"""
防御流水线：读取预处理后的序列数据，按配置施加扰动并写入 data/defended/。

扰动粒度为「每个滑动窗口样本」独立处理，输出 shape 与标签 y 与原数据一致，便于直接复用攻击者模型结构。
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
import time
from typing import Any, Dict, List, Tuple

import numpy as np

from src.config import ExperimentConfig
from src.defenses.adaptive_ldp_defense import AdaptiveLDPDefense
from src.defenses.base_defense import BaseDefense
from src.defenses.ldp_defense import LDPDefense
from src.defenses.noise_defense import NoiseDefense
from src.features import extract_stat_features_matrix
from src.utils import ensure_dir, save_json

logger = logging.getLogger(__name__)


def build_defense(cfg: ExperimentConfig, feature_names: List[str]) -> BaseDefense:
    """根据 defense.method 实例化具体防御器。"""
    dc = cfg.nested("defense")
    method = str(dc.get("method", "noise")).lower().strip()
    sub = {k: v for k, v in dc.items() if k not in ("enabled",)}
    # 与 experiment.random_seed 统一，避免在 defense 段重复配置
    sub["random_seed"] = cfg.random_seed()
    if method == "noise":
        return NoiseDefense(feature_names=feature_names, config=sub)
    if method == "ldp":
        return LDPDefense(feature_names=feature_names, config=sub)
    if method == "adaptive_ldp":
        sub["adaptive_ldp"] = dict(cfg.nested("adaptive_ldp"))
        return AdaptiveLDPDefense(feature_names=feature_names, config=sub)
    raise ValueError(f"未知 defense.method: {method}")


def compute_distortion_metrics(X0: np.ndarray, X1: np.ndarray) -> Dict[str, float]:
    """计算原始张量与扰动后张量之间的简单失真指标（展平后统计）。"""
    if X0.shape != X1.shape:
        raise ValueError("失真度量要求 X0 与 X1 形状一致")
    a = X0.astype(np.float64).ravel()
    b = X1.astype(np.float64).ravel()
    diff = b - a
    mse = float(np.mean(diff**2))
    mae = float(np.mean(np.abs(diff)))
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        pearson = 0.0
    else:
        pearson = float(np.corrcoef(a, b)[0, 1])
        if np.isnan(pearson):
            pearson = 0.0
    denom = np.maximum(np.abs(a), 1e-8)
    mape_like = float(np.mean(np.abs(diff) / denom))
    return {
        "mse": mse,
        "mae": mae,
        "pearson_r": pearson,
        "mean_relative_abs_error": mape_like,
    }


def run_defense_pipeline(cfg: ExperimentConfig) -> Dict[str, Any]:
    """
    读取 data/processed/sequences.npz，施加防御，写入：
    - defended_train.npz / defended_val.npz / defended_test.npz
    - defended_sequences.npz（含全部分割，便于训练脚本）
    - defended_mlp_features.npz（由扰动后序列重算统计特征）
    - defense_artifact.json（防御器参数快照）
    返回摘要 dict（含失真指标），供评估与对比实验使用。
    """
    dcfg = cfg.nested("defense")
    if not bool(dcfg.get("enabled", True)):
        raise ValueError("defense.enabled=false，已跳过防御流水线。如需生成防御数据请改为 true。")

    processed = cfg.path("paths", "processed_dir")
    meta_path = processed / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"缺少 {meta_path}，请先运行预处理。")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta: Dict[str, Any] = json.load(f)
    feature_names: List[str] = list(meta["feature_names"])

    seq_path = processed / "sequences.npz"
    if not seq_path.is_file():
        raise FileNotFoundError(f"缺少 {seq_path}")

    data = np.load(seq_path)
    X_train = data["X_train"].astype(np.float32)
    X_val = data["X_val"].astype(np.float32)
    X_test = data["X_test"].astype(np.float32)
    y_train = data["y_train"]
    y_val = data["y_val"]
    y_test = data["y_test"]

    defense = build_defense(cfg, feature_names)
    defense.fit(np.concatenate([X_train, X_val, X_test], axis=0), None)

    t0 = time.perf_counter()
    Xt_tr = defense.transform(X_train)
    t1 = time.perf_counter()
    Xt_va = defense.transform(X_val)
    t2 = time.perf_counter()
    Xt_te = defense.transform(X_test)
    t3 = time.perf_counter()
    timing = {
        "transform_train_seconds": round(t1 - t0, 4),
        "transform_val_seconds": round(t2 - t1, 4),
        "transform_test_seconds": round(t3 - t2, 4),
        "transform_total_seconds": round(t3 - t0, 4),
        "num_windows_train_val_test": [int(X_train.shape[0]), int(X_val.shape[0]), int(X_test.shape[0])],
    }

    X_all_orig = np.concatenate([X_train, X_val, X_test], axis=0)
    X_all_def = np.concatenate([Xt_tr, Xt_va, Xt_te], axis=0)
    distort = compute_distortion_metrics(X_all_orig, X_all_def)

    defended_root = ensure_dir(cfg.path("paths", "defended_dir"))

    np.savez_compressed(defended_root / "defended_train.npz", X=Xt_tr, y=y_train)
    np.savez_compressed(defended_root / "defended_val.npz", X=Xt_va, y=y_val)
    np.savez_compressed(defended_root / "defended_test.npz", X=Xt_te, y=y_test)

    np.savez_compressed(
        defended_root / "defended_sequences.npz",
        X_train=Xt_tr,
        y_train=y_train,
        X_val=Xt_va,
        y_val=y_val,
        X_test=Xt_te,
        y_test=y_test,
    )

    feat_cfg = cfg.nested("features")
    Xm_tr = extract_stat_features_matrix(Xt_tr, feature_names, feat_cfg)
    Xm_va = extract_stat_features_matrix(Xt_va, feature_names, feat_cfg)
    Xm_te = extract_stat_features_matrix(Xt_te, feature_names, feat_cfg)
    np.savez_compressed(
        defended_root / "defended_mlp_features.npz",
        X_train=Xm_tr,
        y_train=y_train,
        X_val=Xm_va,
        y_val=y_val,
        X_test=Xm_te,
        y_test=y_test,
    )

    art_path = defended_root / "defense_artifact.json"
    defense.save(art_path)

    summary = {
        "defense_method": dcfg.get("method"),
        "defense_config": dcfg,
        "distortion": distort,
        "system_performance": timing,
        "defended_dir": str(defended_root),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "shapes": {
            "train": list(Xt_tr.shape),
            "val": list(Xt_va.shape),
            "test": list(Xt_te.shape),
        },
    }
    save_json(summary, defended_root / "defense_summary.json")
    logger.info(
        "防御数据已写入 %s | MSE=%.6f MAE=%.6f r=%.4f",
        defended_root,
        distort["mse"],
        distort["mae"],
        distort["pearson_r"],
    )
    return summary
