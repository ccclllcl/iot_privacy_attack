"""
防御效果评估：对比「原始数据 vs 扰动后数据」下攻击者模型的识别性能。

模式说明（与论文威胁模型对应）：
- fixed_attacker（模式 A）：攻击者在干净数据上训练好的分类器保持不变，仅将测试输入替换为防御后数据。
  该设定刻画「攻击者已部署识别器、用户侧发布扰动数据」的场景，若准确率显著下降，说明防御使既有模型失效。
- retrain_attacker（模式 B）：攻击者获得与防御后数据同分布的训练集并重新训练，再在防御后测试集上评估。
  用于分析自适应对手在适应扰动机制后是否仍能恢复高识别率，即稳健性风险。

同时报告攻击性能下降与数据失真指标，用于隐私-可用性权衡分析。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from src.plotting import configure_matplotlib_english
import torch

from src.config import ExperimentConfig
from src.defenses.defense_pipeline import run_defense_pipeline
from src.evaluate import (
    evaluate_on_arrays,
    load_model_from_checkpoint,
    _plot_confusion,
)
from src.train import run_training
from src.utils import ensure_dir, get_torch_device, save_json

logger = logging.getLogger(__name__)


def _defense_paths(cfg: ExperimentConfig) -> Tuple[Path, Path]:
    defended = cfg.path("paths", "defended_dir")
    defense_out = ensure_dir(cfg.path("paths", "defense_dir"))
    return defended, defense_out


def compute_fixed_attacker_metrics(
    cfg: ExperimentConfig,
    model_path: Path,
) -> Dict[str, Any]:
    """
    使用已训练攻击模型，分别在干净/防御后测试集上计算指标（不写入完整报告）。
    调用前需已生成 defended_test.npz。
    """
    ev = cfg.nested("evaluate")
    device = get_torch_device(str(ev.get("device") or "auto"))
    model, ckpt = load_model_from_checkpoint(model_path, device)
    model_type = str(ckpt["model_type"])
    class_names: List[str] = list(ckpt["class_names"])
    bs = int(ev.get("batch_size", 128))
    nw = int(ev.get("num_workers", 0))

    processed = cfg.path("paths", "processed_dir")
    defended = cfg.path("paths", "defended_dir")

    if model_type == "lstm":
        clean = np.load(processed / "sequences.npz")
        Xc = clean["X_test"]
        yc = clean["y_test"]
        defd = np.load(defended / "defended_test.npz")
        Xd = defd["X"]
        yd = defd["y"]
    else:
        clean = np.load(processed / "mlp_features.npz")
        Xc, yc = clean["X_test"], clean["y_test"]
        defd = np.load(defended / "defended_mlp_features.npz")
        Xd, yd = defd["X_test"], defd["y_test"]

    base_m = evaluate_on_arrays(
        model, Xc, yc, model_type, class_names, device, bs, nw
    )
    def_m = evaluate_on_arrays(
        model, Xd, yd, model_type, class_names, device, bs, nw
    )
    return {
        "baseline": base_m,
        "defended": def_m,
        "model_type": model_type,
        "class_names": class_names,
    }


def _recall_drop(
    b: Dict[str, float], d: Dict[str, float]
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k in b:
        out[k] = float(b[k]) - float(d.get(k, 0.0))
    return out


def run_defense_evaluation(
    cfg: ExperimentConfig,
    mode: str,
    model_path: Optional[Path] = None,
    skip_pipeline: bool = False,
) -> Dict[str, Any]:
    """
    主入口：根据模式运行评估并写入 outputs/defense/ 下报告与图表。

    skip_pipeline: 若已为当前配置生成过防御数据，可跳过再次扰动以节省时间。
    """
    mode = mode.lower().strip()
    if mode not in ("fixed_attacker", "retrain_attacker"):
        raise ValueError("mode 只能是 fixed_attacker 或 retrain_attacker")

    defended_root, defense_out = _defense_paths(cfg)
    ensure_dir(defense_out)

    if not skip_pipeline:
        pipe_summary = run_defense_pipeline(cfg)
    else:
        summ_path = defended_root / "defense_summary.json"
        if not summ_path.is_file():
            raise FileNotFoundError(f"skip_pipeline=true 但缺少 {summ_path}")
        with open(summ_path, "r", encoding="utf-8") as f:
            pipe_summary = json.load(f)

    distort = pipe_summary.get("distortion", {})

    ev = cfg.nested("evaluate")
    device = get_torch_device(str(ev.get("device") or "auto"))
    bs = int(ev.get("batch_size", 128))
    nw = int(ev.get("num_workers", 0))
    processed = cfg.path("paths", "processed_dir")

    report: Dict[str, Any] = {
        "mode": mode,
        "distortion": distort,
        "defense_config": cfg.nested("defense"),
    }

    if mode == "fixed_attacker":
        if model_path is None or not model_path.is_file():
            raise FileNotFoundError("fixed_attacker 模式需要有效的 --model_path")
        pair = compute_fixed_attacker_metrics(cfg, model_path)
        b = pair["baseline"]
        d = pair["defended"]
        class_names = pair["class_names"]

        acc_drop = float(b["accuracy"]) - float(d["accuracy"])
        rel_drop_pct = (
            (acc_drop / max(float(b["accuracy"]), 1e-8)) * 100.0
            if b["accuracy"] > 0
            else 0.0
        )
        f1_drop = float(b["f1_macro"]) - float(d["f1_macro"])
        recall_drop = _recall_drop(b["per_class_recall"], d["per_class_recall"])

        report["attack_metrics"] = {
            "baseline": {
                "accuracy": b["accuracy"],
                "precision_macro": b["precision_macro"],
                "recall_macro": b["recall_macro"],
                "f1_macro": b["f1_macro"],
                "per_class_recall": b["per_class_recall"],
            },
            "defended_fixed_attacker": {
                "accuracy": d["accuracy"],
                "precision_macro": d["precision_macro"],
                "recall_macro": d["recall_macro"],
                "f1_macro": d["f1_macro"],
                "per_class_recall": d["per_class_recall"],
            },
            "defense_effect": {
                "accuracy_drop": acc_drop,
                "relative_accuracy_drop_percent": rel_drop_pct,
                "macro_f1_drop": f1_drop,
                "per_class_recall_drop": recall_drop,
            },
        }

        _plot_confusion(
            b["confusion_matrix"], class_names, defense_out / "confusion_matrix_baseline.png"
        )
        _plot_confusion(
            d["confusion_matrix"], class_names, defense_out / "confusion_matrix_defended.png"
        )

        configure_matplotlib_english()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(
            ["Clean test", "Defended test"],
            [b["accuracy"], d["accuracy"]],
            color=["#4C72B0", "#DD8452"],
        )
        ax.set_ylabel("Accuracy")
        ax.set_title("Fixed attacker: accuracy comparison")
        ax.set_ylim(0.0, 1.05)
        for i, v in enumerate([b["accuracy"], d["accuracy"]]):
            ax.text(i, v + 0.02, f"{v:.3f}", ha="center")
        fig.tight_layout()
        fig.savefig(defense_out / "accuracy_comparison.png", dpi=150)
        plt.close(fig)

    else:
        decfg = cfg.nested("defense_eval")
        model_type = str(cfg.nested("train").get("model_type", "lstm")).lower()
        if model_type not in ("lstm", "mlp"):
            model_type = "lstm"

        seq_def = defended_root / "defended_sequences.npz"
        mlp_def = defended_root / "defended_mlp_features.npz"
        models_dir = ensure_dir(cfg.path("paths", "models_dir"))
        name = str(
            decfg.get(
                "retrained_model_name",
                f"best_{model_type}_defended_retrain.pt",
            )
        )
        save_path = models_dir / name

        run_training(
            cfg,
            model_type=model_type,
            override_model_path=save_path,
            sequences_npz=seq_def if model_type == "lstm" else None,
            mlp_npz=mlp_def if model_type == "mlp" else None,
            curve_output_path=defense_out / f"train_curve_defended_retrain_{model_type}.png",
            history_output_path=defense_out / f"train_history_defended_retrain_{model_type}.json",
        )

        model, ckpt = load_model_from_checkpoint(save_path, device)
        class_names = list(ckpt["class_names"])

        if model_type == "lstm":
            dt = np.load(seq_def)
            Xd, yd = dt["X_test"], dt["y_test"]
        else:
            dt = np.load(mlp_def)
            Xd, yd = dt["X_test"], dt["y_test"]

        d = evaluate_on_arrays(
            model, Xd, yd, model_type, class_names, device, bs, nw
        )

        clean = (
            np.load(processed / "sequences.npz")
            if model_type == "lstm"
            else np.load(processed / "mlp_features.npz")
        )
        Xc, yc = clean["X_test"], clean["y_test"]
        b = evaluate_on_arrays(
            model, Xc, yc, model_type, class_names, device, bs, nw
        )

        report["attack_metrics"] = {
            "note": "重训后模型在干净测试集上的表现通常下降（分布偏移），在防御后测试集上为自适应攻击者的主指标。",
            "retrained_on_defended": {
                "accuracy_on_defended_test": d["accuracy"],
                "f1_macro_on_defended_test": d["f1_macro"],
                "per_class_recall_on_defended_test": d["per_class_recall"],
            },
            "same_retrained_model_on_clean_test": {
                "accuracy": b["accuracy"],
                "f1_macro": b["f1_macro"],
                "per_class_recall": b["per_class_recall"],
            },
        }
        _plot_confusion(
            d["confusion_matrix"], class_names, defense_out / "confusion_matrix_defended_retrain.png"
        )

    save_json(report, defense_out / "defense_report.json")

    txt_lines = [
        "物联网隐私防御评估报告",
        f"模式: {mode}",
        "",
        "=== 数据失真（防御流水线输出） ===",
        json.dumps(distort, ensure_ascii=False, indent=2),
        "",
        "=== 攻击性能 ===",
        json.dumps(report.get("attack_metrics", {}), ensure_ascii=False, indent=2),
    ]
    with open(defense_out / "defense_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(txt_lines))

    logger.info("防御评估完成，结果目录: %s", defense_out)
    return report
