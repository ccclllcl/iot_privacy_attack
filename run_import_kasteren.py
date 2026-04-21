#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
导入 van Kasteren 智能家居环境传感器数据（CSV.annotated）到本项目统一格式。

数据来源（CSV 由社区项目转换而来）：
https://github.com/aitoralmeida/c4a_activity_recognition

用法:
  python run_import_kasteren.py --config configs/default.yaml --auto-download
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import ExperimentConfig
from src.features import extract_stat_features_matrix
from src.utils import ensure_dir, save_json


logger = logging.getLogger(__name__)


KASTEREN_URL = (
    "https://raw.githubusercontent.com/aitoralmeida/c4a_activity_recognition/"
    "master/experiments/kasteren_dataset/test_kasteren.csv.annotated"
)


def _download(url: str, out: Path) -> None:
    import urllib.request

    out.parent.mkdir(parents=True, exist_ok=True)
    logger.info("下载: %s -> %s", url, out)
    urllib.request.urlretrieve(url, out)  # noqa: S310 (controlled URL)


def _parse_annotated_tsv(path: Path) -> Tuple[List[str], List[str]]:
    """
    返回:
      sensors: 每条事件对应的 sensor 名
      labels:  每条事件对应的 activity label
    """
    sensors: List[str] = []
    labels: List[str] = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 6:
                continue
            sensor = str(parts[2]).strip()
            label = str(parts[5]).strip()
            if sensor and label:
                sensors.append(sensor)
                labels.append(label)
    if not sensors:
        raise ValueError(f"未解析到有效事件: {path}")
    return sensors, labels


def _build_event_windows(
    sensors: List[str],
    labels: List[str],
    *,
    window_len: int,
    stride: int,
    sensor_vocab: List[str],
    label_vocab: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    s2i: Dict[str, int] = {s: i for i, s in enumerate(sensor_vocab)}
    l2i: Dict[str, int] = {l: i for i, l in enumerate(label_vocab)}
    F = len(sensor_vocab)

    Xs: List[np.ndarray] = []
    ys: List[int] = []
    n = len(sensors)
    for start in range(0, n - window_len + 1, stride):
        end = start + window_len
        win = np.zeros((window_len, F), dtype=np.float32)
        for t in range(window_len):
            si = s2i.get(sensors[start + t])
            if si is not None:
                win[t, si] = 1.0
        y = labels[end - 1]
        yi = l2i.get(y)
        if yi is None:
            continue
        Xs.append(win)
        ys.append(int(yi))
    if not Xs:
        raise ValueError("窗口化后样本数为 0：请调大 window_len 或检查数据。")
    return np.stack(Xs, axis=0), np.asarray(ys, dtype=np.int64)


def _split_indices(n: int, seed: int, train_ratio: float, val_ratio: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    idx = np.arange(n, dtype=np.int64)
    rng.shuffle(idx)
    n_tr = int(round(n * train_ratio))
    n_va = int(round(n * val_ratio))
    n_tr = max(1, min(n_tr, n - 2))
    n_va = max(1, min(n_va, n - n_tr - 1))
    tr = idx[:n_tr]
    va = idx[n_tr : n_tr + n_va]
    te = idx[n_tr + n_va :]
    return tr, va, te


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    parser = argparse.ArgumentParser(description="导入 van Kasteren 智能家居数据集")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--auto-download", action="store_true")
    parser.add_argument(
        "--window-len",
        type=int,
        default=30,
        help="事件窗口长度（按事件数滑窗，不是按秒）。",
    )
    parser.add_argument("--stride", type=int, default=1)
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = (ROOT / cfg_path).resolve()
    cfg = ExperimentConfig.from_yaml(cfg_path, project_root=ROOT)

    raw_root = ensure_dir(cfg.project_root / "data" / "raw" / "kasteren")
    raw_path = raw_root / "test_kasteren.csv.annotated"
    if args.auto_download and not raw_path.exists():
        _download(KASTEREN_URL, raw_path)
    if not raw_path.exists():
        raise FileNotFoundError(f"未找到原始文件：{raw_path}（可加 --auto-download）")

    sensors, labels = _parse_annotated_tsv(raw_path)
    sensor_vocab = sorted(set(sensors))
    label_vocab = sorted(set(labels))

    X, y = _build_event_windows(
        sensors,
        labels,
        window_len=max(2, int(args.window_len)),
        stride=max(1, int(args.stride)),
        sensor_vocab=sensor_vocab,
        label_vocab=label_vocab,
    )

    pre = cfg.nested("preprocess")
    tr, va, te = _split_indices(
        int(X.shape[0]),
        seed=cfg.random_seed(),
        train_ratio=float(pre.get("train_ratio", 0.7)),
        val_ratio=float(pre.get("val_ratio", 0.15)),
    )

    out_dir = ensure_dir(cfg.path("paths", "processed_dir"))
    np.savez_compressed(
        out_dir / "sequences.npz",
        X_train=X[tr],
        y_train=y[tr],
        X_val=X[va],
        y_val=y[va],
        X_test=X[te],
        y_test=y[te],
    )

    feat_cfg = cfg.nested("features")
    Xm_tr = extract_stat_features_matrix(X[tr], sensor_vocab, feat_cfg)
    Xm_va = extract_stat_features_matrix(X[va], sensor_vocab, feat_cfg)
    Xm_te = extract_stat_features_matrix(X[te], sensor_vocab, feat_cfg)
    np.savez_compressed(
        out_dir / "mlp_features.npz",
        X_train=Xm_tr,
        y_train=y[tr],
        X_val=Xm_va,
        y_val=y[va],
        X_test=Xm_te,
        y_test=y[te],
    )

    meta = {
        "dataset": "Kasteren (CSV annotated)",
        "source": KASTEREN_URL,
        "feature_names": sensor_vocab,
        "class_names": label_vocab,
        "seq_len": int(X.shape[1]),
        "freq": "event_window",
        "label_source": "kasteren_annotated",
        "mlp_feature_dim": int(Xm_tr.shape[1]) if Xm_tr.ndim == 2 else None,
        "n_train": int(len(tr)),
        "n_val": int(len(va)),
        "n_test": int(len(te)),
    }
    save_json(meta, out_dir / "meta.json")
    logger.info(
        "Kasteren 导入完成: train=%d val=%d test=%d | X=(T=%d,F=%d) | processed=%s",
        len(tr),
        len(va),
        len(te),
        X.shape[1],
        X.shape[2],
        out_dir,
    )


if __name__ == "__main__":
    main()

