#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
导入 CASAS labeled smart home 数据（Zenodo labeled_data.zip）到本项目统一格式。

该数据是智能家居环境传感器事件流（门磁/红外等）+ 活动片段标签（begin/end）。

用法:
  python experiments/real_public/run_import_casas.py --config configs/default.yaml --home hh101 --auto-download
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import ExperimentConfig
from src.features import extract_stat_features_matrix
from src.utils import ensure_dir, save_json


logger = logging.getLogger(__name__)

ZENODO_LABELED_URL = "https://zenodo.org/api/records/15708568/files/labeled_data.zip/content"


def _download(url: str, out: Path) -> None:
    import urllib.request

    out.parent.mkdir(parents=True, exist_ok=True)
    logger.info("下载: %s -> %s", url, out)
    urllib.request.urlretrieve(url, out)  # noqa: S310 (controlled URL)


def _parse_activity_marker(field: str) -> Optional[Tuple[str, str]]:
    """
    Parse e.g. Step_Out="begin" -> ("Step_Out", "begin")
    """
    s = field.strip()
    if "=" not in s:
        return None
    left, right = s.split("=", 1)
    left = left.strip()
    right = right.strip().strip('"').strip("'")
    if not left or not right:
        return None
    if right not in ("begin", "end"):
        return None
    return left, right


def _read_casas_home_from_zip(zip_path: Path, home: str) -> Tuple[List[str], List[str]]:
    """
    Return per-event sensor name and per-event activity label.
    Activity label is taken from the active segment. Events outside segments are labeled "None".
    """
    inner = f"labeled/{home}.csv"
    if not zip_path.is_file():
        raise FileNotFoundError(zip_path)
    with zipfile.ZipFile(zip_path) as z:
        if inner not in set(z.namelist()):
            # Provide a helpful message with available homes.
            homes = sorted(
                p.removeprefix("labeled/").removesuffix(".csv")
                for p in z.namelist()
                if p.startswith("labeled/") and p.endswith(".csv")
            )
            raise FileNotFoundError(f"未找到 {inner}。可用 home 示例: {', '.join(homes[:20])} ...")
        logger.info("读取 CASAS 家庭文件: %s（zip 内路径：%s）", home, inner)
        raw = z.read(inner).decode("utf-8", errors="replace")

    sensors: List[str] = []
    labels: List[str] = []
    current: str = "None"

    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        # Some files contain a single integer line (meta); skip.
        if line.isdigit():
            continue
        parts = list(csv.reader([line]))[0]
        if len(parts) < 4:
            continue
        sensor = str(parts[2]).strip()
        if not sensor:
            continue
        if len(parts) >= 5:
            m = _parse_activity_marker(str(parts[4]))
            if m is not None:
                act, kind = m
                if kind == "begin":
                    current = act
                else:
                    current = "None"
        sensors.append(sensor)
        labels.append(current)

    if not sensors:
        raise ValueError(f"未解析到有效事件: {inner}")
    logger.info("解析完成：events=%d | unique_sensors=%d", len(sensors), len(set(sensors)))
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
        lab = labels[end - 1]
        if lab == "None":
            continue
        yi = l2i.get(lab)
        if yi is None:
            continue
        win = np.zeros((window_len, F), dtype=np.float32)
        for t in range(window_len):
            si = s2i.get(sensors[start + t])
            if si is not None:
                win[t, si] = 1.0
        Xs.append(win)
        ys.append(int(yi))
    if not Xs:
        raise ValueError("窗口化后样本数为 0：可能是活动标签太稀疏或 window_len 太大。")
    logger.info("窗口化完成：samples=%d | window_len=%d | stride=%d", len(Xs), window_len, stride)
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
    parser = argparse.ArgumentParser(description="导入 CASAS labeled 数据集到本项目格式")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--home", type=str, required=True, help="例如 hh101 / rw105 / tm004 等")
    parser.add_argument("--auto-download", action="store_true")
    parser.add_argument("--window-len", type=int, default=30, help="事件窗口长度（按事件数滑窗）")
    parser.add_argument("--stride", type=int, default=1)
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = (ROOT / cfg_path).resolve()
    cfg = ExperimentConfig.from_yaml(cfg_path, project_root=ROOT)

    raw_root = ensure_dir(cfg.project_root / "data" / "raw" / "casas")
    zip_path = raw_root / "labeled_data.zip"
    if args.auto_download and not zip_path.exists():
        _download(ZENODO_LABELED_URL, zip_path)
    if not zip_path.exists():
        raise FileNotFoundError(f"缺少 {zip_path}（可加 --auto-download）")

    logger.info("开始导入 CASAS: home=%s | window_len=%d | stride=%d", args.home, args.window_len, args.stride)
    sensors, seg_labels = _read_casas_home_from_zip(zip_path, home=str(args.home).strip())
    sensor_vocab = sorted(set(sensors))
    label_vocab = sorted({l for l in seg_labels if l != "None"})
    logger.info("标签集合：%d 类（不含 None）", len(label_vocab))

    X, y = _build_event_windows(
        sensors,
        seg_labels,
        window_len=max(2, int(args.window_len)),
        stride=max(1, int(args.stride)),
        sensor_vocab=sensor_vocab,
        label_vocab=label_vocab,
    )

    pre = cfg.nested("preprocess")
    logger.info(
        "划分比例：train=%.2f val=%.2f test=%.2f",
        float(pre.get("train_ratio", 0.7)),
        float(pre.get("val_ratio", 0.15)),
        float(pre.get("test_ratio", 0.15)),
    )
    tr, va, te = _split_indices(
        int(X.shape[0]),
        seed=cfg.random_seed(),
        train_ratio=float(pre.get("train_ratio", 0.7)),
        val_ratio=float(pre.get("val_ratio", 0.15)),
    )

    out_dir = ensure_dir(cfg.path("paths", "processed_dir"))
    logger.info("写入 processed 目录: %s", out_dir)
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
    logger.info("生成 MLP 统计特征...")
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
        "dataset": f"CASAS labeled ({args.home})",
        "source": ZENODO_LABELED_URL,
        "home": str(args.home).strip(),
        "feature_names": sensor_vocab,
        "class_names": label_vocab,
        "seq_len": int(X.shape[1]),
        "freq": "event_window",
        "label_source": "casas_begin_end_segments",
        "mlp_feature_dim": int(Xm_tr.shape[1]) if Xm_tr.ndim == 2 else None,
        "n_train": int(len(tr)),
        "n_val": int(len(va)),
        "n_test": int(len(te)),
    }
    save_json(meta, out_dir / "meta.json")
    logger.info(
        "CASAS 导入完成(%s): train=%d val=%d test=%d | X=(T=%d,F=%d) | processed=%s",
        args.home,
        len(tr),
        len(va),
        len(te),
        X.shape[1],
        X.shape[2],
        out_dir,
    )


if __name__ == "__main__":
    main()

