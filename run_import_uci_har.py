#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
导入真实公开数据集：UCI HAR (Human Activity Recognition Using Smartphones)。

该数据集本身已经是固定窗口（128 点/窗、50Hz）并带有活动标签，因此本脚本会：
- 下载/解压后的目录中读取 Inertial Signals（9 通道）与 y_train/y_test
- 将其转存为本项目统一的 data/processed/sequences.npz + meta.json
- 可选生成 mlp_features.npz（复用本项目的统计特征提取）

用法:
  python run_import_uci_har.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import ExperimentConfig
from src.features import extract_stat_features_matrix
from src.utils import ensure_dir, save_json, set_seed

logger = logging.getLogger(__name__)


UCI_HAR_ZIP_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/"
    "UCI%20HAR%20Dataset.zip"
)


def _har_root(cfg: ExperimentConfig) -> Path:
    """
    默认读取 data/raw/uci_har/UCI HAR Dataset/。
    若你把数据放在其他位置，可直接改这里或扩展为配置项。
    """
    return (cfg.project_root / "data" / "raw" / "uci_har" / "UCI HAR Dataset").resolve()


def _har_zip_path(cfg: ExperimentConfig) -> Path:
    return (cfg.project_root / "data" / "raw" / "uci_har" / "UCI_HAR_Dataset.zip").resolve()


def _cleanup_macos_artifacts(root: Path) -> None:
    """
    清理 macOS 解压附件：__MACOSX/、._*、.DS_Store。
    不影响数据内容，只减少干扰。
    """
    mac_dir = root / "__MACOSX"
    if mac_dir.exists():
        try:
            for p in mac_dir.rglob("*"):
                if p.is_file():
                    p.unlink(missing_ok=True)
            # 递归删除目录
            for p in sorted(mac_dir.rglob("*"), reverse=True):
                if p.is_dir():
                    p.rmdir()
            mac_dir.rmdir()
        except Exception:
            pass

    for ds in root.rglob(".DS_Store"):
        try:
            ds.unlink(missing_ok=True)
        except Exception:
            pass
    for dot_ in root.rglob("._*"):
        try:
            dot_.unlink(missing_ok=True)
        except Exception:
            pass


def _download_file(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".part")
    if tmp.exists():
        tmp.unlink(missing_ok=True)
    logger.info("下载 UCI HAR 压缩包: %s", url)
    urllib.request.urlretrieve(url, tmp)  # nosec - controlled public dataset URL
    tmp.replace(out_path)


def _ensure_har_present(cfg: ExperimentConfig, *, auto_download: bool) -> Path:
    """
    确保 data/raw/uci_har/UCI HAR Dataset/ 存在。
    若缺失且 auto_download=True，则自动下载并解压。
    """
    har_root = _har_root(cfg)
    if har_root.is_dir():
        return har_root

    if not auto_download:
        raise FileNotFoundError(
            f"未找到 UCI HAR 目录: {har_root}\n"
            f"请先手动下载并解压，或在命令中添加 --auto-download。"
        )

    zip_path = _har_zip_path(cfg)
    if not zip_path.is_file():
        _download_file(UCI_HAR_ZIP_URL, zip_path)

    raw_root = zip_path.parent
    logger.info("解压到: %s", raw_root)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(raw_root)

    _cleanup_macos_artifacts(raw_root)

    if not har_root.is_dir():
        raise FileNotFoundError(
            f"解压完成但仍未找到目录: {har_root}\n"
            f"请检查压缩包内容是否为 UCI HAR Dataset.zip。"
        )
    return har_root


def _load_activity_labels(har_root: Path) -> List[str]:
    path = har_root / "activity_labels.txt"
    if not path.is_file():
        raise FileNotFoundError(f"缺少活动标签文件: {path}")
    classes: List[str] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        # 统一为小写，便于与项目其他输出一致
        classes.append(parts[1].strip().lower())
    if not classes:
        raise ValueError("未能解析 activity_labels.txt")
    return classes


def _load_signal_matrix(path: Path) -> np.ndarray:
    """
    读取形如 *_train.txt 的信号矩阵：
    - 每行 128 个浮点数（一个窗口）
    - 返回 shape (N, 128)
    """
    if not path.is_file():
        raise FileNotFoundError(f"缺少信号文件: {path}")
    x = np.loadtxt(path, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"信号矩阵维度异常: {path} -> {x.shape}")
    return x.astype(np.float32)


def _load_split(har_root: Path, split: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    split: "train" | "test"
    返回:
      X: (N, 128, 9)
      y: (N,) 0-index int64
    """
    split = split.lower().strip()
    if split not in ("train", "test"):
        raise ValueError("split 只能为 train 或 test")

    base = har_root / split
    inertial = base / "Inertial Signals"
    y_path = base / f"y_{split}.txt"

    # 选用 9 通道惯性信号（每窗 128 点）
    channels = [
        (f"total_acc_x_{split}.txt", "total_acc_x"),
        (f"total_acc_y_{split}.txt", "total_acc_y"),
        (f"total_acc_z_{split}.txt", "total_acc_z"),
        (f"body_acc_x_{split}.txt", "body_acc_x"),
        (f"body_acc_y_{split}.txt", "body_acc_y"),
        (f"body_acc_z_{split}.txt", "body_acc_z"),
        (f"body_gyro_x_{split}.txt", "body_gyro_x"),
        (f"body_gyro_y_{split}.txt", "body_gyro_y"),
        (f"body_gyro_z_{split}.txt", "body_gyro_z"),
    ]

    mats: List[np.ndarray] = []
    for fname, _ in channels:
        mats.append(_load_signal_matrix(inertial / fname))

    # shape check + stack to (N, 128, 9)
    n = int(mats[0].shape[0])
    t = int(mats[0].shape[1])
    for m in mats[1:]:
        if int(m.shape[0]) != n or int(m.shape[1]) != t:
            raise ValueError("各通道窗口数/长度不一致")
    X = np.stack(mats, axis=-1)  # (N, 128, 9)

    y_raw = np.loadtxt(y_path, dtype=np.int64).reshape(-1)
    if y_raw.shape[0] != n:
        raise ValueError(f"标签长度与样本数不一致: y={y_raw.shape}, X={X.shape}")
    # 原始标签为 1..6，转为 0..5
    y = (y_raw - 1).astype(np.int64)
    return X.astype(np.float32), y


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    parser = argparse.ArgumentParser(description="导入 UCI HAR 数据集到本项目格式")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument(
        "--auto-download",
        action="store_true",
        help="若未找到 data/raw/uci_har/UCI HAR Dataset/，则自动下载并解压官方 UCI HAR 压缩包。",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = (ROOT / cfg_path).resolve()
    cfg = ExperimentConfig.from_yaml(cfg_path, project_root=ROOT)

    set_seed(cfg.random_seed())

    har_root = _ensure_har_present(cfg, auto_download=bool(args.auto_download))

    classes = _load_activity_labels(har_root)
    X_train_full, y_train_full = _load_split(har_root, "train")
    X_test, y_test = _load_split(har_root, "test")

    # 将官方 train 再切一份 val，保证你现有训练脚本可直接复用
    seed = cfg.random_seed()
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.2,
        random_state=seed,
        stratify=y_train_full,
    )

    processed = ensure_dir(cfg.path("paths", "processed_dir"))
    np.savez_compressed(
        processed / "sequences.npz",
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
    )

    feature_names = [
        "total_acc_x",
        "total_acc_y",
        "total_acc_z",
        "body_acc_x",
        "body_acc_y",
        "body_acc_z",
        "body_gyro_x",
        "body_gyro_y",
        "body_gyro_z",
    ]

    pp = cfg.nested("preprocess")
    save_mlp = bool(pp.get("save_mlp_features", True))
    mlp_dim = 0
    if save_mlp:
        feat_cfg = cfg.nested("features")
        Xm_tr = extract_stat_features_matrix(X_train, feature_names, feat_cfg)
        Xm_va = extract_stat_features_matrix(X_val, feature_names, feat_cfg)
        Xm_te = extract_stat_features_matrix(X_test, feature_names, feat_cfg)
        np.savez_compressed(
            processed / "mlp_features.npz",
            X_train=Xm_tr,
            X_val=Xm_va,
            X_test=Xm_te,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
        )
        mlp_dim = int(Xm_tr.shape[1])

    meta: Dict[str, Any] = {
        "dataset": "UCI HAR Dataset",
        "source": "https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones",
        "feature_names": feature_names,
        "class_names": classes,
        "seq_len": int(X_train.shape[1]),
        "freq": "50Hz_windowed_2.56s",
        "label_source": "uci_har",
        "mlp_feature_dim": mlp_dim,
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "n_test": int(len(y_test)),
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }
    save_json(meta, processed / "meta.json")

    logger.info(
        "UCI HAR 导入完成: train=%d val=%d test=%d | X=(T=%d,F=%d) | processed=%s",
        len(y_train),
        len(y_val),
        len(y_test),
        X_train.shape[1],
        X_train.shape[2],
        processed,
    )


if __name__ == "__main__":
    main()

