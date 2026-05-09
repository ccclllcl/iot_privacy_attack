"""
特征工程：从滑窗张量提取统计特征，供 MLP / 传统机器学习基线使用。

也可在后续隐私实验中对比「原始序列 vs 统计特征」的泄露程度。
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np


def _changes_1d(x: np.ndarray) -> float:
    """一维序列相邻差分非零次数（开关/事件跳变次数代理）。"""
    if x.size < 2:
        return 0.0
    d = np.diff(x.astype(np.float64))
    return float(np.sum(np.abs(d) > 1e-8))


def extract_stat_features_for_window(
    window: np.ndarray, feature_names: List[str], feat_cfg: Dict[str, Any]
) -> np.ndarray:
    """
    单个窗口 (seq_len, n_channels) -> 一维特征向量。

    特征顺序稳定，便于保存 meta 与复现实验。
    """
    seq_len, n_ch = window.shape
    per = feat_cfg.get("per_channel", {})
    glob = feat_cfg.get("global", {})
    feats: List[float] = []

    total_energy_proxy = float(window.sum())
    peak_power_proxy = float(window.max()) if window.size else 0.0
    mean_power_proxy = float(window.mean()) if window.size else 0.0

    for c in range(n_ch):
        ch = window[:, c].astype(np.float64)
        if per.get("use_count", True):
            feats.append(float(np.sum(ch > 1e-8)))
        if per.get("use_mean", True):
            feats.append(float(ch.mean()))
        if per.get("use_max", True):
            feats.append(float(ch.max()) if ch.size else 0.0)
        if per.get("use_min", True):
            feats.append(float(ch.min()) if ch.size else 0.0)
        if per.get("use_std", True):
            feats.append(float(ch.std()) if ch.size else 0.0)
        if per.get("use_change_count", True):
            feats.append(_changes_1d(ch))

    if glob.get("active_device_count", True):
        # 窗口内「曾活跃」的通道数
        active = int(np.sum(np.max(window, axis=0) > 1e-8))
        feats.append(float(active))

    if glob.get("total_energy_proxy", True):
        feats.append(total_energy_proxy)
    if glob.get("mean_power_proxy", True):
        feats.append(mean_power_proxy)
    if glob.get("peak_power_proxy", True):
        feats.append(peak_power_proxy)

    return np.asarray(feats, dtype=np.float32)


def extract_stat_features_matrix(
    X: np.ndarray, feature_names: List[str], feat_cfg: Dict[str, Any]
) -> np.ndarray:
    """批量窗口 X (N, seq_len, F) -> (N, feat_dim)。"""
    if X.ndim != 3:
        raise ValueError(f"期望 X 维度为 3，得到 shape={X.shape}")
    # Fast vectorized implementation (critical for large N on CASAS-like datasets).
    per = feat_cfg.get("per_channel", {})
    glob = feat_cfg.get("global", {})

    Xf = X.astype(np.float64, copy=False)
    eps = 1e-8

    feats: List[np.ndarray] = []

    # Per-channel features in the SAME order as extract_stat_features_for_window
    if per.get("use_count", True):
        feats.append((Xf > eps).sum(axis=1))
    if per.get("use_mean", True):
        feats.append(Xf.mean(axis=1))
    if per.get("use_max", True):
        feats.append(Xf.max(axis=1) if Xf.size else np.zeros((Xf.shape[0], Xf.shape[2])))
    if per.get("use_min", True):
        feats.append(Xf.min(axis=1) if Xf.size else np.zeros((Xf.shape[0], Xf.shape[2])))
    if per.get("use_std", True):
        feats.append(Xf.std(axis=1))
    if per.get("use_change_count", True):
        d = np.diff(Xf, axis=1)
        feats.append((np.abs(d) > eps).sum(axis=1))

    # Flatten per-channel blocks: (N, F) * K  -> (N, F*K)
    out_parts: List[np.ndarray] = []
    for block in feats:
        out_parts.append(block.astype(np.float32, copy=False))

    # Global features (append to the end)
    if glob.get("active_device_count", True):
        active = (Xf.max(axis=1) > eps).sum(axis=1).astype(np.float32)
        out_parts.append(active.reshape(-1, 1))

    if glob.get("total_energy_proxy", True):
        out_parts.append(Xf.sum(axis=(1, 2), dtype=np.float64).astype(np.float32).reshape(-1, 1))
    if glob.get("mean_power_proxy", True):
        out_parts.append(Xf.mean(axis=(1, 2)).astype(np.float32).reshape(-1, 1))
    if glob.get("peak_power_proxy", True):
        out_parts.append(Xf.max(axis=(1, 2)).astype(np.float32).reshape(-1, 1))

    if not out_parts:
        return np.zeros((X.shape[0], 0), dtype=np.float32)

    return np.concatenate(out_parts, axis=1)
