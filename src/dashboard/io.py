"""Dashboard 文件读取与绘图工具。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def read_json(path: Path) -> Any | None:
    if not path.exists() or not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or not path.is_file() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def read_text(path: Path, max_chars: int = 120_000) -> str:
    if not path.exists() or not path.is_file():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    if len(text) > max_chars:
        return text[:max_chars] + "\n\n...[内容已截断]...\n"
    return text


def list_artifacts(path: Path) -> list[Path]:
    if not path.exists() or not path.is_dir():
        return []
    return sorted([p for p in path.iterdir() if p.is_file()], key=lambda p: p.name)


def load_metrics(path: Path) -> dict[str, Any]:
    data = read_json(path)
    return data if isinstance(data, dict) else {}


def load_confusion(path: Path) -> dict[str, Any]:
    data = read_json(path)
    return data if isinstance(data, dict) else {}


def _class_names(confusion_json: dict[str, Any], n: int) -> list[str]:
    names = confusion_json.get("class_names") or confusion_json.get("labels")
    if isinstance(names, list) and len(names) == n:
        return [str(x) for x in names]
    return [str(i) for i in range(n)]


def _top_confusions(cm: np.ndarray, names: list[str], top_n: int = 12) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i != j and int(cm[i, j]) > 0:
                rows.append({"true": names[i], "pred": names[j], "count": int(cm[i, j])})
    rows.sort(key=lambda r: r["count"], reverse=True)
    return pd.DataFrame(rows[:top_n])


def plot_confusion_matrix(
    confusion_json: dict[str, Any],
    normalize: bool = True,
    max_labels: int = 30,
) -> tuple[plt.Figure | None, pd.DataFrame]:
    raw = confusion_json.get("confusion_matrix")
    if raw is None:
        return None, pd.DataFrame()
    cm = np.asarray(raw, dtype=float)
    if cm.ndim != 2 or cm.shape[0] == 0 or cm.shape[0] != cm.shape[1]:
        return None, pd.DataFrame()

    names = _class_names(confusion_json, cm.shape[0])
    top_df = _top_confusions(cm.astype(int), names)
    if cm.shape[0] > max_labels:
        support = cm.sum(axis=1)
        keep = np.argsort(support)[-max_labels:]
        keep = np.sort(keep)
        cm_show = cm[np.ix_(keep, keep)]
        names_show = [names[int(i)] for i in keep]
    else:
        cm_show = cm
        names_show = names

    title = "混淆矩阵"
    if normalize:
        denom = cm_show.sum(axis=1, keepdims=True)
        denom[denom == 0] = 1.0
        cm_show = cm_show / denom
        title = "混淆矩阵（按行归一化）"

    fig_size = max(6.0, min(14.0, 0.42 * len(names_show)))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    im = ax.imshow(cm_show, cmap="Blues", interpolation="nearest")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    stride = max(1, int(np.ceil(len(names_show) / max_labels)))
    tick_idx = np.arange(0, len(names_show), stride)
    ax.set_xticks(tick_idx)
    ax.set_yticks(tick_idx)
    ax.set_xticklabels([names_show[i] for i in tick_idx], rotation=45, ha="right")
    ax.set_yticklabels([names_show[i] for i in tick_idx])
    ax.set_xlabel("预测标签")
    ax.set_ylabel("真实标签")
    ax.set_title(title)
    fig.tight_layout()
    return fig, top_df


def plot_parameter_scan(csv_df: pd.DataFrame, method: str) -> tuple[plt.Figure | None, pd.DataFrame]:
    if csv_df.empty:
        return None, pd.DataFrame()
    df = csv_df.copy()
    y_cols = [c for c in ["defended_accuracy", "mse", "pearson_r"] if c in df.columns]
    for col in y_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if method == "adaptive_ldp":
        x_col = "profile_name" if "profile_name" in df.columns else "param_value"
        detail_cols = [
            c
            for c in ["profile_name", "epsilon_min", "epsilon_max", "weight_sensitivity", "weight_traffic"]
            if c in df.columns
        ]
    elif method == "ldp":
        x_col = "epsilon" if "epsilon" in df.columns else "param_value"
        detail_cols = [c for c in ["param_name", "param_value", "epsilon"] if c in df.columns]
    else:
        x_col = "noise_scale" if "noise_scale" in df.columns else "param_value"
        detail_cols = [c for c in ["param_name", "param_value", "noise_scale"] if c in df.columns]

    if x_col not in df.columns:
        x_col = "param_value"
    x = df[x_col].astype(str)
    fig, axes = plt.subplots(len(y_cols), 1, figsize=(9, max(3.2, 2.8 * len(y_cols))), squeeze=False)
    for ax, col in zip(axes.ravel(), y_cols):
        ax.plot(x, df[col], marker="o")
        ax.set_ylabel(col)
        ax.grid(True, alpha=0.25)
        if method == "ldp":
            ax.set_xlabel("epsilon")
        elif method == "noise":
            ax.set_xlabel("noise_scale")
        else:
            ax.set_xlabel("profile")
        ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    detail = df[detail_cols].copy() if detail_cols else pd.DataFrame()
    return fig, detail


def summarize_experiment_selection(
    dataset: str,
    seed: int,
    model: str,
    method: str | None = None,
    mode: str | None = None,
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "seed": int(seed),
        "model": model,
        "method": method,
        "mode": mode,
    }
