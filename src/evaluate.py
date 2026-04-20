"""
评估模块：Accuracy / Precision / Recall / F1、混淆矩阵与分类报告输出。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from src.plotting import configure_matplotlib_english
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader

from src.config import ExperimentConfig
from src.dataset import SequenceDataset, TabularDataset
from src.models.lstm_classifier import LSTMClassifier
from src.models.mlp_baseline import MLPBaseline
from src.utils import ensure_dir, get_torch_device

logger = logging.getLogger(__name__)


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys: List[int] = []
    ps: List[int] = []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb)
        pred = logits.argmax(dim=-1).cpu().numpy().tolist()
        ps.extend(pred)
        ys.extend(yb.numpy().tolist())
    return np.asarray(ys, dtype=np.int64), np.asarray(ps, dtype=np.int64)


def evaluate_on_arrays(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    model_type: str,
    class_names: List[str],
    device: torch.device,
    batch_size: int = 128,
    num_workers: int = 0,
) -> Dict[str, Any]:
    """
    在给定特征矩阵/张量上评估分类器，返回指标与预测结果。

    model_type: lstm -> X 形状 (N, T, F)；mlp -> X 形状 (N, D)。
    """
    model_type = model_type.lower().strip()
    if model_type == "lstm":
        ds = SequenceDataset(X, y)
    elif model_type == "mlp":
        ds = TabularDataset(X, y)
    else:
        raise ValueError("model_type 只能是 lstm 或 mlp")

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    y_true, y_pred = collect_predictions(model, loader, device)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    per_recall = recall_score(
        y_true, y_pred, average=None, zero_division=0, labels=list(range(len(class_names)))
    )
    per_recall_dict = {
        class_names[i]: float(per_recall[i]) for i in range(len(class_names))
    }
    return {
        "accuracy": float(acc),
        "precision_macro": float(prec),
        "recall_macro": float(rec),
        "f1_macro": float(f1),
        "per_class_recall": per_recall_dict,
        "y_true": y_true,
        "y_pred": y_pred,
        "confusion_matrix": cm,
        "classification_report": report,
    }


def load_model_from_checkpoint(
    ckpt_path: Path, device: torch.device
) -> Tuple[nn.Module, Dict[str, Any]]:
    """从 run_training 保存的 checkpoint 恢复模型。"""
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)
    model_type = str(ckpt.get("model_type", "lstm"))
    num_classes = int(ckpt["num_classes"])
    in_dim = int(ckpt["input_dim"])
    tr = ckpt.get("train_config", {})

    if model_type == "lstm":
        model = LSTMClassifier(
            input_dim=in_dim,
            num_classes=num_classes,
            hidden_size=int(tr.get("lstm_hidden_size", 128)),
            num_layers=int(tr.get("lstm_num_layers", 2)),
            dropout=float(tr.get("dropout", 0.3)),
        )
    else:
        hidden = list(tr.get("mlp_hidden_sizes", [256, 128]))
        model = MLPBaseline(
            input_dim=in_dim,
            num_classes=num_classes,
            hidden_sizes=hidden,
            dropout=float(tr.get("dropout", 0.3)),
        )

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    return model, ckpt


def run_evaluate(cfg: ExperimentConfig, model_path: Path) -> None:
    """在测试集上评估并写入 reports/ 与 figures/。"""
    ev = cfg.nested("evaluate")
    device = get_torch_device(str(ev.get("device") or "auto"))
    model, ckpt = load_model_from_checkpoint(model_path, device)
    class_names: List[str] = list(ckpt["class_names"])
    model_type = str(ckpt["model_type"])

    processed = cfg.path("paths", "processed_dir")
    bs = int(ev.get("batch_size", 128))
    nw = int(ev.get("num_workers", 0))

    if model_type == "lstm":
        data = np.load(processed / "sequences.npz")
        X_te, y_te = data["X_test"], data["y_test"]
        ds = SequenceDataset(X_te, y_te)
    else:
        data = np.load(processed / "mlp_features.npz")
        X_te, y_te = data["X_test"], data["y_test"]
        ds = TabularDataset(X_te, y_te)

    loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=nw)
    y_true, y_pred = collect_predictions(model, loader, device)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    reports_dir = ensure_dir(cfg.path("paths", "reports_dir"))
    fig_dir = ensure_dir(cfg.path("paths", "figures_dir"))

    lines = [
        "物联网隐私攻击基线 — 测试集评估结果",
        f"模型文件: {model_path}",
        f"模型类型: {model_type}",
        "",
        f"Accuracy:  {acc:.4f}",
        f"Precision (macro): {prec:.4f}",
        f"Recall (macro):    {rec:.4f}",
        f"F1-score (macro):  {f1:.4f}",
        "",
        "=== classification_report ===",
        report,
    ]
    report_txt = "\n".join(lines)
    rep_path = reports_dir / "classification_report.txt"
    with open(rep_path, "w", encoding="utf-8") as f:
        f.write(report_txt)

    metrics_path = reports_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "accuracy": acc,
                "precision_macro": prec,
                "recall_macro": rec,
                "f1_macro": f1,
                "model_path": str(model_path),
                "model_type": model_type,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    _plot_confusion(cm, class_names, fig_dir / "confusion_matrix.png")

    logger.info("评估完成：\n%s", report_txt)
    logger.info("已写入: %s, %s, %s", rep_path, metrics_path, fig_dir / "confusion_matrix.png")


def _plot_confusion(cm: np.ndarray, class_names: List[str], out_path: Path) -> None:
    configure_matplotlib_english()
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
