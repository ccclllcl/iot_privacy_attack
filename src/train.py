"""
训练脚本逻辑：支持 LSTM / MLP、类别权重、早停、曲线与最佳模型保存。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from src.plotting import configure_matplotlib_english
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import ExperimentConfig
from src.dataset import SequenceDataset, TabularDataset
from src.models.lstm_classifier import LSTMClassifier
from src.models.mlp_baseline import MLPBaseline
from src.utils import (
    compute_class_weights,
    describe_split,
    ensure_dir,
    get_torch_device,
    set_seed,
)

logger = logging.getLogger(__name__)


@dataclass
class TrainHistory:
    epoch: List[int]
    train_loss: List[float]
    val_loss: List[float]
    val_acc: List[float]


def _accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = logits.argmax(dim=-1)
    return (pred == y).float().mean().item()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total = 0.0
    n = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total += float(loss.item()) * xb.size(0)
        n += xb.size(0)
    return total / max(n, 1)


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    total = 0.0
    n = 0
    correct = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        total += float(loss.item()) * xb.size(0)
        n += xb.size(0)
        pred = logits.argmax(dim=-1)
        correct += int((pred == yb).sum().item())
    acc = correct / max(n, 1)
    return total / max(n, 1), acc


def run_training(
    cfg: ExperimentConfig,
    model_type: str,
    override_model_path: Optional[Path] = None,
    sequences_npz: Optional[Path] = None,
    mlp_npz: Optional[Path] = None,
    curve_output_path: Optional[Path] = None,
    history_output_path: Optional[Path] = None,
) -> Path:
    """
    从预处理结果训练模型，返回保存的最佳模型路径。

    sequences_npz / mlp_npz:
        若为 None，则分别使用 data/processed/sequences.npz 与 mlp_features.npz；
        防御实验「重训攻击者」模式可指向 data/defended/defended_sequences.npz 等。
    """
    model_type = model_type.lower().strip()
    if model_type not in ("lstm", "mlp"):
        raise ValueError("model_type 只能是 lstm 或 mlp")

    pp = cfg.nested("preprocess")
    tr = cfg.nested("train")
    seed = cfg.random_seed()
    set_seed(seed)

    processed = cfg.path("paths", "processed_dir")
    meta_path = processed / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(
            f"未找到 {meta_path}，请先运行: python run_preprocess.py --config ..."
        )
    with open(meta_path, "r", encoding="utf-8") as f:
        meta: Dict[str, Any] = json.load(f)

    class_names: List[str] = list(meta["class_names"])
    num_classes = len(class_names)

    seq_file = sequences_npz if sequences_npz is not None else (processed / "sequences.npz")
    mlp_file = mlp_npz if mlp_npz is not None else (processed / "mlp_features.npz")

    if model_type == "lstm":
        if not seq_file.is_file():
            raise FileNotFoundError(f"未找到序列数据: {seq_file}")
        data = np.load(seq_file)
        X_train, y_train = data["X_train"], data["y_train"]
        X_val, y_val = data["X_val"], data["y_val"]
        ds_tr = SequenceDataset(X_train, y_train)
        ds_va = SequenceDataset(X_val, y_val)
        in_dim = int(X_train.shape[2])
        model: nn.Module = LSTMClassifier(
            input_dim=in_dim,
            num_classes=num_classes,
            hidden_size=int(tr.get("lstm_hidden_size", 128)),
            num_layers=int(tr.get("lstm_num_layers", 2)),
            dropout=float(tr.get("dropout", 0.3)),
        )
    else:
        if not mlp_file.is_file():
            raise FileNotFoundError(
                f"未找到 {mlp_file}。请在配置 preprocess.save_mlp_features=true 并重新预处理，或运行防御流水线生成 defended_mlp_features.npz。"
            )
        data = np.load(mlp_file)
        X_train, y_train = data["X_train"], data["y_train"]
        X_val, y_val = data["X_val"], data["y_val"]
        ds_tr = TabularDataset(X_train, y_train)
        ds_va = TabularDataset(X_val, y_val)
        in_dim = int(X_train.shape[1])
        hidden = list(tr.get("mlp_hidden_sizes", [256, 128]))
        model = MLPBaseline(
            input_dim=in_dim,
            num_classes=num_classes,
            hidden_sizes=hidden,
            dropout=float(tr.get("dropout", 0.3)),
        )

    device_pref = str(tr.get("device", "auto"))
    device = get_torch_device(device_pref)
    model = model.to(device)

    logger.info("设备: %s", device)
    logger.info(describe_split("train", y_train, class_names))
    logger.info(describe_split("val", y_val, class_names))

    bs = int(tr.get("batch_size", 64))
    nw = int(tr.get("num_workers", 0))
    loader_tr = DataLoader(ds_tr, batch_size=bs, shuffle=True, num_workers=nw)
    loader_va = DataLoader(ds_va, batch_size=bs, shuffle=False, num_workers=nw)

    use_w = bool(tr.get("use_class_weights", True))
    w_tensor: Optional[torch.Tensor] = None
    if use_w:
        w_tensor = compute_class_weights(y_train, num_classes)
        if w_tensor is not None:
            w_tensor = w_tensor.to(device)

    criterion = nn.CrossEntropyLoss(weight=w_tensor)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(tr.get("learning_rate", 1e-3)),
        weight_decay=float(tr.get("weight_decay", 1e-4)),
    )

    epochs = int(tr.get("num_epochs", 80))
    patience = int(tr.get("early_stopping_patience", 10))
    min_delta = float(tr.get("early_stopping_min_delta", 1e-4))

    history = TrainHistory(epoch=[], train_loss=[], val_loss=[], val_acc=[])

    best_val = float("inf")
    best_state = None
    stall = 0

    models_dir = ensure_dir(cfg.path("paths", "models_dir"))
    default_name = f"best_{model_type}.pt"
    save_path = override_model_path if override_model_path else (models_dir / default_name)

    for ep in tqdm(range(1, epochs + 1), desc="Training", unit="epoch"):
        tl = train_one_epoch(model, loader_tr, optimizer, criterion, device)
        vl, va = evaluate_epoch(model, loader_va, criterion, device)
        history.epoch.append(ep)
        history.train_loss.append(tl)
        history.val_loss.append(vl)
        history.val_acc.append(va)

        logger.info(
            "Epoch %03d | train_loss=%.4f | val_loss=%.4f | val_acc=%.4f",
            ep,
            tl,
            vl,
            va,
        )

        improved = vl < (best_val - min_delta)
        if improved:
            best_val = vl
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            stall = 0
        else:
            stall += 1

        if stall >= patience:
            logger.info("早停触发：patience=%d，最佳 val_loss=%.4f", patience, best_val)
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    ensure_dir(save_path.parent)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_type": model_type,
            "class_names": class_names,
            "num_classes": num_classes,
            "input_dim": in_dim,
            "meta": meta,
            "train_config": tr,
        },
        save_path,
    )

    fig_dir = ensure_dir(cfg.path("paths", "figures_dir"))
    curve_path = curve_output_path if curve_output_path else (fig_dir / "train_curve.png")
    ensure_dir(curve_path.parent)
    _plot_curves(history, curve_path)

    hist_path = (
        history_output_path
        if history_output_path
        else (fig_dir / "train_history.json")
    )
    ensure_dir(hist_path.parent)
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(asdict(history), f, indent=2)

    logger.info("训练结束，最佳模型已保存: %s", save_path)
    return save_path


def _plot_curves(history: TrainHistory, out_path: Path) -> None:
    """Plot training / validation loss and validation accuracy (English labels for figures)."""
    configure_matplotlib_english()
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(history.epoch, history.train_loss, label="train_loss", color="C0")
    ax1.plot(history.epoch, history.val_loss, label="val_loss", color="C1")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(history.epoch, history.val_acc, label="val_acc", color="C2", linestyle="--")
    ax2.set_ylabel("Val Accuracy")

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="center right")
    plt.title("Training curves (attacker baseline)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
