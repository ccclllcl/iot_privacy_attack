"""
通用工具：随机种子、设备选择、日志目录创建等。
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """固定 Python / NumPy / PyTorch 随机种子，保证实验可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_torch_device(preference: str = "auto") -> torch.device:
    """
    返回训练/推理设备。

    preference: "auto" | "cpu" | "cuda" | "cuda:0" 等
    """
    pref = (preference or "auto").lower().strip()
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(pref)


def ensure_dir(path: Path) -> Path:
    """创建目录（若不存在）。"""
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(obj: Any, path: Path) -> None:
    """将对象保存为 UTF-8 JSON。"""
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: Path) -> Any:
    """读取 JSON。"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_class_weights(
    y: np.ndarray, num_classes: int
) -> Optional[torch.Tensor]:
    """
    根据整数标签向量计算 CrossEntropyLoss 用 class weight（逆频率归一化）。

    若 num_classes<=1 或样本过少则返回 None。
    """
    if y.size == 0 or num_classes < 2:
        return None
    counts = np.bincount(y.astype(np.int64), minlength=num_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (num_classes * counts)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def describe_split(name: str, y: np.ndarray, class_names: List[str]) -> str:
    """生成数据集类别分布描述字符串（用于日志）。"""
    if y.size == 0:
        return f"{name}: 空集"
    u, c = np.unique(y.astype(int), return_counts=True)
    parts = []
    for ui, ci in zip(u, c):
        label = class_names[int(ui)] if 0 <= int(ui) < len(class_names) else str(int(ui))
        parts.append(f"{label}={int(ci)}")
    return f"{name}: " + ", ".join(parts)
