"""
PyTorch Dataset：支持 LSTM 原始序列输入与 MLP 统计特征输入。
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class SequenceDataset(Dataset):
    """时序分类数据集，X: (N, seq_len, feat_dim)，y: (N,) 整数标签。"""

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        if X.shape[0] != y.shape[0]:
            raise ValueError("X 与 y 样本数不一致")
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self.X[idx])
        t = torch.tensor(self.y[idx], dtype=torch.long)
        return x, t


class TabularDataset(Dataset):
    """表格特征数据集，用于 MLP：X: (N, d)，y: (N,)。"""

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        if X.shape[0] != y.shape[0]:
            raise ValueError("X 与 y 样本数不一致")
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self.X[idx])
        t = torch.tensor(self.y[idx], dtype=torch.long)
        return x, t
