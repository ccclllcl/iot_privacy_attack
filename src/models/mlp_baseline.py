"""
多层感知机基线：对统计特征向量做非线性分类，用于与 LSTM 对比。
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class MLPBaseline(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_sizes: List[int],
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
