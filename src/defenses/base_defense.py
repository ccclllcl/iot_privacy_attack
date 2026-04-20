"""
防御器抽象基类：统一 fit / transform / 持久化接口，便于扩展新机制。

说明：多数扰动防御无需从数据「学习」参数，fit 可为空实现，仅用于接口一致性。
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar

import numpy as np

T = TypeVar("T", bound="BaseDefense")


class BaseDefense(ABC):
    """对形状为 (N, seq_len, n_features) 的时序窗口张量逐样本独立扰动，标签 y 不变。"""

    def __init__(self, feature_names: List[str], config: Dict[str, Any]) -> None:
        self.feature_names = list(feature_names)
        self.config = dict(config)

    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """对输入 X 做扰动，返回同 dtype/shape 的 X'。"""

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "BaseDefense":
        """可选：从数据估计参数；默认无操作。"""
        return self

    def fit_transform(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> np.ndarray:
        return self.fit(X, y).transform(X)

    def save(self, path: Path) -> None:
        """保存类名与超参数，便于实验复现与论文附录说明。"""
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "class": self.__class__.__name__,
            "feature_names": self.feature_names,
            "config": self.config,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls: Type[T], path: Path) -> T:
        """默认加载：仅支持与本类 constructor 签名兼容的子类。"""
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        name = payload.get("class")
        if name != cls.__name__:
            raise ValueError(f"checkpoint 类型为 {name}，期望 {cls.__name__}")
        return cls(
            feature_names=list(payload["feature_names"]),
            config=dict(payload.get("config", {})),
        )

    def _target_feature_indices(self) -> np.ndarray:
        """
        根据 config.apply_to / selected_features 返回需要扰动的特征列下标（形状 (K,)）。
        """
        apply_to = str(self.config.get("apply_to", "all")).lower().strip()
        if apply_to == "all":
            return np.arange(len(self.feature_names), dtype=np.int64)
        selected = list(self.config.get("selected_features", []))
        idx: List[int] = []
        for s in selected:
            if s in self.feature_names:
                idx.append(self.feature_names.index(s))
        if not idx:
            raise ValueError(
                "apply_to=selected 但 selected_features 与 feature_names 无交集，请检查配置。"
            )
        return np.asarray(sorted(set(idx)), dtype=np.int64)

    def _feature_mask(self, n_features: int) -> np.ndarray:
        """(F,) 布尔掩码，True 表示该列参与扰动。"""
        m = np.zeros(n_features, dtype=bool)
        for i in self._target_feature_indices():
            if 0 <= int(i) < n_features:
                m[int(i)] = True
        return m
