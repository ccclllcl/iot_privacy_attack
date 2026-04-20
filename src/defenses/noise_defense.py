"""
加性噪声防御：对选定特征加入高斯或拉普拉斯噪声。

适用于快速基线对比；不提供形式化差分隐私保证，但可通过噪声强度调节隐私-效用折中。
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from src.defenses.base_defense import BaseDefense


class NoiseDefense(BaseDefense):
    """
    对每个样本窗口 X[i, t, f] 独立加噪（按窗口扰动）。

    noise_type:
      - gaussian: 加 N(0, noise_scale^2)
      - laplace:  加 Laplace(0, noise_scale)，概率密度 p(x)=1/(2b)exp(-|x|/b)
    """

    def __init__(self, feature_names: List[str], config: Dict[str, Any]) -> None:
        super().__init__(feature_names, config)
        self._rng = np.random.default_rng(int(config["random_seed"]))

    def fit(self, X: np.ndarray, y: Any = None) -> "NoiseDefense":
        # 固定种子下可复现；若希望 fit 后再 transform 用不同噪声，可改此处重设 rng
        self._rng = np.random.default_rng(int(self.config["random_seed"]))
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if X.ndim != 3:
            raise ValueError(f"期望 X 形状 (N,T,F)，得到 {X.shape}")
        X = X.astype(np.float32, copy=True)
        n_features = X.shape[2]
        mask = self._feature_mask(n_features)
        if not np.any(mask):
            return X

        noise_type = str(self.config.get("noise_type", "gaussian")).lower().strip()
        scale = float(self.config.get("noise_scale", 0.1))
        if scale < 0:
            raise ValueError("noise_scale 必须非负")

        sub = X[:, :, mask]
        if noise_type == "gaussian":
            noise = self._rng.normal(loc=0.0, scale=scale, size=sub.shape).astype(
                np.float32
            )
        elif noise_type == "laplace":
            # numpy 无 laplace，使用指数分布差分构造
            noise = self._rng.laplace(loc=0.0, scale=scale, size=sub.shape).astype(
                np.float32
            )
        else:
            raise ValueError(f"未知 noise_type: {noise_type}")

        X[:, :, mask] = sub + noise
        return X
