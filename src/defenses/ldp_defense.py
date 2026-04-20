"""
本地差分隐私（LDP）风格扰动：数值通道 Laplace 机制 + 二值通道随机响应（RR）。

论文说明（注释中的公式可直接引用到实验报告）：

1) Laplace 机制（数值）
   对查询 q 释放 q(x) + Lap(Delta_1 / epsilon)，在 (epsilon, 0)-DP 意义下提供保护，
   其中 Delta_1 为查询的 L1 全局敏感度（此处由配置 ldp_sensitivity 给出标量上界代理）。

2) 随机响应（二值）
   对真实比特 B∈{0,1}，以如下概率报告 B'：
     P(B'=1 | B=1) = e^epsilon / (1 + e^epsilon)
     P(B'=1 | B=0) = 1 / (1 + e^epsilon)
   该机制满足 epsilon-局部差分隐私（经典 RR 参数化）。

注意：真实系统中敏感度需结合具体查询与裁剪范围严格推导；本框架为实验演示用途。
"""

from __future__ import annotations

from typing import Any, Dict, List, Set, Tuple

import numpy as np

from src.defenses.base_defense import BaseDefense


class LDPDefense(BaseDefense):
    """对选中特征列：二值列做 RR，其余数值列做 Laplace 加噪。"""

    def __init__(self, feature_names: List[str], config: Dict[str, Any]) -> None:
        super().__init__(feature_names, config)
        self._rng = np.random.default_rng(int(config["random_seed"]))

    def fit(self, X: np.ndarray, y: Any = None) -> "LDPDefense":
        self._rng = np.random.default_rng(int(self.config["random_seed"]))
        return self

    def _binary_index_set(self) -> Set[int]:
        names = set(self.feature_names)
        raw = list(self.config.get("binary_features", []))
        idx = set()
        for s in raw:
            if s in names:
                idx.add(self.feature_names.index(s))
        return idx

    def _numeric_explicit_mode(self) -> Tuple[bool, Set[int]]:
        """
        返回 (是否显式指定数值列集合, 数值列下标集合)。

        - numeric_features 未给出（None）：非显式，等价于「除二值列外全部数值化扰动」。
        - numeric_features: []：显式空集，表示不对任何数值列做 Laplace（仅二值列可 RR）。
        - numeric_features: [..]：仅这些列 Laplace。
        """
        names = set(self.feature_names)
        if "numeric_features" not in self.config:
            raw = None
        else:
            raw = self.config.get("numeric_features")
        if raw is None:
            bi = self._binary_index_set()
            return False, {i for i in range(len(self.feature_names)) if i not in bi}
        if len(raw) == 0:
            return True, set()
        idx = set()
        for s in raw:
            if s in names:
                idx.add(self.feature_names.index(s))
        return True, idx

    def _rr(self, bits: np.ndarray, epsilon: float) -> np.ndarray:
        """bits: 0/1 浮点或整型数组，逐元素独立 RR，输出 float32 0/1。"""
        if epsilon <= 0:
            raise ValueError("epsilon 必须为正数")
        e = float(np.exp(epsilon))
        p1 = e / (1.0 + e)  # P(out=1 | true=1)
        p0 = 1.0 / (1.0 + e)  # P(out=1 | true=0)
        u = self._rng.random(size=bits.shape)
        true_one = bits >= 0.5
        out = np.where(true_one, u < p1, u < p0).astype(np.float32)
        return out

    def transform(self, X: np.ndarray) -> np.ndarray:
        if X.ndim != 3:
            raise ValueError(f"期望 X 形状 (N,T,F)，得到 {X.shape}")
        Xo = X.astype(np.float32, copy=True)
        n_features = Xo.shape[2]
        mask = self._feature_mask(n_features)
        if not np.any(mask):
            return Xo

        epsilon = float(self.config.get("epsilon", 1.0))
        sensitivity = float(self.config.get("ldp_sensitivity", 1.0))
        if epsilon <= 0 or sensitivity <= 0:
            raise ValueError("epsilon 与 ldp_sensitivity 必须为正")

        # Laplace 尺度 b = Delta / epsilon（标量敏感度代理）
        lap_scale = sensitivity / epsilon

        thr = float(self.config.get("binary_threshold", 0.5))
        binary_set = self._binary_index_set()
        numeric_explicit, numeric_set = self._numeric_explicit_mode()

        for f in range(n_features):
            if not mask[f]:
                continue
            col = Xo[:, :, f]
            if f in binary_set:
                bits = (col > thr).astype(np.float32)
                Xo[:, :, f] = self._rr(bits, epsilon)
                continue
            if numeric_explicit and f not in numeric_set:
                continue
            noise = self._rng.laplace(0.0, lap_scale, size=col.shape).astype(
                np.float32
            )
            Xo[:, :, f] = col + noise

        return np.clip(Xo, 0.0, None).astype(np.float32)
