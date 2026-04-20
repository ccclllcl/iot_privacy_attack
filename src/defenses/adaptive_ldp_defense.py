"""
自适应本地差分隐私（仿真）：按「数据敏感度」与「流量强度代理」为每个时间窗口动态分配 epsilon，
再对数值通道施加 Laplace、对二值通道施加随机响应。

与毕设定位的对应关系：
- 「流量分析」侧信道：用窗口内总能量/活动量（L1 和）刻画上报强度，类比流量负载；
- 「敏感度」：用窗口内时序波动（std）刻画数据离散程度，高波动窗口分配更小 epsilon（更强噪声）；
- 「边缘辅助」：可选调用 edge.budget_allocator，在边缘侧对整批窗口的 epsilon 序列做总预算裁剪。

注：形式化 DP 保证需结合具体机制与合成定理；本实现提供可复现实验与论文中的机制描述接口。
"""

from __future__ import annotations

from typing import Any, Dict, List, Set, Tuple

import numpy as np

from src.defenses.base_defense import BaseDefense
from src.edge.budget_allocator import apply_edge_budget_cap


class AdaptiveLDPDefense(BaseDefense):
    """按窗口动态 epsilon 的 LDP 风格扰动；fit 阶段在边缘/全量数据上标定分位数。"""

    def __init__(self, feature_names: List[str], config: Dict[str, Any]) -> None:
        super().__init__(feature_names, config)
        self._rng = np.random.default_rng(int(config["random_seed"]))
        self._s_lo = self._s_hi = 0.0
        self._t_lo = self._t_hi = 0.0
        self._fitted = False

    def fit(self, X: np.ndarray, y: Any = None) -> "AdaptiveLDPDefense":
        self._rng = np.random.default_rng(int(self.config["random_seed"]))
        ac = self._adaptive_cfg()
        p_lo, p_hi = float(ac.get("calibration_percentile_low", 5)), float(
            ac.get("calibration_percentile_high", 95)
        )
        if X.ndim != 3:
            raise ValueError(f"期望 X 形状 (N,T,F)，得到 {X.shape}")
        s_list = [float(np.std(X[i])) for i in range(X.shape[0])]
        t_list = [float(np.sum(np.abs(X[i]))) for i in range(X.shape[0])]
        self._s_lo, self._s_hi = np.percentile(s_list, [p_lo, p_hi])
        self._t_lo, self._t_hi = np.percentile(t_list, [p_lo, p_hi])
        self._fitted = True
        return self

    def _adaptive_cfg(self) -> Dict[str, Any]:
        return dict(self.config.get("adaptive_ldp", {}))

    def _binary_index_set(self) -> Set[int]:
        names = set(self.feature_names)
        idx: Set[int] = set()
        for s in list(self.config.get("binary_features", [])):
            if s in names:
                idx.add(self.feature_names.index(s))
        return idx

    def _numeric_explicit_mode(self) -> Tuple[bool, Set[int]]:
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
        if epsilon <= 0:
            raise ValueError("epsilon 必须为正数")
        e = float(np.exp(epsilon))
        p1 = e / (1.0 + e)
        p0 = 1.0 / (1.0 + e)
        u = self._rng.random(size=bits.shape)
        true_one = bits >= 0.5
        return np.where(true_one, u < p1, u < p0).astype(np.float32)

    def _norm(self, v: float, lo: float, hi: float) -> float:
        if hi - lo < 1e-12:
            return 0.5
        return float(np.clip((v - lo) / (hi - lo + 1e-12), 0.0, 1.0))

    def _epsilon_per_window(self, X: np.ndarray) -> np.ndarray:
        ac = self._adaptive_cfg()
        eps_min = float(ac.get("epsilon_min", 0.3))
        eps_max = float(ac.get("epsilon_max", 3.0))
        w_s = float(ac.get("weight_sensitivity", 0.5))
        w_t = float(ac.get("weight_traffic", 0.5))
        wsum = w_s + w_t
        if wsum > 0:
            w_s, w_t = w_s / wsum, w_t / wsum
        if not self._fitted:
            raise RuntimeError("请先对 AdaptiveLDPDefense 调用 fit（防御流水线已自动调用）。")

        n = X.shape[0]
        eps_list = np.zeros(n, dtype=np.float64)
        for i in range(n):
            w = X[i]
            s_i = float(np.std(w))
            t_i = float(np.sum(np.abs(w)))
            s_n = self._norm(s_i, self._s_lo, self._s_hi)
            t_n = self._norm(t_i, self._t_lo, self._t_hi)
            risk = w_s * s_n + w_t * t_n
            # 风险高 -> epsilon 取更小（更强隐私）；线性插值
            eps_list[i] = eps_max - risk * (eps_max - eps_min)

        eps_list = np.maximum(eps_list, eps_min)
        eps_list = np.minimum(eps_list, eps_max)

        if bool(ac.get("use_edge_budget_cap", False)):
            cap = float(ac.get("edge_inverse_budget_cap", 1.0e9))
            eps_list = apply_edge_budget_cap(eps_list.astype(np.float32), cap).astype(np.float64)
        return eps_list

    def transform(self, X: np.ndarray) -> np.ndarray:
        if X.ndim != 3:
            raise ValueError(f"期望 X 形状 (N,T,F)，得到 {X.shape}")
        Xo = X.astype(np.float32, copy=True)
        n_features = Xo.shape[2]
        mask = self._feature_mask(n_features)
        if not np.any(mask):
            return Xo

        sensitivity = float(self.config.get("ldp_sensitivity", 1.0))
        if sensitivity <= 0:
            raise ValueError("ldp_sensitivity 必须为正")

        thr = float(self.config.get("binary_threshold", 0.5))
        binary_set = self._binary_index_set()
        numeric_explicit, numeric_set = self._numeric_explicit_mode()

        eps_vec = self._epsilon_per_window(X)
        n = Xo.shape[0]
        n_features = Xo.shape[2]

        for i in range(n):
            eps = float(eps_vec[i])
            lap_scale = sensitivity / eps
            for f in range(n_features):
                if not mask[f]:
                    continue
                col = Xo[i, :, f]
                if f in binary_set:
                    bits = (col > thr).astype(np.float32)
                    Xo[i, :, f] = self._rr(bits, eps)
                elif numeric_explicit and f not in numeric_set:
                    continue
                else:
                    noise = self._rng.laplace(0.0, lap_scale, size=col.shape).astype(np.float32)
                    Xo[i, :, f] = col + noise

        return np.clip(Xo, 0.0, None).astype(np.float32)
