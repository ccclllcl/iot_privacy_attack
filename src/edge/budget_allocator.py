"""
边缘隐私预算裁剪（启发式）：当各窗口逆隐私预算之和超过上限时，按比例收紧各窗口 epsilon，
模拟边缘侧对整条上报流的总体约束。

说明：此为工程可实现的近似调度，形式化 DP 合成界需按具体机制另行推导。
"""

from __future__ import annotations

import numpy as np


def apply_edge_budget_cap(
    epsilon_per_window: np.ndarray,
    inverse_budget_cap: float,
) -> np.ndarray:
    """
    若 sum_i (1/epsilon_i) > inverse_budget_cap，则按比例放大各 epsilon（等价于在总约束下略降低单窗噪声强度），
    使 sum_i (1/epsilon_i') <= inverse_budget_cap。

    参数 inverse_budget_cap：允许的 sum(1/eps) 上界，越大允许更强单窗保护（更小 eps）。
    """
    eps = np.maximum(epsilon_per_window.astype(np.float64), 1e-8)
    inv_sum = float(np.sum(1.0 / eps))
    if inv_sum <= inverse_budget_cap or inverse_budget_cap <= 0:
        return eps.astype(np.float32)
    # 设 eps' = k * eps，则 sum(1/eps') = sum(1/(k*eps)) = (1/k) * inv_sum = cap => k = inv_sum / cap
    k = inv_sum / inverse_budget_cap
    return (eps * k).astype(np.float32)
