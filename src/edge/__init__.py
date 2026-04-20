"""
边缘辅助（仿真）：在「设备上报 → 边缘聚合与策略」流程中协调隐私预算。

真实系统中边缘节点可结合局部统计发布机制参数；此处用轻量数值逻辑模拟，
便于毕设中阐述「设备—边缘协同」而不引入网络栈依赖。
"""

from src.edge.budget_allocator import apply_edge_budget_cap

__all__ = ["apply_edge_budget_cap"]
