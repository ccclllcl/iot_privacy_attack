# 代码结构说明

本项目源码已经按职责分层，便于答辩评审、复现实验和后续维护时快速定位代码。

## `src/`

- 配置核心（`src/core`）：负责配置加载、路径解析、通用工具和通用绘图函数。
- 数据处理（`src/data`）：负责 CSV/NPZ 读取、滑窗、异常值处理、统计特征提取和 Dataset 封装。
- 模型定义（`src/models`）：包含 LSTM 分类器和 MLP baseline。
- 训练逻辑（`src/training`）：包含训练循环、早停、checkpoint 写入和训练曲线输出。
- 评估逻辑（`src/evaluation`）：包含 baseline 评估、防御后攻击评估和参数扫描逻辑。
- 防御算法（`src/defenses`）：包含 `noise`、`ldp`、`adaptive_ldp` 和防御流水线。
- 边缘预算（`src/edge`）：包含 `adaptive_ldp` 使用的边缘预算分配逻辑。
- Dashboard 工具（`src/dashboard`）：包含 Dashboard 的路径、IO、绘图、运行器和历史记录工具。
- 产物路径工具（`src/artifacts`）：集中维护 canonical artifact 路径和 summary IO。

旧路径如 `src/config.py`、`src/train.py`、`src/evaluate.py` 仍保留为兼容 wrapper，只负责 re-export 新包中的实现。新代码应优先从上述分层包导入。

## `experiments/`

- `experiments/core/`：单步实验 CLI，包括预处理、训练、评估、防御生成、防御评估、参数比较和混淆矩阵收集。
- `experiments/batches/`：多 seed / 全矩阵批处理入口，仅用于复现实验，不作为日常审查入口。
- `experiments/real_public/imports/`：`uci_har`、`kasteren`、`casas_hh101` 的真实数据导入流程。
- `experiments/real_public/benchmarks/`：真实数据 benchmark 运行和汇总脚本。
- `experiments/cooja/`：Cooja 日志评估和比较脚本。
- `experiments/demo/`：Dashboard 使用的单组合训练/评估 runner。

## `scripts/`

- `scripts/final_thesis/`：最终结果汇总构建逻辑。
- `scripts/audit/`：实验对称性审计、仓库体积审计、代码结构审计、项目文件功能报告生成。
- 根目录脚本保留为兼容入口，以下命令仍可直接运行：
  - `python scripts/build_final_thesis_results.py`
  - `python scripts/audit_experiment_symmetry.py`
  - `python scripts/audit_repository_bloat.py`
  - `python scripts/audit_code_structure.py`
  - `python scripts/generate_project_file_report.py`

这些脚本都是轻量维护或审计命令，不会运行完整实验矩阵。

## `apps/`

- `apps/dashboard.py`：正式 Streamlit Dashboard 入口。
- `apps/legacy/ui_app.py`：旧式 UI 占位入口，仅用于历史说明，不推荐使用。

## `tools/`

- `tools/cooja/`：Cooja 相关外部维护工具。
- `tools/maintenance/`：非日常入口的维护工具。

## 产物路径原则

代码结构重构不改变 canonical artifact layout。最终源产物仍位于 `outputs/experiments/`，最终汇总仍位于 `outputs/summaries/final_thesis/`，最终图像仍位于 `outputs/figures/summaries/final_thesis/`。
