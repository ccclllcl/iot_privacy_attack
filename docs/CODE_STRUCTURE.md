# 代码结构说明

代码结构围绕“攻击建模—防御生成—防御后评估—结果展示”的研究流程组织。各模块既对应论文中的实验方法，也对应最终产物中的数据处理、模型训练、攻击评估和可视化环节。

## 1. `src/` 研究流程模块

- 配置核心（`src/core`）：读取 `configs/default.yaml` 和临时配置，提供路径解析、通用工具和绘图函数，是各实验入口的基础层。
- 数据处理（`src/data`）：完成 CSV/NPZ 读取、滑窗、异常值处理、统计特征提取和 Dataset 封装，为 LSTM 与 MLP 攻击模型提供输入。
- 模型定义（`src/models`）：实现 LSTM 分类器和 MLP baseline，用于比较时序模型与统计特征模型的行为识别能力。
- 训练逻辑（`src/training`）：执行 baseline 攻击者训练或 `retrain_attacker` 训练，写入 checkpoint 和训练曲线。
- 评估逻辑（`src/evaluation`）：完成 baseline 评估、防御后攻击评估和参数扫描，将攻击准确率、F1、MSE、MAE、Pearson 等指标写入实验产物。
- 防御算法（`src/defenses`）：实现 `noise`、`ldp`、`adaptive_ldp` 以及防御流水线，负责把 processed data 转换为 defended data。
- 边缘预算（`src/edge`）：为 `adaptive_ldp` 提供边缘预算分配和裁剪逻辑，支持 profile 级消融分析。
- Dashboard 工具（`src/dashboard`）：支持结果浏览、图像绘制、单组合 demo runner 调用和运行历史记录。
- 产物路径工具（`src/artifacts`）：集中维护标准产物路径和 summary IO，使构建脚本、审计脚本和 Dashboard 使用同一套路径规则。

这些模块共同形成攻击—防御闭环：`src/data` 产生模型输入，`src/models` 与 `src/training` 构建攻击者，`src/defenses` 生成防御数据，`src/evaluation` 衡量攻击抑制与数据失真，`src/dashboard` 和汇总脚本展示最终结论。

`src/` 根目录不再保留旧兼容 wrapper。所有正式实现都按职责放入 `src/core/`、`src/data/`、`src/models/`、`src/training/`、`src/evaluation/`、`src/defenses/`、`src/edge/`、`src/dashboard/` 和 `src/artifacts/`。

## 2. `experiments/` 实验入口

- `experiments/core/`：单步实验 CLI，包括预处理、训练、评估、防御生成、防御评估、参数比较和混淆矩阵收集。
- `experiments/real_public/imports/`：`uci_har`、`kasteren`、`casas_hh101` 的真实数据导入流程。
- `experiments/real_public/benchmarks/`：真实数据 benchmark 运行和汇总脚本。
- `experiments/cooja/`：Cooja 日志评估和比较脚本，用于节点侧 dummy 流量功能性验证。
- `experiments/demo/`：Dashboard 调用的单组合训练/评估 runner，用于答辩演示中的局部流程复现。

## 3. `scripts/` 汇总与审计

- `scripts/final_thesis/`：构建最终汇总结果、覆盖审计、参数扫描汇总和最终图像。
- `scripts/audit/`：实验对称性审计、仓库体积审计、代码结构审计、项目文件功能报告生成。
- 正式维护命令直接使用分层脚本：
  - `python scripts/final_thesis/build_final_thesis_results.py`
  - `python scripts/audit/audit_experiment_symmetry.py`
  - `python scripts/audit/audit_repository_bloat.py`
  - `python scripts/audit/audit_code_structure.py`
  - `python scripts/audit/generate_project_file_report.py`

这些脚本用于维护最终结果包和复核材料，不承担论文结论之外的新实验解释。

## 4. `apps/`

- `apps/dashboard.py`：正式 Streamlit Dashboard 入口，面向结果浏览、图表展示、混淆矩阵查看和单组合演示。

## 5. `tools/`

- `tools/cooja/`：Cooja 相关外部维护工具。
- `tools/maintenance/`：非主流程维护工具。

## 6. 产物路径关系

代码结构重构不改变实验结论对应的标准产物路径：

- 源实验产物：`outputs/experiments/`
- 最终汇总：`outputs/summaries/final_thesis/`
- 最终图像：`outputs/figures/summaries/final_thesis/`
- 单组合图像：`outputs/figures/experiments/`

源码模块、实验入口和审计脚本通过这些路径连接到最终论文结果，便于从研究结论追溯到具体 dataset / seed / model / method / mode 组合。
