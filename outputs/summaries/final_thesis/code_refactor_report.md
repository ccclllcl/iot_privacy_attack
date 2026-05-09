# 代码结构重构报告

本报告记录最终交付阶段的代码结构整理结果。

## 范围

本次只整理代码结构，没有重跑 mock 主矩阵、real 主矩阵、参数扫描或 Cooja 仿真。canonical artifacts 仍保留在：

- `outputs/experiments/`
- `outputs/summaries/final_thesis/`
- `outputs/figures/summaries/final_thesis/`
- `outputs/figures/experiments/`

## 重构前问题

- `src/` 根目录混合了配置、数据处理、特征工程、Dataset、训练、评估、防御评估、参数扫描、Dashboard 工具和通用工具。
- Dashboard 专用工具与核心实验逻辑并列放置。
- 最终汇总脚本和审计脚本都堆在 `scripts/` 根目录。
- 旧 UI 与正式 Dashboard 位于同一层级。
- 部分真实数据脚本没有区分 import 与 benchmark 职责。

## 新 `src` 结构

- `src/core/`：配置、工具和绘图。
- `src/data/`：预处理、特征工程和 Dataset 封装。
- `src/models/`：LSTM 和 MLP 模型定义。
- `src/training/`：训练循环和 checkpoint。
- `src/evaluation/`：baseline 评估、防御评估和参数扫描。
- `src/defenses/`：防御算法和防御流水线。
- `src/edge/`：`adaptive_ldp` 边缘预算分配。
- `src/dashboard/`：Dashboard 路径、IO、运行器和历史记录。
- `src/artifacts/`：canonical artifact 路径和 summary IO。

## 移动的文件

- `src/config.py` -> `src/core/config.py`
- `src/utils.py` -> `src/core/utils.py`
- `src/plotting.py` -> `src/core/plotting.py`
- `src/preprocess.py` -> `src/data/preprocess.py`
- `src/features.py` -> `src/data/features.py`
- `src/dataset.py` -> `src/data/dataset.py`
- `src/train.py` -> `src/training/trainer.py`
- `src/evaluate.py` -> `src/evaluation/evaluator.py`
- `src/defense_eval.py` -> `src/evaluation/defense_evaluator.py`
- `src/experiment_compare.py` -> `src/evaluation/comparison.py`
- `src/dashboard_paths.py` -> `src/dashboard/paths.py`
- `src/dashboard_io.py` -> `src/dashboard/io.py`
- `src/dashboard_runner.py` -> `src/dashboard/runner.py`
- `src/ui_history.py` -> `src/dashboard/history.py`
- `src/defenses/base_defense.py` -> `src/defenses/base.py`
- `apps/ui_app.py` -> `apps/legacy/ui_app.py`
- `scripts/build_final_thesis_results.py` -> `scripts/final_thesis/build_final_thesis_results.py`
- `scripts/audit_experiment_symmetry.py` -> `scripts/audit/audit_experiment_symmetry.py`
- `scripts/audit_repository_bloat.py` -> `scripts/audit/audit_repository_bloat.py`
- 真实数据 import 脚本移动到 `experiments/real_public/imports/`。
- 真实数据 benchmark 脚本移动到 `experiments/real_public/benchmarks/`。
- Cooja 和维护工具整理到 `tools/cooja/` 与 `tools/maintenance/`。

## 删除或下沉

- 删除 `experiments/real_public/run_uci_har_missing_parameter_scans.py`，因为它属于过时补缺脚本。
- 删除 `tools/refresh_web_assets.py`，因为正式 Dashboard 不再依赖 `web_assets/images`。
- 旧 UI 降级为 `apps/legacy/ui_app.py` 下的占位入口。

## import 更新

正式代码已改用分层包，例如：

- `src.core.config`
- `src.data.features`
- `src.training.trainer`
- `src.evaluation.evaluator`
- `src.evaluation.defense_evaluator`
- `src.evaluation.comparison`
- `src.dashboard.paths`
- `src.dashboard.io`
- `src.dashboard.runner`

旧 import 精确扫描已确认正式代码不再依赖旧根模块。

## 兼容 wrapper

旧路径如 `src/config.py`、`src/train.py`、`src/evaluate.py`、`src/experiment_compare.py`、`src/dashboard_paths.py` 和 `src/defenses/base_defense.py` 仍保留为短 wrapper，只 re-export 新模块。

根目录脚本 wrapper 也保留，以下命令仍可运行：

- `python scripts/build_final_thesis_results.py`
- `python scripts/audit_experiment_symmetry.py`
- `python scripts/audit_repository_bloat.py`
- `python scripts/audit_code_structure.py`

## 验证

- `python -m compileall src apps experiments scripts tools`：通过。
- `python scripts/build_final_thesis_results.py`：通过。
- `python scripts/audit_experiment_symmetry.py`：通过。
- `python scripts/audit_repository_bloat.py`：通过。
- `python scripts/audit_code_structure.py`：通过。
- Dashboard import 检查：通过。
- Streamlit Dashboard 启动检查：通过。
- `python experiments/demo/run_dashboard_job.py --help`：通过。
- Dashboard 读取一个 mock confusion、一个 real confusion 和一个 parameter scan CSV：通过。

## 最终结果完整性

- `final_missing_outputs.json`：`[]`
- `parameter_scan_missing_outputs.json`：`[]`
- mock 主矩阵：36/36
- real 主矩阵：108/108
- mock 参数扫描：36/36
- real 参数扫描：108/108
- Cooja canonical：18/18

本次没有重跑实验。
