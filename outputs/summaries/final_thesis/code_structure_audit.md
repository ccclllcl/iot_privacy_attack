# 代码结构审计

- 生成时间：`2026-05-09T19:47:56`
- 扫描文件数：`115`
- 结论：当前代码结构已完成职责分层，暂无必须移动的文件。

## 1. 分类统计

- `artifact_paths`：2
- `audit`：4
- `baseline_evaluation`：1
- `compatibility_wrapper`：27
- `config_core`：3
- `cooja_eval`：4
- `dashboard`：5
- `data_processing`：3
- `defense_algorithms`：5
- `defense_evaluation`：1
- `defense_pipeline`：1
- `docs`：9
- `experiment_cli`：9
- `final_summary_build`：1
- `legacy_file`：3
- `maintenance`：3
- `model_definitions`：2
- `package_init`：24
- `parameter_scan`：1
- `real_dataset_benchmark`：3
- `real_dataset_import`：3
- `training`：1

## 2. 已完成的移动

- `scripts/audit_code_structure.py` -> `scripts/audit/audit_code_structure.py`
- `scripts/audit_experiment_symmetry.py` -> `scripts/audit/audit_experiment_symmetry.py`
- `scripts/audit_repository_bloat.py` -> `scripts/audit/audit_repository_bloat.py`
- `scripts/build_final_thesis_results.py` -> `scripts/final_thesis/build_final_thesis_results.py`
- `scripts/generate_project_file_report.py` -> `scripts/audit/generate_project_file_report.py`
- `src/config.py` -> `src/core/config.py`
- `src/dashboard_io.py` -> `src/dashboard/io.py`
- `src/dashboard_paths.py` -> `src/dashboard/paths.py`
- `src/dashboard_runner.py` -> `src/dashboard/runner.py`
- `src/dataset.py` -> `src/data/dataset.py`
- `src/defense_eval.py` -> `src/evaluation/defense_evaluator.py`
- `src/defenses/base_defense.py` -> `src/defenses/base.py`
- `src/evaluate.py` -> `src/evaluation/evaluator.py`
- `src/experiment_compare.py` -> `src/evaluation/comparison.py`
- `src/features.py` -> `src/data/features.py`
- `src/plotting.py` -> `src/core/plotting.py`
- `src/preprocess.py` -> `src/data/preprocess.py`
- `src/train.py` -> `src/training/trainer.py`
- `src/ui_history.py` -> `src/dashboard/history.py`
- `src/utils.py` -> `src/core/utils.py`

## 3. 兼容 wrapper

- `scripts/audit_code_structure.py`：指向 `scripts/audit/audit_code_structure.py`。
- `scripts/audit_experiment_symmetry.py`：指向 `scripts/audit/audit_experiment_symmetry.py`。
- `scripts/audit_repository_bloat.py`：指向 `scripts/audit/audit_repository_bloat.py`。
- `scripts/build_final_thesis_results.py`：指向 `scripts/final_thesis/build_final_thesis_results.py`。
- `scripts/generate_project_file_report.py`：指向 `scripts/audit/generate_project_file_report.py`。
- `src/config.py`：指向 `src/core/config.py`。
- `src/dashboard_io.py`：指向 `src/dashboard/io.py`。
- `src/dashboard_paths.py`：指向 `src/dashboard/paths.py`。
- `src/dashboard_runner.py`：指向 `src/dashboard/runner.py`。
- `src/dataset.py`：指向 `src/data/dataset.py`。
- `src/defense_eval.py`：指向 `src/evaluation/defense_evaluator.py`。
- `src/defenses/base_defense.py`：指向 `src/defenses/base.py`。
- `src/evaluate.py`：指向 `src/evaluation/evaluator.py`。
- `src/experiment_compare.py`：指向 `src/evaluation/comparison.py`。
- `src/features.py`：指向 `src/data/features.py`。
- `src/plotting.py`：指向 `src/core/plotting.py`。
- `src/preprocess.py`：指向 `src/data/preprocess.py`。
- `src/train.py`：指向 `src/training/trainer.py`。
- `src/ui_history.py`：指向 `src/dashboard/history.py`。
- `src/utils.py`：指向 `src/core/utils.py`。

## 4. legacy 文件

- `apps/legacy/ui_app.py`：旧 UI 占位入口，不作为正式 Dashboard。
- `experiments/batches/run_all_data_multiseed.py`：批处理复现脚本，日常审查不运行。
- `experiments/batches/run_all_methods_multiseed.py`：完整矩阵复现脚本，日常审查不运行。

## 5. unknown 文件

- 无 unknown 文件。

## 6. 仍待处理建议

- 当前代码结构已完成职责分层，暂无必须移动的文件。
