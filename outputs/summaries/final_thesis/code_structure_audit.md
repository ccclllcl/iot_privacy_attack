# Code Structure Audit

- Generated at: `2026-05-09T17:46:01`
- Files scanned: `111`

## Category Counts

- `artifact_io`: 2
- `artifact_paths`: 5
- `audit`: 7
- `baseline_evaluation`: 3
- `batch_runner`: 3
- `config_core`: 7
- `cooja_eval`: 5
- `dashboard`: 1
- `dashboard_runner`: 2
- `data_processing`: 2
- `dataset_wrappers`: 2
- `defense_algorithms`: 8
- `defense_evaluation`: 2
- `defense_pipeline`: 1
- `experiment_cli`: 10
- `feature_engineering`: 2
- `final_summary_build`: 2
- `legacy_ui`: 2
- `maintenance`: 14
- `model_definitions`: 3
- `parameter_scan`: 2
- `real_dataset_import`: 16
- `training`: 3
- `unknown`: 7

## src Root Compatibility Wrappers

- config.py: compatibility wrapper for configuration core
- preprocess.py: compatibility wrapper for data processing
- features.py: compatibility wrapper for feature engineering
- dataset.py: compatibility wrapper for Dataset wrappers
- train.py: compatibility wrapper for training
- evaluate.py: compatibility wrapper for baseline evaluation
- defense_eval.py: compatibility wrapper for defense-side attack evaluation
- experiment_compare.py: compatibility wrapper for parameter scans
- dashboard_paths.py/dashboard_io.py/dashboard_runner.py: compatibility wrappers for dashboard helpers
- utils.py/plotting.py: compatibility wrappers for common utilities

## Move/Delete Recommendations

- `src/config.py` -> `src/core/config.py` (move; wrapper needed)
- `src/dashboard_io.py` -> `src/dashboard/io.py` (move; wrapper needed)
- `src/dashboard_paths.py` -> `src/dashboard/paths.py` (move; wrapper needed)
- `src/dashboard_runner.py` -> `src/dashboard/runner.py` (move; wrapper needed)
- `src/dataset.py` -> `src/data/dataset.py` (move; wrapper needed)
- `src/defense_eval.py` -> `src/evaluation/defense_evaluator.py` (move; wrapper needed)
- `src/evaluate.py` -> `src/evaluation/evaluator.py` (move; wrapper needed)
- `src/experiment_compare.py` -> `src/evaluation/comparison.py` (move; wrapper needed)
- `src/features.py` -> `src/data/features.py` (move; wrapper needed)
- `src/plotting.py` -> `src/core/plotting.py` (move; wrapper needed)
- `src/preprocess.py` -> `src/data/preprocess.py` (move; wrapper needed)
- `src/train.py` -> `src/training/trainer.py` (move; wrapper needed)
- `src/utils.py` -> `src/core/utils.py` (move; wrapper needed)
