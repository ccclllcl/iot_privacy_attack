# Code Refactor Report

Generated for the final delivery code-structure cleanup.

## Scope

This cleanup reorganized code only. It did not rerun the mock main matrix, real main matrix, parameter scans, or Cooja simulations. Canonical artifacts remain under:

- `outputs/experiments/`
- `outputs/summaries/final_thesis/`
- `outputs/figures/summaries/final_thesis/`
- `outputs/figures/experiments/`

## Pre-Refactor Issues

- `src/` mixed configuration, data preprocessing, features, datasets, training, evaluation, defense evaluation, parameter scanning, dashboard helpers, and common utilities in the package root.
- Dashboard-only helpers were stored beside core experiment logic.
- Final thesis scripts and audit scripts were all in the `scripts/` root.
- The legacy UI still existed at the same level as the official dashboard.
- Some real-public workflow scripts were not grouped by import versus benchmark role.

## New `src` Structure

- `src/core/`: config, utilities, and plotting.
- `src/data/`: preprocessing, feature engineering, and dataset wrappers.
- `src/models/`: LSTM and MLP model definitions.
- `src/training/`: training loop and checkpoint logic.
- `src/evaluation/`: baseline evaluation, defense evaluation, and parameter comparison scans.
- `src/defenses/`: defense algorithms and defense pipeline.
- `src/edge/`: adaptive LDP edge-budget allocation.
- `src/dashboard/`: dashboard paths, IO, runner, and run-history utilities.
- `src/artifacts/`: canonical artifact paths and summary IO helpers.

## Files Moved

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
- Real-public import scripts moved to `experiments/real_public/imports/`.
- Real-public benchmark scripts moved to `experiments/real_public/benchmarks/`.
- Cooja and maintenance tools were grouped under `tools/cooja/` and `tools/maintenance/`.

## Deleted or Retired

- `experiments/real_public/run_uci_har_missing_parameter_scans.py` was removed as an obsolete missing-scan helper.
- `tools/refresh_web_assets.py` was removed because the canonical dashboard no longer uses `web_assets/images`.
- The old UI was downgraded to a small legacy placeholder under `apps/legacy/ui_app.py`.

## Import Updates

Formal code now imports from the structured packages, for example:

- `src.core.config`
- `src.data.features`
- `src.training.trainer`
- `src.evaluation.evaluator`
- `src.evaluation.defense_evaluator`
- `src.evaluation.comparison`
- `src.dashboard.paths`
- `src.dashboard.io`
- `src.dashboard.runner`

Exact old import patterns were scanned and no formal code remains dependent on old root modules.

## Compatibility Wrappers

Short wrappers remain at old paths such as `src/config.py`, `src/train.py`, `src/evaluate.py`, `src/experiment_compare.py`, `src/dashboard_paths.py`, and `src/defenses/base_defense.py`. These wrappers only re-export the new package modules.

Root script wrappers remain so these commands still work:

- `python scripts/build_final_thesis_results.py`
- `python scripts/audit_experiment_symmetry.py`
- `python scripts/audit_repository_bloat.py`
- `python scripts/audit_code_structure.py`

## Verification

- `python -m compileall src apps experiments scripts tools`: passed.
- `python scripts/build_final_thesis_results.py`: passed.
- `python scripts/audit_experiment_symmetry.py`: passed.
- `python scripts/audit_repository_bloat.py`: passed.
- `python scripts/audit_code_structure.py`: passed.
- Dashboard import check: passed.
- Streamlit dashboard start check: passed.
- `python experiments/demo/run_dashboard_job.py --help`: passed.
- Dashboard artifact read check for one mock confusion, one real confusion, and one parameter scan CSV: passed.

## Final Thesis Integrity

- `final_missing_outputs.json`: `[]`
- `parameter_scan_missing_outputs.json`: `[]`
- mock main matrix: 36/36
- real main matrix: 108/108
- mock parameter scans: 36/36
- real parameter scans: 108/108
- Cooja canonical outputs: 18/18

No experiments were rerun.
