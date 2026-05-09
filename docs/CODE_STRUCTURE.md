# Code Structure

The repository source code is organized by responsibility so that final delivery code is easier to inspect and maintain.

## `src/`

- `src/core/`: configuration loading, path-aware config resolution, shared utility functions, and common plotting helpers.
- `src/data/`: CSV/NPZ preprocessing, sliding-window sequence preparation, statistical feature extraction, and PyTorch dataset wrappers.
- `src/models/`: model definitions, including the LSTM classifier and MLP baseline.
- `src/training/`: training loop, early stopping, checkpoint writing, and training curve output.
- `src/evaluation/`: baseline evaluation, model loading, prediction metrics, defense-after-attack evaluation, and parameter scans.
- `src/defenses/`: base defense API, noise defense, LDP defense, adaptive LDP defense, and the defense pipeline.
- `src/edge/`: adaptive LDP edge-budget allocation helpers.
- `src/dashboard/`: dashboard canonical path lookup, artifact IO, plotting helpers, subprocess runner, and run-history utilities.
- `src/artifacts/`: shared canonical artifact paths and summary IO helpers.

Compatibility wrappers remain at old import paths such as `src/config.py`, `src/train.py`, and `src/evaluate.py`. They only re-export the new package modules and are kept so older scripts or notes do not fail immediately. New code should import from the structured packages.

## `experiments/`

- `experiments/core/`: single-step CLI entry points for preprocessing, training, evaluation, defense generation, defense evaluation, comparison scans, and confusion collection.
- `experiments/batches/`: multi-seed and full-matrix batch runners retained for reproducibility. They are not part of routine review.
- `experiments/real_public/imports/`: UCI HAR, van Kasteren, and CASAS import workflows.
- `experiments/real_public/benchmarks/`: real public benchmark runners and summarizers.
- `experiments/cooja/`: Cooja log evaluation and comparison scripts.
- `experiments/demo/`: dashboard-safe single-combination runner.

## `scripts/`

- `scripts/final_thesis/`: final thesis summary builder.
- `scripts/audit/`: experiment symmetry, repository bloat, and code-structure audits.
- Root-level script files are compatibility wrappers so the documented commands still work:
  - `python scripts/build_final_thesis_results.py`
  - `python scripts/audit_experiment_symmetry.py`
  - `python scripts/audit_repository_bloat.py`
  - `python scripts/audit_code_structure.py`

## `apps/`

- `apps/dashboard.py`: official Streamlit dashboard entry point.
- `apps/legacy/ui_app.py`: older command-style UI kept for reference only.

## `tools/`

- `tools/cooja/`: external Cooja maintenance helpers.
- `tools/maintenance/`: optional maintenance utilities that are not required for normal thesis review.

## Artifact Policy

Code refactoring does not change canonical artifact paths. Final source artifacts remain under `outputs/experiments/`, final summaries under `outputs/summaries/final_thesis/`, and final figures under `outputs/figures/summaries/final_thesis/`.
