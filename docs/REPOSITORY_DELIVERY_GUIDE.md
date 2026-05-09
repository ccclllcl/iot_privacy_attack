# Repository Delivery Guide

This repository now uses artifact paths that expose the experiment options directly.

## Primary Delivery Locations

Use these paths first when reviewing the project:

- `README.md`
- `configs/default.yaml`
- `experiments/`
- `src/`
- `scripts/`
- `apps/dashboard.py`
- `docs/CODE_STRUCTURE.md`
- `outputs/experiments/`
- `outputs/summaries/final_thesis/`
- `outputs/figures/summaries/final_thesis/`
- `docs/ARTIFACT_LAYOUT.md`

## Preferred Thesis References

For thesis tables, figures, and final numeric claims, prefer:

- `outputs/summaries/final_thesis/final_summary.csv`
- `outputs/summaries/final_thesis/final_summary.json`
- `outputs/summaries/final_thesis/final_coverage_audit.json`
- `outputs/summaries/final_thesis/final_symmetry_audit.json`
- `outputs/summaries/final_thesis/parameter_scan_coverage_audit.json`
- `outputs/summaries/final_thesis/mock/`
- `outputs/summaries/final_thesis/real/`
- `outputs/summaries/final_thesis/cooja/`
- `outputs/figures/summaries/final_thesis/`

## Canonical Source Artifacts

Core source artifacts live under:

- `outputs/experiments/mock/`
- `outputs/experiments/uci_har/`
- `outputs/experiments/kasteren/`
- `outputs/experiments/casas_hh101/`
- `outputs/experiments/cooja/`

The normal experiment path is:

`outputs/experiments/{dataset}/seed_{seed}/{model}/{method}/{mode}/`

The baseline path is:

`outputs/experiments/{dataset}/seed_{seed}/{model}/baseline/`

Cooja uses `random_forest` as the model slot:

`outputs/experiments/cooja/seed_{seed}/random_forest/{dummy_method}/{mode}/`

## Legacy Paths

Old batch-name paths were migrated and should not be cited as final paths. The migration map is stored at `outputs/summaries/layout/migration_map.csv`, and the narrative report is stored at `outputs/summaries/layout/migration_report.md`.

## Dashboard Entry

Use the dashboard for quick review and single-combination demo runs:

`python -m streamlit run apps/dashboard.py`

The dashboard reads from `outputs/experiments/`, `outputs/summaries/final_thesis/`, and `outputs/figures/summaries/final_thesis/`. It does not import datasets, run full matrices, or run Cooja simulations.

## Code Organization

Implementation code is organized by responsibility:

- `src/core/` for config, utilities, and plotting.
- `src/data/` for preprocessing, features, and datasets.
- `src/models/` for model definitions.
- `src/training/` for training logic.
- `src/evaluation/` for evaluation, defense evaluation, and parameter scans.
- `src/defenses/` for defense algorithms and pipelines.
- `src/dashboard/` for dashboard helpers and runner logic.
- `src/artifacts/` for canonical artifact path helpers.

Root-level `src/*.py` modules are compatibility wrappers only. See `docs/CODE_STRUCTURE.md`.

## Cooja Limitations

- `cooja_summary.csv` and `cooja_per_seed.csv` can be used for attack-accuracy reporting.
- Cooja traffic rows may contain null or NaN packet/byte/IAT fields; read `outputs/summaries/final_thesis/cooja/cooja_limitations.md` before interpreting them.
- Do not claim that real energy consumption or real end-to-end latency has been measured.
- Completed Cooja results may keep local WSL log paths to document the evaluation source.
- For reproduction, copy `configs/cooja_defense_dummy_logs.template.json`, set `COOJA_LOG_ROOT` to the local Cooja log directory, and keep generated outputs under the canonical Cooja experiment paths.
