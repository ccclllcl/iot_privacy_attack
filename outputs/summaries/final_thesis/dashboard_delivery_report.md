# Dashboard Delivery Report

Generated: 2026-05-09

## README Updates

- Replaced old Web UI guidance with `python -m streamlit run apps/dashboard.py`.
- Replaced old output navigation with canonical roots: `outputs/experiments/`, `outputs/summaries/final_thesis/`, `outputs/figures/summaries/final_thesis/`, and `outputs/figures/experiments/`.
- Removed the normal-workflow recommendation to use missing-scan or full-matrix helpers.
- Kept old paths only as migration history through `outputs/summaries/layout/migration_report.md`.

## Removed or Downgraded Files

- Removed `scripts/restructure_artifacts.py`.
- Removed `scripts/audit_artifact_layout.py`.
- Removed `experiments/batches/run_missing_parameter_scans.py`.
- Removed the old simple UI entry.
- Kept `apps/ui_app.py` only as a legacy optional command UI; it is not the recommended entry.

## New Dashboard Files

- `apps/dashboard.py`
- `src/dashboard_paths.py`
- `src/dashboard_io.py`
- `src/dashboard_runner.py`
- `experiments/demo/run_dashboard_job.py`
- `docs/DASHBOARD_GUIDE.md`

## Dashboard Scope

Supported datasets:

- `mock`
- `uci_har`
- `kasteren`
- `casas_hh101`
- `cooja` for browsing only

Supported seeds: `42`, `123`, `2026`.

Supported models:

- `lstm`
- `mlp`
- `random_forest` for Cooja browsing

Supported methods:

- `adaptive_ldp`
- `ldp`
- `noise`
- `dummy_noise`, `dummy_ldp`, `dummy_adaptive_ldp` for Cooja browsing

Supported modes:

- `fixed_attacker`
- `retrain_attacker`

## Artifact Access

The dashboard reads experiment artifacts from `outputs/experiments/`, summaries from `outputs/summaries/final_thesis/`, and figures from `outputs/figures/summaries/final_thesis/` plus `outputs/figures/experiments/`.

It can load JSON, CSV, text reports, summary PNGs, confusion matrices, and parameter scan CSVs. Confusion matrices are plotted directly from `confusion.json`; parameter scans are plotted from each selected `parameter_scan/comparison_results.csv`.

## Train / Evaluate Demo

The dashboard calls `experiments/demo/run_dashboard_job.py` for one selected combination. It does not import data, generate mock data, run Cooja simulations, run full matrices, or fill missing Cooja traffic fields.

Output rules:

- baseline: `outputs/experiments/{dataset}/seed_{seed}/{model}/baseline/`
- defense: `outputs/experiments/{dataset}/seed_{seed}/{model}/{method}/{mode}/`
- models: `outputs/models/{dataset}/seed_{seed}/{model}/...`
- temporary configs: `outputs/ui/tmp_configs/`
- history: `outputs/ui/run_history.jsonl`

`overwrite=false` protects existing artifacts. `overwrite=true` is required to overwrite, and the dashboard requires explicit confirmation.

## Verification

- Dashboard import check passed.
- A mock confusion JSON, a real confusion JSON, and an adaptive_ldp parameter scan CSV were loaded and plotted successfully.
- `overwrite=false` refused to replace an existing baseline artifact.
- A single mock `seed_42` `lstm` baseline train/evaluate demo completed with `max_epochs=1` and `overwrite=true`.
- Final build and symmetry audit still report mock 36/36, real 108/108, mock scans 36/36, real scans 108/108.
- `final_missing_outputs.json` remains `[]`.
- `parameter_scan_missing_outputs.json` remains `[]`.
- Cooja remains browsing/reporting only; packet/byte/IAT, real energy, and real delay were not fabricated.
