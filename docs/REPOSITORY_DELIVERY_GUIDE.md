# Repository Delivery Guide

This repository now uses artifact paths that expose the experiment options directly.

## Primary Delivery Locations

Use these paths first when reviewing the project:

- `README.md`
- `configs/default.yaml`
- `experiments/`
- `src/`
- `scripts/`
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

The old batch-name paths were migrated and should not be cited as final paths:

- `full_multiseed` -> `mock`
- `real_public_benchmark/{dataset}` -> `{dataset}`
- `outputs/reports/final_thesis/` -> `outputs/summaries/final_thesis/`
- `outputs/figures/final_thesis/` -> `outputs/figures/summaries/final_thesis/`

The migration map is stored at `outputs/summaries/layout/migration_map.csv`.

## Cooja Limitations

- `cooja_summary.csv` and `cooja_per_seed.csv` can be used for attack-accuracy reporting.
- Cooja traffic rows may contain null or NaN packet/byte/IAT fields; read `outputs/summaries/final_thesis/cooja/cooja_limitations.md` before interpreting them.
- Do not claim that real energy consumption or real end-to-end latency has been measured.
- Completed Cooja results may keep local WSL log paths to document the evaluation source.
- For reproduction, copy `configs/cooja_defense_dummy_logs.template.json`, set `COOJA_LOG_ROOT` to the local Cooja log directory, and keep generated outputs under the canonical Cooja experiment paths.
