# Artifact Layout

The canonical artifact layout encodes experiment choices in the path itself.

## Source Experiments

Normal experiment combination:

`outputs/experiments/{dataset}/seed_{seed}/{model}/{method}/{mode}/`

Each combination contains:

- `metrics.json`
- `confusion.json`
- `classification_report.txt`
- `trace.json`
- `defense_report.json`
- `source_manifest.json`

Baseline artifacts:

`outputs/experiments/{dataset}/seed_{seed}/{model}/baseline/`

Parameter scans:

`outputs/experiments/{dataset}/seed_{seed}/{model}/{method}/{mode}/parameter_scan/`

Adaptive LDP profile scans contain `profile_config.json` in addition to `comparison_results.csv`, `scan_summary.json`, and `scan_trace.json`.

## Datasets and Models

Supported dataset slots:

- `mock`
- `uci_har`
- `kasteren`
- `casas_hh101`
- `cooja`

Supported model slots:

- `lstm`
- `mlp`
- `random_forest` for Cooja only

## Cooja

Cooja uses:

`outputs/experiments/cooja/seed_{seed}/random_forest/{dummy_method}/{mode}/`

This migration only normalizes paths. It does not fabricate packet counts, byte counts, IAT fields, real energy, or real end-to-end delay.

## Summaries and Figures

Final summaries:

`outputs/summaries/final_thesis/`

Final summary figures:

`outputs/figures/summaries/final_thesis/`

Per-combination diagnostic figures:

`outputs/figures/experiments/{dataset}/seed_{seed}/{model}/{method}/{mode}/`

## Migration Record

The migration from legacy batch-name paths is documented in:

- `outputs/summaries/layout/migration_map.csv`
- `outputs/summaries/layout/migration_report.md`

Old batch-name paths are compatibility concepts only and should not be used as final citation roots.

## Dashboard

The official browsing and demonstration UI is:

`python -m streamlit run apps/dashboard.py`

It uses the same canonical paths described above. Demo runs write only the selected combination and record history in `outputs/ui/run_history.jsonl`.

## Related Code Layout

Canonical path helpers live in `src/artifacts/canonical_paths.py`. Dashboard-specific browsing helpers live in `src/dashboard/paths.py` and reuse the canonical path constants. Final summary builders and audit scripts should read canonical artifacts instead of legacy batch-name directories.
