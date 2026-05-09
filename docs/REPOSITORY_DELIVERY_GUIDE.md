# Repository Delivery Guide

This repository keeps final thesis artifacts, source experiment outputs, and legacy/process artifacts in place for traceability.

## Primary Delivery Locations

Use these paths first when reviewing the project:

- `README.md`
- `configs/default.yaml`
- `experiments/`
- `src/`
- `scripts/`
- `outputs/reports/final_thesis/`
- `outputs/figures/final_thesis/`

## Preferred Thesis References

For thesis tables, figures, and final numeric claims, prefer:

- `outputs/reports/final_thesis/final_summary.csv`
- `outputs/reports/final_thesis/final_summary.json`
- `outputs/reports/final_thesis/final_coverage_audit.json`
- `outputs/reports/final_thesis/final_symmetry_audit.json`
- `outputs/reports/final_thesis/parameter_scan_coverage_audit.json`
- `outputs/reports/final_thesis/mock/`
- `outputs/reports/final_thesis/real/`
- `outputs/reports/final_thesis/cooja/`
- `outputs/figures/final_thesis/`

## Process Artifacts Kept for Traceability

These paths are retained but are not recommended as direct thesis citation paths:

- `outputs/reports/full_multiseed/`
- `outputs/reports/real_public_benchmark/`
- `outputs/defense/full_multiseed/`
- `outputs/defense/real_public_benchmark/`
- `data/processed/`
- `data/defended/`
- `outputs/models/`

They are kept because:

- `final_thesis` summary rows contain `source_file` references back to source artifacts.
- Parameter-scan summaries trace back to canonical scan CSV files.
- Deleting these paths would break the review and reproduction chain.

## Cooja Limitations

- `cooja_summary.csv` and `cooja_per_seed.csv` can be used for attack-accuracy reporting.
- If `cooja_traffic_metrics.csv` contains null or NaN traffic fields, read `cooja_limitations.md` before interpreting them.
- Do not claim that real energy consumption or real end-to-end latency has been measured.
