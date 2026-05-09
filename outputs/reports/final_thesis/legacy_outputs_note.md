# Legacy and Process Output Note

The repository intentionally keeps process artifacts and older output paths for traceability. They are not error files.

## Process or Legacy Paths

- `outputs/reports/full_multiseed/`
- `outputs/reports/real_public_benchmark/`
- `outputs/reports/dataset_matrix/`
- `outputs/defense/full_multiseed/`
- `outputs/defense/real_public_benchmark/`
- `configs/generated_*`
- `web_assets/`
- Historical outputs from `apps/ui`

## How to Use Them

- Some files under these paths are source artifacts for `outputs/reports/final_thesis/`.
- Final thesis summaries use `source_file` fields to point back to canonical scan CSVs and source defense outputs.
- Thesis text should cite the `final_thesis` package first, not these process paths.
- If reproducing experiments, use these paths together with `README.md`, `scripts/`, and the relevant experiment entry points.
