# Repository Cleanup Report

Generated at: 2026-05-09 14:37 +08:00

## Deleted Files and Directories

The cleanup removed tracked legacy/process artifacts that were not referenced by `final_thesis` source paths:

- `configs/generated_dataset_matrix/`
- `data/processed/dataset_matrix/`
- `data/defended/dataset_matrix/`
- `outputs/defense/dataset_matrix/`
- `outputs/figures/dataset_matrix/`
- `outputs/models/dataset_matrix/`
- `outputs/reports/dataset_matrix/`
- `outputs/reports/dataset_matrix_manifest.json`
- `outputs/reports/full_multiseed/seed_*/metrics.json`

It also removed untracked live parameter-scan logs from `outputs/reports/final_thesis/` because they were local process logs containing machine-specific paths, not final deliverables. The local UI history file `outputs/ui/run_history.jsonl` was removed as a non-delivery run log.

## Deletion Checks

- `scripts/audit_repository_bloat.py` was run before deletion and marked the removed tracked files as unreferenced by `final_thesis`.
- `scripts/audit_experiment_symmetry.py` confirmed the main matrices and parameter scans were complete before cleanup.
- `README.md`, `docs/RESULTS_STRUCTURE.md`, and `outputs/reports/final_thesis/legacy_outputs_note.md` were updated so the deleted `dataset_matrix` paths are no longer presented as the current final delivery scope.
- `scripts/build_final_thesis_results.py` does not read the deleted `dataset_matrix` paths.

## Size Reduction

- Tracked files removed: 408.
- Tracked file size removed: 536,340,031 bytes, about 511.49 MiB.
- Untracked local live logs removed: about 920,190 bytes.

## Retained Source Artifacts

The cleanup kept the source artifacts required for traceability and final summary rebuilds:

- `outputs/defense/full_multiseed/`
- `outputs/defense/real_public_benchmark/`
- `outputs/defense/final_thesis/`
- `data/processed/`
- `data/defended/`
- `outputs/models/`
- `outputs/reports/full_multiseed/` except old seed-level `metrics.json`
- `outputs/reports/real_public_benchmark/`

## Old-Looking Files Kept

Some process outputs remain because `final_thesis` summaries point back to them through `source_file` or because they are canonical parameter-scan source artifacts:

- canonical scan CSVs under `outputs/defense/full_multiseed/`
- canonical scan CSVs under `outputs/defense/real_public_benchmark/`
- final-thesis main matrix source outputs under `outputs/defense/final_thesis/`

## Path Hygiene

- Windows machine-specific absolute paths were removed from generated `final_thesis` reports.
- Remaining local paths are WSL Cooja radio/app log paths in Cooja `source_log_files`; they are documented in `outputs/reports/final_thesis/cooja/cooja_limitations.md` and summarized in `outputs/reports/final_thesis/path_hygiene_remaining.json`.

## Documentation and Ignore Rules

- `.gitignore` now ignores broad runtime outputs while explicitly allowing `outputs/reports/final_thesis/` and `outputs/figures/final_thesis/`.
- `docs/RESULTS_STRUCTURE.md` was rewritten as a short compatibility note.
- `docs/PROJECT_STRUCTURE.md` was updated to note that `run_compare.py` supports `ldp`, `noise`, and `adaptive_ldp`, and that final-thesis outputs are the delivery exception.

## Final Integrity

- No experiment was rerun during this cleanup.
- `final_thesis_v2` was not created.
- `supplemental` was not created.
- `outputs/reports/final_thesis/` remains the final report package.
- `outputs/figures/final_thesis/` remains the final figure package.
- Post-cleanup audits keep the main matrices and parameter scans complete.
