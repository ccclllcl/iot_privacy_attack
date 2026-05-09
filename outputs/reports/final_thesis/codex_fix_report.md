# Codex Fix Report

Generated at: 2026-05-09 13:30 +08:00

## Code changes

- Added `scripts/audit_experiment_symmetry.py` for read-only symmetry auditing.
- Added `experiments/batches/run_missing_parameter_scans.py` for audit-driven parameter-scan completion. The batch runner has no hard wall-clock timeout for model training.
- Updated `experiments/core/run_compare.py` and `src/experiment_compare.py` so `--method adaptive_ldp` runs the six configured adaptive profiles.
- Updated `scripts/build_final_thesis_results.py` to aggregate canonical scan files once, fall back to legacy files only when needed, export Cooja per-seed/traffic CSVs, and refresh final-thesis figures.
- Updated `experiments/cooja/run_cooja_defense_eval.py` with compatible Cooja log path resolution through `COOJA_LOG_ROOT` while preserving the existing manifest.
- Updated `README.md` with only the requested final-result symmetry section.

## Result files added or refreshed

- `outputs/reports/final_thesis/final_symmetry_audit.json`
- `outputs/reports/final_thesis/final_symmetry_audit.md`
- `outputs/reports/final_thesis/parameter_scan_coverage_audit.json`
- `outputs/reports/final_thesis/parameter_scan_missing_outputs.json`
- `outputs/reports/final_thesis/parameter_scan_run_log.json`
- `outputs/reports/final_thesis/mock/mock_parameter_scan_ldp.csv`
- `outputs/reports/final_thesis/mock/mock_parameter_scan_noise.csv`
- `outputs/reports/final_thesis/mock/mock_parameter_scan_adaptive_ldp.csv`
- `outputs/reports/final_thesis/real/real_parameter_scan_ldp.csv`
- `outputs/reports/final_thesis/real/real_parameter_scan_noise.csv`
- `outputs/reports/final_thesis/real/real_parameter_scan_adaptive_ldp.csv`
- `outputs/reports/final_thesis/cooja/cooja_per_seed.csv`
- `outputs/reports/final_thesis/cooja/cooja_traffic_metrics.csv`
- `outputs/reports/final_thesis/cooja/cooja_limitations.md`
- `outputs/figures/final_thesis/parameter_scan_ldp_all_models_modes.png`
- `outputs/figures/final_thesis/parameter_scan_noise_all_models_modes.png`
- `outputs/figures/final_thesis/parameter_scan_adaptive_ldp_all_models_modes.png`
- `outputs/figures/final_thesis/cooja_per_seed_accuracy.png`
- `outputs/figures/final_thesis/cooja_traffic_metrics.png`

Canonical parameter-scan CSVs now exist at the original symmetric locations:

- Mock: `outputs/defense/full_multiseed/seed_{seed}/{method}/comparisons/{model}_{mode}_comparison_results.csv` = 36/36.
- Real: `outputs/defense/real_public_benchmark/{dataset}/seed_{seed}/{method}/comparisons/{model}_{mode}_comparison_results.csv` = 108/108.

## Reused existing outputs

- Mock main matrix was reused as-is: `outputs/reports/final_thesis/mock/mock_coverage_audit.json` remains 36/36.
- Real main matrix was reused as-is: `outputs/reports/final_thesis/real/real_coverage_audit.json` remains 108/108.
- `outputs/reports/final_thesis/final_missing_outputs.json` remains present and contains `[]`.
- Existing readable canonical scans and legacy `comparison_results.csv` files were not rerun. Legacy files were retained for compatibility.
- Cooja used the existing `outputs/defense/final_thesis/cooja/eval/defense_eval_report.json`; no Cooja simulation was rerun.

## Parameter scans completed in this task

- Mock parameter scans were completed/normalized to canonical filenames for all seeds, methods, models, and modes.
- Real parameter-scan completion generated the remaining 42 real retrain CSVs recorded in `parameter_scan_run_log.json`:
  - `uci_har`: `adaptive_ldp` retrain scans for LSTM and MLP across seeds 42, 123, 2026.
  - `kasteren`: `adaptive_ldp`, `ldp`, and `noise` retrain scans for LSTM and MLP across seeds 42, 123, 2026.
  - `casas_hh101`: `adaptive_ldp`, `ldp`, and `noise` retrain scans for LSTM and MLP across seeds 42, 123, 2026.

## Remaining missing combinations

- None. `final_symmetry_audit.json` reports:
  - `missing_mock_parameter_scans`: 0
  - `missing_real_parameter_scans`: 0
  - `missing_cooja_outputs`: 0
- `parameter_scan_missing_outputs.json` contains `[]`.

## Cooja logs and limitations

- Cooja logs were accessible through the existing configured paths, so `cooja_missing_logs.json` was not needed.
- `cooja_per_seed.csv` contains 18 per-seed rows.
- `cooja_traffic_metrics.csv` contains 9 traffic rows.
- Energy and delay are not fabricated: `energy_metric_available=false` and `delay_metric_available=false`.
- Radio logs do not distinguish dummy packets from real packets; dummy packet and byte ratios are reported as null and documented in `cooja_limitations.md`.

## Duplicate scan rows

- Final aggregation avoids reading both canonical and legacy scan files for the same LSTM fixed-attacker result.
- Validation after rebuild:
  - `mock_parameter_scan_ldp.csv`: 60 rows, 0 duplicate key rows.
  - `mock_parameter_scan_noise.csv`: 48 rows, 0 duplicate key rows.
- `parameter_scan_coverage_audit.json` reports `duplicate_rows_removed=0` for the final canonical aggregation pass.

## Delivery status

- No `final_thesis_v2` directory was created.
- No `supplemental` directory was created.
- During the symmetry-completion pass, no old outputs were deleted; legacy cleanup is documented in the later repository bloat cleanup section.
- Main matrices remain complete: mock 36/36 and real 108/108.
- Parameter-scan symmetry is complete: mock 36/36 and real 108/108.
- Cooja aggregate, per-seed, and traffic-report outputs are present.
- Current project reaches the symmetric-output requirement for undergraduate thesis delivery.

## Repository cleanup and adaptive ablation finalization

- Real-data parameter scans were rechecked and remain complete: 108/108.
- adaptive_ldp profile coverage was rechecked: each mock/real dataset, seed, model, and mode combination has the expected 6 profiles.
- Formal adaptive_ldp ablation summaries were generated from existing parameter-scan CSVs; no main matrix or completed scan was rerun.
- `outputs/reports/final_thesis/artifact_index.md` was added as the final artifact index.
- `docs/REPOSITORY_DELIVERY_GUIDE.md` was added as the repository delivery guide.
- No `final_thesis_v2` directory was created.
- No `supplemental` directory was created.
- Core traceability artifacts were retained, including `outputs/defense/full_multiseed/`, `outputs/defense/real_public_benchmark/`, `data/processed/`, `data/defended/`, and `outputs/models/`.
- Legacy/source paths remain available as process artifacts: `outputs/reports/full_multiseed/`, `outputs/reports/real_public_benchmark/`, `outputs/defense/full_multiseed/`, `outputs/defense/real_public_benchmark/`, `configs/generated_*`, `web_assets/`, and historical `apps/ui` outputs.
- Current repository structure is suitable for undergraduate thesis delivery: final references are centralized under `outputs/reports/final_thesis/` and `outputs/figures/final_thesis/`, while source artifacts remain available for review.

## Repository bloat cleanup

- No experiment was rerun in this cleanup pass.
- Removed the old tracked `dataset_matrix` process package: `configs/generated_dataset_matrix/`, `data/processed/dataset_matrix/`, `data/defended/dataset_matrix/`, `outputs/defense/dataset_matrix/`, `outputs/figures/dataset_matrix/`, `outputs/models/dataset_matrix/`, `outputs/reports/dataset_matrix/`, and `outputs/reports/dataset_matrix_manifest.json`.
- Removed old seed-level single-run reports: `outputs/reports/full_multiseed/seed_*/metrics.json`.
- Removed untracked local live parameter-scan logs from `outputs/reports/final_thesis/` because they were not final deliverables and contained local absolute paths.
- Removed local UI history: `outputs/ui/run_history.jsonl`.
- Tracked cleanup removed 408 files, about 511.49 MiB.
- `final_thesis` remains complete: mock main 36/36, real main 108/108, mock scans 36/36, real scans 108/108, `final_missing_outputs.json=[]`, and `parameter_scan_missing_outputs.json=[]`.
- `.gitignore` now ignores broad runtime outputs while explicitly allowing `outputs/reports/final_thesis/` and `outputs/figures/final_thesis/`.
- Retained source artifacts for traceability include `outputs/defense/full_multiseed/`, `outputs/defense/real_public_benchmark/`, `outputs/defense/final_thesis/`, `data/processed/`, `data/defended/`, and `outputs/models/`.
- Cooja traffic limitations remain documented: traffic proxy fields may be NaN, dummy ratios are not fabricated, and no real energy or end-to-end delay is claimed.

## Final delivery consistency cleanup

- No experiment was rerun in this final consistency pass.
- Updated `final_thesis_summary.md` generation so it no longer describes Kasteren/CASAS parameter scans as future work; the summary now states mock scans 36/36 and real scans 108/108.
- Split final manifest commit metadata into `experiment_result_commit`, `repository_cleanup_commit`, and `latest_verified_commit`. Because the final commit hash is not known before committing, `latest_verified_commit` is recorded as `working_tree_before_final_commit`.
- Added `configs/cooja_defense_dummy_logs.template.json` for portable Cooja log configuration through `COOJA_LOG_ROOT`.
- Added `outputs/reports/final_thesis/thesis_text_sync_suggestions.md`.
- Added `outputs/reports/final_thesis/git_history_size_note.md`.
- The current working tree is cleaned, but GitHub clone size may remain larger because old large files still exist in history.
- Further clone-size reduction would require a separate history rewrite with `git filter-repo` or BFG and a force push; this was not performed.
