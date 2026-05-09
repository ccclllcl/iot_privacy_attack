# Codex Fix Report

## Artifact layout normalization

- No full experiment rerun was performed.
- Missing combinations rerun: none.
- Mock main matrix remains complete: 36/36.
- Real main matrix remains complete: 108/108.
- Mock parameter scans remain complete: 36/36.
- Real parameter scans remain complete: 108/108.
- adaptive_ldp profile coverage remains 6 profiles per dataset/seed/model/mode combination.
- `final_missing_outputs.json` is `[]`.
- `parameter_scan_missing_outputs.json` is `[]`.

## Path Changes

- Old formal source roots were migrated away from batch names.
- New source root: `outputs/experiments/`.
- New final summary root: `outputs/summaries/final_thesis/`.
- New final figure root: `outputs/figures/summaries/final_thesis/`.
- Normal experiment path: `outputs/experiments/{dataset}/seed_{seed}/{model}/{method}/{mode}/`.
- Baseline path: `outputs/experiments/{dataset}/seed_{seed}/{model}/baseline/`.
- Cooja path: `outputs/experiments/cooja/seed_{seed}/random_forest/{dummy_method}/{mode}/`.

## Moved and Regenerated Files

- Moved or normalized: main-matrix JSON/TXT artifacts, parameter-scan CSVs, adaptive profile configs, final summaries, final figures, generated configs, processed/defended data, and local model artifacts.
- Regenerated from canonical paths: final summaries, coverage audits, symmetry audits, repository bloat audit, layout audits, migration map, migration report, artifact index, and summary figures.
- Deleted old batch roots and empty optional Cooja diagnostic placeholders after canonical copies and `source_manifest.json` files were written.

## Cooja

- Cooja was only moved into the canonical structure.
- No real energy, real end-to-end delay, packet/byte/IAT, or dummy ratio values were fabricated.
- Cooja traffic limitations remain documented in `outputs/summaries/final_thesis/cooja/cooja_limitations.md`.

## Delivery Status

The normalized repository structure is suitable for undergraduate thesis delivery. Final citation paths are centralized under `outputs/summaries/final_thesis/` and `outputs/figures/summaries/final_thesis/`, with source artifacts traceable through `outputs/experiments/**/source_manifest.json`.

## Canonical dashboard delivery

- No full experiment rerun was performed.
- Added the formal dashboard entry `apps/dashboard.py`.
- Added shared dashboard helpers in `src/dashboard_paths.py`, `src/dashboard_io.py`, and `src/dashboard_runner.py`.
- Added the single-combination runner `experiments/demo/run_dashboard_job.py`.
- Updated README and delivery docs so `outputs/experiments/`, `outputs/summaries/final_thesis/`, and `outputs/figures/summaries/final_thesis/` are the formal paths.
- Removed migration-only scripts and the old simple UI entry.
- Verified one mock `seed_42` `lstm` baseline train/evaluate demo with `max_epochs=1`; overwrite protection refused an existing artifact when overwrite was false.
- Cooja remains display-only in the dashboard; no packet/byte/IAT, real energy, or real delay metrics were fabricated.
