# Dashboard Precheck Report

Generated: 2026-05-09

## Coverage Status

- mock main matrix: 36/36
- real main matrix: 108/108
- mock parameter scans: 36/36
- real parameter scans: 108/108
- adaptive_ldp profile count: 6 per dataset/seed/model/mode combination
- Cooja canonical outputs: 18/18
- `final_missing_outputs.json`: []
- `parameter_scan_missing_outputs.json`: []
- Experiment rerun needed: no

## Canonical Artifact Roots

- `outputs/experiments/`
- `outputs/summaries/final_thesis/`
- `outputs/figures/summaries/final_thesis/`
- `outputs/figures/experiments/`
- `docs/ARTIFACT_LAYOUT.md`

## README Issues Found

- The Web UI section still recommends `apps/ui_app.py` and `apps/ui_simple.py` instead of `apps/dashboard.py`.
- The old simple UI still depends on `web_assets/images/`, which is no longer the final image entry.
- Several sections still present old batch paths as active result locations, including `outputs/reports/real_public_benchmark/`, `outputs/reports/full_multiseed_summary.json`, and older report roots.
- The full rerun guidance still recommends batch/full-matrix commands as a normal workflow, which conflicts with the completed canonical result package.
- The final symmetry completion section still recommends `experiments/batches/run_missing_parameter_scans.py --skip-existing`; this is no longer a standard delivery command because scans are already complete.

## Script Disposition

- Delete `scripts/restructure_artifacts.py`: one-time migration helper; migration map and report are already preserved.
- Delete `scripts/audit_artifact_layout.py`: one-time layout audit helper; final layout is documented in `docs/ARTIFACT_LAYOUT.md`.
- Delete `experiments/batches/run_missing_parameter_scans.py`: old missing-scan helper still points at legacy paths and is not a final delivery entry.
- Delete `apps/ui_simple.py`: legacy UI tied to old web asset and rerun flow.
- Keep `apps/ui_app.py` as an optional legacy command UI only; it is not the recommended entry.
- Keep `scripts/build_final_thesis_results.py`, `scripts/audit_experiment_symmetry.py`, and `scripts/audit_repository_bloat.py`.

## Decision

No experiment import, Cooja metric completion, full matrix rerun, or parameter-scan rerun is needed. The next work should update README/docs, add the canonical dashboard, and remove migration-only or obsolete UI/script entry points.
