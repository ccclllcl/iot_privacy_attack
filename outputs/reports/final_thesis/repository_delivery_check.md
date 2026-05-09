# Repository Delivery Check

Generated at: 2026-05-09 14:08 +08:00

## Coverage Confirmation

| Item | Expected | Completed | Missing |
|---|---:|---:|---:|
| mock main matrix | 36 | 36 | 0 |
| real main matrix | 108 | 108 | 0 |
| mock parameter scans | 36 | 36 | 0 |
| real parameter scans | 108 | 108 | 0 |

`parameter_scan_missing_outputs.json` is `[]`.

## Adaptive LDP Profile Coverage

- Expected profile count per combination: 6.
- Profiles covered: `adaptive_default`, `adaptive_strong_privacy`, `adaptive_weak_privacy`, `adaptive_sensitivity_only`, `adaptive_traffic_only`, `adaptive_edge_cap_on`.
- Datasets covered: `mock`, `uci_har`, `kasteren`, `casas_hh101`.
- Seeds covered: 42, 123, 2026.
- Models covered: `lstm`, `mlp`.
- Modes covered: `fixed_attacker`, `retrain_attacker`.
- Profile-count audit: every adaptive_ldp dataset/seed/model/mode group has exactly 6 rows.

## Delivery Documents

- `docs/REPOSITORY_DELIVERY_GUIDE.md`: present.
- `outputs/reports/final_thesis/artifact_index.md`: present.
- `outputs/reports/final_thesis/legacy_outputs_note.md`: present.
- `outputs/reports/final_thesis/adaptive_ldp_ablation_overview.md`: present.
- `outputs/reports/final_thesis/mock/mock_adaptive_ldp_ablation_summary.csv`: present.
- `outputs/reports/final_thesis/real/real_adaptive_ldp_ablation_summary.csv`: present.

## Scope

No main matrix or completed parameter scan was rerun for this delivery check. The adaptive_ldp ablation files are summaries derived from existing final_thesis parameter-scan CSVs.
