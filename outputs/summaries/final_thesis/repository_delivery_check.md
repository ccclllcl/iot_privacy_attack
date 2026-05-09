# Repository Delivery Check

- Mock main matrix: expected=36, completed=36, missing=0.
- Real main matrix: expected=108, completed=108, missing=0.
- Mock parameter scans: expected=36, completed=36, missing=[].
- Real parameter scans: expected=108, completed=108, missing=[].
- adaptive_ldp profile count: expected=6 per combination across mock, uci_har, kasteren, casas_hh101; seeds 42/123/2026; models lstm/mlp; modes fixed_attacker/retrain_attacker.
- `parameter_scan_missing_outputs.json`: `[]`.

## Delivery Files

- Artifact index: `outputs/summaries/final_thesis/artifact_index.md`.
- Layout guide: `docs/ARTIFACT_LAYOUT.md`.
- Repository delivery guide: `docs/REPOSITORY_DELIVERY_GUIDE.md`.
- Adaptive ablation overview: `outputs/summaries/final_thesis/adaptive_ldp_ablation_overview.md`.
- Mock adaptive ablation summary: `outputs/summaries/final_thesis/mock/mock_adaptive_ldp_ablation_summary.csv`.
- Real adaptive ablation summary: `outputs/summaries/final_thesis/real/real_adaptive_ldp_ablation_summary.csv`.

No experiments were rerun for this layout normalization.
