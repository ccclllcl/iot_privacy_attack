# Mock adaptive_ldp profile ablation summary

This file summarizes existing adaptive_ldp profile scans. No experiment was rerun for this summary.

- Source: `outputs/reports/final_thesis/mock/mock_parameter_scan_adaptive_ldp.csv`
- Output rows: `24`
- Each profile is aggregated by dataset, model type, and attacker mode across available seeds.

| dataset | profile_name | mean_defended_acc | mean_accuracy_drop | mean_mse |
|---|---:|---:|---:|---:|
| mock | adaptive_default | 0.378311 | 0.077401 | 0.885304 |
| mock | adaptive_strong_privacy | 0.323124 | 0.131347 | 6.091179 |
| mock | adaptive_weak_privacy | 0.406181 | 0.068295 | 0.214581 |
| mock | adaptive_sensitivity_only | 0.377759 | 0.077263 | 1.002515 |
| mock | adaptive_traffic_only | 0.381209 | 0.078918 | 0.850746 |
| mock | adaptive_edge_cap_on | 0.378311 | 0.077401 | 0.885304 |

Interpretation should stay cautious: this is a profile-level empirical ablation summary, not a formal theoretical proof.
