# Real adaptive_ldp profile ablation summary

This file summarizes existing adaptive_ldp profile scans. No experiment was rerun for this summary.

- Source: `outputs/summaries/final_thesis/real/real_parameter_scan_adaptive_ldp.csv`
- Output rows: `72`
- Each profile is aggregated by dataset, model type, and attacker mode across available seeds.

| dataset | profile_name | mean_defended_acc | mean_accuracy_drop | mean_mse |
|---|---:|---:|---:|---:|
| casas_hh101 | adaptive_default | 0.245183 | 0.187742 | 0.206018 |
| casas_hh101 | adaptive_strong_privacy | 0.164006 | 0.218768 | 2.421450 |
| casas_hh101 | adaptive_weak_privacy | 0.306584 | 0.135435 | 0.080229 |
| casas_hh101 | adaptive_sensitivity_only | 0.254378 | 0.166587 | 1.145251 |
| casas_hh101 | adaptive_traffic_only | 0.229331 | 0.184322 | 0.228886 |
| casas_hh101 | adaptive_edge_cap_on | 0.255112 | 0.174025 | 0.206018 |
| kasteren | adaptive_default | 0.056563 | 0.117444 | 0.215381 |
| kasteren | adaptive_strong_privacy | 0.016408 | 0.124784 | 2.792281 |
| kasteren | adaptive_weak_privacy | 0.088515 | 0.090242 | 0.086355 |
| kasteren | adaptive_sensitivity_only | 0.056563 | 0.117444 | 0.215381 |
| kasteren | adaptive_traffic_only | 0.056563 | 0.117444 | 0.215381 |
| kasteren | adaptive_edge_cap_on | 0.056563 | 0.117444 | 0.215381 |
| uci_har | adaptive_default | 0.498360 | 0.039277 | 0.714355 |
| uci_har | adaptive_strong_privacy | 0.391754 | 0.140567 | 4.732988 |
| uci_har | adaptive_weak_privacy | 0.533000 | 0.027627 | 0.199609 |
| uci_har | adaptive_sensitivity_only | 0.505882 | 0.030087 | 0.688785 |
| uci_har | adaptive_traffic_only | 0.495843 | 0.047506 | 0.806484 |
| uci_har | adaptive_edge_cap_on | 0.502799 | 0.038429 | 0.714355 |

Interpretation should stay cautious: this is a profile-level empirical ablation summary, not a formal theoretical proof.
