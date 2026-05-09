# Adaptive LDP Ablation Overview

This is not a newly rerun experiment. It organizes the existing adaptive_ldp profile parameter scans into a formal ablation summary.

## Ablation Dimensions

- `epsilon_min` / `epsilon_max`: adaptive privacy budget range.
- `weight_sensitivity`: weight for the window-variation proxy.
- `weight_traffic`: weight for the traffic-intensity proxy.
- `use_edge_budget_cap`: whether the edge budget clipping interface is enabled.

## Profiles

- `adaptive_default`: balanced sensitivity and traffic weighting.
- `adaptive_strong_privacy`: stronger perturbation through a smaller epsilon range.
- `adaptive_weak_privacy`: weaker perturbation through a larger epsilon range.
- `adaptive_sensitivity_only`: uses only the window-variation proxy.
- `adaptive_traffic_only`: uses only the traffic-intensity proxy.
- `adaptive_edge_cap_on`: enables the edge budget clipping interface.

## Scope and Caution

The summary covers mock, uci_har, kasteren, and casas_hh101; seeds 42, 123, and 2026; LSTM and MLP; fixed_attacker and retrain_attacker.
The results are empirical profile scans and should not be overstated as a formal theoretical proof.

If thesis Section 5.2 still says that ablation experiments can be done later, revise it to: "Current results already include profile-level ablation summaries, while finer-grained real deployment ablations can remain future work."

## Generated Files

- `outputs/summaries/final_thesis/mock/mock_adaptive_ldp_ablation_summary.csv`
- `outputs/summaries/final_thesis/mock/mock_adaptive_ldp_ablation_summary.md`
- `outputs/summaries/final_thesis/real/real_adaptive_ldp_ablation_summary.csv`
- `outputs/summaries/final_thesis/real/real_adaptive_ldp_ablation_summary.md`
