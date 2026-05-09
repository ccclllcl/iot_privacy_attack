# Results Structure Note

This file is kept as a compatibility note for older result-directory documentation.

The final thesis delivery no longer uses batch names such as `full_multiseed`, `real_public_benchmark`, `dataset_matrix`, or `final_thesis` as source-artifact roots. The current structure is:

- Source artifacts: `outputs/experiments/`
- Final summaries: `outputs/summaries/final_thesis/`
- Final figures: `outputs/figures/summaries/final_thesis/`
- Layout guide: `docs/ARTIFACT_LAYOUT.md`
- Artifact index: `outputs/summaries/final_thesis/artifact_index.md`

Old paths were migrated as follows:

- `full_multiseed` became `mock`.
- `real_public_benchmark/{dataset}` became `{dataset}`.
- `outputs/reports/final_thesis/` became `outputs/summaries/final_thesis/`.
- `outputs/figures/final_thesis/` became `outputs/figures/summaries/final_thesis/`.

Cooja was moved under `outputs/experiments/cooja/` at the same level as `mock`, `uci_har`, `kasteren`, and `casas_hh101`. The migration does not add real energy, real end-to-end delay, or fabricated packet/byte/IAT metrics.
