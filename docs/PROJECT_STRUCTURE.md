# Project Structure

This repository keeps the implementation, experiment entry points, configuration, and final thesis artifacts in separate areas.

## Top-Level Directories

- `src/`: core preprocessing, feature extraction, model, defense, evaluation, and comparison logic.
- `experiments/`: command-line experiment entry points. Run them from the repository root.
- `configs/`: default and generated experiment configuration files.
- `scripts/`: final thesis aggregation, audit, and delivery helper scripts.
- `docs/`: repository structure and delivery notes.
- `data/`: raw, processed, and defended datasets. These are generally not newly committed except for already tracked reproducibility artifacts.
- `outputs/`: models, reports, figures, and defense outputs. Most runtime outputs are ignored, but `outputs/reports/final_thesis/` and `outputs/figures/final_thesis/` are the final delivery exceptions.
- `apps/`, `tools/`, `web_assets/`: UI and maintenance utilities.

## Experiment Entrypoints

- `experiments/core/run_train.py`: train LSTM or MLP attackers.
- `experiments/core/run_evaluate.py`: evaluate attack baselines.
- `experiments/core/run_defense.py`: generate defended data.
- `experiments/core/run_defense_eval.py`: evaluate fixed and retrained attackers.
- `experiments/core/run_compare.py`: parameter scans for `ldp`, `noise`, and `adaptive_ldp`.
- `experiments/batches/`: mock multi-seed/multi-model/multi-method workflows.
- `experiments/real_public/`: UCI HAR, van Kasteren, and CASAS real-data workflows.
- `experiments/cooja/`: Cooja log evaluation and defense summaries.

## Final Delivery Paths

For thesis submission, prioritize:

- `outputs/reports/final_thesis/`
- `outputs/figures/final_thesis/`
- `docs/REPOSITORY_DELIVERY_GUIDE.md`

Process artifacts under `data/` and `outputs/` are retained only where they support reproduction or `source_file` traceability. New generated files should generally stay ignored unless they belong to the `final_thesis` delivery package.
