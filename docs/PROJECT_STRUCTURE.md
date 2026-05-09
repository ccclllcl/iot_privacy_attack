# Project Structure

This repository separates implementation code, reproducible experiment entry points, canonical source artifacts, and final thesis summaries.

## Top-Level Directories

- `src/`: core preprocessing, feature extraction, models, defenses, evaluation, and comparison logic.
- `experiments/`: command-line experiment entry points, including the single-combination dashboard runner.
- `configs/`: default, Cooja, and generated experiment configuration files.
- `scripts/`: final aggregation and audit scripts.
- `docs/`: repository structure and delivery notes.
- `data/`: processed and defended data organized by dataset and seed.
- `outputs/experiments/`: canonical source artifacts organized by dataset, seed, model, method, and mode.
- `outputs/summaries/final_thesis/`: final thesis CSV/JSON/Markdown summaries and audits.
- `outputs/figures/summaries/final_thesis/`: final thesis figures.
- `outputs/figures/experiments/`: per-combination diagnostic figures when available.

## Experiment Entrypoints

- `experiments/core/run_train.py`: train LSTM or MLP attackers.
- `experiments/core/run_evaluate.py`: evaluate attack baselines.
- `experiments/core/run_defense.py`: generate defended data.
- `experiments/core/run_defense_eval.py`: evaluate fixed and retrained attackers.
- `experiments/core/run_compare.py`: parameter scans for `ldp`, `noise`, and `adaptive_ldp`.
- `experiments/demo/run_dashboard_job.py`: dashboard-safe single-combination train/evaluate runner.
- `experiments/real_public/`: UCI HAR, van Kasteren, and CASAS real-data workflows.
- `experiments/cooja/`: Cooja log evaluation and defense summaries.

## Dashboard

The recommended demonstration entry is:

`python -m streamlit run apps/dashboard.py`

The dashboard reads canonical artifacts, displays summary figures and confusion matrices, and can run one selected train/evaluate job at a time. It does not import data, run Cooja simulations, or launch a full experiment matrix.

## Canonical Artifact Paths

Normal defense experiment:

`outputs/experiments/{dataset}/seed_{seed}/{model}/{method}/{mode}/`

Baseline:

`outputs/experiments/{dataset}/seed_{seed}/{model}/baseline/`

Parameter scan:

`outputs/experiments/{dataset}/seed_{seed}/{model}/{method}/{mode}/parameter_scan/`

Cooja:

`outputs/experiments/cooja/seed_{seed}/random_forest/{dummy_method}/{mode}/`

## Final Delivery Paths

For thesis submission, prioritize:

- `outputs/summaries/final_thesis/`
- `outputs/figures/summaries/final_thesis/`
- `docs/REPOSITORY_DELIVERY_GUIDE.md`
- `docs/ARTIFACT_LAYOUT.md`

Legacy batch-name roots were migrated to reduce ambiguity. The migration record is kept in `outputs/summaries/layout/migration_report.md`.
