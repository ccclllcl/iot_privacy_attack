# iot_privacy_attack

This project evaluates behavior-inference attacks on IoT time-series data and the effect of privacy defenses such as noise injection, LDP, and adaptive LDP. The repository is organized as a final thesis delivery project: experiment artifacts use canonical paths that expose `dataset / seed / model / method / mode`, and source code is separated by responsibility.

## Current Delivery Status

- mock main matrix: 36/36
- real main matrix: 108/108
- mock parameter scans: 36/36
- real parameter scans: 108/108
- adaptive LDP: 6 profiles for each dataset / seed / model / mode combination
- Cooja canonical results: 18/18
- `outputs/summaries/final_thesis/final_missing_outputs.json`: `[]`
- `outputs/summaries/final_thesis/parameter_scan_missing_outputs.json`: `[]`

Do not rerun the full matrices for normal review. Use the dashboard for browsing artifacts and for single-combination training or evaluation demos.

## Web Dashboard

Official entry point:

```bash
python -m streamlit run apps/dashboard.py
```

The dashboard includes:

- Overview: coverage, missing-output counts, summary tables, and Cooja limitations.
- Artifact Explorer: browse canonical artifacts by dataset, seed, model, method, and mode.
- Figures & Confusion Matrices: view final figures, draw confusion matrices from JSON, and plot parameter scans.
- Train / Evaluate Demo: run one selected training or evaluation job without importing data or running a full matrix.
- Run History: inspect dashboard-triggered jobs recorded in `outputs/ui/run_history.jsonl`.

The old command-style UI is retained only as `apps/legacy/ui_app.py`; it is not the recommended entry point.

## Canonical Artifact Layout

Formal artifact roots:

```text
outputs/
  experiments/
    mock/
    uci_har/
    kasteren/
    casas_hh101/
    cooja/
  summaries/
    final_thesis/
    layout/
  figures/
    summaries/
      final_thesis/
    experiments/
```

Normal defense experiment:

```text
outputs/experiments/{dataset}/seed_{seed}/{model}/{method}/{mode}/
```

Baseline:

```text
outputs/experiments/{dataset}/seed_{seed}/{model}/baseline/
```

Parameter scan:

```text
outputs/experiments/{dataset}/seed_{seed}/{model}/{method}/{mode}/parameter_scan/
```

Cooja:

```text
outputs/experiments/cooja/seed_{seed}/random_forest/{dummy_method}/{mode}/
```

Final summaries and figures:

```text
outputs/summaries/final_thesis/
outputs/figures/summaries/final_thesis/
outputs/figures/experiments/
```

Legacy batch paths were migrated. See `outputs/summaries/layout/migration_report.md` for the migration record.

## Code Layout

The main source package is organized by responsibility:

- `src/core/`: configuration, shared utilities, and plotting helpers.
- `src/data/`: preprocessing, feature extraction, and dataset wrappers.
- `src/models/`: LSTM and MLP model definitions.
- `src/training/`: training loops, early stopping, and checkpoint writing.
- `src/evaluation/`: baseline evaluation, defense evaluation, and parameter scans.
- `src/defenses/`: defense algorithms and the defense pipeline.
- `src/edge/`: adaptive LDP edge-budget allocation.
- `src/dashboard/`: dashboard paths, IO, plotting, subprocess runner, and history.
- `src/artifacts/`: canonical artifact paths and summary IO helpers.

Short compatibility wrappers remain at old `src/*.py` import paths so older scripts do not break immediately. New code should import from the structured packages above. See `docs/CODE_STRUCTURE.md` for details.

## Final Result Navigation

For thesis tables, figures, and delivery review, start with:

- `outputs/summaries/final_thesis/final_summary.csv`
- `outputs/summaries/final_thesis/final_summary.json`
- `outputs/summaries/final_thesis/final_symmetry_audit.json`
- `outputs/summaries/final_thesis/parameter_scan_coverage_audit.json`
- `outputs/summaries/final_thesis/artifact_index.md`
- `outputs/summaries/final_thesis/adaptive_ldp_ablation_overview.md`
- `outputs/summaries/final_thesis/cooja/cooja_limitations.md`
- `outputs/figures/summaries/final_thesis/`

Useful documentation:

- `docs/ARTIFACT_LAYOUT.md`
- `docs/CODE_STRUCTURE.md`
- `docs/PROJECT_STRUCTURE.md`
- `docs/REPOSITORY_DELIVERY_GUIDE.md`
- `docs/DASHBOARD_GUIDE.md`

## Single-Combination Demo Runs

The dashboard and demo runner use existing processed data only:

```text
data/processed/{dataset}/seed_{seed}/
data/defended/{dataset}/seed_{seed}/{method}/
```

They do not download, import, or generate datasets. If an input is missing, the run reports the missing path instead of filling it automatically.

Equivalent command-line examples:

```bash
python experiments/demo/run_dashboard_job.py --dataset mock --seed 42 --model lstm --job train_baseline --max-epochs 5 --device auto
python experiments/demo/run_dashboard_job.py --dataset mock --seed 42 --model lstm --job evaluate_baseline --overwrite
python experiments/demo/run_dashboard_job.py --dataset mock --seed 42 --model lstm --job defense_eval_fixed --method ldp --overwrite
```

Existing artifacts are protected by default. Use `--overwrite` only for an intentional single-combination demo overwrite.

## Core Maintenance Commands

These commands are lightweight and do not run experiment matrices:

```bash
python scripts/build_final_thesis_results.py
python scripts/audit_experiment_symmetry.py
python scripts/audit_repository_bloat.py
python scripts/audit_code_structure.py
```

The root-level scripts are compatibility wrappers. The organized implementations live in `scripts/final_thesis/` and `scripts/audit/`.

## Cooja Notes

Cooja results support fixed/retrain attack-accuracy reporting and functionality validation. The current delivery does not claim measured real energy consumption, measured real end-to-end delay, or distinguishable dummy/real packet ratios.

For reproducing Cooja log evaluation, copy:

```text
configs/cooja_defense_dummy_logs.template.json
```

and set `COOJA_LOG_ROOT` to the local Cooja log directory. Local WSL paths in completed result metadata only document the original evaluation source.

## Environment

```bash
pip install -r requirements.txt
```

Python 3.10+, pandas, numpy, scikit-learn, matplotlib, PyTorch, and Streamlit are the main dependencies. Install the matching CUDA-enabled PyTorch build only if GPU training is needed.
