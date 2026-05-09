# Dashboard Guide

Start the dashboard from the repository root:

```bash
python -m streamlit run apps/dashboard.py
```

## Pages

- Overview: shows final coverage, missing-output counts, summary tables, and Cooja limitations.
- Artifact Explorer: browses canonical experiment folders by dataset, seed, model, method, and mode.
- Figures & Confusion Matrices: displays final figures, draws confusion matrices from JSON, and plots parameter scans.
- Train / Evaluate Demo: runs one selected training or evaluation job and writes to the corresponding canonical path.
- Run History: reads `outputs/ui/run_history.jsonl` and shows previous dashboard-triggered jobs.

## Implementation Files

- `apps/dashboard.py`: official Streamlit entry point.
- `src/dashboard/paths.py`: dashboard selection and canonical path helpers.
- `src/dashboard/io.py`: artifact loading and plotting helpers.
- `src/dashboard/runner.py`: subprocess runner, demo config builder, and history writer.
- `experiments/demo/run_dashboard_job.py`: command-line wrapper used by the dashboard for one selected job.

Legacy UI code is kept under `apps/legacy/` only for reference and is not the recommended entry point.

## Data Policy

The dashboard does not import or download data. It only uses existing processed data:

```text
data/processed/{dataset}/seed_{seed}/
```

Defense evaluation uses existing defended data:

```text
data/defended/{dataset}/seed_{seed}/{method}/
```

If these inputs are missing, the dashboard reports the missing path and does not auto-generate or import anything.

## Overwrite Policy

- `overwrite=false` is the default and protects existing canonical artifacts.
- `overwrite=true` is required to replace the selected output path.
- The Streamlit page requires a second confirmation before overwriting.
- Every run appends a record to `outputs/ui/run_history.jsonl`.

## Cooja

The dashboard only displays existing Cooja results. It does not run Cooja simulations, add real energy measurements, add real end-to-end delay, or fabricate packet/byte/IAT fields.

## Canonical Outputs

Baseline jobs write to:

```text
outputs/experiments/{dataset}/seed_{seed}/{model}/baseline/
```

Defense jobs write to:

```text
outputs/experiments/{dataset}/seed_{seed}/{model}/{method}/{mode}/
```

Models are saved under:

```text
outputs/models/{dataset}/seed_{seed}/{model}/
```
