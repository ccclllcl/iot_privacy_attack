# iot_privacy_attack

本项目用于评估智能家居/物联网时序数据上的行为推断攻击，以及噪声、LDP、adaptive LDP 等数据侧隐私防御对攻击准确率和数据失真的影响。当前仓库已经完成最终实验交付整理，正式产物路径均采用 dataset / seed / model / method / mode 结构表达实验选项。

## 当前交付状态

- mock 主矩阵：36/36
- real 主矩阵：108/108
- mock 参数扫描：36/36
- real 参数扫描：108/108
- adaptive_ldp：每个 dataset / seed / model / mode 组合 6 个 profile
- Cooja canonical 结果：18/18
- `outputs/summaries/final_thesis/final_missing_outputs.json` 为 `[]`
- `outputs/summaries/final_thesis/parameter_scan_missing_outputs.json` 为 `[]`

本仓库不建议重复全量重跑已完成实验。日常浏览、答辩展示和单组合训练/评估演示请优先使用 Dashboard。

## Web Dashboard

正式演示入口：

```bash
python -m streamlit run apps/dashboard.py
```

Dashboard 包含：

- Overview：查看主矩阵、参数扫描、Cooja 和缺失项状态；
- Artifact Explorer：按 dataset / seed / model / method / mode 检索 canonical artifact；
- Figures & Confusion Matrices：查看最终图、实时绘制混淆矩阵、查看参数扫描曲线；
- Train / Evaluate Demo：只对用户选择的单个组合训练或评估，不导入数据、不跑全量矩阵；
- Run History：查看 Dashboard 触发的单组合运行历史。

运行历史写入：

```text
outputs/ui/run_history.jsonl
```

旧的 `apps/ui_app.py` 仅作为 legacy 命令式 UI 保留，不再作为推荐入口；早期 simple UI 已移除。

## Canonical Artifact Layout

正式产物入口如下：

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

普通防御实验：

```text
outputs/experiments/{dataset}/seed_{seed}/{model}/{method}/{mode}/
```

baseline：

```text
outputs/experiments/{dataset}/seed_{seed}/{model}/baseline/
```

参数扫描：

```text
outputs/experiments/{dataset}/seed_{seed}/{model}/{method}/{mode}/parameter_scan/
```

Cooja：

```text
outputs/experiments/cooja/seed_{seed}/random_forest/{dummy_method}/{mode}/
```

最终汇总与图：

```text
outputs/summaries/final_thesis/
outputs/figures/summaries/final_thesis/
outputs/figures/experiments/
```

更详细的结构说明见：

```text
docs/ARTIFACT_LAYOUT.md
docs/PROJECT_STRUCTURE.md
docs/REPOSITORY_DELIVERY_GUIDE.md
docs/DASHBOARD_GUIDE.md
```

## 最终结果导航

论文和答辩优先查看：

- `outputs/summaries/final_thesis/final_summary.csv`
- `outputs/summaries/final_thesis/final_summary.json`
- `outputs/summaries/final_thesis/final_symmetry_audit.json`
- `outputs/summaries/final_thesis/parameter_scan_coverage_audit.json`
- `outputs/summaries/final_thesis/artifact_index.md`
- `outputs/summaries/final_thesis/adaptive_ldp_ablation_overview.md`
- `outputs/summaries/final_thesis/cooja/cooja_limitations.md`
- `outputs/figures/summaries/final_thesis/`

旧批次路径已迁移，迁移记录见：

```text
outputs/summaries/layout/migration_report.md
outputs/summaries/layout/migration_map.csv
```

## 单组合训练与评估

Dashboard 的 Train / Evaluate Demo 只使用已经存在的：

```text
data/processed/{dataset}/seed_{seed}/
data/defended/{dataset}/seed_{seed}/{method}/
```

它不会自动下载、导入或生成数据。缺少 processed 或 defended data 时会提示缺失。

命令行等价入口：

```bash
python experiments/demo/run_dashboard_job.py --dataset mock --seed 42 --model lstm --job train_baseline --max-epochs 5 --device auto
python experiments/demo/run_dashboard_job.py --dataset mock --seed 42 --model lstm --job evaluate_baseline --overwrite
python experiments/demo/run_dashboard_job.py --dataset mock --seed 42 --model lstm --job defense_eval_fixed --method ldp --overwrite
```

默认不会覆盖已有 canonical 产物。需要覆盖时显式传入 `--overwrite`，Dashboard 页面还会要求二次确认。

## Core Scripts

保留的正式维护脚本：

```bash
python scripts/build_final_thesis_results.py
python scripts/audit_experiment_symmetry.py
python scripts/audit_repository_bloat.py
```

`build_final_thesis_results.py` 从 `outputs/experiments/` 读取 canonical artifacts，并写入 `outputs/summaries/final_thesis/` 与 `outputs/figures/summaries/final_thesis/`。

迁移和补缺阶段脚本已经不再作为最终项目入口。当前参数扫描已完整，若确需演示单组合训练或评估，请使用 Dashboard 或 `experiments/demo/run_dashboard_job.py`。

## Cooja 说明

Cooja 结果用于展示 fixed/retrain 攻击准确率和功能性验证。当前结果不声称真实能耗、真实端到端时延或可区分 dummy/real 包比例已测量。

复现实验日志路径时，可复制：

```text
configs/cooja_defense_dummy_logs.template.json
```

并设置 `COOJA_LOG_ROOT`。已完成结果中的本地 WSL 日志路径仅用于记录当时的评估来源。

## 环境

```bash
pip install -r requirements.txt
```

Python 3.10+、pandas、numpy、scikit-learn、matplotlib、PyTorch 和 Streamlit 是主要依赖。GPU 训练可按 PyTorch 官方说明安装对应 CUDA 版本。
