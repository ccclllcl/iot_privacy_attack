# 产物结构说明

canonical artifact layout 的核心原则是：路径本身表达实验选项。

## 源实验产物

普通实验组合：

```text
outputs/experiments/{dataset}/seed_{seed}/{model}/{method}/{mode}/
```

每个组合通常包含：

- `metrics.json`
- `confusion.json`
- `classification_report.txt`
- `trace.json`
- `defense_report.json`
- `source_manifest.json`

baseline 产物：

```text
outputs/experiments/{dataset}/seed_{seed}/{model}/baseline/
```

参数扫描：

```text
outputs/experiments/{dataset}/seed_{seed}/{model}/{method}/{mode}/parameter_scan/
```

`adaptive_ldp` profile 扫描除 `comparison_results.csv`、`scan_summary.json`、`scan_trace.json` 外，还包含 `profile_config.json`。

## Dataset 和 Model

支持的 dataset slot：

- `mock`
- `uci_har`
- `kasteren`
- `casas_hh101`
- `cooja`

支持的 model slot：

- `lstm`
- `mlp`
- `random_forest`，仅用于 Cooja。

## Cooja

Cooja 使用：

```text
outputs/experiments/cooja/seed_{seed}/random_forest/{dummy_method}/{mode}/
```

该路径整理只规范已有结果，不伪造 packet counts、byte counts、IAT、真实能耗或真实端到端时延。

## 汇总与图像

最终汇总：

```text
outputs/summaries/final_thesis/
```

最终论文图像：

```text
outputs/figures/summaries/final_thesis/
```

单组合诊断图像：

```text
outputs/figures/experiments/{dataset}/seed_{seed}/{model}/{method}/{mode}/
```

## 迁移记录

旧批次路径迁移记录保留在：

- `outputs/summaries/layout/migration_map.csv`
- `outputs/summaries/layout/migration_report.md`

旧批次路径只作为历史说明，不应作为最终论文引用路径。

## Dashboard

正式浏览和演示入口：

```bash
python -m streamlit run apps/dashboard.py
```

Dashboard 使用本文件描述的 canonical paths。demo 运行只写入所选组合，并在 `outputs/ui/run_history.jsonl` 记录历史。

## 相关代码

共享路径工具位于 `src/artifacts/canonical_paths.py`。Dashboard 选择和浏览工具位于 `src/dashboard/paths.py`，并复用 canonical path 常量。最终汇总和审计脚本应读取 canonical artifacts，不应再把旧 batch-name 目录作为正式路径。
