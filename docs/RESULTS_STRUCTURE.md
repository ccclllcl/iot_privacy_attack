# 实验结果目录梳理（2026-04）

本文件用于快速定位“已跑完的全量实验”与其产出路径，便于论文写作和 GPT 批量读取。

## 0) 真实公开数据优先（real_public_benchmark，当前主线）

- 数据集（默认）：`uci_har`、`kasteren`（可扩展 `casas_hh101`）
- 随机种子：`42, 123, 2026`（可按运行脚本参数调整）
- 模型：`lstm, mlp`
- 防御：`adaptive_ldp, ldp, noise`
- 防御评估：`fixed_attacker, retrain_attacker`
- 扫描：`ldp epsilon`、`noise scale`

关键索引文件：

- `outputs/reports/real_public_benchmark/real_public_benchmark_manifest.json`
- `outputs/reports/real_public_benchmark/real_public_benchmark_summary.json`
- `outputs/reports/real_public_benchmark/real_public_benchmark_runs.csv`

主要结果目录：

- `outputs/defense/real_public_benchmark/{dataset}/seed_{seed}/{method}/`
  - `defense_report.json` / `defense_report.txt`
  - `json_reports/*.json`（baseline / fixed / retrain 混淆矩阵与指标）
  - `comparisons/comparison_results.csv`（ldp/noise 扫描）

配套数据与模型目录：

- `data/processed/real_public_benchmark/`
- `data/defended/real_public_benchmark/`
- `outputs/models/real_public_benchmark/`

## 1) 合成数据全矩阵（full_multiseed）

- 随机种子：`42, 123, 2026`
- 模型：`lstm, mlp`
- 防御：`adaptive_ldp, ldp, noise`
- 防御评估：`fixed_attacker, retrain_attacker`
- 扫描：`ldp epsilon`、`noise scale`

关键索引文件：

- `outputs/reports/full_methods_multiseed_manifest.json`
- `outputs/reports/full_multiseed_summary.json`

主要结果目录：

- `outputs/defense/full_multiseed/seed_{seed}/{method}/`
  - `defense_report.json` / `defense_report.txt`
  - `json_reports/*.json`（baseline / fixed / retrain 混淆矩阵与指标）
  - `comparisons/comparison_results.csv`（ldp/noise 扫描）

## 2) 真实数据集全矩阵（dataset_matrix）

- 数据集：`uci_har, kasteren`
- 随机种子：`42, 123`
- 模型：`lstm, mlp`
- 防御：`adaptive_ldp, ldp, noise`
- 防御评估：`fixed_attacker, retrain_attacker`
- 扫描：`ldp epsilon`、`noise scale`

关键索引文件：

- `outputs/reports/dataset_matrix_manifest.json`

主要结果目录：

- `outputs/defense/dataset_matrix/{dataset}/seed_{seed}/{method}/`
  - `defense_report.json` / `defense_report.txt`
  - `json_reports/*.json`（baseline / fixed / retrain 混淆矩阵与指标）
  - `comparisons/comparison_results.csv`（ldp/noise 扫描）

## 3) 训练与中间数据目录

- 预处理结果：`data/processed/full_multiseed/`、`data/processed/dataset_matrix/`
- 防御后数据：`data/defended/full_multiseed/`、`data/defended/dataset_matrix/`
- 模型权重：`outputs/models/full_multiseed/`、`outputs/models/dataset_matrix/`
- 图与报告：`outputs/figures/...`、`outputs/reports/...`

## 4) 复现实验脚本

- 合成数据全矩阵：`run_all_methods_multiseed.py`
- 真实数据集全矩阵：`run_full_matrix_real_datasets.py`
