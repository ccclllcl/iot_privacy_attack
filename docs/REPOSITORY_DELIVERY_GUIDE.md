# 仓库交付指南

本仓库已经整理为最终交付结构，实验选项直接体现在产物路径中。

## 优先查看位置

评审项目时建议优先查看：

- `README.md`
- `configs/default.yaml`
- `experiments/`
- `src/`
- `scripts/`
- `apps/dashboard.py`
- `docs/CODE_STRUCTURE.md`
- `docs/ARTIFACT_LAYOUT.md`
- `docs/DASHBOARD_GUIDE.md`
- `outputs/experiments/`
- `outputs/summaries/final_thesis/`
- `outputs/figures/summaries/final_thesis/`

## 论文引用优先路径

论文表格、图像和最终数值应优先引用：

- `outputs/summaries/final_thesis/final_summary.csv`
- `outputs/summaries/final_thesis/final_summary.json`
- `outputs/summaries/final_thesis/final_coverage_audit.json`
- `outputs/summaries/final_thesis/final_symmetry_audit.json`
- `outputs/summaries/final_thesis/parameter_scan_coverage_audit.json`
- `outputs/summaries/final_thesis/mock/`
- `outputs/summaries/final_thesis/real/`
- `outputs/summaries/final_thesis/cooja/`
- `outputs/figures/summaries/final_thesis/`

## 标准源产物

核心源产物位于：

- `outputs/experiments/mock/`
- `outputs/experiments/uci_har/`
- `outputs/experiments/kasteren/`
- `outputs/experiments/casas_hh101/`
- `outputs/experiments/cooja/`

普通实验路径：

```text
outputs/experiments/{dataset}/seed_{seed}/{model}/{method}/{mode}/
```

baseline 路径：

```text
outputs/experiments/{dataset}/seed_{seed}/{model}/baseline/
```

Cooja 使用 `random_forest` 作为 model slot：

```text
outputs/experiments/cooja/seed_{seed}/random_forest/{dummy_method}/{mode}/
```

## 旧路径说明

旧批次路径已经迁移，不应作为最终引用路径。迁移映射见 `outputs/summaries/layout/migration_map.csv`，文字说明见 `outputs/summaries/layout/migration_report.md`。

## Dashboard 入口

用于快速检查和单组合演示：

```bash
python -m streamlit run apps/dashboard.py
```

Dashboard 读取 `outputs/experiments/`、`outputs/summaries/final_thesis/` 和 `outputs/figures/summaries/final_thesis/`。它不导入数据、不运行完整矩阵、不运行 Cooja 仿真。

## 代码组织

- 配置核心（`src/core`）：配置、工具、绘图。
- 数据处理（`src/data`）：预处理、特征、Dataset。
- 模型定义（`src/models`）：LSTM、MLP。
- 训练逻辑（`src/training`）：训练和 checkpoint。
- 评估逻辑（`src/evaluation`）：评估、防御评估、参数扫描。
- 防御算法（`src/defenses`）：防御方法和防御流水线。
- Dashboard 工具（`src/dashboard`）：路径、IO、运行器、历史记录。
- 产物路径工具（`src/artifacts`）：canonical artifact 路径。

根目录 `src/*.py` 模块仅是兼容 wrapper。详细说明见 `docs/CODE_STRUCTURE.md`。

## Cooja 限制

- `cooja_summary.csv` 和 `cooja_per_seed.csv` 可用于攻击准确率报告。
- `cooja_traffic_metrics.csv` 中的 packet/byte/IAT 字段若为 null 或 NaN，应结合 `outputs/summaries/final_thesis/cooja/cooja_limitations.md` 理解。
- 当前结果不声称真实能耗已测量。
- 当前结果不声称真实端到端时延已测量。
- 当前日志无法区分 dummy 包和 real 包时，不应伪造 dummy ratio。
- 已完成结果中的本地 WSL 日志路径仅用于记录原实验来源。
- 复现时请复制 `configs/cooja_defense_dummy_logs.template.json`，设置 `COOJA_LOG_ROOT`，并将输出放入 canonical Cooja 路径。
