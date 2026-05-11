# 产物结构说明

路径结构服务于实验复核，核心研究结论见 `README.md`、`outputs/summaries/final_thesis/final_summary.csv` 和 `outputs/summaries/final_thesis/final_summary.json`。本文件说明实验产物如何按 dataset、seed、model、method 和 mode 组织，便于从汇总结果回溯到单个实验组合。

## 1. 源实验产物

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

`source_manifest.json` 记录该组合的来源、角色和路径信息，用于连接最终汇总与单组合产物。

baseline 产物：

```text
outputs/experiments/{dataset}/seed_{seed}/{model}/baseline/
```

参数扫描：

```text
outputs/experiments/{dataset}/seed_{seed}/{model}/{method}/{mode}/parameter_scan/
```

`adaptive_ldp` profile 扫描除 `comparison_results.csv`、`scan_summary.json`、`scan_trace.json` 外，还包含 `profile_config.json`。

## 2. Dataset 和 Model

支持的 dataset slot：

- `mock`
- `uci_har`
- `kasteren`
- `casas_hh101`
- `cooja`

支持的 model slot：

- `lstm`
- `mlp`
- `random_forest`，用于 Cooja 场景。

## 3. Cooja

Cooja 使用：

```text
outputs/experiments/cooja/seed_{seed}/random_forest/{dummy_method}/{mode}/
```

Cooja 产物用于节点侧 dummy 流量功能性验证和 fixed/retrain 攻击准确率分析。当前结果同时提供 dummy/real 包比例、packet/byte overhead、Cooja 仿真时延和 Contiki-NG Energest 仿真能耗估计；这些能耗与时延不等同于真实硬件部署测量，相关说明见 `outputs/summaries/final_thesis/cooja/cooja_limitations.md`。

## 4. 汇总与图像

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

最终汇总面向论文和答辩复核，单组合产物面向结果追溯和 Dashboard 检索。

## 5. 历史迁移记录

早期批次产物已整理到统一结构中。迁移记录保留在：

- `outputs/summaries/layout/migration_map.csv`
- `outputs/summaries/layout/migration_report.md`

这些记录用于解释历史路径与标准路径之间的映射关系，最终结果引用以 `outputs/summaries/final_thesis/`、`outputs/figures/summaries/final_thesis/` 和 `outputs/experiments/` 为准。

## 6. Dashboard 与相关代码

Dashboard 使用本文描述的产物结构：

```bash
python -m streamlit run apps/dashboard.py
```

共享路径工具位于 `src/artifacts/canonical_paths.py`，Dashboard 选择和浏览工具位于 `src/dashboard/paths.py`。最终汇总和审计脚本基于标准产物路径读取结果，以保证 summary 与单组合 source artifacts 可相互追溯。
