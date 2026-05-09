# 结果结构说明

本文件作为旧结果目录说明的兼容备注保留。

最终交付不再把 `full_multiseed`、`real_public_benchmark`、`dataset_matrix` 或旧 `final_thesis` 作为 source-artifact root。当前正式结构为：

- 源产物：`outputs/experiments/`
- 最终汇总：`outputs/summaries/final_thesis/`
- 最终图像：`outputs/figures/summaries/final_thesis/`
- 产物结构说明：`docs/ARTIFACT_LAYOUT.md`
- 产物索引：`outputs/summaries/final_thesis/artifact_index.md`

旧路径迁移关系：

- `full_multiseed` 已迁移为 `mock`。
- `real_public_benchmark/{dataset}` 已迁移为 `{dataset}`。
- `outputs/reports/final_thesis/` 已迁移为 `outputs/summaries/final_thesis/`。
- `outputs/figures/final_thesis/` 已迁移为 `outputs/figures/summaries/final_thesis/`。

Cooja 已移动到 `outputs/experiments/cooja/`，与 `mock`、`uci_har`、`kasteren`、`casas_hh101` 同级。该迁移只整理路径，不补真实能耗、真实端到端时延，也不伪造 packet/byte/IAT 指标。
