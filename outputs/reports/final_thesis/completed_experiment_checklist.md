# 已完成实验计划任务清单

生成时间：2026-05-07

## 复核结论

本次已重新执行最终结果打包脚本，并复核 `outputs/reports/final_thesis/final_missing_outputs.json`。当前缺失项为 `0`。

核心审计文件：

- `outputs/reports/final_thesis/final_coverage_audit.json`
- `outputs/reports/final_thesis/final_missing_outputs.json`
- `outputs/reports/final_thesis/final_thesis_summary.md`
- `outputs/reports/final_thesis/figure_table_list.md`

## 已完成任务

| 序号 | 实验计划任务 | 完成情况 | 主要产物 |
|---:|---|---|---|
| 1 | mock 合成数据全矩阵实验 | 已完成 36/36 | `outputs/reports/final_thesis/mock/mock_summary.csv`、`outputs/defense/final_thesis/mock/` |
| 2 | mock 数据 LSTM/MLP 双模型评估 | 已完成 | `outputs/reports/final_thesis/mock/mock_summary.json` |
| 3 | mock 数据 adaptive_ldp / ldp / noise 三类防御评估 | 已完成 | `outputs/defense/final_thesis/mock/**/defense_report.json` |
| 4 | mock 数据 fixed_attacker / retrain_attacker 双攻击者口径评估 | 已完成 | `outputs/defense/final_thesis/mock/**/confusion.json` |
| 5 | mock 数据参数扫描汇总 | 已完成 | `outputs/reports/final_thesis/mock/mock_parameter_scan_ldp.csv`、`outputs/reports/final_thesis/mock/mock_parameter_scan_noise.csv` |
| 6 | UCI HAR 真实公开数据全矩阵实验 | 已完成 36/36 | `outputs/reports/final_thesis/real/real_summary.csv`、`outputs/defense/final_thesis/real/uci_har/` |
| 7 | Kasteren 真实公开数据全矩阵实验 | 已完成 36/36 | `outputs/defense/final_thesis/real/kasteren/` |
| 8 | CASAS HH101 真实公开数据全矩阵实验 | 已完成 36/36 | `outputs/defense/final_thesis/real/casas_hh101/` |
| 9 | UCI HAR 补充参数扫描，覆盖 LSTM/MLP 与 fixed/retrain | 已完成 | `outputs/reports/final_thesis/real/real_parameter_scan_ldp.csv`、`outputs/reports/final_thesis/real/real_parameter_scan_noise.csv` |
| 10 | Cooja 节点级日志防御评估 | 已完成 6 条汇总 | `outputs/reports/final_thesis/cooja/cooja_summary.csv` |
| 11 | Cooja 窗口数量代理开销汇总 | 已完成 | `outputs/reports/final_thesis/cooja/cooja_overhead_summary.csv` |
| 12 | 论文图表清单与最终图像 | 已完成 | `outputs/reports/final_thesis/figure_table_list.md`、`outputs/figures/final_thesis/` |
| 13 | 最终论文结果包汇总与缺口审计 | 已完成 | `outputs/reports/final_thesis/final_summary.csv`、`outputs/reports/final_thesis/final_summary.json` |

## 复核命令

本次实际复核使用：

```bash
py -3 experiments/real_public/run_uci_har_missing_parameter_scans.py --skip-existing
py -3 scripts/build_final_thesis_results.py
```

## 论文引用建议

论文正文优先引用以下结果包，避免使用可能被后续单次运行覆盖的旧路径：

- `outputs/reports/final_thesis/*.csv`
- `outputs/reports/final_thesis/*.json`
- `outputs/reports/final_thesis/final_thesis_summary.md`
- `outputs/defense/final_thesis/**`
- `outputs/figures/final_thesis/**`

不建议直接引用：

- `outputs/reports/**/metrics.json`
- `outputs/defense/**/defense_report.json`

上述旧路径可作为过程文件，但正文中的数值与图表应以 `final_thesis` 结果包为准。

## 追加完成项

- adaptive_ldp profile 级消融汇总：已完成，基于已有参数扫描整理，未重跑主矩阵。
