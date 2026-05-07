# 最终实验总结（可追溯）

## 1. 本次运行环境
- git commit: `b516557152acf3268c3142f437183d7faf33d04a`
- python version: `3.14.0 (tags/v3.14.0:ebf955d, Oct  7 2025, 10:15:03) [MSC v.1944 64 bit (AMD64)]`
- OS: `Windows-11-10.0.26200-SP0`
- start time / end time: `2026-05-07T13:36:50` / `2026-05-07T13:37:13`

## 2. mock 实验是否完整
- 完成情况: 已收集 `36` / 期望 `36` 条（dataset=mock）。
- LSTM 主要结果: baseline_acc 均值 `0.7759`，defended_acc 均值 `0.3764`。
- MLP 主要结果: baseline_acc 均值 `0.4735`，defended_acc 均值 `0.2568`。
- 参数扫描结果: mock 保留原有 ldp/noise 扫描 CSV；UCI HAR 已补齐 LSTM/MLP 与 fixed/retrain 扫描口径。
- 可写入论文的结论: fixed_attacker 与 retrain_attacker 在 mock 数据上呈现可观差异，支持隐私-效用分析。
- 不建议写入论文的内容: 缺失组合（见 final_missing_outputs.json）对应的推断结论。

## 3. 真实数据集实验是否完整
- uci_har 完成情况: `36` / `36` 条。
  - 主要结果: baseline_acc 均值 `0.7522`，fixed/retrain defended_acc 均值 `0.4247`。
- kasteren 完成情况: `36` / `36` 条。
  - 主要结果: baseline_acc 均值 `0.2668`，fixed/retrain defended_acc 均值 `0.0422`。
- casas_hh101 完成情况: `36` / `36` 条。
  - 主要结果: baseline_acc 均值 `0.4601`，fixed/retrain defended_acc 均值 `0.2105`。
- 各数据集之间不能直接比较的原因: 类别空间、样本分布、传感器维度和标签定义不同。
- 可写入论文的结论: 在 UCI HAR 与 Kasteren 上可稳定观测防御导致的准确率下降及部分重训恢复。
- 不建议写入论文的内容: 不同数据集之间的绝对准确率直接排序。

## 4. Cooja 节点级实验是否完整
- 日志是否存在: 可用。
- dummy 流量是否跑通: 已运行。
- fixed/retrain 是否跑通: 已运行。
- 流量混淆度是否可计算: 部分可计算。
- 节点开销是否可计算: 能耗/时延真实量化不足，使用代理指标。
- 可写入论文的结论: 见 cooja_summary.csv。
- 不建议写入论文的内容: 未有真实量测支持的能耗结论。

## 5. 文件口径风险
- 覆盖风险: 原始 `outputs/reports/**/metrics.json`、`outputs/defense/**/defense_report.json` 可能被后续运行覆盖。
- 推荐论文引用: `outputs/reports/final_thesis/*.csv|*.json` 与 `outputs/defense/final_thesis/**`。
- 不建议直接引用: 旧路径中未分 model/mode 的单文件报告。

## 6. 下一步建议
- 若需进一步扩展，可把 UCI HAR 的全口径参数扫描推广到 Kasteren 与 CASAS。
- 若需系统开销完整性，可补充真实能耗、时延与带宽测量。
- 论文图表建议优先使用 `outputs/figures/final_thesis/`。

## Missing Count
- total missing entries: `0`