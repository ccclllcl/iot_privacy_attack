# 最终实验总结（可追溯）

## 1. 本次运行环境
- experiment_result_commit: `a43cb0e28822124fcaea8df2f40ff6119b571299`
- repository_cleanup_commit: `22580d40585af923aa1180215098db4403d47b79`
- latest_verified_commit: `working_tree_before_final_commit`
- 说明: 实验结果包生成 commit 与后续仓库清理 commit 可能不同；清理未重跑实验，只修正文档、路径和冗余产物。
- python version: `3.14.0 (tags/v3.14.0:ebf955d, Oct  7 2025, 10:15:03) [MSC v.1944 64 bit (AMD64)]`
- OS: `Windows-11-10.0.26200-SP0`
- start time / end time: `2026-05-09T17:01:20` / `2026-05-09T17:01:30`

## 2. mock 实验是否完整
- mock 主矩阵完整: `36` / `36`。
- mock 参数扫描完整: `36` / `36`；missing=`0`。
- LSTM 主要结果: baseline_acc 均值 `0.6402`，defended_acc 均值 `0.3764`。
- MLP 主要结果: baseline_acc 均值 `0.4735`，defended_acc 均值 `0.2568`。
- adaptive_ldp 已有 `6`-profile 级消融汇总。
- 可写入论文的结论: fixed_attacker 与 retrain_attacker 在 mock 数据上呈现可观差异，支持隐私-效用分析。
- 不建议写入论文的内容: 缺失组合（见 final_missing_outputs.json）对应的推断结论。

## 3. 真实数据集实验是否完整
- real 主矩阵完整: `108` / `108`。
- real 参数扫描完整: `108` / `108`；missing=`0`。
- 参数扫描覆盖: datasets=`uci_har,kasteren,casas_hh101`；methods=`adaptive_ldp,ldp,noise`；models=`lstm,mlp`；modes=`fixed_attacker,retrain_attacker`；seeds=`42,123,2026`。
- Kasteren 和 CASAS 参数扫描已经补齐，不再作为后续扩展建议。
- uci_har 完成情况: `36` / `36` 条。
  - 主要结果: baseline_acc 均值 `0.7522`，fixed/retrain defended_acc 均值 `0.4247`。
- kasteren 完成情况: `36` / `36` 条。
  - 主要结果: baseline_acc 均值 `0.2668`，fixed/retrain defended_acc 均值 `0.0422`。
- casas_hh101 完成情况: `36` / `36` 条。
  - 主要结果: baseline_acc 均值 `0.4601`，fixed/retrain defended_acc 均值 `0.2105`。
- 各数据集之间不能直接比较的原因: 类别空间、样本分布、传感器维度和标签定义不同。
- 可写入论文的结论: 在 UCI HAR、Kasteren 与 CASAS 上可稳定观测防御导致的准确率下降及部分重训恢复。
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
- 覆盖风险: 早期单次运行报告和 defense_report 文件可能被后续运行覆盖。
- 论文复核口径: `outputs/summaries/final_thesis/*.csv|*.json` 与 `outputs/experiments/**/source_manifest.json`。
- 历史路径说明: 早期未分 model/mode 的单文件报告只用于迁移追溯。

## 6. 下一步建议
- Cooja 真实能耗与真实端到端时延仍需真实部署补充。
- 更强攻击模型如 Transformer/TCN 可作为后续工作。
- 更细粒度真实部署消融仍可作为后续工作。

## Missing Count
- total missing entries: `0`
