# 项目复核指南

本文说明评审、老师或查看者如何理解本项目的研究目标、核心结论、实验覆盖和结果追溯方式。项目的详细背景与结论见 `README.md`，标准产物结构见 `docs/ARTIFACT_LAYOUT.md`。

## 1. 研究目标

本项目研究 IoT 行为序列和流量统计特征中的隐私泄露风险。实验首先构建 LSTM 与 MLP 攻击模型识别用户行为，再比较 `noise`、`ldp`、`adaptive_ldp` 三类防御机制在不同威胁模型下对攻击准确率和数据失真的影响。

研究流程覆盖五类场景：

- `mock`：可控模拟场景，用于系统性验证攻击与防御流程。
- `uci_har`、`kasteren`、`casas_hh101`：真实公开数据集，用于验证防御效果是否能迁移到真实行为数据。
- `cooja`：节点侧 dummy 流量功能性验证，用于观察 Cooja 日志场景下 fixed/retrain 攻击准确率变化。

## 2. 核心结论复核

核心结论集中在：

- `outputs/summaries/final_thesis/final_summary.csv`
- `outputs/summaries/final_thesis/final_summary.json`
- `outputs/summaries/final_thesis/mock/`
- `outputs/summaries/final_thesis/real/`
- `outputs/summaries/final_thesis/cooja/`
- `outputs/figures/summaries/final_thesis/`

这些文件对应 README 中总结的主要发现：LSTM 对时序结构更敏感，`ldp` 抑制较强但失真更大，`noise` 保留相关性但剩余识别风险较高，`adaptive_ldp` 提供 profile 级隐私—可用性折中，真实数据集上也能观察到防御后准确率下降。

## 3. 实验覆盖复核

实验覆盖情况由以下审计文件给出：

- `outputs/summaries/final_thesis/final_symmetry_audit.json`
- `outputs/summaries/final_thesis/parameter_scan_coverage_audit.json`
- `outputs/summaries/final_thesis/final_missing_outputs.json`
- `outputs/summaries/final_thesis/parameter_scan_missing_outputs.json`

当前最终结果包覆盖：

- mock 主矩阵：36/36
- real 主矩阵：108/108
- mock 参数扫描：36/36
- real 参数扫描：108/108
- `adaptive_ldp`：每个 dataset / seed / model / mode 组合包含 6 个 profile
- Cooja canonical：18/18

`final_missing_outputs.json` 与 `parameter_scan_missing_outputs.json` 均为 `[]`，说明最终结果包没有已知缺失组合。

## 4. Dashboard 展示方式

Dashboard 是答辩演示和结果复核入口：

```bash
python -m streamlit run apps/dashboard.py
```

Dashboard 可展示：

- 汇总覆盖和最终结果表；
- 按 dataset / seed / model / method / mode 检索单组合产物；
- 最终图像、参数扫描曲线和 confusion matrix；
- 基于已有处理数据的单组合训练/评估演示；
- 单组合演示任务的运行历史。

Dashboard 的演示功能聚焦于已有处理数据上的单组合流程复现。完整矩阵结果和 Cooja 日志实验结果以最终结果包中的汇总为准。

## 5. 单个实验组合追溯

普通防御实验的源产物位于：

```text
outputs/experiments/{dataset}/seed_{seed}/{model}/{method}/{mode}/
```

baseline 产物位于：

```text
outputs/experiments/{dataset}/seed_{seed}/{model}/baseline/
```

参数扫描产物位于：

```text
outputs/experiments/{dataset}/seed_{seed}/{model}/{method}/{mode}/parameter_scan/
```

Cooja 产物位于：

```text
outputs/experiments/cooja/seed_{seed}/random_forest/{dummy_method}/{mode}/
```

每个普通实验组合目录中的 `source_manifest.json` 记录了该组合的来源、角色和路径映射，便于从 summary 回溯到具体实验产物。

## 6. 代码实现复核

源码按攻击—防御闭环组织：

- 数据处理：`src/data/`
- 模型定义：`src/models/`
- 训练流程：`src/training/`
- 评估与参数扫描：`src/evaluation/`
- 防御算法：`src/defenses/`
- Dashboard 与产物浏览：`src/dashboard/`
- 标准产物路径工具：`src/artifacts/`

完整说明见 `docs/CODE_STRUCTURE.md`。

## 7. Cooja 结果边界

Cooja 部分用于 fixed/retrain 攻击准确率展示和节点侧 dummy 流量功能性验证。当前结果不声称真实能耗已测量，不声称真实端到端时延已测量，也不伪造 dummy/real 包比例。相关限制集中说明在：

```text
outputs/summaries/final_thesis/cooja/cooja_limitations.md
```

复现 Cooja 日志评估时，可复制 `configs/cooja_defense_dummy_logs.template.json` 并设置 `COOJA_LOG_ROOT`。已完成结果中的本地 WSL 路径仅记录原实验来源。
