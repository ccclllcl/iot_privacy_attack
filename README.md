# iot_privacy_attack

## 项目概述

`iot_privacy_attack` 是一个围绕物联网行为推断隐私风险的本科毕业设计项目。项目以设备状态序列、传感器活动序列和流量统计特征为输入，构建攻击模型识别用户行为，并评估 `noise`、`ldp`、`adaptive_ldp` 三类数据侧防御机制对攻击准确率和数据失真的影响。

项目的实验组织围绕 `dataset / seed / model / method / mode` 展开。最终结果既包含 `mock` 场景，也包含 `uci_har`、`kasteren`、`casas_hh101` 三个真实数据集，以及基于 Cooja 日志的节点侧 dummy 流量功能性验证。

## 研究问题

本项目主要回答以下问题：

- 设备状态序列和流量统计特征是否会泄露用户行为。
- 数据侧扰动能否降低行为推断攻击的准确率。
- `fixed_attacker` 与 `retrain_attacker` 两种威胁模型下，防御效果是否存在差异。
- 隐私抑制与数据失真之间如何权衡。
- 节点侧 dummy 流量是否能改变窃听者可见的流量模式。

## 主要实现

项目实现了从数据处理、攻击建模、防御生成、攻击评估到结果汇总的完整实验闭环：

- 攻击模型：使用 LSTM 与 MLP 对用户行为进行分类识别。
- 防御方法：实现 `noise`、`ldp`、`adaptive_ldp` 三类数据扰动机制。
- 威胁模型：比较 `fixed_attacker` 与 `retrain_attacker` 下的防御效果。
- 验证场景：覆盖 `mock`、`uci_har`、`kasteren`、`casas_hh101` 和 `cooja`。
- 演示入口：提供 Streamlit Dashboard，用于浏览最终结果、展示图表、查看混淆矩阵，并演示单组合训练/评估流程。

## 核心实验结论

最终结果表明，LSTM 基线攻击能力整体明显强于 MLP，说明用户行为识别较依赖时序结构，单纯统计特征模型在部分场景下难以捕捉完整动态模式。

默认参数下，`ldp` 对攻击准确率的抑制通常最强，但对应的 MSE 更高、Pearson 相关性更低，体现了更强隐私扰动带来的可用性损失。

`noise` 在多数场景中保留了更多数据相关性，失真相对较低，但攻击者仍能保留较高识别能力，说明简单噪声并不总能充分削弱行为推断风险。

`adaptive_ldp` 在默认参数下呈现隐私与可用性之间的折中效果。其 profile 级消融结果显示，不同 `epsilon_min / epsilon_max`、`weight_sensitivity`、`weight_traffic` 和 `use_edge_budget_cap` 配置会改变防御强度与数据失真表现；面对不同模型和重训练攻击者时，该折中表现也存在差异。

在 `uci_har`、`kasteren`、`casas_hh101` 真实数据上，防御后攻击准确率均出现下降，说明防御效果不只存在于 `mock` 场景，也能在真实行为数据中观察到。

Cooja 结果支持节点侧 dummy 流量机制的功能性验证，可用于分析 fixed/retrain 攻击准确率变化；本项目进一步基于结构化 `METRIC_TX` / `METRIC_RX` 日志和 Contiki-NG Energest 计数器补充了 dummy/real 包比例、packet/byte overhead、Cooja 仿真时延和 Energest 仿真能耗估计。这些能耗值是仿真估计，不是硬件功耗仪测量。

## 实验覆盖情况

- mock 主矩阵：36/36
- real 主矩阵：108/108
- mock 参数扫描：36/36
- real 参数扫描：108/108
- `adaptive_ldp`：每个 dataset / seed / model / mode 组合包含 6 个 profile
- Cooja canonical：18/18
- `outputs/summaries/final_thesis/final_missing_outputs.json`：`[]`
- `outputs/summaries/final_thesis/parameter_scan_missing_outputs.json`：`[]`

完整矩阵结果已作为最终实验结果保留；Dashboard 提供单组合训练/评估演示，便于展示流程而不改变主实验结论。

## Web Dashboard

Dashboard 用于答辩演示和结果复核，可展示汇总指标、柱状图、参数扫描曲线、混淆矩阵，并支持单组合训练/评估演示。

启动命令：

```bash
python -m streamlit run apps/dashboard.py
```

Dashboard 包含五个页面：

- 总览：展示实验覆盖、核心汇总表和 Cooja 限制说明。
- 产物检索：按 dataset / seed / model / method / mode 浏览单组合产物。
- 图表与混淆矩阵：展示最终图像、实时绘制 confusion matrix，并查看参数扫描曲线。
- 训练 / 评估演示：基于已有处理数据运行单组合训练或评估任务。
- 运行历史：展示 Dashboard 触发过的单组合任务记录，记录文件为 `outputs/ui/run_history.jsonl`。


## 结果复核与产物索引

为便于复核，项目将研究说明、源码实现、实验结果和可视化入口整理如下：

- 最终汇总：`outputs/summaries/final_thesis/final_summary.csv`
- 最终汇总 JSON：`outputs/summaries/final_thesis/final_summary.json`
- 实验完整性审计：`outputs/summaries/final_thesis/final_symmetry_audit.json`
- 参数扫描覆盖审计：`outputs/summaries/final_thesis/parameter_scan_coverage_audit.json`
- 结果产物索引：`outputs/summaries/final_thesis/artifact_index.md`
- `adaptive_ldp` 消融总览：`outputs/summaries/final_thesis/adaptive_ldp_ablation_overview.md`
- Cooja 开销指标：`outputs/summaries/final_thesis/cooja/cooja_overhead_metrics.csv`
- Cooja 限制说明：`outputs/summaries/final_thesis/cooja/cooja_limitations.md`
- 最终图像目录：`outputs/figures/summaries/final_thesis/`
- 单组合源产物：`outputs/experiments/`

早期批次产物已整理到统一结构中；最终论文和答辩复核以标准结果包为准，历史迁移记录用于追溯：

- `outputs/summaries/layout/migration_map.csv`
- `outputs/summaries/layout/migration_report.md`

## 标准产物结构

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

普通防御实验路径：

```text
outputs/experiments/{dataset}/seed_{seed}/{model}/{method}/{mode}/
```

baseline 路径：

```text
outputs/experiments/{dataset}/seed_{seed}/{model}/baseline/
```

参数扫描路径：

```text
outputs/experiments/{dataset}/seed_{seed}/{model}/{method}/{mode}/parameter_scan/
```

Cooja 路径：

```text
outputs/experiments/cooja/seed_{seed}/random_forest/{dummy_method}/{mode}/
```

## 代码结构

源码按研究流程拆分为数据、模型、训练、评估、防御和展示模块：

- `src/core/`：配置加载、通用工具、绘图工具。
- `src/data/`：预处理、特征提取、Dataset 封装。
- `src/models/`：LSTM 和 MLP 模型定义。
- `src/training/`：训练循环、早停、checkpoint 写入。
- `src/evaluation/`：baseline 评估、防御后评估、参数扫描。
- `src/defenses/`：`noise`、`ldp`、`adaptive_ldp` 和防御流水线。
- `src/edge/`：`adaptive_ldp` 的边缘预算分配。
- `src/dashboard/`：Dashboard 路径、IO、绘图、运行器和历史记录。
- `src/artifacts/`：标准产物路径和 summary IO 工具。

更完整的代码说明见 `docs/CODE_STRUCTURE.md`。

## 单组合训练与评估演示

Dashboard 的演示功能聚焦于已有处理数据上的单组合训练与评估；完整矩阵和 Cooja 日志实验结果以最终结果包中的汇总为准。

等价命令行示例：

```bash
python experiments/demo/run_dashboard_job.py --dataset mock --seed 42 --model lstm --job train_baseline --max-epochs 5 --device auto
python experiments/demo/run_dashboard_job.py --dataset mock --seed 42 --model lstm --job evaluate_baseline --overwrite
python experiments/demo/run_dashboard_job.py --dataset mock --seed 42 --model lstm --job defense_eval_fixed --method ldp --overwrite
```

演示任务读取：

```text
data/processed/{dataset}/seed_{seed}/
data/defended/{dataset}/seed_{seed}/{method}/
```

演示任务写入对应的单组合产物路径和 `outputs/ui/run_history.jsonl`。

## 项目文件功能报告

- `docs/PROJECT_FILE_FUNCTION_REPORT.md`：按文件和目录解释项目功能，以及对应读取/生成的产物。
- `outputs/summaries/final_thesis/project_file_function_report.csv`：机器可读的文件功能索引。
- `outputs/summaries/final_thesis/artifact_index.md`：最终论文结果产物索引。

## 运行环境

```bash
pip install -r requirements.txt
```

主要依赖包括 Python 3.10+、pandas、numpy、scikit-learn、matplotlib、PyTorch 和 Streamlit。需要 GPU 训练时，可按 PyTorch 官方说明安装匹配的 CUDA 版本。

## Cooja 限制说明

Cooja 结果用于 fixed/retrain 攻击准确率展示和节点侧 dummy 流量功能性验证。节点级开销表已经包含 dummy/real 包比例、packet/byte overhead、Cooja 仿真时间下的端到端时延，以及 Contiki-NG Energest 计数器换算得到的仿真能耗估计。

这些指标的口径如下：

- `dummy_packet_ratio`、`packet_overhead_ratio` 和 `byte_overhead_ratio` 来自显式标记的 `METRIC_TX` / `METRIC_RX` 日志。
- `mean_delay_ms` 和 `p95_delay_ms` 是 Cooja 仿真时间下 REAL 包的端到端时延。
- `energy_mj` 是 Energest 计数器结合 `configs/cooja_energy_model.json` 中电压/电流参数换算得到的仿真估计。
- `is_hardware_measurement=false`，因此这些能耗数值不代表真实硬件功耗仪测量。

复现 Cooja 日志评估时，可复制：

```text
configs/cooja_defense_dummy_logs.template.json
```

并设置 `COOJA_LOG_ROOT` 指向本地 Cooja 日志目录。已完成结果中的本地 WSL 路径仅用于记录原实验来源，不作为通用复现路径。

## 维护说明

代码和产物中的 dataset、method、mode、JSON/CSV 字段名保持英文标识，以保证脚本和结果可复核。更细的维护约定见 `docs/MAINTENANCE_NOTES.md`。
