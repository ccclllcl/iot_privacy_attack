# Dashboard 使用指南

Dashboard 是本项目的结果浏览与答辩演示入口。它将最终汇总、单组合产物、图像、混淆矩阵和单组合训练/评估流程集中在同一个页面中，便于展示研究过程和实验结论。

启动命令：

```bash
python -m streamlit run apps/dashboard.py
```

## 1. Dashboard 展示的研究结论

Dashboard 支持从多个角度复核实验发现：

- LSTM/MLP 对比：展示时序模型与统计特征模型在不同 dataset 上的 baseline 与 defended 准确率差异。
- 三种防御方法比较：展示 `noise`、`ldp`、`adaptive_ldp` 对攻击准确率和数据失真的影响。
- fixed/retrain 威胁模型比较：展示固定攻击者与重训练攻击者下的防御稳定性差异。
- 参数扫描趋势：展示 epsilon、noise_scale 和 adaptive profile 对隐私—可用性折中的影响。
- 真实数据集内部变化：在 `uci_har`、`kasteren`、`casas_hh101` 内部比较 baseline 与 defended 结果。
- Cooja 功能性验证：展示节点侧 dummy 流量场景中的攻击准确率变化，并说明能耗、端到端时延和 dummy/real 包比例的当前边界。

## 2. 页面结构

- 总览：展示最终覆盖率、缺失产物数量、汇总表和 Cooja 限制。
- 产物检索：按 dataset、seed、model、method、mode 浏览单组合实验目录。
- 图表与混淆矩阵：展示最终图像，从 JSON 绘制 confusion matrix，并查看参数扫描曲线。
- 训练 / 评估演示：基于已有处理数据运行一个单组合训练或评估任务。
- 运行历史：读取 `outputs/ui/run_history.jsonl`，展示 Dashboard 触发过的任务。

## 3. 实现文件

- `apps/dashboard.py`：正式 Streamlit 入口。
- `src/dashboard/paths.py`：Dashboard 选择项和标准产物路径工具。
- `src/dashboard/io.py`：产物读取和绘图工具。
- `src/dashboard/runner.py`：子进程运行器、demo config 构建和 history 写入。
- `experiments/demo/run_dashboard_job.py`：Dashboard 调用的单组合命令行 runner。

早期 UI 保留在 `apps/legacy/`，用于记录项目演示入口的历史演进。

## 4. 数据与输出

Dashboard 的演示功能使用已有 processed data：

```text
data/processed/{dataset}/seed_{seed}/
```

防御评估使用已有 defended data：

```text
data/defended/{dataset}/seed_{seed}/{method}/
```

baseline 任务写入：

```text
outputs/experiments/{dataset}/seed_{seed}/{model}/baseline/
```

防御任务写入：

```text
outputs/experiments/{dataset}/seed_{seed}/{model}/{method}/{mode}/
```

模型保存到：

```text
outputs/models/{dataset}/seed_{seed}/{model}/
```

Dashboard 的单组合演示不会替代最终矩阵结果；其作用是展示训练、评估和产物写入流程。

## 5. 覆盖策略与运行历史

Dashboard 默认保护已有产物。页面提供 overwrite 选项和确认框，用于明确区分结果浏览与单组合演示运行。每次演示运行会追加记录到：

```text
outputs/ui/run_history.jsonl
```

运行历史包含 timestamp、dataset、seed、model、method、mode、job、status、duration、command 和 output_path，便于复核演示过程。

## 6. Cooja

Dashboard 展示已有 Cooja 结果，用于说明节点侧 dummy 流量机制的功能性验证。Cooja 页面中的 packet/byte/IAT、真实能耗和真实端到端时延字段保持限制说明，不以 Dashboard 演示补充或推断这些指标。
