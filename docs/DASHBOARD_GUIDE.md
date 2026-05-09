# Dashboard 使用指南

从仓库根目录启动：

```bash
python -m streamlit run apps/dashboard.py
```

## 页面说明

- 总览：展示最终覆盖率、缺失产物数量、汇总表和 Cooja 限制。
- 产物检索：按 dataset、seed、model、method、mode 浏览 canonical experiment folders。
- 图表与混淆矩阵：展示最终图像、从 JSON 绘制 confusion matrix，并查看参数扫描曲线。
- 训练 / 评估演示：只运行一个用户选择的训练或评估任务，并写入对应 canonical path。
- 运行历史：读取 `outputs/ui/run_history.jsonl`，展示 Dashboard 触发过的任务。

## 实现文件

- `apps/dashboard.py`：正式 Streamlit 入口。
- `src/dashboard/paths.py`：Dashboard 选择项和 canonical path 工具。
- `src/dashboard/io.py`：产物读取和绘图工具。
- `src/dashboard/runner.py`：子进程运行器、demo config 构建和 history 写入。
- `experiments/demo/run_dashboard_job.py`：Dashboard 调用的单组合命令行 runner。

旧 UI 仅保留在 `apps/legacy/`，不作为推荐入口。

## 数据策略

Dashboard 不导入或下载数据，只使用已经存在的 processed data：

```text
data/processed/{dataset}/seed_{seed}/
```

防御评估使用已经存在的 defended data：

```text
data/defended/{dataset}/seed_{seed}/{method}/
```

如果这些输入缺失，Dashboard 会显示缺失路径，不会自动下载、导入或生成数据。

## 覆盖策略

- 默认 `overwrite=false`，保护已有 canonical artifacts。
- 只有显式启用 `overwrite=true` 时，才覆盖所选输出路径。
- Streamlit 页面会要求二次确认。
- 每次运行会追加记录到 `outputs/ui/run_history.jsonl`。

## Cooja

Dashboard 只展示已有 Cooja 结果。它不运行 Cooja 仿真，不补真实能耗，不补真实端到端时延，也不伪造 packet/byte/IAT 字段。

## 标准输出

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
