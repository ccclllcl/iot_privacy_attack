# iot_privacy_attack

## 项目简介

`iot_privacy_attack` 用于评估 IoT 时序数据上的用户行为推断攻击，以及 `noise`、`ldp`、`adaptive_ldp` 等数据侧隐私防御方法对攻击准确率和数据失真的影响。当前仓库已经整理为本科毕业设计最终交付形态：实验产物路径直接表达 `dataset / seed / model / method / mode`，源码也按配置、数据、训练、评估、防御、Dashboard 和产物路径等职责分层。

## 当前交付状态

- mock 主矩阵：36/36
- real 主矩阵：108/108
- mock 参数扫描：36/36
- real 参数扫描：108/108
- `adaptive_ldp`：每个 dataset / seed / model / mode 组合包含 6 个 profile
- Cooja canonical 结果：18/18
- `outputs/summaries/final_thesis/final_missing_outputs.json`：`[]`
- `outputs/summaries/final_thesis/parameter_scan_missing_outputs.json`：`[]`

日常查看、答辩演示和单组合训练/评估建议使用 Dashboard。除非明确需要复现实验，不建议重复运行完整矩阵。

## Web Dashboard

正式入口：

```bash
python -m streamlit run apps/dashboard.py
```

Dashboard 包含：

- 总览：查看覆盖率、缺失产物数量、汇总表和 Cooja 限制说明。
- 产物检索：按 dataset / seed / model / method / mode 检索 canonical artifacts。
- 图表与混淆矩阵：查看最终图、从 JSON 实时绘制 confusion matrix，并展示参数扫描曲线。
- 训练 / 评估演示：只运行用户选择的单个组合，不导入数据、不跑全量矩阵。
- 运行历史：查看 Dashboard 触发的单组合任务，记录在 `outputs/ui/run_history.jsonl`。

旧式命令 UI 仅保留为 `apps/legacy/ui_app.py`，不作为推荐入口。

## 标准产物结构

正式产物入口：

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

普通防御实验：

```text
outputs/experiments/{dataset}/seed_{seed}/{model}/{method}/{mode}/
```

baseline：

```text
outputs/experiments/{dataset}/seed_{seed}/{model}/baseline/
```

参数扫描：

```text
outputs/experiments/{dataset}/seed_{seed}/{model}/{method}/{mode}/parameter_scan/
```

Cooja：

```text
outputs/experiments/cooja/seed_{seed}/random_forest/{dummy_method}/{mode}/
```

最终汇总和图像：

```text
outputs/summaries/final_thesis/
outputs/figures/summaries/final_thesis/
outputs/figures/experiments/
```

旧批次路径已经迁移，迁移记录见 `outputs/summaries/layout/migration_report.md`。

## 代码结构

主要源码包按职责组织：

- `src/core/`：配置加载、通用工具、绘图工具。
- `src/data/`：预处理、特征提取、Dataset 封装。
- `src/models/`：LSTM 和 MLP 模型定义。
- `src/training/`：训练循环、早停、checkpoint 写入。
- `src/evaluation/`：baseline 评估、防御后评估、参数扫描。
- `src/defenses/`：防御算法和防御流水线。
- `src/edge/`：`adaptive_ldp` 的边缘预算分配。
- `src/dashboard/`：Dashboard 路径、IO、绘图、子进程运行器和历史记录。
- `src/artifacts/`：canonical artifact 路径和 summary IO 工具。

旧的 `src/*.py` import 路径保留为轻量兼容 wrapper，便于历史脚本不立即失效。新代码应优先使用上述分层包。详细说明见 `docs/CODE_STRUCTURE.md`。

## 最终结果导航

论文表格、图像和答辩核查优先查看：

- `outputs/summaries/final_thesis/final_summary.csv`
- `outputs/summaries/final_thesis/final_summary.json`
- `outputs/summaries/final_thesis/final_symmetry_audit.json`
- `outputs/summaries/final_thesis/parameter_scan_coverage_audit.json`
- `outputs/summaries/final_thesis/artifact_index.md`
- `outputs/summaries/final_thesis/adaptive_ldp_ablation_overview.md`
- `outputs/summaries/final_thesis/cooja/cooja_limitations.md`
- `outputs/figures/summaries/final_thesis/`

推荐阅读文档：

- `docs/REPOSITORY_DELIVERY_GUIDE.md`
- `docs/ARTIFACT_LAYOUT.md`
- `docs/CODE_STRUCTURE.md`
- `docs/DASHBOARD_GUIDE.md`
- `docs/PROJECT_FILE_FUNCTION_REPORT.md`

## 项目文件功能报告

- `docs/PROJECT_FILE_FUNCTION_REPORT.md`：按文件和目录解释项目功能，以及对应读取/生成的产物。
- `outputs/summaries/final_thesis/project_file_function_report.csv`：机器可读的文件功能索引。
- `outputs/summaries/final_thesis/artifact_index.md`：最终论文结果产物索引。

## 单组合训练与评估演示

Dashboard 和 demo runner 只使用已经存在的处理后数据：

```text
data/processed/{dataset}/seed_{seed}/
data/defended/{dataset}/seed_{seed}/{method}/
```

它们不会下载、导入或生成数据。若输入缺失，会提示缺失路径，而不会自动补齐。

等价命令行示例：

```bash
python experiments/demo/run_dashboard_job.py --dataset mock --seed 42 --model lstm --job train_baseline --max-epochs 5 --device auto
python experiments/demo/run_dashboard_job.py --dataset mock --seed 42 --model lstm --job evaluate_baseline --overwrite
python experiments/demo/run_dashboard_job.py --dataset mock --seed 42 --model lstm --job defense_eval_fixed --method ldp --overwrite
```

默认不会覆盖已有产物。只有明确传入 `--overwrite` 时，才会覆盖所选单组合对应的 canonical path。

## 常用维护命令

以下命令是轻量维护命令，不会运行完整实验矩阵：

```bash
python scripts/build_final_thesis_results.py
python scripts/audit_experiment_symmetry.py
python scripts/audit_repository_bloat.py
python scripts/audit_code_structure.py
python scripts/generate_project_file_report.py
```

根目录脚本是兼容入口，实际实现位于 `scripts/final_thesis/` 和 `scripts/audit/`。

## Cooja 说明

Cooja 结果用于 fixed/retrain 攻击准确率展示和功能性验证。当前交付不声称已经测量真实能耗、真实端到端时延，也不伪造 dummy/real 包比例。

复现 Cooja 日志评估时，可复制：

```text
configs/cooja_defense_dummy_logs.template.json
```

并设置 `COOJA_LOG_ROOT` 指向本地 Cooja 日志目录。已完成结果中的本地 WSL 路径仅用于记录原实验来源，不作为通用复现路径。

## 环境安装

```bash
pip install -r requirements.txt
```

主要依赖包括 Python 3.10+、pandas、numpy、scikit-learn、matplotlib、PyTorch 和 Streamlit。需要 GPU 训练时，请按 PyTorch 官方说明安装匹配的 CUDA 版本。

## 注意事项

- 不要把 `mock`、`uci_har`、`kasteren`、`casas_hh101`、`cooja` 翻译成中文。
- 不要把 `adaptive_ldp`、`ldp`、`noise`、`fixed_attacker`、`retrain_attacker` 翻译成中文。
- 不要修改 JSON/CSV 字段名、命令参数名或路径模板。
- 不要为了演示而重跑完整矩阵；Dashboard 只用于浏览和单组合演示。
- Cooja 中不可用的 packet/byte/IAT、能耗、时延字段应保留为限制说明，不应伪造。
