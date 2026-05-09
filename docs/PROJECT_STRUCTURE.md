# 项目结构

本仓库将实现代码、实验入口、标准产物、最终汇总和文档说明分开管理。

## 顶层目录

- `src/`：按职责分层的核心源码，包括配置、数据、模型、训练、评估、防御、Dashboard 工具和产物路径工具。
- `experiments/`：命令行实验入口，包括单步实验、批处理实验、真实数据流程、Cooja 流程和 Dashboard demo runner。
- `configs/`：默认配置、Cooja 配置和 generated 配置。
- `scripts/`：最终汇总构建与审计入口。
- `apps/`：正式 Dashboard 和 legacy UI。
- `docs/`：项目结构、产物结构、交付说明和 Dashboard 使用说明。
- `data/`：按 dataset / seed 组织的 processed data 和 defended data。
- `outputs/experiments/`：按 dataset / seed / model / method / mode 组织的 canonical source artifacts。
- `outputs/summaries/final_thesis/`：最终论文结果汇总、审计和说明文件。
- `outputs/figures/summaries/final_thesis/`：最终论文图像。
- `outputs/figures/experiments/`：单组合诊断图像。

## 源码分层

- 配置核心（`src/core`）：配置加载、通用工具、通用绘图。
- 数据处理（`src/data`）：预处理、特征提取、Dataset 封装。
- 模型定义（`src/models`）：LSTM 和 MLP。
- 训练逻辑（`src/training`）：训练循环、早停、checkpoint。
- 评估逻辑（`src/evaluation`）：baseline 评估、防御评估、参数扫描。
- 防御算法（`src/defenses`）：`noise`、`ldp`、`adaptive_ldp` 和防御流水线。
- Dashboard 工具（`src/dashboard`）：路径、IO、运行器和历史记录。
- 产物路径工具（`src/artifacts`）：canonical artifact 路径和 summary IO。

旧 `src/*.py` 文件只作为兼容 wrapper 保留。

## 实验入口

- `experiments/core/run_train.py`：训练 LSTM 或 MLP attacker。
- `experiments/core/run_evaluate.py`：评估 baseline attacker。
- `experiments/core/run_defense.py`：生成 defended data。
- `experiments/core/run_defense_eval.py`：评估 fixed/retrain attacker。
- `experiments/core/run_compare.py`：对 `ldp`、`noise`、`adaptive_ldp` 做参数扫描。
- `experiments/demo/run_dashboard_job.py`：Dashboard 使用的单组合训练/评估 runner。
- `experiments/real_public/imports/`：真实数据导入流程。
- `experiments/real_public/benchmarks/`：真实数据 benchmark 流程。
- `experiments/cooja/`：Cooja 日志实验。

## Dashboard

推荐演示入口：

```bash
python -m streamlit run apps/dashboard.py
```

Dashboard 从 `outputs/experiments/`、`outputs/summaries/final_thesis/` 和 `outputs/figures/summaries/final_thesis/` 读取产物。它可以浏览结果和运行单组合 demo，不导入数据、不运行 Cooja 仿真、不启动完整实验矩阵。

## 标准产物路径

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

## 最终交付路径

论文和答辩优先查看：

- `outputs/summaries/final_thesis/`
- `outputs/figures/summaries/final_thesis/`
- `docs/REPOSITORY_DELIVERY_GUIDE.md`
- `docs/ARTIFACT_LAYOUT.md`
- `docs/CODE_STRUCTURE.md`

旧批次路径已经迁移，迁移记录保留在 `outputs/summaries/layout/migration_report.md`。
