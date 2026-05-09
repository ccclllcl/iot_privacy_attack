# 项目整理与交付修订记录

本文记录最终交付阶段对产物结构、Dashboard、代码结构和文档说明的整理情况。记录中的改动均围绕已有实验结果展开，不改变实验数值和论文结论。

## 产物结构标准化

- 实验源产物统一整理到 `outputs/experiments/`。
- 最终汇总结果集中在 `outputs/summaries/final_thesis/`。
- 最终图像集中在 `outputs/figures/summaries/final_thesis/`。
- 普通实验路径为 `outputs/experiments/{dataset}/seed_{seed}/{model}/{method}/{mode}/`。
- baseline 路径为 `outputs/experiments/{dataset}/seed_{seed}/{model}/baseline/`。
- Cooja 路径为 `outputs/experiments/cooja/seed_{seed}/random_forest/{dummy_method}/{mode}/`。
- 早期批次路径与标准路径的映射记录保留在 `outputs/summaries/layout/migration_map.csv` 和 `outputs/summaries/layout/migration_report.md`。

## 实验覆盖状态

- mock 主矩阵：36/36。
- real 主矩阵：108/108。
- mock 参数扫描：36/36。
- real 参数扫描：108/108。
- `adaptive_ldp` profile 覆盖：每个 dataset / seed / model / mode 组合 6 个 profile。
- `final_missing_outputs.json` 为 `[]`。
- `parameter_scan_missing_outputs.json` 为 `[]`。

## Dashboard 交付

- 正式 Dashboard 入口为 `apps/dashboard.py`。
- Dashboard 支持总览、产物检索、图表与混淆矩阵、训练 / 评估演示和运行历史。
- 单组合演示 runner 为 `experiments/demo/run_dashboard_job.py`。
- Dashboard 读取 `outputs/experiments/`、`outputs/summaries/final_thesis/` 和 `outputs/figures/summaries/final_thesis/`。
- Cooja 在 Dashboard 中作为已有结果展示，不补充真实能耗、真实端到端时延、packet/byte/IAT 或 dummy ratio。

## 代码结构整理

- 源码按职责分为 `src/core`、`src/data`、`src/models`、`src/training`、`src/evaluation`、`src/defenses`、`src/edge`、`src/dashboard` 和 `src/artifacts`。
- 根目录 `src/*.py` 保留为兼容 wrapper，用于连接早期脚本与当前分层源码。
- 代码结构审计输出为 `outputs/summaries/final_thesis/code_structure_audit.json` 和 `outputs/summaries/final_thesis/code_structure_audit.md`。

## 中文化与文件功能报告

- README、核心 docs、Dashboard 页面文案、Dashboard 后端提示和主要 docstring 已中文化。
- 路径、命令、CLI 参数、JSON/CSV 字段名、dataset、method、mode 和机器进度标记保持英文标识。
- `scripts/audit/audit_code_structure.py` 已修正，报告区分 `completed_moves`、`compatibility_wrappers`、`legacy_files`、`unknown_files` 和 `pending_recommendations`。
- 已生成 `docs/PROJECT_FILE_FUNCTION_REPORT.md`、`outputs/summaries/final_thesis/project_file_function_report.md`、`project_file_function_report.csv` 和 `project_file_function_report.json`。

## 统一项目文档叙述视角

- README 已调整为本科毕业设计项目说明视角，结构包括项目概述、研究问题、主要实现、核心实验结论、实验覆盖情况、Dashboard 和结果复核入口。
- `docs/REPOSITORY_DELIVERY_GUIDE.md` 已改写为项目复核指南，重点说明研究目标、核心结论、实验覆盖、Dashboard 展示和单组合追溯方式。
- `docs/DASHBOARD_GUIDE.md` 增加 Dashboard 展示的研究结论，包括 LSTM/MLP 对比、三种防御方法比较、fixed/retrain 威胁模型比较、参数扫描趋势、真实数据集变化和 Cooja 功能性验证边界。
- `docs/ARTIFACT_LAYOUT.md` 和 `docs/CODE_STRUCTURE.md` 已从纯工程目录说明改为服务实验复核和攻击—防御闭环的说明。
- `docs/PROJECT_STRUCTURE.md` 已同步调整为项目结构与研究复核关系说明。
- 内部维护规则下沉到 `docs/MAINTENANCE_NOTES.md`。
- 已生成 `outputs/summaries/final_thesis/text_tone_audit.md`、`text_tone_audit.json`、`text_tone_revision_report.md` 和 `text_tone_revision_report.json`。

## Cooja 结果边界

Cooja 结果用于 fixed/retrain 攻击准确率和节点侧 dummy 流量功能性验证。当前结果不声称真实能耗已测量，不声称真实端到端时延已测量，也不伪造 dummy/real 包比例。相关限制见 `outputs/summaries/final_thesis/cooja/cooja_limitations.md`。
