# 项目文件功能与产物对应报告

生成时间：`2026-05-09T18:27:50`

## 1. 顶层文件

### `README.md`
- 类型：顶层说明
- 作用：项目入口说明，介绍交付状态、Dashboard、标准产物结构、代码结构和注意事项。
- 相关产物：docs/*; outputs/summaries/final_thesis/*

### `requirements.txt`
- 类型：依赖清单
- 作用：记录 Python 运行依赖。
- 相关产物：所有训练、评估、Dashboard 脚本

### `.gitignore`
- 类型：仓库配置
- 作用：控制运行产物、缓存和最终交付产物的跟踪策略。
- 相关产物：outputs/; data/; configs/; docs/

## 2. 配置文件 configs/

### `configs/default.yaml`
- 类型：配置文件
- 作用：默认实验配置，供 core CLI、Dashboard demo runner 和批处理脚本读取。
- 写入：outputs/ui/tmp_configs/ 可基于它生成临时配置
- 相关产物：experiments/core/*; experiments/demo/run_dashboard_job.py

### `configs/generated/{dataset}/seed_{seed}/{model}/`
- 类型：generated 配置
- 作用：按 dataset / seed / model / method / mode 保存已整理的 canonical 配置。
- 读取：configs/default.yaml
- 写入：单组合运行使用临时副本
- 相关产物：outputs/experiments/{dataset}/seed_{seed}/

### `configs/cooja*.json`
- 类型：Cooja 配置
- 作用：记录 Cooja 日志路径或模板路径。
- 读取：COOJA_LOG_ROOT 或本地 WSL 路径
- 写入：outputs/experiments/cooja/
- 相关产物：outputs/summaries/final_thesis/cooja/
- 备注：本地 WSL 路径仅记录原实验来源。

## 3. 应用入口 apps/

### `apps/dashboard.py`
- 类型：应用入口
- 作用：正式 Streamlit Dashboard，用于浏览产物、绘制图表、查看混淆矩阵和运行单组合 demo。
- 读取：outputs/experiments/; outputs/summaries/final_thesis/; outputs/figures/summaries/final_thesis/
- 写入：outputs/ui/run_history.jsonl; outputs/ui/tmp_configs/
- 相关产物：src/dashboard/*; experiments/demo/run_dashboard_job.py

### `apps/legacy/ui_app.py`
- 类型：legacy UI
- 作用：旧 UI 占位入口，仅保留历史说明，不推荐使用。
- 相关产物：apps/dashboard.py

## 4. 源码 src/

### `src/core/`
- 类型：源码模块
- 作用：配置、通用工具和通用绘图。
- 读取：configs/*.yaml
- 写入：各实验输出目录
- 相关产物：src/core/config.py; src/core/utils.py; src/core/plotting.py

### `src/data/`
- 类型：源码模块
- 作用：数据预处理、特征工程和 Dataset 封装。
- 读取：data/processed/{dataset}/seed_{seed}/
- 写入：data/processed/{dataset}/seed_{seed}/
- 相关产物：experiments/core/run_preprocess.py

### `src/models/`
- 类型：源码模块
- 作用：LSTM 和 MLP 模型定义。
- 写入：outputs/models/{dataset}/seed_{seed}/{model}/
- 相关产物：src/training/trainer.py; src/evaluation/evaluator.py

### `src/training/`
- 类型：源码模块
- 作用：训练 baseline 或 retrain attacker，并写入模型和训练曲线。
- 读取：data/processed/{dataset}/seed_{seed}/; data/defended/{dataset}/seed_{seed}/{method}/
- 写入：outputs/models/{dataset}/seed_{seed}/{model}/; outputs/figures/experiments/
- 相关产物：experiments/core/run_train.py; experiments/demo/run_dashboard_job.py

### `src/evaluation/`
- 类型：源码模块
- 作用：评估 baseline、防御后攻击和参数扫描。
- 读取：outputs/models/; data/processed/; data/defended/
- 写入：outputs/experiments/; outputs/figures/experiments/
- 相关产物：experiments/core/run_evaluate.py; run_defense_eval.py; run_compare.py

### `src/defenses/`
- 类型：源码模块
- 作用：实现 `noise`、`ldp`、`adaptive_ldp` 和防御流水线。
- 读取：data/processed/{dataset}/seed_{seed}/
- 写入：data/defended/{dataset}/seed_{seed}/{method}/
- 相关产物：experiments/core/run_defense.py

### `src/edge/`
- 类型：源码模块
- 作用：`adaptive_ldp` 使用的边缘预算分配工具。
- 读取：configs/default.yaml 中 adaptive_ldp 配置
- 写入：defense_report.json
- 相关产物：src/defenses/adaptive_ldp_defense.py

### `src/dashboard/`
- 类型：源码模块
- 作用：Dashboard 路径、IO、绘图、子进程运行器和运行历史工具。
- 读取：outputs/experiments/; outputs/summaries/final_thesis/
- 写入：outputs/ui/run_history.jsonl; outputs/ui/tmp_configs/
- 相关产物：apps/dashboard.py

### `src/artifacts/`
- 类型：源码模块
- 作用：集中维护 canonical artifact 路径和 summary IO。
- 相关产物：scripts/audit/*; apps/dashboard.py

### `src/*.py`
- 类型：兼容 wrapper
- 作用：旧 import 路径的兼容层，只 re-export 新分层包。
- 读取：src/core/; src/data/; src/evaluation/; src/dashboard/
- 相关产物：历史脚本

## 5. 实验入口 experiments/

### `experiments/core/`
- 类型：实验入口
- 作用：单步 CLI：预处理、训练、评估、防御、参数扫描和混淆矩阵收集。
- 读取：configs/default.yaml; data/processed/; data/defended/; outputs/models/
- 写入：outputs/experiments/; data/defended/; outputs/models/; outputs/figures/experiments/
- 相关产物：src/*

### `experiments/batches/`
- 类型：批处理入口
- 作用：多 seed / 全矩阵复现脚本，日常交付审查不运行。
- 读取：configs/default.yaml
- 写入：outputs/experiments/ 等批量产物
- 相关产物：README.md 中不作为常规命令

### `experiments/real_public/imports/`
- 类型：真实数据导入
- 作用：导入 `uci_har`、`kasteren`、`casas_hh101` 原始数据并生成 processed data。
- 读取：data/raw/; 外部公开数据集
- 写入：data/processed/{dataset}/seed_{seed}/
- 相关产物：experiments/real_public/benchmarks/

### `experiments/real_public/benchmarks/`
- 类型：真实数据 benchmark
- 作用：真实数据完整 benchmark 和汇总脚本。
- 读取：data/processed/{dataset}/seed_{seed}/; configs/generated/
- 写入：outputs/experiments/{dataset}/; outputs/summaries/final_thesis/real/
- 相关产物：outputs/summaries/final_thesis/parameter_scan_coverage_audit.json

### `experiments/cooja/`
- 类型：Cooja 实验入口
- 作用：读取 Cooja 日志，生成攻击准确率和防御评估结果。
- 读取：configs/cooja*.json; Cooja 日志
- 写入：outputs/experiments/cooja/; outputs/summaries/final_thesis/cooja/
- 相关产物：cooja_limitations.md

### `experiments/demo/run_dashboard_job.py`
- 类型：Dashboard demo
- 作用：Dashboard 调用的单组合训练/评估 runner。
- 读取：data/processed/; data/defended/; outputs/models/
- 写入：outputs/experiments/; outputs/models/; outputs/ui/run_history.jsonl
- 相关产物：apps/dashboard.py

## 6. 汇总与审计 scripts/

### `scripts/final_thesis/build_final_thesis_results.py`
- 类型：汇总脚本
- 作用：从 canonical artifacts 构建最终 summary 和 figure。
- 读取：outputs/experiments/
- 写入：outputs/summaries/final_thesis/; outputs/figures/summaries/final_thesis/
- 相关产物：scripts/build_final_thesis_results.py

### `scripts/audit/audit_experiment_symmetry.py`
- 类型：审计脚本
- 作用：检查主矩阵、参数扫描和 Cooja canonical 产物完整性。
- 读取：outputs/experiments/; outputs/summaries/final_thesis/
- 写入：outputs/summaries/final_thesis/final_symmetry_audit.*
- 相关产物：scripts/audit_experiment_symmetry.py

### `scripts/audit/audit_repository_bloat.py`
- 类型：审计脚本
- 作用：检查仓库 tracked 文件、路径卫生和删除候选。
- 读取：git ls-files; outputs/summaries/final_thesis/
- 写入：outputs/summaries/final_thesis/repository_bloat_audit.*
- 相关产物：scripts/audit_repository_bloat.py

### `scripts/audit/audit_code_structure.py`
- 类型：审计脚本
- 作用：检查代码职责分层、兼容 wrapper、legacy 文件和 unknown 文件。
- 读取：src/; apps/; experiments/; scripts/; tools/; docs/
- 写入：outputs/summaries/final_thesis/code_structure_audit.*
- 相关产物：scripts/audit_code_structure.py

### `scripts/audit/generate_project_file_report.py`
- 类型：报告脚本
- 作用：生成本文件功能与产物对应报告。
- 读取：git ls-files; 项目目录结构
- 写入：docs/PROJECT_FILE_FUNCTION_REPORT.md; outputs/summaries/final_thesis/project_file_function_report.*
- 相关产物：scripts/generate_project_file_report.py

## 7. 数据产物 data/

### `data/processed/{dataset}/seed_{seed}/`
- 类型：数据产物
- 作用：处理后的训练/验证/测试数据，供训练和评估读取。
- 读取：data/raw/ 或导入脚本
- 写入：sequences.npz; mlp_features.npz; meta.json
- 相关产物：src/data/; src/training/; src/evaluation/

### `data/defended/{dataset}/seed_{seed}/{method}/`
- 类型：数据产物
- 作用：防御后的训练/验证/测试数据，供 fixed/retrain attacker 评估读取。
- 读取：data/processed/{dataset}/seed_{seed}/
- 写入：defended_sequences.npz; defended_mlp_features.npz
- 相关产物：src/defenses/; src/evaluation/defense_evaluator.py

## 8. 实验产物 outputs/experiments/

### `outputs/experiments/{dataset}/seed_{seed}/{model}/baseline/`
- 类型：实验产物
- 作用：baseline 指标、混淆矩阵、分类报告和 source manifest。
- 读取：outputs/models/{dataset}/seed_{seed}/{model}/baseline/
- 写入：baseline_metrics.json; baseline_confusion.json; baseline_classification_report.txt
- 相关产物：outputs/summaries/final_thesis/final_summary.csv

### `outputs/experiments/{dataset}/seed_{seed}/{model}/{method}/{mode}/`
- 类型：实验产物
- 作用：防御实验指标、混淆矩阵、trace、defense report 和 source manifest。
- 读取：data/defended/{dataset}/seed_{seed}/{method}/; outputs/models/
- 写入：metrics.json; confusion.json; classification_report.txt; trace.json; defense_report.json
- 相关产物：outputs/summaries/final_thesis/final_summary.csv

### `outputs/experiments/{dataset}/seed_{seed}/{model}/{method}/{mode}/parameter_scan/`
- 类型：实验产物
- 作用：参数扫描结果和 profile 配置。
- 读取：outputs/experiments/{dataset}/seed_{seed}/{model}/{method}/{mode}/
- 写入：comparison_results.csv; scan_summary.json; scan_trace.json; profile_config.json
- 相关产物：parameter_scan_coverage_audit.json

### `outputs/experiments/cooja/seed_{seed}/random_forest/{dummy_method}/{mode}/`
- 类型：Cooja 产物
- 作用：Cooja fixed/retrain 攻击准确率和 source manifest。
- 读取：Cooja 日志; outputs/experiments/cooja/eval/
- 写入：metrics.json; source_manifest.json
- 相关产物：outputs/summaries/final_thesis/cooja/

## 9. 汇总产物 outputs/summaries/final_thesis/

### `outputs/summaries/final_thesis/`
- 类型：最终汇总
- 作用：论文结果包根目录，包含 final summary、覆盖审计、参数扫描审计、报告和索引。
- 读取：outputs/experiments/
- 写入：final_summary.csv; final_summary.json; final_symmetry_audit.json; parameter_scan_coverage_audit.json
- 相关产物：README.md; docs/REPOSITORY_DELIVERY_GUIDE.md

### `outputs/summaries/final_thesis/mock/`
- 类型：最终汇总
- 作用：mock 数据集 summary、参数扫描 summary 和 adaptive_ldp 消融 summary。
- 读取：outputs/experiments/mock/
- 写入：mock_summary.csv; mock_parameter_scan_*.csv; mock_adaptive_ldp_ablation_summary.csv
- 相关产物：final_summary.csv

### `outputs/summaries/final_thesis/real/`
- 类型：最终汇总
- 作用：真实数据 summary、参数扫描 summary、dataset meta 和 adaptive_ldp 消融 summary。
- 读取：outputs/experiments/uci_har/; outputs/experiments/kasteren/; outputs/experiments/casas_hh101/
- 写入：real_summary.csv; real_parameter_scan_*.csv; real_adaptive_ldp_ablation_summary.csv
- 相关产物：final_summary.csv

### `outputs/summaries/final_thesis/cooja/`
- 类型：最终汇总
- 作用：Cooja summary、per-seed 结果、traffic metrics 和限制说明。
- 读取：outputs/experiments/cooja/
- 写入：cooja_summary.csv; cooja_per_seed.csv; cooja_traffic_metrics.csv; cooja_limitations.md
- 相关产物：final_summary.csv

## 10. 图像产物 outputs/figures/

### `outputs/figures/summaries/final_thesis/`
- 类型：图像产物
- 作用：最终论文图像，包括准确率、失真、参数扫描、消融和 Cooja 图。
- 读取：outputs/summaries/final_thesis/
- 写入：*.png
- 相关产物：figure_table_list.md

### `outputs/figures/experiments/{dataset}/seed_{seed}/{model}/{method}/{mode}/`
- 类型：图像产物
- 作用：单组合诊断图像。
- 读取：outputs/experiments/{dataset}/seed_{seed}/{model}/{method}/{mode}/
- 写入：*.png
- 相关产物：Dashboard 图表页面

## 11. 模型产物 outputs/models/

### `outputs/models/{dataset}/seed_{seed}/{model}/`
- 类型：模型产物
- 作用：baseline 模型、retrain 模型和 Dashboard demo 生成模型。
- 读取：src/training/trainer.py
- 写入：*.pt
- 相关产物：experiments/demo/run_dashboard_job.py; src/evaluation/evaluator.py

## 12. 推荐阅读顺序

1. `README.md`
2. `docs/REPOSITORY_DELIVERY_GUIDE.md`
3. `docs/ARTIFACT_LAYOUT.md`
4. `docs/CODE_STRUCTURE.md`
5. `docs/DASHBOARD_GUIDE.md`
6. `docs/PROJECT_FILE_FUNCTION_REPORT.md`
7. `outputs/summaries/final_thesis/artifact_index.md`
8. `outputs/summaries/final_thesis/final_summary.csv`
