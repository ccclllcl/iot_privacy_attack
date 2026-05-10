# 最终结果产物索引

本索引用于连接论文结论、答辩展示和单组合产物追溯。核心研究结论见 `README.md` 与 `outputs/summaries/final_thesis/final_summary.csv`。

## A. 主矩阵结果

- mock 汇总：`outputs/summaries/final_thesis/mock/mock_summary.csv`
- real 汇总：`outputs/summaries/final_thesis/real/real_summary.csv`
- 最终总表：`outputs/summaries/final_thesis/final_summary.csv`
- 最终总表 JSON：`outputs/summaries/final_thesis/final_summary.json`
- 覆盖审计：`outputs/summaries/final_thesis/final_coverage_audit.json`
- 对称性审计：`outputs/summaries/final_thesis/final_symmetry_audit.json`

## B. 参数扫描结果

- `outputs/summaries/final_thesis/mock/mock_parameter_scan_ldp.csv`
- `outputs/summaries/final_thesis/mock/mock_parameter_scan_noise.csv`
- `outputs/summaries/final_thesis/mock/mock_parameter_scan_adaptive_ldp.csv`
- `outputs/summaries/final_thesis/real/real_parameter_scan_ldp.csv`
- `outputs/summaries/final_thesis/real/real_parameter_scan_noise.csv`
- `outputs/summaries/final_thesis/real/real_parameter_scan_adaptive_ldp.csv`
- `outputs/summaries/final_thesis/parameter_scan_coverage_audit.json`

## C. `adaptive_ldp` 消融汇总

- `outputs/summaries/final_thesis/mock/mock_adaptive_ldp_ablation_summary.csv`
- `outputs/summaries/final_thesis/mock/mock_adaptive_ldp_ablation_summary.md`
- `outputs/summaries/final_thesis/real/real_adaptive_ldp_ablation_summary.csv`
- `outputs/summaries/final_thesis/real/real_adaptive_ldp_ablation_summary.md`
- `outputs/summaries/final_thesis/adaptive_ldp_ablation_overview.md`

## D. 单组合源产物

- mock 实验：`outputs/experiments/mock/`
- UCI HAR 实验：`outputs/experiments/uci_har/`
- Kasteren 实验：`outputs/experiments/kasteren/`
- CASAS HH101 实验：`outputs/experiments/casas_hh101/`
- Cooja 实验：`outputs/experiments/cooja/`
- 历史迁移映射：`outputs/summaries/layout/migration_map.csv`

## E. Cooja

- `outputs/summaries/final_thesis/cooja/cooja_summary.csv`
- `outputs/summaries/final_thesis/cooja/cooja_per_seed.csv`
- `outputs/summaries/final_thesis/cooja/cooja_traffic_metrics.csv`
- `outputs/summaries/final_thesis/cooja/cooja_limitations.md`

## F. 图像

- `outputs/figures/summaries/final_thesis/mock_model_mode_accuracy.png`
- `outputs/figures/summaries/final_thesis/mock_method_distortion.png`
- `outputs/figures/summaries/final_thesis/real_uci_har_model_mode_accuracy.png`
- `outputs/figures/summaries/final_thesis/real_kasteren_model_mode_accuracy.png`
- `outputs/figures/summaries/final_thesis/real_casas_hh101_model_mode_accuracy.png`
- `outputs/figures/summaries/final_thesis/parameter_scan_ldp_all_models_modes.png`
- `outputs/figures/summaries/final_thesis/parameter_scan_noise_all_models_modes.png`
- `outputs/figures/summaries/final_thesis/parameter_scan_adaptive_ldp_all_models_modes.png`
- `outputs/figures/summaries/final_thesis/adaptive_ldp_ablation_mock_accuracy.png`
- `outputs/figures/summaries/final_thesis/adaptive_ldp_ablation_mock_distortion.png`
- `outputs/figures/summaries/final_thesis/adaptive_ldp_ablation_real_accuracy.png`
- `outputs/figures/summaries/final_thesis/adaptive_ldp_ablation_real_distortion.png`
- `outputs/figures/summaries/final_thesis/cooja_per_seed_accuracy.png`
- `outputs/figures/summaries/final_thesis/cooja_traffic_metrics.png`
- `outputs/figures/summaries/final_thesis/cooja_mode_accuracy.png`
- `outputs/figures/summaries/final_thesis/cooja_window_overhead_proxy.png`
- `outputs/figures/summaries/final_thesis/confusion_mock.png`
- `outputs/figures/summaries/final_thesis/confusion_uci_har.png`
- `outputs/figures/summaries/final_thesis/confusion_kasteren.png`
- `outputs/figures/summaries/final_thesis/confusion_casas_hh101.png`

## G. 第四章论文专用图

- `outputs/figures/summaries/final_thesis/thesis_fig4_01_mock_accuracy.png`：图4.1 mock 场景准确率对比；数据来源 `outputs/summaries/final_thesis/mock/mock_summary.csv`；适合第 4.2、4.3 节；比较 LSTM/MLP 在 baseline、fixed_attacker、retrain_attacker 下的识别准确率变化。；不同 seed 已取均值，正文解释以趋势为主。
- `outputs/figures/summaries/final_thesis/thesis_fig4_02_mock_distortion.png`：图4.2 mock 场景失真指标对比；数据来源 `outputs/summaries/final_thesis/mock/mock_summary.csv`；适合第 4.4 节；展示 MSE、MAE、Pearson_r 对隐私—可用性权衡的支持。；Pearson 与误差指标分轴/分面展示，避免混合解释。
- `outputs/figures/summaries/final_thesis/thesis_fig4_03_ldp_parameter_scan.png`：图4.3 LDP 参数扫描；数据来源 `outputs/summaries/final_thesis/mock/mock_parameter_scan_ldp.csv`；适合第 4.3.3、4.4 节；展示 epsilon 增大时 defended_acc 与 MSE 的同步变化。；口径为 mock、LSTM、fixed_attacker，三组 seed 平均。
- `outputs/figures/summaries/final_thesis/thesis_fig4_04_noise_parameter_scan.png`：图4.4 noise 参数扫描；数据来源 `outputs/summaries/final_thesis/mock/mock_parameter_scan_noise.csv`；适合第 4.3.3、4.4 节；展示 noise_scale 增大时攻击准确率和失真指标的变化。；口径为 mock、LSTM、fixed_attacker，三组 seed 平均。
- `outputs/figures/summaries/final_thesis/thesis_fig4_05_adaptive_ldp_parameter_scan.png`：图4.5 adaptive_ldp profile 参数扫描；数据来源 `outputs/summaries/final_thesis/mock/mock_parameter_scan_adaptive_ldp.csv`；适合第 4.3.3、4.4 节；展示 6 个 adaptive_ldp profile 的 defended_acc 与 MSE。；属于 profile 级实验观察，不作为形式化理论证明。
- `outputs/figures/summaries/final_thesis/thesis_fig4_06_adaptive_ldp_ablation.png`：图4.6 adaptive_ldp 消融图；数据来源 `outputs/summaries/final_thesis/mock/mock_adaptive_ldp_ablation_summary.csv`；适合第 4.4 节；展示不同预算范围、风险权重和边缘预算裁剪接口下的结果差异。；口径为 mock、LSTM、fixed_attacker。
- `outputs/figures/summaries/final_thesis/thesis_fig4_07_confusion_mock_baseline.png`：图4.7 LSTM 基线混淆矩阵；数据来源 `outputs/experiments/mock/seed_42/lstm/baseline/baseline_confusion.json`；适合第 4.5 节；展示无防御状态下的主要误分布。；单 seed 代表性样本，不替代全矩阵均值。
- `outputs/figures/summaries/final_thesis/thesis_fig4_08_confusion_mock_adaptive_lstm_fixed.png`：图4.8 adaptive_ldp 下 LSTM fixed_attacker 混淆矩阵；数据来源 `outputs/experiments/mock/seed_42/lstm/adaptive_ldp/fixed_attacker/confusion.json`；适合第 4.5 节；展示防御后类别预测分布如何变化。；单 seed 代表性样本。
- `outputs/figures/summaries/final_thesis/thesis_fig4_09_confusion_mock_mlp_fixed.png`：图4.9 adaptive_ldp 下 MLP fixed_attacker 混淆矩阵；数据来源 `outputs/experiments/mock/seed_42/mlp/adaptive_ldp/fixed_attacker/confusion.json`；适合第 4.5 节；展示 MLP 在相同防御下的错误集中情况。；如篇幅有限，正文可只选用 LSTM 相关矩阵。
- `outputs/figures/summaries/final_thesis/thesis_fig4_10_real_dataset_accuracy.png`：图4.10 真实数据集准确率对比；数据来源 `outputs/summaries/final_thesis/real/real_summary.csv`；适合第 4.7 节；展示 UCI HAR、Kasteren、CASAS 各自内部 baseline 到 defended 的变化。；不同数据集类别数和任务定义不同，不做绝对排名。
- `outputs/figures/summaries/final_thesis/thesis_fig4_11_real_dataset_parameter_scan.png`：图4.11 真实数据集 LDP 参数扫描；数据来源 `outputs/summaries/final_thesis/real/real_parameter_scan_ldp.csv`；适合第 4.7 节；展示真实数据上参数扫描覆盖后的趋势支持。；按数据集分面，口径为 LSTM fixed_attacker。
- `outputs/figures/summaries/final_thesis/thesis_fig4_12_cooja_accuracy.png`：图4.12 Cooja 节点级准确率对比；数据来源 `outputs/summaries/final_thesis/cooja/cooja_summary.csv`；适合第 4.6 节；展示 dummy_noise、dummy_ldp、dummy_adaptive_ldp 在 fixed/retrain 下的攻击准确率变化。；Cooja 部分只作节点侧功能性验证，不表示真实能耗或端到端时延测量。
