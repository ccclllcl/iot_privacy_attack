# 第四章论文图像审计

## 覆盖确认
- mock 主矩阵：36/36
- real 主矩阵：108/108
- mock 参数扫描：36/36
- real 参数扫描：108/108
- adaptive_ldp：每个组合 6 个 profile
- Cooja canonical：18/18
- final_missing_outputs.json 与 parameter_scan_missing_outputs.json 均为空数组。

## 可直接使用的现有图
- `outputs/figures/summaries/final_thesis/mock_model_mode_accuracy.png`
- `outputs/figures/summaries/final_thesis/mock_method_distortion.png`
- `outputs/figures/summaries/final_thesis/cooja_mode_accuracy.png`

## 本次为第四章重新生成的论文专用图
- `outputs/figures/summaries/final_thesis/thesis_fig4_01_mock_accuracy.png`
- `outputs/figures/summaries/final_thesis/thesis_fig4_02_mock_distortion.png`
- `outputs/figures/summaries/final_thesis/thesis_fig4_03_ldp_parameter_scan.png`
- `outputs/figures/summaries/final_thesis/thesis_fig4_04_noise_parameter_scan.png`
- `outputs/figures/summaries/final_thesis/thesis_fig4_05_adaptive_ldp_parameter_scan.png`
- `outputs/figures/summaries/final_thesis/thesis_fig4_06_adaptive_ldp_ablation.png`
- `outputs/figures/summaries/final_thesis/thesis_fig4_07_confusion_mock_baseline.png`
- `outputs/figures/summaries/final_thesis/thesis_fig4_08_confusion_mock_adaptive_lstm_fixed.png`
- `outputs/figures/summaries/final_thesis/thesis_fig4_09_confusion_mock_mlp_fixed.png`
- `outputs/figures/summaries/final_thesis/thesis_fig4_10_real_dataset_accuracy.png`
- `outputs/figures/summaries/final_thesis/thesis_fig4_11_real_dataset_parameter_scan.png`
- `outputs/figures/summaries/final_thesis/thesis_fig4_12_cooja_accuracy.png`

## 图例与索引检查
- LDP、noise、adaptive_ldp 参数扫描论文图均已包含 defended accuracy 与 MSE 图例，并标明 left axis / right axis。
- 论文混淆矩阵统一使用白色到深蓝色的蓝白色带。
- `artifact_index.md` 与 `figure_table_list.md` 已加入本次论文专用图路径。
- 早期 `real_uci_ldp_scan.png` 与 `real_uci_noise_scan.png` 仍可作为历史汇总图存在，但第四章改用本次重新绘制的论文专用图。

## 结论
本次只基于已有 CSV/JSON 重新绘图，没有重跑训练实验、参数扫描或 Cooja 仿真。
参数扫描图已经补充清晰图例，并在双 y 轴图中标明 left axis / right axis。
