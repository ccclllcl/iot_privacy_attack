# 图表清单

## 1. Mock LSTM/MLP baseline vs fixed/retrain accuracy 对比图
- 图路径: `D:\毕业设计毕业设计毕业设计毕业设计\Projects\iot_privacy_attack\outputs\figures\final_thesis\mock_model_mode_accuracy.png`
- 源文件: `outputs/reports/final_thesis/mock/mock_summary.csv`
- 可写入论文结论: 可用于展示 fixed_attacker 与 retrain_attacker 的差异趋势。
- 口径限制: 均值汇总会掩盖个别 seed 波动。

## 2. Mock 三种防御方法 MSE/MAE/Pearson 对比图
- 图路径: `D:\毕业设计毕业设计毕业设计毕业设计\Projects\iot_privacy_attack\outputs\figures\final_thesis\mock_method_distortion.png`
- 源文件: `outputs/reports/final_thesis/mock/mock_summary.csv`
- 可写入论文结论: 可用于展示防御强度与信号保真度之间权衡。
- 口径限制: 不同 mode 下共享同一 distortion 指标。

## 3. real uci_har LSTM/MLP baseline vs fixed/retrain accuracy 对比图
- 图路径: `D:\毕业设计毕业设计毕业设计毕业设计\Projects\iot_privacy_attack\outputs\figures\final_thesis\real_uci_har_model_mode_accuracy.png`
- 源文件: `outputs/reports/final_thesis/real/real_summary.csv`
- 可写入论文结论: 可用于展示 uci_har 数据集的防御效果。
- 口径限制: 若样本不平衡，宏平均与准确率可能有偏差。

## 4. real kasteren LSTM/MLP baseline vs fixed/retrain accuracy 对比图
- 图路径: `D:\毕业设计毕业设计毕业设计毕业设计\Projects\iot_privacy_attack\outputs\figures\final_thesis\real_kasteren_model_mode_accuracy.png`
- 源文件: `outputs/reports/final_thesis/real/real_summary.csv`
- 可写入论文结论: 可用于展示 kasteren 数据集的防御效果。
- 口径限制: 若样本不平衡，宏平均与准确率可能有偏差。

## 5. real casas_hh101 LSTM/MLP baseline vs fixed/retrain accuracy 对比图
- 图路径: `D:\毕业设计毕业设计毕业设计毕业设计\Projects\iot_privacy_attack\outputs\figures\final_thesis\real_casas_hh101_model_mode_accuracy.png`
- 源文件: `outputs/reports/final_thesis/real/real_summary.csv`
- 可写入论文结论: 可用于展示 casas_hh101 数据集的防御效果。
- 口径限制: 若样本不平衡，宏平均与准确率可能有偏差。

## 6. ldp epsilon 参数扫描曲线
- 图路径: `D:\毕业设计毕业设计毕业设计毕业设计\Projects\iot_privacy_attack\outputs\figures\final_thesis\real_uci_ldp_scan.png`
- 源文件: `outputs/reports/final_thesis/real/real_parameter_scan_ldp.csv`
- 可写入论文结论: 可用于展示 epsilon 变大时准确率恢复趋势。
- 口径限制: UCI HAR 扫描覆盖 LSTM/MLP 与 fixed/retrain；其他数据集仅作辅助口径。

## 7. noise scale 参数扫描曲线
- 图路径: `D:\毕业设计毕业设计毕业设计毕业设计\Projects\iot_privacy_attack\outputs\figures\final_thesis\real_uci_noise_scan.png`
- 源文件: `outputs/reports/final_thesis/real/real_parameter_scan_noise.csv`
- 可写入论文结论: 可用于展示噪声强度上升时攻击准确率下降趋势。
- 口径限制: UCI HAR 扫描覆盖 LSTM/MLP 与 fixed/retrain；其他数据集仅作辅助口径。

## 8. mock 代表性 confusion matrix
- 图路径: `D:\毕业设计毕业设计毕业设计毕业设计\Projects\iot_privacy_attack\outputs\figures\final_thesis\confusion_mock.png`
- 源文件: `D:\毕业设计毕业设计毕业设计毕业设计\Projects\iot_privacy_attack\outputs\defense\final_thesis\mock\seed_42\lstm\adaptive_ldp\fixed_attacker\confusion.json`
- 可写入论文结论: 可用于展示主要误分类模式。
- 口径限制: 仅展示单个 seed/model/method 样本。

## 9. uci_har 代表性 confusion matrix
- 图路径: `D:\毕业设计毕业设计毕业设计毕业设计\Projects\iot_privacy_attack\outputs\figures\final_thesis\confusion_uci_har.png`
- 源文件: `D:\毕业设计毕业设计毕业设计毕业设计\Projects\iot_privacy_attack\outputs\defense\final_thesis\real\uci_har\seed_42\lstm\adaptive_ldp\fixed_attacker\confusion.json`
- 可写入论文结论: 可用于展示主要误分类模式。
- 口径限制: 仅展示单个 seed/model/method 样本。

## 10. kasteren 代表性 confusion matrix
- 图路径: `D:\毕业设计毕业设计毕业设计毕业设计\Projects\iot_privacy_attack\outputs\figures\final_thesis\confusion_kasteren.png`
- 源文件: `D:\毕业设计毕业设计毕业设计毕业设计\Projects\iot_privacy_attack\outputs\defense\final_thesis\real\kasteren\seed_42\lstm\adaptive_ldp\fixed_attacker\confusion.json`
- 可写入论文结论: 可用于展示主要误分类模式。
- 口径限制: 仅展示单个 seed/model/method 样本。

## 11. casas_hh101 代表性 confusion matrix
- 图路径: `D:\毕业设计毕业设计毕业设计毕业设计\Projects\iot_privacy_attack\outputs\figures\final_thesis\confusion_casas_hh101.png`
- 源文件: `D:\毕业设计毕业设计毕业设计毕业设计\Projects\iot_privacy_attack\outputs\defense\final_thesis\real\casas_hh101\seed_42\lstm\adaptive_ldp\fixed_attacker\confusion.json`
- 可写入论文结论: 可用于展示主要误分类模式。
- 口径限制: 仅展示单个 seed/model/method 样本。

## 12. Cooja fixed/retrain accuracy 对比图
- 图路径: `D:\毕业设计毕业设计毕业设计毕业设计\Projects\iot_privacy_attack\outputs\figures\final_thesis\cooja_mode_accuracy.png`
- 源文件: `outputs/reports/final_thesis/cooja/cooja_summary.csv`
- 可写入论文结论: 可用于展示节点级防御在流量侧攻击下的变化。
- 口径限制: 依赖 Cooja 日志质量与可获得性。

## 13. Cooja 窗口数量代理开销图
- 图路径: `D:\毕业设计毕业设计毕业设计毕业设计\Projects\iot_privacy_attack\outputs\figures\final_thesis\cooja_window_overhead_proxy.png`
- 源文件: `outputs/reports/final_thesis/cooja/cooja_overhead_summary.csv`
- 可写入论文结论: 可用于说明当前日志只能支持窗口数量代理，而不能支持真实能耗或时延结论。
- 口径限制: 该图不是能耗或时延实测，只反映当前导出日志形成的窗口规模差异。

## 14. ldp parameter scans across available models/modes
- 图路径: `D:\毕业设计毕业设计毕业设计毕业设计\Projects\iot_privacy_attack\outputs\figures\final_thesis\parameter_scan_ldp_all_models_modes.png`
- 源文件: `outputs/reports/final_thesis/mock/mock_parameter_scan_ldp.csv;outputs/reports/final_thesis/real/real_parameter_scan_ldp.csv`
- 可写入论文结论: Shows parameter sensitivity separately by dataset; missing combinations are documented in parameter_scan_coverage_audit.json.
- 口径限制: Curves average available seeds and do not rank different datasets against each other.

## 15. noise parameter scans across available models/modes
- 图路径: `D:\毕业设计毕业设计毕业设计毕业设计\Projects\iot_privacy_attack\outputs\figures\final_thesis\parameter_scan_noise_all_models_modes.png`
- 源文件: `outputs/reports/final_thesis/mock/mock_parameter_scan_noise.csv;outputs/reports/final_thesis/real/real_parameter_scan_noise.csv`
- 可写入论文结论: Shows parameter sensitivity separately by dataset; missing combinations are documented in parameter_scan_coverage_audit.json.
- 口径限制: Curves average available seeds and do not rank different datasets against each other.

## 16. adaptive_ldp parameter scans across available models/modes
- 图路径: `D:\毕业设计毕业设计毕业设计毕业设计\Projects\iot_privacy_attack\outputs\figures\final_thesis\parameter_scan_adaptive_ldp_all_models_modes.png`
- 源文件: `outputs/reports/final_thesis/mock/mock_parameter_scan_adaptive_ldp.csv;outputs/reports/final_thesis/real/real_parameter_scan_adaptive_ldp.csv`
- 可写入论文结论: Shows parameter sensitivity separately by dataset; missing combinations are documented in parameter_scan_coverage_audit.json.
- 口径限制: Curves average available seeds and do not rank different datasets against each other.

## 17. Cooja per-seed accuracy
- 图路径: `D:\毕业设计毕业设计毕业设计毕业设计\Projects\iot_privacy_attack\outputs\figures\final_thesis\cooja_per_seed_accuracy.png`
- 源文件: `outputs/reports/final_thesis/cooja/cooja_per_seed.csv`
- 可写入论文结论: Shows fixed/retrain attacker behavior per seed for each Cooja dummy method.
- 口径限制: Depends on available Cooja radio/app logs and exported per-seed runs.

## 18. Cooja traffic metrics
- 图路径: `D:\毕业设计毕业设计毕业设计毕业设计\Projects\iot_privacy_attack\outputs\figures\final_thesis\cooja_traffic_metrics.png`
- 源文件: `outputs/reports/final_thesis/cooja/cooja_traffic_metrics.csv`
- 可写入论文结论: Shows available packet/byte overhead proxies from Cooja traffic windows.
- 口径限制: Not real energy or delay; dummy packet ratios are null when logs do not label dummy packets.
