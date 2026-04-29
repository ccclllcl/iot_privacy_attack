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
- 口径限制: 当前扫描来自 fixed_attacker；retrain 扫描缺失。

## 7. noise scale 参数扫描曲线
- 图路径: `D:\毕业设计毕业设计毕业设计毕业设计\Projects\iot_privacy_attack\outputs\figures\final_thesis\real_uci_noise_scan.png`
- 源文件: `outputs/reports/final_thesis/real/real_parameter_scan_noise.csv`
- 可写入论文结论: 可用于展示噪声强度上升时攻击准确率下降趋势。
- 口径限制: 当前扫描来自 fixed_attacker；retrain 扫描缺失。

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
