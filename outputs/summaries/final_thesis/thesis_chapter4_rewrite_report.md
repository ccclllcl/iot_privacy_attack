# 第四章重写报告

- 输入 Word：`毕业论文_第四章实验结果重写版.docx`
- 输出 Word：`毕业论文_第四章图表与结果最终重写版.docx`
- 输出位置：本地指定路径，未作为仓库实验产物提交。
- 删除旧第四章 XML block 数：119
- 本次没有重跑训练实验、参数扫描或 Cooja 仿真。

## 使用的 CSV/JSON 结果
- `outputs/summaries/final_thesis/mock/mock_summary.csv`
- `outputs/summaries/final_thesis/real/real_summary.csv`
- `outputs/summaries/final_thesis/cooja/cooja_summary.csv`
- `outputs/summaries/final_thesis/mock/mock_parameter_scan_ldp.csv`
- `outputs/summaries/final_thesis/mock/mock_parameter_scan_noise.csv`
- `outputs/summaries/final_thesis/mock/mock_parameter_scan_adaptive_ldp.csv`
- `outputs/summaries/final_thesis/mock/mock_adaptive_ldp_ablation_summary.csv`
- `outputs/experiments/mock/seed_42/**/confusion.json`

## 插入或引用的论文图
- `outputs/figures/summaries/final_thesis/thesis_fig4_01_mock_accuracy.png`
- `outputs/figures/summaries/final_thesis/thesis_fig4_02_mock_distortion.png`
- `outputs/figures/summaries/final_thesis/thesis_fig4_03_ldp_parameter_scan.png`
- `outputs/figures/summaries/final_thesis/thesis_fig4_04_noise_parameter_scan.png`
- `outputs/figures/summaries/final_thesis/thesis_fig4_05_adaptive_ldp_parameter_scan.png`
- `outputs/figures/summaries/final_thesis/thesis_fig4_07_confusion_mock_baseline.png`
- `outputs/figures/summaries/final_thesis/thesis_fig4_08_confusion_mock_adaptive_lstm_fixed.png`
- `outputs/figures/summaries/final_thesis/thesis_fig4_10_real_dataset_accuracy.png`
- `outputs/figures/summaries/final_thesis/thesis_fig4_12_cooja_accuracy.png`

## 修正内容
- 第四章已按最终完整实验矩阵重新组织。
- 真实数据参数扫描表述已更新为覆盖 UCI HAR、Kasteren 与 CASAS。
- adaptive_ldp 已加入 6-profile 级消融解释。
- Cooja 部分只写 fixed/retrain 攻击准确率与节点侧功能性验证，不写真实能耗、真实端到端时延或 dummy/real 包比例量化。

## 格式检查
- 字体目标：中文宋体小四；英文和数字 Times New Roman 小四
- 检查正文 run 数：37
- 非 Times New Roman run 数：0
- 非宋体 eastAsia run 数：0
- 检查范围：仅检查本次重写的第四章正文与图题
- 目录：已通过本机 Word 自动化刷新目录和文档域。
- 打开验证：本机 Word 自动化能够打开、更新并保存输出稿；PNG 渲染预览工具因当前环境缺少 LibreOffice 未执行成功。
