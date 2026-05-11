# 第四章混淆矩阵图修复报告

- 修改脚本: `scripts/final_thesis/generate_chapter4_figures.py`
- 修复方式: 使用 GridSpec 为 colorbar 单独预留轴，避免色条覆盖混淆矩阵主体。
- colorbar 是否取消: 否。
- 色彩方案: `Blues` 蓝白色。
- 是否重跑实验: 否，仅基于已有 `confusion.json` 重新绘图。
- 是否修改 Word 文档: 否。

## 重新生成的混淆矩阵图

- `outputs/figures/summaries/final_thesis/thesis_fig4_07_confusion_mock_baseline.png`
  - 尺寸: `1981 x 1483` px
  - DPI: `[300.0, 300.0]`
  - 遮挡风险: colorbar 位于独立轴，不覆盖热力图格子或数字标注。
- `outputs/figures/summaries/final_thesis/thesis_fig4_08_confusion_mock_adaptive_lstm_fixed.png`
  - 尺寸: `1981 x 1483` px
  - DPI: `[300.0, 300.0]`
  - 遮挡风险: colorbar 位于独立轴，不覆盖热力图格子或数字标注。
- `outputs/figures/summaries/final_thesis/thesis_fig4_09_confusion_mock_mlp_fixed.png`
  - 尺寸: `1981 x 1483` px
  - DPI: `[300.0, 300.0]`
  - 遮挡风险: colorbar 位于独立轴，不覆盖热力图格子或数字标注。
