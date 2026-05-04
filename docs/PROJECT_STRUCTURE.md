# 项目结构说明

本项目按“核心代码、实验入口、配置、数据产物、展示界面”拆分，根目录只保留项目级文件。

## 顶层目录

- `src/`：核心实现，包括预处理、特征、模型训练、评估、防御机制和参数扫描逻辑。
- `experiments/`：命令行实验入口，所有脚本都应从项目根目录运行。
- `configs/`：默认配置和批量实验生成的配置。
- `data/`：原始数据、预处理数据、防御后数据。该目录通常不提交。
- `outputs/`：模型、图表、报告、防御评估结果。该目录通常不提交。
- `apps/`：Streamlit 界面入口。
- `tools/`：维护工具，例如刷新前端图表、改写 Cooja 场景。
- `scripts/`：论文最终结果汇总与打包脚本。
- `web_assets/`：前端展示用图片资源。

## experiments 子目录

- `experiments/core/`：单次流水线入口。
  - `generate_mock_data.py`：生成 mock 智能家居事件。
  - `run_preprocess.py`：CSV 到滑窗序列与 MLP 特征。
  - `run_train.py`：训练 LSTM / MLP 攻击者。
  - `run_evaluate.py`：评估攻击者基线。
  - `run_defense.py`：生成防御后数据。
  - `run_defense_eval.py`：fixed / retrain 攻击者防御评估。
  - `run_compare.py`：LDP epsilon / noise scale 参数扫描。
  - `collect_confusion.py`：导出混淆矩阵 JSON。
- `experiments/batches/`：mock 多 seed、多模型、多防御方法矩阵。
- `experiments/real_public/`：UCI HAR、van Kasteren、CASAS 导入和真实数据矩阵。
- `experiments/cooja/`：Cooja 日志攻击、两组日志对比和节点级防御评估。

## 常用命令

```bash
python experiments/core/run_train.py --config configs/default.yaml --model lstm
python experiments/real_public/run_real_public_benchmark.py --datasets uci_har,kasteren,casas_hh101 --seeds 42,123 --models lstm,mlp --max-epochs 25 --skip-existing
python scripts/build_final_thesis_results.py
```
