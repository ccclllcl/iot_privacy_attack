# 实验入口说明

所有实验脚本都建议在项目根目录运行，配置文件路径仍然相对于项目根目录，例如 `configs/default.yaml`。

## core

`experiments/core/` 放单次流水线入口，适合调试、演示和小规模复现：

```bash
python experiments/core/generate_mock_data.py
python experiments/core/run_preprocess.py --config configs/default.yaml
python experiments/core/run_train.py --config configs/default.yaml --model lstm
python experiments/core/run_evaluate.py --config configs/default.yaml --model_path outputs/models/best_lstm.pt
python experiments/core/run_defense.py --config configs/default.yaml
python experiments/core/run_defense_eval.py --config configs/default.yaml --mode fixed_attacker --model_path outputs/models/best_lstm.pt
```

## batches

`experiments/batches/` 放 mock 数据的多 seed / 多模型 / 多防御矩阵：

```bash
python experiments/batches/run_all_methods_multiseed.py
```

## real_public

`experiments/real_public/` 放真实公开数据导入与全矩阵：

```bash
python experiments/real_public/run_import_uci_har.py --config configs/default.yaml --auto-download
python experiments/real_public/run_real_public_benchmark.py --datasets uci_har,kasteren,casas_hh101 --seeds 42,123 --models lstm,mlp --max-epochs 25 --skip-existing
python experiments/real_public/summarize_real_public_benchmark.py
```

## cooja

`experiments/cooja/` 放节点级日志实验：

```bash
python experiments/cooja/run_cooja_defense_eval.py --manifest configs/cooja_defense_dummy_logs.json --out_dir outputs/cooja_defense_eval_dummy --seeds "42,123,2026"
```
