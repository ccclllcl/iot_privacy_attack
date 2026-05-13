# 实验入口说明

所有实验脚本都应在项目根目录运行，配置文件路径相对于项目根目录解析。

## core

`experiments/core/` 放单步流水线入口，适合调试、演示和小规模复现：

```bash
python experiments/core/generate_mock_data.py
python experiments/core/run_preprocess.py --config configs/default.yaml
python experiments/core/run_train.py --config configs/default.yaml --model lstm
python experiments/core/run_evaluate.py --config configs/default.yaml --model_path outputs/models/best_lstm.pt
python experiments/core/run_defense.py --config configs/default.yaml
python experiments/core/run_defense_eval.py --config configs/default.yaml --mode fixed_attacker --model_path outputs/models/best_lstm.pt
```

## real_public

`experiments/real_public/` 放真实公开数据导入与 benchmark 流程：

```bash
python experiments/real_public/imports/run_import_uci_har.py --config configs/default.yaml --auto-download
python experiments/real_public/benchmarks/run_real_public_benchmark.py --datasets uci_har,kasteren,casas_hh101 --seeds 42,123,2026 --models lstm,mlp --max-epochs 25 --skip-existing
python experiments/real_public/benchmarks/summarize_real_public_benchmark.py
```

## cooja

`experiments/cooja/` 放节点级日志实验和开销指标解析：

```bash
python experiments/cooja/run_cooja_defense_eval.py --manifest configs/cooja_defense_dummy_logs.json --out_dir outputs/experiments/cooja/eval --seeds "42,123,2026"
python experiments/cooja/parse_cooja_overhead_metrics.py --help
```

## demo

`experiments/demo/run_dashboard_job.py` 是 Dashboard 调用的单组合训练/评估入口：

```bash
python experiments/demo/run_dashboard_job.py --dataset mock --seed 42 --model lstm --job evaluate_baseline
```
