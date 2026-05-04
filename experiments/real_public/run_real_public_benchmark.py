#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run full benchmark matrix on real public datasets.

Matrix coverage:
- datasets: uci_har, kasteren, casas_hh101
- seeds: configurable (default 42,123,2026)
- models: lstm, mlp
- defense methods: noise, ldp, adaptive_ldp
- attacker modes: fixed_attacker, retrain_attacker
- parameter scans: ldp epsilon + noise scale (on lstm baseline checkpoint)

All outputs are written under:
- data/processed/real_public_benchmark/
- data/defended/real_public_benchmark/
- outputs/models/real_public_benchmark/
- outputs/defense/real_public_benchmark/
- outputs/reports/real_public_benchmark/
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
BASE_CONFIG = ROOT / "configs" / "default.yaml"
GEN_DIR = ROOT / "configs" / "generated_real_public"

DEFAULT_DATASETS = ["uci_har", "kasteren"]
DEFAULT_SEEDS = [42, 123, 2026]
DEFAULT_MODELS = ["lstm", "mlp"]
METHODS = ["adaptive_ldp", "ldp", "noise"]


def _rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _run(cmd: list[str]) -> None:
    print(f"\n[RUN] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def _save_cfg(cfg: dict[str, Any], name: str) -> Path:
    GEN_DIR.mkdir(parents=True, exist_ok=True)
    out = GEN_DIR / f"{name}.yaml"
    with out.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    return out


def _parse_list(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def _parse_seeds(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _dataset_tag(dataset_key: str) -> str:
    if dataset_key == "casas_hh101":
        return "casas_hh101"
    return dataset_key


def _base_cfg(dataset_key: str, seed: int, max_epochs: int) -> dict[str, Any]:
    with BASE_CONFIG.open("r", encoding="utf-8") as f:
        cfg: dict[str, Any] = yaml.safe_load(f)

    cfg.setdefault("experiment", {})
    cfg["experiment"]["random_seed"] = int(seed)

    ds_tag = _dataset_tag(dataset_key)
    run_tag = f"{ds_tag}/seed_{seed}"
    p = cfg.setdefault("paths", {})
    p["processed_dir"] = f"data/processed/real_public_benchmark/{run_tag}"
    p["defended_dir"] = f"data/defended/real_public_benchmark/{run_tag}/adaptive_ldp"
    p["models_dir"] = f"outputs/models/real_public_benchmark/{run_tag}"
    p["figures_dir"] = f"outputs/figures/real_public_benchmark/{run_tag}"
    p["reports_dir"] = f"outputs/reports/real_public_benchmark/{run_tag}"
    p["defense_dir"] = f"outputs/defense/real_public_benchmark/{run_tag}/adaptive_ldp"

    # Keep runs practical but still representative.
    train = cfg.setdefault("train", {})
    train["num_epochs"] = int(min(int(train.get("num_epochs", 80)), int(max_epochs)))
    train["early_stopping_patience"] = int(min(int(train.get("early_stopping_patience", 10)), 6))
    return cfg


def _run_import(dataset_key: str, cfg_path: Path) -> None:
    if dataset_key == "uci_har":
        _run([sys.executable, "experiments/real_public/run_import_uci_har.py", "--config", _rel(cfg_path), "--auto-download"])
        return
    if dataset_key == "kasteren":
        _run([sys.executable, "experiments/real_public/run_import_kasteren.py", "--config", _rel(cfg_path), "--auto-download"])
        return
    if dataset_key == "casas_hh101":
        _run(
            [
                sys.executable,
                "experiments/real_public/run_import_casas.py",
                "--config",
                _rel(cfg_path),
                "--home",
                "hh101",
                "--auto-download",
            ]
        )
        return
    raise ValueError(f"Unsupported dataset key: {dataset_key}")


def _maybe_run(cmd: list[str], outputs: list[Path], skip_existing: bool) -> None:
    if skip_existing and outputs and all(p.exists() for p in outputs):
        print(f"[SKIP] outputs exist: {outputs[0]}", flush=True)
        return
    _run(cmd)


def run_one(
    dataset_key: str,
    seed: int,
    models: list[str],
    max_epochs: int,
    skip_existing: bool,
) -> dict[str, Any]:
    base = _base_cfg(dataset_key, seed, max_epochs=max_epochs)
    ds_tag = _dataset_tag(dataset_key)
    base_cfg_path = _save_cfg(base, f"{ds_tag}.seed_{seed}.base")
    paths = base["paths"]
    processed_seq = ROOT / paths["processed_dir"] / "sequences.npz"
    if skip_existing and processed_seq.exists():
        print(f"[SKIP] dataset import exists: {processed_seq}", flush=True)
    else:
        _run_import(dataset_key, base_cfg_path)

    model_lstm = ROOT / paths["models_dir"] / "best_lstm.pt"
    model_mlp = ROOT / paths["models_dir"] / "best_mlp.pt"

    # Baseline train/eval on clean real dataset.
    for model in models:
        model_path = model_lstm if model == "lstm" else model_mlp
        _maybe_run(
            [sys.executable, "experiments/core/run_train.py", "--config", _rel(base_cfg_path), "--model", model],
            [model_path],
            skip_existing=skip_existing,
        )
        _run([sys.executable, "experiments/core/run_evaluate.py", "--config", _rel(base_cfg_path), "--model_path", _rel(model_path)])

    method_records: list[dict[str, Any]] = []
    for method in METHODS:
        method_cfg = copy.deepcopy(base)
        method_cfg["defense"]["method"] = method
        tag = f"{ds_tag}/seed_{seed}/{method}"
        method_cfg["paths"]["defended_dir"] = f"data/defended/real_public_benchmark/{tag}"
        method_cfg["paths"]["defense_dir"] = f"outputs/defense/real_public_benchmark/{tag}"
        method_cfg_path = _save_cfg(method_cfg, f"{ds_tag}.seed_{seed}.{method}")

        defended_seq = ROOT / method_cfg["paths"]["defended_dir"] / "defended_sequences.npz"
        _maybe_run(
            [sys.executable, "experiments/core/run_defense.py", "--config", _rel(method_cfg_path)],
            [defended_seq],
            skip_existing=skip_existing,
        )

        for model in models:
            model_path = model_lstm if model == "lstm" else model_mlp
            _run(
                [
                    sys.executable,
                    "experiments/core/run_defense_eval.py",
                    "--config",
                    _rel(method_cfg_path),
                    "--mode",
                    "fixed_attacker",
                    "--model_path",
                    _rel(model_path),
                ]
            )

            retrain_cfg = copy.deepcopy(method_cfg)
            retrain_cfg.setdefault("train", {})
            retrain_cfg["train"]["model_type"] = model
            retrain_cfg.setdefault("defense_eval", {})
            retrain_cfg["defense_eval"]["retrained_model_name"] = (
                f"best_{model}_{ds_tag}_{method}_defended_retrain.pt"
            )
            retrain_cfg_path = _save_cfg(retrain_cfg, f"{ds_tag}.seed_{seed}.{method}.{model}.retrain")
            retrained_model = (
                ROOT
                / method_cfg["paths"]["models_dir"]
                / f"best_{model}_{ds_tag}_{method}_defended_retrain.pt"
            )
            if skip_existing and retrained_model.exists():
                print(f"[SKIP] retrained model exists: {retrained_model}", flush=True)
            else:
                _run(
                    [
                        sys.executable,
                        "experiments/core/run_defense_eval.py",
                        "--config",
                        _rel(retrain_cfg_path),
                        "--mode",
                        "retrain_attacker",
                    ]
                )

            clean_npz = (
                ROOT / method_cfg["paths"]["processed_dir"] / "sequences.npz"
                if model == "lstm"
                else ROOT / method_cfg["paths"]["processed_dir"] / "mlp_features.npz"
            )
            defended_npz = (
                ROOT / method_cfg["paths"]["defended_dir"] / "defended_sequences.npz"
                if model == "lstm"
                else ROOT / method_cfg["paths"]["defended_dir"] / "defended_mlp_features.npz"
            )
            reports_dir = ROOT / method_cfg["paths"]["defense_dir"] / "json_reports"
            reports_dir.mkdir(parents=True, exist_ok=True)

            _maybe_run(
                [
                    sys.executable,
                    "experiments/core/collect_confusion.py",
                    "--model_path",
                    _rel(model_path),
                    "--npz_path",
                    _rel(clean_npz),
                    "--split",
                    "test",
                    "--model_type",
                    model,
                    "--out",
                    _rel(reports_dir / f"{model}_baseline_confusion_test.json"),
                ]
                ,
                [reports_dir / f"{model}_baseline_confusion_test.json"],
                skip_existing=skip_existing,
            )
            _maybe_run(
                [
                    sys.executable,
                    "experiments/core/collect_confusion.py",
                    "--model_path",
                    _rel(model_path),
                    "--npz_path",
                    _rel(defended_npz),
                    "--split",
                    "test",
                    "--model_type",
                    model,
                    "--out",
                    _rel(reports_dir / f"{model}_defended_confusion_test_fixed_attacker.json"),
                ]
                ,
                [reports_dir / f"{model}_defended_confusion_test_fixed_attacker.json"],
                skip_existing=skip_existing,
            )
            _maybe_run(
                [
                    sys.executable,
                    "experiments/core/collect_confusion.py",
                    "--model_path",
                    _rel(retrained_model),
                    "--npz_path",
                    _rel(defended_npz),
                    "--split",
                    "test",
                    "--model_type",
                    model,
                    "--out",
                    _rel(reports_dir / f"{model}_defended_confusion_test_retrained_attacker.json"),
                ]
                ,
                [reports_dir / f"{model}_defended_confusion_test_retrained_attacker.json"],
                skip_existing=skip_existing,
            )

        if method in ("ldp", "noise"):
            compare_csv = ROOT / method_cfg["paths"]["defense_dir"] / "comparisons" / "comparison_results.csv"
            _maybe_run(
                [
                    sys.executable,
                    "experiments/core/run_compare.py",
                    "--config",
                    _rel(method_cfg_path),
                    "--method",
                    method,
                    "--model_path",
                    _rel(model_lstm),
                ]
                ,
                [compare_csv],
                skip_existing=skip_existing,
            )

        method_records.append(
            {
                "method": method,
                "config": _rel(method_cfg_path),
                "defended_dir": method_cfg["paths"]["defended_dir"],
                "defense_dir": method_cfg["paths"]["defense_dir"],
            }
        )

    return {
        "dataset": dataset_key,
        "seed": seed,
        "base_config": _rel(base_cfg_path),
        "processed_dir": paths["processed_dir"],
        "models_dir": paths["models_dir"],
        "methods": method_records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full benchmark on real public datasets")
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS), help="Comma-separated dataset keys")
    parser.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS), help="Comma-separated seeds")
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS), help="Comma-separated models")
    parser.add_argument("--max-epochs", type=int, default=25)
    parser.add_argument("--skip-existing", action="store_true", help="Skip steps whose output files already exist")
    args = parser.parse_args()

    datasets = _parse_list(args.datasets)
    seeds = _parse_seeds(args.seeds)
    models = _parse_list(args.models)

    records: list[dict[str, Any]] = []
    for dataset in datasets:
        for seed in seeds:
            print(f"\n========== REAL DATASET {dataset} / SEED {seed} ==========", flush=True)
            records.append(
                run_one(
                    dataset,
                    seed,
                    models=models,
                    max_epochs=int(args.max_epochs),
                    skip_existing=bool(args.skip_existing),
                )
            )

    manifest_path = ROOT / "outputs" / "reports" / "real_public_benchmark" / "real_public_benchmark_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nAll done. Manifest written: {manifest_path.as_posix()}", flush=True)


if __name__ == "__main__":
    main()
