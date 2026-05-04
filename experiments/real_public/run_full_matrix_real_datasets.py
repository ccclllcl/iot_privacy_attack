#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run full defense matrix on real datasets (UCI HAR, Kasteren)."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
BASE_CONFIG = ROOT / "configs" / "default.yaml"
GEN_DIR = ROOT / "configs" / "generated_dataset_matrix"

SEEDS = [42, 123]
DATASETS = ["uci_har", "kasteren"]
METHODS = ["adaptive_ldp", "ldp", "noise"]
MODELS = ["lstm", "mlp"]


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


def _base_cfg(dataset: str, seed: int) -> dict[str, Any]:
    with BASE_CONFIG.open("r", encoding="utf-8") as f:
        cfg: dict[str, Any] = yaml.safe_load(f)

    cfg.setdefault("experiment", {})
    cfg["experiment"]["random_seed"] = int(seed)
    tag = f"{dataset}/seed_{seed}"

    p = cfg.setdefault("paths", {})
    p["processed_dir"] = f"data/processed/dataset_matrix/{tag}"
    p["models_dir"] = f"outputs/models/dataset_matrix/{tag}"
    p["figures_dir"] = f"outputs/figures/dataset_matrix/{tag}"
    p["reports_dir"] = f"outputs/reports/dataset_matrix/{tag}"
    p["defended_dir"] = f"data/defended/dataset_matrix/{tag}/adaptive_ldp"
    p["defense_dir"] = f"outputs/defense/dataset_matrix/{tag}/adaptive_ldp"

    # Coverage-oriented run: keep full combination matrix, reduce runtime.
    train = cfg.setdefault("train", {})
    train["num_epochs"] = int(min(int(train.get("num_epochs", 80)), 25))
    train["early_stopping_patience"] = int(min(int(train.get("early_stopping_patience", 10)), 5))

    return cfg


def _run_import(dataset: str, cfg_path: Path) -> None:
    if dataset == "uci_har":
        _run([sys.executable, "experiments/real_public/run_import_uci_har.py", "--config", _rel(cfg_path), "--auto-download"])
    elif dataset == "kasteren":
        _run([sys.executable, "experiments/real_public/run_import_kasteren.py", "--config", _rel(cfg_path), "--auto-download"])
    else:
        raise ValueError(f"unsupported dataset: {dataset}")


def run_one(dataset: str, seed: int) -> dict[str, Any]:
    base = _base_cfg(dataset, seed)
    base_cfg = _save_cfg(base, f"{dataset}.seed_{seed}.base")
    _run_import(dataset, base_cfg)

    paths = base["paths"]
    model_lstm = ROOT / paths["models_dir"] / "best_lstm.pt"
    model_mlp = ROOT / paths["models_dir"] / "best_mlp.pt"

    # Baseline
    for model in MODELS:
        _run([sys.executable, "experiments/core/run_train.py", "--config", _rel(base_cfg), "--model", model])
        mp = model_lstm if model == "lstm" else model_mlp
        _run([sys.executable, "experiments/core/run_evaluate.py", "--config", _rel(base_cfg), "--model_path", _rel(mp)])

    method_records: list[dict[str, str]] = []
    for method in METHODS:
        cfg = copy.deepcopy(base)
        cfg["defense"]["method"] = method
        tag = f"{dataset}/seed_{seed}/{method}"
        cfg["paths"]["defended_dir"] = f"data/defended/dataset_matrix/{tag}"
        cfg["paths"]["defense_dir"] = f"outputs/defense/dataset_matrix/{tag}"
        cfg_path = _save_cfg(cfg, f"{dataset}.seed_{seed}.{method}")

        _run([sys.executable, "experiments/core/run_defense.py", "--config", _rel(cfg_path)])

        for model in MODELS:
            model_path = model_lstm if model == "lstm" else model_mlp
            _run(
                [
                    sys.executable,
                    "experiments/core/run_defense_eval.py",
                    "--config",
                    _rel(cfg_path),
                    "--mode",
                    "fixed_attacker",
                    "--model_path",
                    _rel(model_path),
                ]
            )

            retrain_cfg = copy.deepcopy(cfg)
            retrain_cfg.setdefault("train", {})
            retrain_cfg["train"]["model_type"] = model
            retrain_cfg.setdefault("defense_eval", {})
            retrain_cfg["defense_eval"]["retrained_model_name"] = (
                f"best_{model}_{dataset}_{method}_defended_retrain.pt"
            )
            retrain_cfg_path = _save_cfg(retrain_cfg, f"{dataset}.seed_{seed}.{method}.{model}.retrain")
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
                ROOT / cfg["paths"]["processed_dir"] / "sequences.npz"
                if model == "lstm"
                else ROOT / cfg["paths"]["processed_dir"] / "mlp_features.npz"
            )
            defended_npz = (
                ROOT / cfg["paths"]["defended_dir"] / "defended_sequences.npz"
                if model == "lstm"
                else ROOT / cfg["paths"]["defended_dir"] / "defended_mlp_features.npz"
            )
            retrained_model = (
                ROOT
                / cfg["paths"]["models_dir"]
                / f"best_{model}_{dataset}_{method}_defended_retrain.pt"
            )
            out_dir = ROOT / cfg["paths"]["defense_dir"] / "json_reports"
            out_dir.mkdir(parents=True, exist_ok=True)

            _run(
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
                    _rel(out_dir / f"{model}_baseline_confusion_test.json"),
                ]
            )
            _run(
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
                    _rel(out_dir / f"{model}_defended_confusion_test_fixed_attacker.json"),
                ]
            )
            _run(
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
                    _rel(out_dir / f"{model}_defended_confusion_test_retrained_attacker.json"),
                ]
            )

        if method in ("ldp", "noise"):
            _run(
                [
                    sys.executable,
                    "experiments/core/run_compare.py",
                    "--config",
                    _rel(cfg_path),
                    "--method",
                    method,
                    "--model_path",
                    _rel(model_lstm),
                ]
            )

        method_records.append(
            {
                "method": method,
                "config": _rel(cfg_path),
                "defended_dir": cfg["paths"]["defended_dir"],
                "defense_dir": cfg["paths"]["defense_dir"],
            }
        )

    return {
        "dataset": dataset,
        "seed": seed,
        "base_config": _rel(base_cfg),
        "processed_dir": paths["processed_dir"],
        "models_dir": paths["models_dir"],
        "methods": method_records,
    }


def main() -> None:
    records: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for seed in SEEDS:
            print(f"\n========== DATASET {dataset} / SEED {seed} ==========", flush=True)
            records.append(run_one(dataset, seed))

    out = ROOT / "outputs" / "reports" / "dataset_matrix_manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nAll done. Manifest written: {out.as_posix()}", flush=True)


if __name__ == "__main__":
    main()
