#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run all implemented defense methods across multiple seeds."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parent
BASE_CONFIG = ROOT / "configs" / "default.yaml"
GEN_DIR = ROOT / "configs" / "generated_all_methods"
SEEDS = [42, 123, 2026]
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


def _base_seed_cfg(seed: int) -> dict[str, Any]:
    with BASE_CONFIG.open("r", encoding="utf-8") as f:
        cfg: dict[str, Any] = yaml.safe_load(f)

    tag = f"seed_{seed}"
    cfg.setdefault("experiment", {})
    cfg["experiment"]["random_seed"] = int(seed)

    p = cfg.setdefault("paths", {})
    p["raw_csv"] = f"data/raw/full_multiseed/{tag}/smart_home_events.csv"
    p["processed_dir"] = f"data/processed/full_multiseed/{tag}"
    p["models_dir"] = f"outputs/models/full_multiseed/{tag}"
    p["figures_dir"] = f"outputs/figures/full_multiseed/{tag}"
    p["reports_dir"] = f"outputs/reports/full_multiseed/{tag}"
    p["defended_dir"] = f"data/defended/full_multiseed/{tag}/adaptive_ldp"
    p["defense_dir"] = f"outputs/defense/full_multiseed/{tag}/adaptive_ldp"
    return cfg


def run_seed(seed: int) -> dict[str, Any]:
    tag = f"seed_{seed}"
    seed_cfg = _base_seed_cfg(seed)
    seed_cfg_path = _save_cfg(seed_cfg, f"default.{tag}.base")

    paths = seed_cfg["paths"]
    raw_csv = ROOT / paths["raw_csv"]
    model_lstm = ROOT / paths["models_dir"] / "best_lstm.pt"
    model_mlp = ROOT / paths["models_dir"] / "best_mlp.pt"

    _run(
        [
            sys.executable,
            "generate_mock_data.py",
            "--config",
            _rel(seed_cfg_path),
            "--output",
            _rel(raw_csv),
            "--seed",
            str(seed),
        ]
    )
    _run([sys.executable, "run_preprocess.py", "--config", _rel(seed_cfg_path)])

    # Baseline models: lstm + mlp
    for model in MODELS:
        _run([sys.executable, "run_train.py", "--config", _rel(seed_cfg_path), "--model", model])
        model_path = model_lstm if model == "lstm" else model_mlp
        _run(
            [
                sys.executable,
                "run_evaluate.py",
                "--config",
                _rel(seed_cfg_path),
                "--model_path",
                _rel(model_path),
            ]
        )

    method_outputs: list[dict[str, str]] = []
    for method in METHODS:
        method_cfg = copy.deepcopy(seed_cfg)
        method_cfg["defense"]["method"] = method
        method_cfg["paths"]["defended_dir"] = f"data/defended/full_multiseed/{tag}/{method}"
        method_cfg["paths"]["defense_dir"] = f"outputs/defense/full_multiseed/{tag}/{method}"
        method_cfg_path = _save_cfg(method_cfg, f"default.{tag}.{method}")

        # Explicit defense pipeline
        _run([sys.executable, "run_defense.py", "--config", _rel(method_cfg_path)])

        # Fixed attacker + retrain attacker for both models
        for model in MODELS:
            model_path = model_lstm if model == "lstm" else model_mlp
            _run(
                [
                    sys.executable,
                    "run_defense_eval.py",
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
                f"best_{model}_{method}_defended_retrain.pt"
            )
            retrain_cfg_path = _save_cfg(retrain_cfg, f"default.{tag}.{method}.{model}.retrain")
            _run(
                [
                    sys.executable,
                    "run_defense_eval.py",
                    "--config",
                    _rel(retrain_cfg_path),
                    "--mode",
                    "retrain_attacker",
                ]
            )

            # confusion exports
            defended_npz = (
                ROOT / method_cfg["paths"]["defended_dir"] / "defended_sequences.npz"
                if model == "lstm"
                else ROOT / method_cfg["paths"]["defended_dir"] / "defended_mlp_features.npz"
            )
            clean_npz = (
                ROOT / method_cfg["paths"]["processed_dir"] / "sequences.npz"
                if model == "lstm"
                else ROOT / method_cfg["paths"]["processed_dir"] / "mlp_features.npz"
            )
            reports_dir = ROOT / method_cfg["paths"]["defense_dir"] / "json_reports"
            reports_dir.mkdir(parents=True, exist_ok=True)
            retrained_model = (
                ROOT
                / method_cfg["paths"]["models_dir"]
                / f"best_{model}_{method}_defended_retrain.pt"
            )

            _run(
                [
                    sys.executable,
                    "collect_confusion.py",
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
            )
            _run(
                [
                    sys.executable,
                    "collect_confusion.py",
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
            )
            _run(
                [
                    sys.executable,
                    "collect_confusion.py",
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
            )

        # parameter compare is implemented for ldp/noise
        if method in ("ldp", "noise"):
            _run(
                [
                    sys.executable,
                    "run_compare.py",
                    "--config",
                    _rel(method_cfg_path),
                    "--method",
                    method,
                    "--model_path",
                    _rel(model_lstm),
                ]
            )

        method_outputs.append(
            {
                "method": method,
                "config": _rel(method_cfg_path),
                "defended_dir": method_cfg["paths"]["defended_dir"],
                "defense_dir": method_cfg["paths"]["defense_dir"],
            }
        )

    return {
        "seed": seed,
        "base_config": _rel(seed_cfg_path),
        "raw_csv": paths["raw_csv"],
        "processed_dir": paths["processed_dir"],
        "models_dir": paths["models_dir"],
        "methods": method_outputs,
    }


def main() -> None:
    results: list[dict[str, Any]] = []
    for seed in SEEDS:
        print(f"\n========== FULL RUN Seed {seed} ==========", flush=True)
        results.append(run_seed(seed))

    out = ROOT / "outputs" / "reports" / "full_methods_multiseed_manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nAll done. Manifest written: {out.as_posix()}", flush=True)


if __name__ == "__main__":
    main()
