#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch-run all data-producing experiments for multiple random seeds."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
BASE_CONFIG = ROOT / "configs" / "default.yaml"
GENERATED_CONFIG_DIR = ROOT / "configs" / "generated"
SEEDS = [42, 123, 2026]


def _to_rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def make_seed_config(seed: int) -> Path:
    with BASE_CONFIG.open("r", encoding="utf-8") as f:
        cfg: dict[str, Any] = yaml.safe_load(f)

    seed_tag = f"seed_{seed}"
    cfg.setdefault("experiment", {})
    cfg["experiment"]["random_seed"] = int(seed)

    paths = cfg.setdefault("paths", {})
    paths["raw_csv"] = f"data/raw/multiseed/{seed_tag}/smart_home_events.csv"
    paths["processed_dir"] = f"data/processed/multiseed/{seed_tag}"
    paths["defended_dir"] = f"data/defended/multiseed/{seed_tag}"
    paths["models_dir"] = f"outputs/models/multiseed/{seed_tag}"
    paths["figures_dir"] = f"outputs/figures/multiseed/{seed_tag}"
    paths["reports_dir"] = f"outputs/reports/multiseed/{seed_tag}"
    paths["defense_dir"] = f"outputs/defense/multiseed/{seed_tag}"

    GENERATED_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    out = GENERATED_CONFIG_DIR / f"default.{seed_tag}.yaml"
    with out.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    return out


def run(cmd: list[str]) -> None:
    print(f"\n[RUN] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def run_for_seed(seed: int) -> dict[str, str]:
    cfg_path = make_seed_config(seed)
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    p = cfg["paths"]

    raw_csv = ROOT / p["raw_csv"]
    model_path = ROOT / p["models_dir"] / "best_lstm.pt"
    retrain_model_path = ROOT / p["models_dir"] / "best_lstm_defended_retrain.pt"
    seq_npz = ROOT / p["processed_dir"] / "sequences.npz"
    defended_npz = ROOT / p["defended_dir"] / "defended_sequences.npz"
    defense_reports_dir = ROOT / p["defense_dir"] / "json_reports"

    run(
        [
            sys.executable,
            "experiments/core/generate_mock_data.py",
            "--config",
            _to_rel(cfg_path),
            "--output",
            _to_rel(raw_csv),
            "--seed",
            str(seed),
        ]
    )
    run([sys.executable, "experiments/core/run_preprocess.py", "--config", _to_rel(cfg_path)])
    run([sys.executable, "experiments/core/run_train.py", "--config", _to_rel(cfg_path), "--model", "lstm"])
    run(
        [
            sys.executable,
            "experiments/core/run_evaluate.py",
            "--config",
            _to_rel(cfg_path),
            "--model_path",
            _to_rel(model_path),
        ]
    )
    run([sys.executable, "experiments/core/run_defense.py", "--config", _to_rel(cfg_path)])
    run(
        [
            sys.executable,
            "experiments/core/run_defense_eval.py",
            "--config",
            _to_rel(cfg_path),
            "--mode",
            "fixed_attacker",
            "--model_path",
            _to_rel(model_path),
        ]
    )
    run([sys.executable, "experiments/core/run_defense_eval.py", "--config", _to_rel(cfg_path), "--mode", "retrain_attacker"])
    run(
        [
            sys.executable,
            "experiments/core/run_compare.py",
            "--config",
            _to_rel(cfg_path),
            "--method",
            "ldp",
            "--model_path",
            _to_rel(model_path),
        ]
    )
    run(
        [
            sys.executable,
            "experiments/core/run_compare.py",
            "--config",
            _to_rel(cfg_path),
            "--method",
            "noise",
            "--model_path",
            _to_rel(model_path),
        ]
    )

    defense_reports_dir.mkdir(parents=True, exist_ok=True)
    run(
        [
            sys.executable,
            "experiments/core/collect_confusion.py",
            "--model_path",
            _to_rel(model_path),
            "--npz_path",
            _to_rel(seq_npz),
            "--split",
            "test",
            "--model_type",
            "lstm",
            "--out",
            _to_rel(defense_reports_dir / "baseline_confusion_test.json"),
        ]
    )
    run(
        [
            sys.executable,
            "experiments/core/collect_confusion.py",
            "--model_path",
            _to_rel(model_path),
            "--npz_path",
            _to_rel(defended_npz),
            "--split",
            "test",
            "--model_type",
            "lstm",
            "--out",
            _to_rel(defense_reports_dir / "defended_confusion_test_fixed_attacker.json"),
        ]
    )
    run(
        [
            sys.executable,
            "experiments/core/collect_confusion.py",
            "--model_path",
            _to_rel(retrain_model_path),
            "--npz_path",
            _to_rel(defended_npz),
            "--split",
            "test",
            "--model_type",
            "lstm",
            "--out",
            _to_rel(defense_reports_dir / "defended_confusion_test_retrained_attacker.json"),
        ]
    )

    return {
        "seed": str(seed),
        "config": _to_rel(cfg_path),
        "raw_csv": _to_rel(raw_csv),
        "processed_dir": p["processed_dir"],
        "defended_dir": p["defended_dir"],
        "models_dir": p["models_dir"],
        "defense_dir": p["defense_dir"],
    }


def main() -> None:
    all_results: list[dict[str, str]] = []
    for seed in SEEDS:
        print(f"\n========== Seed {seed} ==========", flush=True)
        one = run_for_seed(seed)
        all_results.append(one)

    out = ROOT / "outputs" / "reports" / "multiseed_run_manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nAll done. Manifest written to: {out.as_posix()}", flush=True)


if __name__ == "__main__":
    main()
