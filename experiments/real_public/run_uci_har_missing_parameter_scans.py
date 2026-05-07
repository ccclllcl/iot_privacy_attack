#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Complete UCI HAR parameter scans for final thesis coverage.

The original benchmark pipeline generated fixed-attacker LSTM scans only.
This entry point fills the remaining UCI HAR scan matrix:

- seeds: 42, 123, 2026
- methods: ldp, noise
- models: lstm, mlp
- modes: fixed_attacker, retrain_attacker

Outputs are stored beside the existing scan CSVs under
``outputs/defense/real_public_benchmark/uci_har/seed_*/{method}/comparisons/``.
Existing fixed-attacker LSTM scans are preserved and copied into the new
model/mode-specific naming scheme.
"""

from __future__ import annotations

import argparse
import copy
import csv
import logging
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import ExperimentConfig
from src.defense_eval import compute_fixed_attacker_metrics
from src.defenses.defense_pipeline import run_defense_pipeline
from src.evaluate import evaluate_on_arrays, load_model_from_checkpoint
from src.plotting import configure_matplotlib_english
from src.train import run_training
from src.utils import ensure_dir, get_torch_device


SEEDS = [42, 123, 2026]
METHODS = ["ldp", "noise"]
MODELS = ["lstm", "mlp"]
MODES = ["fixed_attacker", "retrain_attacker"]
GEN_DIR = ROOT / "configs" / "generated_real_public"


def _rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _parse_csv_list(raw: str, cast=str) -> list[Any]:
    return [cast(x.strip()) for x in raw.split(",") if x.strip()]


def _load_method_config(seed: int, method: str, max_epochs: int | None) -> ExperimentConfig:
    cfg_path = GEN_DIR / f"uci_har.seed_{seed}.{method}.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing generated config: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f)
    if max_epochs is not None:
        train = raw.setdefault("train", {})
        train["num_epochs"] = int(min(int(train.get("num_epochs", max_epochs)), int(max_epochs)))
        train["early_stopping_patience"] = int(min(int(train.get("early_stopping_patience", 6)), 6))
    return ExperimentConfig(raw, ROOT)


def _clone_cfg(cfg: ExperimentConfig) -> ExperimentConfig:
    return ExperimentConfig(copy.deepcopy(cfg.raw), cfg.project_root)


def _scan_values(cfg: ExperimentConfig, method: str) -> tuple[str, list[float]]:
    cmp = cfg.nested("compare")
    if method == "ldp":
        return "epsilon", [float(x) for x in cmp.get("ldp_epsilon_list", [0.5, 1.0, 2.0])]
    if method == "noise":
        return "noise_scale", [float(x) for x in cmp.get("noise_scale_list", [0.1, 0.3, 0.5])]
    raise ValueError(f"Unsupported method: {method}")


def _set_param(cfg: ExperimentConfig, method: str, value: float) -> None:
    cfg.raw.setdefault("defense", {})
    cfg.raw["defense"]["enabled"] = True
    cfg.raw["defense"]["method"] = method
    if method == "ldp":
        cfg.raw["defense"]["epsilon"] = float(value)
    else:
        cfg.raw["defense"]["noise_scale"] = float(value)


def _model_path(cfg: ExperimentConfig, model_type: str) -> Path:
    return cfg.path("paths", "models_dir") / f"best_{model_type}.pt"


def _evaluate_retrained(
    cfg: ExperimentConfig,
    *,
    model_type: str,
    method: str,
    seed: int,
    param_name: str,
    param_value: float,
    out_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    cfg.raw.setdefault("train", {})
    cfg.raw["train"]["model_type"] = model_type

    defended = cfg.path("paths", "defended_dir")
    model_dir = ensure_dir(
        cfg.path("paths", "models_dir")
        / "parameter_scans"
        / method
        / f"seed_{seed}"
        / model_type
        / param_name
    )
    param_slug = str(param_value).replace(".", "p")
    save_path = model_dir / f"best_{model_type}_{method}_{param_name}_{param_slug}_retrain.pt"
    curve_path = out_dir / f"{model_type}_retrain_{param_name}_{param_slug}_curve.png"
    history_path = out_dir / f"{model_type}_retrain_{param_name}_{param_slug}_history.json"

    if not save_path.exists():
        run_training(
            cfg,
            model_type=model_type,
            override_model_path=save_path,
            sequences_npz=defended / "defended_sequences.npz" if model_type == "lstm" else None,
            mlp_npz=defended / "defended_mlp_features.npz" if model_type == "mlp" else None,
            curve_output_path=curve_path,
            history_output_path=history_path,
        )

    ev = cfg.nested("evaluate")
    device = get_torch_device(str(ev.get("device") or "auto"))
    batch_size = int(ev.get("batch_size", 128))
    num_workers = int(ev.get("num_workers", 0))
    model, ckpt = load_model_from_checkpoint(save_path, device)
    class_names = list(ckpt["class_names"])

    processed = cfg.path("paths", "processed_dir")
    if model_type == "lstm":
        clean = np.load(processed / "sequences.npz")
        defended_npz = np.load(defended / "defended_sequences.npz")
        x_clean, y_clean = clean["X_test"], clean["y_test"]
        x_def, y_def = defended_npz["X_test"], defended_npz["y_test"]
    else:
        clean = np.load(processed / "mlp_features.npz")
        defended_npz = np.load(defended / "defended_mlp_features.npz")
        x_clean, y_clean = clean["X_test"], clean["y_test"]
        x_def, y_def = defended_npz["X_test"], defended_npz["y_test"]

    clean_metrics = evaluate_on_arrays(
        model, x_clean, y_clean, model_type, class_names, device, batch_size, num_workers
    )
    defended_metrics = evaluate_on_arrays(
        model, x_def, y_def, model_type, class_names, device, batch_size, num_workers
    )
    return clean_metrics, defended_metrics, save_path


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _plot_accuracy(rows: list[dict[str, Any]], out: Path, title: str) -> None:
    configure_matplotlib_english()
    if not rows:
        return
    xs = [float(r["param_value"]) for r in rows]
    ys = [float(r["defended_accuracy"]) for r in rows]
    baseline = float(rows[0]["baseline_accuracy"])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(xs, ys, marker="o", label="Defended accuracy")
    ax.axhline(baseline, color="gray", linestyle="--", label="Clean reference")
    if rows[0]["method"] == "ldp":
        ax.set_xlabel("epsilon")
    else:
        ax.set_xlabel("noise_scale")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def run_scan(
    *,
    seed: int,
    method: str,
    model_type: str,
    mode: str,
    skip_existing: bool,
    max_epochs: int | None,
) -> Path:
    cfg = _load_method_config(seed, method, max_epochs=max_epochs)
    comp_dir = ensure_dir(cfg.path("paths", "defense_dir") / "comparisons")
    out_csv = comp_dir / f"{model_type}_{mode}_comparison_results.csv"
    if skip_existing and out_csv.exists():
        return out_csv

    param_name, values = _scan_values(cfg, method)
    model_path = _model_path(cfg, model_type)
    if mode == "fixed_attacker" and not model_path.exists():
        raise FileNotFoundError(f"Missing clean attacker model: {model_path}")

    rows: list[dict[str, Any]] = []
    for value in values:
        run_cfg = _clone_cfg(cfg)
        _set_param(run_cfg, method, value)
        summary = run_defense_pipeline(run_cfg)
        distortion = summary.get("distortion", {})

        if mode == "fixed_attacker":
            pair = compute_fixed_attacker_metrics(run_cfg, model_path)
            baseline_metrics = pair["baseline"]
            defended_metrics = pair["defended"]
            model_source = model_path
        else:
            baseline_metrics, defended_metrics, model_source = _evaluate_retrained(
                run_cfg,
                model_type=model_type,
                method=method,
                seed=seed,
                param_name=param_name,
                param_value=float(value),
                out_dir=comp_dir,
            )

        baseline_acc = float(baseline_metrics["accuracy"])
        defended_acc = float(defended_metrics["accuracy"])
        rows.append(
            {
                "dataset": "uci_har",
                "seed": int(seed),
                "model_type": model_type,
                "mode": mode,
                "method": method,
                "param_name": param_name,
                "param_value": float(value),
                "baseline_accuracy": baseline_acc,
                "defended_accuracy": defended_acc,
                "accuracy_drop": baseline_acc - defended_acc,
                "defended_f1_macro": float(defended_metrics["f1_macro"]),
                "mse": float(distortion.get("mse", 0.0)),
                "mae": float(distortion.get("mae", 0.0)),
                "pearson_r": float(distortion.get("pearson_r", 0.0)),
                "model_source": _rel(model_source),
            }
        )

    _write_csv(out_csv, rows)
    _plot_accuracy(
        rows,
        comp_dir / f"{model_type}_{mode}_{method}_accuracy.png",
        f"UCI HAR {method}: {model_type} {mode}",
    )
    return out_csv


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    parser = argparse.ArgumentParser(description="Complete missing UCI HAR parameter scans")
    parser.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--modes", default=",".join(MODES))
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--max-epochs", type=int, default=25)
    args = parser.parse_args()

    seeds = _parse_csv_list(args.seeds, int)
    methods = _parse_csv_list(args.methods, str)
    models = _parse_csv_list(args.models, str)
    modes = _parse_csv_list(args.modes, str)

    outputs: list[str] = []
    for seed in seeds:
        for method in methods:
            for model_type in models:
                for mode in modes:
                    path = run_scan(
                        seed=int(seed),
                        method=str(method),
                        model_type=str(model_type),
                        mode=str(mode),
                        skip_existing=bool(args.skip_existing),
                        max_epochs=int(args.max_epochs),
                    )
                    outputs.append(_rel(path))
                    logging.info("scan ready: %s", path)

    manifest = ROOT / "outputs" / "reports" / "real_public_benchmark" / "uci_har_parameter_scan_completion.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        yaml.safe_dump({"outputs": outputs}, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
