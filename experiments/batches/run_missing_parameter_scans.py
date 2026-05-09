#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fill only missing final-thesis parameter scans reported by the symmetry audit."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
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
from src.experiment_compare import ADAPTIVE_LDP_PROFILES
from src.plotting import configure_matplotlib_english
from src.train import run_training
from src.utils import ensure_dir, get_torch_device


SEEDS = [42, 123, 2026]
MODELS = ["lstm", "mlp"]
METHODS = ["adaptive_ldp", "ldp", "noise"]
MODES = ["fixed_attacker", "retrain_attacker"]
DATASETS = ["uci_har", "kasteren", "casas_hh101"]
DATASET_PRIORITY = {"mock": 0, "uci_har": 1, "kasteren": 2, "casas_hh101": 3}
PARAM_ROWS = {"ldp": 5, "noise": 4, "adaptive_ldp": 6}

OUT_REPORT = ROOT / "outputs" / "reports" / "final_thesis"
AUDIT_PATH = OUT_REPORT / "final_symmetry_audit.json"
MISSING_PATH = OUT_REPORT / "parameter_scan_missing_outputs.json"
RUN_LOG_PATH = OUT_REPORT / "parameter_scan_run_log.json"

SCAN_FIELDNAMES = [
    "dataset",
    "seed",
    "model_type",
    "mode",
    "method",
    "profile_name",
    "param_name",
    "param_value",
    "epsilon_min",
    "epsilon_max",
    "weight_sensitivity",
    "weight_traffic",
    "use_edge_budget_cap",
    "edge_inverse_budget_cap",
    "baseline_accuracy",
    "defended_accuracy",
    "accuracy_drop",
    "defended_f1_macro",
    "mse",
    "mae",
    "pearson_r",
    "model_source",
    "source_file",
]


@dataclass(frozen=True)
class ScanTarget:
    scope: str
    dataset: str
    seed: int
    method: str
    model_type: str
    mode: str


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except Exception:
        return path.as_posix()


def _load_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_csv_rows(path: Path) -> list[dict[str, str]] | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))
    except Exception:
        return None


def _csv_is_complete(path: Path, method: str) -> bool:
    rows = _read_csv_rows(path)
    return rows is not None and len(rows) == PARAM_ROWS[method]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SCAN_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in SCAN_FIELDNAMES})


def _target_path(target: ScanTarget) -> Path:
    if target.scope == "mock":
        return (
            ROOT
            / "outputs"
            / "defense"
            / "full_multiseed"
            / f"seed_{target.seed}"
            / target.method
            / "comparisons"
            / f"{target.model_type}_{target.mode}_comparison_results.csv"
        )
    return (
        ROOT
        / "outputs"
        / "defense"
        / "real_public_benchmark"
        / target.dataset
        / f"seed_{target.seed}"
        / target.method
        / "comparisons"
        / f"{target.model_type}_{target.mode}_comparison_results.csv"
    )


def _legacy_path(target: ScanTarget) -> Path:
    if target.scope == "mock":
        return (
            ROOT
            / "outputs"
            / "defense"
            / "full_multiseed"
            / f"seed_{target.seed}"
            / target.method
            / "comparisons"
            / "comparison_results.csv"
        )
    return (
        ROOT
        / "outputs"
        / "defense"
        / "real_public_benchmark"
        / target.dataset
        / f"seed_{target.seed}"
        / target.method
        / "comparisons"
        / "comparison_results.csv"
    )


def _config_path(target: ScanTarget) -> Path:
    if target.scope == "mock":
        return ROOT / "configs" / "generated_all_methods" / f"default.seed_{target.seed}.{target.method}.yaml"
    return (
        ROOT
        / "configs"
        / "generated_real_public"
        / f"{target.dataset}.seed_{target.seed}.{target.method}.yaml"
    )


def _load_cfg(target: ScanTarget, max_epochs: int | None) -> ExperimentConfig:
    path = _config_path(target)
    if not path.exists():
        raise FileNotFoundError(f"Missing config: {_rel(path)}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid config: {_rel(path)}")
    if max_epochs is not None:
        train = raw.setdefault("train", {})
        train["num_epochs"] = int(min(int(train.get("num_epochs", max_epochs)), int(max_epochs)))
        train["early_stopping_patience"] = int(min(int(train.get("early_stopping_patience", 6)), 6))
    return ExperimentConfig(raw, ROOT)


def _clone_cfg(cfg: ExperimentConfig) -> ExperimentConfig:
    return ExperimentConfig(copy.deepcopy(cfg.raw), cfg.project_root)


def _clean_model_path(cfg: ExperimentConfig, model_type: str) -> Path:
    return cfg.path("paths", "models_dir") / f"best_{model_type}.pt"


def _param_specs(method: str) -> list[dict[str, Any]]:
    if method == "ldp":
        return [{"param_name": "epsilon", "param_value": v, "epsilon": v} for v in [0.1, 0.5, 1.0, 2.0, 5.0]]
    if method == "noise":
        return [{"param_name": "noise_scale", "param_value": v, "noise_scale": v} for v in [0.1, 0.3, 0.5, 1.0]]
    if method == "adaptive_ldp":
        specs: list[dict[str, Any]] = []
        for idx, profile in enumerate(ADAPTIVE_LDP_PROFILES, start=1):
            specs.append({"param_name": "profile", "param_value": idx, **profile})
        return specs
    raise ValueError(f"Unsupported method: {method}")


def _apply_param(cfg: ExperimentConfig, method: str, spec: dict[str, Any]) -> None:
    cfg.raw.setdefault("defense", {})
    cfg.raw["defense"]["enabled"] = True
    cfg.raw["defense"]["method"] = method
    if method == "ldp":
        cfg.raw["defense"]["epsilon"] = float(spec["epsilon"])
    elif method == "noise":
        cfg.raw["defense"]["noise_scale"] = float(spec["noise_scale"])
    else:
        cfg.raw.setdefault("adaptive_ldp", {})
        cfg.raw["adaptive_ldp"].update(
            {
                "epsilon_min": float(spec["epsilon_min"]),
                "epsilon_max": float(spec["epsilon_max"]),
                "weight_sensitivity": float(spec["weight_sensitivity"]),
                "weight_traffic": float(spec["weight_traffic"]),
                "use_edge_budget_cap": bool(spec["use_edge_budget_cap"]),
            }
        )
        if "edge_inverse_budget_cap" in spec:
            cfg.raw["adaptive_ldp"]["edge_inverse_budget_cap"] = int(spec["edge_inverse_budget_cap"])


def _slug(spec: dict[str, Any]) -> str:
    if spec.get("profile_name"):
        return str(spec["profile_name"])
    value = str(spec["param_value"]).replace(".", "p")
    return f"{spec['param_name']}_{value}"


def _evaluate_retrained(
    cfg: ExperimentConfig,
    *,
    target: ScanTarget,
    spec: dict[str, Any],
    out_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    cfg.raw.setdefault("train", {})
    cfg.raw["train"]["model_type"] = target.model_type

    defended = cfg.path("paths", "defended_dir")
    model_dir = ensure_dir(
        cfg.path("paths", "models_dir")
        / "parameter_scans"
        / target.method
        / target.model_type
        / target.mode
    )
    param_slug = _slug(spec)
    save_path = model_dir / f"best_{target.model_type}_{target.method}_{param_slug}_retrain.pt"
    curve_path = out_dir / f"{target.model_type}_{target.mode}_{target.method}_{param_slug}_curve.png"
    history_path = out_dir / f"{target.model_type}_{target.mode}_{target.method}_{param_slug}_history.json"

    if not save_path.exists():
        run_training(
            cfg,
            model_type=target.model_type,
            override_model_path=save_path,
            sequences_npz=defended / "defended_sequences.npz" if target.model_type == "lstm" else None,
            mlp_npz=defended / "defended_mlp_features.npz" if target.model_type == "mlp" else None,
            curve_output_path=curve_path,
            history_output_path=history_path,
        )

    ev = cfg.nested("evaluate")
    device = get_torch_device(str(ev.get("device") or "auto"))
    batch_size = int(ev.get("batch_size", 128))
    num_workers = int(ev.get("num_workers", 0))
    try:
        model, ckpt = load_model_from_checkpoint(save_path, device)
    except Exception:
        run_training(
            cfg,
            model_type=target.model_type,
            override_model_path=save_path,
            sequences_npz=defended / "defended_sequences.npz" if target.model_type == "lstm" else None,
            mlp_npz=defended / "defended_mlp_features.npz" if target.model_type == "mlp" else None,
            curve_output_path=curve_path,
            history_output_path=history_path,
        )
        model, ckpt = load_model_from_checkpoint(save_path, device)
    class_names = list(ckpt["class_names"])

    processed = cfg.path("paths", "processed_dir")
    if target.model_type == "lstm":
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
        model, x_clean, y_clean, target.model_type, class_names, device, batch_size, num_workers
    )
    defended_metrics = evaluate_on_arrays(
        model, x_def, y_def, target.model_type, class_names, device, batch_size, num_workers
    )
    return clean_metrics, defended_metrics, save_path


def _row(
    *,
    target: ScanTarget,
    spec: dict[str, Any],
    distortion: dict[str, Any],
    baseline_metrics: dict[str, Any],
    defended_metrics: dict[str, Any],
    model_source: Path,
) -> dict[str, Any]:
    baseline_acc = float(baseline_metrics["accuracy"])
    defended_acc = float(defended_metrics["accuracy"])
    out_file = _target_path(target)
    return {
        "dataset": target.dataset,
        "seed": int(target.seed),
        "model_type": target.model_type,
        "mode": target.mode,
        "method": target.method,
        "profile_name": spec.get("profile_name", ""),
        "param_name": spec["param_name"],
        "param_value": spec["param_value"],
        "epsilon_min": spec.get("epsilon_min", ""),
        "epsilon_max": spec.get("epsilon_max", ""),
        "weight_sensitivity": spec.get("weight_sensitivity", ""),
        "weight_traffic": spec.get("weight_traffic", ""),
        "use_edge_budget_cap": spec.get("use_edge_budget_cap", ""),
        "edge_inverse_budget_cap": spec.get("edge_inverse_budget_cap", ""),
        "baseline_accuracy": baseline_acc,
        "defended_accuracy": defended_acc,
        "accuracy_drop": baseline_acc - defended_acc,
        "defended_f1_macro": float(defended_metrics["f1_macro"]),
        "mse": float(distortion.get("mse", 0.0)),
        "mae": float(distortion.get("mae", 0.0)),
        "pearson_r": float(distortion.get("pearson_r", 0.0)),
        "model_source": _rel(model_source),
        "source_file": _rel(out_file),
    }


def _plot_rows(rows: list[dict[str, Any]], out_dir: Path, target: ScanTarget) -> None:
    if not rows:
        return
    configure_matplotlib_english()
    xs = [float(r["param_value"]) for r in rows]
    ys = [float(r["defended_accuracy"]) for r in rows]
    mse = [float(r["mse"]) for r in rows]

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(xs, ys, marker="o")
    if target.method == "ldp":
        ax.set_xscale("log")
        ax.set_xlabel("epsilon")
    elif target.method == "noise":
        ax.set_xlabel("noise_scale")
    else:
        ax.set_xlabel("adaptive profile")
        ax.set_xticks(xs)
        ax.set_xticklabels([str(r["profile_name"]).replace("adaptive_", "") for r in rows], rotation=25, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{target.dataset} {target.method}: {target.model_type} {target.mode}")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{target.model_type}_{target.mode}_{target.method}_accuracy.png", dpi=150)
    if target.method == "adaptive_ldp" and target.model_type == "lstm" and target.mode == "fixed_attacker":
        fig.savefig(out_dir / "adaptive_profile_vs_accuracy.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(xs, mse, marker="s", color="#C44E52")
    if target.method == "ldp":
        ax.set_xscale("log")
        ax.set_xlabel("epsilon")
    elif target.method == "noise":
        ax.set_xlabel("noise_scale")
    else:
        ax.set_xlabel("adaptive profile")
        ax.set_xticks(xs)
        ax.set_xticklabels([str(r["profile_name"]).replace("adaptive_", "") for r in rows], rotation=25, ha="right")
    ax.set_ylabel("MSE")
    ax.set_title(f"{target.dataset} {target.method}: distortion")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"{target.model_type}_{target.mode}_{target.method}_distortion.png", dpi=150)
    if target.method == "adaptive_ldp" and target.model_type == "lstm" and target.mode == "fixed_attacker":
        fig.savefig(out_dir / "adaptive_profile_vs_distortion.png", dpi=150)
    plt.close(fig)


def _normalize_legacy(target: ScanTarget, log: list[dict[str, Any]]) -> bool:
    if target.method not in {"ldp", "noise"} or target.model_type != "lstm" or target.mode != "fixed_attacker":
        return False
    legacy = _legacy_path(target)
    if not _csv_is_complete(legacy, target.method):
        return False
    rows = _read_csv_rows(legacy) or []
    out_rows: list[dict[str, Any]] = []
    norm = _target_path(target)
    for r in rows:
        baseline = float(r.get("baseline_accuracy", 0.0))
        defended = float(r.get("defended_accuracy", 0.0))
        out_rows.append(
            {
                "dataset": target.dataset,
                "seed": target.seed,
                "model_type": target.model_type,
                "mode": target.mode,
                "method": target.method,
                "param_name": r.get("param_name", "epsilon" if target.method == "ldp" else "noise_scale"),
                "param_value": r.get("param_value", ""),
                "baseline_accuracy": baseline,
                "defended_accuracy": defended,
                "accuracy_drop": r.get("accuracy_drop", baseline - defended),
                "defended_f1_macro": r.get("defended_f1_macro", ""),
                "mse": r.get("mse", ""),
                "mae": r.get("mae", ""),
                "pearson_r": r.get("pearson_r", ""),
                "model_source": r.get("model_source", ""),
                "source_file": _rel(norm),
            }
        )
    _write_csv(norm, out_rows)
    log.append({"target": _rel(norm), "action": "copied_legacy", "source": _rel(legacy)})
    return True


def _audit_targets() -> list[ScanTarget]:
    if not AUDIT_PATH.exists():
        subprocess.run([sys.executable, "scripts/audit_experiment_symmetry.py"], cwd=ROOT, check=False)
    audit = _load_json(AUDIT_PATH)
    if not isinstance(audit, dict):
        raise RuntimeError("Could not load final_symmetry_audit.json")
    targets: list[ScanTarget] = []
    for item in audit.get("missing_mock_parameter_scans", []):
        targets.append(
            ScanTarget(
                scope="mock",
                dataset="mock",
                seed=int(item["seed"]),
                method=str(item["method"]),
                model_type=str(item["model_type"]),
                mode=str(item["mode"]),
            )
        )
    for item in audit.get("missing_real_parameter_scans", []):
        targets.append(
            ScanTarget(
                scope="real",
                dataset=str(item["dataset"]),
                seed=int(item["seed"]),
                method=str(item["method"]),
                model_type=str(item["model_type"]),
                mode=str(item["mode"]),
            )
        )
    return sorted(
        set(targets),
        key=lambda t: (
            0 if t.scope == "mock" else 1,
            DATASET_PRIORITY.get(t.dataset, 99),
            t.seed,
            t.method,
            t.model_type,
            t.mode,
        ),
    )


def _group_targets(targets: list[ScanTarget]) -> dict[tuple[str, str, int, str], list[ScanTarget]]:
    groups: dict[tuple[str, str, int, str], list[ScanTarget]] = {}
    for target in targets:
        groups.setdefault((target.scope, target.dataset, target.seed, target.method), []).append(target)
    return groups


def run_group(
    targets: list[ScanTarget],
    *,
    skip_existing: bool,
    max_epochs: int | None,
    run_log: list[dict[str, Any]],
    missing: list[dict[str, Any]],
) -> None:
    runnable: list[ScanTarget] = []
    for target in targets:
        out_csv = _target_path(target)
        if skip_existing and _csv_is_complete(out_csv, target.method):
            run_log.append({"target": _rel(out_csv), "action": "skipped_existing"})
            continue
        if _normalize_legacy(target, run_log):
            continue
        runnable.append(target)

    if not runnable:
        return

    base_target = runnable[0]
    cfg = _load_cfg(base_target, max_epochs=max_epochs)
    rows_by_target: dict[ScanTarget, list[dict[str, Any]]] = {target: [] for target in runnable}
    comp_dir = _target_path(base_target).parent

    for spec in _param_specs(base_target.method):
        run_cfg = _clone_cfg(cfg)
        _apply_param(run_cfg, base_target.method, spec)
        try:
            summary = run_defense_pipeline(run_cfg)
        except Exception as exc:
            for target in runnable:
                missing.append(
                    {
                        "section": "parameter_scan",
                        "dataset": target.dataset,
                        "seed": target.seed,
                        "method": target.method,
                        "model_type": target.model_type,
                        "mode": target.mode,
                        "param_name": spec["param_name"],
                        "param_value": spec["param_value"],
                        "reason": "defense_pipeline_failed",
                        "error": repr(exc),
                        "expected_file": _rel(_target_path(target)),
                    }
                )
            continue

        distortion = summary.get("distortion", {})
        for target in runnable:
            try:
                if target.mode == "fixed_attacker":
                    model_path = _clean_model_path(run_cfg, target.model_type)
                    pair = compute_fixed_attacker_metrics(run_cfg, model_path)
                    baseline_metrics = pair["baseline"]
                    defended_metrics = pair["defended"]
                    model_source = model_path
                else:
                    baseline_metrics, defended_metrics, model_source = _evaluate_retrained(
                        run_cfg,
                        target=target,
                        spec=spec,
                        out_dir=_target_path(target).parent,
                    )
                rows_by_target[target].append(
                    _row(
                        target=target,
                        spec=spec,
                        distortion=distortion,
                        baseline_metrics=baseline_metrics,
                        defended_metrics=defended_metrics,
                        model_source=model_source,
                    )
                )
            except Exception as exc:
                missing.append(
                    {
                        "section": "parameter_scan",
                        "dataset": target.dataset,
                        "seed": target.seed,
                        "method": target.method,
                        "model_type": target.model_type,
                        "mode": target.mode,
                        "param_name": spec["param_name"],
                        "param_value": spec["param_value"],
                        "reason": "evaluation_failed",
                        "error": repr(exc),
                        "expected_file": _rel(_target_path(target)),
                    }
                )

    for target, rows in rows_by_target.items():
        out_csv = _target_path(target)
        if len(rows) == PARAM_ROWS[target.method]:
            _write_csv(out_csv, rows)
            _plot_rows(rows, out_csv.parent, target)
            run_log.append({"target": _rel(out_csv), "action": "generated", "rows": len(rows)})
        else:
            missing.append(
                {
                    "section": "parameter_scan",
                    "dataset": target.dataset,
                    "seed": target.seed,
                    "method": target.method,
                    "model_type": target.model_type,
                    "mode": target.mode,
                    "reason": f"incomplete_rows_{len(rows)}_expected_{PARAM_ROWS[target.method]}",
                    "expected_file": _rel(out_csv),
                }
            )
            run_log.append({"target": _rel(out_csv), "action": "incomplete", "rows": len(rows)})


def _flush(run_log: list[dict[str, Any]], missing: list[dict[str, Any]]) -> None:
    _write_json(RUN_LOG_PATH, run_log)
    _write_json(MISSING_PATH, missing)


def _append_pending_as_missing(
    groups: list[tuple[tuple[str, str, int, str], list[ScanTarget]]],
    *,
    reason: str,
    missing: list[dict[str, Any]],
) -> None:
    for _, targets in groups:
        for target in targets:
            out_csv = _target_path(target)
            if _csv_is_complete(out_csv, target.method):
                continue
            missing.append(
                {
                    "section": "parameter_scan",
                    "dataset": target.dataset,
                    "seed": target.seed,
                    "method": target.method,
                    "model_type": target.model_type,
                    "mode": target.mode,
                    "reason": reason,
                    "expected_file": _rel(out_csv),
                }
            )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    parser = argparse.ArgumentParser(description="Run only missing parameter scans from the final symmetry audit")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--max-epochs", type=int, default=25)
    args = parser.parse_args()

    OUT_REPORT.mkdir(parents=True, exist_ok=True)
    targets = _audit_targets()
    run_log: list[dict[str, Any]] = [
        {
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "skip_existing": bool(args.skip_existing),
            "max_epochs": int(args.max_epochs),
            "targets_from_audit": len(targets),
        }
    ]
    missing: list[dict[str, Any]] = []

    groups = sorted(
        _group_targets(targets).items(),
        key=lambda item: (
            0 if item[0][0] == "mock" else 1,
            DATASET_PRIORITY.get(item[0][1], 99),
            item[0][2],
            item[0][3],
        ),
    )
    for key, group_targets in groups:
        logging.info("running group %s with %d target(s)", key, len(group_targets))
        run_group(
            group_targets,
            skip_existing=bool(args.skip_existing),
            max_epochs=int(args.max_epochs),
            run_log=run_log,
            missing=missing,
        )
        _flush(run_log, missing)

    run_log.append({"finished_at": datetime.now().isoformat(timespec="seconds"), "missing_count": len(missing)})
    _flush(run_log, missing)
    print(f"parameter_scan_run_log={_rel(RUN_LOG_PATH)}")
    print(f"parameter_scan_missing_outputs={_rel(MISSING_PATH)}")
    print(f"parameter_scan_missing_count={len(missing)}")


if __name__ == "__main__":
    main()
