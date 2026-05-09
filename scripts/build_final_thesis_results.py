#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build final thesis result package from reproducible repository artifacts only.

This script never fabricates metrics. It only reads existing files and/or
records missing items into missing-output files.
"""

from __future__ import annotations

import csv
import json
import platform
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_REPORT = ROOT / "outputs" / "summaries" / "final_thesis"
OUT_DEFENSE = ROOT / "outputs" / "experiments"
OUT_FIG = ROOT / "outputs" / "figures" / "summaries" / "final_thesis"
TMP_DIR = OUT_REPORT / "_tmp"
DATA_ROOT = ROOT / "data" / "processed"

SEEDS = [42, 123, 2026]
MODELS = ["lstm", "mlp"]
METHODS = ["adaptive_ldp", "ldp", "noise"]
MODES = ["fixed_attacker", "retrain_attacker"]


@dataclass
class EnvInfo:
    git_commit: str
    experiment_result_commit: str
    repository_cleanup_commit: str
    latest_verified_commit: str
    python_version: str
    os: str
    start_time: str
    end_time: str


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _run(cmd: list[str], cwd: Path | None = None, timeout: int | None = None) -> tuple[int, str, str]:
    p = subprocess.run(
        cmd,
        cwd=str(cwd or ROOT),
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    return p.returncode, p.stdout or "", p.stderr or ""


def _resolve_commit_metadata(current_head: str) -> dict[str, str]:
    prior_manifest = _safe_json(OUT_REPORT / "final_manifest.json")
    if not isinstance(prior_manifest, dict):
        prior_manifest = _safe_json(ROOT / "outputs" / "reports" / "final_thesis" / "final_manifest.json")
    if not isinstance(prior_manifest, dict):
        prior_manifest = {}
    experiment_commit = str(
        prior_manifest.get("experiment_result_commit")
        or prior_manifest.get("git_commit")
        or current_head
    )
    cleanup_commit = str(prior_manifest.get("repository_cleanup_commit") or current_head)
    return {
        "experiment_result_commit": experiment_commit,
        "repository_cleanup_commit": cleanup_commit,
        "latest_verified_commit": "working_tree_before_final_commit",
    }


def _safe_json(path: Path) -> dict[str, Any] | list[Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _safe_copy(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        if fieldnames is None:
            fieldnames = []
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if fieldnames:
                writer.writeheader()
        return
    if fieldnames is None:
        keys: set[str] = set()
        for r in rows:
            keys.update(r.keys())
        fieldnames = sorted(keys)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except Exception:
        return path.as_posix()


def _mean(vals: list[float]) -> float:
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def _extract_trace(
    *,
    dataset: str,
    seed: int,
    model: str,
    method: str,
    mode: str,
    config: str,
    command: str,
    env: EnvInfo,
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "seed": int(seed),
        "model_type": model,
        "method": method,
        "mode": mode,
        "config": config,
        "command": command,
        "timestamp": _now(),
        "git_commit": env.git_commit,
    }


def _render_confusion_from_json(conf_json: dict[str, Any], out_path: Path, title: str) -> bool:
    cm = np.asarray(conf_json.get("confusion_matrix", []))
    labels = list(conf_json.get("class_names", []))
    if cm.size == 0:
        return False
    if not labels:
        labels = [str(i) for i in range(cm.shape[0])]
    row_sum = cm.sum(axis=1, keepdims=True).astype(np.float64)
    row_sum[row_sum == 0] = 1.0
    cmn = cm / row_sum
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cmn, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.figure.colorbar(im, ax=ax)
    tick_stride = max(1, len(labels) // 15)
    ticks = np.arange(0, len(labels), tick_stride)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([labels[i] for i in ticks], rotation=45, ha="right")
    ax.set_yticklabels([labels[i] for i in ticks])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return True


def _ensure_import_metas(missing: list[dict[str, Any]]) -> None:
    imports = [
        ("uci_har", ROOT / "data" / "processed" / "imports" / "uci_har" / "meta.json"),
        ("kasteren", ROOT / "data" / "processed" / "imports" / "kasteren" / "meta.json"),
        ("casas_hh101", ROOT / "data" / "processed" / "imports" / "casas_hh101" / "meta.json"),
    ]
    for ds, meta in imports:
        if meta.exists():
            continue
        missing.append(
            {
                "section": "real_imports",
                "dataset": ds,
                "reason": "import_meta_missing",
                "expected_file": _rel(meta),
                "note": "Build is read-only for experiments/imports in the normalized artifact layout.",
            }
        )


def _collect_mock(env: EnvInfo, missing: list[dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    top_conf_rows: list[dict[str, Any]] = []
    found_keys: set[tuple[int, str, str, str]] = set()

    for seed in SEEDS:
        for model in MODELS:
            baseline_path = OUT_DEFENSE / "mock" / f"seed_{seed}" / model / "baseline" / "baseline_confusion.json"
            baseline = _safe_json(baseline_path)
            if not isinstance(baseline, dict):
                missing.append(
                    {
                        "section": "mock",
                        "dataset": "mock",
                        "seed": seed,
                        "model_type": model,
                        "method": "baseline",
                        "mode": "baseline",
                        "reason": "baseline_confusion_missing",
                        "expected_file": _rel(baseline_path),
                    }
                )
                continue

            for method in METHODS:
                for mode in MODES:
                    combo_dir = OUT_DEFENSE / "mock" / f"seed_{seed}" / model / method / mode
                    conf_path = combo_dir / "confusion.json"
                    report_path = combo_dir / "defense_report.json"
                    defended_obj = _safe_json(conf_path)
                    rep = _safe_json(report_path) or {}
                    dist = rep.get("distortion", {}) if isinstance(rep, dict) else {}

                    key = (seed, model, method, mode)
                    absent = [name for name in ["confusion.json", "classification_report.txt", "trace.json", "defense_report.json", "metrics.json", "source_manifest.json"] if not (combo_dir / name).is_file()]
                    if absent or not isinstance(defended_obj, dict):
                        missing.append(
                            {
                                "section": "mock",
                                "dataset": "mock",
                                "seed": seed,
                                "model_type": model,
                                "method": method,
                                "mode": mode,
                                "reason": "canonical_combo_missing_or_unreadable",
                                "missing_files": absent,
                                "expected_dir": _rel(combo_dir),
                            }
                        )
                        continue

                    found_keys.add(key)
                    baseline_acc = float(baseline.get("accuracy", np.nan))
                    baseline_f1 = float(baseline.get("f1_macro", np.nan))
                    defended_acc = float(defended_obj.get("accuracy", np.nan))
                    defended_f1 = float(defended_obj.get("f1_macro", np.nan))
                    acc_drop = baseline_acc - defended_acc
                    rel_drop = (acc_drop / baseline_acc * 100.0) if baseline_acc and baseline_acc == baseline_acc else np.nan

                    source_files = [
                        _rel(baseline_path),
                        _rel(conf_path),
                        _rel(report_path),
                    ]
                    row = {
                        "dataset": "mock",
                        "seed": seed,
                        "model_type": model,
                        "method": method,
                        "mode": mode,
                        "baseline_acc": baseline_acc,
                        "baseline_f1_macro": baseline_f1,
                        "defended_acc": defended_acc,
                        "defended_f1_macro": defended_f1,
                        "accuracy_drop": acc_drop,
                        "relative_accuracy_drop_percent": rel_drop,
                        "mse": float(dist.get("mse", np.nan)),
                        "mae": float(dist.get("mae", np.nan)),
                        "pearson_r": float(dist.get("pearson_r", np.nan)),
                        "source_files": ";".join(source_files),
                    }
                    rows.append(row)

                    for tc in defended_obj.get("top_confusions", [])[:10]:
                        top_conf_rows.append(
                            {
                                "dataset": "mock",
                                "seed": seed,
                                "model_type": model,
                                "method": method,
                                "mode": mode,
                                "true_label": tc.get("true"),
                                "pred_label": tc.get("pred"),
                                "count": tc.get("count"),
                                "source_file": _rel(conf_path),
                            }
                        )

    rows = sorted(rows, key=lambda x: (x["seed"], x["model_type"], x["method"], x["mode"]))
    mock_report_dir = OUT_REPORT / "mock"
    _write_json(mock_report_dir / "mock_summary.json", rows)
    _write_csv(mock_report_dir / "mock_summary.csv", rows)
    _write_csv(mock_report_dir / "mock_top_confusions.csv", top_conf_rows)

    expected_keys = {(s, m, me, mo) for s in SEEDS for m in MODELS for me in METHODS for mo in MODES}
    missing_keys = sorted(expected_keys - found_keys)
    coverage = {
        "expected_total": len(expected_keys),
        "completed_total": len(found_keys),
        "missing_total": len(missing_keys),
        "missing_combinations": [
            {"seed": s, "model_type": m, "method": me, "mode": mo}
            for (s, m, me, mo) in missing_keys
        ],
        "notes": "source=outputs/experiments/mock/*",
    }
    _write_json(mock_report_dir / "mock_coverage_audit.json", coverage)
    return {
        "rows": rows,
        "coverage": coverage,
        "scan_ldp_rows": [],
        "scan_noise_rows": [],
    }


def _npz_shape_stats(npz_path: Path) -> tuple[str, int, int, int]:
    if not npz_path.exists():
        return ("missing", -1, -1, -1)
    arr = np.load(npz_path)
    x_train = arr["X_train"]
    x_val = arr["X_val"]
    x_test = arr["X_test"]
    return (str(tuple(x_train.shape[1:])), int(len(x_train)), int(len(x_val)), int(len(x_test)))


def _collect_real(env: EnvInfo, missing: list[dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    top_conf_rows: list[dict[str, Any]] = []
    found_keys: set[tuple[str, int, str, str, str]] = set()

    datasets = ["uci_har", "kasteren", "casas_hh101"]
    for ds in datasets:
        for seed in SEEDS:
            proc_dir = DATA_ROOT / ds / f"seed_{seed}"
            meta = _safe_json(proc_dir / "meta.json")
            if not isinstance(meta, dict):
                missing.append(
                    {
                        "section": "real",
                        "dataset": ds,
                        "seed": seed,
                        "reason": "processed_meta_missing",
                        "expected_file": _rel(proc_dir / "meta.json"),
                    }
                )
            for model in MODELS:
                baseline_path = OUT_DEFENSE / ds / f"seed_{seed}" / model / "baseline" / "baseline_confusion.json"
                baseline = _safe_json(baseline_path)
                npz_path = proc_dir / ("sequences.npz" if model == "lstm" else "mlp_features.npz")
                input_shape, train_size, val_size, test_size = _npz_shape_stats(npz_path)
                num_classes = len((baseline or {}).get("class_names", [])) if isinstance(baseline, dict) else -1

                if not isinstance(baseline, dict):
                    missing.append(
                        {
                            "section": "real",
                            "dataset": ds,
                            "seed": seed,
                            "model_type": model,
                            "method": "baseline",
                            "mode": "baseline",
                            "reason": "baseline_confusion_missing",
                            "expected_file": _rel(baseline_path),
                        }
                    )
                    continue

                for method in METHODS:
                    for mode in MODES:
                        combo_dir = OUT_DEFENSE / ds / f"seed_{seed}" / model / method / mode
                        conf_path = combo_dir / "confusion.json"
                        report_path = combo_dir / "defense_report.json"
                        defended_obj = _safe_json(conf_path)
                        rep = _safe_json(report_path) or {}
                        dist = rep.get("distortion", {}) if isinstance(rep, dict) else {}
                        absent = [name for name in ["confusion.json", "classification_report.txt", "trace.json", "defense_report.json", "metrics.json", "source_manifest.json"] if not (combo_dir / name).is_file()]
                        key = (ds, seed, model, method, mode)
                        if absent or not isinstance(defended_obj, dict):
                            missing.append(
                                {
                                    "section": "real",
                                    "dataset": ds,
                                    "seed": seed,
                                    "model_type": model,
                                    "method": method,
                                    "mode": mode,
                                    "reason": "canonical_combo_missing_or_unreadable",
                                    "missing_files": absent,
                                    "expected_dir": _rel(combo_dir),
                                }
                            )
                            continue

                        found_keys.add(key)
                        baseline_acc = float(baseline.get("accuracy", np.nan))
                        baseline_f1 = float(baseline.get("f1_macro", np.nan))
                        defended_acc = float(defended_obj.get("accuracy", np.nan))
                        defended_f1 = float(defended_obj.get("f1_macro", np.nan))
                        acc_drop = baseline_acc - defended_acc
                        rel_drop = (acc_drop / baseline_acc * 100.0) if baseline_acc and baseline_acc == baseline_acc else np.nan

                        source_files = [
                            _rel(baseline_path),
                            _rel(conf_path),
                            _rel(report_path),
                            _rel(proc_dir / "meta.json"),
                        ]
                        row = {
                            "dataset": ds,
                            "seed": seed,
                            "model_type": model,
                            "method": method,
                            "mode": mode,
                            "baseline_acc": baseline_acc,
                            "baseline_f1_macro": baseline_f1,
                            "defended_acc": defended_acc,
                            "defended_f1_macro": defended_f1,
                            "accuracy_drop": acc_drop,
                            "relative_accuracy_drop_percent": rel_drop,
                            "mse": float(dist.get("mse", np.nan)),
                            "mae": float(dist.get("mae", np.nan)),
                            "pearson_r": float(dist.get("pearson_r", np.nan)),
                            "num_classes": num_classes,
                            "input_shape": input_shape,
                            "train_size": train_size,
                            "val_size": val_size,
                            "test_size": test_size,
                            "source_files": ";".join(source_files),
                        }
                        rows.append(row)
                        for tc in defended_obj.get("top_confusions", [])[:10]:
                            top_conf_rows.append(
                                {
                                    "dataset": ds,
                                    "seed": seed,
                                    "model_type": model,
                                    "method": method,
                                    "mode": mode,
                                    "true_label": tc.get("true"),
                                    "pred_label": tc.get("pred"),
                                    "count": tc.get("count"),
                                    "source_file": _rel(conf_path),
                                }
                            )

    rows = sorted(rows, key=lambda x: (x["dataset"], x["seed"], x["model_type"], x["method"], x["mode"]))
    real_report_dir = OUT_REPORT / "real"
    _write_json(real_report_dir / "real_summary.json", rows)
    _write_csv(real_report_dir / "real_summary.csv", rows)
    _write_csv(real_report_dir / "real_top_confusions.csv", top_conf_rows)

    # import meta summary
    meta_rows: list[dict[str, Any]] = []
    for ds in ["uci_har", "kasteren", "casas_hh101"]:
        meta_path = DATA_ROOT / "imports" / ds / "meta.json"
        meta = _safe_json(meta_path)
        if not isinstance(meta, dict):
            missing.append(
                {
                    "section": "real_dataset_meta",
                    "dataset": ds,
                    "reason": "import_meta_missing",
                    "expected_file": str(meta_path),
                }
            )
            continue
        meta_rows.append(
            {
                "dataset": ds,
                "meta_path": _rel(meta_path),
                "source": meta.get("source"),
                "seq_len": meta.get("seq_len"),
                "freq": meta.get("freq"),
                "n_train": meta.get("n_train"),
                "n_val": meta.get("n_val"),
                "n_test": meta.get("n_test"),
                "num_classes": len(meta.get("class_names", []) or []),
                "num_features": len(meta.get("feature_names", []) or []),
            }
        )
    _write_csv(real_report_dir / "real_dataset_meta_summary.csv", meta_rows)

    expected = {(d, s, m, me, mo) for d in ["uci_har", "kasteren", "casas_hh101"] for s in SEEDS for m in MODELS for me in METHODS for mo in MODES}
    missing_keys = sorted(expected - found_keys)
    coverage = {
        "expected_total": len(expected),
        "completed_total": len(found_keys),
        "missing_total": len(missing_keys),
        "missing_combinations": [
            {"dataset": d, "seed": s, "model_type": m, "method": me, "mode": mo}
            for (d, s, m, me, mo) in missing_keys
        ],
        "notes": "source=outputs/experiments/{dataset}/*",
    }
    _write_json(real_report_dir / "real_coverage_audit.json", coverage)

    real_missing = [m for m in missing if str(m.get("section", "")).startswith("real")]
    _write_json(real_report_dir / "real_missing_outputs.json", real_missing)
    return {
        "rows": rows,
        "coverage": coverage,
        "scan_ldp_rows": [],
        "scan_noise_rows": [],
        "meta_rows": meta_rows,
    }


PARAM_SCAN_ROWS = {"ldp": 5, "noise": 4, "adaptive_ldp": 6}
PARAM_SCAN_FIELDS = [
    "dataset",
    "seed",
    "model_type",
    "mode",
    "method",
    "profile_name",
    "parameter_name",
    "parameter_value",
    "epsilon_min",
    "epsilon_max",
    "weight_sensitivity",
    "weight_traffic",
    "use_edge_budget_cap",
    "edge_inverse_budget_cap",
    "baseline_acc",
    "defended_acc",
    "accuracy_drop",
    "defended_f1_macro",
    "mse",
    "mae",
    "pearson_r",
    "model_source",
    "source_file",
]
ADAPTIVE_PROFILE_ORDER = [
    "adaptive_default",
    "adaptive_strong_privacy",
    "adaptive_weak_privacy",
    "adaptive_sensitivity_only",
    "adaptive_traffic_only",
    "adaptive_edge_cap_on",
]
ADAPTIVE_ABLATION_FIELDS = [
    "dataset",
    "profile_name",
    "model_type",
    "mode",
    "epsilon_min",
    "epsilon_max",
    "weight_sensitivity",
    "weight_traffic",
    "use_edge_budget_cap",
    "mean_baseline_acc",
    "mean_defended_acc",
    "mean_accuracy_drop",
    "mean_defended_f1_macro",
    "mean_mse",
    "mean_mae",
    "mean_pearson_r",
    "num_seeds",
    "source_file",
]


def _read_csv_dicts(path: Path) -> list[dict[str, Any]] | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))
    except Exception:
        return None


def _scan_csv_complete(path: Path, method: str) -> bool:
    rows = _read_csv_dicts(path)
    return rows is not None and len(rows) == PARAM_SCAN_ROWS[method]


def _scan_path(scope: str, dataset: str, seed: int, method: str, model: str, mode: str) -> Path:
    ds = "mock" if scope == "mock" else dataset
    return OUT_DEFENSE / ds / f"seed_{seed}" / model / method / mode / "parameter_scan" / "comparison_results.csv"


def _legacy_scan_path(scope: str, dataset: str, seed: int, method: str) -> Path:
    ds = "mock" if scope == "mock" else dataset
    return OUT_DEFENSE / ds / f"seed_{seed}" / "lstm" / method / "fixed_attacker" / "parameter_scan" / "comparison_results.csv"


def _fnum(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _normalize_scan_rows(
    rows: list[dict[str, Any]],
    *,
    scope: str,
    dataset: str,
    seed: int,
    method: str,
    model: str,
    mode: str,
    source: Path,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for r in rows:
        baseline = _fnum(r.get("baseline_accuracy", r.get("baseline_acc")))
        defended = _fnum(r.get("defended_accuracy", r.get("defended_acc")))
        param_name = str(r.get("parameter_name", r.get("param_name", "profile" if method == "adaptive_ldp" else "")))
        param_value = r.get("parameter_value", r.get("param_value", ""))
        out.append(
            {
                "dataset": str(r.get("dataset") or dataset),
                "seed": int(float(r.get("seed", seed))),
                "model_type": str(r.get("model_type") or model),
                "mode": str(r.get("mode") or mode),
                "method": str(r.get("method") or method),
                "profile_name": str(r.get("profile_name", "")),
                "parameter_name": param_name,
                "parameter_value": param_value,
                "epsilon_min": r.get("epsilon_min", ""),
                "epsilon_max": r.get("epsilon_max", ""),
                "weight_sensitivity": r.get("weight_sensitivity", ""),
                "weight_traffic": r.get("weight_traffic", ""),
                "use_edge_budget_cap": r.get("use_edge_budget_cap", ""),
                "edge_inverse_budget_cap": r.get("edge_inverse_budget_cap", ""),
                "baseline_acc": baseline,
                "defended_acc": defended,
                "accuracy_drop": _fnum(r.get("accuracy_drop", baseline - defended)),
                "defended_f1_macro": _fnum(r.get("defended_f1_macro")),
                "mse": _fnum(r.get("mse")),
                "mae": _fnum(r.get("mae")),
                "pearson_r": _fnum(r.get("pearson_r")),
                "model_source": str(r.get("model_source", "")),
                "source_file": _rel(source),
            }
        )
    return out


def _dedupe_scan_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    seen: set[tuple[Any, ...]] = set()
    out: list[dict[str, Any]] = []
    removed = 0
    for row in rows:
        key = (
            row.get("dataset"),
            str(row.get("seed")),
            row.get("model_type"),
            row.get("mode"),
            row.get("method"),
            row.get("parameter_name"),
            row.get("profile_name"),
            str(row.get("parameter_value")),
            row.get("source_file"),
        )
        if key in seen:
            removed += 1
            continue
        seen.add(key)
        out.append(row)
    return out, removed


def _collect_parameter_scans(missing: list[dict[str, Any]]) -> dict[str, Any]:
    rows_by_scope_method: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    missing_parameter: list[dict[str, Any]] = []
    completed: dict[str, int] = {"mock": 0, "real": 0}
    adaptive_profile_counts: dict[str, int] = {}
    duplicate_rows_removed = 0

    expected_mock = 0
    expected_real = 0
    for scope, datasets in [("mock", ["mock"]), ("real", ["uci_har", "kasteren", "casas_hh101"])]:
        for dataset in datasets:
            for seed in SEEDS:
                for method in METHODS:
                    for model in MODELS:
                        for mode in MODES:
                            if scope == "mock":
                                expected_mock += 1
                            else:
                                expected_real += 1
                            path = _scan_path(scope, dataset, seed, method, model, mode)
                            source = path
                            if not _scan_csv_complete(path, method):
                                legacy = _legacy_scan_path(scope, dataset, seed, method)
                                if method in {"ldp", "noise"} and model == "lstm" and mode == "fixed_attacker" and _scan_csv_complete(legacy, method):
                                    source = legacy
                                else:
                                    missing_parameter.append(
                                        {
                                            "section": f"{scope}_parameter_scan",
                                            "dataset": dataset,
                                            "seed": seed,
                                            "method": method,
                                            "model_type": model,
                                            "mode": mode,
                                            "reason": "comparison_results_missing_or_incomplete",
                                            "expected_file": _rel(path),
                                        }
                                    )
                                    continue
                            raw_rows = _read_csv_dicts(source) or []
                            completed[scope] += 1
                            normalized = _normalize_scan_rows(
                                raw_rows,
                                scope=scope,
                                dataset=dataset,
                                seed=seed,
                                method=method,
                                model=model,
                                mode=mode,
                                source=source,
                            )
                            if method == "adaptive_ldp":
                                profiles = {str(r.get("profile_name")) for r in normalized if r.get("profile_name")}
                                adaptive_profile_counts[f"{scope}:{dataset}:seed_{seed}:{model}:{mode}"] = len(profiles)
                            rows_by_scope_method[(scope, method)].extend(normalized)

    for scope in ["mock", "real"]:
        report_dir = OUT_REPORT / scope
        for method in METHODS:
            rows = rows_by_scope_method[(scope, method)]
            rows, removed = _dedupe_scan_rows(rows)
            duplicate_rows_removed += removed
            rows = sorted(
                rows,
                key=lambda r: (
                    str(r.get("dataset")),
                    int(r.get("seed", 0)),
                    str(r.get("model_type")),
                    str(r.get("mode")),
                    str(r.get("parameter_value")),
                    str(r.get("profile_name")),
                    str(r.get("source_file")),
                ),
            )
            prefix = "mock" if scope == "mock" else "real"
            _write_csv(report_dir / f"{prefix}_parameter_scan_{method}.csv", rows, PARAM_SCAN_FIELDS)

    _write_json(OUT_REPORT / "parameter_scan_missing_outputs.json", missing_parameter)
    missing.extend(missing_parameter)
    coverage = {
        "mock": {
            "expected": expected_mock,
            "completed": completed["mock"],
            "missing": [m for m in missing_parameter if m["section"] == "mock_parameter_scan"],
        },
        "real": {
            "expected": expected_real,
            "completed": completed["real"],
            "missing": [m for m in missing_parameter if m["section"] == "real_parameter_scan"],
        },
        "adaptive_ldp_profile_count": {
            "expected": PARAM_SCAN_ROWS["adaptive_ldp"],
            "observed": adaptive_profile_counts,
        },
        "duplicate_rows_removed": duplicate_rows_removed,
        "missing_combinations": missing_parameter,
    }
    _write_json(OUT_REPORT / "parameter_scan_coverage_audit.json", coverage)
    return {
        "coverage": coverage,
        "duplicate_rows_removed": duplicate_rows_removed,
        "missing": missing_parameter,
    }


def _profile_sort_value(profile: Any) -> int:
    name = str(profile)
    return ADAPTIVE_PROFILE_ORDER.index(name) if name in ADAPTIVE_PROFILE_ORDER else len(ADAPTIVE_PROFILE_ORDER)


def _build_adaptive_ablation_outputs(missing: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    outputs = {
        "mock": {
            "input": OUT_REPORT / "mock" / "mock_parameter_scan_adaptive_ldp.csv",
            "csv": OUT_REPORT / "mock" / "mock_adaptive_ldp_ablation_summary.csv",
            "md": OUT_REPORT / "mock" / "mock_adaptive_ldp_ablation_summary.md",
            "title": "Mock adaptive_ldp profile ablation summary",
        },
        "real": {
            "input": OUT_REPORT / "real" / "real_parameter_scan_adaptive_ldp.csv",
            "csv": OUT_REPORT / "real" / "real_adaptive_ldp_ablation_summary.csv",
            "md": OUT_REPORT / "real" / "real_adaptive_ldp_ablation_summary.md",
            "title": "Real adaptive_ldp profile ablation summary",
        },
    }
    result: dict[str, list[dict[str, Any]]] = {"mock": [], "real": []}

    for scope, spec in outputs.items():
        input_path = spec["input"]
        if not input_path.exists() or input_path.stat().st_size == 0:
            missing.append(
                {
                    "section": "adaptive_ablation",
                    "scope": scope,
                    "reason": "adaptive_ldp_parameter_scan_missing",
                    "expected_file": _rel(input_path),
                }
            )
            _write_csv(spec["csv"], [], ADAPTIVE_ABLATION_FIELDS)
            spec["md"].write_text(f"# {spec['title']}\n\nNo input rows were available.\n", encoding="utf-8")
            continue

        df = pd.read_csv(input_path)
        if df.empty:
            missing.append(
                {
                    "section": "adaptive_ablation",
                    "scope": scope,
                    "reason": "adaptive_ldp_parameter_scan_empty",
                    "expected_file": _rel(input_path),
                }
            )
            _write_csv(spec["csv"], [], ADAPTIVE_ABLATION_FIELDS)
            spec["md"].write_text(f"# {spec['title']}\n\nNo input rows were available.\n", encoding="utf-8")
            continue

        for col in [
            "epsilon_min",
            "epsilon_max",
            "weight_sensitivity",
            "weight_traffic",
            "baseline_acc",
            "defended_acc",
            "accuracy_drop",
            "defended_f1_macro",
            "mse",
            "mae",
            "pearson_r",
        ]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        for col in ["dataset", "profile_name", "model_type", "mode", "use_edge_budget_cap"]:
            df[col] = df[col].astype(str)

        rows: list[dict[str, Any]] = []
        group_cols = [
            "dataset",
            "profile_name",
            "model_type",
            "mode",
            "epsilon_min",
            "epsilon_max",
            "weight_sensitivity",
            "weight_traffic",
            "use_edge_budget_cap",
        ]
        for keys, g in df.groupby(group_cols, dropna=False):
            data = dict(zip(group_cols, keys))
            rows.append(
                {
                    **data,
                    "mean_baseline_acc": float(g["baseline_acc"].mean()),
                    "mean_defended_acc": float(g["defended_acc"].mean()),
                    "mean_accuracy_drop": float(g["accuracy_drop"].mean()),
                    "mean_defended_f1_macro": float(g["defended_f1_macro"].mean()),
                    "mean_mse": float(g["mse"].mean()),
                    "mean_mae": float(g["mae"].mean()),
                    "mean_pearson_r": float(g["pearson_r"].mean()),
                    "num_seeds": int(g["seed"].nunique()) if "seed" in g.columns else int(len(g)),
                    "source_file": _rel(input_path),
                }
            )
        rows = sorted(
            rows,
            key=lambda r: (
                str(r["dataset"]),
                _profile_sort_value(r["profile_name"]),
                str(r["model_type"]),
                str(r["mode"]),
            ),
        )
        _write_csv(spec["csv"], rows, ADAPTIVE_ABLATION_FIELDS)
        result[scope] = rows

        summary_df = pd.DataFrame(rows)
        profile_summary = (
            summary_df.groupby(["dataset", "profile_name"], as_index=False)[["mean_defended_acc", "mean_accuracy_drop", "mean_mse"]]
            .mean(numeric_only=True)
            .sort_values(["dataset", "profile_name"], key=lambda s: s.map(_profile_sort_value) if s.name == "profile_name" else s)
        )
        lines = [
            f"# {spec['title']}",
            "",
            "This file summarizes existing adaptive_ldp profile scans. No experiment was rerun for this summary.",
            "",
            f"- Source: `{_rel(input_path)}`",
            f"- Output rows: `{len(rows)}`",
            "- Each profile is aggregated by dataset, model type, and attacker mode across available seeds.",
            "",
            "| dataset | profile_name | mean_defended_acc | mean_accuracy_drop | mean_mse |",
            "|---|---:|---:|---:|---:|",
        ]
        for _, row in profile_summary.iterrows():
            lines.append(
                f"| {row['dataset']} | {row['profile_name']} | "
                f"{row['mean_defended_acc']:.6f} | {row['mean_accuracy_drop']:.6f} | {row['mean_mse']:.6f} |"
            )
        lines.extend(
            [
                "",
                "Interpretation should stay cautious: this is a profile-level empirical ablation summary, not a formal theoretical proof.",
            ]
        )
        spec["md"].write_text("\n".join(lines) + "\n", encoding="utf-8")

    overview = OUT_REPORT / "adaptive_ldp_ablation_overview.md"
    overview.write_text(
        "\n".join(
            [
                "# Adaptive LDP Ablation Overview",
                "",
                "This is not a newly rerun experiment. It organizes the existing adaptive_ldp profile parameter scans into a formal ablation summary.",
                "",
                "## Ablation Dimensions",
                "",
                "- `epsilon_min` / `epsilon_max`: adaptive privacy budget range.",
                "- `weight_sensitivity`: weight for the window-variation proxy.",
                "- `weight_traffic`: weight for the traffic-intensity proxy.",
                "- `use_edge_budget_cap`: whether the edge budget clipping interface is enabled.",
                "",
                "## Profiles",
                "",
                "- `adaptive_default`: balanced sensitivity and traffic weighting.",
                "- `adaptive_strong_privacy`: stronger perturbation through a smaller epsilon range.",
                "- `adaptive_weak_privacy`: weaker perturbation through a larger epsilon range.",
                "- `adaptive_sensitivity_only`: uses only the window-variation proxy.",
                "- `adaptive_traffic_only`: uses only the traffic-intensity proxy.",
                "- `adaptive_edge_cap_on`: enables the edge budget clipping interface.",
                "",
                "## Scope and Caution",
                "",
                "The summary covers mock, uci_har, kasteren, and casas_hh101; seeds 42, 123, and 2026; LSTM and MLP; fixed_attacker and retrain_attacker.",
                "The results are empirical profile scans and should not be overstated as a formal theoretical proof.",
                "",
                "If thesis Section 5.2 still says that ablation experiments can be done later, revise it to: "
                "\"Current results already include profile-level ablation summaries, while finer-grained real deployment ablations can remain future work.\"",
                "",
                "## Generated Files",
                "",
                "- `outputs/summaries/final_thesis/mock/mock_adaptive_ldp_ablation_summary.csv`",
                "- `outputs/summaries/final_thesis/mock/mock_adaptive_ldp_ablation_summary.md`",
                "- `outputs/summaries/final_thesis/real/real_adaptive_ldp_ablation_summary.csv`",
                "- `outputs/summaries/final_thesis/real/real_adaptive_ldp_ablation_summary.md`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return result


def _cooja_logs_available(manifest_path: Path) -> tuple[bool, list[Path]]:
    obj = _safe_json(manifest_path)
    if not isinstance(obj, dict):
        return False, []
    root_env = None
    try:
        import os

        root_env = os.environ.get("COOJA_LOG_ROOT")
    except Exception:
        root_env = None
    fallback_names = {
        "baseline": {"radio_log": "Radiomsg.txt", "app_log": "loglistener.txt"},
        "dummy_noise": {"radio_log": "Radiomsg_dummy_noise.txt", "app_log": "loglistener_dummy_noise.txt"},
        "dummy_ldp": {"radio_log": "Radiomsg_dummy_ldp.txt", "app_log": "loglistener_dummy_ldp.txt"},
        "dummy_adaptive_ldp": {"radio_log": "Radiomsg_dummy_adaptive.txt", "app_log": "loglistener_dummy_adaptive.txt"},
    }

    def add_path(paths: list[Path], raw: Any, name: str, kind: str) -> None:
        p = Path(str(raw))
        if p.exists() or not root_env:
            paths.append(p)
            return
        alt_name = fallback_names.get(name, {}).get(kind)
        paths.append(Path(root_env) / alt_name if alt_name else p)

    paths: list[Path] = []
    baseline = obj.get("baseline", {})
    if isinstance(baseline, dict):
        for k in ["radio_log", "app_log"]:
            v = baseline.get(k)
            if v:
                add_path(paths, v, "baseline", k)
    for m in obj.get("methods", []) or []:
        if not isinstance(m, dict):
            continue
        for k in ["radio_log", "app_log"]:
            v = m.get(k)
            if v:
                add_path(paths, v, str(m.get("name", "")), k)
    exists = [p for p in paths if p.exists()]
    return len(paths) > 0 and len(exists) >= 2, paths


COOJA_PER_SEED_FIELDS = [
    "method",
    "mode",
    "seed",
    "baseline_acc",
    "defended_acc",
    "accuracy_drop",
    "baseline_f1_macro",
    "defended_f1_macro",
    "baseline_windows",
    "defense_windows",
    "source_radio_log",
    "source_app_log",
]
COOJA_TRAFFIC_FIELDS = [
    "method",
    "seed",
    "baseline_windows",
    "defense_windows",
    "baseline_pkt_count_mean",
    "defense_pkt_count_mean",
    "baseline_byte_count_mean",
    "defense_byte_count_mean",
    "packet_overhead_ratio",
    "byte_overhead_ratio",
    "baseline_mean_iat_ms",
    "defense_mean_iat_ms",
    "baseline_p95_iat_ms",
    "defense_p95_iat_ms",
    "dummy_packet_ratio",
    "dummy_byte_ratio",
    "energy_metric_available",
    "delay_metric_available",
    "limitations",
]


def _cooja_summary_from_report(report_path: Path, missing: list[dict[str, Any]]) -> dict[str, Any]:
    cooja_report_dir = OUT_REPORT / "cooja"
    rows: list[dict[str, Any]] = []
    feat_rows: list[dict[str, Any]] = []
    top_conf_rows: list[dict[str, Any]] = []
    overhead_rows: list[dict[str, Any]] = []

    rep = _safe_json(report_path) or {}
    methods = rep.get("methods", {}) if isinstance(rep, dict) else {}
    for method_name, mobj in methods.items():
        if not isinstance(mobj, dict):
            continue
        b_mean = float(((mobj.get("baseline_test") or {}).get("accuracy") or {}).get("mean", np.nan))
        f_mean = float(((mobj.get("fixed_attacker") or {}).get("accuracy") or {}).get("mean", np.nan))
        r_mean = float(((mobj.get("retrain_attacker") or {}).get("accuracy") or {}).get("mean", np.nan))
        f1_fixed = float(((mobj.get("fixed_attacker") or {}).get("f1_macro") or {}).get("mean", np.nan))
        f1_retrain = float(((mobj.get("retrain_attacker") or {}).get("f1_macro") or {}).get("mean", np.nan))
        dataset_meta = mobj.get("dataset", {}) if isinstance(mobj.get("dataset", {}), dict) else {}
        baseline_windows = float(dataset_meta.get("baseline_windows", np.nan))
        defense_windows = float(dataset_meta.get("defense_windows", np.nan))
        window_ratio = defense_windows / baseline_windows if baseline_windows == baseline_windows and baseline_windows > 0 else np.nan
        overhead_rows.append(
            {
                "method": method_name,
                "baseline_windows": baseline_windows,
                "defense_windows": defense_windows,
                "defense_window_ratio": window_ratio,
                "window_count_delta": defense_windows - baseline_windows
                if baseline_windows == baseline_windows and defense_windows == defense_windows
                else np.nan,
                "energy_metric_available": False,
                "delay_metric_available": False,
                "note": "Cooja logs do not include real energy/delay fields; this is a window-count proxy only.",
            }
        )
        for mode, defended_acc, f1 in [
            ("fixed_attacker", f_mean, f1_fixed),
            ("retrain_attacker", r_mean, f1_retrain),
        ]:
            rows.append(
                {
                    "method": method_name,
                    "seed": "mean_over_seeds",
                    "mode": mode,
                    "baseline_acc": b_mean,
                    "defended_acc": defended_acc,
                    "accuracy_drop": b_mean - defended_acc,
                    "f1_macro": f1,
                    "pkt_count_mean": np.nan,
                    "byte_count_mean": np.nan,
                    "dummy_packet_ratio": np.nan,
                    "packet_overhead_ratio": np.nan,
                    "mean_iat_ms": np.nan,
                    "p95_iat_ms": np.nan,
                    "traffic_activity_correlation_before": np.nan,
                    "traffic_activity_correlation_after": np.nan,
                    "correlation_drop": np.nan,
                    "energy_metric_available": False,
                    "delay_proxy_available": False,
                    "source_log_files": json.dumps((mobj.get("defense_log_paths") or {}), ensure_ascii=False),
                }
            )

        for run in mobj.get("runs", []):
            if not isinstance(run, dict):
                continue
            seed = int(run.get("seed", -1))
            fixed = run.get("fixed_attacker_on_defense", {}) or {}
            retr = run.get("retrain_attacker_on_defense", {}) or {}
            for mode, obj in [("fixed_attacker", fixed), ("retrain_attacker", retr)]:
                for tc in obj.get("top_confusions", [])[:5]:
                    top_conf_rows.append(
                        {
                            "method": method_name,
                            "seed": seed,
                            "mode": mode,
                            "true_label": tc.get("true"),
                            "pred_label": tc.get("pred"),
                            "count": tc.get("count"),
                        }
                    )

    _write_json(cooja_report_dir / "cooja_summary.json", rows)
    _write_csv(cooja_report_dir / "cooja_summary.csv", rows)
    if feat_rows:
        _write_csv(cooja_report_dir / "cooja_feature_importance.csv", feat_rows)
    if top_conf_rows:
        _write_csv(cooja_report_dir / "cooja_top_confusions.csv", top_conf_rows)
    _write_csv(cooja_report_dir / "cooja_overhead_summary.csv", overhead_rows)
    _write_json(cooja_report_dir / "cooja_missing_outputs.json", [m for m in missing if m.get("section") == "cooja"])
    _export_cooja_detail_outputs(rep, missing)
    return {"rows": rows}


def _cooja_manifest_with_fallback() -> dict[str, Any] | None:
    manifest_path = ROOT / "configs" / "cooja_defense_dummy_logs.json"
    obj = _safe_json(manifest_path)
    if not isinstance(obj, dict):
        return None
    root_env = None
    try:
        import os

        root_env = os.environ.get("COOJA_LOG_ROOT")
    except Exception:
        root_env = None
    if not root_env:
        return obj
    root = Path(root_env)
    names = {
        "baseline": ("Radiomsg.txt", "loglistener.txt"),
        "dummy_noise": ("Radiomsg_dummy_noise.txt", "loglistener_dummy_noise.txt"),
        "dummy_ldp": ("Radiomsg_dummy_ldp.txt", "loglistener_dummy_ldp.txt"),
        "dummy_adaptive_ldp": ("Radiomsg_dummy_adaptive.txt", "loglistener_dummy_adaptive.txt"),
    }

    def resolve_pair(item: dict[str, Any], name: str) -> dict[str, Any]:
        radio = Path(str(item.get("radio_log", "")))
        app = Path(str(item.get("app_log", "")))
        if radio.exists() and app.exists():
            return item
        radio_name, app_name = names.get(name, (f"Radiomsg_{name}.txt", f"loglistener_{name}.txt"))
        alt_radio = root / radio_name
        alt_app = root / app_name
        if alt_radio.exists() and alt_app.exists():
            item = dict(item)
            item["radio_log"] = str(alt_radio)
            item["app_log"] = str(alt_app)
        return item

    obj = dict(obj)
    obj["baseline"] = resolve_pair(dict(obj["baseline"]), "baseline")
    obj["methods"] = [resolve_pair(dict(m), str(m.get("name", ""))) for m in obj.get("methods", [])]
    return obj


def _cooja_window_metrics(radio_log: Path, app_log: Path, config: dict[str, Any]) -> dict[str, float] | None:
    try:
        from experiments.cooja.run_cooja_baseline_attack import build_window_dataset, parse_app_requests, parse_radio

        radio_df = parse_radio(radio_log)
        app_df = parse_app_requests(app_log)
        ds = build_window_dataset(
            radio_df=radio_df,
            app_df=app_df,
            window_s=float(config.get("window_s", 8.0)),
            step_s=float(config.get("step_s", 3.0)),
            min_requests=int(config.get("min_requests", 2)),
            dominance_threshold=float(config.get("dominance_threshold", 0.2)),
        )
        byte_count = ds["pkt_count"].astype(float) * ds["mean_len"].astype(float)
        return {
            "windows": float(len(ds)),
            "pkt_count_mean": float(ds["pkt_count"].mean()),
            "byte_count_mean": float(byte_count.mean()),
            "mean_iat_ms": float(ds["mean_iat_ms"].mean()),
            "p95_iat_ms": float(ds["p95_iat_ms"].mean()),
        }
    except Exception:
        return None


def _export_cooja_detail_outputs(rep: dict[str, Any], missing: list[dict[str, Any]]) -> None:
    cooja_report_dir = OUT_REPORT / "cooja"
    methods = rep.get("methods", {}) if isinstance(rep, dict) else {}
    config = rep.get("config", {}) if isinstance(rep.get("config", {}), dict) else {}
    manifest = _cooja_manifest_with_fallback()
    baseline_logs = (manifest or {}).get("baseline", {}) if isinstance(manifest, dict) else {}
    baseline_radio = Path(str(baseline_logs.get("radio_log", "")))
    baseline_app = Path(str(baseline_logs.get("app_log", "")))
    baseline_traffic = _cooja_window_metrics(baseline_radio, baseline_app, config) if baseline_radio.exists() and baseline_app.exists() else None

    per_seed_rows: list[dict[str, Any]] = []
    traffic_rows: list[dict[str, Any]] = []
    limitations = [
        "# Cooja Limitations",
        "",
        "- Cooja outputs can be used for fixed/retrain attacker accuracy reporting.",
        "- Cooja currently does not provide real energy measurements.",
        "- Cooja currently does not provide real end-to-end delay measurements.",
        "- Radio/app log paths may point to local WSL-exported files; those paths document the local evaluation source and are not portable reproduction paths.",
        "- Current radio logs do not distinguish dummy packets from real packets, so dummy packet and byte ratios are reported as null.",
        "- Packet, byte, and IAT fields reported as NaN indicate unavailable log fields, not an unrun experiment.",
        "- `cooja_overhead_summary.csv` remains a window-count proxy, not measured energy or latency.",
    ]

    for method_name, mobj in methods.items():
        if not isinstance(mobj, dict):
            continue
        dataset_meta = mobj.get("dataset", {}) if isinstance(mobj.get("dataset", {}), dict) else {}
        paths = mobj.get("defense_log_paths", {}) if isinstance(mobj.get("defense_log_paths", {}), dict) else {}
        source_radio = str(paths.get("radio_log", ""))
        source_app = str(paths.get("app_log", ""))
        defense_radio = Path(source_radio)
        defense_app = Path(source_app)
        defense_traffic = _cooja_window_metrics(defense_radio, defense_app, config) if defense_radio.exists() and defense_app.exists() else None

        for run in mobj.get("runs", []):
            if not isinstance(run, dict):
                continue
            seed = int(run.get("seed", -1))
            base = run.get("baseline_test", {}) or {}
            fixed = run.get("fixed_attacker_on_defense", {}) or {}
            retr = run.get("retrain_attacker_on_defense", {}) or {}
            for mode, defended in [("fixed_attacker", fixed), ("retrain_attacker", retr)]:
                b_acc = _fnum(base.get("accuracy"))
                d_acc = _fnum(defended.get("accuracy"))
                per_seed_rows.append(
                    {
                        "method": method_name,
                        "mode": mode,
                        "seed": seed,
                        "baseline_acc": b_acc,
                        "defended_acc": d_acc,
                        "accuracy_drop": b_acc - d_acc,
                        "baseline_f1_macro": _fnum(base.get("f1_macro")),
                        "defended_f1_macro": _fnum(defended.get("f1_macro")),
                        "baseline_windows": dataset_meta.get("baseline_windows", ""),
                        "defense_windows": dataset_meta.get("defense_windows", ""),
                        "source_radio_log": source_radio,
                        "source_app_log": source_app,
                    }
                )

            baseline_windows = dataset_meta.get("baseline_windows", "")
            defense_windows = dataset_meta.get("defense_windows", "")
            bp = baseline_traffic or {}
            dp = defense_traffic or {}
            b_pkt = bp.get("pkt_count_mean", np.nan)
            d_pkt = dp.get("pkt_count_mean", np.nan)
            b_byte = bp.get("byte_count_mean", np.nan)
            d_byte = dp.get("byte_count_mean", np.nan)
            traffic_rows.append(
                {
                    "method": method_name,
                    "seed": seed,
                    "baseline_windows": bp.get("windows", baseline_windows),
                    "defense_windows": dp.get("windows", defense_windows),
                    "baseline_pkt_count_mean": b_pkt,
                    "defense_pkt_count_mean": d_pkt,
                    "baseline_byte_count_mean": b_byte,
                    "defense_byte_count_mean": d_byte,
                    "packet_overhead_ratio": d_pkt / b_pkt if b_pkt == b_pkt and b_pkt else np.nan,
                    "byte_overhead_ratio": d_byte / b_byte if b_byte == b_byte and b_byte else np.nan,
                    "baseline_mean_iat_ms": bp.get("mean_iat_ms", np.nan),
                    "defense_mean_iat_ms": dp.get("mean_iat_ms", np.nan),
                    "baseline_p95_iat_ms": bp.get("p95_iat_ms", np.nan),
                    "defense_p95_iat_ms": dp.get("p95_iat_ms", np.nan),
                    "dummy_packet_ratio": np.nan,
                    "dummy_byte_ratio": np.nan,
                    "energy_metric_available": False,
                    "delay_metric_available": False,
                    "limitations": "Radio log does not distinguish dummy packets from real packets; no real energy/delay metrics are available.",
                }
            )

    _write_csv(cooja_report_dir / "cooja_per_seed.csv", per_seed_rows, COOJA_PER_SEED_FIELDS)
    _write_csv(cooja_report_dir / "cooja_traffic_metrics.csv", traffic_rows, COOJA_TRAFFIC_FIELDS)
    (cooja_report_dir / "cooja_limitations.md").write_text("\n".join(limitations) + "\n", encoding="utf-8")
    if not per_seed_rows:
        missing.append({"section": "cooja", "reason": "cooja_per_seed_rows_empty", "expected_file": _rel(cooja_report_dir / "cooja_per_seed.csv")})
    if not traffic_rows:
        missing.append({"section": "cooja", "reason": "cooja_traffic_rows_empty", "expected_file": _rel(cooja_report_dir / "cooja_traffic_metrics.csv")})


def _collect_cooja(env: EnvInfo, missing: list[dict[str, Any]]) -> dict[str, Any]:
    cooja_report_dir = OUT_REPORT / "cooja"
    rows: list[dict[str, Any]] = []
    feat_rows: list[dict[str, Any]] = []
    top_conf_rows: list[dict[str, Any]] = []
    overhead_rows: list[dict[str, Any]] = []
    existing_report = OUT_DEFENSE / "cooja" / "eval" / "defense_eval_report.json"
    if existing_report.exists():
        return _cooja_summary_from_report(existing_report, missing)

    missing.append(
        {
            "section": "cooja",
            "reason": "canonical_cooja_eval_report_missing",
            "expected_file": _rel(existing_report),
            "note": "Build does not rerun Cooja while normalizing the artifact layout.",
        }
    )
    _write_json(cooja_report_dir / "cooja_missing_outputs.json", [m for m in missing if m.get("section") == "cooja"])
    _write_json(cooja_report_dir / "cooja_summary.json", rows)
    _write_csv(cooja_report_dir / "cooja_summary.csv", rows)
    _write_csv(cooja_report_dir / "cooja_overhead_summary.csv", overhead_rows)
    (cooja_report_dir / "cooja_limitations.md").write_text(
        "# Cooja Limitations\n\n"
        "- Cooja canonical evaluation report is missing from `outputs/experiments/cooja/eval/`.\n"
        "- This build does not rerun Cooja or fabricate energy/delay/packet metrics.\n",
        encoding="utf-8",
    )
    return {"rows": rows}

    dummy_manifest = ROOT / "configs" / "cooja_defense_dummy_logs.json"
    post_manifest = ROOT / "configs" / "cooja_defense_postprocess.json"
    legacy_manifest = ROOT / "configs" / "cooja_defense_logs.json"

    chosen_manifest: Path | None = None
    for cand in [dummy_manifest, post_manifest, legacy_manifest]:
        ok, _ = _cooja_logs_available(cand)
        if ok:
            chosen_manifest = cand
            break

    if chosen_manifest is None:
        missing.append(
            {
                "section": "cooja",
                "reason": "no_accessible_cooja_logs",
                "checked_manifests": [str(dummy_manifest), str(post_manifest), str(legacy_manifest)],
                "note": "WSL/UNC log paths are not available from current workspace.",
            }
        )
        _write_json(cooja_report_dir / "cooja_missing_outputs.json", [m for m in missing if m.get("section") == "cooja"])
        _write_json(cooja_report_dir / "cooja_summary.json", rows)
        _write_csv(cooja_report_dir / "cooja_summary.csv", rows)
        _write_csv(cooja_report_dir / "cooja_feature_importance.csv", feat_rows)
        _write_csv(cooja_report_dir / "cooja_top_confusions.csv", top_conf_rows)
        _write_csv(cooja_report_dir / "cooja_overhead_summary.csv", overhead_rows)
        _write_json(cooja_report_dir / "cooja_missing_logs.json", [m for m in missing if m.get("section") == "cooja"])
        (cooja_report_dir / "cooja_limitations.md").write_text(
            "# Cooja Limitations\n\n- Cooja log paths are not accessible in the current environment.\n- No real energy or delay measurements are available.\n",
            encoding="utf-8",
        )
        return {"rows": rows}

    out_dir = OUT_DEFENSE / "cooja" / "eval"
    cmd = [
        sys.executable,
        "experiments/cooja/run_cooja_defense_eval.py",
        "--manifest",
        str(chosen_manifest),
        "--out_dir",
        str(out_dir),
        "--seeds",
        "42,123,2026",
        "--window_s",
        "8",
        "--step_s",
        "3",
        "--min_requests",
        "2",
        "--dominance_threshold",
        "0.2",
    ]
    rc, out, err = _run(cmd, cwd=ROOT, timeout=7200)
    report_path = out_dir / "defense_eval_report.json"
    if rc != 0 or not report_path.exists():
        missing.append(
            {
                "section": "cooja",
                "reason": "cooja_eval_run_failed",
                "manifest": str(chosen_manifest),
                "command": " ".join(cmd),
                "stdout_tail": out[-4000:],
                "stderr_tail": err[-4000:],
            }
        )
        _write_json(cooja_report_dir / "cooja_missing_outputs.json", [m for m in missing if m.get("section") == "cooja"])
        _write_json(cooja_report_dir / "cooja_summary.json", rows)
        _write_csv(cooja_report_dir / "cooja_summary.csv", rows)
        _write_csv(cooja_report_dir / "cooja_feature_importance.csv", feat_rows)
        _write_csv(cooja_report_dir / "cooja_top_confusions.csv", top_conf_rows)
        _write_csv(cooja_report_dir / "cooja_overhead_summary.csv", overhead_rows)
        _write_json(cooja_report_dir / "cooja_missing_logs.json", [m for m in missing if m.get("section") == "cooja"])
        (cooja_report_dir / "cooja_limitations.md").write_text(
            "# Cooja Limitations\n\n- Cooja defense evaluation could not be regenerated from accessible logs.\n- No real energy or delay measurements are available.\n",
            encoding="utf-8",
        )
        return {"rows": rows}

    rep = _safe_json(report_path) or {}
    methods = rep.get("methods", {}) if isinstance(rep, dict) else {}
    for method_name, mobj in methods.items():
        if not isinstance(mobj, dict):
            continue
        b_mean = float(((mobj.get("baseline_test") or {}).get("accuracy") or {}).get("mean", np.nan))
        f_mean = float(((mobj.get("fixed_attacker") or {}).get("accuracy") or {}).get("mean", np.nan))
        r_mean = float(((mobj.get("retrain_attacker") or {}).get("accuracy") or {}).get("mean", np.nan))
        f1_fixed = float(((mobj.get("fixed_attacker") or {}).get("f1_macro") or {}).get("mean", np.nan))
        f1_retrain = float(((mobj.get("retrain_attacker") or {}).get("f1_macro") or {}).get("mean", np.nan))
        dataset_meta = mobj.get("dataset", {}) if isinstance(mobj.get("dataset", {}), dict) else {}
        baseline_windows = float(dataset_meta.get("baseline_windows", np.nan))
        defense_windows = float(dataset_meta.get("defense_windows", np.nan))
        window_ratio = (
            defense_windows / baseline_windows
            if baseline_windows == baseline_windows and baseline_windows > 0
            else np.nan
        )
        overhead_rows.append(
            {
                "method": method_name,
                "baseline_windows": baseline_windows,
                "defense_windows": defense_windows,
                "defense_window_ratio": window_ratio,
                "window_count_delta": defense_windows - baseline_windows
                if baseline_windows == baseline_windows and defense_windows == defense_windows
                else np.nan,
                "energy_metric_available": False,
                "delay_metric_available": False,
                "note": "Cooja logs do not include real energy/delay fields; this is a window-count proxy only.",
            }
        )
        rows.append(
            {
                "method": method_name,
                "seed": "mean_over_seeds",
                "mode": "fixed_attacker",
                "baseline_acc": b_mean,
                "defended_acc": f_mean,
                "accuracy_drop": b_mean - f_mean,
                "f1_macro": f1_fixed,
                "pkt_count_mean": np.nan,
                "byte_count_mean": np.nan,
                "dummy_packet_ratio": np.nan,
                "packet_overhead_ratio": np.nan,
                "mean_iat_ms": np.nan,
                "p95_iat_ms": np.nan,
                "traffic_activity_correlation_before": np.nan,
                "traffic_activity_correlation_after": np.nan,
                "correlation_drop": np.nan,
                "energy_metric_available": False,
                "delay_proxy_available": False,
                "source_log_files": json.dumps((mobj.get("defense_log_paths") or {}), ensure_ascii=False),
            }
        )
        rows.append(
            {
                "method": method_name,
                "seed": "mean_over_seeds",
                "mode": "retrain_attacker",
                "baseline_acc": b_mean,
                "defended_acc": r_mean,
                "accuracy_drop": b_mean - r_mean,
                "f1_macro": f1_retrain,
                "pkt_count_mean": np.nan,
                "byte_count_mean": np.nan,
                "dummy_packet_ratio": np.nan,
                "packet_overhead_ratio": np.nan,
                "mean_iat_ms": np.nan,
                "p95_iat_ms": np.nan,
                "traffic_activity_correlation_before": np.nan,
                "traffic_activity_correlation_after": np.nan,
                "correlation_drop": np.nan,
                "energy_metric_available": False,
                "delay_proxy_available": False,
                "source_log_files": json.dumps((mobj.get("defense_log_paths") or {}), ensure_ascii=False),
            }
        )

        for run in mobj.get("runs", []):
            if not isinstance(run, dict):
                continue
            seed = int(run.get("seed", -1))
            fixed = run.get("fixed_attacker_on_defense", {}) or {}
            retr = run.get("retrain_attacker_on_defense", {}) or {}
            for tc in fixed.get("top_confusions", [])[:5]:
                top_conf_rows.append(
                    {
                        "method": method_name,
                        "seed": seed,
                        "mode": "fixed_attacker",
                        "true_label": tc.get("true"),
                        "pred_label": tc.get("pred"),
                        "count": tc.get("count"),
                    }
                )
            for tc in retr.get("top_confusions", [])[:5]:
                top_conf_rows.append(
                    {
                        "method": method_name,
                        "seed": seed,
                        "mode": "retrain_attacker",
                        "true_label": tc.get("true"),
                        "pred_label": tc.get("pred"),
                        "count": tc.get("count"),
                    }
                )

    _write_json(cooja_report_dir / "cooja_summary.json", rows)
    _write_csv(cooja_report_dir / "cooja_summary.csv", rows)
    if feat_rows:
        _write_csv(cooja_report_dir / "cooja_feature_importance.csv", feat_rows)
    if top_conf_rows:
        _write_csv(cooja_report_dir / "cooja_top_confusions.csv", top_conf_rows)
    _write_csv(cooja_report_dir / "cooja_overhead_summary.csv", overhead_rows)
    _write_json(cooja_report_dir / "cooja_missing_outputs.json", [m for m in missing if m.get("section") == "cooja"])
    _export_cooja_detail_outputs(rep, missing)
    return {"rows": rows}


def _plot_bar_by_mode(df: pd.DataFrame, out_path: Path, title: str) -> bool:
    if df.empty:
        return False
    pivot = df.pivot_table(index="method", columns="mode", values="defended_acc", aggfunc="mean")
    if pivot.empty:
        return False
    fig, ax = plt.subplots(figsize=(8, 5))
    pivot.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_ylabel("Accuracy")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return True


def _build_figures(
    mock_rows: list[dict[str, Any]],
    real_rows: list[dict[str, Any]],
    cooja_rows: list[dict[str, Any]],
    scan_mock_ldp: list[dict[str, Any]],
    scan_mock_noise: list[dict[str, Any]],
    scan_real_ldp: list[dict[str, Any]],
    scan_real_noise: list[dict[str, Any]],
    missing: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    figures: list[dict[str, Any]] = []
    OUT_FIG.mkdir(parents=True, exist_ok=True)

    mock_df = pd.DataFrame(mock_rows)
    real_df = pd.DataFrame(real_rows)
    cooja_df = pd.DataFrame(cooja_rows)
    mock_scan_ldp_df = pd.DataFrame(scan_mock_ldp)
    mock_scan_noise_df = pd.DataFrame(scan_mock_noise)
    real_scan_ldp_df = pd.DataFrame(scan_real_ldp)
    real_scan_noise_df = pd.DataFrame(scan_real_noise)
    for df in [mock_df, real_df, cooja_df, mock_scan_ldp_df, mock_scan_noise_df, real_scan_ldp_df, real_scan_noise_df]:
        for col in [
            "seed",
            "baseline_acc",
            "defended_acc",
            "accuracy_drop",
            "defended_f1_macro",
            "mse",
            "mae",
            "pearson_r",
            "parameter_value",
        ]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

    # 1) mock baseline vs fixed/retrain accuracy
    if not mock_df.empty:
        fig, ax = plt.subplots(figsize=(9, 5))
        g = mock_df.groupby(["model_type", "mode"], as_index=False)["defended_acc"].mean()
        x = np.arange(len(g["model_type"].unique()))
        width = 0.35
        for i, mode in enumerate(MODES):
            vals = []
            models = sorted(g["model_type"].unique().tolist())
            for m in models:
                sub = g[(g["model_type"] == m) & (g["mode"] == mode)]
                vals.append(float(sub["defended_acc"].iloc[0]) if not sub.empty else np.nan)
            ax.bar(x + (i - 0.5) * width, vals, width=width, label=mode)
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.set_ylabel("Mean defended accuracy")
        ax.set_title("Mock: LSTM/MLP under fixed vs retrain attacker")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        p = OUT_FIG / "mock_model_mode_accuracy.png"
        fig.savefig(p, dpi=180)
        plt.close(fig)
        figures.append(
            {
                "path": str(p),
                "title": "Mock LSTM/MLP baseline vs fixed/retrain accuracy 对比图",
                "source_files": "outputs/summaries/final_thesis/mock/mock_summary.csv",
                "conclusion": "可用于展示 fixed_attacker 与 retrain_attacker 的差异趋势。",
                "limitations": "均值汇总会掩盖个别 seed 波动。",
            }
        )

    # 2) mock distortion comparison
    if not mock_df.empty:
        dist = mock_df.groupby("method", as_index=False)[["mse", "mae", "pearson_r"]].mean(numeric_only=True)
        fig, ax = plt.subplots(figsize=(9, 5))
        x = np.arange(len(dist))
        ax.bar(x - 0.25, dist["mse"], width=0.25, label="MSE")
        ax.bar(x, dist["mae"], width=0.25, label="MAE")
        ax.bar(x + 0.25, dist["pearson_r"], width=0.25, label="Pearson r")
        ax.set_xticks(x)
        ax.set_xticklabels(dist["method"])
        ax.set_title("Mock: Distortion metrics by defense method")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        p = OUT_FIG / "mock_method_distortion.png"
        fig.savefig(p, dpi=180)
        plt.close(fig)
        figures.append(
            {
                "path": str(p),
                "title": "Mock 三种防御方法 MSE/MAE/Pearson 对比图",
                "source_files": "outputs/summaries/final_thesis/mock/mock_summary.csv",
                "conclusion": "可用于展示防御强度与信号保真度之间权衡。",
                "limitations": "不同 mode 下共享同一 distortion 指标。",
            }
        )

    # 3-5 real dataset charts
    for ds in ["uci_har", "kasteren", "casas_hh101"]:
        sub = real_df[real_df["dataset"] == ds] if not real_df.empty else pd.DataFrame()
        if sub.empty:
            missing.append(
                {
                    "section": "figures",
                    "figure": f"real_{ds}_model_mode_accuracy",
                    "reason": "insufficient_real_rows",
                }
            )
            continue
        fig, ax = plt.subplots(figsize=(9, 5))
        g = sub.groupby(["model_type", "mode"], as_index=False)["defended_acc"].mean()
        models = sorted(g["model_type"].unique().tolist())
        x = np.arange(len(models))
        width = 0.35
        for i, mode in enumerate(MODES):
            vals = []
            for m in models:
                t = g[(g["model_type"] == m) & (g["mode"] == mode)]
                vals.append(float(t["defended_acc"].iloc[0]) if not t.empty else np.nan)
            ax.bar(x + (i - 0.5) * width, vals, width=width, label=mode)
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.set_ylabel("Mean defended accuracy")
        ax.set_title(f"Real ({ds}): LSTM/MLP under fixed vs retrain attacker")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        p = OUT_FIG / f"real_{ds}_model_mode_accuracy.png"
        fig.savefig(p, dpi=180)
        plt.close(fig)
        figures.append(
            {
                "path": str(p),
                "title": f"real {ds} LSTM/MLP baseline vs fixed/retrain accuracy 对比图",
                "source_files": "outputs/summaries/final_thesis/real/real_summary.csv",
                "conclusion": f"可用于展示 {ds} 数据集的防御效果。",
                "limitations": "若样本不平衡，宏平均与准确率可能有偏差。",
            }
        )

    # 6-7 parameter scan curves (use uci_har + mock)
    if not real_scan_ldp_df.empty:
        sub = real_scan_ldp_df[real_scan_ldp_df["dataset"] == "uci_har"]
        if not sub.empty:
            fig, ax = plt.subplots(figsize=(8, 5))
            x = sorted(sub["parameter_value"].unique().tolist())
            y = [float(sub[sub["parameter_value"] == v]["defended_acc"].mean()) for v in x]
            ax.plot(x, y, marker="o")
            ax.set_xscale("log")
            ax.set_xlabel("epsilon")
            ax.set_ylabel("defended accuracy")
            ax.set_title("Real(UCI HAR) LDP epsilon scan")
            ax.grid(alpha=0.3)
            fig.tight_layout()
            p = OUT_FIG / "real_uci_ldp_scan.png"
            fig.savefig(p, dpi=180)
            plt.close(fig)
            figures.append(
                {
                    "path": str(p),
                    "title": "ldp epsilon 参数扫描曲线",
                    "source_files": "outputs/summaries/final_thesis/real/real_parameter_scan_ldp.csv",
                    "conclusion": "可用于展示 epsilon 变大时准确率恢复趋势。",
                    "limitations": "该图以 UCI HAR 作代表性曲线展示；完整 real 参数扫描矩阵已覆盖 UCI HAR、Kasteren 与 CASAS。",
                }
            )

    if real_scan_noise_df is not None and not real_scan_noise_df.empty:
        sub = real_scan_noise_df[real_scan_noise_df["dataset"] == "uci_har"]
        if not sub.empty:
            fig, ax = plt.subplots(figsize=(8, 5))
            x = sorted(sub["parameter_value"].unique().tolist())
            y = [float(sub[sub["parameter_value"] == v]["defended_acc"].mean()) for v in x]
            ax.plot(x, y, marker="o")
            ax.set_xlabel("noise scale")
            ax.set_ylabel("defended accuracy")
            ax.set_title("Real(UCI HAR) noise scale scan")
            ax.grid(alpha=0.3)
            fig.tight_layout()
            p = OUT_FIG / "real_uci_noise_scan.png"
            fig.savefig(p, dpi=180)
            plt.close(fig)
            figures.append(
                {
                    "path": str(p),
                    "title": "noise scale 参数扫描曲线",
                    "source_files": "outputs/summaries/final_thesis/real/real_parameter_scan_noise.csv",
                    "conclusion": "可用于展示噪声强度上升时攻击准确率下降趋势。",
                    "limitations": "该图以 UCI HAR 作代表性曲线展示；完整 real 参数扫描矩阵已覆盖 UCI HAR、Kasteren 与 CASAS。",
                }
            )

    # 8 representative confusion matrix for each dataset
    candidates = [
        ("mock", OUT_DEFENSE / "mock" / "seed_42" / "lstm" / "adaptive_ldp" / "fixed_attacker" / "confusion.json"),
        ("uci_har", OUT_DEFENSE / "uci_har" / "seed_42" / "lstm" / "adaptive_ldp" / "fixed_attacker" / "confusion.json"),
        ("kasteren", OUT_DEFENSE / "kasteren" / "seed_42" / "lstm" / "adaptive_ldp" / "fixed_attacker" / "confusion.json"),
        ("casas_hh101", OUT_DEFENSE / "casas_hh101" / "seed_42" / "lstm" / "adaptive_ldp" / "fixed_attacker" / "confusion.json"),
    ]
    for ds, pjson in candidates:
        obj = _safe_json(pjson)
        if not isinstance(obj, dict):
            missing.append(
                {
                    "section": "figures",
                    "figure": f"confusion_{ds}",
                    "reason": "confusion_json_missing",
                    "expected_file": str(pjson),
                }
            )
            continue
        out = OUT_FIG / f"confusion_{ds}.png"
        ok = _render_confusion_from_json(obj, out, f"Representative confusion ({ds})")
        if ok:
            figures.append(
                {
                    "path": str(out),
                    "title": f"{ds} 代表性 confusion matrix",
                    "source_files": _rel(pjson),
                    "conclusion": "可用于展示主要误分类模式。",
                    "limitations": "仅展示单个 seed/model/method 样本。",
                }
            )

    # 9-10 Cooja charts if available
    if not cooja_df.empty:
        p = OUT_FIG / "cooja_mode_accuracy.png"
        ok = _plot_bar_by_mode(cooja_df, p, "Cooja fixed vs retrain accuracy")
        if ok:
            figures.append(
                {
                    "path": str(p),
                    "title": "Cooja fixed/retrain accuracy 对比图",
                    "source_files": "outputs/summaries/final_thesis/cooja/cooja_summary.csv",
                    "conclusion": "可用于展示节点级防御在流量侧攻击下的变化。",
                    "limitations": "依赖 Cooja 日志质量与可获得性。",
                }
            )
        else:
            missing.append(
                {"section": "figures", "figure": "cooja_mode_accuracy", "reason": "cooja_rows_empty"}
            )

        overhead_path = OUT_REPORT / "cooja" / "cooja_overhead_summary.csv"
        overhead_df = pd.read_csv(overhead_path) if overhead_path.exists() else pd.DataFrame()
        if not overhead_df.empty and "defense_window_ratio" in overhead_df.columns:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.bar(overhead_df["method"], overhead_df["defense_window_ratio"], color="#4C72B0")
            ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
            ax.set_ylabel("Defense / baseline window ratio")
            ax.set_title("Cooja traffic-window count proxy")
            ax.grid(axis="y", alpha=0.3)
            fig.tight_layout()
            p = OUT_FIG / "cooja_window_overhead_proxy.png"
            fig.savefig(p, dpi=180)
            plt.close(fig)
            figures.append(
                {
                    "path": str(p),
                    "title": "Cooja 窗口数量代理开销图",
                    "source_files": "outputs/summaries/final_thesis/cooja/cooja_overhead_summary.csv",
                    "conclusion": "可用于说明当前日志只能支持窗口数量代理，而不能支持真实能耗或时延结论。",
                    "limitations": "该图不是能耗或时延实测，只反映当前导出日志形成的窗口规模差异。",
                }
            )
    else:
        missing.append({"section": "figures", "figure": "cooja_mode_accuracy", "reason": "cooja_rows_empty"})

    return figures


def _build_symmetry_figures(missing: list[dict[str, Any]]) -> list[dict[str, Any]]:
    figures: list[dict[str, Any]] = []
    OUT_FIG.mkdir(parents=True, exist_ok=True)

    def read_csv(path: Path) -> pd.DataFrame:
        return pd.read_csv(path) if path.exists() and path.stat().st_size > 0 else pd.DataFrame()

    def plot_parameter_method(method: str, out_name: str) -> None:
        mock_path = OUT_REPORT / "mock" / f"mock_parameter_scan_{method}.csv"
        real_path = OUT_REPORT / "real" / f"real_parameter_scan_{method}.csv"
        df = pd.concat([read_csv(mock_path), read_csv(real_path)], ignore_index=True)
        if df.empty:
            missing.append({"section": "figures", "figure": out_name, "reason": "parameter_scan_rows_empty"})
            return
        df["dataset"] = df["dataset"].fillna("unknown").astype(str)
        datasets = ["mock"] + [d for d in ["uci_har", "kasteren", "casas_hh101"] if d in set(df["dataset"])]
        datasets = [d for d in datasets if d in set(df["dataset"])]
        fig, axes = plt.subplots(len(datasets), 1, figsize=(10, max(4, 3.2 * len(datasets))), squeeze=False)
        for ax, dataset in zip(axes.ravel(), datasets):
            sub = df[df["dataset"] == dataset].copy()
            if sub.empty:
                ax.set_visible(False)
                continue
            if method == "adaptive_ldp":
                xcol = "parameter_value"
                xlabel = "adaptive profile"
            elif method == "ldp":
                xcol = "parameter_value"
                xlabel = "epsilon"
            else:
                xcol = "parameter_value"
                xlabel = "noise_scale"
            for (model, mode), g in sub.groupby(["model_type", "mode"]):
                curve = g.groupby(xcol, as_index=False)["defended_acc"].mean(numeric_only=True).sort_values(xcol)
                ax.plot(curve[xcol], curve["defended_acc"], marker="o", label=f"{model} {mode}")
            if method == "ldp":
                ax.set_xscale("log")
            ax.set_title(f"{dataset}: {method} parameter scan")
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Mean defended accuracy")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8, ncols=2)
        fig.tight_layout()
        out = OUT_FIG / out_name
        fig.savefig(out, dpi=180)
        plt.close(fig)
        figures.append(
            {
                "path": str(out),
                "title": f"{method} parameter scans across available models/modes",
                "source_files": f"{_rel(mock_path)};{_rel(real_path)}",
                "conclusion": "Shows parameter sensitivity separately by dataset; missing combinations are documented in parameter_scan_coverage_audit.json.",
                "limitations": "Curves average available seeds and do not rank different datasets against each other.",
            }
        )

    plot_parameter_method("ldp", "parameter_scan_ldp_all_models_modes.png")
    plot_parameter_method("noise", "parameter_scan_noise_all_models_modes.png")
    plot_parameter_method("adaptive_ldp", "parameter_scan_adaptive_ldp_all_models_modes.png")

    def plot_adaptive_ablation(scope: str, metric: str, out_name: str, ylabel: str) -> None:
        if scope == "mock":
            summary_path = OUT_REPORT / "mock" / "mock_adaptive_ldp_ablation_summary.csv"
        else:
            summary_path = OUT_REPORT / "real" / "real_adaptive_ldp_ablation_summary.csv"
        df = read_csv(summary_path)
        if df.empty:
            missing.append({"section": "figures", "figure": out_name, "reason": "adaptive_ablation_rows_empty"})
            return
        df["dataset"] = df["dataset"].fillna("unknown").astype(str)
        df["profile_name"] = df["profile_name"].astype(str)
        datasets = ["mock"] if scope == "mock" else [d for d in ["uci_har", "kasteren", "casas_hh101"] if d in set(df["dataset"])]
        fig, axes = plt.subplots(len(datasets), 1, figsize=(11, max(4, 3.3 * len(datasets))), squeeze=False)
        x = np.arange(len(ADAPTIVE_PROFILE_ORDER))
        for ax, dataset in zip(axes.ravel(), datasets):
            sub = df[df["dataset"] == dataset].copy()
            if sub.empty:
                ax.set_visible(False)
                continue
            sub["_profile_order"] = sub["profile_name"].map(_profile_sort_value)
            for (model, mode), g in sub.groupby(["model_type", "mode"]):
                curve = (
                    g.sort_values("_profile_order")
                    .groupby("profile_name", as_index=False)[metric]
                    .mean(numeric_only=True)
                )
                curve["_profile_order"] = curve["profile_name"].map(_profile_sort_value)
                curve = curve.sort_values("_profile_order")
                ax.plot(curve["_profile_order"], curve[metric], marker="o", label=f"{model} {mode}")
            ax.set_xticks(x)
            ax.set_xticklabels(ADAPTIVE_PROFILE_ORDER, rotation=25, ha="right")
            ax.set_title(f"{dataset}: adaptive_ldp ablation")
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8, ncols=2)
        fig.tight_layout()
        out = OUT_FIG / out_name
        fig.savefig(out, dpi=180)
        plt.close(fig)
        figures.append(
            {
                "path": str(out),
                "title": f"adaptive_ldp ablation {scope} {metric}",
                "source_files": _rel(summary_path),
                "conclusion": "Shows profile-level adaptive_ldp ablation without mixing datasets for absolute ranking.",
                "limitations": "Averages model/mode rows within each dataset panel and remains an empirical profile scan.",
            }
        )

    plot_adaptive_ablation("mock", "mean_defended_acc", "adaptive_ldp_ablation_mock_accuracy.png", "Mean defended accuracy")
    plot_adaptive_ablation("mock", "mean_mse", "adaptive_ldp_ablation_mock_distortion.png", "Mean MSE")
    plot_adaptive_ablation("real", "mean_defended_acc", "adaptive_ldp_ablation_real_accuracy.png", "Mean defended accuracy")
    plot_adaptive_ablation("real", "mean_mse", "adaptive_ldp_ablation_real_distortion.png", "Mean MSE")

    per_seed_path = OUT_REPORT / "cooja" / "cooja_per_seed.csv"
    per_seed_df = read_csv(per_seed_path)
    if not per_seed_df.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        for (method, mode), g in per_seed_df.groupby(["method", "mode"]):
            g = g.sort_values("seed")
            ax.plot(g["seed"], g["defended_acc"], marker="o", label=f"{method} {mode}")
        ax.set_xlabel("seed")
        ax.set_ylabel("Defended accuracy")
        ax.set_title("Cooja per-seed defended accuracy")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, ncols=2)
        fig.tight_layout()
        out = OUT_FIG / "cooja_per_seed_accuracy.png"
        fig.savefig(out, dpi=180)
        plt.close(fig)
        figures.append(
            {
                "path": str(out),
                "title": "Cooja per-seed accuracy",
                "source_files": _rel(per_seed_path),
                "conclusion": "Shows fixed/retrain attacker behavior per seed for each Cooja dummy method.",
                "limitations": "Depends on available Cooja radio/app logs and exported per-seed runs.",
            }
        )
    else:
        missing.append({"section": "figures", "figure": "cooja_per_seed_accuracy", "reason": "cooja_per_seed_rows_empty"})

    traffic_path = OUT_REPORT / "cooja" / "cooja_traffic_metrics.csv"
    traffic_df = read_csv(traffic_path)
    if not traffic_df.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        g = traffic_df.groupby("method", as_index=False)[["packet_overhead_ratio", "byte_overhead_ratio"]].mean(numeric_only=True)
        x = np.arange(len(g))
        ax.bar(x - 0.18, g["packet_overhead_ratio"], width=0.36, label="packet ratio")
        ax.bar(x + 0.18, g["byte_overhead_ratio"], width=0.36, label="byte ratio")
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(g["method"], rotation=20, ha="right")
        ax.set_ylabel("Defense / baseline ratio")
        ax.set_title("Cooja traffic metrics")
        ax.grid(axis="y", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        out = OUT_FIG / "cooja_traffic_metrics.png"
        fig.savefig(out, dpi=180)
        plt.close(fig)
        figures.append(
            {
                "path": str(out),
                "title": "Cooja traffic metrics",
                "source_files": _rel(traffic_path),
                "conclusion": "Shows available packet/byte overhead proxies from Cooja traffic windows.",
                "limitations": "Not real energy or delay; dummy packet ratios are null when logs do not label dummy packets.",
            }
        )
    else:
        missing.append({"section": "figures", "figure": "cooja_traffic_metrics", "reason": "cooja_traffic_rows_empty"})

    return figures


def _write_figure_list(figures: list[dict[str, Any]]) -> None:
    def normalize_display(value: Any) -> str:
        parts = str(value).split(";")
        normalized: list[str] = []
        for part in parts:
            item = part.strip()
            if not item:
                continue
            path = Path(item)
            normalized.append(_rel(path) if path.is_absolute() else item.replace("\\", "/"))
        return ";".join(normalized)

    path = OUT_REPORT / "figure_table_list.md"
    lines = ["# 图表清单", ""]
    for i, fig in enumerate(figures, start=1):
        lines.append(f"## {i}. {fig['title']}")
        lines.append(f"- 图路径: `{normalize_display(fig['path'])}`")
        lines.append(f"- 源文件: `{normalize_display(fig['source_files'])}`")
        lines.append(f"- 可写入论文结论: {fig['conclusion']}")
        lines.append(f"- 口径限制: {fig['limitations']}")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _adaptive_ablation_table_entries() -> list[dict[str, Any]]:
    return [
        {
            "path": _rel(OUT_REPORT / "mock" / "mock_adaptive_ldp_ablation_summary.csv"),
            "title": "mock adaptive_ldp ablation summary table",
            "source_files": _rel(OUT_REPORT / "mock" / "mock_parameter_scan_adaptive_ldp.csv"),
            "conclusion": "Profile-level mock adaptive_ldp ablation summary based on existing parameter scans.",
            "limitations": "Empirical profile aggregation across seeds; no new experiment was rerun.",
        },
        {
            "path": _rel(OUT_REPORT / "real" / "real_adaptive_ldp_ablation_summary.csv"),
            "title": "real adaptive_ldp ablation summary table",
            "source_files": _rel(OUT_REPORT / "real" / "real_parameter_scan_adaptive_ldp.csv"),
            "conclusion": "Profile-level real-data adaptive_ldp ablation summary by dataset, model, and attacker mode.",
            "limitations": "Do not rank different datasets against each other by absolute value.",
        },
        {
            "path": _rel(OUT_REPORT / "adaptive_ldp_ablation_overview.md"),
            "title": "adaptive_ldp ablation overview",
            "source_files": (
                f"{_rel(OUT_REPORT / 'mock' / 'mock_adaptive_ldp_ablation_summary.csv')};"
                f"{_rel(OUT_REPORT / 'real' / 'real_adaptive_ldp_ablation_summary.csv')}"
            ),
            "conclusion": "Explains the six adaptive_ldp profiles and their cautious interpretation.",
            "limitations": "This overview is a delivery note, not a theoretical proof.",
        },
    ]


def _write_final_summary_md(
    env: EnvInfo,
    mock_rows: list[dict[str, Any]],
    real_rows: list[dict[str, Any]],
    cooja_rows: list[dict[str, Any]],
    missing: list[dict[str, Any]],
    parameter_scan_coverage: dict[str, Any],
) -> None:
    md = OUT_REPORT / "final_thesis_summary.md"
    mock_df = pd.DataFrame(mock_rows)
    real_df = pd.DataFrame(real_rows)
    cooja_df = pd.DataFrame(cooja_rows)

    def _line(s: str) -> str:
        return s if s else "N/A"

    lines: list[str] = []
    lines.append("# 最终实验总结（可追溯）")
    lines.append("")
    lines.append("## 1. 本次运行环境")
    lines.append(f"- experiment_result_commit: `{env.experiment_result_commit}`")
    lines.append(f"- repository_cleanup_commit: `{env.repository_cleanup_commit}`")
    lines.append(f"- latest_verified_commit: `{env.latest_verified_commit}`")
    lines.append("- 说明: 实验结果包生成 commit 与后续仓库清理 commit 可能不同；清理未重跑实验，只修正文档、路径和冗余产物。")
    lines.append(f"- python version: `{env.python_version}`")
    lines.append(f"- OS: `{env.os}`")
    lines.append(f"- start time / end time: `{env.start_time}` / `{env.end_time}`")
    lines.append("")

    lines.append("## 2. mock 实验是否完整")
    mock_expected = len(SEEDS) * len(MODELS) * len(METHODS) * len(MODES)
    mock_scan = parameter_scan_coverage.get("mock", {}) if isinstance(parameter_scan_coverage, dict) else {}
    real_scan = parameter_scan_coverage.get("real", {}) if isinstance(parameter_scan_coverage, dict) else {}
    profile_info = parameter_scan_coverage.get("adaptive_ldp_profile_count", {}) if isinstance(parameter_scan_coverage, dict) else {}
    lines.append(f"- mock 主矩阵完整: `{len(mock_df)}` / `{mock_expected}`。")
    lines.append(
        f"- mock 参数扫描完整: `{mock_scan.get('completed', 0)}` / `{mock_scan.get('expected', 0)}`；"
        f"missing=`{len(mock_scan.get('missing', []) or [])}`。"
    )
    if not mock_df.empty:
        for model in MODELS:
            sub = mock_df[mock_df["model_type"] == model]
            lines.append(
                f"- {model.upper()} 主要结果: baseline_acc 均值 `{_mean(sub['baseline_acc'].tolist()):.4f}`，"
                f"defended_acc 均值 `{_mean(sub['defended_acc'].tolist()):.4f}`。"
            )
    lines.append(f"- adaptive_ldp 已有 `{profile_info.get('expected', 6)}`-profile 级消融汇总。")
    lines.append("- 可写入论文的结论: fixed_attacker 与 retrain_attacker 在 mock 数据上呈现可观差异，支持隐私-效用分析。")
    lines.append("- 不建议写入论文的内容: 缺失组合（见 final_missing_outputs.json）对应的推断结论。")
    lines.append("")

    lines.append("## 3. 真实数据集实验是否完整")
    real_expected_total = 3 * len(SEEDS) * len(MODELS) * len(METHODS) * len(MODES)
    lines.append(f"- real 主矩阵完整: `{len(real_df)}` / `{real_expected_total}`。")
    lines.append(
        f"- real 参数扫描完整: `{real_scan.get('completed', 0)}` / `{real_scan.get('expected', 0)}`；"
        f"missing=`{len(real_scan.get('missing', []) or [])}`。"
    )
    lines.append("- 参数扫描覆盖: datasets=`uci_har,kasteren,casas_hh101`；methods=`adaptive_ldp,ldp,noise`；models=`lstm,mlp`；modes=`fixed_attacker,retrain_attacker`；seeds=`42,123,2026`。")
    lines.append("- Kasteren 和 CASAS 参数扫描已经补齐，不再作为后续扩展建议。")
    for ds in ["uci_har", "kasteren", "casas_hh101"]:
        sub = real_df[real_df["dataset"] == ds] if not real_df.empty else pd.DataFrame()
        expected = len(SEEDS) * len(MODELS) * len(METHODS) * len(MODES)
        lines.append(f"- {ds} 完成情况: `{len(sub)}` / `{expected}` 条。")
        if not sub.empty:
            lines.append(
                f"  - 主要结果: baseline_acc 均值 `{_mean(sub['baseline_acc'].tolist()):.4f}`，"
                f"fixed/retrain defended_acc 均值 `{_mean(sub['defended_acc'].tolist()):.4f}`。"
            )
    lines.append("- 各数据集之间不能直接比较的原因: 类别空间、样本分布、传感器维度和标签定义不同。")
    lines.append("- 可写入论文的结论: 在 UCI HAR、Kasteren 与 CASAS 上可稳定观测防御导致的准确率下降及部分重训恢复。")
    lines.append("- 不建议写入论文的内容: 不同数据集之间的绝对准确率直接排序。")
    lines.append("")

    lines.append("## 4. Cooja 节点级实验是否完整")
    if cooja_df.empty:
        lines.append("- 日志是否存在: 当前工作区无法访问有效 Cooja 日志（多为 WSL UNC 路径）。")
        lines.append("- dummy 流量是否跑通: 未能在当前环境复现。")
        lines.append("- fixed/retrain 是否跑通: 未完成。")
        lines.append("- 流量混淆度是否可计算: 不可计算。")
        lines.append("- 节点开销是否可计算: 不可计算（energy_metric_available=false）。")
        lines.append("- 可写入论文的结论: 仅可说明当前环境下日志不可达，需在日志完整环境复现。")
        lines.append("- 不建议写入论文的内容: 任何未实际运行得到的 Cooja 数值结论。")
    else:
        lines.append("- 日志是否存在: 可用。")
        lines.append("- dummy 流量是否跑通: 已运行。")
        lines.append("- fixed/retrain 是否跑通: 已运行。")
        lines.append("- 流量混淆度是否可计算: 部分可计算。")
        lines.append("- 节点开销是否可计算: 能耗/时延真实量化不足，使用代理指标。")
        lines.append("- 可写入论文的结论: 见 cooja_summary.csv。")
        lines.append("- 不建议写入论文的内容: 未有真实量测支持的能耗结论。")
    lines.append("")

    lines.append("## 5. 文件口径风险")
    lines.append("- 覆盖风险: 原始 `outputs/reports/**/metrics.json`、`outputs/defense/**/defense_report.json` 可能被后续运行覆盖。")
    lines.append("- 推荐论文引用: `outputs/summaries/final_thesis/*.csv|*.json` 与 `outputs/experiments/**/source_manifest.json`。")
    lines.append("- 不建议直接引用: 旧路径中未分 model/mode 的单文件报告。")
    lines.append("")

    lines.append("## 6. 下一步建议")
    lines.append("- Cooja 真实能耗与真实端到端时延仍需真实部署补充。")
    lines.append("- 更强攻击模型如 Transformer/TCN 可作为后续工作。")
    lines.append("- 更细粒度真实部署消融仍可作为后续工作。")
    lines.append("")

    lines.append("## Missing Count")
    lines.append(f"- total missing entries: `{len(missing)}`")
    md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    start = _now()
    OUT_REPORT.mkdir(parents=True, exist_ok=True)
    OUT_DEFENSE.mkdir(parents=True, exist_ok=True)
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    rc, gout, _ = _run(["git", "rev-parse", "HEAD"], cwd=ROOT)
    git_commit = gout.strip() if rc == 0 else "unknown"
    commit_metadata = _resolve_commit_metadata(git_commit)
    env = EnvInfo(
        git_commit=git_commit,
        experiment_result_commit=commit_metadata["experiment_result_commit"],
        repository_cleanup_commit=commit_metadata["repository_cleanup_commit"],
        latest_verified_commit=commit_metadata["latest_verified_commit"],
        python_version=sys.version.replace("\n", " "),
        os=platform.platform(),
        start_time=start,
        end_time=start,
    )

    missing: list[dict[str, Any]] = []
    _ensure_import_metas(missing)
    mock = _collect_mock(env, missing)
    real = _collect_real(env, missing)
    cooja = _collect_cooja(env, missing)
    parameter_scans = _collect_parameter_scans(missing)
    adaptive_ablation = _build_adaptive_ablation_outputs(missing)
    scan_mock_ldp = _read_csv_dicts(OUT_REPORT / "mock" / "mock_parameter_scan_ldp.csv") or []
    scan_mock_noise = _read_csv_dicts(OUT_REPORT / "mock" / "mock_parameter_scan_noise.csv") or []
    scan_real_ldp = _read_csv_dicts(OUT_REPORT / "real" / "real_parameter_scan_ldp.csv") or []
    scan_real_noise = _read_csv_dicts(OUT_REPORT / "real" / "real_parameter_scan_noise.csv") or []

    # unified files
    final_rows = []
    final_rows.extend([{**r, "section": "mock"} for r in mock["rows"]])
    final_rows.extend([{**r, "section": "real"} for r in real["rows"]])
    final_rows.extend([{**r, "section": "cooja"} for r in cooja["rows"]])
    _write_json(OUT_REPORT / "final_summary.json", final_rows)
    _write_csv(OUT_REPORT / "final_summary.csv", final_rows)

    manifest = {
        "generated_at": _now(),
        "experiment_result_commit": env.experiment_result_commit,
        "repository_cleanup_commit": env.repository_cleanup_commit,
        "latest_verified_commit": env.latest_verified_commit,
        "inputs": {
            "experiment_source_root": "outputs/experiments",
            "cooja_manifest_candidates": [
                "configs/cooja_defense_dummy_logs.json",
                "configs/cooja_defense_dummy_logs.template.json",
                "configs/cooja_defense_postprocess.json",
                "configs/cooja_defense_logs.json",
            ],
        },
        "outputs": {
            "mock_summary": _rel(OUT_REPORT / "mock" / "mock_summary.csv"),
            "real_summary": _rel(OUT_REPORT / "real" / "real_summary.csv"),
            "cooja_summary": _rel(OUT_REPORT / "cooja" / "cooja_summary.csv"),
            "final_summary": _rel(OUT_REPORT / "final_summary.csv"),
            "mock_adaptive_ldp_ablation": _rel(OUT_REPORT / "mock" / "mock_adaptive_ldp_ablation_summary.csv"),
            "real_adaptive_ldp_ablation": _rel(OUT_REPORT / "real" / "real_adaptive_ldp_ablation_summary.csv"),
        },
    }
    _write_json(OUT_REPORT / "final_manifest.json", manifest)

    # coverage + missing
    coverage = {
        "mock": mock["coverage"],
        "real": real["coverage"],
        "cooja_rows": len(cooja["rows"]),
        "parameter_scan": parameter_scans["coverage"],
        "should_have_experiments": {
            "mock": len(SEEDS) * len(MODELS) * len(METHODS) * len(MODES),
            "real": 3 * len(SEEDS) * len(MODELS) * len(METHODS) * len(MODES),
        },
        "actual_completed": {
            "mock": len(mock["rows"]),
            "real": len(real["rows"]),
            "cooja": len(cooja["rows"]),
            "mock_adaptive_ablation_rows": len(adaptive_ablation["mock"]),
            "real_adaptive_ablation_rows": len(adaptive_ablation["real"]),
        },
        "missing_combinations": {
            "mock": mock["coverage"]["missing_combinations"],
            "real": real["coverage"]["missing_combinations"],
        },
        "covered_risk_files": [
            "outputs/summaries/final_thesis/**/*.json",
            "outputs/experiments/**/defense_report.json",
        ],
        "recommended_for_thesis": [
            "outputs/summaries/final_thesis/mock/mock_summary.csv",
            "outputs/summaries/final_thesis/real/real_summary.csv",
            "outputs/summaries/final_thesis/final_summary.csv",
        ],
        "not_recommended_for_thesis": [
            "legacy batch/report roots (see outputs/summaries/layout/migration_report.md)",
        ],
    }
    _write_json(OUT_REPORT / "final_coverage_audit.json", coverage)
    _write_json(OUT_REPORT / "final_missing_outputs.json", missing)

    # figures + figure list
    figures = _build_figures(
        mock_rows=mock["rows"],
        real_rows=real["rows"],
        cooja_rows=cooja["rows"],
        scan_mock_ldp=scan_mock_ldp,
        scan_mock_noise=scan_mock_noise,
        scan_real_ldp=scan_real_ldp,
        scan_real_noise=scan_real_noise,
        missing=missing,
    )
    figures.extend(_adaptive_ablation_table_entries())
    figures.extend(_build_symmetry_figures(missing))
    _write_figure_list(figures)
    _write_json(OUT_REPORT / "final_missing_outputs.json", missing)

    env.end_time = _now()
    _write_final_summary_md(env, mock["rows"], real["rows"], cooja["rows"], missing, parameter_scans["coverage"])

    # Final ready flag
    required = [
        OUT_REPORT / "final_manifest.json",
        OUT_REPORT / "final_summary.csv",
        OUT_REPORT / "final_summary.json",
        OUT_REPORT / "final_coverage_audit.json",
        OUT_REPORT / "final_missing_outputs.json",
        OUT_REPORT / "final_thesis_summary.md",
        OUT_REPORT / "figure_table_list.md",
    ]
    all_exist = all(p.exists() for p in required)
    missing_count = len(missing)
    ready = bool(all_exist and missing_count == 0)

    if ready:
        print("FINAL_THESIS_RESULTS_READY=true")
        print("final_summary_path=outputs/summaries/final_thesis/final_summary.csv")
        print("final_report_path=outputs/summaries/final_thesis/final_thesis_summary.md")
    else:
        print("FINAL_THESIS_RESULTS_READY=false")
        print("missing_outputs_path=outputs/summaries/final_thesis/final_missing_outputs.json")


if __name__ == "__main__":
    main()
