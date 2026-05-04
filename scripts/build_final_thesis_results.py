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
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_REPORT = ROOT / "outputs" / "reports" / "final_thesis"
OUT_DEFENSE = ROOT / "outputs" / "defense" / "final_thesis"
OUT_FIG = ROOT / "outputs" / "figures" / "final_thesis"
TMP_DIR = OUT_REPORT / "_tmp"

SEEDS = [42, 123, 2026]
MODELS = ["lstm", "mlp"]
METHODS = ["adaptive_ldp", "ldp", "noise"]
MODES = ["fixed_attacker", "retrain_attacker"]


@dataclass
class EnvInfo:
    git_commit: str
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
        (
            "uci_har",
            ROOT / "data" / "processed" / "imports" / "uci_har" / "meta.json",
            [sys.executable, "experiments/real_public/run_import_uci_har.py", "--config", "configs/default.yaml", "--auto-download"],
        ),
        (
            "kasteren",
            ROOT / "data" / "processed" / "imports" / "kasteren" / "meta.json",
            [sys.executable, "experiments/real_public/run_import_kasteren.py", "--config", "configs/default.yaml", "--auto-download"],
        ),
        (
            "casas_hh101",
            ROOT / "data" / "processed" / "imports" / "casas_hh101" / "meta.json",
            [sys.executable, "experiments/real_public/run_import_casas.py", "--config", "configs/default.yaml", "--home", "hh101", "--auto-download"],
        ),
    ]
    for ds, meta, cmd in imports:
        if meta.exists():
            continue
        rc, out, err = _run(cmd, cwd=ROOT, timeout=7200)
        if rc != 0 or not meta.exists():
            missing.append(
                {
                    "section": "real_imports",
                    "dataset": ds,
                    "reason": "import_meta_missing_or_import_failed",
                    "expected_file": str(meta),
                    "command": " ".join(cmd),
                    "stdout_tail": out[-4000:],
                    "stderr_tail": err[-4000:],
                }
            )


def _collect_mock(env: EnvInfo, missing: list[dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    top_conf_rows: list[dict[str, Any]] = []
    scan_ldp_rows: list[dict[str, Any]] = []
    scan_noise_rows: list[dict[str, Any]] = []
    found_keys: set[tuple[int, str, str, str]] = set()

    for seed in SEEDS:
        for method in METHODS:
            base = ROOT / "outputs" / "defense" / "full_multiseed" / f"seed_{seed}" / method
            report_path = base / "defense_report.json"
            rep = _safe_json(report_path) or {}
            dist = rep.get("distortion", {}) if isinstance(rep, dict) else {}

            for model in MODELS:
                baseline_path = base / "json_reports" / f"{model}_baseline_confusion_test.json"
                fixed_path = base / "json_reports" / f"{model}_defended_confusion_test_fixed_attacker.json"
                retrain_path = base / "json_reports" / f"{model}_defended_confusion_test_retrained_attacker.json"
                baseline = _safe_json(baseline_path)
                fixed = _safe_json(fixed_path)
                retrain = _safe_json(retrain_path)

                if not isinstance(baseline, dict):
                    missing.append(
                        {
                            "section": "mock",
                            "dataset": "mock",
                            "seed": seed,
                            "model_type": model,
                            "method": method,
                            "mode": "baseline",
                            "reason": "baseline_confusion_missing",
                            "expected_file": str(baseline_path),
                        }
                    )
                    continue

                for mode, defended_obj, conf_path in [
                    ("fixed_attacker", fixed, fixed_path),
                    ("retrain_attacker", retrain, retrain_path),
                ]:
                    key = (seed, model, method, mode)
                    if not isinstance(defended_obj, dict):
                        missing.append(
                            {
                                "section": "mock",
                                "dataset": "mock",
                                "seed": seed,
                                "model_type": model,
                                "method": method,
                                "mode": mode,
                                "reason": "defended_confusion_missing",
                                "expected_file": str(conf_path),
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
                        str(baseline_path),
                        str(conf_path),
                        str(report_path),
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

                    # Copy to final thesis defense structure.
                    dst_base = OUT_DEFENSE / "mock" / f"seed_{seed}" / model / method / mode
                    dst_base.mkdir(parents=True, exist_ok=True)
                    _safe_copy(conf_path, dst_base / "confusion.json")
                    _safe_copy(report_path, dst_base / "defense_report.json")
                    (dst_base / "classification_report.txt").write_text(
                        f"dataset=mock\nseed={seed}\nmodel={model}\nmethod={method}\nmode={mode}\n"
                        f"baseline_acc={baseline_acc:.6f}\ndefended_acc={defended_acc:.6f}\n"
                        f"baseline_f1_macro={baseline_f1:.6f}\ndefended_f1_macro={defended_f1:.6f}\n",
                        encoding="utf-8",
                    )
                    _write_json(
                        dst_base / "trace.json",
                        _extract_trace(
                            dataset="mock",
                            seed=seed,
                            model=model,
                            method=method,
                            mode=mode,
                            config=f"configs/generated_all_methods/default.seed_{seed}.{method}.yaml",
                            command=f"python experiments/core/run_defense_eval.py --mode {mode}",
                            env=env,
                        ),
                    )

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
                                "source_file": str(conf_path),
                            }
                        )

            for scan_method in ["ldp", "noise"]:
                scan_path = ROOT / "outputs" / "defense" / "full_multiseed" / f"seed_{seed}" / scan_method / "comparisons" / "comparison_results.csv"
                if not scan_path.exists():
                    missing.append(
                        {
                            "section": "mock_parameter_scan",
                            "dataset": "mock",
                            "seed": seed,
                            "method": scan_method,
                            "reason": "comparison_results_missing",
                            "expected_file": str(scan_path),
                        }
                    )
                    continue
                df = pd.read_csv(scan_path)
                for _, r in df.iterrows():
                    out_row = {
                        "dataset": "mock",
                        "seed": seed,
                        "model_type": "lstm",  # run_compare in matrix scripts uses lstm baseline checkpoint
                        "method": str(r.get("method", scan_method)),
                        "mode": "fixed_attacker",
                        "parameter_name": str(r.get("param_name")),
                        "parameter_value": float(r.get("param_value")),
                        "baseline_acc": float(r.get("baseline_accuracy")),
                        "defended_acc": float(r.get("defended_accuracy")),
                        "accuracy_drop": float(r.get("accuracy_drop")),
                        "defended_f1_macro": float(r.get("defended_f1_macro")),
                        "mse": float(r.get("mse")),
                        "mae": float(r.get("mae")),
                        "pearson_r": float(r.get("pearson_r")),
                        "source_file": str(scan_path),
                    }
                    if scan_method == "ldp":
                        scan_ldp_rows.append(out_row)
                    else:
                        scan_noise_rows.append(out_row)

                # Mock scan requirement does not enforce retrain/MLP scans explicitly.

    rows = sorted(rows, key=lambda x: (x["seed"], x["model_type"], x["method"], x["mode"]))
    mock_report_dir = OUT_REPORT / "mock"
    _write_json(mock_report_dir / "mock_summary.json", rows)
    _write_csv(mock_report_dir / "mock_summary.csv", rows)
    _write_csv(mock_report_dir / "mock_parameter_scan_ldp.csv", scan_ldp_rows)
    _write_csv(mock_report_dir / "mock_parameter_scan_noise.csv", scan_noise_rows)
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
        "notes": "source=outputs/defense/full_multiseed/*",
    }
    _write_json(mock_report_dir / "mock_coverage_audit.json", coverage)
    return {
        "rows": rows,
        "coverage": coverage,
        "scan_ldp_rows": scan_ldp_rows,
        "scan_noise_rows": scan_noise_rows,
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
    scan_ldp_rows: list[dict[str, Any]] = []
    scan_noise_rows: list[dict[str, Any]] = []
    found_keys: set[tuple[str, int, str, str, str]] = set()

    datasets = ["uci_har", "kasteren", "casas_hh101"]
    for ds in datasets:
        for seed in SEEDS:
            proc_dir = ROOT / "data" / "processed" / "real_public_benchmark" / ds / f"seed_{seed}"
            meta = _safe_json(proc_dir / "meta.json")
            if not isinstance(meta, dict):
                missing.append(
                    {
                        "section": "real",
                        "dataset": ds,
                        "seed": seed,
                        "reason": "processed_meta_missing",
                        "expected_file": str(proc_dir / "meta.json"),
                    }
                )
            for method in METHODS:
                base = ROOT / "outputs" / "defense" / "real_public_benchmark" / ds / f"seed_{seed}" / method
                report_path = base / "defense_report.json"
                rep = _safe_json(report_path) or {}
                dist = rep.get("distortion", {}) if isinstance(rep, dict) else {}

                for model in MODELS:
                    baseline_path = base / "json_reports" / f"{model}_baseline_confusion_test.json"
                    fixed_path = base / "json_reports" / f"{model}_defended_confusion_test_fixed_attacker.json"
                    retrain_path = base / "json_reports" / f"{model}_defended_confusion_test_retrained_attacker.json"
                    baseline = _safe_json(baseline_path)
                    fixed = _safe_json(fixed_path)
                    retrain = _safe_json(retrain_path)

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
                                "method": method,
                                "mode": "baseline",
                                "reason": "baseline_confusion_missing",
                                "expected_file": str(baseline_path),
                            }
                        )
                        continue

                    for mode, defended_obj, conf_path in [
                        ("fixed_attacker", fixed, fixed_path),
                        ("retrain_attacker", retrain, retrain_path),
                    ]:
                        key = (ds, seed, model, method, mode)
                        if not isinstance(defended_obj, dict):
                            missing.append(
                                {
                                    "section": "real",
                                    "dataset": ds,
                                    "seed": seed,
                                    "model_type": model,
                                    "method": method,
                                    "mode": mode,
                                    "reason": "defended_confusion_missing",
                                    "expected_file": str(conf_path),
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

                        source_files = [str(baseline_path), str(conf_path), str(report_path), str(proc_dir / "meta.json")]
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

                        dst_base = OUT_DEFENSE / "real" / ds / f"seed_{seed}" / model / method / mode
                        dst_base.mkdir(parents=True, exist_ok=True)
                        _safe_copy(conf_path, dst_base / "confusion.json")
                        _safe_copy(report_path, dst_base / "defense_report.json")
                        (dst_base / "classification_report.txt").write_text(
                            f"dataset={ds}\nseed={seed}\nmodel={model}\nmethod={method}\nmode={mode}\n"
                            f"baseline_acc={baseline_acc:.6f}\ndefended_acc={defended_acc:.6f}\n"
                            f"baseline_f1_macro={baseline_f1:.6f}\ndefended_f1_macro={defended_f1:.6f}\n",
                            encoding="utf-8",
                        )
                        _write_json(
                            dst_base / "trace.json",
                            _extract_trace(
                                dataset=ds,
                                seed=seed,
                                model=model,
                                method=method,
                                mode=mode,
                                config=f"configs/generated_real_public/{ds}.seed_{seed}.{method}.yaml",
                                command=f"python experiments/core/run_defense_eval.py --mode {mode}",
                                env=env,
                            ),
                        )
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
                                    "source_file": str(conf_path),
                                }
                            )

            # scans per dataset/seed
            for scan_method in ["ldp", "noise"]:
                scan_path = ROOT / "outputs" / "defense" / "real_public_benchmark" / ds / f"seed_{seed}" / scan_method / "comparisons" / "comparison_results.csv"
                if not scan_path.exists():
                    missing.append(
                        {
                            "section": "real_parameter_scan",
                            "dataset": ds,
                            "seed": seed,
                            "method": scan_method,
                            "reason": "comparison_results_missing",
                            "expected_file": str(scan_path),
                        }
                    )
                    continue
                df = pd.read_csv(scan_path)
                for _, r in df.iterrows():
                    out_row = {
                        "dataset": ds,
                        "seed": seed,
                        "model_type": "lstm",
                        "method": str(r.get("method", scan_method)),
                        "mode": "fixed_attacker",
                        "parameter_name": str(r.get("param_name")),
                        "parameter_value": float(r.get("param_value")),
                        "baseline_acc": float(r.get("baseline_accuracy")),
                        "defended_acc": float(r.get("defended_accuracy")),
                        "accuracy_drop": float(r.get("accuracy_drop")),
                        "defended_f1_macro": float(r.get("defended_f1_macro")),
                        "mse": float(r.get("mse")),
                        "mae": float(r.get("mae")),
                        "pearson_r": float(r.get("pearson_r")),
                        "source_file": str(scan_path),
                    }
                    if scan_method == "ldp":
                        scan_ldp_rows.append(out_row)
                    else:
                        scan_noise_rows.append(out_row)

                # Requirement asks to at least cover UCI HAR; mark unmet mandatory parts.
                if ds == "uci_har":
                    missing.append(
                        {
                            "section": "real_parameter_scan",
                            "dataset": ds,
                            "seed": seed,
                            "model_type": "lstm",
                            "method": scan_method,
                            "mode": "retrain_attacker",
                            "reason": "retrain_scan_not_supported_by_run_compare",
                        }
                    )
                    missing.append(
                        {
                            "section": "real_parameter_scan",
                            "dataset": ds,
                            "seed": seed,
                            "model_type": "mlp",
                            "method": scan_method,
                            "mode": "fixed_attacker",
                            "reason": "mlp_scan_not_generated_by_current_pipeline",
                        }
                    )
                    missing.append(
                        {
                            "section": "real_parameter_scan",
                            "dataset": ds,
                            "seed": seed,
                            "model_type": "mlp",
                            "method": scan_method,
                            "mode": "retrain_attacker",
                            "reason": "retrain_scan_not_supported_by_run_compare",
                        }
                    )

    rows = sorted(rows, key=lambda x: (x["dataset"], x["seed"], x["model_type"], x["method"], x["mode"]))
    real_report_dir = OUT_REPORT / "real"
    _write_json(real_report_dir / "real_summary.json", rows)
    _write_csv(real_report_dir / "real_summary.csv", rows)
    _write_csv(real_report_dir / "real_parameter_scan_ldp.csv", scan_ldp_rows)
    _write_csv(real_report_dir / "real_parameter_scan_noise.csv", scan_noise_rows)
    _write_csv(real_report_dir / "real_top_confusions.csv", top_conf_rows)

    # import meta summary
    meta_rows: list[dict[str, Any]] = []
    for ds in ["uci_har", "kasteren", "casas_hh101"]:
        meta_path = ROOT / "data" / "processed" / "imports" / ds / "meta.json"
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
                "meta_path": str(meta_path),
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
        "notes": "source=outputs/defense/real_public_benchmark/*",
    }
    _write_json(real_report_dir / "real_coverage_audit.json", coverage)

    real_missing = [m for m in missing if str(m.get("section", "")).startswith("real")]
    _write_json(real_report_dir / "real_missing_outputs.json", real_missing)
    return {
        "rows": rows,
        "coverage": coverage,
        "scan_ldp_rows": scan_ldp_rows,
        "scan_noise_rows": scan_noise_rows,
        "meta_rows": meta_rows,
    }


def _cooja_logs_available(manifest_path: Path) -> tuple[bool, list[Path]]:
    obj = _safe_json(manifest_path)
    if not isinstance(obj, dict):
        return False, []
    paths: list[Path] = []
    baseline = obj.get("baseline", {})
    if isinstance(baseline, dict):
        for k in ["radio_log", "app_log"]:
            v = baseline.get(k)
            if v:
                paths.append(Path(str(v)))
    for m in obj.get("methods", []) or []:
        if not isinstance(m, dict):
            continue
        for k in ["radio_log", "app_log"]:
            v = m.get(k)
            if v:
                paths.append(Path(str(v)))
    exists = [p for p in paths if p.exists()]
    return len(paths) > 0 and len(exists) >= 2, paths


def _collect_cooja(env: EnvInfo, missing: list[dict[str, Any]]) -> dict[str, Any]:
    cooja_report_dir = OUT_REPORT / "cooja"
    rows: list[dict[str, Any]] = []
    feat_rows: list[dict[str, Any]] = []
    top_conf_rows: list[dict[str, Any]] = []
    overhead_rows: list[dict[str, Any]] = []

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
    _write_csv(cooja_report_dir / "cooja_feature_importance.csv", feat_rows)
    _write_csv(cooja_report_dir / "cooja_top_confusions.csv", top_conf_rows)
    _write_csv(cooja_report_dir / "cooja_overhead_summary.csv", overhead_rows)
    _write_json(cooja_report_dir / "cooja_missing_outputs.json", [m for m in missing if m.get("section") == "cooja"])
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
                "source_files": "outputs/reports/final_thesis/mock/mock_summary.csv",
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
                "source_files": "outputs/reports/final_thesis/mock/mock_summary.csv",
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
                "source_files": "outputs/reports/final_thesis/real/real_summary.csv",
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
                    "source_files": "outputs/reports/final_thesis/real/real_parameter_scan_ldp.csv",
                    "conclusion": "可用于展示 epsilon 变大时准确率恢复趋势。",
                    "limitations": "当前扫描来自 fixed_attacker；retrain 扫描缺失。",
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
                    "source_files": "outputs/reports/final_thesis/real/real_parameter_scan_noise.csv",
                    "conclusion": "可用于展示噪声强度上升时攻击准确率下降趋势。",
                    "limitations": "当前扫描来自 fixed_attacker；retrain 扫描缺失。",
                }
            )

    # 8 representative confusion matrix for each dataset
    candidates = [
        ("mock", OUT_DEFENSE / "mock" / "seed_42" / "lstm" / "adaptive_ldp" / "fixed_attacker" / "confusion.json"),
        ("uci_har", OUT_DEFENSE / "real" / "uci_har" / "seed_42" / "lstm" / "adaptive_ldp" / "fixed_attacker" / "confusion.json"),
        ("kasteren", OUT_DEFENSE / "real" / "kasteren" / "seed_42" / "lstm" / "adaptive_ldp" / "fixed_attacker" / "confusion.json"),
        ("casas_hh101", OUT_DEFENSE / "real" / "casas_hh101" / "seed_42" / "lstm" / "adaptive_ldp" / "fixed_attacker" / "confusion.json"),
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
                    "source_files": str(pjson),
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
                    "source_files": "outputs/reports/final_thesis/cooja/cooja_summary.csv",
                    "conclusion": "可用于展示节点级防御在流量侧攻击下的变化。",
                    "limitations": "依赖 Cooja 日志质量与可获得性。",
                }
            )
        else:
            missing.append(
                {"section": "figures", "figure": "cooja_mode_accuracy", "reason": "cooja_rows_empty"}
            )
        missing.append(
            {
                "section": "figures",
                "figure": "cooja_overhead",
                "reason": "cooja_overhead_metrics_unavailable",
                "note": "energy_metric_available=false and delay_proxy unavailable in current logs.",
            }
        )
    else:
        missing.append({"section": "figures", "figure": "cooja_mode_accuracy", "reason": "cooja_rows_empty"})
        missing.append({"section": "figures", "figure": "cooja_overhead", "reason": "cooja_rows_empty"})

    return figures


def _write_figure_list(figures: list[dict[str, Any]]) -> None:
    path = OUT_REPORT / "figure_table_list.md"
    lines = ["# 图表清单", ""]
    for i, fig in enumerate(figures, start=1):
        lines.append(f"## {i}. {fig['title']}")
        lines.append(f"- 图路径: `{fig['path']}`")
        lines.append(f"- 源文件: `{fig['source_files']}`")
        lines.append(f"- 可写入论文结论: {fig['conclusion']}")
        lines.append(f"- 口径限制: {fig['limitations']}")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_final_summary_md(
    env: EnvInfo,
    mock_rows: list[dict[str, Any]],
    real_rows: list[dict[str, Any]],
    cooja_rows: list[dict[str, Any]],
    missing: list[dict[str, Any]],
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
    lines.append(f"- git commit: `{env.git_commit}`")
    lines.append(f"- python version: `{env.python_version}`")
    lines.append(f"- OS: `{env.os}`")
    lines.append(f"- start time / end time: `{env.start_time}` / `{env.end_time}`")
    lines.append("")

    lines.append("## 2. mock 实验是否完整")
    mock_expected = len(SEEDS) * len(MODELS) * len(METHODS) * len(MODES)
    lines.append(f"- 完成情况: 已收集 `{len(mock_df)}` / 期望 `{mock_expected}` 条（dataset=mock）。")
    if not mock_df.empty:
        for model in MODELS:
            sub = mock_df[mock_df["model_type"] == model]
            lines.append(
                f"- {model.upper()} 主要结果: baseline_acc 均值 `{_mean(sub['baseline_acc'].tolist()):.4f}`，"
                f"defended_acc 均值 `{_mean(sub['defended_acc'].tolist()):.4f}`。"
            )
    lines.append("- 参数扫描结果: 已输出 ldp/noise 扫描 CSV；retrain 与 MLP 扫描缺项已写入 missing_outputs。")
    lines.append("- 可写入论文的结论: fixed_attacker 与 retrain_attacker 在 mock 数据上呈现可观差异，支持隐私-效用分析。")
    lines.append("- 不建议写入论文的内容: 缺失组合（见 final_missing_outputs.json）对应的推断结论。")
    lines.append("")

    lines.append("## 3. 真实数据集实验是否完整")
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
    lines.append("- 可写入论文的结论: 在 UCI HAR 与 Kasteren 上可稳定观测防御导致的准确率下降及部分重训恢复。")
    lines.append("- 不建议写入论文的内容: CASAS 缺失 seed_2026 的完整矩阵组合。")
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
    lines.append("- 推荐论文引用: `outputs/reports/final_thesis/*.csv|*.json` 与 `outputs/defense/final_thesis/**`。")
    lines.append("- 不建议直接引用: 旧路径中未分 model/mode 的单文件报告。")
    lines.append("")

    lines.append("## 6. 下一步建议")
    lines.append("- 优先补齐 CASAS seed_2026 组合与 Cooja 日志可达性，再重新执行本脚本。")
    lines.append("- 若需真实参数扫描完整性，新增 run_compare 的 retrain 模式并补齐 MLP 扫描。")
    lines.append("- 论文图表建议优先使用 `outputs/figures/final_thesis/`。")
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
    env = EnvInfo(
        git_commit=git_commit,
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

    # unified files
    final_rows = []
    final_rows.extend([{**r, "section": "mock"} for r in mock["rows"]])
    final_rows.extend([{**r, "section": "real"} for r in real["rows"]])
    final_rows.extend([{**r, "section": "cooja"} for r in cooja["rows"]])
    _write_json(OUT_REPORT / "final_summary.json", final_rows)
    _write_csv(OUT_REPORT / "final_summary.csv", final_rows)

    manifest = {
        "generated_at": _now(),
        "git_commit": env.git_commit,
        "inputs": {
            "mock_source_root": "outputs/defense/full_multiseed",
            "real_source_root": "outputs/defense/real_public_benchmark",
            "cooja_manifest_candidates": [
                "configs/cooja_defense_dummy_logs.json",
                "configs/cooja_defense_postprocess.json",
                "configs/cooja_defense_logs.json",
            ],
        },
        "outputs": {
            "mock_summary": str(OUT_REPORT / "mock" / "mock_summary.csv"),
            "real_summary": str(OUT_REPORT / "real" / "real_summary.csv"),
            "cooja_summary": str(OUT_REPORT / "cooja" / "cooja_summary.csv"),
            "final_summary": str(OUT_REPORT / "final_summary.csv"),
        },
    }
    _write_json(OUT_REPORT / "final_manifest.json", manifest)

    # coverage + missing
    coverage = {
        "mock": mock["coverage"],
        "real": real["coverage"],
        "cooja_rows": len(cooja["rows"]),
        "should_have_experiments": {
            "mock": len(SEEDS) * len(MODELS) * len(METHODS) * len(MODES),
            "real": 3 * len(SEEDS) * len(MODELS) * len(METHODS) * len(MODES),
        },
        "actual_completed": {
            "mock": len(mock["rows"]),
            "real": len(real["rows"]),
            "cooja": len(cooja["rows"]),
        },
        "missing_combinations": {
            "mock": mock["coverage"]["missing_combinations"],
            "real": real["coverage"]["missing_combinations"],
        },
        "covered_risk_files": [
            "outputs/reports/**/metrics.json",
            "outputs/defense/**/defense_report.json",
        ],
        "recommended_for_thesis": [
            "outputs/reports/final_thesis/mock/mock_summary.csv",
            "outputs/reports/final_thesis/real/real_summary.csv",
            "outputs/reports/final_thesis/final_summary.csv",
        ],
        "not_recommended_for_thesis": [
            "outputs/reports/**/metrics.json (legacy single-file outputs)",
            "outputs/defense/**/defense_report.json (legacy overwritten paths)",
        ],
    }
    _write_json(OUT_REPORT / "final_coverage_audit.json", coverage)
    _write_json(OUT_REPORT / "final_missing_outputs.json", missing)

    # figures + figure list
    figures = _build_figures(
        mock_rows=mock["rows"],
        real_rows=real["rows"],
        cooja_rows=cooja["rows"],
        scan_mock_ldp=mock["scan_ldp_rows"],
        scan_mock_noise=mock["scan_noise_rows"],
        scan_real_ldp=real["scan_ldp_rows"],
        scan_real_noise=real["scan_noise_rows"],
        missing=missing,
    )
    _write_figure_list(figures)
    _write_json(OUT_REPORT / "final_missing_outputs.json", missing)

    env.end_time = _now()
    _write_final_summary_md(env, mock["rows"], real["rows"], cooja["rows"], missing)

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
        print("final_summary_path=outputs/reports/final_thesis/final_summary.csv")
        print("final_report_path=outputs/reports/final_thesis/final_thesis_summary.md")
    else:
        print("FINAL_THESIS_RESULTS_READY=false")
        print("missing_outputs_path=outputs/reports/final_thesis/final_missing_outputs.json")


if __name__ == "__main__":
    main()
