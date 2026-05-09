#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit canonical experiment-output symmetry without running experiments."""

from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT_REPORT = ROOT / "outputs" / "summaries" / "final_thesis"
EXPERIMENTS = ROOT / "outputs" / "experiments"

SEEDS = [42, 123, 2026]
MODELS = ["lstm", "mlp"]
METHODS = ["adaptive_ldp", "ldp", "noise"]
MODES = ["fixed_attacker", "retrain_attacker"]
DATASETS = ["uci_har", "kasteren", "casas_hh101"]
MAIN_FILES = ["metrics.json", "confusion.json", "classification_report.txt", "trace.json", "defense_report.json", "source_manifest.json"]
PARAM_ROWS = {"ldp": 5, "noise": 4, "adaptive_ldp": 6}
COOJA_METHODS = ["dummy_noise", "dummy_ldp", "dummy_adaptive_ldp"]


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except Exception:
        return path.as_posix().replace("\\", "/")


def _read_csv_rows(path: Path) -> list[dict[str, str]] | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))
    except Exception:
        return None


def _read_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _main_dir(dataset: str, seed: int, model: str, method: str, mode: str) -> Path:
    return EXPERIMENTS / dataset / f"seed_{seed}" / model / method / mode


def _scan_file(dataset: str, seed: int, model: str, method: str, mode: str) -> Path:
    return _main_dir(dataset, seed, model, method, mode) / "parameter_scan" / "comparison_results.csv"


def _csv_complete(path: Path, method: str) -> tuple[bool, str]:
    rows = _read_csv_rows(path)
    if rows is None:
        return False, "missing_or_unreadable"
    if len(rows) != PARAM_ROWS[method]:
        return False, f"row_count_{len(rows)}_expected_{PARAM_ROWS[method]}"
    return True, "ok"


def audit_main(dataset: str) -> tuple[bool, list[dict[str, Any]], dict[str, int]]:
    missing: list[dict[str, Any]] = []
    completed = 0
    for seed in SEEDS:
        for model in MODELS:
            baseline = EXPERIMENTS / dataset / f"seed_{seed}" / model / "baseline"
            baseline_absent = [name for name in ["baseline_metrics.json", "baseline_confusion.json", "baseline_classification_report.txt", "source_manifest.json"] if not (baseline / name).is_file()]
            if baseline_absent:
                missing.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "model_type": model,
                        "method": "baseline",
                        "mode": "baseline",
                        "missing_files": baseline_absent,
                        "expected_dir": _rel(baseline),
                    }
                )
            for method in METHODS:
                for mode in MODES:
                    base = _main_dir(dataset, seed, model, method, mode)
                    absent = [name for name in MAIN_FILES if not (base / name).is_file()]
                    if absent:
                        missing.append(
                            {
                                "dataset": dataset,
                                "seed": seed,
                                "model_type": model,
                                "method": method,
                                "mode": mode,
                                "missing_files": absent,
                                "expected_dir": _rel(base),
                            }
                        )
                    else:
                        completed += 1
    expected = len(SEEDS) * len(MODELS) * len(METHODS) * len(MODES)
    return not any(m.get("mode") != "baseline" for m in missing), missing, {"expected": expected, "completed": completed, "missing": expected - completed}


def audit_parameter_scans(scope: str, datasets: list[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    missing: list[dict[str, Any]] = []
    completed = 0
    adaptive_counts: dict[str, int] = {}
    expected = 0
    for dataset in datasets:
        for seed in SEEDS:
            for method in METHODS:
                for model in MODELS:
                    for mode in MODES:
                        expected += 1
                        path = _scan_file(dataset, seed, model, method, mode)
                        ok, reason = _csv_complete(path, method)
                        if not ok:
                            missing.append(
                                {
                                    "section": f"{scope}_parameter_scan",
                                    "dataset": dataset,
                                    "seed": seed,
                                    "method": method,
                                    "model_type": model,
                                    "mode": mode,
                                    "reason": reason,
                                    "expected_file": _rel(path),
                                }
                            )
                            continue
                        completed += 1
                        if method == "adaptive_ldp":
                            rows = _read_csv_rows(path) or []
                            profiles = {str(r.get("profile_name")) for r in rows if r.get("profile_name")}
                            adaptive_counts[f"{scope}:{dataset}:seed_{seed}:{model}:{mode}"] = len(profiles)
    return missing, {"expected": expected, "completed": completed, "missing": missing, "adaptive_profile_count": adaptive_counts}


def audit_cooja_outputs() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    missing: list[dict[str, Any]] = []
    summary = OUT_REPORT / "cooja" / "cooja_summary.csv"
    per_seed = OUT_REPORT / "cooja" / "cooja_per_seed.csv"
    traffic = OUT_REPORT / "cooja" / "cooja_traffic_metrics.csv"
    report = EXPERIMENTS / "cooja" / "eval" / "defense_eval_report.json"
    for path, label in [(summary, "cooja_summary"), (per_seed, "cooja_per_seed"), (traffic, "cooja_traffic_metrics"), (report, "defense_eval_report")]:
        if not path.exists() or path.stat().st_size == 0:
            missing.append({"section": "cooja", "reason": f"{label}_missing", "expected_file": _rel(path)})
    per_rows = _read_csv_rows(per_seed) or []
    expected_dirs = 0
    observed_dirs = 0
    for seed in SEEDS:
        for method in COOJA_METHODS:
            for mode in MODES:
                expected_dirs += 1
                base = EXPERIMENTS / "cooja" / f"seed_{seed}" / "random_forest" / method / mode
                if (base / "metrics.json").is_file() and (base / "source_manifest.json").is_file():
                    observed_dirs += 1
                else:
                    missing.append({"section": "cooja", "reason": "canonical_cooja_combo_missing", "expected_dir": _rel(base)})
    traffic_rows = _read_csv_rows(traffic) or []
    numeric_available = False
    for row in traffic_rows:
        for key in ["baseline_pkt_count_mean", "defense_pkt_count_mean", "baseline_byte_count_mean", "defense_byte_count_mean", "baseline_mean_iat_ms", "defense_mean_iat_ms"]:
            value = str(row.get(key, "")).strip().lower()
            if value and value not in {"nan", "none", "null"}:
                numeric_available = True
                break
    traffic_status = {
        "rows_expected": 9,
        "rows_observed": len(traffic_rows),
        "numeric_metrics_available": numeric_available,
        "reason": "Radio/app logs available for accuracy evaluation, but exported traffic rows do not expose enough labeled packet fields for packet/byte/IAT proxy metrics.",
    }
    return missing, {"per_seed_rows": len(per_rows), "canonical_expected": expected_dirs, "canonical_completed": observed_dirs, "traffic_status": traffic_status}


def detect_duplicates() -> list[dict[str, Any]]:
    duplicates: list[dict[str, Any]] = []
    for path in [
        OUT_REPORT / "mock" / "mock_parameter_scan_ldp.csv",
        OUT_REPORT / "mock" / "mock_parameter_scan_noise.csv",
        OUT_REPORT / "mock" / "mock_parameter_scan_adaptive_ldp.csv",
        OUT_REPORT / "real" / "real_parameter_scan_ldp.csv",
        OUT_REPORT / "real" / "real_parameter_scan_noise.csv",
        OUT_REPORT / "real" / "real_parameter_scan_adaptive_ldp.csv",
    ]:
        rows = _read_csv_rows(path) or []
        counter: Counter[tuple[str, ...]] = Counter()
        for row in rows:
            key = (
                str(row.get("dataset")),
                str(row.get("seed")),
                str(row.get("model_type")),
                str(row.get("mode")),
                str(row.get("method")),
                str(row.get("parameter_name") or row.get("profile_name")),
                str(row.get("parameter_value")),
                str(row.get("source_file")),
            )
            counter[key] += 1
        dup = [k for k, v in counter.items() if v > 1]
        if dup:
            duplicates.append({"file": _rel(path), "duplicate_keys": len(dup)})
    return duplicates


def delivery_docs_status() -> tuple[dict[str, bool], list[dict[str, Any]]]:
    docs = {
        "repository_delivery_guide": ROOT / "docs" / "REPOSITORY_DELIVERY_GUIDE.md",
        "artifact_layout": ROOT / "docs" / "ARTIFACT_LAYOUT.md",
        "artifact_index": OUT_REPORT / "artifact_index.md",
        "adaptive_ablation_overview": OUT_REPORT / "adaptive_ldp_ablation_overview.md",
        "mock_adaptive_ablation_summary": OUT_REPORT / "mock" / "mock_adaptive_ldp_ablation_summary.csv",
        "real_adaptive_ablation_summary": OUT_REPORT / "real" / "real_adaptive_ldp_ablation_summary.csv",
    }
    status = {name: path.exists() for name, path in docs.items()}
    missing = [{"section": "delivery_docs", "reason": "delivery_doc_missing", "name": name, "expected_file": _rel(path)} for name, path in docs.items() if not path.exists()]
    return status, missing


def build_audit() -> dict[str, Any]:
    mock_ok, mock_missing, mock_counts = audit_main("mock")
    real_missing_all: list[dict[str, Any]] = []
    real_completed = 0
    real_expected = 0
    for dataset in DATASETS:
        _, missing, counts = audit_main(dataset)
        real_missing_all.extend([m for m in missing if m.get("mode") != "baseline"])
        real_completed += counts["completed"]
        real_expected += counts["expected"]
    mock_scan_missing, mock_scan = audit_parameter_scans("mock", ["mock"])
    real_scan_missing, real_scan = audit_parameter_scans("real", DATASETS)
    cooja_missing, cooja = audit_cooja_outputs()
    docs_status, docs_missing = delivery_docs_status()
    duplicates = detect_duplicates()
    adaptive_counts = {**mock_scan["adaptive_profile_count"], **real_scan["adaptive_profile_count"]}
    audit = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "canonical_roots": {
            "experiments": "outputs/experiments",
            "summaries": "outputs/summaries/final_thesis",
            "figures": "outputs/figures/summaries/final_thesis",
        },
        "complete_main_matrix_mock": mock_ok,
        "complete_main_matrix_real": real_completed == real_expected and not real_missing_all,
        "mock_main_matrix": mock_counts,
        "real_main_matrix": {"expected": real_expected, "completed": real_completed, "missing": real_expected - real_completed},
        "missing_mock_main_matrix": mock_missing,
        "missing_real_main_matrix": real_missing_all,
        "missing_mock_parameter_scans": mock_scan_missing,
        "missing_real_parameter_scans": real_scan_missing,
        "parameter_scan_counts": {
            "mock": {"expected": mock_scan["expected"], "completed": mock_scan["completed"], "missing": len(mock_scan_missing)},
            "real": {"expected": real_scan["expected"], "completed": real_scan["completed"], "missing": len(real_scan_missing)},
        },
        "adaptive_ldp_profile_count": {"expected": PARAM_ROWS["adaptive_ldp"], "observed": adaptive_counts},
        "missing_cooja_outputs": cooja_missing,
        "cooja": cooja,
        "cooja_traffic_metrics_status": cooja["traffic_status"],
        "duplicated_rows_detected": duplicates,
        "delivery_docs_status": docs_status,
        "delivery_docs_missing": docs_missing,
        "actions_recommended": [],
    }
    if mock_scan_missing or real_scan_missing:
        audit["actions_recommended"].append("Run only the missing parameter-scan combinations; do not rerun completed matrices.")
    if cooja_missing:
        audit["actions_recommended"].append("Keep Cooja limitations explicit; do not fabricate unavailable packet/byte/IAT or energy/delay metrics.")
    if docs_missing:
        audit["actions_recommended"].append("Regenerate or move missing delivery documents under outputs/summaries/final_thesis and docs/.")
    return audit


def write_markdown(audit: dict[str, Any]) -> None:
    lines = [
        "# Final Symmetry Audit",
        "",
        f"- Generated at: `{audit['generated_at']}`",
        f"- Canonical experiment root: `{audit['canonical_roots']['experiments']}`",
        f"- Mock main matrix: `{audit['mock_main_matrix']['completed']}` / `{audit['mock_main_matrix']['expected']}`",
        f"- Real main matrix: `{audit['real_main_matrix']['completed']}` / `{audit['real_main_matrix']['expected']}`",
        f"- Mock parameter scans: `{audit['parameter_scan_counts']['mock']['completed']}` / `{audit['parameter_scan_counts']['mock']['expected']}`",
        f"- Real parameter scans: `{audit['parameter_scan_counts']['real']['completed']}` / `{audit['parameter_scan_counts']['real']['expected']}`",
        f"- Missing mock parameter scans: `{len(audit['missing_mock_parameter_scans'])}`",
        f"- Missing real parameter scans: `{len(audit['missing_real_parameter_scans'])}`",
        f"- Cooja canonical dirs: `{audit['cooja']['canonical_completed']}` / `{audit['cooja']['canonical_expected']}`",
        f"- Duplicate scan rows detected: `{len(audit['duplicated_rows_detected'])}`",
        "",
        "## Notes",
        "",
        "- This audit checks the normalized artifact layout only.",
        "- Cooja unavailable traffic metrics are treated as limitations, not missing experiments.",
    ]
    (OUT_REPORT / "final_symmetry_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_REPORT.mkdir(parents=True, exist_ok=True)
    audit = build_audit()
    (OUT_REPORT / "final_symmetry_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(audit)
    print(f"final_symmetry_audit={_rel(OUT_REPORT / 'final_symmetry_audit.json')}")


if __name__ == "__main__":
    main()
