#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit final thesis experiment-output symmetry without running experiments."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT_REPORT = ROOT / "outputs" / "reports" / "final_thesis"

SEEDS = [42, 123, 2026]
MODELS = ["lstm", "mlp"]
METHODS = ["adaptive_ldp", "ldp", "noise"]
MODES = ["fixed_attacker", "retrain_attacker"]
DATASETS = ["uci_har", "kasteren", "casas_hh101"]

MAIN_FILES = [
    "confusion.json",
    "classification_report.txt",
    "trace.json",
    "defense_report.json",
]
PARAM_ROWS = {"ldp": 5, "noise": 4, "adaptive_ldp": 6}
COOJA_METHODS = ["dummy_noise", "dummy_ldp", "dummy_adaptive_ldp"]
DELIVERY_DOCS = {
    "repository_delivery_guide": ROOT / "docs" / "REPOSITORY_DELIVERY_GUIDE.md",
    "artifact_index": OUT_REPORT / "artifact_index.md",
    "adaptive_ablation_overview": OUT_REPORT / "adaptive_ldp_ablation_overview.md",
    "mock_adaptive_ablation_summary": OUT_REPORT / "mock" / "mock_adaptive_ldp_ablation_summary.csv",
    "real_adaptive_ablation_summary": OUT_REPORT / "real" / "real_adaptive_ldp_ablation_summary.csv",
}


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except Exception:
        return path.as_posix()


def _read_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _read_csv_rows(path: Path) -> list[dict[str, str]] | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))
    except Exception:
        return None


def _csv_is_complete(path: Path, method: str) -> tuple[bool, str]:
    rows = _read_csv_rows(path)
    if rows is None:
        return False, "missing_or_unreadable"
    expected = PARAM_ROWS[method]
    if len(rows) != expected:
        return False, f"row_count_{len(rows)}_expected_{expected}"
    return True, "ok"


def _norm_scan_path(root: Path, seed: int, method: str, model: str, mode: str) -> Path:
    return root / f"seed_{seed}" / method / "comparisons" / f"{model}_{mode}_comparison_results.csv"


def _legacy_scan_path(root: Path, seed: int, method: str) -> Path:
    return root / f"seed_{seed}" / method / "comparisons" / "comparison_results.csv"


def _real_norm_scan_path(dataset: str, seed: int, method: str, model: str, mode: str) -> Path:
    return (
        ROOT
        / "outputs"
        / "defense"
        / "real_public_benchmark"
        / dataset
        / f"seed_{seed}"
        / method
        / "comparisons"
        / f"{model}_{mode}_comparison_results.csv"
    )


def _real_legacy_scan_path(dataset: str, seed: int, method: str) -> Path:
    return (
        ROOT
        / "outputs"
        / "defense"
        / "real_public_benchmark"
        / dataset
        / f"seed_{seed}"
        / method
        / "comparisons"
        / "comparison_results.csv"
    )


def audit_main_mock() -> tuple[bool, list[dict[str, Any]], dict[str, int]]:
    missing: list[dict[str, Any]] = []
    completed = 0
    root = ROOT / "outputs" / "defense" / "final_thesis" / "mock"
    for seed in SEEDS:
        for model in MODELS:
            for method in METHODS:
                for mode in MODES:
                    base = root / f"seed_{seed}" / model / method / mode
                    absent = [name for name in MAIN_FILES if not (base / name).is_file()]
                    if absent:
                        missing.append(
                            {
                                "dataset": "mock",
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
    return not missing, missing, {"expected": expected, "completed": completed, "missing": len(missing)}


def audit_main_real() -> tuple[bool, list[dict[str, Any]], dict[str, int]]:
    missing: list[dict[str, Any]] = []
    completed = 0
    root = ROOT / "outputs" / "defense" / "final_thesis" / "real"
    for dataset in DATASETS:
        for seed in SEEDS:
            for model in MODELS:
                for method in METHODS:
                    for mode in MODES:
                        base = root / dataset / f"seed_{seed}" / model / method / mode
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
    expected = len(DATASETS) * len(SEEDS) * len(MODELS) * len(METHODS) * len(MODES)
    return not missing, missing, {"expected": expected, "completed": completed, "missing": len(missing)}


def audit_mock_parameter_scans() -> list[dict[str, Any]]:
    missing: list[dict[str, Any]] = []
    root = ROOT / "outputs" / "defense" / "full_multiseed"
    for seed in SEEDS:
        for method in METHODS:
            for model in MODELS:
                for mode in MODES:
                    norm = _norm_scan_path(root, seed, method, model, mode)
                    ok, reason = _csv_is_complete(norm, method)
                    used = norm
                    if not ok and method in {"ldp", "noise"} and model == "lstm" and mode == "fixed_attacker":
                        legacy = _legacy_scan_path(root, seed, method)
                        legacy_ok, legacy_reason = _csv_is_complete(legacy, method)
                        if legacy_ok:
                            ok, reason, used = True, "legacy_ok", legacy
                        else:
                            reason = f"{reason};legacy_{legacy_reason}"
                    if not ok:
                        missing.append(
                            {
                                "dataset": "mock",
                                "seed": seed,
                                "method": method,
                                "model_type": model,
                                "mode": mode,
                                "reason": reason,
                                "expected_file": _rel(norm),
                            }
                        )
                    elif used.name == "comparison_results.csv" and not norm.exists():
                        missing.append(
                            {
                                "dataset": "mock",
                                "seed": seed,
                                "method": method,
                                "model_type": model,
                                "mode": mode,
                                "reason": "canonical_copy_missing",
                                "legacy_file": _rel(used),
                                "expected_file": _rel(norm),
                            }
                        )
    return missing


def audit_real_parameter_scans() -> list[dict[str, Any]]:
    missing: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for seed in SEEDS:
            for method in METHODS:
                for model in MODELS:
                    for mode in MODES:
                        norm = _real_norm_scan_path(dataset, seed, method, model, mode)
                        ok, reason = _csv_is_complete(norm, method)
                        used = norm
                        if not ok and method in {"ldp", "noise"} and model == "lstm" and mode == "fixed_attacker":
                            legacy = _real_legacy_scan_path(dataset, seed, method)
                            legacy_ok, legacy_reason = _csv_is_complete(legacy, method)
                            if legacy_ok:
                                ok, reason, used = True, "legacy_ok", legacy
                            else:
                                reason = f"{reason};legacy_{legacy_reason}"
                        if not ok:
                            missing.append(
                                {
                                    "dataset": dataset,
                                    "seed": seed,
                                    "method": method,
                                    "model_type": model,
                                    "mode": mode,
                                    "reason": reason,
                                    "expected_file": _rel(norm),
                                }
                            )
                        elif used.name == "comparison_results.csv" and not norm.exists():
                            missing.append(
                                {
                                    "dataset": dataset,
                                    "seed": seed,
                                    "method": method,
                                    "model_type": model,
                                    "mode": mode,
                                    "reason": "canonical_copy_missing",
                                    "legacy_file": _rel(used),
                                    "expected_file": _rel(norm),
                                }
                            )
    return missing


def audit_cooja_outputs() -> list[dict[str, Any]]:
    missing: list[dict[str, Any]] = []
    report_dir = OUT_REPORT / "cooja"
    summary_rows = _read_csv_rows(report_dir / "cooja_summary.csv")
    if summary_rows is None:
        missing.append({"section": "cooja", "reason": "cooja_summary_missing", "expected_file": _rel(report_dir / "cooja_summary.csv")})
    else:
        have = {(r.get("method"), r.get("mode")) for r in summary_rows}
        for method in COOJA_METHODS:
            for mode in MODES:
                if (method, mode) not in have:
                    missing.append({"section": "cooja", "method": method, "mode": mode, "reason": "summary_row_missing"})

    eval_report = ROOT / "outputs" / "defense" / "final_thesis" / "cooja" / "eval" / "defense_eval_report.json"
    obj = _read_json(eval_report)
    if not isinstance(obj, dict):
        missing.append({"section": "cooja", "reason": "defense_eval_report_missing_or_unreadable", "expected_file": _rel(eval_report)})
    else:
        methods = obj.get("methods", {})
        for method in COOJA_METHODS:
            mobj = methods.get(method, {}) if isinstance(methods, dict) else {}
            runs = mobj.get("runs", []) if isinstance(mobj, dict) else []
            have_seeds = {int(r.get("seed")) for r in runs if isinstance(r, dict) and str(r.get("seed", "")).isdigit()}
            for seed in SEEDS:
                if seed not in have_seeds:
                    missing.append({"section": "cooja", "method": method, "seed": seed, "reason": "per_seed_run_missing"})

    for name in ["cooja_per_seed.csv", "cooja_traffic_metrics.csv", "cooja_limitations.md"]:
        path = report_dir / name
        if not path.exists():
            missing.append({"section": "cooja", "reason": f"{name}_missing", "expected_file": _rel(path)})
    return missing


def detect_duplicates() -> list[dict[str, Any]]:
    duplicates: list[dict[str, Any]] = []
    files = [
        OUT_REPORT / "mock" / f"mock_parameter_scan_{m}.csv"
        for m in METHODS
    ] + [
        OUT_REPORT / "real" / f"real_parameter_scan_{m}.csv"
        for m in METHODS
    ]
    for path in files:
        rows = _read_csv_rows(path)
        if rows is None:
            continue
        keys: Counter[tuple[Any, ...]] = Counter()
        for r in rows:
            key = (
                r.get("dataset"),
                r.get("seed"),
                r.get("model_type"),
                r.get("mode"),
                r.get("method"),
                r.get("parameter_name") or r.get("param_name"),
                r.get("profile_name"),
                r.get("parameter_value") or r.get("param_value"),
                r.get("source_file"),
            )
            keys[key] += 1
        dup_count = sum(v - 1 for v in keys.values() if v > 1)
        if dup_count:
            duplicates.append({"file": _rel(path), "duplicate_rows": dup_count})
    return duplicates


def audit_delivery_docs() -> tuple[dict[str, bool], list[dict[str, Any]]]:
    status = {name: path.exists() and path.stat().st_size > 0 for name, path in DELIVERY_DOCS.items()}
    missing = [
        {
            "section": "delivery_docs",
            "name": name,
            "reason": "delivery_doc_missing",
            "expected_file": _rel(path),
        }
        for name, path in DELIVERY_DOCS.items()
        if not status[name]
    ]
    return status, missing


def _group_missing(items: list[dict[str, Any]]) -> dict[str, Any]:
    by_method: dict[str, int] = defaultdict(int)
    by_dataset: dict[str, int] = defaultdict(int)
    for item in items:
        if item.get("method"):
            by_method[str(item["method"])] += 1
        if item.get("dataset"):
            by_dataset[str(item["dataset"])] += 1
    return {
        "total": len(items),
        "by_method": dict(sorted(by_method.items())),
        "by_dataset": dict(sorted(by_dataset.items())),
        "items": items,
    }


def build_audit() -> dict[str, Any]:
    mock_complete, missing_mock_main, mock_counts = audit_main_mock()
    real_complete, missing_real_main, real_counts = audit_main_real()
    missing_mock_scans = audit_mock_parameter_scans()
    missing_real_scans = audit_real_parameter_scans()
    missing_cooja = audit_cooja_outputs()
    duplicates = detect_duplicates()
    delivery_docs_status, delivery_docs_missing = audit_delivery_docs()

    actions: list[str] = []
    if missing_mock_scans or missing_real_scans:
        actions.append("Run experiments/batches/run_missing_parameter_scans.py --skip-existing to fill only missing parameter-scan CSVs.")
    if missing_cooja:
        actions.append("Export Cooja per-seed and traffic CSVs from the existing defense_eval_report.json, or record missing logs if logs are inaccessible.")
    if duplicates:
        actions.append("Rebuild final thesis summaries with de-duplicated parameter scan aggregation.")
    if delivery_docs_missing:
        actions.append("Generate delivery documentation and adaptive_ldp ablation summaries without rerunning experiments.")
    if not actions:
        actions.append("No action required; audited outputs are symmetric.")

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "complete_main_matrix_mock": mock_complete,
        "complete_main_matrix_real": real_complete,
        "main_matrix_counts": {"mock": mock_counts, "real": real_counts},
        "missing_main_matrix_mock": missing_mock_main,
        "missing_main_matrix_real": missing_real_main,
        "missing_mock_parameter_scans": missing_mock_scans,
        "missing_real_parameter_scans": missing_real_scans,
        "missing_cooja_outputs": missing_cooja,
        "delivery_docs_status": delivery_docs_status,
        "delivery_docs_missing": delivery_docs_missing,
        "duplicated_rows_detected": duplicates,
        "actions_recommended": actions,
        "summary": {
            "mock_parameter_scans": _group_missing(missing_mock_scans),
            "real_parameter_scans": _group_missing(missing_real_scans),
            "cooja": {"total": len(missing_cooja)},
            "delivery_docs": {"total": len(delivery_docs_missing)},
        },
    }


def write_markdown(audit: dict[str, Any], path: Path) -> None:
    lines = [
        "# Final Thesis Symmetry Audit",
        "",
        f"- Generated at: `{audit['generated_at']}`",
        f"- Mock main matrix complete: `{audit['complete_main_matrix_mock']}` ({audit['main_matrix_counts']['mock']['completed']}/{audit['main_matrix_counts']['mock']['expected']})",
        f"- Real main matrix complete: `{audit['complete_main_matrix_real']}` ({audit['main_matrix_counts']['real']['completed']}/{audit['main_matrix_counts']['real']['expected']})",
        f"- Missing mock parameter scans: `{len(audit['missing_mock_parameter_scans'])}`",
        f"- Missing real parameter scans: `{len(audit['missing_real_parameter_scans'])}`",
        f"- Missing Cooja outputs: `{len(audit['missing_cooja_outputs'])}`",
        f"- Missing delivery docs: `{len(audit['delivery_docs_missing'])}`",
        f"- Duplicate row findings: `{len(audit['duplicated_rows_detected'])}`",
        "",
        "## Recommended Actions",
        "",
    ]
    for action in audit["actions_recommended"]:
        lines.append(f"- {action}")
    lines.extend(["", "## Missing Parameter Scan Summary", ""])
    for scope in ["mock_parameter_scans", "real_parameter_scans"]:
        summary = audit["summary"][scope]
        lines.append(f"- {scope}: total `{summary['total']}`, by method `{summary['by_method']}`, by dataset `{summary['by_dataset']}`")
    if audit["missing_cooja_outputs"]:
        lines.extend(["", "## Cooja Missing Outputs", ""])
        for item in audit["missing_cooja_outputs"]:
            lines.append(f"- `{item.get('reason')}` {item.get('method', '')} {item.get('mode', '')} {item.get('seed', '')}".rstrip())
    if audit["delivery_docs_missing"]:
        lines.extend(["", "## Delivery Docs Missing", ""])
        for item in audit["delivery_docs_missing"]:
            lines.append(f"- `{item.get('name')}` -> `{item.get('expected_file')}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_REPORT.mkdir(parents=True, exist_ok=True)
    audit = build_audit()
    json_path = OUT_REPORT / "final_symmetry_audit.json"
    md_path = OUT_REPORT / "final_symmetry_audit.md"
    json_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(audit, md_path)
    print(f"symmetry_audit_json={_rel(json_path)}")
    print(f"symmetry_audit_md={_rel(md_path)}")
    print(f"mock_main={audit['main_matrix_counts']['mock']['completed']}/{audit['main_matrix_counts']['mock']['expected']}")
    print(f"real_main={audit['main_matrix_counts']['real']['completed']}/{audit['main_matrix_counts']['real']['expected']}")
    print(f"missing_mock_parameter_scans={len(audit['missing_mock_parameter_scans'])}")
    print(f"missing_real_parameter_scans={len(audit['missing_real_parameter_scans'])}")
    print(f"missing_cooja_outputs={len(audit['missing_cooja_outputs'])}")


if __name__ == "__main__":
    main()
