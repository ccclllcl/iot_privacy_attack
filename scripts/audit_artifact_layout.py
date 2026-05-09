#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit artifact layout and propose canonical delivery paths.

This script is read-only. It scans repository artifacts and writes a compact
layout inventory before or after artifact migration.
"""

from __future__ import annotations

import csv
import json
import re
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LAYOUT_DIR = ROOT / "outputs" / "summaries" / "layout"
SCAN_ROOTS = ["outputs", "data", "configs", "scripts", "docs"]

DATASETS = {"mock", "uci_har", "kasteren", "casas_hh101", "cooja"}
MODELS = {"lstm", "mlp", "random_forest"}
METHODS = {"adaptive_ldp", "ldp", "noise", "dummy_noise", "dummy_ldp", "dummy_adaptive_ldp"}
MODES = {"fixed_attacker", "retrain_attacker"}


def _run(cmd: list[str]) -> tuple[int, str, str]:
    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    )
    return proc.returncode, proc.stdout or "", proc.stderr or ""


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except Exception:
        return path.as_posix().replace("\\", "/")


def _tracked_files() -> set[str]:
    rc, out, err = _run(["git", "ls-files"])
    if rc != 0:
        raise RuntimeError(err.strip() or "git ls-files failed")
    return {line.strip().replace("\\", "/") for line in out.splitlines() if line.strip()}


def _iter_files() -> list[Path]:
    files: list[Path] = []
    for root_name in SCAN_ROOTS:
        root = ROOT / root_name
        if not root.exists():
            continue
        files.extend(p for p in root.rglob("*") if p.is_file())
    return sorted(files, key=lambda p: _rel(p))


def _state_name() -> str:
    after_markers = [
        ROOT / "outputs" / "summaries" / "layout" / "migration_map.csv",
        ROOT / "outputs" / "experiments",
    ]
    old_summary = ROOT / "outputs" / "reports" / "final_thesis"
    if any(p.exists() for p in after_markers) and not old_summary.exists():
        return "after"
    return "before"


def _category(path: str) -> str:
    p = path.replace("\\", "/")
    if p.startswith("outputs/summaries/final_thesis/") or p.startswith("outputs/reports/final_thesis/"):
        return "final summary artifacts"
    if p.startswith("outputs/figures/summaries/final_thesis/") or p.startswith("outputs/figures/final_thesis/"):
        return "final figures"
    if p.startswith(("outputs/defense/full_multiseed/", "outputs/reports/full_multiseed/", "data/processed/full_multiseed/", "data/defended/full_multiseed/", "outputs/models/full_multiseed/")):
        return "mock source artifacts"
    if p.startswith(("outputs/defense/real_public_benchmark/", "outputs/reports/real_public_benchmark/", "data/processed/real_public_benchmark/", "data/defended/real_public_benchmark/", "outputs/models/real_public_benchmark/")):
        return "real source artifacts"
    if p.startswith("outputs/defense/final_thesis/"):
        return "curated final source artifacts"
    if p.startswith("outputs/experiments/cooja/") or p.startswith("outputs/summaries/final_thesis/cooja/") or p.startswith("outputs/reports/final_thesis/cooja/") or p.startswith("configs/cooja"):
        return "Cooja artifacts"
    if p.startswith(("configs/generated/", "configs/generated_all_methods/", "configs/generated_real_public/")):
        return "generated configs"
    if p.startswith("outputs/experiments/"):
        return "canonical experiment artifacts"
    if p.startswith(("data/processed/mock/", "data/processed/uci_har/", "data/processed/kasteren/", "data/processed/casas_hh101/", "data/processed/imports/", "data/defended/mock/", "data/defended/uci_har/", "data/defended/kasteren/", "data/defended/casas_hh101/")):
        return "canonical data artifacts"
    if p.startswith(("outputs/models/mock/", "outputs/models/uci_har/", "outputs/models/kasteren/", "outputs/models/casas_hh101/")):
        return "canonical model artifacts"
    if p.startswith(("scripts/", "docs/")) or p in {"README.md", ".gitignore"}:
        return "scripts and docs"
    if "__pycache__" in p or ".pytest_cache" in p or ".mypy_cache" in p or p.endswith((".tmp", ".bak")) or Path(p).name in {".DS_Store", "Thumbs.db"} or Path(p).name.startswith("~$"):
        return "temp_or_cache"
    return "other"


def _infer(path: str) -> dict[str, str]:
    p = path.replace("\\", "/")
    parts = p.split("/")
    inferred = {
        "inferred_dataset": "",
        "inferred_seed": "",
        "inferred_model": "",
        "inferred_method": "",
        "inferred_mode": "",
        "inferred_role": "",
    }
    for part in parts:
        if part in DATASETS:
            inferred["inferred_dataset"] = part
        if re.fullmatch(r"seed_\d+", part):
            inferred["inferred_seed"] = part
        if part in MODELS:
            inferred["inferred_model"] = part
        if part in METHODS:
            inferred["inferred_method"] = part
        if part in MODES:
            inferred["inferred_mode"] = part
    name = Path(p).name
    if "parameter_scan" in p or "comparisons" in p:
        inferred["inferred_role"] = "parameter_scan"
    elif "baseline" in name:
        inferred["inferred_role"] = "baseline"
    elif name in {"confusion.json", "classification_report.txt", "trace.json", "defense_report.json", "metrics.json"}:
        inferred["inferred_role"] = "main_matrix"
    elif name.endswith((".png", ".jpg", ".jpeg")):
        inferred["inferred_role"] = "figure"
    elif name.endswith((".csv", ".json", ".md")) and ("summary" in name or "audit" in name or "manifest" in name):
        inferred["inferred_role"] = "summary"
    return inferred


def _canonical_scan_path(path: str) -> str:
    p = path.replace("\\", "/")
    name = Path(p).name
    m = re.search(r"outputs/defense/full_multiseed/seed_(\d+)/(adaptive_ldp|ldp|noise)/comparisons/(lstm|mlp)_(fixed_attacker|retrain_attacker)_comparison_results\.csv$", p)
    if m:
        seed, method, model, mode = m.groups()
        return f"outputs/experiments/mock/seed_{seed}/{model}/{method}/{mode}/parameter_scan/comparison_results.csv"
    m = re.search(r"outputs/defense/real_public_benchmark/(uci_har|kasteren|casas_hh101)/seed_(\d+)/(adaptive_ldp|ldp|noise)/comparisons/(lstm|mlp)_(fixed_attacker|retrain_attacker)_comparison_results\.csv$", p)
    if m:
        dataset, seed, method, model, mode = m.groups()
        return f"outputs/experiments/{dataset}/seed_{seed}/{model}/{method}/{mode}/parameter_scan/comparison_results.csv"
    if name == "comparison_results.csv" and "/comparisons/" in p:
        return ""
    return ""


def _proposed_new_path(path: str) -> str:
    p = path.replace("\\", "/")
    scan = _canonical_scan_path(p)
    if scan:
        return scan
    m = re.search(r"outputs/defense/final_thesis/mock/seed_(\d+)/(lstm|mlp)/(adaptive_ldp|ldp|noise)/(fixed_attacker|retrain_attacker)/(confusion|classification_report|trace|defense_report)\.(json|txt)$", p)
    if m:
        seed, model, method, mode, stem, ext = m.groups()
        target_name = f"{stem}.{ext}"
        if stem == "confusion":
            target_name = "confusion.json"
        return f"outputs/experiments/mock/seed_{seed}/{model}/{method}/{mode}/{target_name}"
    m = re.search(r"outputs/defense/final_thesis/real/(uci_har|kasteren|casas_hh101)/seed_(\d+)/(lstm|mlp)/(adaptive_ldp|ldp|noise)/(fixed_attacker|retrain_attacker)/(confusion|classification_report|trace|defense_report)\.(json|txt)$", p)
    if m:
        dataset, seed, model, method, mode, stem, ext = m.groups()
        return f"outputs/experiments/{dataset}/seed_{seed}/{model}/{method}/{mode}/{stem}.{ext}"
    if p.startswith("outputs/reports/final_thesis/"):
        return p.replace("outputs/reports/final_thesis/", "outputs/summaries/final_thesis/", 1)
    if p.startswith("outputs/figures/final_thesis/"):
        return p.replace("outputs/figures/final_thesis/", "outputs/figures/summaries/final_thesis/", 1)
    if p.startswith("data/processed/full_multiseed/"):
        return p.replace("data/processed/full_multiseed/", "data/processed/mock/", 1)
    if p.startswith("data/defended/full_multiseed/"):
        return p.replace("data/defended/full_multiseed/", "data/defended/mock/", 1)
    if p.startswith("data/processed/real_public_benchmark/"):
        return p.replace("data/processed/real_public_benchmark/", "data/processed/", 1)
    if p.startswith("data/defended/real_public_benchmark/"):
        return p.replace("data/defended/real_public_benchmark/", "data/defended/", 1)
    if p.startswith("outputs/models/full_multiseed/"):
        return p.replace("outputs/models/full_multiseed/", "outputs/models/mock/", 1)
    if p.startswith("outputs/models/real_public_benchmark/"):
        return p.replace("outputs/models/real_public_benchmark/", "outputs/models/", 1)
    if p.startswith("configs/generated_all_methods/"):
        return p.replace("configs/generated_all_methods/", "configs/generated/mock/", 1)
    if p.startswith("configs/generated_real_public/"):
        return p.replace("configs/generated_real_public/", "configs/generated/", 1)
    return ""


def _action(path: str, category: str, proposed: str, size: int) -> tuple[str, str]:
    p = path.replace("\\", "/")
    if size == 0 and not p.startswith(("outputs/summaries/final_thesis/", "outputs/reports/final_thesis/")):
        return "delete", "Zero-byte artifact outside required final summaries."
    if p.startswith(("outputs/experiments/", "outputs/summaries/", "outputs/figures/summaries/", "outputs/figures/experiments/", "configs/generated/", "data/processed/", "data/defended/", "outputs/models/mock/", "outputs/models/uci_har/", "outputs/models/kasteren/", "outputs/models/casas_hh101/")):
        return "keep", "Already under canonical layout."
    if proposed:
        return "move", "Legacy path can be migrated to canonical layout."
    if category in {"scripts and docs", "Cooja artifacts"}:
        return "keep", "Code, documentation, or retained Cooja configuration."
    if category in {"mock source artifacts", "real source artifacts", "curated final source artifacts", "final summary artifacts", "final figures", "generated configs"}:
        return "investigate", "Legacy artifact class needs migration or explicit retention."
    if category == "temp_or_cache":
        return "delete", "Temporary/cache artifact."
    return "optional", "Not part of final canonical layout."


def build_rows() -> list[dict[str, Any]]:
    tracked = _tracked_files()
    rows: list[dict[str, Any]] = []
    for full in _iter_files():
        rel = _rel(full)
        category = _category(rel)
        proposed = _proposed_new_path(rel)
        size = full.stat().st_size
        action, reason = _action(rel, category, proposed, size)
        row = {
            "old_path": rel,
            "size_bytes": size,
            "tracked_by_git": rel in tracked,
            "category": category,
            **_infer(rel),
            "source_or_summary": "summary" if "summary" in category or "figures" in category else "source",
            "proposed_new_path": proposed,
            "action": action,
            "reason": reason,
        }
        rows.append(row)
    return rows


def write_outputs(rows: list[dict[str, Any]]) -> None:
    state = _state_name()
    LAYOUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = LAYOUT_DIR / f"artifact_layout_{state}.json"
    csv_path = LAYOUT_DIR / f"artifact_layout_{state}.csv"
    md_path = LAYOUT_DIR / f"artifact_layout_{state}.md"
    fields = [
        "old_path",
        "size_bytes",
        "tracked_by_git",
        "category",
        "inferred_dataset",
        "inferred_seed",
        "inferred_model",
        "inferred_method",
        "inferred_mode",
        "inferred_role",
        "source_or_summary",
        "proposed_new_path",
        "action",
        "reason",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "state": state,
        "total_files": len(rows),
        "tracked_files": sum(1 for r in rows if r["tracked_by_git"]),
        "total_bytes": sum(int(r["size_bytes"]) for r in rows),
        "by_category": dict(Counter(str(r["category"]) for r in rows)),
        "by_action": dict(Counter(str(r["action"]) for r in rows)),
    }
    json_path.write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        f"# Artifact Layout Audit ({state})",
        "",
        f"- Generated at: `{summary['generated_at']}`",
        f"- Files scanned: `{summary['total_files']}`",
        f"- Git tracked files: `{summary['tracked_files']}`",
        f"- Total bytes: `{summary['total_bytes']}`",
        "",
        "## By Category",
        "",
    ]
    for key, count in sorted(summary["by_category"].items()):
        lines.append(f"- {key}: `{count}`")
    lines.extend(["", "## By Action", ""])
    for key, count in sorted(summary["by_action"].items()):
        lines.append(f"- {key}: `{count}`")
    lines.extend(["", "## Notes", ""])
    lines.append("- Cooja packet/byte/IAT NaN values are treated as metric-field limitations, not missing experiment runs.")
    lines.append("- Proposed paths are structural migration targets; this audit does not move or delete files.")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"artifact_layout_{state}={_rel(json_path)}")


def main() -> None:
    write_outputs(build_rows())


if __name__ == "__main__":
    main()
