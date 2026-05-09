#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit tracked repository artifacts and cleanup candidates.

This script is read-only. It treats the normalized artifact layout as the
official delivery structure.
"""

from __future__ import annotations

import csv
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "summaries" / "final_thesis"
CSV_OUT = OUT_DIR / "repository_bloat_audit.csv"
JSON_OUT = OUT_DIR / "repository_bloat_audit.json"
MD_OUT = OUT_DIR / "repository_bloat_audit.md"

LOCAL_PATH_PATTERNS = ["D:\\", "D:/", "姣曚笟璁捐姣曚笟璁捐", "\\\\wsl$"]
LEGACY_ROOTS = [
    "outputs/defense/full_multiseed/",
    "outputs/defense/real_public_benchmark/",
    "outputs/defense/final_thesis/",
    "outputs/reports/final_thesis/",
    "outputs/reports/full_multiseed/",
    "outputs/reports/real_public_benchmark/",
    "outputs/figures/final_thesis/",
    "data/processed/full_multiseed/",
    "data/processed/real_public_benchmark/",
    "data/defended/full_multiseed/",
    "data/defended/real_public_benchmark/",
    "outputs/models/full_multiseed/",
    "outputs/models/real_public_benchmark/",
    "configs/generated_all_methods/",
    "configs/generated_real_public/",
]


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


def _tracked_files() -> list[str]:
    rc, out, err = _run(["git", "ls-files"])
    if rc != 0:
        raise RuntimeError(err.strip() or "git ls-files failed")
    return [line.strip().replace("\\", "/") for line in out.splitlines() if line.strip()]


def _last_commit(path: str) -> tuple[str, str]:
    rc, out, _ = _run(["git", "log", "-1", "--format=%H|%cI", "--", path])
    if rc != 0 or not out.strip():
        return "", ""
    commit_hash, _, commit_time = out.strip().partition("|")
    return commit_time, commit_hash


def _reference_files() -> list[Path]:
    files: list[Path] = []
    for pattern in ("*.csv", "*.json", "*.md"):
        files.extend(OUT_DIR.rglob(pattern))
    for path in [
        ROOT / "scripts" / "build_final_thesis_results.py",
        ROOT / "scripts" / "audit_experiment_symmetry.py",
        ROOT / "README.md",
    ]:
        if path.exists():
            files.append(path)
    docs = ROOT / "docs"
    if docs.exists():
        files.extend(docs.glob("*.md"))
    return list({str(p.resolve()): p for p in files}.values())


def _load_reference_texts() -> str:
    chunks: list[str] = []
    for path in _reference_files():
        try:
            chunks.append(path.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            continue
    return "\n".join(chunks)


def _text_contains_local_path(path: Path) -> bool:
    if not path.exists() or not path.is_file() or path.stat().st_size > 8_000_000:
        return False
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return False
    return any(pattern in text for pattern in LOCAL_PATH_PATTERNS)


def _category(path: str) -> str:
    p = path.replace("\\", "/")
    name = Path(p).name
    if p.startswith("outputs/summaries/final_thesis/"):
        return "final_thesis_required"
    if p.startswith(("outputs/figures/summaries/final_thesis/", "outputs/figures/experiments/")):
        return "final_figure_required"
    if p.startswith("outputs/experiments/"):
        return "source_artifact_referenced"
    if p.startswith(tuple(LEGACY_ROOTS)) or "/dataset_matrix/" in p:
        return "legacy_batch_path"
    if p.startswith("configs/generated/"):
        return "generated_config"
    if p.startswith("outputs/ui/") or "run_history" in p:
        return "ui_history"
    if p.startswith("web_assets/"):
        return "web_asset"
    if name == "metrics.json" and p.startswith("outputs/reports/"):
        return "legacy_metrics_json"
    if "__pycache__" in p or ".pytest_cache" in p or ".mypy_cache" in p or name in {".DS_Store", "Thumbs.db"} or name.endswith((".tmp", ".bak")) or name.startswith("~"):
        return "temp_or_cache"
    if p.startswith(("docs/",)) or name.lower().endswith((".md", ".docx")):
        return "docs"
    if p.endswith((".py", ".yaml", ".yml", ".toml", ".json", ".ps1", ".sh")) or p.startswith(("src/", "scripts/", "experiments/", "configs/", "apps/", "tools/")):
        return "code"
    return "unknown"


def _recommendation(path: str, category: str, referenced: bool, hygiene: bool) -> tuple[str, str]:
    p = path.replace("\\", "/")
    if p == "outputs/reports/README.md":
        return "keep", "Small redirect explaining the migrated reports path."
    if category in {"legacy_batch_path", "legacy_metrics_json", "temp_or_cache", "ui_history"}:
        return "delete_candidate", "Legacy batch path or temporary artifact should not remain in normalized delivery layout."
    if referenced:
        reason = "Referenced by final summaries or delivery documentation."
        if hygiene:
            reason += " Contains local path content that should be reviewed."
        return "keep", reason
    if category in {"final_thesis_required", "final_figure_required", "source_artifact_referenced", "code", "docs", "generated_config"}:
        reason = "Required canonical delivery/code/documentation file."
        if hygiene:
            reason += " Contains local path content that should be reviewed."
        return "keep", reason
    if category in {"web_asset"}:
        return "review", "Web/UI asset; keep only if documented."
    return "review", "Unknown tracked file category."


def build_audit() -> list[dict[str, Any]]:
    reference_text = _load_reference_texts()
    rows: list[dict[str, Any]] = []
    for path in _tracked_files():
        full = ROOT / path
        if not full.exists():
            continue
        size = full.stat().st_size if full.exists() else 0
        last_time, last_hash = _last_commit(path)
        category = _category(path)
        referenced = path in reference_text or path.replace("/", "\\") in reference_text
        hygiene = full.exists() and full.is_file() and _text_contains_local_path(full)
        deletion_recommendation, reason = _recommendation(path, category, referenced, hygiene)
        if hygiene:
            reason = f"path_hygiene_issue; {reason}"
        rows.append(
            {
                "path": path,
                "file_size_bytes": size,
                "git_last_commit_time": last_time,
                "git_last_commit_hash": last_hash,
                "category": category,
                "referenced_by_final_thesis": referenced,
                "deletion_recommendation": deletion_recommendation,
                "reason": reason,
            }
        )
    return rows


def write_outputs(rows: list[dict[str, Any]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fields = [
        "path",
        "file_size_bytes",
        "git_last_commit_time",
        "git_last_commit_hash",
        "category",
        "referenced_by_final_thesis",
        "deletion_recommendation",
        "reason",
    ]
    with CSV_OUT.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "tracked_files": len(rows),
        "total_tracked_bytes": sum(int(r["file_size_bytes"]) for r in rows),
        "by_category": {},
        "by_recommendation": {},
        "path_hygiene_issues": [r for r in rows if "path_hygiene_issue" in str(r["reason"])],
        "delete_candidates": [r for r in rows if r["deletion_recommendation"] == "delete_candidate"],
    }
    for key in ("category", "deletion_recommendation"):
        counts: dict[str, int] = {}
        for row in rows:
            counts[str(row[key])] = counts.get(str(row[key]), 0) + 1
        summary[f"by_{key.replace('deletion_', '')}"] = dict(sorted(counts.items()))
    JSON_OUT.write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Repository Bloat Audit",
        "",
        f"- Generated at: `{summary['generated_at']}`",
        f"- Tracked files: `{summary['tracked_files']}`",
        f"- Total tracked bytes: `{summary['total_tracked_bytes']}`",
        f"- Delete candidates: `{len(summary['delete_candidates'])}`",
        f"- Path hygiene issues: `{len(summary['path_hygiene_issues'])}`",
        "",
        "## Path Hygiene",
        "",
        "- Cooja local WSL log paths may remain in Cooja limitations/source-log fields as provenance, not portable reproduction paths.",
        "- Portable Cooja configuration uses `configs/cooja_defense_dummy_logs.template.json`.",
        "",
        "## Delete Candidates",
        "",
    ]
    if summary["delete_candidates"]:
        for row in summary["delete_candidates"][:100]:
            lines.append(f"- `{row['path']}`: {row['reason']}")
    else:
        lines.append("- None")
    MD_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows = build_audit()
    write_outputs(rows)
    print(f"repository_bloat_audit={_rel(JSON_OUT)}")


if __name__ == "__main__":
    main()
