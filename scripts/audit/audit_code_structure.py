#!/usr/bin/env python3
"""Audit Python code layout and suggest responsibility-based packages."""

from __future__ import annotations

import ast
import json
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs" / "summaries" / "final_thesis"
JSON_OUT = OUT_DIR / "code_structure_audit.json"
MD_OUT = OUT_DIR / "code_structure_audit.md"
SCAN_ROOTS = ["src", "apps", "experiments", "scripts", "tools", "docs"]


CATEGORY_BY_PATH = [
    ("src/config.py", "config_core", "src/core/config.py", True),
    ("src/utils.py", "config_core", "src/core/utils.py", True),
    ("src/plotting.py", "config_core", "src/core/plotting.py", True),
    ("src/preprocess.py", "data_processing", "src/data/preprocess.py", True),
    ("src/features.py", "feature_engineering", "src/data/features.py", True),
    ("src/dataset.py", "dataset_wrappers", "src/data/dataset.py", True),
    ("src/train.py", "training", "src/training/trainer.py", True),
    ("src/evaluate.py", "baseline_evaluation", "src/evaluation/evaluator.py", True),
    ("src/defense_eval.py", "defense_evaluation", "src/evaluation/defense_evaluator.py", True),
    ("src/experiment_compare.py", "parameter_scan", "src/evaluation/comparison.py", True),
    ("src/dashboard_paths.py", "artifact_paths", "src/dashboard/paths.py", True),
    ("src/dashboard_io.py", "artifact_io", "src/dashboard/io.py", True),
    ("src/dashboard_runner.py", "dashboard_runner", "src/dashboard/runner.py", True),
]


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except Exception:
        return path.as_posix().replace("\\", "/")


def iter_code_files() -> list[Path]:
    files: list[Path] = []
    for root_name in SCAN_ROOTS:
        root = ROOT / root_name
        if not root.exists():
            continue
        files.extend(p for p in root.rglob("*") if p.is_file() and p.suffix in {".py", ".ps1", ".md"})
    return sorted(files, key=rel)


def imports_for(path: Path) -> list[str]:
    if path.suffix != ".py":
        return []
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
        return []
    out: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = "." * int(node.level or 0) + (node.module or "")
            out.append(module)
    return sorted(set(out))


def classify(path: Path) -> tuple[str, str, bool, bool, str]:
    p = rel(path)
    for exact, category, proposed, wrapper in CATEGORY_BY_PATH:
        if p == exact:
            return category, proposed, False, wrapper, "Compatibility wrapper for old import path."
    if p.startswith("src/core/"):
        return "config_core", p, False, False, "Core configuration or utility package."
    if p.startswith("src/data/preprocess.py"):
        return "data_processing", p, False, False, "Data preprocessing."
    if p.startswith("src/data/features.py"):
        return "feature_engineering", p, False, False, "Feature engineering."
    if p.startswith("src/data/dataset.py"):
        return "dataset_wrappers", p, False, False, "Dataset wrappers."
    if p.startswith("src/training/"):
        return "training", p, False, False, "Training logic."
    if p.startswith("src/evaluation/defense_evaluator.py"):
        return "defense_evaluation", p, False, False, "Defense-side attack evaluation."
    if p.startswith("src/evaluation/comparison.py"):
        return "parameter_scan", p, False, False, "Parameter scan logic."
    if p.startswith("src/evaluation/"):
        return "baseline_evaluation", p, False, False, "Baseline evaluation."
    if p.startswith("src/dashboard/paths.py"):
        return "artifact_paths", p, False, False, "Dashboard path helper."
    if p.startswith("src/dashboard/io.py"):
        return "artifact_io", p, False, False, "Dashboard artifact IO."
    if p.startswith("src/dashboard/runner.py"):
        return "dashboard_runner", p, False, False, "Dashboard single-combination runner."
    if p.startswith("src/artifacts/"):
        return "artifact_paths", p, False, False, "Canonical artifact helper."
    if p.startswith("src/defenses/defense_pipeline.py"):
        return "defense_pipeline", p, False, False, "Defense pipeline package."
    if p.startswith("src/defenses/"):
        return "defense_algorithms", p, False, False, "Defense algorithm package."
    if p.startswith("src/models/"):
        return "model_definitions", p, False, False, "Model definition package."
    if p.startswith("src/edge/"):
        return "defense_algorithms", p, False, False, "Edge budget helper."
    if p.startswith("apps/dashboard.py"):
        return "dashboard", p, False, False, "Formal Streamlit dashboard."
    if p.startswith("apps/legacy/"):
        return "legacy_ui", p, False, False, "Retained legacy UI."
    if p.startswith("apps/"):
        return "legacy_ui", "apps/legacy/" + Path(p).name, False, False, "Legacy UI should live under apps/legacy."
    if p.startswith("experiments/demo/"):
        return "experiment_cli", p, False, False, "Dashboard demo CLI."
    if p.startswith("experiments/core/"):
        return "experiment_cli", p, False, False, "Single-step experiment CLI."
    if p.startswith("experiments/batches/"):
        return "batch_runner", p, False, False, "Batch runner."
    if p.startswith("experiments/real_public/run_import"):
        return "real_dataset_import", p, False, False, "Real dataset import."
    if p.startswith("experiments/real_public/"):
        return "real_dataset_import", p, False, False, "Real dataset workflow."
    if p.startswith("experiments/cooja/"):
        return "cooja_eval", p, False, False, "Cooja evaluation."
    if p.startswith("scripts/final_thesis/"):
        return "final_summary_build", p, False, False, "Final summary builder."
    if p.startswith("scripts/audit/") or p.startswith("scripts/audit_"):
        return "audit", p, False, False, "Audit script."
    if p.startswith("scripts/"):
        return "maintenance", p, False, False, "Compatibility wrapper or maintenance helper."
    if p.startswith("tools/maintenance/refresh_web_assets.py"):
        return "maintenance", p, True, False, "Old web-assets helper is not used by the dashboard."
    if p.startswith("tools/cooja/"):
        return "maintenance", p, False, False, "Cooja maintenance helper."
    if p.startswith("tools/"):
        return "maintenance", p, False, False, "Maintenance helper."
    if p.startswith("docs/"):
        return "maintenance", p, False, False, "Documentation."
    return "unknown", p, False, False, "No specific classification rule."


def main() -> None:
    rows: list[dict[str, Any]] = []
    for path in iter_code_files():
        category, proposed, delete, wrapper, reason = classify(path)
        rows.append(
            {
                "path": rel(path),
                "size_bytes": path.stat().st_size,
                "category": category,
                "imports": imports_for(path),
                "proposed_new_path": proposed,
                "delete_recommended": delete,
                "compatibility_wrapper_needed": wrapper,
                "reason": reason,
            }
        )

    root_mix = [
        "config.py: compatibility wrapper for configuration core",
        "preprocess.py: compatibility wrapper for data processing",
        "features.py: compatibility wrapper for feature engineering",
        "dataset.py: compatibility wrapper for Dataset wrappers",
        "train.py: compatibility wrapper for training",
        "evaluate.py: compatibility wrapper for baseline evaluation",
        "defense_eval.py: compatibility wrapper for defense-side attack evaluation",
        "experiment_compare.py: compatibility wrapper for parameter scans",
        "dashboard_paths.py/dashboard_io.py/dashboard_runner.py: compatibility wrappers for dashboard helpers",
        "utils.py/plotting.py: compatibility wrappers for common utilities",
    ]
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "files": rows,
        "src_root_mixed_responsibilities": root_mix,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    by_cat: dict[str, int] = {}
    for row in rows:
        by_cat[row["category"]] = by_cat.get(row["category"], 0) + 1
    lines = [
        "# Code Structure Audit",
        "",
        f"- Generated at: `{payload['generated_at']}`",
        f"- Files scanned: `{len(rows)}`",
        "",
        "## Category Counts",
        "",
    ]
    for cat, count in sorted(by_cat.items()):
        lines.append(f"- `{cat}`: {count}")
    lines.extend(["", "## src Root Compatibility Wrappers", ""])
    lines.extend([f"- {item}" for item in root_mix])
    lines.extend(["", "## Move/Delete Recommendations", ""])
    for row in rows:
        if row["path"] != row["proposed_new_path"] or row["delete_recommended"] or row["compatibility_wrapper_needed"]:
            action = "delete" if row["delete_recommended"] else "move"
            wrapper_note = "; wrapper needed" if row["compatibility_wrapper_needed"] else ""
            lines.append(f"- `{row['path']}` -> `{row['proposed_new_path']}` ({action}{wrapper_note})")
    MD_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"code_structure_audit={rel(JSON_OUT)}")


if __name__ == "__main__":
    main()
