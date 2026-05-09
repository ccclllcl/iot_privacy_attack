#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""审计仓库中可清理的冗余文件、跳转目录和占位文件。

脚本只读取仓库状态并生成审计报告，不移动实验产物，也不运行实验。
"""

from __future__ import annotations

import csv
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs" / "summaries" / "final_thesis"
CSV_OUT = OUT_DIR / "repository_bloat_audit.csv"
JSON_OUT = OUT_DIR / "repository_bloat_audit.json"
MD_OUT = OUT_DIR / "repository_bloat_audit.md"
REDIRECT_CSV_OUT = OUT_DIR / "redirect_placeholder_audit.csv"
REDIRECT_JSON_OUT = OUT_DIR / "redirect_placeholder_audit.json"
REDIRECT_MD_OUT = OUT_DIR / "redirect_placeholder_audit.md"
CLEANUP_REPORT_JSON = OUT_DIR / "redirect_placeholder_cleanup_report.json"

LOCAL_PATH_PATTERNS = ["D:\\", "D:/", "毕业设计毕业设计", "\\\\wsl$"]
LEGACY_ROOTS = [
    "outputs/defense/full_multiseed/",
    "outputs/defense/real_public_benchmark/",
    "outputs/defense/final_thesis/",
    "outputs/reports/",
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
SCAN_ROOTS = ("outputs/", "data/", "web_assets/", "docs/", "apps/", "scripts/", "experiments/", "tools/")
PROTECTED_ROOTS = (
    "outputs/experiments/",
    "outputs/summaries/final_thesis/",
    "outputs/summaries/layout/",
    "outputs/figures/summaries/final_thesis/",
    "outputs/figures/experiments/",
    "data/processed/",
    "data/defended/",
)
PLACEHOLDER_NAMES = {"README.md", "README.txt", ".gitkeep"}
PLACEHOLDER_KEYWORDS = [
    "migrated",
    "moved to",
    "old path",
    "legacy path",
    "redirect",
    "已迁移",
    "旧路径",
    "请查看另一个目录",
    "迁移到",
    "不再作为正式路径",
]
SUBSTANTIVE_SUFFIXES = {
    ".csv",
    ".json",
    ".png",
    ".jpg",
    ".jpeg",
    ".svg",
    ".pt",
    ".pth",
    ".npz",
    ".npy",
    ".yaml",
    ".yml",
    ".py",
    ".ipynb",
    ".docx",
}


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
    excluded = {
        CSV_OUT.resolve(),
        JSON_OUT.resolve(),
        MD_OUT.resolve(),
        REDIRECT_CSV_OUT.resolve(),
        REDIRECT_JSON_OUT.resolve(),
        REDIRECT_MD_OUT.resolve(),
        CLEANUP_REPORT_JSON.resolve(),
    }
    for path in [
        ROOT / "scripts" / "build_final_thesis_results.py",
        ROOT / "scripts" / "audit_experiment_symmetry.py",
        ROOT / "README.md",
        ROOT / ".gitignore",
    ]:
        if path.exists():
            files.append(path)
    docs = ROOT / "docs"
    if docs.exists():
        files.extend(docs.glob("*.md"))
    unique = []
    seen: set[str] = set()
    for path in files:
        try:
            resolved = path.resolve()
        except Exception:
            continue
        if resolved in excluded:
            continue
        key = str(resolved)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def _load_reference_map() -> dict[str, str]:
    refs: dict[str, str] = {}
    for path in _reference_files():
        try:
            refs[_rel(path)] = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
    return refs


def _referenced_by(path: str, reference_map: dict[str, str]) -> list[str]:
    hits: list[str] = []
    win_path = path.replace("/", "\\")
    for ref_path, text in reference_map.items():
        if path in text or win_path in text:
            hits.append(ref_path)
    return sorted(set(hits))


def _text_contains_local_path(path: Path) -> bool:
    if not path.exists() or not path.is_file() or path.stat().st_size > 8_000_000:
        return False
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return False
    return any(pattern in text for pattern in LOCAL_PATH_PATTERNS)


def _read_text(path: str) -> str:
    full = ROOT / path
    if not full.exists() or not full.is_file():
        return ""
    try:
        return full.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def _is_placeholder_readme(path: str) -> bool:
    name = Path(path).name
    if name not in {"README.md", "README.txt"}:
        return False
    text = _read_text(path).strip().lower()
    if not text:
        return True
    return any(keyword.lower() in text for keyword in PLACEHOLDER_KEYWORDS)


def _readme_excerpt(path: str, max_len: int = 220) -> str:
    text = " ".join(_read_text(path).strip().split())
    return text[:max_len]


def _extract_target_path(text: str) -> str:
    patterns = [
        r"`((?:outputs|data|docs|apps|scripts|experiments|tools|configs)/[^`]+)`",
        r"((?:outputs|data|docs|apps|scripts|experiments|tools|configs)/[A-Za-z0-9_./{}-]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).rstrip(".,)")
    return ""


def _is_scan_root(path: str) -> bool:
    return path.startswith(SCAN_ROOTS)


def _is_protected(path: str) -> bool:
    return path.startswith(PROTECTED_ROOTS)


def _is_substantive(path: str) -> bool:
    p = Path(path)
    if p.name in PLACEHOLDER_NAMES:
        return False
    if p.suffix.lower() in SUBSTANTIVE_SUFFIXES:
        return True
    full = ROOT / path
    return full.exists() and full.is_file() and full.stat().st_size > 0


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
    if category in {"legacy_batch_path", "legacy_metrics_json", "temp_or_cache", "ui_history"}:
        return "delete_candidate", "Legacy batch path or temporary artifact should not remain in normalized delivery layout."
    if referenced:
        reason = "Referenced by final summaries or delivery documentation."
        if hygiene:
            reason += " Contains local path content that should be reviewed."
        return "keep", reason
    if category in {"final_thesis_required", "final_figure_required", "source_artifact_referenced", "code", "docs", "generated_config"}:
        reason = "Required delivery/code/documentation file."
        if hygiene:
            reason += " Contains local path content that should be reviewed."
        return "keep", reason
    if category in {"web_asset"}:
        return "review", "Web/UI asset; keep when documented by Dashboard or project docs."
    return "review", "Unknown tracked file category."


def build_redirect_audit(tracked: list[str], reference_map: dict[str, str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    tracked_set = set(tracked)
    dirs: dict[str, list[str]] = {}
    for path in tracked:
        if not _is_scan_root(path):
            continue
        parts = path.split("/")
        for i in range(1, len(parts)):
            directory = "/".join(parts[:i]) + "/"
            dirs.setdefault(directory, []).append(path)

    for path in tracked:
        name = Path(path).name
        if name not in {"README.md", "README.txt"} and "placeholder" not in name.lower():
            continue
        if not _is_scan_root(path):
            continue
        if not _is_placeholder_readme(path):
            continue
        excerpt = _readme_excerpt(path)
        refs = _referenced_by(path, reference_map)
        blocking_refs = [ref for ref in refs if ref != ".gitignore"]
        recommendation = "keep" if _is_protected(path) or blocking_refs else "delete"
        reason = "README 只描述旧路径迁移或跳转，未承载实验结果、使用说明或研究解释。"
        if blocking_refs:
            reason = "该 README 被文档或脚本引用，清理前需要同步更新引用。"
        rows.append(
            {
                "path": path,
                "type": "placeholder_readme",
                "file_count": 1,
                "files": path,
                "readme_excerpt": excerpt,
                "target_path_if_any": _extract_target_path(excerpt),
                "referenced_by": "; ".join(refs),
                "recommendation": recommendation,
                "reason": reason,
            }
        )

    for directory, files in sorted(dirs.items()):
        if not files:
            continue
        if _is_protected(directory):
            continue
        direct_files = [f for f in files if f.count("/") == directory.rstrip("/").count("/") + 1]
        if not direct_files:
            continue
        names = {Path(f).name for f in direct_files}
        if names <= PLACEHOLDER_NAMES:
            readmes = [f for f in direct_files if Path(f).name in {"README.md", "README.txt"}]
            gitkeeps = [f for f in direct_files if Path(f).name == ".gitkeep"]
            substantive = [f for f in direct_files if _is_substantive(f)]
            refs = sorted({ref for f in direct_files for ref in _referenced_by(f, reference_map)})
            blocking_refs = [ref for ref in refs if ref != ".gitignore"]
            if readmes and all(_is_placeholder_readme(f) for f in readmes) and not substantive:
                excerpt = _readme_excerpt(readmes[0])
                row_type = "legacy_shell_directory" if directory.startswith(tuple(LEGACY_ROOTS)) else "redirect_only_directory"
                recommendation = "keep" if blocking_refs else "delete"
                reason = "目录仅包含跳转 README 或占位文件，没有真实产物。"
                if blocking_refs:
                    reason = "目录中的占位文件仍被引用，清理前需要同步更新引用。"
                rows.append(
                    {
                        "path": directory.rstrip("/"),
                        "type": row_type,
                        "file_count": len(direct_files),
                        "files": "; ".join(direct_files),
                        "readme_excerpt": excerpt,
                        "target_path_if_any": _extract_target_path(excerpt),
                        "referenced_by": "; ".join(refs),
                        "recommendation": recommendation,
                        "reason": reason,
                    }
                )
            elif gitkeeps and not substantive:
                rows.append(
                    {
                        "path": directory.rstrip("/"),
                        "type": "empty_or_gitkeep_only_directory",
                        "file_count": len(direct_files),
                        "files": "; ".join(direct_files),
                        "readme_excerpt": "",
                        "target_path_if_any": "",
                        "referenced_by": "; ".join(refs),
                        "recommendation": "keep" if refs else "delete",
                        "reason": "目录只包含 .gitkeep 或空占位文件。",
                    }
                )
    # 去重：文件行和目录行都保留，但同一路径只出现一次。
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = (str(row["path"]), str(row["type"]))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return sorted(deduped, key=lambda r: (str(r["recommendation"]), str(r["path"])))


def _deleted_redirect_paths() -> list[str]:
    if not CLEANUP_REPORT_JSON.exists():
        return []
    try:
        data = json.loads(CLEANUP_REPORT_JSON.read_text(encoding="utf-8"))
    except Exception:
        return []
    paths: list[str] = []
    for item in data.get("deleted_items", []):
        if isinstance(item, dict) and item.get("path"):
            paths.append(str(item["path"]))
        elif isinstance(item, str):
            paths.append(item)
    return sorted(set(paths))


def build_bloat_audit(tracked: list[str], reference_map: dict[str, str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in tracked:
        full = ROOT / path
        if not full.exists():
            continue
        size = full.stat().st_size if full.exists() else 0
        last_time, last_hash = _last_commit(path)
        category = _category(path)
        refs = _referenced_by(path, reference_map)
        referenced = bool(refs)
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


def _write_redirect_outputs(rows: list[dict[str, Any]]) -> None:
    fields = [
        "path",
        "type",
        "file_count",
        "files",
        "readme_excerpt",
        "target_path_if_any",
        "referenced_by",
        "recommendation",
        "reason",
    ]
    with REDIRECT_CSV_OUT.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "rows": rows,
        "summary": {
            "total": len(rows),
            "delete": sum(1 for r in rows if r["recommendation"] == "delete"),
            "keep": sum(1 for r in rows if r["recommendation"] == "keep"),
            "investigate": sum(1 for r in rows if r["recommendation"] == "investigate"),
        },
    }
    REDIRECT_JSON_OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 跳转目录与占位文件专项审计",
        "",
        f"- Generated at: `{payload['generated_at']}`",
        f"- Total findings: `{payload['summary']['total']}`",
        f"- Delete recommendations: `{payload['summary']['delete']}`",
        "",
        "## Findings",
        "",
    ]
    if rows:
        for row in rows:
            lines.append(f"- `{row['path']}` ({row['type']}): {row['recommendation']} - {row['reason']}")
    else:
        lines.append("- 未发现跳转目录、占位 README 或空壳 legacy 目录。")
    REDIRECT_MD_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(rows: list[dict[str, Any]], redirect_rows: list[dict[str, Any]]) -> None:
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

    redirect_only_delete_candidates = [
        r for r in redirect_rows if r["recommendation"] == "delete" and r["type"] in {"redirect_only_directory", "legacy_shell_directory"}
    ]
    placeholder_readme_delete_candidates = [
        r for r in redirect_rows if r["recommendation"] == "delete" and r["type"] == "placeholder_readme"
    ]
    empty_or_gitkeep_only_dirs = [
        r for r in redirect_rows if r["recommendation"] == "delete" and r["type"] == "empty_or_gitkeep_only_directory"
    ]
    deleted_redirect_paths = _deleted_redirect_paths()

    summary: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "tracked_files": len(rows),
        "total_tracked_bytes": sum(int(r["file_size_bytes"]) for r in rows),
        "by_category": {},
        "by_recommendation": {},
        "path_hygiene_issues": [r for r in rows if "path_hygiene_issue" in str(r["reason"])],
        "delete_candidates": [r for r in rows if r["deletion_recommendation"] == "delete_candidate"],
        "redirect_only_delete_candidates": redirect_only_delete_candidates,
        "placeholder_readme_delete_candidates": placeholder_readme_delete_candidates,
        "empty_or_gitkeep_only_dirs": empty_or_gitkeep_only_dirs,
        "legacy_shell_dirs": [r for r in redirect_rows if r["type"] == "legacy_shell_directory"],
        "deleted_redirect_paths": deleted_redirect_paths,
    }
    for key in ("category", "deletion_recommendation"):
        counts: dict[str, int] = {}
        for row in rows:
            counts[str(row[key])] = counts.get(str(row[key]), 0) + 1
        summary[f"by_{key.replace('deletion_', '')}"] = dict(sorted(counts.items()))

    JSON_OUT.write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_redirect_outputs(redirect_rows)

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
        "## 跳转目录与占位文件",
        "",
        f"- Redirect-only delete candidates: `{len(redirect_only_delete_candidates)}`",
        f"- Placeholder README delete candidates: `{len(placeholder_readme_delete_candidates)}`",
        f"- Empty/.gitkeep-only delete candidates: `{len(empty_or_gitkeep_only_dirs)}`",
        "",
    ]
    if deleted_redirect_paths:
        lines.append("### 已删除的跳转目录或占位文件")
        lines.append("")
        for path in deleted_redirect_paths:
            lines.append(f"- `{path}`")
        lines.append("")
    if redirect_only_delete_candidates or placeholder_readme_delete_candidates or empty_or_gitkeep_only_dirs:
        lines.append("### 仍需清理的候选项")
        lines.append("")
        for row in redirect_only_delete_candidates + placeholder_readme_delete_candidates + empty_or_gitkeep_only_dirs:
            lines.append(f"- `{row['path']}` ({row['type']}): {row['reason']}")
    else:
        lines.append("- 当前没有 redirect-only 或 placeholder README 删除候选。")
    lines.extend(["", "## Delete Candidates", ""])
    if summary["delete_candidates"]:
        for row in summary["delete_candidates"][:100]:
            lines.append(f"- `{row['path']}`: {row['reason']}")
    else:
        lines.append("- None")
    MD_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    tracked = _tracked_files()
    reference_map = _load_reference_map()
    rows = build_bloat_audit(tracked, reference_map)
    redirect_rows = build_redirect_audit(tracked, reference_map)
    write_outputs(rows, redirect_rows)
    print(f"repository_bloat_audit={_rel(JSON_OUT)}")


if __name__ == "__main__":
    main()
