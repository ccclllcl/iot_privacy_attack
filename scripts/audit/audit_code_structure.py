#!/usr/bin/env python3
"""审计当前代码职责分层，并生成中文结构报告。"""

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

WRAPPER_TARGETS = {
    "src/config.py": "src/core/config.py",
    "src/utils.py": "src/core/utils.py",
    "src/plotting.py": "src/core/plotting.py",
    "src/preprocess.py": "src/data/preprocess.py",
    "src/features.py": "src/data/features.py",
    "src/dataset.py": "src/data/dataset.py",
    "src/train.py": "src/training/trainer.py",
    "src/evaluate.py": "src/evaluation/evaluator.py",
    "src/defense_eval.py": "src/evaluation/defense_evaluator.py",
    "src/experiment_compare.py": "src/evaluation/comparison.py",
    "src/dashboard_paths.py": "src/dashboard/paths.py",
    "src/dashboard_io.py": "src/dashboard/io.py",
    "src/dashboard_runner.py": "src/dashboard/runner.py",
    "src/ui_history.py": "src/dashboard/history.py",
    "src/defenses/base_defense.py": "src/defenses/base.py",
    "scripts/build_final_thesis_results.py": "scripts/final_thesis/build_final_thesis_results.py",
    "scripts/audit_experiment_symmetry.py": "scripts/audit/audit_experiment_symmetry.py",
    "scripts/audit_repository_bloat.py": "scripts/audit/audit_repository_bloat.py",
    "scripts/audit_code_structure.py": "scripts/audit/audit_code_structure.py",
    "scripts/generate_project_file_report.py": "scripts/audit/generate_project_file_report.py",
}

COMPLETED_MOVE_TARGETS = set(WRAPPER_TARGETS.values()) | {
    "apps/legacy/ui_app.py",
    "experiments/real_public/imports/run_import_uci_har.py",
    "experiments/real_public/imports/run_import_kasteren.py",
    "experiments/real_public/imports/run_import_casas.py",
    "experiments/real_public/benchmarks/run_real_public_benchmark.py",
    "experiments/real_public/benchmarks/run_full_matrix_real_datasets.py",
    "experiments/real_public/benchmarks/summarize_real_public_benchmark.py",
    "tools/cooja/rewrite_cooja_client_type.py",
    "tools/maintenance/refresh_confusion_matrices.py",
}

LEGACY_FILES = {
    "apps/legacy/ui_app.py": "旧 UI 占位入口，不作为正式 Dashboard。",
    "experiments/batches/run_all_methods_multiseed.py": "完整矩阵复现脚本，日常审查不运行。",
    "experiments/batches/run_all_data_multiseed.py": "批处理复现脚本，日常审查不运行。",
}


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except Exception:
        return path.as_posix().replace("\\", "/")


def iter_code_files() -> list[Path]:
    files: list[Path] = []
    for root_name in SCAN_ROOTS:
        root = ROOT / root_name
        if root.exists():
            files.extend(p for p in root.rglob("*") if p.is_file() and p.suffix in {".py", ".ps1", ".md"})
    return sorted(files, key=rel)


def imports_for(path: Path) -> list[str]:
    if path.suffix != ".py":
        return []
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
        return []
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append("." * int(node.level or 0) + (node.module or ""))
    return sorted(set(imports))


def classify(path: Path) -> tuple[str, str]:
    p = rel(path)
    if p.endswith("__init__.py"):
        return "package_init", "Python package 初始化文件。"
    if p in WRAPPER_TARGETS:
        return "compatibility_wrapper", "兼容旧 import 或旧命令入口。"
    if p in LEGACY_FILES or p.startswith("apps/legacy/"):
        return "legacy_file", LEGACY_FILES.get(p, "legacy 文件，保留但不是正式入口。")
    if p.startswith("src/core/"):
        return "config_core", "配置核心、通用工具或绘图工具。"
    if p.startswith("src/data/"):
        return "data_processing", "数据处理、特征工程或 Dataset 封装。"
    if p.startswith("src/models/"):
        return "model_definitions", "模型定义。"
    if p.startswith("src/training/"):
        return "training", "训练逻辑。"
    if p.startswith("src/evaluation/comparison.py"):
        return "parameter_scan", "参数扫描和对比实验逻辑。"
    if p.startswith("src/evaluation/defense_evaluator.py"):
        return "defense_evaluation", "防御后攻击评估。"
    if p.startswith("src/evaluation/"):
        return "baseline_evaluation", "baseline 评估。"
    if p.startswith("src/defenses/defense_pipeline.py"):
        return "defense_pipeline", "防御流水线。"
    if p.startswith("src/defenses/"):
        return "defense_algorithms", "防御算法。"
    if p.startswith("src/edge/"):
        return "defense_algorithms", "边缘预算分配。"
    if p.startswith("src/dashboard/"):
        return "dashboard", "Dashboard 工具。"
    if p.startswith("src/artifacts/"):
        return "artifact_paths", "canonical artifact 路径或 summary IO。"
    if p == "apps/dashboard.py":
        return "dashboard", "正式 Streamlit Dashboard。"
    if p.startswith("experiments/core/"):
        return "experiment_cli", "单步实验 CLI。"
    if p == "experiments/README.md":
        return "docs", "experiments 目录说明。"
    if p.startswith("experiments/demo/"):
        return "experiment_cli", "Dashboard demo runner。"
    if p.startswith("experiments/batches/"):
        return "batch_runner", "批处理复现脚本。"
    if p.startswith("experiments/real_public/imports/"):
        return "real_dataset_import", "真实数据导入流程。"
    if p.startswith("experiments/real_public/benchmarks/"):
        return "real_dataset_benchmark", "真实数据 benchmark 流程。"
    if p.startswith("experiments/real_public/"):
        return "compatibility_wrapper", "真实数据旧入口兼容 wrapper。"
    if p.startswith("experiments/cooja/"):
        return "cooja_eval", "Cooja 日志评估。"
    if p.startswith("scripts/final_thesis/"):
        return "final_summary_build", "最终结果汇总构建。"
    if p.startswith("scripts/audit/"):
        return "audit", "审计或报告生成脚本。"
    if p.startswith("scripts/"):
        return "compatibility_wrapper", "脚本兼容入口或脚本说明。"
    if p.startswith("tools/cooja/"):
        return "maintenance", "Cooja 维护工具。"
    if p.startswith("tools/maintenance/"):
        return "maintenance", "维护工具。"
    if p.startswith("tools/"):
        return "maintenance", "维护工具或未跟踪辅助脚本。"
    if p.startswith("docs/"):
        return "docs", "项目文档。"
    return "unknown", "暂无明确分类规则。"


def main() -> None:
    rows: list[dict[str, Any]] = []
    for path in iter_code_files():
        p = rel(path)
        category, reason = classify(path)
        target = WRAPPER_TARGETS.get(p, p)
        rows.append(
            {
                "path": p,
                "size_bytes": path.stat().st_size,
                "category": category,
                "imports": imports_for(path),
                "proposed_new_path": target,
                "reason": reason,
                "is_compatibility_wrapper": p in WRAPPER_TARGETS,
                "is_completed_move_target": p in COMPLETED_MOVE_TARGETS,
                "is_legacy_file": category == "legacy_file",
            }
        )

    pending = [
        row
        for row in rows
        if row["category"] == "unknown" or (row["path"] in WRAPPER_TARGETS and not (ROOT / row["proposed_new_path"]).exists())
    ]
    completed_moves = sorted(
        [{"old_path": old, "new_path": new} for old, new in WRAPPER_TARGETS.items() if (ROOT / new).exists()],
        key=lambda x: x["old_path"],
    )
    compatibility_wrappers = sorted(
        [row for row in rows if row["is_compatibility_wrapper"]],
        key=lambda x: x["path"],
    )
    unknown_files = sorted([row for row in rows if row["category"] == "unknown"], key=lambda x: x["path"])
    legacy_files = sorted([row for row in rows if row["is_legacy_file"]], key=lambda x: x["path"])

    by_cat: dict[str, int] = {}
    for row in rows:
        by_cat[row["category"]] = by_cat.get(row["category"], 0) + 1

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "files_scanned": len(rows),
        "category_counts": dict(sorted(by_cat.items())),
        "files": rows,
        "completed_moves": completed_moves,
        "compatibility_wrappers": compatibility_wrappers,
        "legacy_files": legacy_files,
        "unknown_files": unknown_files,
        "pending_recommendations": pending,
        "summary": "当前代码结构已完成职责分层，暂无必须移动的文件。" if not pending else "仍有文件需要人工确认。",
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 代码结构审计",
        "",
        f"- 生成时间：`{payload['generated_at']}`",
        f"- 扫描文件数：`{payload['files_scanned']}`",
        f"- 结论：{payload['summary']}",
        "",
        "## 1. 分类统计",
        "",
    ]
    for cat, count in payload["category_counts"].items():
        lines.append(f"- `{cat}`：{count}")

    lines += ["", "## 2. 已完成的移动", ""]
    if completed_moves:
        for item in completed_moves:
            lines.append(f"- `{item['old_path']}` -> `{item['new_path']}`")
    else:
        lines.append("- 暂无记录。")

    lines += ["", "## 3. 兼容 wrapper", ""]
    for row in compatibility_wrappers:
        lines.append(f"- `{row['path']}`：指向 `{row['proposed_new_path']}`。")

    lines += ["", "## 4. legacy 文件", ""]
    if legacy_files:
        for row in legacy_files:
            lines.append(f"- `{row['path']}`：{row['reason']}")
    else:
        lines.append("- 暂无 legacy 文件。")

    lines += ["", "## 5. unknown 文件", ""]
    if unknown_files:
        for row in unknown_files:
            lines.append(f"- `{row['path']}`：{row['reason']}")
    else:
        lines.append("- 无 unknown 文件。")

    lines += ["", "## 6. 仍待处理建议", ""]
    if pending:
        for row in pending:
            lines.append(f"- `{row['path']}`：{row['reason']}")
    else:
        lines.append("- 当前代码结构已完成职责分层，暂无必须移动的文件。")

    MD_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"code_structure_audit={rel(JSON_OUT)}")


if __name__ == "__main__":
    main()
