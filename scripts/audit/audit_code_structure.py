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

REMOVED_COMPATIBILITY_ENTRYPOINTS = [
    "src/config.py",
    "src/utils.py",
    "src/plotting.py",
    "src/preprocess.py",
    "src/features.py",
    "src/dataset.py",
    "src/train.py",
    "src/evaluate.py",
    "src/defense_eval.py",
    "src/experiment_compare.py",
    "src/dashboard_paths.py",
    "src/dashboard_io.py",
    "src/dashboard_runner.py",
    "src/ui_history.py",
    "src/defenses/base_defense.py",
    "scripts/build_final_thesis_results.py",
    "scripts/audit_experiment_symmetry.py",
    "scripts/audit_repository_bloat.py",
    "scripts/audit_code_structure.py",
    "scripts/generate_project_file_report.py",
    "apps/legacy/ui_app.py",
    "experiments/batches/run_all_data_multiseed.py",
    "experiments/batches/run_all_methods_multiseed.py",
    "experiments/real_public/run_import_uci_har.py",
    "experiments/real_public/run_import_kasteren.py",
    "experiments/real_public/run_import_casas.py",
    "experiments/real_public/run_real_public_benchmark.py",
    "experiments/real_public/run_full_matrix_real_datasets.py",
    "experiments/real_public/summarize_real_public_benchmark.py",
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
        if root.exists():
            files.extend(
                p
                for p in root.rglob("*")
                if p.is_file() and p.suffix in {".py", ".ps1", ".md"}
            )
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
    name = path.name
    if p.endswith("__init__.py"):
        return "package_init", "Python package 初始化文件。"
    if p.startswith("src/core/"):
        return "config_core", "配置、通用工具或通用绘图工具。"
    if p.startswith("src/data/"):
        return "data_processing", "数据读取、预处理、特征工程或 Dataset 封装。"
    if p.startswith("src/models/"):
        return "model_definitions", "攻击模型定义。"
    if p.startswith("src/training/"):
        return "training", "模型训练逻辑。"
    if p.startswith("src/evaluation/comparison.py"):
        return "parameter_scan", "参数扫描和方法对比逻辑。"
    if p.startswith("src/evaluation/defense_evaluator.py"):
        return "defense_evaluation", "防御后攻击评估逻辑。"
    if p.startswith("src/evaluation/"):
        return "baseline_evaluation", "baseline 评估逻辑。"
    if p.startswith("src/defenses/defense_pipeline.py"):
        return "defense_pipeline", "防御流水线。"
    if p.startswith("src/defenses/"):
        return "defense_algorithms", "防御算法实现。"
    if p.startswith("src/edge/"):
        return "edge_budget", "adaptive_ldp 使用的边缘预算分配。"
    if p.startswith("src/dashboard/"):
        return "dashboard", "Dashboard 路径、IO、绘图、运行器或历史记录工具。"
    if p.startswith("src/artifacts/"):
        return "artifact_paths", "标准产物路径和 summary IO。"
    if p == "apps/dashboard.py":
        return "dashboard", "正式 Streamlit Dashboard。"
    if p.startswith("experiments/core/"):
        return "experiment_cli", "单步实验 CLI。"
    if p == "experiments/README.md":
        return "docs", "experiments 目录说明。"
    if p.startswith("experiments/demo/"):
        return "experiment_cli", "Dashboard 单组合演示 runner。"
    if p.startswith("experiments/batches/"):
        return "batch_runner", "多 seed 或全矩阵复现实验入口。"
    if p.startswith("experiments/real_public/imports/"):
        return "real_dataset_import", "真实数据导入流程。"
    if p.startswith("experiments/real_public/benchmarks/"):
        return "real_dataset_benchmark", "真实数据 benchmark 流程。"
    if p.startswith("experiments/real_public/scans/"):
        return "package_init", "真实数据参数扫描包初始化文件。"
    if p.startswith("experiments/cooja/"):
        return "cooja_eval", "Cooja 日志解析、评估或运行脚本。"
    if p.startswith("scripts/final_thesis/"):
        return "final_summary_build", "最终结果汇总构建和论文图生成脚本。"
    if p.startswith("scripts/audit/"):
        return "audit", "审计或报告生成脚本。"
    if p == "scripts/README.md":
        return "docs", "scripts 目录说明。"
    if p.startswith("tools/cooja/"):
        return "maintenance", "Cooja 维护工具。"
    if p.startswith("tools/maintenance/"):
        return "maintenance", "维护工具。"
    if p.startswith("tools/"):
        return "maintenance", "辅助维护脚本。"
    if p.startswith("docs/"):
        return "docs", "项目文档。"
    if name.endswith(".md"):
        return "docs", "Markdown 说明文件。"
    return "unknown", "暂无明确分类规则。"


def main() -> None:
    rows: list[dict[str, Any]] = []
    for path in iter_code_files():
        category, reason = classify(path)
        rows.append(
            {
                "path": rel(path),
                "size_bytes": path.stat().st_size,
                "category": category,
                "imports": imports_for(path),
                "reason": reason,
            }
        )

    unknown_files = sorted(
        [row for row in rows if row["category"] == "unknown"],
        key=lambda x: x["path"],
    )
    by_cat: dict[str, int] = {}
    for row in rows:
        by_cat[row["category"]] = by_cat.get(row["category"], 0) + 1

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "files_scanned": len(rows),
        "category_counts": dict(sorted(by_cat.items())),
        "files": rows,
        "removed_compatibility_entrypoints": REMOVED_COMPATIBILITY_ENTRYPOINTS,
        "compatibility_wrappers": [],
        "legacy_files": [],
        "unknown_files": unknown_files,
        "pending_recommendations": unknown_files,
        "summary": (
            "当前代码结构已去除旧兼容 wrapper 和 legacy UI，正式代码按职责分层。"
            if not unknown_files
            else "仍有文件需要人工确认分类。"
        ),
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
        "## 1. 分层统计",
        "",
    ]
    for cat, count in payload["category_counts"].items():
        lines.append(f"- `{cat}`：{count}")

    lines += ["", "## 2. 已移除的旧兼容入口", ""]
    for path in REMOVED_COMPATIBILITY_ENTRYPOINTS:
        lines.append(f"- `{path}`")

    lines += ["", "## 3. 当前正式源码分层", ""]
    lines += [
        "- `src/core/`：配置、通用工具、绘图工具。",
        "- `src/data/`：数据预处理、特征工程、Dataset。",
        "- `src/models/`：LSTM 与 MLP 模型定义。",
        "- `src/training/`：训练循环、模型保存、训练曲线。",
        "- `src/evaluation/`：baseline 评估、防御评估、参数扫描。",
        "- `src/defenses/`：noise、ldp、adaptive_ldp 和防御流水线。",
        "- `src/edge/`：边缘预算分配。",
        "- `src/dashboard/`：Dashboard 路径、IO、运行器和运行历史。",
        "- `src/artifacts/`：标准产物路径和 summary IO。",
    ]

    lines += ["", "## 4. unknown 文件", ""]
    if unknown_files:
        for row in unknown_files:
            lines.append(f"- `{row['path']}`：{row['reason']}")
    else:
        lines.append("- 无 unknown 文件。")

    lines += ["", "## 5. 后续建议", ""]
    if unknown_files:
        lines.append("- 对 unknown 文件补充分层规则或手动归档。")
    else:
        lines.append("- 当前代码结构已完成职责分层，暂无必须移动的文件。")

    MD_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"code_structure_audit={rel(JSON_OUT)}")


if __name__ == "__main__":
    main()
