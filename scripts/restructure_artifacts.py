#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Normalize artifact layout by dataset/seed/model/method/mode.

The script moves or copies already-completed artifacts into the canonical
delivery layout. It does not run experiments.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "outputs" / "experiments"
SUMMARIES = ROOT / "outputs" / "summaries" / "final_thesis"
LAYOUT = ROOT / "outputs" / "summaries" / "layout"
SUMMARY_FIGURES = ROOT / "outputs" / "figures" / "summaries" / "final_thesis"
EXPERIMENT_FIGURES = ROOT / "outputs" / "figures" / "experiments"

SEEDS = [42, 123, 2026]
MODELS = ["lstm", "mlp"]
METHODS = ["adaptive_ldp", "ldp", "noise"]
MODES = ["fixed_attacker", "retrain_attacker"]
REAL_DATASETS = ["uci_har", "kasteren", "casas_hh101"]
COOJA_METHODS = ["dummy_noise", "dummy_ldp", "dummy_adaptive_ldp"]
PARAM_ROWS = {"ldp": 5, "noise": 4, "adaptive_ldp": 6}


@dataclass
class Options:
    apply: bool
    dry_run: bool
    skip_existing: bool
    write_map: bool


class Migrator:
    def __init__(self, options: Options) -> None:
        self.options = options
        self.rows: list[dict[str, Any]] = []
        self.deleted: list[dict[str, Any]] = []

    def rel(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(ROOT).as_posix()
        except Exception:
            return path.as_posix().replace("\\", "/")

    def record(self, old: Path | str, new: Path | str, action: str, reason: str, size: int = 0) -> None:
        old_s = self.rel(old) if isinstance(old, Path) else old
        new_s = self.rel(new) if isinstance(new, Path) else new
        self.rows.append(
            {
                "old_path": old_s,
                "new_path": new_s,
                "action": action,
                "reason": reason,
                "size_bytes": size,
            }
        )

    def _safe_inside_root(self, path: Path) -> bool:
        try:
            path.resolve().relative_to(ROOT.resolve())
            return True
        except Exception:
            return False

    def copy_file(self, src: Path, dst: Path, reason: str, transform: str | None = None) -> bool:
        if not src.exists() or not src.is_file():
            self.record(src, dst, "missing", "Source file does not exist.")
            return False
        if self.options.skip_existing and dst.exists() and dst.stat().st_size > 0:
            self.record(src, dst, "keep_existing", reason, src.stat().st_size)
            return True
        self.record(src, dst, "copy", reason, src.stat().st_size)
        if self.options.apply:
            dst.parent.mkdir(parents=True, exist_ok=True)
            if transform == "canonicalize_csv":
                self._copy_csv_with_canonical_paths(src, dst)
            else:
                shutil.copy2(src, dst)
        return True

    def write_json(self, dst: Path, obj: Any, reason: str) -> None:
        self.record("generated", dst, "write", reason)
        if self.options.apply:
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

    def write_text(self, dst: Path, text: str, reason: str) -> None:
        self.record("generated", dst, "write", reason)
        if self.options.apply:
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(text, encoding="utf-8")

    def delete_path(self, path: Path, reason: str) -> None:
        if not path.exists():
            return
        if not self._safe_inside_root(path):
            raise RuntimeError(f"Refusing to delete outside repository: {path}")
        size = self._path_size(path)
        self.record(path, "", "delete", reason, size)
        self.deleted.append({"path": self.rel(path), "reason": reason, "size_bytes": size})
        if self.options.apply:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()

    def move_tree_merge(self, src: Path, dst: Path, reason: str) -> None:
        if not src.exists():
            return
        for file in sorted([p for p in src.rglob("*") if p.is_file()]):
            rel = file.relative_to(src)
            target = dst / rel
            if self.options.skip_existing and target.exists() and target.stat().st_size > 0:
                self.record(file, target, "keep_existing", reason, file.stat().st_size)
                continue
            self.record(file, target, "move", reason, file.stat().st_size)
            if self.options.apply:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(file), str(target))
        if self.options.apply and src.exists():
            shutil.rmtree(src, ignore_errors=True)

    def _path_size(self, path: Path) -> int:
        if path.is_file():
            return path.stat().st_size
        return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())

    def _copy_csv_with_canonical_paths(self, src: Path, dst: Path) -> None:
        with src.open("r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
            fields = list(rows[0].keys()) if rows else list((csv.DictReader(f)).fieldnames or [])
        if not fields:
            dst.write_text("", encoding="utf-8")
            return
        dst_rel = self.rel(dst)
        for row in rows:
            if "source_file" in row:
                row["source_file"] = dst_rel
            if "model_source" in row:
                row["model_source"] = _canonical_model_path(row.get("model_source", ""))
        with dst.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


def _read_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _canonical_model_path(path: str) -> str:
    p = str(path).replace("\\", "/")
    p = re.sub(r"^.*?/outputs/models/", "outputs/models/", p)
    p = p.replace("outputs/models/full_multiseed/", "outputs/models/mock/")
    p = p.replace("outputs/models/real_public_benchmark/", "outputs/models/")
    return p


def _baseline_old_root(dataset: str, seed: int) -> Path:
    if dataset == "mock":
        return ROOT / "outputs" / "defense" / "full_multiseed" / f"seed_{seed}"
    return ROOT / "outputs" / "defense" / "real_public_benchmark" / dataset / f"seed_{seed}"


def _main_old_dir(dataset: str, seed: int, model: str, method: str, mode: str) -> Path:
    if dataset == "mock":
        return ROOT / "outputs" / "defense" / "final_thesis" / "mock" / f"seed_{seed}" / model / method / mode
    return ROOT / "outputs" / "defense" / "final_thesis" / "real" / dataset / f"seed_{seed}" / model / method / mode


def _experiment_dir(dataset: str, seed: int, model: str, method: str, mode: str) -> Path:
    return EXPERIMENTS / dataset / f"seed_{seed}" / model / method / mode


def _old_scan_file(dataset: str, seed: int, model: str, method: str, mode: str) -> Path:
    if dataset == "mock":
        return ROOT / "outputs" / "defense" / "full_multiseed" / f"seed_{seed}" / method / "comparisons" / f"{model}_{mode}_comparison_results.csv"
    return ROOT / "outputs" / "defense" / "real_public_benchmark" / dataset / f"seed_{seed}" / method / "comparisons" / f"{model}_{mode}_comparison_results.csv"


def _old_legacy_scan_file(dataset: str, seed: int, method: str) -> Path:
    if dataset == "mock":
        return ROOT / "outputs" / "defense" / "full_multiseed" / f"seed_{seed}" / method / "comparisons" / "comparison_results.csv"
    return ROOT / "outputs" / "defense" / "real_public_benchmark" / dataset / f"seed_{seed}" / method / "comparisons" / "comparison_results.csv"


def _baseline_source(dataset: str, seed: int, model: str) -> Path | None:
    root = _baseline_old_root(dataset, seed)
    for method in METHODS:
        path = root / method / "json_reports" / f"{model}_baseline_confusion_test.json"
        if path.exists():
            return path
    return None


def migrate_summaries(m: Migrator) -> None:
    old = ROOT / "outputs" / "reports" / "final_thesis"
    if old.exists():
        for src in sorted([p for p in old.rglob("*") if p.is_file()]):
            rel = src.relative_to(old)
            if src.stat().st_size == 0 and src.name in {"cooja_feature_importance.csv", "cooja_top_confusions.csv"}:
                m.record(src, "", "drop_optional", "Empty Cooja optional diagnostic is not part of final delivery.", src.stat().st_size)
                continue
            m.copy_file(src, SUMMARIES / rel, "Move final thesis summaries to outputs/summaries/final_thesis.")
    old_fig = ROOT / "outputs" / "figures" / "final_thesis"
    if old_fig.exists():
        for src in sorted([p for p in old_fig.rglob("*") if p.is_file()]):
            m.copy_file(src, SUMMARY_FIGURES / src.relative_to(old_fig), "Move final thesis figures to outputs/figures/summaries/final_thesis.")


def migrate_main_matrix(m: Migrator) -> None:
    datasets = ["mock"] + REAL_DATASETS
    for dataset in datasets:
        for seed in SEEDS:
            for model in MODELS:
                baseline_dst = EXPERIMENTS / dataset / f"seed_{seed}" / model / "baseline"
                baseline_src = _baseline_source(dataset, seed, model)
                baseline_manifest: dict[str, Any] = {
                    "dataset": dataset,
                    "seed": seed,
                    "model": model,
                    "role": "baseline",
                    "source_files": [],
                    "unavailable_reason": {},
                }
                if baseline_src:
                    m.copy_file(baseline_src, baseline_dst / "baseline_confusion.json", "Normalize baseline confusion file.")
                    m.copy_file(baseline_src, baseline_dst / "baseline_metrics.json", "Use baseline confusion metrics as baseline metrics artifact.")
                    baseline = _read_json(baseline_src) or {}
                    text = (
                        f"dataset={dataset}\nseed={seed}\nmodel={model}\n"
                        f"baseline_acc={baseline.get('accuracy', '')}\n"
                        f"baseline_f1_macro={baseline.get('f1_macro', '')}\n"
                        "note=Derived from baseline_confusion.json; no separate legacy text report existed.\n"
                    )
                    m.write_text(baseline_dst / "baseline_classification_report.txt", text, "Write derived baseline report from existing metrics.")
                    baseline_manifest["source_files"].append({"role": "baseline_confusion", "old_path": m.rel(baseline_src), "new_path": m.rel(baseline_dst / "baseline_confusion.json")})
                    baseline_manifest["unavailable_reason"]["baseline_trace.json"] = "No separate baseline trace existed in the legacy artifacts."
                else:
                    baseline_manifest["unavailable_reason"]["baseline_confusion.json"] = "No legacy baseline confusion source found."
                    baseline_manifest["unavailable_reason"]["baseline_trace.json"] = "No legacy baseline trace source found."
                m.write_json(baseline_dst / "source_manifest.json", baseline_manifest, "Write baseline source manifest.")

                for method in METHODS:
                    for mode in MODES:
                        src_dir = _main_old_dir(dataset, seed, model, method, mode)
                        dst_dir = _experiment_dir(dataset, seed, model, method, mode)
                        manifest = {
                            "dataset": dataset,
                            "seed": seed,
                            "model": model,
                            "method": method,
                            "mode": mode,
                            "role": "main_matrix",
                            "baseline_dir": m.rel(baseline_dst),
                            "source_files": [],
                            "unavailable_reason": {},
                        }
                        for name in ["confusion.json", "classification_report.txt", "trace.json", "defense_report.json"]:
                            src = src_dir / name
                            dst = dst_dir / name
                            if m.copy_file(src, dst, "Normalize main matrix artifact."):
                                manifest["source_files"].append({"role": name, "old_path": m.rel(src), "new_path": m.rel(dst)})
                            else:
                                manifest["unavailable_reason"][name] = "Missing from legacy curated final thesis source."
                        conf_dst = dst_dir / "confusion.json"
                        if (src_dir / "confusion.json").exists():
                            m.copy_file(src_dir / "confusion.json", dst_dir / "metrics.json", "Expose defended confusion metrics under standard metrics.json.")
                            manifest["source_files"].append({"role": "metrics.json", "old_path": m.rel(src_dir / "confusion.json"), "new_path": m.rel(dst_dir / "metrics.json")})
                        m.write_json(dst_dir / "source_manifest.json", manifest, "Write main matrix source manifest.")


def migrate_parameter_scans(m: Migrator) -> None:
    for dataset in ["mock"] + REAL_DATASETS:
        for seed in SEEDS:
            for method in METHODS:
                for model in MODELS:
                    for mode in MODES:
                        src = _old_scan_file(dataset, seed, model, method, mode)
                        if not src.exists() and method in {"ldp", "noise"} and model == "lstm" and mode == "fixed_attacker":
                            src = _old_legacy_scan_file(dataset, seed, method)
                        dst_dir = _experiment_dir(dataset, seed, model, method, mode) / "parameter_scan"
                        dst = dst_dir / "comparison_results.csv"
                        copied = m.copy_file(src, dst, "Move parameter scan CSV to matching experiment combination.", transform="canonicalize_csv")
                        rows = _read_csv_rows(src) if src.exists() else []
                        summary = {
                            "dataset": dataset,
                            "seed": seed,
                            "model": model,
                            "method": method,
                            "mode": mode,
                            "row_count": len(rows),
                            "expected_rows": PARAM_ROWS[method],
                            "complete": len(rows) == PARAM_ROWS[method],
                            "source_file": m.rel(dst),
                            "old_path": m.rel(src),
                        }
                        m.write_json(dst_dir / "scan_summary.json", summary, "Write parameter scan summary.")
                        m.write_json(
                            dst_dir / "scan_trace.json",
                            {"source_file": m.rel(dst), "old_path": m.rel(src), "generated_at": datetime.now().isoformat(timespec="seconds")},
                            "Write parameter scan trace.",
                        )
                        if method == "adaptive_ldp" and copied:
                            profiles = [
                                {
                                    "profile_name": r.get("profile_name"),
                                    "epsilon_min": r.get("epsilon_min"),
                                    "epsilon_max": r.get("epsilon_max"),
                                    "weight_sensitivity": r.get("weight_sensitivity"),
                                    "weight_traffic": r.get("weight_traffic"),
                                    "use_edge_budget_cap": r.get("use_edge_budget_cap"),
                                    "edge_inverse_budget_cap": r.get("edge_inverse_budget_cap"),
                                }
                                for r in rows
                            ]
                            m.write_json(dst_dir / "profile_config.json", {"profiles": profiles}, "Write adaptive profile scan configuration.")
                        copy_scan_plots(m, dataset, seed, model, method, mode, src.parent if src.exists() else None, dst_dir)


def copy_scan_plots(m: Migrator, dataset: str, seed: int, model: str, method: str, mode: str, old_dir: Path | None, dst_dir: Path) -> None:
    if old_dir is None or not old_dir.exists():
        return
    candidates = {
        "plot_accuracy.png": [
            old_dir / f"{model}_{mode}_{method}_accuracy.png",
            old_dir / "adaptive_profile_vs_accuracy.png",
            old_dir / "epsilon_vs_accuracy.png",
            old_dir / "noise_vs_accuracy.png",
        ],
        "plot_distortion.png": [
            old_dir / f"{model}_{mode}_{method}_distortion.png",
            old_dir / "adaptive_profile_vs_distortion.png",
            old_dir / "epsilon_vs_distortion.png",
            old_dir / "distortion_vs_noise.png",
        ],
    }
    fig_base = EXPERIMENT_FIGURES / dataset / f"seed_{seed}" / model / method / mode
    for target_name, srcs in candidates.items():
        for src in srcs:
            if src.exists():
                m.copy_file(src, dst_dir / target_name, "Move parameter scan plot into parameter_scan directory.")
                m.copy_file(src, fig_base / target_name, "Move experiment diagnostic plot into figures/experiments.")
                break


def migrate_cooja(m: Migrator) -> None:
    report = ROOT / "outputs" / "defense" / "final_thesis" / "cooja" / "eval" / "defense_eval_report.json"
    canonical_report = EXPERIMENTS / "cooja" / "eval" / "defense_eval_report.json"
    if report.exists():
        m.copy_file(report, canonical_report, "Move Cooja evaluation report into canonical experiments tree.")
    per_seed = SUMMARIES / "cooja" / "cooja_per_seed.csv"
    rows = _read_csv_rows(per_seed)
    for row in rows:
        method = row.get("method", "")
        mode = row.get("mode", "")
        try:
            seed = int(row.get("seed", ""))
        except Exception:
            continue
        if method not in COOJA_METHODS or mode not in MODES:
            continue
        dst_dir = EXPERIMENTS / "cooja" / f"seed_{seed}" / "random_forest" / method / mode
        metrics = {
            "dataset": "cooja",
            "seed": seed,
            "model": "random_forest",
            "method": method,
            "mode": mode,
            "baseline_acc": row.get("baseline_acc"),
            "defended_acc": row.get("defended_acc"),
            "accuracy_drop": row.get("accuracy_drop"),
            "baseline_f1_macro": row.get("baseline_f1_macro"),
            "defended_f1_macro": row.get("defended_f1_macro"),
            "source_radio_log": row.get("source_radio_log"),
            "source_app_log": row.get("source_app_log"),
        }
        m.write_json(dst_dir / "metrics.json", metrics, "Write Cooja per-seed metrics from existing summary.")
        m.write_json(
            dst_dir / "source_manifest.json",
            {
                "dataset": "cooja",
                "seed": seed,
                "model": "random_forest",
                "method": method,
                "mode": mode,
                "role": "cooja_per_seed",
                "source_files": [{"role": "per_seed_row", "old_path": m.rel(per_seed), "new_path": m.rel(dst_dir / "metrics.json")}],
                "unavailable_reason": {
                    "packet_byte_iat_metrics": "Current Cooja logs do not expose enough labeled packet fields for packet/byte/IAT proxy metrics.",
                    "energy_delay_metrics": "No real energy or end-to-end delay measurements are available.",
                },
            },
            "Write Cooja source manifest without fabricating unavailable metrics.",
        )


def migrate_data(m: Migrator) -> None:
    for seed in SEEDS:
        m.move_tree_merge(ROOT / "data" / "processed" / "full_multiseed" / f"seed_{seed}", ROOT / "data" / "processed" / "mock" / f"seed_{seed}", "Move mock processed data by dataset/seed.")
        m.move_tree_merge(ROOT / "data" / "defended" / "full_multiseed" / f"seed_{seed}", ROOT / "data" / "defended" / "mock" / f"seed_{seed}", "Move mock defended data by dataset/seed/method.")
    for dataset in REAL_DATASETS:
        for seed in SEEDS:
            m.move_tree_merge(ROOT / "data" / "processed" / "real_public_benchmark" / dataset / f"seed_{seed}", ROOT / "data" / "processed" / dataset / f"seed_{seed}", "Move real processed data by dataset/seed.")
            m.move_tree_merge(ROOT / "data" / "defended" / "real_public_benchmark" / dataset / f"seed_{seed}", ROOT / "data" / "defended" / dataset / f"seed_{seed}", "Move real defended data by dataset/seed/method.")


def migrate_models(m: Migrator) -> None:
    migrate_model_root(m, ROOT / "outputs" / "models" / "full_multiseed", "mock")
    real_root = ROOT / "outputs" / "models" / "real_public_benchmark"
    for dataset in REAL_DATASETS:
        migrate_model_root(m, real_root / dataset, dataset)


def migrate_model_root(m: Migrator, root: Path, dataset: str) -> None:
    if not root.exists():
        return
    for seed_dir in sorted(root.glob("seed_*")):
        if not seed_dir.is_dir():
            continue
        seed_name = seed_dir.name
        for file in sorted(seed_dir.glob("best_*.pt")):
            name = file.name
            model = "lstm" if name.startswith("best_lstm") else "mlp" if name.startswith("best_mlp") else "unknown"
            if model == "unknown":
                continue
            method_match = re.search(r"_(adaptive_ldp|ldp|noise)_defended_retrain\.pt$", name)
            if method_match:
                method = method_match.group(1)
                dst = ROOT / "outputs" / "models" / dataset / seed_name / model / method / "retrain_attacker" / name
            else:
                dst = ROOT / "outputs" / "models" / dataset / seed_name / model / "baseline" / name
            if m.options.skip_existing and dst.exists():
                m.record(file, dst, "keep_existing", "Model already exists at canonical path.", file.stat().st_size)
            else:
                m.record(file, dst, "move", "Move model artifact to dataset/seed/model layout.", file.stat().st_size)
                if m.options.apply:
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(file), str(dst))
        param_root = seed_dir / "parameter_scans"
        if param_root.exists():
            for file in sorted(param_root.rglob("*.pt")):
                rel = file.relative_to(param_root)
                parts = rel.parts
                method = parts[0] if parts else "unknown"
                model = "lstm" if "lstm" in file.name else "mlp" if "mlp" in file.name else "unknown"
                dst = ROOT / "outputs" / "models" / dataset / seed_name / model / method / "parameter_scan" / Path(*parts[1:])
                if m.options.skip_existing and dst.exists():
                    m.record(file, dst, "keep_existing", "Parameter scan model already exists at canonical path.", file.stat().st_size)
                else:
                    m.record(file, dst, "move", "Move parameter scan model artifact to canonical model tree.", file.stat().st_size)
                    if m.options.apply:
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(file), str(dst))
    if m.options.apply and root.exists():
        shutil.rmtree(root, ignore_errors=True)


def migrate_configs(m: Migrator) -> None:
    old_mock = ROOT / "configs" / "generated_all_methods"
    if old_mock.exists():
        for src in sorted(old_mock.glob("*.yaml")):
            name = src.name
            if name.startswith("import."):
                dst = ROOT / "configs" / "generated" / "imports" / name.replace("import.", "")
                m.copy_file(src, dst, "Move import helper config under generated/imports.")
                continue
            parsed = _parse_mock_config_name(name)
            if parsed is None:
                m.copy_file(src, ROOT / "configs" / "generated" / "mock" / "unclassified" / name, "Move unclassified mock generated config.")
                continue
            seed, method, model, mode = parsed
            models = [model] if model else MODELS
            for model_name in models:
                if method == "baseline":
                    dst = ROOT / "configs" / "generated" / "mock" / f"seed_{seed}" / model_name / "baseline.yaml"
                else:
                    dst = ROOT / "configs" / "generated" / "mock" / f"seed_{seed}" / model_name / method / f"{mode}.yaml"
                m.copy_file(src, dst, "Move mock generated config to canonical path.")
    old_real = ROOT / "configs" / "generated_real_public"
    if old_real.exists():
        for src in sorted(old_real.glob("*.yaml")):
            parsed = _parse_real_config_name(src.name)
            if parsed is None:
                m.copy_file(src, ROOT / "configs" / "generated" / "unclassified" / src.name, "Move unclassified real generated config.")
                continue
            dataset, seed, method, model, mode = parsed
            models = [model] if model else MODELS
            for model_name in models:
                if method == "baseline":
                    dst = ROOT / "configs" / "generated" / dataset / f"seed_{seed}" / model_name / "baseline.yaml"
                else:
                    dst = ROOT / "configs" / "generated" / dataset / f"seed_{seed}" / model_name / method / f"{mode}.yaml"
                m.copy_file(src, dst, "Move real generated config to canonical path.")
    old_seed = ROOT / "configs" / "generated"
    for src in sorted(old_seed.glob("default.seed_*.yaml")):
        m.copy_file(src, old_seed / "mock" / src.name.replace("default.", ""), "Move legacy generated seed config under mock.")
    cooja_dst = ROOT / "configs" / "generated" / "cooja"
    for src in sorted((ROOT / "configs").glob("cooja*.json")):
        m.copy_file(src, cooja_dst / src.name, "Copy Cooja config/template into generated/cooja index.")


def _parse_mock_config_name(name: str) -> tuple[int, str, str | None, str] | None:
    m = re.fullmatch(r"default\.seed_(\d+)\.base\.yaml", name)
    if m:
        return int(m.group(1)), "baseline", None, "baseline"
    m = re.fullmatch(r"default\.seed_(\d+)\.(adaptive_ldp|ldp|noise)\.yaml", name)
    if m:
        return int(m.group(1)), m.group(2), None, "fixed_attacker"
    m = re.fullmatch(r"default\.seed_(\d+)\.(adaptive_ldp|ldp|noise)\.(lstm|mlp)\.retrain\.yaml", name)
    if m:
        return int(m.group(1)), m.group(2), m.group(3), "retrain_attacker"
    return None


def _parse_real_config_name(name: str) -> tuple[str, int, str, str | None, str] | None:
    m = re.fullmatch(r"(uci_har|kasteren|casas_hh101)\.seed_(\d+)\.base\.yaml", name)
    if m:
        return m.group(1), int(m.group(2)), "baseline", None, "baseline"
    m = re.fullmatch(r"(uci_har|kasteren|casas_hh101)\.seed_(\d+)\.(adaptive_ldp|ldp|noise)\.yaml", name)
    if m:
        return m.group(1), int(m.group(2)), m.group(3), None, "fixed_attacker"
    m = re.fullmatch(r"(uci_har|kasteren|casas_hh101)\.seed_(\d+)\.(adaptive_ldp|ldp|noise)\.(lstm|mlp)\.retrain\.yaml", name)
    if m:
        return m.group(1), int(m.group(2)), m.group(3), m.group(4), "retrain_attacker"
    return None


def cleanup_old_paths(m: Migrator) -> None:
    old_paths = [
        ROOT / "outputs" / "defense" / "full_multiseed",
        ROOT / "outputs" / "defense" / "real_public_benchmark",
        ROOT / "outputs" / "defense" / "final_thesis",
        ROOT / "outputs" / "reports" / "full_multiseed",
        ROOT / "outputs" / "reports" / "real_public_benchmark",
        ROOT / "outputs" / "reports" / "final_thesis",
        ROOT / "outputs" / "figures" / "final_thesis",
        ROOT / "data" / "processed" / "full_multiseed",
        ROOT / "data" / "processed" / "real_public_benchmark",
        ROOT / "data" / "defended" / "full_multiseed",
        ROOT / "data" / "defended" / "real_public_benchmark",
        ROOT / "outputs" / "models" / "full_multiseed",
        ROOT / "outputs" / "models" / "real_public_benchmark",
        ROOT / "configs" / "generated_all_methods",
        ROOT / "configs" / "generated_real_public",
    ]
    for path in old_paths:
        m.delete_path(path, "Legacy batch-name path migrated to canonical layout.")
    for path in [
        ROOT / "outputs" / "ui" / "run_history.jsonl",
        ROOT / ".pytest_cache",
        ROOT / ".mypy_cache",
    ]:
        m.delete_path(path, "Temporary/cache artifact.")
    for pattern in ["__pycache__", "*.tmp", "*.bak", "~$*"]:
        for path in ROOT.rglob(pattern):
            if ".git" not in path.parts:
                m.delete_path(path, "Temporary/cache artifact.")
    if m.options.apply:
        readme = ROOT / "outputs" / "reports" / "README.md"
        readme.parent.mkdir(parents=True, exist_ok=True)
        readme.write_text(
            "# Reports path migrated\n\n"
            "Final thesis summaries moved to `outputs/summaries/final_thesis/`.\n"
            "The old `outputs/reports/final_thesis/` package was removed to avoid duplicate delivery roots.\n",
            encoding="utf-8",
        )


def write_map_and_report(m: Migrator) -> None:
    LAYOUT.mkdir(parents=True, exist_ok=True)
    map_path = LAYOUT / "migration_map.csv"
    fields = ["old_path", "new_path", "action", "reason", "size_bytes"]
    with map_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(m.rows)
    deleted_bytes = sum(int(row.get("size_bytes", 0)) for row in m.deleted)
    report = [
        "# Artifact Layout Migration Report",
        "",
        f"- Generated at: `{datetime.now().isoformat(timespec='seconds')}`",
        "- Experiments were not rerun; artifacts were moved, copied, or summarized from existing outputs.",
        "- Canonical experiment root: `outputs/experiments/`",
        "- Canonical summary root: `outputs/summaries/final_thesis/`",
        "- Canonical summary figure root: `outputs/figures/summaries/final_thesis/`",
        f"- Migration rows: `{len(m.rows)}`",
        f"- Deleted/migrated legacy path bytes: `{deleted_bytes}`",
        "",
        "## Deleted Legacy Roots",
        "",
    ]
    if m.deleted:
        for row in m.deleted:
            report.append(f"- `{row['path']}`: {row['reason']}")
    else:
        report.append("- None")
    report.extend(
        [
            "",
            "## Cooja Note",
            "",
            "Cooja artifacts were only moved into the canonical structure. Packet/byte/IAT, real energy, and real end-to-end delay metrics were not fabricated.",
        ]
    )
    (LAYOUT / "migration_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def run(options: Options) -> Migrator:
    m = Migrator(options)
    migrate_summaries(m)
    migrate_main_matrix(m)
    migrate_parameter_scans(m)
    migrate_cooja(m)
    migrate_data(m)
    migrate_models(m)
    migrate_configs(m)
    if options.apply:
        cleanup_old_paths(m)
    if options.write_map:
        write_map_and_report(m)
    return m


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--apply", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--write-map", action="store_true")
    args = parser.parse_args()
    options = Options(apply=args.apply, dry_run=args.dry_run, skip_existing=args.skip_existing, write_map=args.write_map)
    migrator = run(options)
    print(f"migration_rows={len(migrator.rows)}")
    print(f"mode={'apply' if args.apply else 'dry-run'}")
    if args.write_map:
        print("migration_map=outputs/summaries/layout/migration_map.csv")


if __name__ == "__main__":
    main()
