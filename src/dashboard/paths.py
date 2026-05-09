"""Dashboard 使用的 canonical artifact 路径工具。"""

from __future__ import annotations

from pathlib import Path

from src.artifacts.canonical_paths import (
    COOJA_METHODS,
    EXPERIMENT_FIGURE_ROOT,
    DATASETS,
    EXPERIMENT_ROOT,
    METHODS,
    MODES,
    MODELS,
    PROJECT_ROOT,
    REAL_DATASETS,
    SEEDS,
    SUMMARY_FIGURE_ROOT,
    SUMMARY_ROOT,
    baseline_path,
    cooja_path,
    experiment_path,
    figure_summary_path,
    parameter_scan_path,
    rel_path,
    summary_path,
)


def _existing_names(root: Path, fallback: list[str]) -> list[str]:
    if root.exists():
        names = sorted(p.name for p in root.iterdir() if p.is_dir())
        if names:
            return names
    return fallback


def list_datasets() -> list[str]:
    names = _existing_names(EXPERIMENT_ROOT, DATASETS)
    ordered = [x for x in DATASETS if x in names]
    return ordered + [x for x in names if x not in ordered]


def list_seeds(dataset: str) -> list[int]:
    root = EXPERIMENT_ROOT / dataset
    if root.exists():
        found: list[int] = []
        for child in root.iterdir():
            if child.is_dir() and child.name.startswith("seed_"):
                try:
                    found.append(int(child.name.split("_", 1)[1]))
                except ValueError:
                    pass
        if found:
            return sorted(found)
    return SEEDS.copy()


def list_models(dataset: str) -> list[str]:
    if dataset == "cooja":
        return ["random_forest"]
    return MODELS.copy()


def list_methods(dataset: str) -> list[str]:
    if dataset == "cooja":
        return COOJA_METHODS.copy()
    return METHODS.copy()


def list_modes() -> list[str]:
    return MODES.copy()


def validate_selection(
    dataset: str,
    seed: int | None = None,
    model: str | None = None,
    method: str | None = None,
    mode: str | None = None,
) -> tuple[bool, str]:
    if dataset not in DATASETS:
        return False, f"不支持的 dataset：{dataset}"
    if seed is not None and int(seed) not in SEEDS:
        return False, f"不支持的 seed：{seed}"
    if model is not None and model not in list_models(dataset):
        return False, f"{dataset} 不支持该 model：{model}"
    if method is not None and method not in list_methods(dataset):
        return False, f"{dataset} 不支持该 method：{method}"
    if mode is not None and mode not in MODES:
        return False, f"不支持的 mode：{mode}"
    return True, ""
