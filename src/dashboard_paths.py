"""Canonical artifact paths used by the Streamlit dashboard."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = PROJECT_ROOT / "outputs" / "experiments"
SUMMARY_ROOT = PROJECT_ROOT / "outputs" / "summaries" / "final_thesis"
SUMMARY_FIGURE_ROOT = PROJECT_ROOT / "outputs" / "figures" / "summaries" / "final_thesis"
EXPERIMENT_FIGURE_ROOT = PROJECT_ROOT / "outputs" / "figures" / "experiments"

DATASETS = ["mock", "uci_har", "kasteren", "casas_hh101", "cooja"]
REAL_DATASETS = ["uci_har", "kasteren", "casas_hh101"]
SEEDS = [42, 123, 2026]
MODELS = ["lstm", "mlp"]
METHODS = ["adaptive_ldp", "ldp", "noise"]
MODES = ["fixed_attacker", "retrain_attacker"]
COOJA_METHODS = ["dummy_noise", "dummy_ldp", "dummy_adaptive_ldp"]


def _existing_names(root: Path, fallback: list[str]) -> list[str]:
    if root.exists():
        names = sorted(p.name for p in root.iterdir() if p.is_dir())
        if names:
            return names
    return fallback


def rel_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except Exception:
        return path.as_posix().replace("\\", "/")


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


def baseline_path(dataset: str, seed: int, model: str) -> Path:
    return EXPERIMENT_ROOT / dataset / f"seed_{seed}" / model / "baseline"


def experiment_path(dataset: str, seed: int, model: str, method: str, mode: str) -> Path:
    return EXPERIMENT_ROOT / dataset / f"seed_{seed}" / model / method / mode


def parameter_scan_path(dataset: str, seed: int, model: str, method: str, mode: str) -> Path:
    return experiment_path(dataset, seed, model, method, mode) / "parameter_scan"


def cooja_path(seed: int, dummy_method: str, mode: str) -> Path:
    return EXPERIMENT_ROOT / "cooja" / f"seed_{seed}" / "random_forest" / dummy_method / mode


def summary_path(name: str) -> Path:
    return SUMMARY_ROOT / name


def figure_summary_path(name: str) -> Path:
    return SUMMARY_FIGURE_ROOT / name


def validate_selection(
    dataset: str,
    seed: int | None = None,
    model: str | None = None,
    method: str | None = None,
    mode: str | None = None,
) -> tuple[bool, str]:
    if dataset not in DATASETS:
        return False, f"Unsupported dataset: {dataset}"
    if seed is not None and int(seed) not in SEEDS:
        return False, f"Unsupported seed: {seed}"
    if model is not None and model not in list_models(dataset):
        return False, f"Unsupported model for {dataset}: {model}"
    if method is not None and method not in list_methods(dataset):
        return False, f"Unsupported method for {dataset}: {method}"
    if mode is not None and mode not in MODES:
        return False, f"Unsupported mode: {mode}"
    return True, ""
