"""Canonical artifact path constants shared by audits, builders, and dashboards."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
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


def rel_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except Exception:
        return path.as_posix().replace("\\", "/")


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
