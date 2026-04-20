"""
批量参数扫描：在不同 epsilon 或噪声强度下重复「防御流水线 + 固定攻击者评估」，汇总 CSV 并自动作图。

用于论文中展示「隐私强度（epsilon↓）与攻击准确率、数据失真」之间的折中曲线。
"""

from __future__ import annotations

import copy
import csv
import logging
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt

from src.config import ExperimentConfig
from src.plotting import configure_matplotlib_english
from src.defense_eval import compute_fixed_attacker_metrics
from src.defenses.defense_pipeline import run_defense_pipeline
from src.utils import ensure_dir

logger = logging.getLogger(__name__)


def _clone_cfg(cfg: ExperimentConfig) -> ExperimentConfig:
    raw = copy.deepcopy(cfg.raw)
    return ExperimentConfig(raw, cfg.project_root)


def run_parameter_compare(
    cfg: ExperimentConfig,
    method: str,
    model_path: Path,
) -> Path:
    """
    method: ldp -> 扫描 compare.ldp_epsilon_list；noise -> 扫描 compare.noise_scale_list。

    输出：
    - outputs/defense/comparisons/comparison_results.csv
    - epsilon_vs_accuracy.png（LDP）或 noise 相关曲线
    - distortion_vs_noise.png（噪声扫描时：噪声强度 vs 失真 & 准确率）
    """
    configure_matplotlib_english()
    method = method.lower().strip()
    if method not in ("ldp", "noise"):
        raise ValueError("method 只能是 ldp 或 noise")

    cmp = cfg.nested("compare")
    defense_root = ensure_dir(cfg.path("paths", "defense_dir"))
    comp_dir = ensure_dir(defense_root / "comparisons")

    if not model_path.is_file():
        raise FileNotFoundError(f"未找到攻击者模型: {model_path}")

    rows: List[Dict[str, Any]] = []

    if method == "ldp":
        params: List[float] = [float(x) for x in cmp.get("ldp_epsilon_list", [0.5, 1.0, 2.0])]
        param_name = "epsilon"
        for eps in params:
            c = _clone_cfg(cfg)
            c.raw.setdefault("defense", {})
            c.raw["defense"]["method"] = "ldp"
            c.raw["defense"]["epsilon"] = float(eps)
            c.raw["defense"]["enabled"] = True
            summ = run_defense_pipeline(c)
            distort = summ.get("distortion", {})
            pair = compute_fixed_attacker_metrics(c, model_path)
            b = pair["baseline"]
            d = pair["defended"]
            rows.append(
                {
                    "method": "ldp",
                    "param_name": param_name,
                    "param_value": float(eps),
                    "baseline_accuracy": b["accuracy"],
                    "defended_accuracy": d["accuracy"],
                    "accuracy_drop": b["accuracy"] - d["accuracy"],
                    "defended_f1_macro": d["f1_macro"],
                    "mse": distort.get("mse", 0.0),
                    "mae": distort.get("mae", 0.0),
                    "pearson_r": distort.get("pearson_r", 0.0),
                }
            )
            logger.info(
                "LDP eps=%.3f | defended_acc=%.4f | mse=%.6f",
                eps,
                d["accuracy"],
                distort.get("mse", 0.0),
            )

        csv_path = comp_dir / "comparison_results.csv"
        _write_csv(csv_path, rows)
        _plot_epsilon_vs_accuracy(rows, comp_dir / "epsilon_vs_accuracy.png")
        _plot_epsilon_vs_distortion(rows, comp_dir / "epsilon_vs_distortion.png")
        return csv_path

    # noise
    scales: List[float] = [float(x) for x in cmp.get("noise_scale_list", [0.1, 0.3, 0.5])]
    param_name = "noise_scale"
    for sc in scales:
        c = _clone_cfg(cfg)
        c.raw.setdefault("defense", {})
        c.raw["defense"]["method"] = "noise"
        c.raw["defense"]["noise_scale"] = float(sc)
        c.raw["defense"]["enabled"] = True
        summ = run_defense_pipeline(c)
        distort = summ.get("distortion", {})
        pair = compute_fixed_attacker_metrics(c, model_path)
        b = pair["baseline"]
        d = pair["defended"]
        rows.append(
            {
                "method": "noise",
                "param_name": param_name,
                "param_value": float(sc),
                "baseline_accuracy": b["accuracy"],
                "defended_accuracy": d["accuracy"],
                "accuracy_drop": b["accuracy"] - d["accuracy"],
                "defended_f1_macro": d["f1_macro"],
                "mse": distort.get("mse", 0.0),
                "mae": distort.get("mae", 0.0),
                "pearson_r": distort.get("pearson_r", 0.0),
            }
        )
        logger.info(
            "Noise scale=%.3f | defended_acc=%.4f | mse=%.6f",
            sc,
            d["accuracy"],
            distort.get("mse", 0.0),
        )

    csv_path = comp_dir / "comparison_results.csv"
    _write_csv(csv_path, rows)
    _plot_distortion_vs_noise(rows, comp_dir / "distortion_vs_noise.png")
    _plot_noise_vs_accuracy(rows, comp_dir / "noise_scale_vs_accuracy.png")
    return csv_path


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _plot_epsilon_vs_accuracy(rows: List[Dict[str, Any]], out: Path) -> None:
    xs = [r["param_value"] for r in rows]
    ys = [r["defended_accuracy"] for r in rows]
    yb = rows[0]["baseline_accuracy"] if rows else 0.0
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(xs, ys, marker="o", label="Defended accuracy (fixed attacker)")
    ax.axhline(yb, color="gray", linestyle="--", label="Clean test baseline")
    ax.set_xlabel("epsilon (LDP; lower = stronger privacy)")
    ax.set_ylabel("Accuracy")
    ax.set_title("LDP epsilon vs. defended accuracy (fixed attacker)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _plot_epsilon_vs_distortion(rows: List[Dict[str, Any]], out: Path) -> None:
    xs = [r["param_value"] for r in rows]
    mse = [r["mse"] for r in rows]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(xs, mse, marker="s", color="#C44E52")
    ax.set_xlabel("epsilon")
    ax.set_ylabel("MSE vs. clean data")
    ax.set_title("LDP epsilon vs. distortion (MSE)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _plot_distortion_vs_noise(rows: List[Dict[str, Any]], out: Path) -> None:
    xs = [r["param_value"] for r in rows]
    mse = [r["mse"] for r in rows]
    acc = [r["defended_accuracy"] for r in rows]
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(xs, mse, marker="o", color="#C44E52", label="MSE")
    ax1.set_xlabel("noise_scale")
    ax1.set_ylabel("MSE", color="#C44E52")
    ax1.tick_params(axis="y", labelcolor="#C44E52")
    ax2 = ax1.twinx()
    ax2.plot(xs, acc, marker="^", color="#4C72B0", label="Defended Acc")
    ax2.set_ylabel("Accuracy", color="#4C72B0")
    ax2.tick_params(axis="y", labelcolor="#4C72B0")
    ax1.set_title("Noise scale: distortion vs. attack accuracy")
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _plot_noise_vs_accuracy(rows: List[Dict[str, Any]], out: Path) -> None:
    xs = [r["param_value"] for r in rows]
    ys = [r["defended_accuracy"] for r in rows]
    yb = rows[0]["baseline_accuracy"] if rows else 0.0
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(xs, ys, marker="o")
    ax.axhline(yb, color="gray", linestyle="--", label="Baseline")
    ax.set_xlabel("noise_scale")
    ax.set_ylabel("Accuracy")
    ax.set_title("Noise scale vs. defended accuracy")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
