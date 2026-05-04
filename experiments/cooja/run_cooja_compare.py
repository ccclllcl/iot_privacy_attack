#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare baseline vs defended Cooja traffic attacker performance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from run_cooja_baseline_attack import run_attack_pipeline


def parse_seed_list(s: str) -> list[int]:
    out: list[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    if not out:
        raise ValueError("Seed list is empty.")
    return out


def summarize(vals: list[float]) -> dict[str, float]:
    if not vals:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(mean(vals)),
        "std": float(pstdev(vals)) if len(vals) > 1 else 0.0,
        "min": float(min(vals)),
        "max": float(max(vals)),
    }


def plot_comparison(
    baseline_acc: list[float],
    defense_acc: list[float],
    baseline_f1: list[float],
    defense_f1: list[float],
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    x = np.arange(max(len(baseline_acc), len(defense_acc)))

    axes[0].plot(x[: len(baseline_acc)], baseline_acc, marker="o", label="baseline")
    axes[0].plot(x[: len(defense_acc)], defense_acc, marker="o", label="defense")
    axes[0].set_title("Accuracy by seed")
    axes[0].set_xlabel("Seed index")
    axes[0].set_ylabel("Accuracy")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(x[: len(baseline_f1)], baseline_f1, marker="o", label="baseline")
    axes[1].plot(x[: len(defense_f1)], defense_f1, marker="o", label="defense")
    axes[1].set_title("Macro-F1 by seed")
    axes[1].set_xlabel("Seed index")
    axes[1].set_ylabel("Macro-F1")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare baseline and defended Cooja logs")
    ap.add_argument("--baseline_radio_log", required=True)
    ap.add_argument("--baseline_app_log", required=True)
    ap.add_argument("--defense_radio_log", required=True)
    ap.add_argument("--defense_app_log", required=True)
    ap.add_argument("--out_dir", default="outputs/cooja_compare")
    ap.add_argument("--window_s", type=float, default=8.0)
    ap.add_argument("--step_s", type=float, default=3.0)
    ap.add_argument("--min_requests", type=int, default=2)
    ap.add_argument("--dominance_threshold", type=float, default=0.2)
    ap.add_argument("--test_ratio", type=float, default=0.3)
    ap.add_argument("--seeds", default="42,123,2026", help="Comma-separated seeds")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = parse_seed_list(str(args.seeds))

    baseline_runs: list[dict[str, Any]] = []
    defense_runs: list[dict[str, Any]] = []
    for seed in seeds:
        baseline_out = out_dir / "baseline" / f"seed_{seed}"
        defense_out = out_dir / "defense" / f"seed_{seed}"
        b = run_attack_pipeline(
            radio_log=Path(args.baseline_radio_log),
            app_log=Path(args.baseline_app_log),
            out_dir=baseline_out,
            window_s=float(args.window_s),
            step_s=float(args.step_s),
            min_requests=int(args.min_requests),
            dominance_threshold=float(args.dominance_threshold),
            test_ratio=float(args.test_ratio),
            random_seed=int(seed),
            write_outputs=True,
        )
        d = run_attack_pipeline(
            radio_log=Path(args.defense_radio_log),
            app_log=Path(args.defense_app_log),
            out_dir=defense_out,
            window_s=float(args.window_s),
            step_s=float(args.step_s),
            min_requests=int(args.min_requests),
            dominance_threshold=float(args.dominance_threshold),
            test_ratio=float(args.test_ratio),
            random_seed=int(seed),
            write_outputs=True,
        )
        b["seed"] = int(seed)
        d["seed"] = int(seed)
        baseline_runs.append(b)
        defense_runs.append(d)
        print(
            f"[seed={seed}] baseline acc={b['metrics']['accuracy']:.4f}, defense acc={d['metrics']['accuracy']:.4f}"
        )

    baseline_acc = [float(x["metrics"]["accuracy"]) for x in baseline_runs]
    defense_acc = [float(x["metrics"]["accuracy"]) for x in defense_runs]
    baseline_f1 = [float(x["metrics"]["f1_macro"]) for x in baseline_runs]
    defense_f1 = [float(x["metrics"]["f1_macro"]) for x in defense_runs]

    summary = {
        "config": {
            "window_s": float(args.window_s),
            "step_s": float(args.step_s),
            "min_requests": int(args.min_requests),
            "dominance_threshold": float(args.dominance_threshold),
            "test_ratio": float(args.test_ratio),
            "seeds": seeds,
        },
        "baseline": {
            "accuracy": summarize(baseline_acc),
            "f1_macro": summarize(baseline_f1),
        },
        "defense": {
            "accuracy": summarize(defense_acc),
            "f1_macro": summarize(defense_f1),
        },
        "delta_defense_minus_baseline": {
            "accuracy_mean": float(mean(defense_acc) - mean(baseline_acc)),
            "f1_macro_mean": float(mean(defense_f1) - mean(baseline_f1)),
        },
        "runs": {
            "baseline": baseline_runs,
            "defense": defense_runs,
        },
    }

    (out_dir / "compare_report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    plot_comparison(
        baseline_acc=baseline_acc,
        defense_acc=defense_acc,
        baseline_f1=baseline_f1,
        defense_f1=defense_f1,
        out_path=out_dir / "accuracy_f1_by_seed.png",
    )
    print(f"[OK] Compare report: {out_dir / 'compare_report.json'}")


if __name__ == "__main__":
    main()
