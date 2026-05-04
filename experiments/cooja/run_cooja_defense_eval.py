#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate Cooja defense logs under fixed/retrain attacker settings.

This script consumes a manifest JSON with one baseline log pair and multiple
defense-method log pairs, then reports:
- fixed_attacker: train on baseline, test on defense
- retrain_attacker: train/test on defense

It also reports baseline test metrics (train/test on baseline) for reference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

from run_cooja_baseline_attack import (
    build_window_dataset,
    parse_app_requests,
    parse_radio,
)


def parse_seed_list(s: str) -> list[int]:
    out: list[int] = []
    for part in s.split(","):
        part = part.strip()
        if part:
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


def dataset_from_logs(
    radio_log: Path,
    app_log: Path,
    window_s: float,
    step_s: float,
    min_requests: int,
    dominance_threshold: float,
) -> pd.DataFrame:
    radio_df = parse_radio(radio_log)
    app_df = parse_app_requests(app_log)
    return build_window_dataset(
        radio_df=radio_df,
        app_df=app_df,
        window_s=window_s,
        step_s=step_s,
        min_requests=min_requests,
        dominance_threshold=dominance_threshold,
    )


def dataset_from_frames(
    radio_df: pd.DataFrame,
    app_df: pd.DataFrame,
    window_s: float,
    step_s: float,
    min_requests: int,
    dominance_threshold: float,
) -> pd.DataFrame:
    return build_window_dataset(
        radio_df=radio_df,
        app_df=app_df,
        window_s=window_s,
        step_s=step_s,
        min_requests=min_requests,
        dominance_threshold=dominance_threshold,
    )


def _normalize01(x: np.ndarray) -> np.ndarray:
    lo = float(np.min(x))
    hi = float(np.max(x))
    if hi - lo < 1e-12:
        return np.zeros_like(x, dtype=np.float64)
    return (x - lo) / (hi - lo)


def apply_radio_transform(
    radio_df: pd.DataFrame,
    method_cfg: dict[str, Any],
    seed: int,
) -> pd.DataFrame:
    """
    Apply traffic-side perturbation on radio frames while preserving packet rows.
    Supported `method_cfg["transform"]`:
    - noise
    - ldp
    - adaptive_ldp
    """
    kind = str(method_cfg.get("transform", "")).strip().lower()
    if not kind:
        return radio_df.copy()

    rng = np.random.default_rng(int(seed))
    out = radio_df.copy()
    t = out["t_ms"].to_numpy(dtype=np.float64)
    frame_len = out["frame_len"].to_numpy(dtype=np.float64)
    src = out["src_id"].to_numpy(dtype=np.int64)

    if kind == "noise":
        time_noise_std_ms = float(method_cfg.get("time_noise_std_ms", 5.0))
        len_noise_std = float(method_cfg.get("len_noise_std", 2.5))
        t_new = t + rng.normal(0.0, time_noise_std_ms, size=t.shape)
        len_new = frame_len + rng.normal(0.0, len_noise_std, size=frame_len.shape)
    elif kind == "ldp":
        epsilon = float(method_cfg.get("epsilon", 1.0))
        sens_t = float(method_cfg.get("sensitivity_time_ms", 8.0))
        sens_l = float(method_cfg.get("sensitivity_len", 4.0))
        scale_t = sens_t / max(epsilon, 1e-6)
        scale_l = sens_l / max(epsilon, 1e-6)
        t_new = t + rng.laplace(0.0, scale_t, size=t.shape)
        len_new = frame_len + rng.laplace(0.0, scale_l, size=frame_len.shape)
    elif kind == "adaptive_ldp":
        eps_min = float(method_cfg.get("epsilon_min", 0.6))
        eps_max = float(method_cfg.get("epsilon_max", 2.0))
        sens_t = float(method_cfg.get("sensitivity_time_ms", 8.0))
        sens_l = float(method_cfg.get("sensitivity_len", 4.0))
        weight_iat = float(method_cfg.get("weight_iat", 0.7))
        weight_src = float(method_cfg.get("weight_src", 0.3))

        iat = np.diff(np.r_[t[0], t])
        iat_risk = _normalize01(np.abs(iat - np.median(iat)))
        src_counts = pd.Series(src).value_counts(normalize=True).to_dict()
        src_freq = np.array([float(src_counts.get(int(s), 0.0)) for s in src], dtype=np.float64)
        src_risk = 1.0 - _normalize01(src_freq)
        risk = np.clip(weight_iat * iat_risk + weight_src * src_risk, 0.0, 1.0)
        eps_i = eps_max - risk * (eps_max - eps_min)

        t_new = t + rng.laplace(0.0, sens_t / np.maximum(eps_i, 1e-6), size=t.shape)
        len_new = frame_len + rng.laplace(0.0, sens_l / np.maximum(eps_i, 1e-6), size=frame_len.shape)
    else:
        raise ValueError(f"Unknown transform kind: {kind}")

    order = np.argsort(t_new)
    out = out.iloc[order].reset_index(drop=True)
    t_sorted = np.maximum(np.round(t_new[order]), 0.0).astype(np.int64)
    out["t_ms"] = np.maximum.accumulate(t_sorted)
    out["frame_len"] = np.clip(np.round(len_new[order]), 5, 127).astype(np.int64)
    return out


def make_xy(ds: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    feature_cols = [
        c
        for c in ds.columns
        if c
        not in {
            "window_start_ms",
            "window_end_ms",
            "label",
            "label_count",
            "request_count",
            "label_dominance",
        }
    ]
    x = ds[feature_cols].to_numpy(dtype=np.float32)
    y = ds["label"].to_numpy()
    return x, y, feature_cols


def train_rf(x_train: np.ndarray, y_train: np.ndarray, seed: int) -> RandomForestClassifier:
    clf = RandomForestClassifier(
        n_estimators=300,
        random_state=int(seed),
        class_weight="balanced_subsample",
        n_jobs=-1,
    )
    clf.fit(x_train, y_train)
    return clf


def eval_metrics(
    clf: RandomForestClassifier,
    x_test: np.ndarray,
    y_test: np.ndarray,
    labels: list[str],
) -> dict[str, Any]:
    y_pred = clf.predict(x_test)
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
        "labels": labels,
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=labels).tolist(),
    }


def load_manifest(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "baseline" not in data or "methods" not in data:
        raise ValueError("Manifest must contain 'baseline' and 'methods'.")
    return data


def plot_method_bars(summary: dict[str, Any], out_path: Path) -> None:
    methods = list(summary.keys())
    fixed_acc = [summary[m]["fixed_attacker"]["accuracy"]["mean"] for m in methods]
    retr_acc = [summary[m]["retrain_attacker"]["accuracy"]["mean"] for m in methods]

    x = np.arange(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, fixed_acc, width=width, label="fixed_attacker")
    ax.bar(x + width / 2, retr_acc, width=width, label="retrain_attacker")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=20, ha="right")
    ax.set_ylabel("Accuracy (mean over seeds)")
    ax.set_title("Cooja Defense Methods vs Attacker Modes")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate Cooja defenses with fixed/retrain attacker")
    ap.add_argument("--manifest", default="configs/cooja_defense_logs.json")
    ap.add_argument("--out_dir", default="outputs/cooja_defense_eval")
    ap.add_argument("--window_s", type=float, default=8.0)
    ap.add_argument("--step_s", type=float, default=3.0)
    ap.add_argument("--min_requests", type=int, default=2)
    ap.add_argument("--dominance_threshold", type=float, default=0.2)
    ap.add_argument("--test_ratio", type=float, default=0.3)
    ap.add_argument("--seeds", default="42,123,2026")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    out_dir = (root / args.out_dir).resolve() if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = (root / manifest_path).resolve()
    manifest = load_manifest(manifest_path)
    seeds = parse_seed_list(str(args.seeds))

    baseline_radio = Path(manifest["baseline"]["radio_log"]).resolve()
    baseline_app = Path(manifest["baseline"]["app_log"]).resolve()
    baseline_radio_df = parse_radio(baseline_radio)
    baseline_app_df = parse_app_requests(baseline_app)
    baseline_ds = dataset_from_frames(
        baseline_radio_df,
        baseline_app_df,
        float(args.window_s),
        float(args.step_s),
        int(args.min_requests),
        float(args.dominance_threshold),
    )
    x_base, y_base, _ = make_xy(baseline_ds)

    methods: list[dict[str, Any]] = list(manifest["methods"])
    methods_summary: dict[str, Any] = {}

    for m in methods:
        method_name = str(m["name"])
        transform_kind = str(m.get("transform", "")).strip().lower()

        defense_radio: Path | None = None
        defense_app: Path | None = None
        defense_radio_df_static: pd.DataFrame | None = None
        defense_app_df_static: pd.DataFrame | None = None
        defense_ds_static: pd.DataFrame | None = None
        x_def_static: np.ndarray | None = None
        y_def_static: np.ndarray | None = None

        if transform_kind:
            defense_radio = baseline_radio
            defense_app = baseline_app
            defense_app_df_static = baseline_app_df
        else:
            defense_radio = Path(str(m["radio_log"])).resolve()
            defense_app = Path(str(m["app_log"])).resolve()
            if not defense_radio.exists() or not defense_app.exists():
                print(f"[SKIP] {method_name}: missing defense logs")
                continue
            defense_radio_df_static = parse_radio(defense_radio)
            defense_app_df_static = parse_app_requests(defense_app)
            defense_ds_static = dataset_from_frames(
                defense_radio_df_static,
                defense_app_df_static,
                float(args.window_s),
                float(args.step_s),
                int(args.min_requests),
                float(args.dominance_threshold),
            )
            x_def_static, y_def_static, _ = make_xy(defense_ds_static)

        per_seed_runs: list[dict[str, Any]] = []
        b_acc: list[float] = []
        b_f1: list[float] = []
        f_acc: list[float] = []
        f_f1: list[float] = []
        r_acc: list[float] = []
        r_f1: list[float] = []
        defense_windows_for_summary: int | None = None
        defense_labels_for_summary: list[str] | None = None

        for seed in seeds:
            xb_train, xb_test, yb_train, yb_test = train_test_split(
                x_base,
                y_base,
                test_size=float(args.test_ratio),
                random_state=int(seed),
                stratify=y_base,
            )

            if transform_kind:
                assert defense_app_df_static is not None
                transformed_radio_df = apply_radio_transform(
                    baseline_radio_df,
                    method_cfg=m,
                    seed=seed,
                )
                defense_ds = dataset_from_frames(
                    transformed_radio_df,
                    defense_app_df_static,
                    float(args.window_s),
                    float(args.step_s),
                    int(args.min_requests),
                    float(args.dominance_threshold),
                )
                x_def, y_def, _ = make_xy(defense_ds)
            else:
                assert defense_ds_static is not None
                assert x_def_static is not None
                assert y_def_static is not None
                defense_ds = defense_ds_static
                x_def = x_def_static
                y_def = y_def_static

            if defense_windows_for_summary is None:
                defense_windows_for_summary = int(len(defense_ds))
                defense_labels_for_summary = sorted(np.unique(y_def).tolist())

            xd_train, xd_test, yd_train, yd_test = train_test_split(
                x_def,
                y_def,
                test_size=float(args.test_ratio),
                random_state=int(seed),
                stratify=y_def,
            )

            # Baseline attacker trained on baseline.
            base_clf = train_rf(xb_train, yb_train, seed=seed)
            base_labels = sorted(np.unique(yb_test).tolist())
            base_metrics = eval_metrics(base_clf, xb_test, yb_test, labels=base_labels)

            # Fixed attacker: baseline-trained model evaluated on defense.
            fixed_labels = sorted(np.unique(yd_test).tolist())
            fixed_metrics = eval_metrics(base_clf, xd_test, yd_test, labels=fixed_labels)

            # Retrain attacker: defense-trained model evaluated on defense.
            retrain_clf = train_rf(xd_train, yd_train, seed=seed)
            retrain_metrics = eval_metrics(retrain_clf, xd_test, yd_test, labels=fixed_labels)

            b_acc.append(base_metrics["accuracy"])
            b_f1.append(base_metrics["f1_macro"])
            f_acc.append(fixed_metrics["accuracy"])
            f_f1.append(fixed_metrics["f1_macro"])
            r_acc.append(retrain_metrics["accuracy"])
            r_f1.append(retrain_metrics["f1_macro"])

            per_seed_runs.append(
                {
                    "seed": int(seed),
                    "baseline_test": base_metrics,
                    "fixed_attacker_on_defense": fixed_metrics,
                    "retrain_attacker_on_defense": retrain_metrics,
                }
            )

            print(
                f"[{method_name}][seed={seed}] "
                f"baseline={base_metrics['accuracy']:.4f}, "
                f"fixed={fixed_metrics['accuracy']:.4f}, "
                f"retrain={retrain_metrics['accuracy']:.4f}"
            )

        methods_summary[method_name] = {
            "defense_log_paths": {
                "radio_log": str(defense_radio),
                "app_log": str(defense_app),
            },
            "generation": {
                "mode": "generated_transform" if transform_kind else "from_defense_logs",
                "transform": transform_kind if transform_kind else None,
                "transform_config": m if transform_kind else None,
            },
            "dataset": {
                "baseline_windows": int(len(baseline_ds)),
                "defense_windows": int(defense_windows_for_summary or 0),
                "baseline_labels": sorted(np.unique(y_base).tolist()),
                "defense_labels": defense_labels_for_summary or [],
            },
            "baseline_test": {
                "accuracy": summarize(b_acc),
                "f1_macro": summarize(b_f1),
            },
            "fixed_attacker": {
                "accuracy": summarize(f_acc),
                "f1_macro": summarize(f_f1),
                "delta_vs_baseline_accuracy_mean": float(mean(f_acc) - mean(b_acc)),
                "delta_vs_baseline_f1_mean": float(mean(f_f1) - mean(b_f1)),
            },
            "retrain_attacker": {
                "accuracy": summarize(r_acc),
                "f1_macro": summarize(r_f1),
                "delta_vs_baseline_accuracy_mean": float(mean(r_acc) - mean(b_acc)),
                "delta_vs_baseline_f1_mean": float(mean(r_f1) - mean(b_f1)),
            },
            "runs": per_seed_runs,
        }

    if not methods_summary:
        raise ValueError("No valid defense methods were evaluated. Check manifest paths.")

    final_report = {
        "config": {
            "manifest": str(manifest_path),
            "window_s": float(args.window_s),
            "step_s": float(args.step_s),
            "min_requests": int(args.min_requests),
            "dominance_threshold": float(args.dominance_threshold),
            "test_ratio": float(args.test_ratio),
            "seeds": seeds,
        },
        "methods": methods_summary,
    }
    (out_dir / "defense_eval_report.json").write_text(
        json.dumps(final_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    plot_method_bars(methods_summary, out_dir / "method_accuracy_bars.png")
    print(f"[OK] report: {out_dir / 'defense_eval_report.json'}")
    print(f"[OK] plot:   {out_dir / 'method_accuracy_bars.png'}")


if __name__ == "__main__":
    main()
