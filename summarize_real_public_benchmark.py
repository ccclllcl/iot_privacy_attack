#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aggregate real-public benchmark outputs into concise summary files."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "outputs" / "reports" / "real_public_benchmark" / "real_public_benchmark_manifest.json"
OUT_DIR = ROOT / "outputs" / "reports" / "real_public_benchmark"


def _f(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_acc(path: Path) -> float:
    if not path.exists():
        return float("nan")
    return _f(_load_json(path).get("accuracy"))


def main() -> None:
    if not MANIFEST.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = _load_json(MANIFEST)
    rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for item in manifest:
        dataset = str(item["dataset"])
        seed = int(item["seed"])
        for method_item in item["methods"]:
            method = str(method_item["method"])
            defense_dir = ROOT / str(method_item["defense_dir"])
            report = defense_dir / "defense_report.json"
            if not report.exists():
                continue
            obj = _load_json(report)
            attack = obj.get("attack_metrics", {})
            baseline = attack.get("baseline", {})
            fixed = attack.get("defended_fixed_attacker", {})
            retrain = attack.get("defended_retrain_attacker", {})
            effect = attack.get("defense_effect", {})
            dist = obj.get("distortion", {})
            reports_dir = defense_dir / "json_reports"

            baseline_acc = _f(baseline.get("accuracy"))
            fixed_acc = _f(fixed.get("accuracy"))
            retrain_acc = _f(retrain.get("accuracy"))
            if baseline_acc != baseline_acc:  # NaN check
                baseline_acc = _read_acc(reports_dir / "lstm_baseline_confusion_test.json")
            if fixed_acc != fixed_acc:
                fixed_acc = _read_acc(reports_dir / "lstm_defended_confusion_test_fixed_attacker.json")
            if retrain_acc != retrain_acc:
                retrain_acc = _read_acc(reports_dir / "lstm_defended_confusion_test_retrained_attacker.json")

            acc_drop = _f(effect.get("accuracy_drop"))
            if acc_drop != acc_drop and baseline_acc == baseline_acc and fixed_acc == fixed_acc:
                acc_drop = baseline_acc - fixed_acc
            acc_drop_pct = _f(effect.get("relative_accuracy_drop_percent"))
            if acc_drop_pct != acc_drop_pct and baseline_acc and baseline_acc == baseline_acc and acc_drop == acc_drop:
                acc_drop_pct = (acc_drop / baseline_acc) * 100.0

            row = {
                "dataset": dataset,
                "seed": seed,
                "method": method,
                "baseline_acc": baseline_acc,
                "fixed_acc": fixed_acc,
                "retrain_acc": retrain_acc,
                "acc_drop": acc_drop,
                "acc_drop_pct": acc_drop_pct,
                "mse": _f(dist.get("mse")),
                "mae": _f(dist.get("mae")),
                "pearson_r": _f(dist.get("pearson_r")),
            }
            rows.append(row)

            g = grouped[(dataset, method)]
            for k, v in row.items():
                if k in {"dataset", "seed", "method"}:
                    continue
                g[k].append(v)

    rows.sort(key=lambda x: (x["dataset"], x["method"], x["seed"]))
    csv_path = OUT_DIR / "real_public_benchmark_runs.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "seed",
                "method",
                "baseline_acc",
                "fixed_acc",
                "retrain_acc",
                "acc_drop",
                "acc_drop_pct",
                "mse",
                "mae",
                "pearson_r",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    agg_rows: list[dict[str, Any]] = []
    for (dataset, method), stats in sorted(grouped.items()):
        agg_rows.append(
            {
                "dataset": dataset,
                "method": method,
                "n_runs": len(stats["baseline_acc"]),
                "baseline_acc_mean": mean(stats["baseline_acc"]),
                "fixed_acc_mean": mean(stats["fixed_acc"]),
                "retrain_acc_mean": mean(stats["retrain_acc"]),
                "acc_drop_mean": mean(stats["acc_drop"]),
                "acc_drop_pct_mean": mean(stats["acc_drop_pct"]),
                "mse_mean": mean(stats["mse"]),
                "mae_mean": mean(stats["mae"]),
                "pearson_r_mean": mean(stats["pearson_r"]),
            }
        )

    json_path = OUT_DIR / "real_public_benchmark_summary.json"
    json_path.write_text(json.dumps(agg_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved run-level csv: {csv_path.as_posix()}")
    print(f"Saved aggregated json: {json_path.as_posix()}")


if __name__ == "__main__":
    main()
