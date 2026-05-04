#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a baseline traffic-side attacker from Cooja logs.

Inputs:
- Radiomsg.txt: radio-level frames with timestamp/source/length
- loglistener.txt: app logs with "Sending request" lines used as labels

Output:
- Window-level feature CSV
- JSON report (dataset stats + model metrics)
- Confusion matrix PNG
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split


APP_SEND_RE = re.compile(
    r"^\s*(?P<h>\d+):(?P<m>\d+):(?P<s>\d+(?:\.\d+)?)\s+ID:(?P<id>\d+)\s+\[INFO:\s*App\s*\]\s+Sending request"
)
RADIO_RE = re.compile(
    r"^\s*(?P<t>\d+)\t(?P<src>\d+)\t(?P<dst>[^\t]+)\t(?P<len>\d+):"
)


def hms_to_ms(h: int, m: int, s: float) -> int:
    return int(round((h * 3600 + m * 60 + s) * 1000.0))


def parse_app_requests(path: Path) -> pd.DataFrame:
    rows: list[dict[str, int]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = APP_SEND_RE.match(line)
        if not m:
            continue
        t_ms = hms_to_ms(int(m.group("h")), int(m.group("m")), float(m.group("s")))
        mote_id = int(m.group("id"))
        rows.append({"t_ms": t_ms, "client_id": mote_id})
    if not rows:
        raise ValueError("No 'Sending request' lines found in app log.")
    return pd.DataFrame(rows).sort_values("t_ms").reset_index(drop=True)


def parse_radio(path: Path) -> pd.DataFrame:
    rows: list[dict[str, int]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = RADIO_RE.match(line)
        if not m:
            continue
        rows.append(
            {
                "t_ms": int(m.group("t")),
                "src_id": int(m.group("src")),
                "frame_len": int(m.group("len")),
            }
        )
    if not rows:
        raise ValueError("No radio frame lines matched in Radiomsg log.")
    return pd.DataFrame(rows).sort_values("t_ms").reset_index(drop=True)


def entropy_from_counts(counter: Counter[int]) -> float:
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    ent = 0.0
    for c in counter.values():
        p = c / total
        ent -= p * math.log(p + 1e-12, 2)
    return float(ent)


def window_features(df: pd.DataFrame) -> dict[str, float]:
    lens = df["frame_len"].to_numpy(dtype=np.float64)
    t = df["t_ms"].to_numpy(dtype=np.float64)
    dts = np.diff(t) if len(t) > 1 else np.array([], dtype=np.float64)

    counts = Counter(df["src_id"].tolist())
    pkt_count = float(len(df))
    return {
        "pkt_count": pkt_count,
        "uniq_src_count": float(len(counts)),
        "src_entropy": entropy_from_counts(counts),
        "mean_len": float(np.mean(lens)) if lens.size else 0.0,
        "std_len": float(np.std(lens)) if lens.size else 0.0,
        "min_len": float(np.min(lens)) if lens.size else 0.0,
        "max_len": float(np.max(lens)) if lens.size else 0.0,
        "small_pkt_ratio": float(np.mean(lens <= 10.0)) if lens.size else 0.0,
        "large_pkt_ratio": float(np.mean(lens >= 80.0)) if lens.size else 0.0,
        "mean_iat_ms": float(np.mean(dts)) if dts.size else 0.0,
        "std_iat_ms": float(np.std(dts)) if dts.size else 0.0,
        "p95_iat_ms": float(np.percentile(dts, 95)) if dts.size else 0.0,
    }


def build_window_dataset(
    radio_df: pd.DataFrame,
    app_df: pd.DataFrame,
    window_s: float,
    step_s: float,
    min_requests: int,
    dominance_threshold: float,
) -> pd.DataFrame:
    w_ms = int(round(window_s * 1000.0))
    step_ms = int(round(step_s * 1000.0))

    start_ms = max(int(radio_df["t_ms"].min()), int(app_df["t_ms"].min()))
    end_ms = min(int(radio_df["t_ms"].max()), int(app_df["t_ms"].max()))
    if end_ms <= start_ms + w_ms:
        raise ValueError("Insufficient overlap between radio/app timelines.")

    rows: list[dict[str, Any]] = []
    for ws in range(start_ms, end_ms - w_ms + 1, step_ms):
        we = ws + w_ms
        r = radio_df[(radio_df["t_ms"] >= ws) & (radio_df["t_ms"] < we)]
        if r.empty:
            continue
        a = app_df[(app_df["t_ms"] >= ws) & (app_df["t_ms"] < we)]
        if len(a) < min_requests:
            continue
        label_counts = Counter(a["client_id"].tolist())
        top_label, top_count = label_counts.most_common(1)[0]
        dominance = float(top_count / max(len(a), 1))
        if dominance < dominance_threshold:
            continue
        feat = window_features(r)
        feat["window_start_ms"] = int(ws)
        feat["window_end_ms"] = int(we)
        feat["label"] = f"client_{top_label}"
        feat["label_count"] = int(top_count)
        feat["request_count"] = int(len(a))
        feat["label_dominance"] = dominance
        rows.append(feat)

    if not rows:
        raise ValueError(
            "No valid windows after filtering. Try lower min_requests or dominance_threshold."
        )
    return pd.DataFrame(rows)


def plot_confusion(cm: np.ndarray, labels: list[str], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Cooja Baseline Attacker Confusion Matrix")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_attack_pipeline(
    *,
    radio_log: Path,
    app_log: Path,
    out_dir: Path,
    window_s: float,
    step_s: float,
    min_requests: int,
    dominance_threshold: float,
    test_ratio: float,
    random_seed: int,
    write_outputs: bool = True,
) -> dict[str, Any]:
    radio_log = radio_log.resolve()
    app_log = app_log.resolve()
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    radio_df = parse_radio(radio_log)
    app_df = parse_app_requests(app_log)
    ds = build_window_dataset(
        radio_df=radio_df,
        app_df=app_df,
        window_s=float(window_s),
        step_s=float(step_s),
        min_requests=int(min_requests),
        dominance_threshold=float(dominance_threshold),
    )

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
    X = ds[feature_cols].to_numpy(dtype=np.float32)
    y = ds["label"].to_numpy()

    x_train, x_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=float(test_ratio),
        random_state=int(random_seed),
        stratify=y,
    )

    clf = RandomForestClassifier(
        n_estimators=300,
        random_state=int(random_seed),
        class_weight="balanced_subsample",
        n_jobs=-1,
    )
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    labels = sorted(np.unique(y))
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    acc = float(accuracy_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred, average="macro"))
    report = classification_report(y_test, y_pred, labels=labels, output_dict=True, zero_division=0)

    feature_imp = sorted(
        (
            {"feature": f, "importance": float(v)}
            for f, v in zip(feature_cols, clf.feature_importances_)
        ),
        key=lambda x: x["importance"],
        reverse=True,
    )

    result = {
        "inputs": {
            "radio_log": str(radio_log),
            "app_log": str(app_log),
        },
        "config": {
            "window_s": float(window_s),
            "step_s": float(step_s),
            "min_requests": int(min_requests),
            "dominance_threshold": float(dominance_threshold),
            "test_ratio": float(test_ratio),
            "random_seed": int(random_seed),
        },
        "dataset": {
            "total_windows": int(len(ds)),
            "train_windows": int(len(x_train)),
            "test_windows": int(len(x_test)),
            "label_distribution": {k: int(v) for k, v in Counter(y).items()},
        },
        "metrics": {
            "accuracy": acc,
            "f1_macro": f1,
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
            "labels": labels,
        },
        "top_feature_importance": feature_imp[:10],
    }
    if write_outputs:
        ds.to_csv(out_dir / "window_dataset.csv", index=False, encoding="utf-8")
        plot_confusion(cm, labels, out_dir / "confusion_matrix.png")
        (out_dir / "report.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Train baseline attacker from Cooja logs")
    ap.add_argument("--radio_log", required=True, help="Path to Radiomsg.txt")
    ap.add_argument("--app_log", required=True, help="Path to loglistener.txt")
    ap.add_argument("--out_dir", default="outputs/cooja_baseline", help="Output directory")
    ap.add_argument("--window_s", type=float, default=30.0, help="Window size in seconds")
    ap.add_argument("--step_s", type=float, default=10.0, help="Window step in seconds")
    ap.add_argument("--min_requests", type=int, default=5, help="Min app requests per window")
    ap.add_argument(
        "--dominance_threshold",
        type=float,
        default=0.55,
        help="Minimum dominant-label ratio in a window",
    )
    ap.add_argument("--test_ratio", type=float, default=0.3, help="Test split ratio")
    ap.add_argument("--random_seed", type=int, default=42, help="Random seed")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    result = run_attack_pipeline(
        radio_log=Path(args.radio_log),
        app_log=Path(args.app_log),
        out_dir=out_dir,
        window_s=float(args.window_s),
        step_s=float(args.step_s),
        min_requests=int(args.min_requests),
        dominance_threshold=float(args.dominance_threshold),
        test_ratio=float(args.test_ratio),
        random_seed=int(args.random_seed),
        write_outputs=True,
    )

    print(f"[OK] Dataset windows: {result['dataset']['total_windows']}")
    print(f"[OK] Accuracy: {result['metrics']['accuracy']:.4f}, Macro-F1: {result['metrics']['f1_macro']:.4f}")
    print(f"[OK] Outputs in: {out_dir}")


if __name__ == "__main__":
    main()
