#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collect confusion matrix + key metrics into a JSON file.

This is used for multi-seed paper experiments where PNG confusion matrices
are not convenient to summarize or aggregate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from src.evaluate import evaluate_on_arrays, load_model_from_checkpoint
from src.utils import get_torch_device


def top_confusions(cm: np.ndarray, class_names: List[str], k: int = 8) -> List[Dict[str, Any]]:
    """Return top-k off-diagonal confusions as list of dicts."""
    out: List[Tuple[int, int, int]] = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i == j:
                continue
            c = int(cm[i, j])
            if c > 0:
                out.append((c, i, j))
    out.sort(reverse=True)
    res: List[Dict[str, Any]] = []
    for c, i, j in out[:k]:
        res.append(
            {
                "true": class_names[i] if i < len(class_names) else str(i),
                "pred": class_names[j] if j < len(class_names) else str(j),
                "count": int(c),
            }
        )
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description="Export confusion matrix and metrics to JSON")
    ap.add_argument("--model_path", required=True, help="Path to .pt checkpoint")
    ap.add_argument(
        "--npz_path",
        required=True,
        help="NPZ containing X_* and y_* arrays (e.g. data/processed/sequences.npz)",
    )
    ap.add_argument("--split", required=True, choices=["train", "val", "test"])
    ap.add_argument(
        "--model_type",
        default="lstm",
        choices=["lstm", "mlp"],
        help="How to interpret X arrays",
    )
    ap.add_argument(
        "--out",
        required=True,
        help="Output JSON path",
    )
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=0)
    args = ap.parse_args()

    device = get_torch_device("cpu")
    model, ckpt = load_model_from_checkpoint(Path(args.model_path), device)
    class_names: List[str] = list(ckpt["class_names"])

    data = np.load(Path(args.npz_path))
    X = data[f"X_{args.split}"]
    y = data[f"y_{args.split}"]

    metrics = evaluate_on_arrays(
        model,
        X,
        y,
        args.model_type,
        class_names,
        device,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
    )

    cm = metrics["confusion_matrix"]
    payload: Dict[str, Any] = {
        "split": args.split,
        "model_type": args.model_type,
        "accuracy": float(metrics["accuracy"]),
        "f1_macro": float(metrics["f1_macro"]),
        "precision_macro": float(metrics["precision_macro"]),
        "recall_macro": float(metrics["recall_macro"]),
        "per_class_recall": metrics["per_class_recall"],
        "confusion_matrix": cm.tolist(),
        "top_confusions": top_confusions(cm, class_names, k=10),
        "class_names": class_names,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

