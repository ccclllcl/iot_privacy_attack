#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Delete old confusion matrix images and rebuild them from metrics.json."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import SequenceDataset, TabularDataset
from src.evaluate import _plot_confusion, collect_predictions, load_model_from_checkpoint
from src.utils import get_torch_device


def main() -> None:
    root = ROOT
    reports_root = root / "outputs" / "reports"
    figures_root = root / "outputs" / "figures"

    removed = 0
    for p in figures_root.rglob("confusion_matrix*.png"):
        p.unlink(missing_ok=True)
        removed += 1

    device = get_torch_device("auto")
    rebuilt = 0
    skipped: list[tuple[str, str]] = []

    for metrics_path in sorted(reports_root.rglob("metrics.json")):
        rel = metrics_path.parent.relative_to(reports_root)
        figures_dir = figures_root / rel
        figures_dir.mkdir(parents=True, exist_ok=True)

        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        model_path = Path(str(metrics.get("model_path", "")))
        if not model_path.is_absolute():
            model_path = (root / model_path).resolve()
        if not model_path.exists():
            skipped.append((str(metrics_path), f"missing model: {model_path}"))
            continue

        processed_dir = root / "data" / "processed" / rel

        model, ckpt = load_model_from_checkpoint(model_path, device)
        class_names = list(ckpt["class_names"])
        model_type = str(ckpt["model_type"]).lower().strip()

        npz_path = processed_dir / ("sequences.npz" if model_type == "lstm" else "mlp_features.npz")
        if not npz_path.exists():
            alt = processed_dir / ("mlp_features.npz" if model_type == "lstm" else "sequences.npz")
            if alt.exists():
                npz_path = alt
                model_type = "mlp" if model_type == "lstm" else "lstm"
            else:
                skipped.append((str(metrics_path), f"missing npz: {npz_path}"))
                continue

        arr = np.load(npz_path)
        x_test, y_test = arr["X_test"], arr["y_test"]
        dataset = SequenceDataset(x_test, y_test) if model_type == "lstm" else TabularDataset(x_test, y_test)
        loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0)
        y_true, y_pred = collect_predictions(model, loader, device)
        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
        _plot_confusion(cm, class_names, figures_dir / "confusion_matrix.png")
        rebuilt += 1

    print(f"removed={removed}, rebuilt={rebuilt}, skipped={len(skipped)}")
    for i, (a, b) in enumerate(skipped[:20], start=1):
        print(f"skip[{i}] {a} -> {b}")


if __name__ == "__main__":
    main()
