#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Collect representative charts into web_assets/images for frontend display."""

from __future__ import annotations

import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "web_assets" / "images"


def _copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def main() -> None:
    copied = 0
    checked = 0

    # Mock track (default + multiseed sample)
    mock_pairs = [
        (ROOT / "outputs" / "figures" / "train_curve.png", ASSET_DIR / "mock" / "default_train_curve.png"),
        (ROOT / "outputs" / "figures" / "confusion_matrix.png", ASSET_DIR / "mock" / "default_confusion_matrix.png"),
        (ROOT / "outputs" / "defense" / "accuracy_comparison.png", ASSET_DIR / "mock" / "default_accuracy_comparison.png"),
        (
            ROOT / "outputs" / "defense" / "comparisons" / "noise_scale_vs_accuracy.png",
            ASSET_DIR / "mock" / "default_noise_scale_vs_accuracy.png",
        ),
        (
            ROOT / "outputs" / "defense" / "comparisons" / "epsilon_vs_accuracy.png",
            ASSET_DIR / "mock" / "default_epsilon_vs_accuracy.png",
        ),
        (
            ROOT / "outputs" / "figures" / "full_multiseed" / "seed_42" / "confusion_matrix.png",
            ASSET_DIR / "mock" / "full_multiseed_seed42_confusion_matrix.png",
        ),
    ]

    # Real dataset track (sample seeds for quick visual comparison)
    real_pairs = [
        (
            ROOT / "outputs" / "figures" / "dataset_matrix" / "uci_har" / "seed_42" / "confusion_matrix.png",
            ASSET_DIR / "real" / "dataset_matrix_uci_har_seed42_confusion_matrix.png",
        ),
        (
            ROOT / "outputs" / "figures" / "dataset_matrix" / "kasteren" / "seed_42" / "confusion_matrix.png",
            ASSET_DIR / "real" / "dataset_matrix_kasteren_seed42_confusion_matrix.png",
        ),
        (
            ROOT / "outputs" / "figures" / "real_public_benchmark" / "uci_har" / "seed_42" / "confusion_matrix.png",
            ASSET_DIR / "real" / "benchmark_uci_har_seed42_confusion_matrix.png",
        ),
        (
            ROOT / "outputs" / "figures" / "real_public_benchmark" / "kasteren" / "seed_42" / "confusion_matrix.png",
            ASSET_DIR / "real" / "benchmark_kasteren_seed42_confusion_matrix.png",
        ),
        (
            ROOT / "outputs" / "figures" / "real_public_benchmark" / "casas_hh101" / "seed_42" / "confusion_matrix.png",
            ASSET_DIR / "real" / "benchmark_casas_hh101_seed42_confusion_matrix.png",
        ),
    ]

    for src, dst in [*mock_pairs, *real_pairs]:
        checked += 1
        copied += int(_copy_if_exists(src, dst))

    # Keep a tiny marker file so the folder is explicit in repo.
    marker = ASSET_DIR / "README.txt"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(
        "This folder stores image resources shown by apps/ui_simple.py.\n"
        "Run: python tools/refresh_web_assets.py\n",
        encoding="utf-8",
    )
    print(f"checked={checked}, copied={copied}, asset_dir={ASSET_DIR.as_posix()}")


if __name__ == "__main__":
    main()
