#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
命令行入口：训练攻击者模型。
用法: python run_train.py --config configs/default.yaml --model lstm
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import ExperimentConfig
from src.train import run_training


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    parser = argparse.ArgumentParser(description="训练 LSTM / MLP 攻击者模型")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument(
        "--model",
        type=str,
        default="lstm",
        choices=["lstm", "mlp"],
        help="lstm: 原始序列；mlp: 统计特征",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="可选：覆盖模型保存路径（默认 outputs/models/best_<model>.pt）",
    )
    args = parser.parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = (ROOT / cfg_path).resolve()
    cfg = ExperimentConfig.from_yaml(cfg_path, project_root=ROOT)
    out = Path(args.output) if args.output else None
    run_training(cfg, model_type=args.model, override_model_path=out)


if __name__ == "__main__":
    main()
