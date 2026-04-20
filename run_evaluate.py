#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
命令行入口：测试集评估。
用法: python run_evaluate.py --config configs/default.yaml --model_path outputs/models/best_lstm.pt
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
from src.evaluate import run_evaluate


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    parser = argparse.ArgumentParser(description="评估已训练模型")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="run_train 保存的 .pt 文件路径",
    )
    args = parser.parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = (ROOT / cfg_path).resolve()
    mp = Path(args.model_path)
    if not mp.is_absolute():
        mp = (ROOT / mp).resolve()
    cfg = ExperimentConfig.from_yaml(cfg_path, project_root=ROOT)
    run_evaluate(cfg, mp)


if __name__ == "__main__":
    main()
