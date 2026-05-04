#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
防御效果评估。

用法示例:
  python experiments/core/run_defense_eval.py --config configs/default.yaml --mode fixed_attacker --model_path outputs/models/best_lstm.pt
  python experiments/core/run_defense_eval.py --config configs/default.yaml --mode retrain_attacker
  python experiments/core/run_defense_eval.py --config configs/default.yaml --mode fixed_attacker --model_path outputs/models/best_lstm.pt --skip-pipeline
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import ExperimentConfig
from src.defense_eval import run_defense_evaluation


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    parser = argparse.ArgumentParser(description="防御前后攻击模型性能对比评估")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["fixed_attacker", "retrain_attacker"],
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="",
        help="fixed_attacker 模式必填：原始训练得到的 .pt",
    )
    parser.add_argument(
        "--skip-pipeline",
        action="store_true",
        help="跳过重新扰动，直接使用 data/defended 下已有数据",
    )
    args = parser.parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = (ROOT / cfg_path).resolve()
    cfg = ExperimentConfig.from_yaml(cfg_path, project_root=ROOT)

    mp: Path | None = None
    if args.model_path:
        mp = Path(args.model_path)
        if not mp.is_absolute():
            mp = (ROOT / mp).resolve()

    if args.mode == "fixed_attacker" and mp is None:
        raise SystemExit("fixed_attacker 模式必须提供 --model_path")

    run_defense_evaluation(
        cfg,
        mode=args.mode,
        model_path=mp,
        skip_pipeline=bool(args.skip_pipeline),
    )


if __name__ == "__main__":
    main()
