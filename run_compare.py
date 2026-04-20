#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量扫描 epsilon / 噪声强度并汇总结果（固定攻击者模式）。

用法:
  python run_compare.py --config configs/default.yaml --method ldp --model_path outputs/models/best_lstm.pt
  python run_compare.py --config configs/default.yaml --method noise --model_path outputs/models/best_lstm.pt
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
from src.experiment_compare import run_parameter_compare


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logger = logging.getLogger(__name__)
    parser = argparse.ArgumentParser(description="防御参数批量对比实验")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--method", type=str, required=True, choices=["ldp", "noise"])
    parser.add_argument("--model_path", type=str, required=True)
    args = parser.parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = (ROOT / cfg_path).resolve()
    mp = Path(args.model_path)
    if not mp.is_absolute():
        mp = (ROOT / mp).resolve()
    cfg = ExperimentConfig.from_yaml(cfg_path, project_root=ROOT)
    out = run_parameter_compare(cfg, method=args.method, model_path=mp)
    logger.info("对比结果 CSV: %s", out)


if __name__ == "__main__":
    main()
