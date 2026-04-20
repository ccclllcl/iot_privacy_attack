#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成防御（扰动）后的数据集。

用法: python run_defense.py --config configs/default.yaml
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
from src.defenses.defense_pipeline import run_defense_pipeline


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    parser = argparse.ArgumentParser(description="对预处理序列施加防御并写入 data/defended/")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = (ROOT / cfg_path).resolve()
    cfg = ExperimentConfig.from_yaml(cfg_path, project_root=ROOT)
    run_defense_pipeline(cfg)


if __name__ == "__main__":
    main()
