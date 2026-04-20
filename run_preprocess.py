#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
命令行入口：数据预处理。
用法: python run_preprocess.py --config configs/default.yaml
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
from src.preprocess import run_preprocess


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    parser = argparse.ArgumentParser(description="Smart* 风格数据预处理")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="YAML 配置文件路径（相对项目根目录）",
    )
    args = parser.parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = (ROOT / cfg_path).resolve()
    cfg = ExperimentConfig.from_yaml(cfg_path, project_root=ROOT)
    run_preprocess(cfg)


if __name__ == "__main__":
    main()
