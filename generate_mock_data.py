#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成合成智能家居事件 CSV，用于在无真实 Smart* 数据时跑通全流程。

输出列与 configs/default.yaml 中 columns 映射一致：
timestamp, device_id, value, behavior_label

用法（在项目根目录）:
  python generate_mock_data.py
  python generate_mock_data.py --config configs/default.yaml
  python generate_mock_data.py --days 21 --output data/raw/smart_home_events.csv --seed 123
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml

def _pick_behavior(hour: int, rng: np.random.Generator) -> str:
    """按小时生成「真实」行为标签（带随机性，便于分类器学习）。"""
    r = rng.random()
    if 0 <= hour < 7 or hour >= 23:
        return "sleep" if r < 0.85 else "other"
    if 10 <= hour < 16 and r < 0.2:
        return "away"
    if r < 0.08:
        return "away"
    if 11 <= hour <= 13 or 17 <= hour <= 19:
        if r < 0.55:
            return "cooking"
    if 9 <= hour <= 22 and r < 0.45:
        return "using_computer"
    return "other"


def _emit_for_behavior(
    ts: datetime, behavior: str, rng: np.random.Generator
) -> List[Tuple[datetime, str, float, str]]:
    rows: List[Tuple[datetime, str, float, str]] = []
    noise = lambda: float(rng.normal(0, 0.15))

    def add(dev: str, base: float) -> None:
        rows.append((ts, dev, max(0.0, base + noise()), behavior))

    if behavior == "sleep":
        if rng.random() < 0.3:
            add("bedroom_motion", 0.4)
    elif behavior == "away":
        pass
    elif behavior == "cooking":
        add("kitchen_light", 1.2)
        add("kitchen_microwave", rng.choice([0.0, 2.5, 3.0]))
        add("kitchen_fridge", 0.8)
        add("living_motion", 0.5)
    elif behavior == "using_computer":
        add("office_pc", 3.0 + rng.random() * 2.0)
        add("monitor_power", 1.5)
        add("desk_lamp", 1.0)
    else:
        add("living_motion", 0.6)
        if rng.random() < 0.3:
            add("bedroom_motion", 0.3)

    if not rows:
        rows.append((ts, "living_motion", 0.05 + abs(noise()), behavior))
    return rows


def read_random_seed_from_config(config_path: Path) -> int:
    """读取 experiment.random_seed；兼容旧版 preprocess.random_seed。"""
    if not config_path.is_file():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        data: Dict[str, Any] = yaml.safe_load(f)
    if not isinstance(data, dict):
        return 42
    ex = data.get("experiment") or {}
    if isinstance(ex, dict) and "random_seed" in ex:
        return int(ex["random_seed"])
    pp = data.get("preprocess") or {}
    if isinstance(pp, dict) and "random_seed" in pp:
        return int(pp["random_seed"])
    return 42


def generate(
    days: int = 14,
    freq_minutes: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = datetime(2024, 1, 1, 0, 0, 0)
    delta = timedelta(minutes=freq_minutes)
    all_rows: List[Tuple[datetime, str, float, str]] = []
    t = start
    end = start + timedelta(days=days)
    while t < end:
        h = t.hour
        beh = _pick_behavior(h, rng)
        all_rows.extend(_emit_for_behavior(t, beh, rng))
        t += delta
    df = pd.DataFrame(all_rows, columns=["timestamp", "device_id", "value", "behavior_label"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="生成合成 Smart* 风格事件 CSV")
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument("--freq_minutes", type=int, default=5)
    parser.add_argument(
        "--output",
        type=str,
        default="data/raw/smart_home_events.csv",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="用于读取 experiment.random_seed（与主实验统一）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="若指定则覆盖配置文件中的随机种子",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = root / cfg_path
    seed = int(args.seed) if args.seed is not None else read_random_seed_from_config(cfg_path)
    out = Path(args.output)
    if not out.is_absolute():
        out = root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    df = generate(days=args.days, freq_minutes=args.freq_minutes, seed=seed)
    df.to_csv(out, index=False, encoding="utf-8")
    print(f"已写入 {out.resolve()} ，共 {len(df)} 行。")


if __name__ == "__main__":
    main()
