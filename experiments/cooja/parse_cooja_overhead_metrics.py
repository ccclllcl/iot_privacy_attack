#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""解析 Cooja 结构化开销日志，生成 dummy/real、时延和 Energest 指标。"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from statistics import mean, median
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENERGY_MODEL = ROOT / "configs" / "cooja_energy_model.json"

TX_RE = re.compile(
    r"METRIC_TX\s+type=(?P<type>\w+)\s+node=(?P<node>\d+)\s+seq=(?P<seq>\d+)\s+bytes=(?P<bytes>\d+)\s+time_ms=(?P<time_ms>\d+)"
)
RX_RE = re.compile(
    r"METRIC_RX\s+type=(?P<type>\w+)\s+src=(?P<src>\d+)\s+seq=(?P<seq>\d+)\s+send_ms=(?P<send_ms>\d+)\s+recv_ms=(?P<recv_ms>\d+)\s+bytes=(?P<bytes>\d+)"
)
ENERGEST_RE = re.compile(
    r"ENERGEST\s+node=(?P<node>\d+)\s+cpu_ticks=(?P<cpu>\d+)\s+lpm_ticks=(?P<lpm>\d+)\s+tx_ticks=(?P<tx>\d+)\s+rx_ticks=(?P<rx>\d+)\s+total_ticks=(?P<total>\d+)\s+time_ms=(?P<time_ms>\d+)"
)
SIM_TIME_RE = re.compile(r"^(?P<sim_us>\d+)\s+ID:(?P<log_node>\d+)\s+")


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except Exception:
        return path.as_posix().replace("\\", "/")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return _rel(value)
    return value


def _safe_ratio(num: float | int | None, den: float | int | None) -> float | None:
    if num is None or den is None or float(den) == 0:
        return None
    return float(num) / float(den)


def _percentile(vals: list[float], pct: float) -> float | None:
    if not vals:
        return None
    ordered = sorted(vals)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = (len(ordered) - 1) * pct
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(ordered[lo])
    weight = pos - lo
    return float(ordered[lo] * (1 - weight) + ordered[hi] * weight)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def load_energy_model(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "voltage_v": 3.0,
            "current_ma": {"cpu": 1.8, "lpm": 0.0545, "tx": 17.4, "rx": 18.8},
            "rtimer_second": 32768,
            "metric_note": "Cooja/Contiki-NG Energest based simulation-level energy estimate, not hardware power-meter measurement.",
        }
    return json.loads(path.read_text(encoding="utf-8"))


def parse_log(app_log: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, int]]]:
    events: list[dict[str, Any]] = []
    latest_energest: dict[str, dict[str, int]] = {}
    with app_log.open("r", encoding="utf-8", errors="replace") as f:
        for line_number, line in enumerate(f, start=1):
            sim_match = SIM_TIME_RE.search(line)
            sim_time_ms = float(sim_match.group("sim_us")) / 1000.0 if sim_match else None
            tx = TX_RE.search(line)
            if tx:
                payload_time_ms = int(tx.group("time_ms"))
                events.append(
                    {
                        "event": "TX",
                        "packet_type": tx.group("type"),
                        "node": int(tx.group("node")),
                        "src": "",
                        "seq": int(tx.group("seq")),
                        "bytes": int(tx.group("bytes")),
                        "time_ms": sim_time_ms if sim_time_ms is not None else payload_time_ms,
                        "payload_time_ms": payload_time_ms,
                        "sim_time_ms": sim_time_ms,
                        "send_ms": "",
                        "recv_ms": "",
                        "delay_ms": "",
                        "line_number": line_number,
                        "raw_line": line.strip(),
                    }
                )
                continue
            rx = RX_RE.search(line)
            if rx:
                send_ms = int(rx.group("send_ms"))
                recv_ms = int(rx.group("recv_ms"))
                packet_type = rx.group("type")
                events.append(
                    {
                        "event": "RX",
                        "packet_type": packet_type,
                        "node": "",
                        "src": int(rx.group("src")),
                        "seq": int(rx.group("seq")),
                        "bytes": int(rx.group("bytes")),
                        "time_ms": sim_time_ms if sim_time_ms is not None else recv_ms,
                        "payload_time_ms": recv_ms,
                        "sim_time_ms": sim_time_ms,
                        "send_ms": send_ms,
                        "recv_ms": recv_ms,
                        "delay_ms": "",
                        "line_number": line_number,
                        "raw_line": line.strip(),
                    }
                )
                continue
            eg = ENERGEST_RE.search(line)
            if eg:
                latest_energest[eg.group("node")] = {
                    "cpu_ticks": int(eg.group("cpu")),
                    "lpm_ticks": int(eg.group("lpm")),
                    "tx_ticks": int(eg.group("tx")),
                    "rx_ticks": int(eg.group("rx")),
                    "total_ticks": int(eg.group("total")),
                    "time_ms": int(eg.group("time_ms")),
                    "line_number": line_number,
                }
    tx_by_key: dict[tuple[str, int, int], float] = {}
    for event in events:
        if event["event"] == "TX" and event.get("sim_time_ms") not in {"", None}:
            tx_by_key[(str(event["packet_type"]), int(event["node"]), int(event["seq"]))] = float(event["sim_time_ms"])
    for event in events:
        if event["event"] != "RX" or event["packet_type"] != "REAL":
            continue
        key = (str(event["packet_type"]), int(event["src"]), int(event["seq"]))
        tx_time = tx_by_key.get(key)
        if tx_time is not None and event.get("sim_time_ms") not in {"", None}:
            event["delay_ms"] = float(event["sim_time_ms"]) - tx_time
    return events, latest_energest


def write_events(path: Path, events: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "event",
        "packet_type",
        "node",
        "src",
        "seq",
        "bytes",
        "time_ms",
        "payload_time_ms",
        "sim_time_ms",
        "send_ms",
        "recv_ms",
        "delay_ms",
        "line_number",
        "raw_line",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(events)


def summarize_events(
    *,
    seed: int,
    method: str,
    app_log: Path,
    radio_log: Path | None,
    events: list[dict[str, Any]],
    latest_energest: dict[str, dict[str, int]],
    energy_model: dict[str, Any],
    baseline_metrics: dict[str, Any] | None,
) -> dict[str, Any]:
    reasons: list[str] = []
    tx_events = [e for e in events if e["event"] == "TX"]
    rx_events = [e for e in events if e["event"] == "RX"]
    count_events = rx_events if rx_events else tx_events

    real_events = [e for e in count_events if e["packet_type"] == "REAL"]
    dummy_events = [e for e in count_events if e["packet_type"] == "DUMMY"]
    real_packet_count = len(real_events)
    dummy_packet_count = len(dummy_events)
    total_packet_count = real_packet_count + dummy_packet_count
    real_byte_count = sum(int(e["bytes"]) for e in real_events)
    dummy_byte_count = sum(int(e["bytes"]) for e in dummy_events)
    total_byte_count = real_byte_count + dummy_byte_count

    real_rx_delays = [float(e["delay_ms"]) for e in rx_events if e["packet_type"] == "REAL" and e["delay_ms"] != ""]
    tx_times = sorted(float(e["time_ms"]) for e in tx_events)
    iats = [b - a for a, b in zip(tx_times, tx_times[1:])]

    if not tx_events and not rx_events:
        reasons.append("log_does_not_contain_metric_tx_or_metric_rx")
    if not real_rx_delays:
        reasons.append("no_real_metric_rx_delay_samples")

    tick_second = float(energy_model.get("rtimer_second") or energy_model.get("energest_second") or 32768)
    current = dict(energy_model.get("current_ma", {}))
    voltage = float(energy_model.get("voltage_v", 3.0))

    cpu_ticks = sum(v["cpu_ticks"] for v in latest_energest.values())
    lpm_ticks = sum(v["lpm_ticks"] for v in latest_energest.values())
    tx_ticks = sum(v["tx_ticks"] for v in latest_energest.values())
    rx_ticks = sum(v["rx_ticks"] for v in latest_energest.values())

    cpu_time_s = cpu_ticks / tick_second if tick_second else None
    lpm_time_s = lpm_ticks / tick_second if tick_second else None
    tx_time_s = tx_ticks / tick_second if tick_second else None
    rx_time_s = rx_ticks / tick_second if tick_second else None
    energy_mj = None
    if latest_energest and all(v is not None for v in [cpu_time_s, lpm_time_s, tx_time_s, rx_time_s]):
        energy_mj = voltage * (
            float(cpu_time_s) * float(current.get("cpu", 0.0))
            + float(lpm_time_s) * float(current.get("lpm", 0.0))
            + float(tx_time_s) * float(current.get("tx", 0.0))
            + float(rx_time_s) * float(current.get("rx", 0.0))
        )
    else:
        reasons.append("log_does_not_contain_energest_counters")

    baseline_energy = None
    if baseline_metrics:
        baseline_energy = baseline_metrics.get("energy_mj")
    energy_overhead = None
    if method != "baseline":
        energy_overhead = _safe_ratio(
            None if energy_mj is None or baseline_energy is None else energy_mj - float(baseline_energy),
            baseline_energy,
        )
        if energy_overhead is None:
            reasons.append("baseline_energy_unavailable_for_overhead_ratio")

    metrics = {
        "seed": int(seed),
        "method": method,
        "source_app_log": _rel(app_log),
        "source_radio_log": _rel(radio_log) if radio_log else None,
        "real_packet_count": real_packet_count,
        "dummy_packet_count": dummy_packet_count,
        "total_packet_count": total_packet_count,
        "dummy_packet_ratio": _safe_ratio(dummy_packet_count, total_packet_count),
        "packet_overhead_ratio": _safe_ratio(dummy_packet_count, real_packet_count),
        "real_byte_count": real_byte_count,
        "dummy_byte_count": dummy_byte_count,
        "total_byte_count": total_byte_count,
        "dummy_byte_ratio": _safe_ratio(dummy_byte_count, total_byte_count),
        "byte_overhead_ratio": _safe_ratio(dummy_byte_count, real_byte_count),
        "mean_iat_ms": mean(iats) if iats else None,
        "p95_iat_ms": _percentile(iats, 0.95),
        "mean_delay_ms": mean(real_rx_delays) if real_rx_delays else None,
        "median_delay_ms": median(real_rx_delays) if real_rx_delays else None,
        "p95_delay_ms": _percentile(real_rx_delays, 0.95),
        "delay_sample_count": len(real_rx_delays),
        "cpu_time_s": cpu_time_s,
        "lpm_time_s": lpm_time_s,
        "tx_time_s": tx_time_s,
        "rx_time_s": rx_time_s,
        "energy_mj": energy_mj,
        "energy_overhead_ratio_vs_baseline": energy_overhead,
        "metric_type": "cooja_simulation_estimate",
        "delay_metric_type": "cooja_simulation_time",
        "energy_metric_type": "energest_simulation_estimate",
        "is_hardware_measurement": False,
        "energy_model": energy_model,
        "unavailable_reason": sorted(set(reasons)),
    }
    return metrics


def parse_one(
    *,
    app_log: Path,
    radio_log: Path | None,
    method: str,
    seed: int,
    out_dir: Path,
    energy_model_path: Path,
    baseline_metrics_path: Path | None = None,
) -> dict[str, Any]:
    energy_model = load_energy_model(energy_model_path)
    baseline_metrics = _read_json(baseline_metrics_path) if baseline_metrics_path else None
    events, latest_energest = parse_log(app_log)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_events(out_dir / "overhead_events.csv", events)
    metrics = summarize_events(
        seed=seed,
        method=method,
        app_log=app_log,
        radio_log=radio_log,
        events=events,
        latest_energest=latest_energest,
        energy_model=energy_model,
        baseline_metrics=baseline_metrics,
    )
    (out_dir / "overhead_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description="解析 Cooja METRIC_TX/METRIC_RX/ENERGEST 结构化日志。")
    ap.add_argument("--app-log", required=True, help="Cooja LogListener 输出文件。")
    ap.add_argument("--radio-log", default=None, help="可选 RadioLogger 输出文件，仅记录来源。")
    ap.add_argument("--method", required=True, choices=["baseline", "dummy_noise", "dummy_ldp", "dummy_adaptive_ldp"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--energy-model", default=str(DEFAULT_ENERGY_MODEL))
    ap.add_argument("--baseline-metrics", default=None, help="同一 seed 的 baseline overhead_metrics.json，用于计算能耗开销比例。")
    args = ap.parse_args()

    app_log = Path(args.app_log)
    radio_log = Path(args.radio_log) if args.radio_log else None
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
      out_dir = ROOT / out_dir
    metrics = parse_one(
        app_log=app_log,
        radio_log=radio_log,
        method=args.method,
        seed=args.seed,
        out_dir=out_dir,
        energy_model_path=Path(args.energy_model),
        baseline_metrics_path=Path(args.baseline_metrics) if args.baseline_metrics else None,
    )
    print(json.dumps({"output": _rel(out_dir / "overhead_metrics.json"), "metric_type": metrics["metric_type"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
