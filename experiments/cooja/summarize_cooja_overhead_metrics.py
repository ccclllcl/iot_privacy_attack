#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""汇总 Cooja 节点级开销指标，并回写 final_thesis Cooja 结果表。"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = ROOT / "outputs" / "experiments" / "cooja"
SUMMARY_ROOT = ROOT / "outputs" / "summaries" / "final_thesis"
COOJA_SUMMARY_DIR = SUMMARY_ROOT / "cooja"
METHODS = ["baseline", "dummy_noise", "dummy_ldp", "dummy_adaptive_ldp"]
DEFENSE_METHODS = ["dummy_noise", "dummy_ldp", "dummy_adaptive_ldp"]
SEEDS = [42, 123, 2026]


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except Exception:
        return path.as_posix().replace("\\", "/")


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _to_float(value: Any) -> float | None:
    try:
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def _mean(series: pd.Series) -> float | None:
    vals = pd.to_numeric(series, errors="coerce").dropna()
    if vals.empty:
        return None
    return float(vals.mean())


def _metric_mean(report: dict[str, Any], section: str, metric: str) -> float | None:
    try:
        return float(((report.get(section) or {}).get(metric) or {}).get("mean"))
    except Exception:
        return None


def _metric_value(report: dict[str, Any], metric: str) -> float | None:
    try:
        return float(report.get(metric))
    except Exception:
        return None


def update_accuracy_from_eval_report() -> bool:
    report_path = EXPERIMENT_ROOT / "eval" / "defense_eval_report.json"
    report = _read_json(report_path)
    if not isinstance(report, dict):
        return False
    methods = report.get("methods", {})
    if not isinstance(methods, dict) or not methods:
        return False

    summary_rows: list[dict[str, Any]] = []
    per_seed_rows: list[dict[str, Any]] = []
    for method_name, method_obj in methods.items():
        if method_name not in DEFENSE_METHODS or not isinstance(method_obj, dict):
            continue
        paths = method_obj.get("defense_log_paths", {}) if isinstance(method_obj.get("defense_log_paths"), dict) else {}
        source_files = json.dumps(paths, ensure_ascii=False)
        baseline_acc = _metric_mean(method_obj, "baseline_test", "accuracy")
        for mode, section in [
            ("fixed_attacker", "fixed_attacker"),
            ("retrain_attacker", "retrain_attacker"),
        ]:
            defended_acc = _metric_mean(method_obj, section, "accuracy")
            f1_macro = _metric_mean(method_obj, section, "f1_macro")
            summary_rows.append(
                {
                    "accuracy_drop": None
                    if baseline_acc is None or defended_acc is None
                    else baseline_acc - defended_acc,
                    "baseline_acc": baseline_acc,
                    "byte_count_mean": None,
                    "correlation_drop": None,
                    "defended_acc": defended_acc,
                    "delay_proxy_available": None,
                    "dummy_packet_ratio": None,
                    "energy_metric_available": None,
                    "f1_macro": f1_macro,
                    "mean_iat_ms": None,
                    "method": method_name,
                    "mode": mode,
                    "p95_iat_ms": None,
                    "packet_overhead_ratio": None,
                    "pkt_count_mean": None,
                    "seed": "mean_over_seeds",
                    "source_log_files": source_files,
                    "traffic_activity_correlation_after": None,
                    "traffic_activity_correlation_before": None,
                }
            )

        for run in method_obj.get("runs", []) or []:
            if not isinstance(run, dict):
                continue
            seed = int(run.get("seed"))
            run_dataset = run.get("dataset", {}) if isinstance(run.get("dataset"), dict) else {}
            run_paths = run.get("source_log_paths", {}) if isinstance(run.get("source_log_paths"), dict) else {}
            base = run.get("baseline_test", {}) if isinstance(run.get("baseline_test"), dict) else {}
            for mode, key in [
                ("fixed_attacker", "fixed_attacker_on_defense"),
                ("retrain_attacker", "retrain_attacker_on_defense"),
            ]:
                defended = run.get(key, {}) if isinstance(run.get(key), dict) else {}
                b_acc = _metric_value(base, "accuracy")
                d_acc = _metric_value(defended, "accuracy")
                per_seed_rows.append(
                    {
                        "method": method_name,
                        "mode": mode,
                        "seed": seed,
                        "baseline_acc": b_acc,
                        "defended_acc": d_acc,
                        "accuracy_drop": None if b_acc is None or d_acc is None else b_acc - d_acc,
                        "baseline_f1_macro": _metric_value(base, "f1_macro"),
                        "defended_f1_macro": _metric_value(defended, "f1_macro"),
                        "baseline_windows": run_dataset.get("baseline_windows"),
                        "defense_windows": run_dataset.get("defense_windows"),
                        "source_radio_log": run_paths.get("defense_radio_log"),
                        "source_app_log": run_paths.get("defense_app_log") or paths.get("app_log"),
                    }
                )

    if not summary_rows:
        return False
    COOJA_SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(COOJA_SUMMARY_DIR / "cooja_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(per_seed_rows).to_csv(COOJA_SUMMARY_DIR / "cooja_per_seed.csv", index=False, encoding="utf-8-sig")
    _write_json(COOJA_SUMMARY_DIR / "cooja_summary.json", summary_rows)
    return True


def load_metrics(experiment_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for seed in SEEDS:
        for method in METHODS:
            path = experiment_root / f"seed_{seed}" / "random_forest" / method / "overhead_metrics.json"
            data = _read_json(path)
            if data is None:
                missing.append({"seed": seed, "method": method, "expected_file": _rel(path), "reason": "missing_or_unreadable"})
                continue
            data = dict(data)
            data["source_file"] = _rel(path)
            rows.append(data)
    rows.sort(key=lambda r: (int(r.get("seed", 0)), METHODS.index(str(r.get("method")))))
    return rows, missing


def write_metrics_tables(rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    out_csv = COOJA_SUMMARY_DIR / "cooja_overhead_metrics.csv"
    out_json = COOJA_SUMMARY_DIR / "cooja_overhead_metrics.json"
    COOJA_SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    _write_json(out_json, rows)
    return df


def build_traffic_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if df.empty:
        return rows
    baseline_by_seed = {int(r["seed"]): r for _, r in df[df["method"] == "baseline"].iterrows()}
    for _, row in df[df["method"].isin(DEFENSE_METHODS)].iterrows():
        seed = int(row["seed"])
        base = baseline_by_seed.get(seed)
        rows.append(
            {
                "method": row.get("method"),
                "seed": seed,
                "baseline_windows": "",
                "defense_windows": "",
                "baseline_pkt_count_mean": _to_float(base.get("total_packet_count")) if base is not None else None,
                "defense_pkt_count_mean": _to_float(row.get("total_packet_count")),
                "baseline_byte_count_mean": _to_float(base.get("total_byte_count")) if base is not None else None,
                "defense_byte_count_mean": _to_float(row.get("total_byte_count")),
                "packet_overhead_ratio": _to_float(row.get("packet_overhead_ratio")),
                "byte_overhead_ratio": _to_float(row.get("byte_overhead_ratio")),
                "baseline_mean_iat_ms": _to_float(base.get("mean_iat_ms")) if base is not None else None,
                "defense_mean_iat_ms": _to_float(row.get("mean_iat_ms")),
                "baseline_p95_iat_ms": _to_float(base.get("p95_iat_ms")) if base is not None else None,
                "defense_p95_iat_ms": _to_float(row.get("p95_iat_ms")),
                "dummy_packet_ratio": _to_float(row.get("dummy_packet_ratio")),
                "dummy_byte_ratio": _to_float(row.get("dummy_byte_ratio")),
                "real_packet_count": _to_float(row.get("real_packet_count")),
                "dummy_packet_count": _to_float(row.get("dummy_packet_count")),
                "real_byte_count": _to_float(row.get("real_byte_count")),
                "dummy_byte_count": _to_float(row.get("dummy_byte_count")),
                "mean_delay_ms": _to_float(row.get("mean_delay_ms")),
                "p95_delay_ms": _to_float(row.get("p95_delay_ms")),
                "energy_mj": _to_float(row.get("energy_mj")),
                "energy_overhead_ratio_vs_baseline": _to_float(row.get("energy_overhead_ratio_vs_baseline")),
                "energy_metric_available": pd.notna(row.get("energy_mj")),
                "delay_metric_available": pd.notna(row.get("mean_delay_ms")),
                "metric_type": "cooja_simulation_and_energest_estimate",
                "is_hardware_measurement": False,
                "limitations": "Cooja simulation-time delay and Energest estimate; not hardware measurement.",
            }
        )
    return rows


def build_overhead_summary(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if df.empty:
        return rows
    for method in METHODS:
        part = df[df["method"] == method]
        if part.empty:
            continue
        rows.append(
            {
                "method": method,
                "seed_count": int(part["seed"].nunique()),
                "real_packet_count_mean": _mean(part["real_packet_count"]),
                "dummy_packet_count_mean": _mean(part["dummy_packet_count"]),
                "total_packet_count_mean": _mean(part["total_packet_count"]),
                "dummy_packet_ratio_mean": _mean(part["dummy_packet_ratio"]),
                "packet_overhead_ratio_mean": _mean(part["packet_overhead_ratio"]),
                "real_byte_count_mean": _mean(part["real_byte_count"]),
                "dummy_byte_count_mean": _mean(part["dummy_byte_count"]),
                "total_byte_count_mean": _mean(part["total_byte_count"]),
                "dummy_byte_ratio_mean": _mean(part["dummy_byte_ratio"]),
                "byte_overhead_ratio_mean": _mean(part["byte_overhead_ratio"]),
                "mean_iat_ms_mean": _mean(part["mean_iat_ms"]),
                "p95_iat_ms_mean": _mean(part["p95_iat_ms"]),
                "mean_delay_ms_mean": _mean(part["mean_delay_ms"]),
                "p95_delay_ms_mean": _mean(part["p95_delay_ms"]),
                "energy_mj_mean": _mean(part["energy_mj"]),
                "energy_overhead_ratio_mean": _mean(part["energy_overhead_ratio_vs_baseline"]),
                "delay_metric_available": bool(pd.to_numeric(part["mean_delay_ms"], errors="coerce").notna().any()),
                "energy_metric_available": bool(pd.to_numeric(part["energy_mj"], errors="coerce").notna().any()),
                "metric_type": "cooja_simulation_and_energest_estimate",
                "is_hardware_measurement": False,
            }
        )
    return rows


def update_cooja_summary(df: pd.DataFrame) -> None:
    summary_path = COOJA_SUMMARY_DIR / "cooja_summary.csv"
    if not summary_path.exists():
        return
    summary = pd.read_csv(summary_path)
    for col in ["delay_proxy_available", "energy_metric_available", "is_hardware_measurement"]:
        if col in summary.columns:
            summary[col] = summary[col].astype("object")
    overhead = pd.DataFrame(build_overhead_summary(df))
    if overhead.empty:
        return
    for method in DEFENSE_METHODS:
        if method not in set(overhead["method"]):
            continue
        stat = overhead[overhead["method"] == method].iloc[0].to_dict()
        mask = summary["method"] == method
        summary.loc[mask, "byte_count_mean"] = stat.get("total_byte_count_mean")
        summary.loc[mask, "dummy_packet_ratio"] = stat.get("dummy_packet_ratio_mean")
        summary.loc[mask, "mean_iat_ms"] = stat.get("mean_iat_ms_mean")
        summary.loc[mask, "p95_iat_ms"] = stat.get("p95_iat_ms_mean")
        summary.loc[mask, "packet_overhead_ratio"] = stat.get("packet_overhead_ratio_mean")
        summary.loc[mask, "pkt_count_mean"] = stat.get("total_packet_count_mean")
        summary.loc[mask, "delay_proxy_available"] = bool(stat.get("delay_metric_available"))
        summary.loc[mask, "energy_metric_available"] = bool(stat.get("energy_metric_available"))
        summary.loc[mask, "real_packet_count_mean"] = stat.get("real_packet_count_mean")
        summary.loc[mask, "dummy_packet_count_mean"] = stat.get("dummy_packet_count_mean")
        summary.loc[mask, "real_byte_count_mean"] = stat.get("real_byte_count_mean")
        summary.loc[mask, "dummy_byte_count_mean"] = stat.get("dummy_byte_count_mean")
        summary.loc[mask, "byte_overhead_ratio"] = stat.get("byte_overhead_ratio_mean")
        summary.loc[mask, "mean_delay_ms"] = stat.get("mean_delay_ms_mean")
        summary.loc[mask, "p95_delay_ms"] = stat.get("p95_delay_ms_mean")
        summary.loc[mask, "energy_mj_mean"] = stat.get("energy_mj_mean")
        summary.loc[mask, "energy_overhead_ratio"] = stat.get("energy_overhead_ratio_mean")
        summary.loc[mask, "metric_type"] = "cooja_simulation_and_energest_estimate"
        summary.loc[mask, "is_hardware_measurement"] = False
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    _write_json(COOJA_SUMMARY_DIR / "cooja_summary.json", summary.where(pd.notna(summary), None).to_dict(orient="records"))


def update_final_summary() -> None:
    final_csv = SUMMARY_ROOT / "final_summary.csv"
    cooja_csv = COOJA_SUMMARY_DIR / "cooja_summary.csv"
    if not final_csv.exists() or not cooja_csv.exists():
        return
    final = pd.read_csv(final_csv)
    cooja = pd.read_csv(cooja_csv)
    non_cooja = final[final.get("section") != "cooja"].copy()
    cooja = cooja.copy()
    cooja["section"] = "cooja"
    combined = pd.concat([non_cooja, cooja], ignore_index=True, sort=False)
    combined.to_csv(final_csv, index=False, encoding="utf-8-sig")
    _write_json(SUMMARY_ROOT / "final_summary.json", combined.where(pd.notna(combined), None).to_dict(orient="records"))


def write_limitations(success: bool, missing: list[dict[str, Any]]) -> None:
    if success:
        lines = [
            "# Cooja Limitations",
            "",
            "- Cooja outputs can be used for fixed/retrain attacker accuracy reporting.",
            "- Cooja overhead metrics now include dummy/real packet ratio, packet/byte overhead, Cooja simulation-time delay, and Contiki-NG Energest-based energy estimate.",
            "- Energy values are simulation-level estimates based on Energest counters and current-draw configuration, not hardware power-meter measurements.",
            "- Delay values are Cooja simulation-time end-to-end delays, not real deployment latency.",
            "- Dummy/real packet ratios are computed from explicitly labeled METRIC_TX/METRIC_RX logs.",
        ]
    else:
        lines = [
            "# Cooja Limitations",
            "",
            "- Cooja overhead metrics are not complete.",
            "- Unavailable packet, byte, delay, or energy fields remain null and are not fabricated.",
            f"- Missing entries: {len(missing)}.",
        ]
    (COOJA_SUMMARY_DIR / "cooja_limitations.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_completion_report(df: pd.DataFrame, missing: list[dict[str, Any]]) -> None:
    rows = df.to_dict(orient="records") if not df.empty else []
    success = not missing and not df.empty
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "modified_contiki_sources": [
            "cooja/contiki-ng/udp-client-baseline.c",
            "cooja/contiki-ng/udp-client-mix.c",
            "cooja/contiki-ng/udp-client-mix-adaptive.c",
            "cooja/contiki-ng/udp-server-metrics.c",
            "cooja/contiki-ng/udp-metric-common.h",
        ],
        "structured_log_fields": ["METRIC_TX", "METRIC_RX", "ENERGEST"],
        "methods": METHODS,
        "seeds": SEEDS,
        "metrics_generated": [
            "dummy_packet_ratio",
            "packet_overhead_ratio",
            "byte_overhead_ratio",
            "mean_delay_ms",
            "p95_delay_ms",
            "energy_mj",
            "energy_overhead_ratio_vs_baseline",
        ],
        "missing": missing,
        "success": success,
        "word_document_modified": False,
        "metric_note": "Energy is an Energest simulation estimate; delay is Cooja simulation-time end-to-end delay. dummy_adaptive_ldp uses recent send-intensity driven adaptive epsilon and dummy probability.",
    }
    _write_json(SUMMARY_ROOT / "cooja_overhead_completion_report.json", report)

    lines = [
        "# Cooja 节点级开销补全报告",
        "",
        f"- 生成时间: `{report['generated_at']}`",
        "- 是否修改论文 Word: 否",
        "- 日志字段: `METRIC_TX`、`METRIC_RX`、`ENERGEST`",
        "- 能耗口径: Contiki-NG Energest 仿真估计，不是硬件功耗仪测量。",
        "- 时延口径: Cooja 仿真时间下的 REAL 包端到端时延。",
        f"- 覆盖方法: `{', '.join(METHODS)}`",
        f"- 覆盖 seed: `{', '.join(str(s) for s in SEEDS)}`",
        f"- 缺失项数量: `{len(missing)}`",
        "",
        "## 生成产物",
        "",
        "- `outputs/summaries/final_thesis/cooja/cooja_overhead_metrics.csv`",
        "- `outputs/summaries/final_thesis/cooja/cooja_overhead_metrics.json`",
        "- `outputs/summaries/final_thesis/cooja/cooja_traffic_metrics.csv`",
        "- `outputs/summaries/final_thesis/cooja/cooja_overhead_summary.csv`",
        "- `outputs/figures/summaries/final_thesis/thesis_fig4_13_cooja_overhead_metrics.png`",
        "- `outputs/figures/summaries/final_thesis/thesis_fig4_14_cooja_dummy_ratio.png`",
        "- `outputs/figures/summaries/final_thesis/thesis_fig4_15_cooja_energy_delay.png`",
    ]
    if rows:
        lines.extend(["", "## 指标概览", ""])
        overview = pd.DataFrame(build_overhead_summary(pd.DataFrame(rows)))
        if not overview.empty:
            lines.append("```csv")
            lines.append(overview.to_csv(index=False, lineterminator="\n").strip())
            lines.append("```")
    if missing:
        lines.extend(["", "## 仍不可用字段或组合", ""])
        for item in missing:
            lines.append(f"- `{item.get('method')}` seed `{item.get('seed')}`: {item.get('reason')} ({item.get('expected_file')})")
    (SUMMARY_ROOT / "cooja_overhead_completion_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize(experiment_root: Path = EXPERIMENT_ROOT) -> dict[str, Any]:
    rows, missing = load_metrics(experiment_root)
    df = write_metrics_tables(rows)
    traffic_rows = build_traffic_rows(df)
    overhead_rows = build_overhead_summary(df)
    pd.DataFrame(traffic_rows).to_csv(COOJA_SUMMARY_DIR / "cooja_traffic_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(overhead_rows).to_csv(COOJA_SUMMARY_DIR / "cooja_overhead_summary.csv", index=False, encoding="utf-8-sig")
    update_accuracy_from_eval_report()
    update_cooja_summary(df)
    update_final_summary()
    write_limitations(not missing and not df.empty, missing)
    write_completion_report(df, missing)
    return {"rows": len(rows), "missing": missing}


def main() -> None:
    ap = argparse.ArgumentParser(description="汇总 Cooja 节点级 packet/byte/delay/Energest 开销指标。")
    ap.add_argument("--experiment-root", default=str(EXPERIMENT_ROOT))
    args = ap.parse_args()
    result = summarize(Path(args.experiment_root))
    print(json.dumps({"rows": result["rows"], "missing": len(result["missing"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
