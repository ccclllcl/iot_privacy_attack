#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""运行 Cooja 开销仿真，并调用解析器生成节点级开销产物。"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.cooja.parse_cooja_overhead_metrics import parse_one
from experiments.cooja.summarize_cooja_overhead_metrics import summarize


ROOT = Path(__file__).resolve().parents[2]
TEMPLATE_DIR = ROOT / "cooja" / "contiki-ng"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "experiments" / "cooja"
DEFAULT_CONTIKI_ROOT = "/home/linchen/iot-privacy-project/contiki-ng"
DEFAULT_SCENARIO_ROOT = "/home/linchen/iot-privacy-project"
DEFAULT_WSL_OUTPUT_ROOT = "/home/linchen/iot-privacy-project/cooja_overhead_runs"
METHOD_SCENARIOS = {
    "baseline": "baseline_no_defense.csc",
    "dummy_noise": "dummy_noise.csc",
    "dummy_ldp": "dummy_ldp.csc",
    "dummy_adaptive_ldp": "dummy_adaptive_ldp.csc",
}
METHODS = list(METHOD_SCENARIOS)
SEEDS = [42, 123, 2026]


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except Exception:
        return path.as_posix().replace("\\", "/")


def _wsl_to_unc(path: str, distro: str) -> Path:
    return Path("\\\\wsl$") / distro / path.strip("/").replace("/", "\\")


def _quote_sh(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def parse_csv_list(value: str, allowed: list[str] | None = None) -> list[str]:
    items = [part.strip() for part in value.split(",") if part.strip()]
    if allowed:
        bad = [x for x in items if x not in allowed]
        if bad:
            raise ValueError(f"不支持的取值: {bad}")
    return items


def parse_seed_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def sync_sources(contiki_root: str, distro: str) -> list[str]:
    target = _wsl_to_unc(f"{contiki_root}/examples/rpl-udp", distro)
    if not target.exists():
        raise FileNotFoundError(f"找不到 WSL Contiki-NG rpl-udp 目录: {target}")
    copied: list[str] = []
    for src in TEMPLATE_DIR.iterdir():
        if src.is_file() and src.suffix in {".c", ".h"}:
            dst = target / src.name
            shutil.copy2(src, dst)
            copied.append(str(dst))
    return copied


def prepare_scenario(
    *,
    scenario_root: str,
    method: str,
    seed: int,
    duration_seconds: int,
    wsl_run_dir: str,
    distro: str,
) -> str:
    scenario_name = METHOD_SCENARIOS[method]
    src_unc = _wsl_to_unc(f"{scenario_root}/{scenario_name}", distro)
    if not src_unc.exists():
        raise FileNotFoundError(f"找不到 Cooja 场景文件: {src_unc}")
    text = src_unc.read_text(encoding="utf-8", errors="replace")
    text = re.sub(r"<randomseed>.*?</randomseed>", f"<randomseed>{seed}</randomseed>", text, count=1)
    text = re.sub(r"<logoutput>.*?</logoutput>", f"<logoutput>{int(duration_seconds) * 1000}</logoutput>", text, count=1)
    if "org.contikios.cooja.plugins.ScriptRunner" not in text:
        duration_ms = int(duration_seconds) * 1000
        script_plugin = f"""
  <plugin>
    org.contikios.cooja.plugins.ScriptRunner
    <plugin_config>
      <script>TIMEOUT({duration_ms + 60000});
GENERATE_MSG({duration_ms}, "end test");
while (true) {{
  YIELD();
  log.log(time + "\\tID:" + id + "\\t" + msg + "\\n");
  if (msg.equals("end test")) {{
    log.testOK();
  }}
}}</script>
      <active>true</active>
    </plugin_config>
    <bounds x="100" y="100" height="300" width="500" />
  </plugin>
"""
        text = text.replace("</simconf>", script_plugin + "</simconf>")
    run_unc = _wsl_to_unc(wsl_run_dir, distro)
    run_unc.mkdir(parents=True, exist_ok=True)
    dst_unc = run_unc / f"{method}_seed_{seed}.csc"
    dst_unc.write_text(text, encoding="utf-8")
    return f"{wsl_run_dir}/{dst_unc.name}"


def run_cooja(
    *,
    contiki_root: str,
    scenario_path: str,
    wsl_log_dir: str,
    seed: int,
    timeout_seconds: int | None,
) -> subprocess.CompletedProcess[str]:
    cooja_dir = f"{contiki_root}/tools/cooja"
    args = f"--no-gui --logdir={wsl_log_dir} --contiki={contiki_root} --random-seed={seed} --autostart {scenario_path}"
    command = f"cd {_quote_sh(cooja_dir)} && ./gradlew run --args={_quote_sh(args)}"
    return subprocess.run(
        ["wsl", "-e", "sh", "-lc", command],
        cwd=ROOT,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_seconds if timeout_seconds and timeout_seconds > 0 else None,
    )


def _find_app_log(canonical_dir: Path, wsl_log_dir: str, distro: str) -> Path:
    log_unc = _wsl_to_unc(wsl_log_dir, distro)
    candidates: list[Path] = []
    if log_unc.exists():
        candidates.extend(sorted(log_unc.glob("*.testlog")))
        candidates.extend(sorted(log_unc.glob("*log*.txt")))
        candidates.extend(sorted(log_unc.glob("*.log")))
    for path in candidates:
        if path.is_file() and path.stat().st_size > 0:
            dst = canonical_dir / path.name
            shutil.copy2(path, dst)
            return dst
    stdout = canonical_dir / "cooja_stdout.log"
    return stdout


def _copy_log_dir(canonical_dir: Path, wsl_log_dir: str, distro: str) -> list[str]:
    copied: list[str] = []
    log_unc = _wsl_to_unc(wsl_log_dir, distro)
    if not log_unc.exists():
        return copied
    raw_dir = canonical_dir / "raw_logs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for path in log_unc.iterdir():
        if path.is_file():
            dst = raw_dir / path.name
            shutil.copy2(path, dst)
            copied.append(_rel(dst))
    return copied


def run_one(
    *,
    method: str,
    seed: int,
    contiki_root: str,
    scenario_root: str,
    wsl_output_root: str,
    output_root: Path,
    duration_seconds: int,
    distro: str,
    timeout_seconds: int | None,
    skip_existing: bool,
    dry_run: bool,
) -> dict[str, Any]:
    canonical_dir = output_root / f"seed_{seed}" / "random_forest" / method
    metrics_path = canonical_dir / "overhead_metrics.json"
    if skip_existing and metrics_path.exists() and metrics_path.stat().st_size > 0:
        return {"method": method, "seed": seed, "status": "skipped_existing", "output": _rel(metrics_path)}
    canonical_dir.mkdir(parents=True, exist_ok=True)
    wsl_run_dir = f"{wsl_output_root}/seed_{seed}/{method}"
    scenario_path = prepare_scenario(
        scenario_root=scenario_root,
        method=method,
        seed=seed,
        duration_seconds=duration_seconds,
        wsl_run_dir=wsl_run_dir,
        distro=distro,
    )
    if dry_run:
        return {"method": method, "seed": seed, "status": "dry_run", "scenario": scenario_path}

    proc = run_cooja(
        contiki_root=contiki_root,
        scenario_path=scenario_path,
        wsl_log_dir=wsl_run_dir,
        seed=seed,
        timeout_seconds=timeout_seconds,
    )
    (canonical_dir / "cooja_stdout.log").write_text(proc.stdout or "", encoding="utf-8", errors="replace")
    (canonical_dir / "cooja_stderr.log").write_text(proc.stderr or "", encoding="utf-8", errors="replace")
    copied_logs = _copy_log_dir(canonical_dir, wsl_run_dir, distro)
    app_log = _find_app_log(canonical_dir, wsl_run_dir, distro)
    radio_log = None
    for copied in copied_logs:
        name = Path(copied).name.lower()
        if "radio" in name:
            radio_log = ROOT / copied
            break

    baseline_metrics = None
    if method != "baseline":
        baseline_metrics = output_root / f"seed_{seed}" / "random_forest" / "baseline" / "overhead_metrics.json"
    metrics = parse_one(
        app_log=app_log,
        radio_log=radio_log,
        method=method,
        seed=seed,
        out_dir=canonical_dir,
        energy_model_path=ROOT / "configs" / "cooja_energy_model.json",
        baseline_metrics_path=baseline_metrics if baseline_metrics and baseline_metrics.exists() else None,
    )
    status = "ok" if proc.returncode == 0 else "cooja_returned_nonzero"
    return {
        "method": method,
        "seed": seed,
        "status": status,
        "returncode": proc.returncode,
        "output": _rel(metrics_path),
        "events": len((canonical_dir / "overhead_events.csv").read_text(encoding="utf-8").splitlines()) - 1,
        "unavailable_reason": metrics.get("unavailable_reason", []),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="运行 Cooja 节点级开销仿真并解析 METRIC_TX/METRIC_RX/ENERGEST 日志。")
    ap.add_argument("--contiki-ng-root", default=DEFAULT_CONTIKI_ROOT)
    ap.add_argument("--scenario-root", default=DEFAULT_SCENARIO_ROOT)
    ap.add_argument("--methods", default="baseline,dummy_noise,dummy_ldp,dummy_adaptive_ldp")
    ap.add_argument("--seeds", default="42,123,2026")
    ap.add_argument("--duration-seconds", type=int, default=1800)
    ap.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    ap.add_argument("--wsl-output-root", default=DEFAULT_WSL_OUTPUT_ROOT)
    ap.add_argument("--wsl-distro", default="Ubuntu-22.04")
    ap.add_argument("--timeout-seconds", type=int, default=0, help="0 表示不设置额外超时，依赖 Cooja logoutput 结束。")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--no-sync-sources", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    methods = parse_csv_list(args.methods, METHODS)
    seeds = parse_seed_list(args.seeds)
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = ROOT / output_root

    copied_sources: list[str] = []
    if not args.no_sync_sources:
        copied_sources = sync_sources(args.contiki_ng_root, args.wsl_distro)

    results: list[dict[str, Any]] = []
    for seed in seeds:
        ordered_methods = [m for m in METHODS if m in methods]
        for method in ordered_methods:
            result = run_one(
                method=method,
                seed=seed,
                contiki_root=args.contiki_ng_root,
                scenario_root=args.scenario_root,
                wsl_output_root=args.wsl_output_root,
                output_root=output_root,
                duration_seconds=args.duration_seconds,
                distro=args.wsl_distro,
                timeout_seconds=args.timeout_seconds,
                skip_existing=args.skip_existing,
                dry_run=args.dry_run,
            )
            print(json.dumps(result, ensure_ascii=False))
            results.append(result)

    if not args.dry_run:
        summarize(output_root)

    run_log = {
        "copied_sources": copied_sources,
        "results": results,
        "output_root": _rel(output_root),
    }
    log_path = ROOT / "outputs" / "summaries" / "final_thesis" / "cooja" / "cooja_overhead_run_log.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(run_log, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
