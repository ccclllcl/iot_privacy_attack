from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    timestamp_utc: str
    action: str
    command: list[str]
    cwd: str
    config_path: str | None
    status: str  # "success" | "failed"
    return_code: int
    duration_seconds: float
    stdout: str
    stderr: str
    artifacts: dict[str, str]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _new_run_id() -> str:
    # short + sortable enough for local experiments
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"-{os.getpid()}"


def append_history(history_path: str | Path, record: RunRecord) -> None:
    path = Path(history_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def load_history(history_path: str | Path, limit: int = 200) -> list[dict[str, Any]]:
    path = Path(history_path)
    if not path.exists():
        return []

    # Read last N lines (simple approach; file is expected to be small)
    lines: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(line)

    tail = lines[-limit:]
    out: list[dict[str, Any]] = []
    for line in reversed(tail):
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _python_cmd_unbuffered(command: list[str]) -> list[str]:
    """Insert CPython -u so child logging/tqdm lines flush promptly when stdout is a pipe."""
    c = list(command)
    if len(c) >= 2 and c[1] != "-u":
        c.insert(1, "-u")
    return c


def run_and_record(
    *,
    action: str,
    command: list[str],
    cwd: str | Path,
    history_path: str | Path,
    config_path: str | None = None,
    env: dict[str, str] | None = None,
    artifacts: dict[str, str] | None = None,
    timeout_seconds: int | None = None,
    on_output_line: Callable[[str], None] | None = None,
) -> RunRecord:
    import subprocess  # local import: keep base deps minimal

    start = time.perf_counter()
    merged_env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    if env is not None:
        merged_env.update(env)

    if on_output_line is None:
        proc = subprocess.run(
            command,
            cwd=str(cwd),
            env=merged_env,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=timeout_seconds,
        )
        end = time.perf_counter()
        record = RunRecord(
            run_id=_new_run_id(),
            timestamp_utc=_utc_now_iso(),
            action=action,
            command=command,
            cwd=str(cwd),
            config_path=config_path,
            status="success" if proc.returncode == 0 else "failed",
            return_code=int(proc.returncode),
            duration_seconds=float(end - start),
            stdout=(proc.stdout or "")[-200_000:],
            stderr=(proc.stderr or "")[-200_000:],
            artifacts={} if artifacts is None else dict(artifacts),
        )
        append_history(history_path, record)
        return record

    cmd = _python_cmd_unbuffered(command)
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=merged_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    stdout_parts: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        stdout_parts.append(line)
        on_output_line(line.rstrip("\r\n"))
    # stdout 关闭后子进程通常已退出；流式模式下不设读超时（与原先 run 的 timeout 行为一致）
    return_code = proc.wait()

    end = time.perf_counter()
    full_out = "".join(stdout_parts)
    record = RunRecord(
        run_id=_new_run_id(),
        timestamp_utc=_utc_now_iso(),
        action=action,
        command=command,
        cwd=str(cwd),
        config_path=config_path,
        status="success" if return_code == 0 else "failed",
        return_code=int(return_code),
        duration_seconds=float(end - start),
        stdout=full_out[-200_000:],
        stderr="",
        artifacts={} if artifacts is None else dict(artifacts),
    )
    append_history(history_path, record)
    return record


def guess_latest_artifacts(project_root: str | Path) -> dict[str, str]:
    """
    Best-effort helper for the UI to link likely outputs.
    This avoids hard-coding internals while still being useful.
    """
    root = Path(project_root)
    candidates: dict[str, Path] = {
        "processed_sequences": root / "data" / "processed" / "sequences.npz",
        "processed_mlp_features": root / "data" / "processed" / "mlp_features.npz",
        "defended_sequences": root / "data" / "defended" / "defended_sequences.npz",
        "defense_summary": root / "data" / "defended" / "defense_summary.json",
        "defense_report": root / "outputs" / "defense" / "defense_report.json",
        "eval_confusion_matrix_png": root / "outputs" / "figures" / "confusion_matrix.png",
        "eval_classification_report_txt": root / "outputs" / "reports" / "classification_report.txt",
        "eval_metrics_json": root / "outputs" / "reports" / "metrics.json",
        "reports_dir": root / "outputs" / "reports",
        "figures_dir": root / "outputs" / "figures",
        "models_dir": root / "outputs" / "models",
    }
    out: dict[str, str] = {}
    for k, p in candidates.items():
        if p.exists():
            out[k] = str(p)
    return out

