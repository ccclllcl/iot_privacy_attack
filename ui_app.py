from __future__ import annotations

import json
import math
import re
import sys
import time
from pathlib import Path

import streamlit as st
import yaml

from src.ui_history import guess_latest_artifacts, load_history, run_and_record


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"
DEFAULT_HISTORY = PROJECT_ROOT / "outputs" / "ui" / "run_history.jsonl"
TMP_CONFIG_DIR = PROJECT_ROOT / "outputs" / "ui" / "tmp_configs"


def _py_exe() -> str:
    # Use current interpreter (works with `py -3 -m streamlit ...`)
    return sys.executable


def _script_path(script_name: str) -> Path:
    return PROJECT_ROOT / script_name


def _list_model_files() -> list[Path]:
    models_dir = PROJECT_ROOT / "outputs" / "models"
    if not models_dir.exists():
        return []
    files = sorted(models_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files


def _prepare_config_with_seed(config_path: Path, seed: int | None) -> Path:
    """
    If seed is provided, write a temporary YAML config that only overrides
    `experiment.random_seed`, and return its path. Otherwise return original config.
    """
    if seed is None:
        return config_path

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("配置文件顶层必须是字典结构")
    exp = raw.get("experiment")
    if not isinstance(exp, dict):
        exp = {}
        raw["experiment"] = exp
    exp["random_seed"] = int(seed)

    TMP_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = TMP_CONFIG_DIR / f"{config_path.stem}.seed{int(seed)}.yaml"
    out_path.write_text(yaml.safe_dump(raw, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return out_path


def _safe_read_text(path: Path, max_chars: int = 120_000) -> str:
    try:
        txt = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    if len(txt) > max_chars:
        return txt[:max_chars] + "\n\n...(内容过长，已截断)...\n"
    return txt


def _render_evaluate_outputs() -> None:
    """
    `run_evaluate.py` outputs (by design):
    - outputs/figures/confusion_matrix.png
    - outputs/reports/classification_report.txt
    - outputs/reports/metrics.json
    """
    fig_cm = PROJECT_ROOT / "outputs" / "figures" / "confusion_matrix.png"
    rep_txt = PROJECT_ROOT / "outputs" / "reports" / "classification_report.txt"
    metrics_json = PROJECT_ROOT / "outputs" / "reports" / "metrics.json"

    st.markdown("### 评估结果展示")

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("**混淆矩阵（Confusion Matrix）**")
        if fig_cm.exists():
            st.image(str(fig_cm), use_container_width=True)
            st.caption(str(fig_cm.relative_to(PROJECT_ROOT)))
        else:
            st.info("未找到混淆矩阵图片：请确认评估已成功运行。")

    with c2:
        st.markdown("**关键指标（metrics.json）**")
        if metrics_json.exists():
            try:
                st.json(json.loads(metrics_json.read_text(encoding="utf-8")))
            except Exception:
                st.code(_safe_read_text(metrics_json), language="json")
            st.caption(str(metrics_json.relative_to(PROJECT_ROOT)))
        else:
            st.info("未找到 metrics.json：请确认评估已成功运行。")

    st.markdown("**分类报告（classification_report.txt）**")
    if rep_txt.exists():
        st.code(_safe_read_text(rep_txt), language="text")
        st.caption(str(rep_txt.relative_to(PROJECT_ROOT)))
    else:
        st.info("未找到分类报告：请确认评估已成功运行。")


def _phase_label(action: str) -> str:
    return {
        "preprocess": "数据预处理",
        "train": "模型训练",
        "evaluate": "结果评估",
        "defense": "防御流水线",
        "defense_eval": "防御评估",
        "compare": "参数扫描对比",
    }.get(action, "运行任务")


def _read_train_num_epochs(config_path: Path, default: int = 80) -> int:
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        tr = raw.get("train") if isinstance(raw, dict) else None
        if isinstance(tr, dict) and "num_epochs" in tr:
            return max(1, int(tr["num_epochs"]))
    except Exception:
        pass
    return default


def _asymptotic_progress(start: float, cap: float = 0.88) -> float:
    t = time.perf_counter() - start
    return min(cap, 0.05 + (cap - 0.05) * (1.0 - math.exp(-t / 42.0)))


def _make_progress_callback(
    action: str,
    progress_bar,
    *,
    train_max_epochs: int | None = None,
):
    state = {"start": time.perf_counter(), "lines": 0, "last_frac": 0.02}
    epoch_re = re.compile(r"Epoch\s+(\d+)", re.IGNORECASE)
    tqdm_re = re.compile(r"(\d+)\s*/\s*(\d+)\s*\[")

    def on_line(line: str) -> None:
        state["lines"] += 1
        frac: float
        if action == "train":
            frac = state["last_frac"]
            m = tqdm_re.search(line)
            if m:
                cur, tot = int(m.group(1)), int(m.group(2))
                if tot > 0:
                    frac = 0.05 + 0.90 * min(1.0, cur / float(tot))
            else:
                m = epoch_re.search(line)
                if m and train_max_epochs:
                    ep = int(m.group(1))
                    frac = 0.05 + 0.90 * min(1.0, ep / float(train_max_epochs))
                else:
                    frac = max(state["last_frac"], _asymptotic_progress(state["start"], 0.90))
        elif action == "preprocess":
            if "预处理完成" in line:
                frac = 0.92
            else:
                frac = max(state["last_frac"], _asymptotic_progress(state["start"], 0.85))
        elif action == "evaluate":
            if "评估完成" in line:
                frac = 0.92
            else:
                frac = min(0.88, 0.12 + state["lines"] * 0.08)
        else:
            frac = max(state["last_frac"], _asymptotic_progress(state["start"], 0.88))
        state["last_frac"] = max(state["last_frac"], frac)
        try:
            progress_bar.progress(state["last_frac"], text=_phase_label(action))
        except TypeError:
            progress_bar.progress(state["last_frac"])

    return on_line


def _run_action(
    action: str,
    script_name: str,
    config_path: Path,
    extra_args: list[str],
    *,
    train_max_epochs: int | None = None,
) -> dict:
    cmd = [_py_exe(), str(_script_path(script_name)), "--config", str(config_path), *extra_args]
    artifacts = guess_latest_artifacts(PROJECT_ROOT)

    st.divider()
    st.caption("任务执行进度（根据子进程输出估算；预处理阶段日志较少时进度会随时间缓慢推进）")
    bar = st.progress(0, text=_phase_label(action))
    log_exp = st.expander("实时输出（最近一行）", expanded=True)
    log_slot = log_exp.empty()

    on_line = _make_progress_callback(action, bar, train_max_epochs=train_max_epochs)

    def on_output_line(line: str) -> None:
        if line.strip():
            log_slot.code(line.strip()[-800:], language="text")
        on_line(line)

    rec = run_and_record(
        action=action,
        command=cmd,
        cwd=PROJECT_ROOT,
        history_path=DEFAULT_HISTORY,
        config_path=str(config_path),
        artifacts=artifacts,
        timeout_seconds=None,
        on_output_line=on_output_line,
    )
    try:
        if rec.status == "success":
            bar.progress(1.0, text=f"{_phase_label(action)} · 已完成")
        else:
            bar.progress(1.0, text=f"{_phase_label(action)} · 已结束（失败）")
    except TypeError:
        bar.progress(1.0)

    return {
        "record": rec,
        "artifacts": {**artifacts, **rec.artifacts},
    }


def _render_instructions() -> None:
    st.markdown(
        """
### 这个界面能做什么
- **不用命令行也能跑实验**：预处理、训练、评估、防御、防御评估、参数扫描。
- **侧边栏导航**：说明 / 运行 / 历史。
- **运行历史**：自动记录每次运行的命令、状态、耗时、stdout/stderr（尾部），以及常见输出文件路径（便于定位结果）。

### 典型流程（建议）
- **第 0 步**：（可选）生成模拟数据：运行 `generate_mock_data.py`。
- **第 1 步**：预处理（raw → processed）。
- **第 2 步**：训练攻击模型（LSTM 或 MLP）。
- **第 3 步**：在干净测试集上评估攻击模型。
- **第 4 步**：运行防御（noise / LDP / adaptive LDP），生成 defended 数据。
- **第 5 步**：防御评估（固定攻击者 / 重新训练攻击者）。
- **第 6 步**：参数扫描对比（例如扫 ε 或噪声强度）。

### 备注（Windows）
- 本界面使用 Streamlit 的**同一个 Python 解释器**，通过 `sys.executable` 来调用项目脚本。
- 输出文件仍然写入 `data/` 与 `outputs/`。
"""
    )


def _render_run_page() -> None:
    st.markdown("### 运行实验")

    config_path = Path(
        st.text_input("配置文件", value=str(DEFAULT_CONFIG), help="YAML 配置文件路径（例如 configs/default.yaml）。")
    ).expanduser()
    if not config_path.exists():
        st.error("未找到配置文件。")
        return

    seed_enabled = st.checkbox("覆盖随机种子（本次运行生效）", value=False)
    seed_value: int | None = None
    if seed_enabled:
        seed_value = int(
            st.number_input(
                "随机种子 random_seed",
                min_value=0,
                max_value=2_147_483_647,
                value=42,
                step=1,
                help="会生成临时配置文件，仅覆盖 experiment.random_seed；不会修改原始配置文件。",
            )
        )

    try:
        effective_config_path = _prepare_config_with_seed(config_path, seed_value)
    except Exception as e:
        st.error(f"生成临时配置失败：{e}")
        return

    if effective_config_path != config_path:
        st.caption(f"本次将使用临时配置：{effective_config_path}")

    with st.expander("常用操作", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("预处理", use_container_width=True):
                out = _run_action("preprocess", "run_preprocess.py", effective_config_path, [])
                st.session_state["last_run"] = out
        with c2:
            model = st.selectbox("模型", options=["lstm", "mlp"], index=0)
            if st.button("训练攻击模型", use_container_width=True):
                ne = _read_train_num_epochs(effective_config_path)
                out = _run_action(
                    "train",
                    "run_train.py",
                    effective_config_path,
                    ["--model", model],
                    train_max_epochs=ne,
                )
                st.session_state["last_run"] = out
        with c3:
            model_files = _list_model_files()
            if model_files:
                selected = st.selectbox(
                    "选择模型文件（用于评估）",
                    options=[str(p.relative_to(PROJECT_ROOT)) for p in model_files],
                    index=0,
                    help="默认选择最新生成的 .pt 文件。",
                )
                model_path = selected
            else:
                model_path = st.text_input(
                    "模型文件路径（用于评估）",
                    value="",
                    help="未检测到 outputs/models/*.pt，请先训练，或手动填写模型路径。",
                ).strip()

            if st.button("评估攻击模型", use_container_width=True):
                if not model_path:
                    st.error("评估需要提供 --model_path：请先训练生成模型，或填写模型文件路径。")
                else:
                    out = _run_action(
                        "evaluate",
                        "run_evaluate.py",
                        effective_config_path,
                        ["--model_path", model_path],
                    )
                    st.session_state["last_run"] = out

    with st.expander("防御与对比", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("运行防御（生成 defended 数据）", use_container_width=True):
                out = _run_action("defense", "run_defense.py", effective_config_path, [])
                st.session_state["last_run"] = out
        with c2:
            mode = st.selectbox("防御评估模式", options=["fixed_attacker", "retrain_attacker"], index=0)
            model_path = st.text_input(
                "基线模型路径（fixed_attacker 必填）",
                value="",
                help="例如：outputs/models/lstm_best.pt",
            )
            args: list[str] = ["--mode", mode]
            if mode == "fixed_attacker":
                if model_path.strip():
                    args += ["--model_path", model_path.strip()]
            if st.button("运行防御评估", use_container_width=True):
                out = _run_action("defense_eval", "run_defense_eval.py", effective_config_path, args)
                st.session_state["last_run"] = out
        with c3:
            if st.button("运行参数扫描对比", use_container_width=True):
                out = _run_action("compare", "run_compare.py", effective_config_path, [])
                st.session_state["last_run"] = out

    last = st.session_state.get("last_run")
    if last is not None:
        rec = last["record"]
        st.markdown("### 最近一次运行结果")
        st.write(
            {
                "action": rec.action,
                "status": rec.status,
                "return_code": rec.return_code,
                "duration_seconds": round(rec.duration_seconds, 3),
                "timestamp_utc": rec.timestamp_utc,
            }
        )
        st.markdown("**可能的输出文件（自动猜测/便捷跳转）**")
        st.json(last["artifacts"])
        with st.expander("stdout（尾部）", expanded=False):
            st.code(rec.stdout or "", language="text")
        with st.expander("stderr（尾部）", expanded=False):
            st.code(rec.stderr or "", language="text")

        # Inline visualization for evaluation outputs
        if rec.action == "evaluate" and rec.status == "success":
            _render_evaluate_outputs()


def _render_history_page() -> None:
    st.markdown("### 运行历史")
    st.caption(f"历史文件：{DEFAULT_HISTORY}")

    records = load_history(DEFAULT_HISTORY, limit=300)
    if not records:
        st.info("暂无历史记录。请先在「运行」页面执行一次任务。")
        return

    # quick filters
    actions = sorted({r.get("action", "") for r in records if r.get("action")})
    action_filter = st.multiselect("按动作筛选", options=actions, default=actions)
    status_filter = st.multiselect("按状态筛选", options=["success", "failed"], default=["success", "failed"])

    filtered = [
        r
        for r in records
        if (r.get("action") in action_filter) and (r.get("status") in status_filter)
    ]

    st.write(f"显示 {len(filtered)} / {len(records)} 条记录（最新在前）。")
    for r in filtered[:50]:
        title = f"[{r.get('status')}] {r.get('timestamp_utc')}  —  {r.get('action')}  ({r.get('duration_seconds', 0):.2f}s)"
        with st.expander(title, expanded=False):
            st.write(
                {
                    "run_id": r.get("run_id"),
                    "cwd": r.get("cwd"),
                    "config_path": r.get("config_path"),
                    "return_code": r.get("return_code"),
                    "command": r.get("command"),
                }
            )
            artifacts = r.get("artifacts") or {}
            if artifacts:
                st.markdown("**输出文件**")
                st.json(artifacts)
            st.markdown("**stdout（尾部）**")
            st.code((r.get("stdout") or "")[:200_000], language="text")
            st.markdown("**stderr（尾部）**")
            st.code((r.get("stderr") or "")[:200_000], language="text")

    with st.expander("原始 JSON（最近 20 条）", expanded=False):
        st.code(json.dumps(records[:20], ensure_ascii=False, indent=2), language="json")


def main() -> None:
    st.set_page_config(
        page_title="物联网隐私攻击与防御实验平台",
        page_icon=None,
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.sidebar.markdown("### 实验平台")
    page = st.sidebar.radio("导航", options=["说明", "运行", "历史"], index=1)
    st.sidebar.markdown("---")
    st.sidebar.caption(f"项目路径：{PROJECT_ROOT}")

    if page == "说明":
        _render_instructions()
    elif page == "运行":
        _render_run_page()
    else:
        _render_history_page()


if __name__ == "__main__":
    main()

