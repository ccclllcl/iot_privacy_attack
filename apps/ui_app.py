from __future__ import annotations

import json
import math
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import yaml

from src.ui_history import guess_latest_artifacts, load_history, run_and_record


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"
DEFAULT_HISTORY = PROJECT_ROOT / "outputs" / "ui" / "run_history.jsonl"
TMP_CONFIG_DIR = PROJECT_ROOT / "outputs" / "ui" / "tmp_configs"


def _py_exe() -> str:
    # Use current interpreter (works with `py -3 -m streamlit ...`)
    return sys.executable


def _script_path(script_name: str) -> Path:
    return PROJECT_ROOT / script_name


def _models_dir_from_config(config_path: Path) -> Path:
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            paths = raw.get("paths")
            if isinstance(paths, dict) and paths.get("models_dir"):
                p = Path(str(paths["models_dir"]))
                return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()
    except Exception:
        pass
    return (PROJECT_ROOT / "outputs" / "models").resolve()


def _list_model_files(config_path: Path) -> list[Path]:
    models_dir = _models_dir_from_config(config_path)
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


def _prepare_config_with_overrides(config_path: Path, overrides: dict) -> Path:
    """
    Write a temporary YAML config that applies shallow overrides to selected nested keys.
    Used by the Web UI to change defense parameters without editing the original config.

    overrides example:
      {"defense.method": "ldp", "defense.epsilon": 1.0}
      {"adaptive_ldp.epsilon_min": 0.4, "adaptive_ldp.epsilon_max": 4.0}
    """
    if not overrides:
        return config_path

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("配置文件顶层必须是字典结构")

    def _set_path(d: dict, dotted: str, value) -> None:
        parts = dotted.split(".")
        cur = d
        for p in parts[:-1]:
            nxt = cur.get(p)
            if not isinstance(nxt, dict):
                nxt = {}
                cur[p] = nxt
            cur = nxt
        cur[parts[-1]] = value

    for k, v in overrides.items():
        _set_path(raw, str(k), v)

    TMP_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "_".join([re.sub(r"[^a-zA-Z0-9]+", "", str(k))[:20] for k in sorted(overrides.keys())])[:80]
    out_path = TMP_CONFIG_DIR / f"{config_path.stem}.override.{suffix}.yaml"
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


def _processed_dir_from_config(config_path: Path) -> Path:
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            paths = raw.get("paths")
            if isinstance(paths, dict) and paths.get("processed_dir"):
                p = Path(str(paths["processed_dir"]))
                return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()
    except Exception:
        pass
    return (PROJECT_ROOT / "data" / "processed").resolve()


def _render_current_processed_meta(config_path: Path) -> None:
    processed_dir = _processed_dir_from_config(config_path)
    meta_path = processed_dir / "meta.json"
    st.markdown("### 当前使用的数据（data/processed）")
    if not meta_path.exists():
        st.info("未找到 data/processed/meta.json。请先导入真实数据或完成预处理。")
        return
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        st.code(_safe_read_text(meta_path), language="json")
        return

    # Keep it compact and high-signal.
    info = {
        "dataset": meta.get("dataset") or meta.get("label_source") or "unknown",
        "seq_len": meta.get("seq_len"),
        "freq": meta.get("freq"),
        "feature_dim": len(meta.get("feature_names") or []),
        "n_train": meta.get("n_train"),
        "n_val": meta.get("n_val"),
        "n_test": meta.get("n_test"),
        "saved_at": meta.get("saved_at"),
    }
    st.json(info)
    try:
        st.caption(str(meta_path.relative_to(PROJECT_ROOT)))
    except Exception:
        st.caption(str(meta_path))

    ds = str(meta.get("dataset") or "")
    if ds.strip().lower().startswith("uci har"):
        st.info("检测到 UCI HAR 已导入：通常不需要再运行「预处理」。请直接训练/评估。")
    else:
        st.info("当前 processed 看起来不是 UCI HAR（可能是 Smart* / mock CSV 预处理结果）。")


def _get_processed_dataset_tag() -> str:
    meta_path = PROJECT_ROOT / "data" / "processed" / "meta.json"
    if not meta_path.exists():
        return "none"
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return "unknown"
    ds = str(meta.get("dataset") or "").strip()
    if ds.lower().startswith("uci har"):
        return "uci_har"
    return "other"


def _get_current_feature_names(config_path: Path) -> list[str]:
    processed_dir = _processed_dir_from_config(config_path)
    meta_path = processed_dir / "meta.json"
    if not meta_path.exists():
        return []
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    names = meta.get("feature_names")
    if not isinstance(names, list):
        return []
    return [str(x) for x in names if str(x)]


def _danger_confirm_checkbox(key: str, label: str) -> bool:
    return bool(
        st.checkbox(
            label,
            value=False,
            key=key,
        )
    )


def _wizard_state() -> dict:
    # Central place to keep wizard-related session state.
    ss = st.session_state
    if "wizard_source" not in ss:
        ss["wizard_source"] = "uci_har"
    if "wizard_model" not in ss:
        ss["wizard_model"] = "lstm"
    if "wizard_last_model_path" not in ss:
        ss["wizard_last_model_path"] = ""
    if "wizard_autorun" not in ss:
        ss["wizard_autorun"] = False
    return ss


def _latest_model_relpath(config_path: Path) -> str:
    files = _list_model_files(config_path)
    if not files:
        return ""
    try:
        return str(files[0].relative_to(PROJECT_ROOT))
    except Exception:
        return str(files[0])


def _run_and_store_last_model(
    action: str,
    script_name: str,
    config_path: Path,
    extra_args: list[str],
    *,
    train_max_epochs: int | None = None,
) -> dict:
    out = _run_action(
        action,
        script_name,
        config_path,
        extra_args,
        train_max_epochs=train_max_epochs,
    )
    st.session_state["last_run"] = out
    # After training, attempt to remember the latest model path for evaluation.
    if action == "train" and out.get("record") and out["record"].status == "success":
        st.session_state["wizard_last_model_path"] = _latest_model_relpath(config_path)
    return out


def _render_wizard(config_path: Path) -> None:
    ss = _wizard_state()
    # Note: wizard uses per-dataset processed dirs via temporary config overrides.

    st.markdown("### 一键流程 / 向导")
    st.caption("这个向导会根据数据来源，按正确顺序引导你跑通：数据准备 → 训练 → 评估。")

    c0, c1, c2 = st.columns([1.2, 1, 1])
    with c0:
        dataset_options = ["uci_har", "casas_hh101", "casas_rw105", "casas_tm004", "kasteren", "csv"]
        labels = {
            "uci_har": "UCI HAR（可穿戴行为，已集成）",
            "casas_hh101": "CASAS（hh101，智能家居传感器）",
            "casas_rw105": "CASAS（rw105，智能家居传感器）",
            "casas_tm004": "CASAS（tm004，智能家居传感器）",
            "kasteren": "Kasteren（智能家居传感器）",
            "csv": "CSV（你自己的长表 Smart* / 自采）",
        }
        cur = ss.get("wizard_source")
        idx = dataset_options.index(cur) if cur in dataset_options else 0
        ss["wizard_source"] = st.radio(
            "选择数据集",
            options=dataset_options,
            format_func=lambda x: labels.get(x, x),
            index=idx,
            horizontal=False,
        )
    with c1:
        ss["wizard_model"] = st.selectbox(
            "选择模型结构",
            options=["lstm", "mlp"],
            index=0 if ss.get("wizard_model") == "lstm" else 1,
        )
    with c2:
        ss["wizard_autorun"] = st.checkbox("一键跑到评估（自动串行）", value=bool(ss.get("wizard_autorun")))

    st.divider()

    ds = ss["wizard_source"]
    # Per-dataset isolated paths to avoid overwriting/caching across datasets.
    ds_tag = ds
    path_overrides = {
        "paths.processed_dir": f"data/processed/{ds_tag}",
        "paths.defended_dir": f"data/defended/{ds_tag}",
        "paths.models_dir": f"outputs/models/{ds_tag}",
        "paths.figures_dir": f"outputs/figures/{ds_tag}",
        "paths.reports_dir": f"outputs/reports/{ds_tag}",
        "paths.defense_dir": f"outputs/defense/{ds_tag}",
    }
    ds_cfg = _prepare_config_with_overrides(config_path, path_overrides)
    st.session_state["active_dataset_config_path"] = str(ds_cfg)

    st.markdown("**Step 1：准备数据**（会写入/覆盖该数据集专属的 processed 目录）")
    confirm = _danger_confirm_checkbox(
        "wiz_confirm_prepare",
        "我确认要写入/覆盖本数据集的 `data/processed/<dataset>/`",
    )
    cA, cB, cC = st.columns(3)
    with cA:
        do_prepare = st.button("Step 1：准备数据", use_container_width=True, disabled=not confirm)
    with cB:
        do_train = st.button("Step 2：训练", use_container_width=True)
    with cC:
        do_eval = st.button("Step 3：评估", use_container_width=True)

    prep_action = "preprocess"
    prep_script = "experiments/core/run_preprocess.py"
    prep_args: list[str] = []
    if ds == "uci_har":
        prep_action, prep_script, prep_args = "import_real", "experiments/real_public/run_import_uci_har.py", ["--auto-download"]
    elif ds == "kasteren":
        prep_action, prep_script, prep_args = "import_kasteren", "experiments/real_public/run_import_kasteren.py", ["--auto-download"]
    elif ds.startswith("casas_"):
        home = ds.split("_", 1)[1]
        prep_action, prep_script, prep_args = "import_casas", "experiments/real_public/run_import_casas.py", ["--home", home, "--auto-download"]

    if do_prepare:
        _run_and_store_last_model(prep_action, prep_script, ds_cfg, prep_args)
        if ss["wizard_autorun"]:
            ne = _read_train_num_epochs(ds_cfg)
            _run_and_store_last_model(
                "train",
                "experiments/core/run_train.py",
                ds_cfg,
                ["--model", ss["wizard_model"]],
                train_max_epochs=ne,
            )
            mp = _latest_model_relpath(ds_cfg)
            if mp:
                _run_and_store_last_model("evaluate", "experiments/core/run_evaluate.py", ds_cfg, ["--model_path", mp])
    elif do_train:
        ne = _read_train_num_epochs(ds_cfg)
        _run_and_store_last_model(
            "train",
            "experiments/core/run_train.py",
            ds_cfg,
            ["--model", ss["wizard_model"]],
            train_max_epochs=ne,
        )
        if ss["wizard_autorun"]:
            mp = _latest_model_relpath(ds_cfg)
            if mp:
                _run_and_store_last_model("evaluate", "experiments/core/run_evaluate.py", ds_cfg, ["--model_path", mp])
    elif do_eval:
        mp = _latest_model_relpath(ds_cfg)
        if not mp:
            st.error("未检测到可用模型文件。请先完成训练。")
        else:
            _run_and_store_last_model("evaluate", "experiments/core/run_evaluate.py", ds_cfg, ["--model_path", mp])


def _render_advanced_run_controls(effective_config_path: Path) -> None:
    processed_tag = _get_processed_dataset_tag()

    st.markdown("### 数据入口（高级）")
    st.caption(
        "提示：`导入真实数据` 和 `预处理` 都会写入/覆盖 `data/processed/`。"
        "如果你刚导入 UCI HAR，请不要再运行预处理。"
    )

    with st.expander("数据准备", expanded=True):
        left, right = st.columns([1, 1])
        with left:
            st.markdown("**路线 A：真实数据（UCI HAR）**")
            st.caption("推荐：会直接生成统一格式的 `data/processed/*.npz`，无需再跑预处理。")
            confirm_import = _danger_confirm_checkbox(
                "confirm_import_real",
                "我确认要写入/覆盖 `data/processed/`（导入 UCI HAR）",
            )
            if st.button(
                "导入真实数据（UCI HAR）",
                use_container_width=True,
                disabled=not confirm_import,
            ):
                out = _run_action(
                    "import_real",
                    "experiments/real_public/run_import_uci_har.py",
                    effective_config_path,
                    ["--auto-download"],
                )
                st.session_state["last_run"] = out

        with right:
            st.markdown("**路线 B：CSV（Smart* / 你自己的长表）→ 预处理**")
            st.caption("仅当你确实要用 `paths.raw_csv` 指向的 CSV 时，才需要预处理。")
            if processed_tag == "uci_har":
                st.warning(
                    "当前检测到 `data/processed` 是 UCI HAR。运行「预处理」会覆盖掉 UCI HAR 的 processed，"
                    "导致后续训练/评估回到 CSV/模拟数据结果。"
                )
            confirm_pre = _danger_confirm_checkbox(
                "confirm_preprocess_overwrite",
                "我确认要覆盖 `data/processed/`（运行预处理）",
            )
            disable_preprocess = (processed_tag == "uci_har") and (not confirm_pre)
            if st.button(
                "预处理（raw CSV → processed）",
                use_container_width=True,
                disabled=disable_preprocess,
            ):
                if processed_tag == "uci_har" and not confirm_pre:
                    st.error("为了避免误操作：请先勾选确认，再运行预处理。")
                else:
                    out = _run_action("preprocess", "experiments/core/run_preprocess.py", effective_config_path, [])
                    st.session_state["last_run"] = out

    st.markdown("### 训练与评估（高级）")
    with st.expander("训练与评估（在当前 processed 上运行）", expanded=True):
        c2, c3 = st.columns([1, 1])
        with c2:
            model = st.selectbox("模型", options=["lstm", "mlp"], index=0, key="advanced_model_select")
            if st.button("训练攻击模型", use_container_width=True, key="advanced_train_btn"):
                ne = _read_train_num_epochs(effective_config_path)
                out = _run_action(
                    "train",
                    "experiments/core/run_train.py",
                    effective_config_path,
                    ["--model", model],
                    train_max_epochs=ne,
                )
                st.session_state["last_run"] = out
        with c3:
            model_files = _list_model_files(effective_config_path)
            if model_files:
                selected = st.selectbox(
                    "选择模型文件（用于评估）",
                    options=[str(p.relative_to(PROJECT_ROOT)) for p in model_files],
                    index=0,
                    help="默认选择最新生成的 .pt 文件。",
                    key="advanced_eval_model_select",
                )
                model_path = selected
            else:
                model_path = st.text_input(
                    "模型文件路径（用于评估）",
                    value="",
                    help="未检测到 outputs/models/*.pt，请先训练，或手动填写模型路径。",
                    key="advanced_eval_model_input",
                ).strip()

            if st.button("评估攻击模型", use_container_width=True, key="advanced_eval_btn"):
                if not model_path:
                    st.error("评估需要提供 --model_path：请先训练生成模型，或填写模型文件路径。")
                else:
                    out = _run_action(
                        "evaluate",
                        "experiments/core/run_evaluate.py",
                        effective_config_path,
                        ["--model_path", model_path],
                    )
                    st.session_state["last_run"] = out


def _render_evaluate_outputs(config_path: Path) -> None:
    """
    `experiments/core/run_evaluate.py` outputs (by design):
    - outputs/figures/confusion_matrix.png
    - outputs/reports/classification_report.txt
    - outputs/reports/metrics.json
    """
    figures_dir = ensure_dir(_processed_dir_from_config(config_path).parent.parent / "outputs" / "figures")  # fallback
    reports_dir = ensure_dir(_processed_dir_from_config(config_path).parent.parent / "outputs" / "reports")  # fallback
    # Prefer configured dirs (per-dataset isolation)
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            paths = raw.get("paths")
            if isinstance(paths, dict):
                if paths.get("figures_dir"):
                    p = Path(str(paths["figures_dir"]))
                    figures_dir = p if p.is_absolute() else (PROJECT_ROOT / p).resolve()
                if paths.get("reports_dir"):
                    p = Path(str(paths["reports_dir"]))
                    reports_dir = p if p.is_absolute() else (PROJECT_ROOT / p).resolve()
    except Exception:
        pass

    fig_cm = figures_dir / "confusion_matrix.png"
    rep_txt = reports_dir / "classification_report.txt"
    metrics_json = reports_dir / "metrics.json"

    st.markdown("### 评估结果展示")

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("**混淆矩阵（Confusion Matrix）**")
        if fig_cm.exists():
            # Use image bytes to avoid browser caching the old PNG
            # when the filename stays constant across runs.
            st.image(fig_cm.read_bytes(), use_container_width=True)
            mtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(fig_cm.stat().st_mtime))
            try:
                rel = str(fig_cm.relative_to(PROJECT_ROOT))
            except Exception:
                rel = str(fig_cm)
            st.caption(f"{rel} · updated_at={mtime}")
        else:
            st.info("未找到混淆矩阵图片：请确认评估已成功运行。")

    with c2:
        st.markdown("**关键指标（metrics.json）**")
        if metrics_json.exists():
            try:
                st.json(json.loads(metrics_json.read_text(encoding="utf-8")))
            except Exception:
                st.code(_safe_read_text(metrics_json), language="json")
            try:
                st.caption(str(metrics_json.relative_to(PROJECT_ROOT)))
            except Exception:
                st.caption(str(metrics_json))
        else:
            st.info("未找到 metrics.json：请确认评估已成功运行。")

    st.markdown("**分类报告（classification_report.txt）**")
    if rep_txt.exists():
        st.code(_safe_read_text(rep_txt), language="text")
        try:
            st.caption(str(rep_txt.relative_to(PROJECT_ROOT)))
        except Exception:
            st.caption(str(rep_txt))
    else:
        st.info("未找到分类报告：请确认评估已成功运行。")


def _render_defense_eval_outputs() -> None:
    """
    `experiments/core/run_defense_eval.py` writes into outputs/defense/:
    - fixed_attacker: confusion_matrix_baseline.png, confusion_matrix_defended.png, accuracy_comparison.png
    - retrain_attacker: confusion_matrix_defended_retrain.png
    - defense_report.json / defense_report.txt
    """
    out_dir = PROJECT_ROOT / "outputs" / "defense"
    st.markdown("### 防御评估结果展示")

    fig_acc = out_dir / "accuracy_comparison.png"
    fig_base = out_dir / "confusion_matrix_baseline.png"
    fig_def = out_dir / "confusion_matrix_defended.png"
    fig_retrain = out_dir / "confusion_matrix_defended_retrain.png"

    def _show_img(p: Path, title: str) -> None:
        st.markdown(f"**{title}**")
        if p.exists():
            st.image(p.read_bytes(), use_container_width=True)
            mtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(p.stat().st_mtime))
            st.caption(f"{p.relative_to(PROJECT_ROOT)} · updated_at={mtime}")
        else:
            st.info(f"未找到图片：{p.relative_to(PROJECT_ROOT)}")

    cols = st.columns([1, 1])
    with cols[0]:
        if fig_base.exists():
            _show_img(fig_base, "混淆矩阵（干净测试集 / 基线）")
        elif fig_retrain.exists():
            _show_img(fig_retrain, "混淆矩阵（防御后重训攻击者）")
        else:
            _show_img(fig_def, "混淆矩阵（防御后）")
    with cols[1]:
        if fig_def.exists():
            _show_img(fig_def, "混淆矩阵（防御后 / fixed_attacker）")
        elif fig_acc.exists():
            _show_img(fig_acc, "准确率对比（fixed_attacker）")
        else:
            _show_img(fig_retrain, "混淆矩阵（防御后重训攻击者）")

    if fig_acc.exists():
        _show_img(fig_acc, "准确率对比（fixed_attacker）")

    rep_json = out_dir / "defense_report.json"
    rep_txt = out_dir / "defense_report.txt"

    st.markdown("**报告（defense_report.json）**")
    if rep_json.exists():
        try:
            st.json(json.loads(rep_json.read_text(encoding="utf-8")))
        except Exception:
            st.code(_safe_read_text(rep_json), language="json")
        st.caption(str(rep_json.relative_to(PROJECT_ROOT)))
    else:
        st.info("未找到 defense_report.json：请确认防御评估已成功运行。")

    st.markdown("**报告文本（defense_report.txt）**")
    if rep_txt.exists():
        st.code(_safe_read_text(rep_txt), language="text")
        st.caption(str(rep_txt.relative_to(PROJECT_ROOT)))
    else:
        st.info("未找到 defense_report.txt：请确认防御评估已成功运行。")


def _render_compare_outputs() -> None:
    """
    `experiments/core/run_compare.py` outputs into outputs/defense/comparisons/:
    - comparison_results.csv
    - LDP: epsilon_vs_accuracy.png, epsilon_vs_distortion.png
    - Noise: distortion_vs_noise.png, noise_scale_vs_accuracy.png
    """
    comp_dir = PROJECT_ROOT / "outputs" / "defense" / "comparisons"
    st.markdown("### 参数扫描对比结果展示")

    if not comp_dir.exists():
        st.info("未找到 outputs/defense/comparisons/：请先运行一次参数扫描对比。")
        return

    csv_path = comp_dir / "comparison_results.csv"
    detected_method: str | None = None
    if csv_path.exists():
        st.markdown("**汇总表（comparison_results.csv）**")
        csv_text = _safe_read_text(csv_path, max_chars=80_000)
        st.code(csv_text, language="text")
        st.caption(str(csv_path.relative_to(PROJECT_ROOT)))
        # Detect which method was actually scanned to avoid showing "missing" plots for the other method.
        # CSV has a "method" column with values "ldp" or "noise".
        m = re.search(r"^\s*(ldp|noise)\s*$", csv_text, flags=re.IGNORECASE | re.MULTILINE)
        if m:
            detected_method = m.group(1).lower()
    else:
        st.info("未找到 comparison_results.csv：请确认参数扫描成功完成。")

    def _show_img(p: Path, title: str) -> None:
        st.markdown(f"**{title}**")
        if p.exists():
            st.image(p.read_bytes(), use_container_width=True)
            mtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(p.stat().st_mtime))
            st.caption(f"{p.relative_to(PROJECT_ROOT)} · updated_at={mtime}")
        else:
            st.info(f"未找到图片：{p.relative_to(PROJECT_ROOT)}")

    c1, c2 = st.columns([1, 1])
    with c1:
        if detected_method in (None, "ldp"):
            _show_img(comp_dir / "epsilon_vs_accuracy.png", "LDP：epsilon vs defended accuracy（fixed attacker）")
        if detected_method in (None, "noise"):
            _show_img(comp_dir / "noise_scale_vs_accuracy.png", "Noise：noise_scale vs defended accuracy（fixed attacker）")
    with c2:
        if detected_method in (None, "ldp"):
            _show_img(comp_dir / "epsilon_vs_distortion.png", "LDP：epsilon vs distortion（MSE）")
        if detected_method in (None, "noise"):
            _show_img(comp_dir / "distortion_vs_noise.png", "Noise：distortion vs accuracy（双轴）")


def _phase_label(action: str) -> str:
    return {
        "import_real": "导入真实数据",
        "import_casas": "导入 CASAS",
        "import_kasteren": "导入 Kasteren",
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
- **不用命令行也能跑实验**：导入真实数据、预处理、训练、评估、防御、防御评估、参数扫描。
- **侧边栏导航**：说明 / 运行 / 历史。
- **运行历史**：自动记录每次运行的命令、状态、耗时、stdout/stderr（尾部），以及常见输出文件路径（便于定位结果）。

### 典型流程（建议）
- **第 0 步**：导入真实数据（推荐 UCI HAR）：运行 `experiments/real_public/run_import_uci_har.py --auto-download`。
- **第 0.5 步**：（可选）生成模拟数据：运行 `experiments/core/generate_mock_data.py`。
- **第 1 步**：预处理（raw → processed；仅当你使用智能家居长表 CSV 时需要）。
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

    active_cfg = Path(st.session_state.get("active_dataset_config_path") or str(effective_config_path))
    _render_current_processed_meta(active_cfg)

    with st.expander("一键清空（避免数据污染）", expanded=False):
        st.caption("会删除预处理数据、扰动数据、模型、图表、报告、防御输出与临时配置。不可恢复，请谨慎。")
        confirm_wipe = _danger_confirm_checkbox("confirm_wipe_all", "我确认要清空（不可恢复）")
        if st.button("一键格式化 / 清空所有缓存", use_container_width=True, disabled=not confirm_wipe):
            import shutil

            targets = [
                PROJECT_ROOT / "data" / "processed",
                PROJECT_ROOT / "data" / "defended",
                PROJECT_ROOT / "outputs" / "models",
                PROJECT_ROOT / "outputs" / "figures",
                PROJECT_ROOT / "outputs" / "reports",
                PROJECT_ROOT / "outputs" / "defense",
                PROJECT_ROOT / "outputs" / "ui" / "tmp_configs",
            ]
            removed: list[str] = []
            for p in targets:
                try:
                    if p.exists():
                        shutil.rmtree(p, ignore_errors=True)
                        removed.append(str(p.relative_to(PROJECT_ROOT)))
                except Exception:
                    continue
            st.session_state.pop("wizard_last_model_path", None)
            st.success("已清空：" + (", ".join(removed) if removed else "（无可清空内容）"))

    _render_wizard(effective_config_path)
    active_cfg = Path(st.session_state.get("active_dataset_config_path") or str(effective_config_path))

    with st.expander("防御与对比", expanded=True):
        st.markdown("### 防御设置（3 种手段）")
        m1, m2, m3 = st.columns([1, 1, 1])
        with m1:
            defense_method = st.selectbox(
                "防御手段",
                options=["noise", "ldp", "adaptive_ldp"],
                index=2,
                help="noise：简单加噪；ldp：固定 ε 的本地差分隐私；adaptive_ldp：按窗口风险自适应分配 ε。",
                key="defense_method_select",
            )
        with m2:
            apply_to = st.selectbox(
                "作用范围",
                options=["all", "selected"],
                index=0,
                help="all：全特征；selected：仅对 selected_features 扰动（见配置文件）。",
                key="defense_apply_to_select",
            )
        with m3:
            skip_pipeline = st.checkbox(
                "防御评估时跳过重复扰动（--skip-pipeline）",
                value=False,
                help="如果你没有改动防御参数/数据，可勾选以节省时间。",
                key="defense_skip_pipeline_chk",
            )
            if skip_pipeline:
                st.warning("已启用跳过扰动：请确保 `data/defended/` 与当前防御参数一致，否则评估可能用到旧数据。")

        overrides: dict = {"defense.method": defense_method, "defense.apply_to": apply_to}
        if apply_to == "selected":
            feat_names = _get_current_feature_names(active_cfg)
            if not feat_names:
                st.warning("未检测到当前数据的 feature_names（缺少 data/processed/meta.json）。建议先完成数据准备。")
            else:
                picked = st.multiselect(
                    "选择要扰动的特征列（selected_features）",
                    options=feat_names,
                    default=feat_names[: min(3, len(feat_names))],
                    help="这里的列表来自当前 data/processed/meta.json，避免 selected_features 与数据集不匹配导致流水线失败。",
                    key="defense_selected_features_ms",
                )
                if not picked:
                    st.warning("你选择了 apply_to=selected，但未选择任何特征列。将会导致防御流水线失败。")
                overrides["defense.selected_features"] = picked

        if defense_method == "noise":
            n1, n2 = st.columns([1, 1])
            with n1:
                noise_type = st.selectbox(
                    "噪声类型",
                    options=["laplace", "gaussian"],
                    index=0,
                    key="defense_noise_type",
                )
            with n2:
                noise_scale = float(
                    st.number_input(
                        "噪声强度 noise_scale",
                        min_value=0.0,
                        value=0.5,
                        step=0.1,
                        key="defense_noise_scale",
                    )
                )
            overrides.update({"defense.noise_type": noise_type, "defense.noise_scale": noise_scale})
        elif defense_method == "ldp":
            l1, l2 = st.columns([1, 1])
            with l1:
                epsilon = float(
                    st.number_input(
                        "ε（epsilon）",
                        min_value=0.01,
                        value=1.0,
                        step=0.1,
                        help="ε 越小隐私越强、噪声越大。",
                        key="defense_ldp_epsilon",
                    )
                )
            with l2:
                sens = float(
                    st.number_input(
                        "敏感度代理 ldp_sensitivity",
                        min_value=0.01,
                        value=1.0,
                        step=0.1,
                        key="defense_ldp_sens",
                    )
                )
            overrides.update({"defense.epsilon": epsilon, "defense.ldp_sensitivity": sens})
        else:
            a1, a2, a3, a4 = st.columns([1, 1, 1, 1])
            with a1:
                eps_min = float(
                    st.number_input("epsilon_min", min_value=0.01, value=0.4, step=0.1, key="adaptive_eps_min")
                )
            with a2:
                eps_max = float(
                    st.number_input("epsilon_max", min_value=0.01, value=4.0, step=0.2, key="adaptive_eps_max")
                )
            with a3:
                w_s = float(
                    st.number_input(
                        "weight_sensitivity",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.5,
                        step=0.1,
                        key="adaptive_w_s",
                    )
                )
            with a4:
                w_t = float(
                    st.number_input(
                        "weight_traffic",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.5,
                        step=0.1,
                        key="adaptive_w_t",
                    )
                )
            overrides.update(
                {
                    "adaptive_ldp.epsilon_min": eps_min,
                    "adaptive_ldp.epsilon_max": eps_max,
                    "adaptive_ldp.weight_sensitivity": w_s,
                    "adaptive_ldp.weight_traffic": w_t,
                }
            )

        effective_defense_config = _prepare_config_with_overrides(active_cfg, overrides)
        if effective_defense_config != effective_config_path:
            st.caption(f"本次防御相关操作将使用临时配置：{effective_defense_config}")

        st.divider()
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**生成防御后数据**")
            confirm_def = _danger_confirm_checkbox(
                "confirm_run_defense",
                "我确认要写入/覆盖 `data/defended/`（运行防御）",
            )
            if st.button(
                "运行防御（生成 defended 数据）",
                use_container_width=True,
                disabled=not confirm_def,
                key="run_defense_btn",
            ):
                out = _run_action("defense", "experiments/core/run_defense.py", effective_defense_config, [])
                st.session_state["last_run"] = out
        with c2:
            st.markdown("**防御评估（2 种对手）**")
            mode = st.selectbox(
                "对手模式",
                options=["fixed_attacker", "retrain_attacker"],
                index=0,
                help="fixed：模型不变；retrain：拿到防御数据后重训再攻击。",
                key="def_eval_mode_select",
            )
            # IMPORTANT: avoid cross-dataset model mix-ups.
            # Always pick the latest model under the *current config's* models_dir.
            default_mp = _latest_model_relpath(effective_defense_config)
            if mode == "fixed_attacker":
                mp = st.text_input(
                    "基线攻击者模型路径（自动填入最新）",
                    value=default_mp,
                    help="fixed_attacker 必填：干净数据上训练得到的 .pt（一般是 outputs/models 下最新）。",
                    key="def_eval_model_path",
                ).strip()
            else:
                mp = ""
                st.caption("retrain_attacker：会在防御后数据上重训攻击者模型，无需提供 model_path。")
            args: list[str] = ["--mode", mode]
            if mode == "fixed_attacker":
                if not mp:
                    st.warning("fixed_attacker 需要模型路径；请先训练或填写 outputs/models/*.pt。")
                else:
                    args += ["--model_path", mp]
            if skip_pipeline:
                args += ["--skip-pipeline"]
            if st.button("运行防御评估", use_container_width=True, key="run_defense_eval_btn"):
                if mode == "fixed_attacker" and not mp:
                    st.error("fixed_attacker 模式必须提供 --model_path。")
                else:
                    out = _run_action("defense_eval", "experiments/core/run_defense_eval.py", effective_defense_config, args)
                    st.session_state["last_run"] = out
        with c3:
            st.markdown("**参数扫描对比**")
            method = st.selectbox(
                "扫描方法",
                options=["ldp", "noise"],
                index=0,
                help="ldp：扫 epsilon；noise：扫噪声强度（列表来自 configs/default.yaml 的 compare.*）。",
                key="compare_method_select",
            )
            model_files = _list_model_files(effective_defense_config)
            if model_files:
                options = [str(p.relative_to(PROJECT_ROOT)) for p in model_files]
                idx = options.index(default_mp) if default_mp in options else 0
                selected = st.selectbox(
                    "选择基线模型（fixed_attacker）",
                    options=options,
                    index=idx,
                    help="参数扫描在固定攻击者模式下运行，需要一个已训练的基线模型。",
                    key="compare_model_select",
                )
                model_path = selected
            else:
                model_path = st.text_input(
                    "基线模型路径（必填）",
                    value=default_mp,
                    help="例如：outputs/models/best_lstm.pt（先训练生成 .pt）。",
                    key="compare_model_input",
                ).strip()
            if st.button("运行参数扫描对比", use_container_width=True, key="compare_run_btn"):
                if not model_path:
                    st.error("参数扫描需要提供 --model_path：请先训练生成模型，或填写模型路径。")
                else:
                    out = _run_action(
                        "compare",
                        "experiments/core/run_compare.py",
                        effective_defense_config,
                        ["--method", method, "--model_path", model_path],
                    )
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
            _render_evaluate_outputs(effective_config_path)
        if rec.action == "defense_eval" and rec.status == "success":
            _render_defense_eval_outputs()
        if rec.action == "compare" and rec.status == "success":
            _render_compare_outputs()


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

