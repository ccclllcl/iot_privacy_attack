from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dashboard.io import (  # noqa: E402
    list_artifacts,
    load_confusion,
    plot_confusion_matrix,
    plot_parameter_scan,
    read_csv,
    read_json,
    read_text,
)
from src.dashboard.paths import (  # noqa: E402
    COOJA_METHODS,
    METHODS,
    MODES,
    SUMMARY_FIGURE_ROOT,
    baseline_path,
    cooja_path,
    experiment_path,
    list_datasets,
    list_models,
    list_seeds,
    parameter_scan_path,
    rel_path,
    summary_path,
)
from src.dashboard.runner import RUN_HISTORY, parse_epoch_progress, stream_subprocess  # noqa: E402


SUMMARY_FIGURES = [
    "mock_model_mode_accuracy.png",
    "mock_method_distortion.png",
    "real_uci_har_model_mode_accuracy.png",
    "real_kasteren_model_mode_accuracy.png",
    "real_casas_hh101_model_mode_accuracy.png",
    "parameter_scan_ldp_all_models_modes.png",
    "parameter_scan_noise_all_models_modes.png",
    "parameter_scan_adaptive_ldp_all_models_modes.png",
    "adaptive_ldp_ablation_mock_accuracy.png",
    "adaptive_ldp_ablation_mock_distortion.png",
    "adaptive_ldp_ablation_real_accuracy.png",
    "adaptive_ldp_ablation_real_distortion.png",
    "cooja_mode_accuracy.png",
    "cooja_per_seed_accuracy.png",
    "cooja_traffic_metrics.png",
    "cooja_window_overhead_proxy.png",
    "confusion_mock.png",
    "confusion_uci_har.png",
    "confusion_kasteren.png",
    "confusion_casas_hh101.png",
]


def _metric_value(data: dict, *path: str, default: str | int = "-"):
    cur = data
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _missing_count(path: Path) -> int:
    data = read_json(path)
    return len(data) if isinstance(data, list) else 0


def _display_file(path: Path) -> None:
    st.caption(rel_path(path))
    if not path.exists():
        st.info("File is not available.")
        return
    suffix = path.suffix.lower()
    if suffix == ".json":
        st.json(read_json(path))
    elif suffix == ".csv":
        df = read_csv(path)
        st.dataframe(df, use_container_width=True)
    elif suffix in {".txt", ".md", ".log"}:
        st.code(read_text(path), language="text")
    elif suffix in {".png", ".jpg", ".jpeg"}:
        st.image(str(path), use_container_width=True)
    else:
        st.write({"size_bytes": path.stat().st_size})


def _overview_tab() -> None:
    audit = read_json(summary_path("final_symmetry_audit.json")) or {}
    coverage = read_json(summary_path("parameter_scan_coverage_audit.json")) or {}
    final_summary = read_csv(summary_path("final_summary.csv"))
    mock_summary = read_csv(summary_path("mock/mock_summary.csv"))
    real_summary = read_csv(summary_path("real/real_summary.csv"))
    cooja_summary = read_csv(summary_path("cooja/cooja_summary.csv"))

    st.subheader("Coverage")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("mock main matrix", f"{_metric_value(audit, 'mock_main_matrix', 'completed')}/{_metric_value(audit, 'mock_main_matrix', 'expected')}")
    c2.metric("real main matrix", f"{_metric_value(audit, 'real_main_matrix', 'completed')}/{_metric_value(audit, 'real_main_matrix', 'expected')}")
    c3.metric("mock scans", f"{_metric_value(audit, 'parameter_scan_counts', 'mock', 'completed')}/{_metric_value(audit, 'parameter_scan_counts', 'mock', 'expected')}")
    c4.metric("real scans", f"{_metric_value(audit, 'parameter_scan_counts', 'real', 'completed')}/{_metric_value(audit, 'parameter_scan_counts', 'real', 'expected')}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("adaptive profiles", str(_metric_value(audit, "adaptive_ldp_profile_count", "expected")))
    c6.metric("Cooja canonical", f"{_metric_value(audit, 'cooja', 'canonical_completed')}/{_metric_value(audit, 'cooja', 'canonical_expected')}")
    c7.metric("final missing", str(_missing_count(summary_path("final_missing_outputs.json"))))
    c8.metric("scan missing", str(_missing_count(summary_path("parameter_scan_missing_outputs.json"))))

    st.info("Cooja is shown as functionality validation only; no real energy or end-to-end delay measurement is claimed.")
    with st.expander("Cooja limitations", expanded=False):
        st.markdown(read_text(summary_path("cooja/cooja_limitations.md")) or "No limitation note found.")

    st.subheader("Grouped Summary")
    for title, df in [
        ("final_summary.csv", final_summary),
        ("mock_summary.csv", mock_summary),
        ("real_summary.csv", real_summary),
        ("cooja_summary.csv", cooja_summary),
    ]:
        with st.expander(title, expanded=False):
            if df.empty:
                st.info("No rows available.")
                continue
            st.dataframe(df.head(200), use_container_width=True)
            cols = [c for c in ["dataset", "method", "mode", "model_type", "baseline_acc", "defended_acc", "accuracy"] if c in df.columns]
            numeric = [c for c in ["baseline_acc", "defended_acc", "accuracy"] if c in df.columns]
            if "dataset" in df.columns and numeric:
                view = df[cols].copy()
                for col in numeric:
                    view[col] = pd.to_numeric(view[col], errors="coerce")
                st.dataframe(view.groupby("dataset")[numeric].mean(numeric_only=True), use_container_width=True)


def _artifact_explorer_tab() -> None:
    st.subheader("Artifact Explorer")
    dataset = st.selectbox("dataset", list_datasets(), key="artifact_dataset")
    seed = st.selectbox("seed", list_seeds(dataset), key="artifact_seed")
    if dataset == "cooja":
        dummy_method = st.selectbox("dummy_method", COOJA_METHODS, key="artifact_dummy_method")
        mode = st.selectbox("mode", MODES, key="artifact_cooja_mode")
        path = cooja_path(int(seed), dummy_method, mode)
        st.warning("Cooja packet/byte/IAT, real energy, and end-to-end delay are not fabricated; unavailable fields remain limitations.")
    else:
        model = st.selectbox("model", list_models(dataset), key="artifact_model")
        view_type = st.radio("view_type", ["baseline", "defense_result", "parameter_scan"], horizontal=True)
        if view_type == "baseline":
            path = baseline_path(dataset, int(seed), model)
        else:
            method = st.selectbox("method", METHODS, key=f"artifact_method_{view_type}")
            mode = st.selectbox("mode", MODES, key=f"artifact_mode_{view_type}")
            path = experiment_path(dataset, int(seed), model, method, mode)
            if view_type == "parameter_scan":
                path = parameter_scan_path(dataset, int(seed), model, method, mode)

    st.markdown(f"**Selected path:** `{rel_path(path)}`")
    files = list_artifacts(path)
    if not files:
        st.info("No files found at this path.")
        return
    selected = st.selectbox("artifact file", files, format_func=lambda p: p.name)
    if selected.name.endswith("confusion.json") or selected.name == "baseline_confusion.json":
        st.session_state["selected_confusion_path"] = str(selected)
    _display_file(selected)


def _figures_tab() -> None:
    st.subheader("Summary Figures Gallery")
    cols = st.columns(2)
    for idx, name in enumerate(SUMMARY_FIGURES):
        path = SUMMARY_FIGURE_ROOT / name
        with cols[idx % 2]:
            if path.exists():
                st.image(str(path), caption=rel_path(path), use_container_width=True)
            else:
                st.info(f"Missing: {name}")

    st.subheader("Confusion Matrix Viewer")
    default_conf_text = str(st.session_state.get("selected_confusion_path", ""))
    default_conf = Path(default_conf_text) if default_conf_text else Path("__none__")
    conf_path_text = st.text_input(
        "confusion.json path",
        value=rel_path(default_conf) if default_conf.is_file() else "outputs/experiments/mock/seed_42/lstm/ldp/fixed_attacker/confusion.json",
    )
    conf_path = PROJECT_ROOT / conf_path_text if not Path(conf_path_text).is_absolute() else Path(conf_path_text)
    normalize = st.checkbox("row-normalized", value=True)
    max_labels = st.slider("max labels", min_value=8, max_value=60, value=30)
    conf = load_confusion(conf_path)
    fig, top_df = plot_confusion_matrix(conf, normalize=normalize, max_labels=max_labels)
    if fig is not None:
        st.pyplot(fig, use_container_width=True)
        st.dataframe(top_df, use_container_width=True)
    else:
        st.info("Choose a readable confusion JSON.")

    st.subheader("Parameter Scan Viewer")
    c1, c2, c3, c4, c5 = st.columns(5)
    ds = c1.selectbox("scan dataset", [d for d in list_datasets() if d != "cooja"], key="scan_dataset")
    sd = c2.selectbox("scan seed", list_seeds(ds), key="scan_seed")
    md = c3.selectbox("scan model", list_models(ds), key="scan_model")
    mt = c4.selectbox("scan method", METHODS, key="scan_method")
    mo = c5.selectbox("scan mode", MODES, key="scan_mode")
    scan_path = parameter_scan_path(ds, int(sd), md, mt, mo) / "comparison_results.csv"
    st.caption(rel_path(scan_path))
    scan_df = read_csv(scan_path)
    fig, detail = plot_parameter_scan(scan_df, mt)
    if fig is not None:
        st.pyplot(fig, use_container_width=True)
        if not detail.empty:
            st.dataframe(detail, use_container_width=True)
    else:
        st.info("No readable parameter scan CSV for this selection.")


def _train_eval_tab() -> None:
    st.subheader("Train / Evaluate Demo")
    st.caption("Runs one selected combination only. It does not import data, generate mock data, run Cooja, or run a full matrix.")
    c1, c2, c3 = st.columns(3)
    dataset = c1.selectbox("dataset", ["mock", "uci_har", "kasteren", "casas_hh101"], key="run_dataset")
    seed = c2.selectbox("seed", [42, 123, 2026], key="run_seed")
    model = c3.selectbox("model", ["lstm", "mlp"], key="run_model")
    run_type = st.selectbox("run_type", ["train_baseline", "evaluate_baseline", "defense_eval_fixed", "defense_eval_retrain"])
    method = None
    if run_type.startswith("defense"):
        method = st.selectbox("method", METHODS, key="run_method")
        st.text_input("mode", value="fixed_attacker" if run_type == "defense_eval_fixed" else "retrain_attacker", disabled=True)

    c4, c5, c6 = st.columns(3)
    max_epochs = c4.number_input("max_epochs", min_value=1, max_value=20, value=5, step=1)
    batch_size = c5.selectbox("batch_size", [None, 16, 32, 64, 128], format_func=lambda x: "config default" if x is None else str(x))
    device = c6.selectbox("device", ["auto", "cpu", "cuda"])
    overwrite = st.checkbox("overwrite existing artifacts", value=False)
    confirmed = st.checkbox("I understand this writes to the selected canonical path", value=False)

    cmd = [
        sys.executable,
        "experiments/demo/run_dashboard_job.py",
        "--dataset",
        dataset,
        "--seed",
        str(seed),
        "--model",
        model,
        "--job",
        run_type,
        "--max-epochs",
        str(int(max_epochs)),
        "--device",
        device,
    ]
    if method:
        cmd.extend(["--method", method])
    if batch_size:
        cmd.extend(["--batch-size", str(batch_size)])
    if overwrite:
        cmd.append("--overwrite")

    st.code(" ".join(cmd), language="bash")
    if st.button("Run selected job", type="primary"):
        if overwrite and not confirmed:
            st.error("Please confirm before overwriting canonical artifacts.")
            return
        progress = st.progress(0)
        output_box = st.empty()
        lines: list[str] = []
        current = 5
        progress.progress(current)
        returncode = 1
        for event in stream_subprocess(cmd):
            if event["type"] == "line":
                line = event["text"]
                lines.append(line)
                if "CONFIG_PREPARED" in line:
                    current = max(current, 10)
                elif "TRAINING_STARTED" in line or "DEFENSE_EVALUATION_STARTED" in line:
                    current = max(current, 20)
                elif "EVALUATION_STARTED" in line:
                    current = max(current, 85)
                elif "WRITING_ARTIFACTS" in line:
                    current = max(current, 95)
                elif "DONE" in line:
                    current = 100
                else:
                    parsed = parse_epoch_progress(line, int(max_epochs))
                    if parsed is not None:
                        current = max(current, parsed)
                progress.progress(min(current, 100))
                output_box.code("\n".join(lines[-80:]), language="text")
            else:
                returncode = int(event["returncode"])
        if returncode == 0:
            progress.progress(100)
            st.success("Job completed.")
            result_line = next((x for x in reversed(lines) if x.startswith("RESULT_JSON ")), "")
            if result_line:
                result = json.loads(result_line.replace("RESULT_JSON ", "", 1))
                st.markdown(f"Artifacts written to: `{result.get('output_path')}`")
                out_path = PROJECT_ROOT / str(result.get("output_path", ""))
                conf = out_path / ("baseline_confusion.json" if run_type.endswith("baseline") else "confusion.json")
                report = out_path / ("baseline_classification_report.txt" if run_type.endswith("baseline") else "classification_report.txt")
                fig, top_df = plot_confusion_matrix(load_confusion(conf), normalize=True)
                if fig is not None:
                    st.pyplot(fig, use_container_width=True)
                    st.dataframe(top_df, use_container_width=True)
                if report.exists():
                    st.code(read_text(report), language="text")
        else:
            st.error("Job failed. See output above.")


def _run_history_tab() -> None:
    st.subheader("Run History")
    if not RUN_HISTORY.exists():
        st.info("No dashboard run history yet.")
        return
    rows = []
    for line in RUN_HISTORY.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            pass
    if not rows:
        st.info("Run history is empty.")
        return
    df = pd.DataFrame(rows)
    filters = st.columns(6)
    for col_name, widget in [
        ("dataset", filters[0]),
        ("seed", filters[1]),
        ("model", filters[2]),
        ("method", filters[3]),
        ("mode", filters[4]),
        ("status", filters[5]),
    ]:
        if col_name in df.columns:
            vals = ["all"] + sorted([str(x) for x in df[col_name].dropna().unique()])
            chosen = widget.selectbox(col_name, vals, key=f"hist_{col_name}")
            if chosen != "all":
                df = df[df[col_name].astype(str) == chosen]
    cols = [c for c in ["timestamp", "dataset", "seed", "model", "method", "mode", "job", "status", "duration_seconds", "command", "output_path"] if c in df.columns]
    st.dataframe(df[cols], use_container_width=True)
    if not df.empty:
        idx = st.selectbox("record", list(df.index), format_func=lambda i: f"{df.loc[i].get('timestamp', '')} {df.loc[i].get('job', '')} {df.loc[i].get('status', '')}")
        st.json(df.loc[idx].to_dict())
        out = df.loc[idx].get("output_path")
        if isinstance(out, str) and out:
            path = PROJECT_ROOT / out
            st.markdown(f"Output path: `{out}`")
            for file in list_artifacts(path):
                with st.expander(file.name, expanded=False):
                    _display_file(file)


def main() -> None:
    st.set_page_config(page_title="IoT Privacy Artifact Dashboard", layout="wide")
    st.title("IoT Privacy Artifact Dashboard")
    st.caption("Canonical artifact browser and single-combination train/evaluate demo.")
    tabs = st.tabs([
        "Overview",
        "Artifact Explorer",
        "Figures & Confusion Matrices",
        "Train / Evaluate Demo",
        "Run History",
    ])
    with tabs[0]:
        _overview_tab()
    with tabs[1]:
        _artifact_explorer_tab()
    with tabs[2]:
        _figures_tab()
    with tabs[3]:
        _train_eval_tab()
    with tabs[4]:
        _run_history_tab()


if __name__ == "__main__":
    main()
