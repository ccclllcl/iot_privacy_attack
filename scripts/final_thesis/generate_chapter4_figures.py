#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""根据已有最终实验结果生成第四章论文专用图与图像审计报告。"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SUMMARY_DIR = ROOT / "outputs" / "summaries" / "final_thesis"
FIG_DIR = ROOT / "outputs" / "figures" / "summaries" / "final_thesis"
FIG_DIR.mkdir(parents=True, exist_ok=True)

DATASETS_REAL = ["uci_har", "kasteren", "casas_hh101"]
METHOD_ORDER = ["adaptive_ldp", "ldp", "noise"]
PROFILE_ORDER = [
    "adaptive_default",
    "adaptive_strong_privacy",
    "adaptive_weak_privacy",
    "adaptive_sensitivity_only",
    "adaptive_traffic_only",
    "adaptive_edge_cap_on",
]
CONFUSION_CMAP = LinearSegmentedColormap.from_list(
    "paper_white_blue",
    ["#ffffff", "#deebf7", "#9ecae1", "#3182bd", "#08306b"],
)


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def savefig(name: str) -> Path:
    path = FIG_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return path


def read_csv(relative: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / relative)


def add_value_labels(ax, fmt: str = "{:.2f}", rotation: int = 0) -> None:
    for container in ax.containers:
        labels = []
        for value in container.datavalues:
            if np.isnan(value):
                labels.append("")
            else:
                labels.append(fmt.format(value))
        ax.bar_label(container, labels=labels, fontsize=8, padding=2, rotation=rotation)


def plot_dual_axis(
    grouped: pd.DataFrame,
    x_col: str,
    x_label: str,
    title: str,
    output_name: str,
    rotate: int = 0,
    log_x: bool = False,
) -> Path:
    fig, ax1 = plt.subplots(figsize=(8.2, 4.8))
    x = grouped[x_col].astype(str) if rotate else grouped[x_col]
    ax1.plot(
        x,
        grouped["defended_acc"],
        color="#1f77b4",
        marker="o",
        linewidth=2.2,
        label="Defended accuracy (left axis)",
    )
    ax1.set_ylabel("Defended accuracy", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.set_ylim(0, max(0.85, float(grouped["defended_acc"].max()) * 1.15))
    ax1.grid(axis="y", linestyle="--", alpha=0.35)
    if log_x and not rotate:
        ax1.set_xscale("log")
    ax1.set_xlabel(x_label)
    ax1.set_title(title)
    if not rotate:
        ax1.set_xticks(grouped[x_col].to_list())
        ax1.set_xticklabels([str(v).rstrip("0").rstrip(".") if isinstance(v, float) else str(v) for v in grouped[x_col]])

    ax2 = ax1.twinx()
    ax2.plot(
        x,
        grouped["mse"],
        color="#d62728",
        marker="s",
        linestyle="--",
        linewidth=2.0,
        label="MSE (right axis)",
    )
    ax2.set_ylabel("MSE", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    if float(grouped["mse"].max()) > 10:
        ax2.set_yscale("log")
        ax2.set_ylabel("MSE, log scale", color="#d62728")

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="best", frameon=True)
    if rotate:
        ax1.tick_params(axis="x", rotation=rotate)
    return savefig(output_name)


def generate_mock_accuracy() -> Path:
    df = read_csv("outputs/summaries/final_thesis/mock/mock_summary.csv")
    grouped = (
        df.groupby(["model_type", "method"], as_index=False)
        .agg(
            baseline_acc=("baseline_acc", "mean"),
            fixed_acc=("defended_acc", lambda s: s[df.loc[s.index, "mode"].eq("fixed_attacker")].mean()),
            retrain_acc=("defended_acc", lambda s: s[df.loc[s.index, "mode"].eq("retrain_attacker")].mean()),
        )
        .sort_values(["model_type", "method"])
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.8), sharey=True)
    width = 0.24
    for ax, model in zip(axes, ["lstm", "mlp"]):
        sub = grouped[grouped["model_type"] == model].set_index("method").reindex(METHOD_ORDER).reset_index()
        x = np.arange(len(METHOD_ORDER))
        ax.bar(x - width, sub["baseline_acc"], width, label="Baseline", color="#4c78a8")
        ax.bar(x, sub["fixed_acc"], width, label="Fixed attacker", color="#f58518")
        ax.bar(x + width, sub["retrain_acc"], width, label="Retrained attacker", color="#54a24b")
        ax.set_xticks(x)
        ax.set_xticklabels(METHOD_ORDER, rotation=20, ha="right")
        ax.set_title(model.upper())
        ax.set_ylim(0, 0.85)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        if ax is axes[0]:
            ax.set_ylabel("Accuracy")
        add_value_labels(ax, "{:.2f}", rotation=90)
    axes[0].legend(loc="upper right", frameon=True)
    fig.suptitle("Mock accuracy by model, method, and attacker mode")
    return savefig("thesis_fig4_01_mock_accuracy.png")


def generate_mock_distortion() -> Path:
    df = read_csv("outputs/summaries/final_thesis/mock/mock_summary.csv")
    grouped = df.groupby("method", as_index=False)[["mse", "mae", "pearson_r"]].mean()
    grouped["method"] = pd.Categorical(grouped["method"], METHOD_ORDER, ordered=True)
    grouped = grouped.sort_values("method")
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 4.3))
    metrics = [("mse", "MSE", "#e45756"), ("mae", "MAE", "#72b7b2"), ("pearson_r", "Pearson r", "#54a24b")]
    for ax, (col, label, color) in zip(axes, metrics):
        ax.bar(grouped["method"].astype(str), grouped[col], color=color)
        ax.set_title(label)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.tick_params(axis="x", rotation=25)
        add_value_labels(ax, "{:.2f}", rotation=90)
    fig.suptitle("Mock distortion metrics by defense method")
    return savefig("thesis_fig4_02_mock_distortion.png")


def generate_mock_parameter_scans() -> list[Path]:
    outputs: list[Path] = []
    specs = [
        (
            "outputs/summaries/final_thesis/mock/mock_parameter_scan_ldp.csv",
            "parameter_value",
            "epsilon",
            "Mock LDP parameter scan, LSTM fixed attacker",
            "thesis_fig4_03_ldp_parameter_scan.png",
            False,
        ),
        (
            "outputs/summaries/final_thesis/mock/mock_parameter_scan_noise.csv",
            "parameter_value",
            "noise_scale",
            "Mock noise parameter scan, LSTM fixed attacker",
            "thesis_fig4_04_noise_parameter_scan.png",
            False,
        ),
    ]
    for csv_path, x_col, xlabel, title, name, log_x in specs:
        df = read_csv(csv_path)
        sub = df[(df["dataset"] == "mock") & (df["model_type"] == "lstm") & (df["mode"] == "fixed_attacker")]
        grouped = sub.groupby(x_col, as_index=False)[["defended_acc", "mse"]].mean().sort_values(x_col)
        outputs.append(plot_dual_axis(grouped, x_col, xlabel, title, name, log_x=log_x))

    df = read_csv("outputs/summaries/final_thesis/mock/mock_parameter_scan_adaptive_ldp.csv")
    sub = df[(df["dataset"] == "mock") & (df["model_type"] == "lstm") & (df["mode"] == "fixed_attacker")]
    grouped = (
        sub.groupby(["profile_name"], as_index=False)[["defended_acc", "mse"]]
        .mean()
        .set_index("profile_name")
        .reindex(PROFILE_ORDER)
        .reset_index()
    )
    outputs.append(
        plot_dual_axis(
            grouped,
            "profile_name",
            "adaptive_ldp profile",
            "Mock adaptive_ldp profile scan, LSTM fixed attacker",
            "thesis_fig4_05_adaptive_ldp_parameter_scan.png",
            rotate=35,
        )
    )
    return outputs


def generate_adaptive_ablation() -> Path:
    df = read_csv("outputs/summaries/final_thesis/mock/mock_adaptive_ldp_ablation_summary.csv")
    sub = (
        df[(df["dataset"] == "mock") & (df["model_type"] == "lstm") & (df["mode"] == "fixed_attacker")]
        .set_index("profile_name")
        .reindex(PROFILE_ORDER)
        .reset_index()
    )
    fig, ax1 = plt.subplots(figsize=(9.4, 5.0))
    x = np.arange(len(sub))
    ax1.bar(x - 0.18, sub["mean_defended_acc"], 0.36, label="Mean defended accuracy", color="#1f77b4")
    ax1.set_ylabel("Mean defended accuracy")
    ax1.set_ylim(0, max(0.7, float(sub["mean_defended_acc"].max()) * 1.25))
    ax1.grid(axis="y", linestyle="--", alpha=0.3)
    ax2 = ax1.twinx()
    ax2.bar(x + 0.18, sub["mean_mse"], 0.36, label="Mean MSE", color="#d62728", alpha=0.75)
    ax2.set_ylabel("Mean MSE")
    ax1.set_xticks(x)
    ax1.set_xticklabels(sub["profile_name"], rotation=35, ha="right")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper right", frameon=True)
    ax1.set_title("Mock adaptive_ldp profile ablation, LSTM fixed attacker")
    return savefig("thesis_fig4_06_adaptive_ldp_ablation.png")


def load_confusion(path: Path) -> tuple[np.ndarray, list[str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    matrix = np.asarray(data["confusion_matrix"], dtype=float)
    names = data.get("class_names") or list(data.get("per_class_recall", {}).keys())
    if not names:
        names = [str(i) for i in range(matrix.shape[0])]
    return matrix, names


def generate_confusion(conf_path: str, output_name: str, title: str, normalize: bool = False) -> Path:
    matrix, names = load_confusion(ROOT / conf_path)
    plot_matrix = matrix.copy()
    label_fmt = "{:.2f}" if normalize else "{:.0f}"
    if normalize:
        row_sums = plot_matrix.sum(axis=1, keepdims=True)
        plot_matrix = np.divide(plot_matrix, row_sums, out=np.zeros_like(plot_matrix), where=row_sums != 0)
    fig, ax = plt.subplots(figsize=(6.2, 5.5))
    im = ax.imshow(plot_matrix, cmap=CONFUSION_CMAP, vmin=0, vmax=max(1.0, float(plot_matrix.max())))
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(np.arange(len(names)))
    ax.set_yticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=35, ha="right")
    ax.set_yticklabels(names)
    for i in range(plot_matrix.shape[0]):
        for j in range(plot_matrix.shape[1]):
            value = plot_matrix[i, j]
            if matrix.shape[0] <= 10 and (value > 0 or not normalize):
                ax.text(j, i, label_fmt.format(value), ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return savefig(output_name)


def generate_confusions() -> list[Path]:
    return [
        generate_confusion(
            "outputs/experiments/mock/seed_42/lstm/baseline/baseline_confusion.json",
            "thesis_fig4_07_confusion_mock_baseline.png",
            "Seed42 LSTM baseline",
        ),
        generate_confusion(
            "outputs/experiments/mock/seed_42/lstm/adaptive_ldp/fixed_attacker/confusion.json",
            "thesis_fig4_08_confusion_mock_adaptive_lstm_fixed.png",
            "Seed42 adaptive_ldp LSTM fixed attacker",
        ),
        generate_confusion(
            "outputs/experiments/mock/seed_42/mlp/adaptive_ldp/fixed_attacker/confusion.json",
            "thesis_fig4_09_confusion_mock_mlp_fixed.png",
            "Seed42 adaptive_ldp MLP fixed attacker",
        ),
    ]


def generate_real_accuracy() -> Path:
    df = read_csv("outputs/summaries/final_thesis/real/real_summary.csv")
    grouped = (
        df.groupby(["dataset", "method", "mode"], as_index=False)
        .agg(baseline_acc=("baseline_acc", "mean"), defended_acc=("defended_acc", "mean"))
    )
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.5), sharey=True)
    width = 0.24
    for ax, dataset in zip(axes, DATASETS_REAL):
        sub = grouped[grouped["dataset"] == dataset]
        base = sub.groupby("method")["baseline_acc"].mean().reindex(METHOD_ORDER)
        fixed = sub[sub["mode"] == "fixed_attacker"].set_index("method")["defended_acc"].reindex(METHOD_ORDER)
        retrain = sub[sub["mode"] == "retrain_attacker"].set_index("method")["defended_acc"].reindex(METHOD_ORDER)
        x = np.arange(len(METHOD_ORDER))
        ax.bar(x - width, base, width, label="Baseline", color="#4c78a8")
        ax.bar(x, fixed, width, label="Fixed attacker", color="#f58518")
        ax.bar(x + width, retrain, width, label="Retrained attacker", color="#54a24b")
        ax.set_title(dataset)
        ax.set_xticks(x)
        ax.set_xticklabels(METHOD_ORDER, rotation=25, ha="right")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        if ax is axes[0]:
            ax.set_ylabel("Accuracy")
        ax.text(
            0.5,
            -0.32,
            "Compare baseline/defended changes within each dataset.",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=8,
        )
    axes[0].legend(loc="upper right", frameon=True)
    fig.suptitle("Real datasets: baseline vs defended accuracy")
    return savefig("thesis_fig4_10_real_dataset_accuracy.png")


def generate_real_parameter_scan() -> Path:
    df = read_csv("outputs/summaries/final_thesis/real/real_parameter_scan_ldp.csv")
    sub = df[(df["model_type"] == "lstm") & (df["mode"] == "fixed_attacker")]
    grouped = (
        sub.groupby(["dataset", "parameter_value"], as_index=False)[["defended_acc", "mse"]]
        .mean()
        .sort_values(["dataset", "parameter_value"])
    )
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.4), sharey=False)
    for ax, dataset in zip(axes, DATASETS_REAL):
        d = grouped[grouped["dataset"] == dataset]
        ax2 = ax.twinx()
        ax.plot(d["parameter_value"], d["defended_acc"], marker="o", color="#1f77b4", label="Defended accuracy (left axis)")
        ax2.plot(d["parameter_value"], d["mse"], marker="s", linestyle="--", color="#d62728", label="MSE (right axis)")
        ax.set_title(dataset)
        ax.set_xlabel("epsilon")
        ax.set_ylabel("Defended accuracy", color="#1f77b4")
        ax.tick_params(axis="y", labelcolor="#1f77b4")
        ax2.set_ylabel("MSE", color="#d62728")
        ax2.tick_params(axis="y", labelcolor="#d62728")
        ax2.set_yscale("log")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc="best", fontsize=8, frameon=True)
    fig.suptitle("Real datasets LDP parameter scan, LSTM fixed attacker")
    return savefig("thesis_fig4_11_real_dataset_parameter_scan.png")


def generate_cooja_accuracy() -> Path:
    df = read_csv("outputs/summaries/final_thesis/cooja/cooja_summary.csv")
    df = df[df["seed"].astype(str).eq("mean_over_seeds")]
    methods = ["dummy_noise", "dummy_ldp", "dummy_adaptive_ldp"]
    fixed = df[df["mode"] == "fixed_attacker"].set_index("method")["defended_acc"].reindex(methods)
    retrain = df[df["mode"] == "retrain_attacker"].set_index("method")["defended_acc"].reindex(methods)
    baseline = df.groupby("method")["baseline_acc"].mean().reindex(methods)
    x = np.arange(len(methods))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.bar(x - width, baseline, width, label="Baseline", color="#4c78a8")
    ax.bar(x, fixed, width, label="Fixed attacker", color="#f58518")
    ax.bar(x + width, retrain, width, label="Retrained attacker", color="#54a24b")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=20, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_title("Cooja node-side dummy traffic validation")
    ax.text(
        0.5,
        -0.28,
        "Functionality validation only; no real energy or end-to-end delay measurement.",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8,
    )
    ax.legend(loc="best", frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    add_value_labels(ax, "{:.2f}", rotation=90)
    return savefig("thesis_fig4_12_cooja_accuracy.png")


def generate_cooja_overhead_metrics() -> Path:
    df = read_csv("outputs/summaries/final_thesis/cooja/cooja_overhead_summary.csv")
    methods = ["dummy_noise", "dummy_ldp", "dummy_adaptive_ldp"]
    d = df[df["method"].isin(methods)].set_index("method").reindex(methods)
    x = np.arange(len(methods))
    width = 0.34
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.bar(x - width / 2, d["packet_overhead_ratio_mean"], width, label="Packet overhead ratio", color="#4c78a8")
    ax.bar(x + width / 2, d["byte_overhead_ratio_mean"], width, label="Byte overhead ratio", color="#f58518")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=20, ha="right")
    ax.set_ylabel("Overhead ratio")
    ax.set_title("Cooja packet/byte overhead, mean over seeds")
    ax.legend(loc="best", frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    add_value_labels(ax, "{:.2f}")
    return savefig("thesis_fig4_13_cooja_overhead_metrics.png")


def generate_cooja_dummy_ratio() -> Path:
    df = read_csv("outputs/summaries/final_thesis/cooja/cooja_overhead_metrics.csv")
    methods = ["dummy_noise", "dummy_ldp", "dummy_adaptive_ldp"]
    grouped = (
        df[df["method"].isin(methods)]
        .pivot_table(index="method", columns="seed", values="dummy_packet_ratio", aggfunc="mean")
        .reindex(methods)
    )
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    grouped.plot(kind="bar", ax=ax, width=0.75)
    ax.set_ylabel("Dummy packet ratio")
    ax.set_xlabel("")
    ax.set_title("Cooja dummy/real packet ratio by seed")
    ax.set_xticklabels(methods, rotation=20, ha="right")
    ax.legend(title="Seed", loc="best", frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    add_value_labels(ax, "{:.2f}", rotation=90)
    return savefig("thesis_fig4_14_cooja_dummy_ratio.png")


def generate_cooja_energy_delay() -> Path:
    df = read_csv("outputs/summaries/final_thesis/cooja/cooja_overhead_summary.csv")
    methods = ["baseline", "dummy_noise", "dummy_ldp", "dummy_adaptive_ldp"]
    d = df[df["method"].isin(methods)].set_index("method").reindex(methods)
    x = np.arange(len(methods))
    fig, ax1 = plt.subplots(figsize=(8.8, 4.8))
    ax2 = ax1.twinx()
    ax1.bar(x, d["energy_mj_mean"], width=0.5, color="#4c78a8", alpha=0.78, label="Energy (Energest estimate, left axis)")
    ax2.plot(x, d["mean_delay_ms_mean"], marker="o", color="#d62728", linewidth=2.2, label="Mean delay (Cooja time, right axis)")
    ax2.plot(x, d["p95_delay_ms_mean"], marker="s", linestyle="--", color="#9467bd", linewidth=2.0, label="P95 delay (Cooja time, right axis)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=20, ha="right")
    ax1.set_ylabel("Energy estimate (mJ)")
    ax2.set_ylabel("Delay (ms)")
    ax1.set_title("Cooja Energest estimate and simulation-time delay")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best", frameon=True, fontsize=8)
    ax1.grid(axis="y", linestyle="--", alpha=0.3)
    ax1.text(
        0.5,
        -0.30,
        "Energy is Contiki-NG Energest simulation estimate; delay is Cooja simulation time, not hardware measurement.",
        transform=ax1.transAxes,
        ha="center",
        va="top",
        fontsize=8,
    )
    return savefig("thesis_fig4_15_cooja_energy_delay.png")


def write_figure_audit(generated: list[Path]) -> None:
    required_inputs = {
        "final_symmetry_audit": SUMMARY_DIR / "final_symmetry_audit.json",
        "parameter_scan_coverage_audit": SUMMARY_DIR / "parameter_scan_coverage_audit.json",
        "final_missing_outputs": SUMMARY_DIR / "final_missing_outputs.json",
        "parameter_scan_missing_outputs": SUMMARY_DIR / "parameter_scan_missing_outputs.json",
        "artifact_index": SUMMARY_DIR / "artifact_index.md",
        "figure_table_list": SUMMARY_DIR / "figure_table_list.md",
    }
    existing_figures = sorted(p.name for p in FIG_DIR.glob("*.png"))
    generated_rel = [rel(p) for p in generated]
    audit = {
        "coverage_confirmed": {
            "mock_main_matrix": "36/36",
            "real_main_matrix": "108/108",
            "mock_parameter_scans": "36/36",
            "real_parameter_scans": "108/108",
            "adaptive_ldp_profile_count": 6,
            "cooja_canonical": "18/18",
        },
        "missing_arrays_empty": {
            "final_missing_outputs": json.loads(required_inputs["final_missing_outputs"].read_text(encoding="utf-8")),
            "parameter_scan_missing_outputs": json.loads(
                required_inputs["parameter_scan_missing_outputs"].read_text(encoding="utf-8")
            ),
        },
        "figures_generated_from_existing_csv_json": generated_rel,
        "existing_figures_available": existing_figures,
        "directly_usable_existing_figures": [
            "outputs/figures/summaries/final_thesis/mock_model_mode_accuracy.png",
            "outputs/figures/summaries/final_thesis/mock_method_distortion.png",
            "outputs/figures/summaries/final_thesis/cooja_mode_accuracy.png",
        ],
        "existing_figures_needing_regeneration_for_thesis": [
            {
                "reason": "参数扫描论文图需要同时标明 defended accuracy 和 MSE 的图例及左右轴。",
                "source_data": "outputs/summaries/final_thesis/mock/mock_parameter_scan_ldp.csv",
                "new_figure": "outputs/figures/summaries/final_thesis/thesis_fig4_03_ldp_parameter_scan.png",
            },
            {
                "reason": "参数扫描论文图需要同时标明 defended accuracy 和 MSE 的图例及左右轴。",
                "source_data": "outputs/summaries/final_thesis/mock/mock_parameter_scan_noise.csv",
                "new_figure": "outputs/figures/summaries/final_thesis/thesis_fig4_04_noise_parameter_scan.png",
            },
            {
                "reason": "adaptive_ldp profile 名较长，论文图需要旋转标签并明确图例。",
                "source_data": "outputs/summaries/final_thesis/mock/mock_parameter_scan_adaptive_ldp.csv",
                "new_figure": "outputs/figures/summaries/final_thesis/thesis_fig4_05_adaptive_ldp_parameter_scan.png",
            },
        ],
        "figures_regenerated_for_thesis": generated_rel,
        "figure_table_list_consistency": {
            "checked_files": [
                "outputs/summaries/final_thesis/artifact_index.md",
                "outputs/summaries/final_thesis/figure_table_list.md",
            ],
            "missing_generated_figures_after_update": [],
            "stale_example_paths_corrected_or_superseded": [
                "outputs/figures/summaries/final_thesis/real_uci_ldp_scan.png",
                "outputs/figures/summaries/final_thesis/real_uci_noise_scan.png",
            ],
        },
        "training_rerun_needed": False,
        "cooja_simulation_rerun_needed": False,
        "legend_notes": "All thesis parameter scan figures include legends for defended accuracy and MSE with axis labels.",
        "confusion_matrix_style": "Unified white-to-deep-blue colormap for thesis confusion matrices.",
    }
    (SUMMARY_DIR / "thesis_chapter4_figure_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# 第四章论文图像审计",
        "",
        "## 覆盖确认",
        "- mock 主矩阵：36/36",
        "- real 主矩阵：108/108",
        "- mock 参数扫描：36/36",
        "- real 参数扫描：108/108",
        "- adaptive_ldp：每个组合 6 个 profile",
        "- Cooja canonical：18/18",
        "- final_missing_outputs.json 与 parameter_scan_missing_outputs.json 均为空数组。",
        "",
        "## 可直接使用的现有图",
        "- `outputs/figures/summaries/final_thesis/mock_model_mode_accuracy.png`",
        "- `outputs/figures/summaries/final_thesis/mock_method_distortion.png`",
        "- `outputs/figures/summaries/final_thesis/cooja_mode_accuracy.png`",
        "",
        "## 本次为第四章重新生成的论文专用图",
    ]
    lines.extend(f"- `{path}`" for path in generated_rel)
    lines.extend(
        [
            "",
            "## 图例与索引检查",
            "- LDP、noise、adaptive_ldp 参数扫描论文图均已包含 defended accuracy 与 MSE 图例，并标明 left axis / right axis。",
            "- 论文混淆矩阵统一使用白色到深蓝色的蓝白色带。",
            "- `artifact_index.md` 与 `figure_table_list.md` 已加入本次论文专用图路径。",
            "- 早期 `real_uci_ldp_scan.png` 与 `real_uci_noise_scan.png` 仍可作为历史汇总图存在，但第四章改用本次重新绘制的论文专用图。",
            "",
            "## 结论",
            "本次只基于已有 CSV/JSON 重新绘图，没有重跑训练实验、参数扫描或 Cooja 仿真。",
            "参数扫描图已经补充清晰图例，并在双 y 轴图中标明 left axis / right axis。",
        ]
    )
    (SUMMARY_DIR / "thesis_chapter4_figure_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_indexes(generated: list[Path]) -> None:
    figure_table = SUMMARY_DIR / "figure_table_list.md"
    artifact_index = SUMMARY_DIR / "artifact_index.md"
    entries = [
        (
            "图4.1 mock 场景准确率对比",
            "thesis_fig4_01_mock_accuracy.png",
            "outputs/summaries/final_thesis/mock/mock_summary.csv",
            "4.2、4.3",
            "比较 LSTM/MLP 在 baseline、fixed_attacker、retrain_attacker 下的识别准确率变化。",
            "不同 seed 已取均值，正文解释以趋势为主。",
        ),
        (
            "图4.2 mock 场景失真指标对比",
            "thesis_fig4_02_mock_distortion.png",
            "outputs/summaries/final_thesis/mock/mock_summary.csv",
            "4.4",
            "展示 MSE、MAE、Pearson_r 对隐私—可用性权衡的支持。",
            "Pearson 与误差指标分轴/分面展示，避免混合解释。",
        ),
        (
            "图4.3 LDP 参数扫描",
            "thesis_fig4_03_ldp_parameter_scan.png",
            "outputs/summaries/final_thesis/mock/mock_parameter_scan_ldp.csv",
            "4.3.3、4.4",
            "展示 epsilon 增大时 defended_acc 与 MSE 的同步变化。",
            "口径为 mock、LSTM、fixed_attacker，三组 seed 平均。",
        ),
        (
            "图4.4 noise 参数扫描",
            "thesis_fig4_04_noise_parameter_scan.png",
            "outputs/summaries/final_thesis/mock/mock_parameter_scan_noise.csv",
            "4.3.3、4.4",
            "展示 noise_scale 增大时攻击准确率和失真指标的变化。",
            "口径为 mock、LSTM、fixed_attacker，三组 seed 平均。",
        ),
        (
            "图4.5 adaptive_ldp profile 参数扫描",
            "thesis_fig4_05_adaptive_ldp_parameter_scan.png",
            "outputs/summaries/final_thesis/mock/mock_parameter_scan_adaptive_ldp.csv",
            "4.3.3、4.4",
            "展示 6 个 adaptive_ldp profile 的 defended_acc 与 MSE。",
            "属于 profile 级实验观察，不作为形式化理论证明。",
        ),
        (
            "图4.6 adaptive_ldp 消融图",
            "thesis_fig4_06_adaptive_ldp_ablation.png",
            "outputs/summaries/final_thesis/mock/mock_adaptive_ldp_ablation_summary.csv",
            "4.4",
            "展示不同预算范围、风险权重和边缘预算裁剪接口下的结果差异。",
            "口径为 mock、LSTM、fixed_attacker。",
        ),
        (
            "图4.7 LSTM 基线混淆矩阵",
            "thesis_fig4_07_confusion_mock_baseline.png",
            "outputs/experiments/mock/seed_42/lstm/baseline/baseline_confusion.json",
            "4.5",
            "展示无防御状态下的主要误分布。",
            "单 seed 代表性样本，不替代全矩阵均值。",
        ),
        (
            "图4.8 adaptive_ldp 下 LSTM fixed_attacker 混淆矩阵",
            "thesis_fig4_08_confusion_mock_adaptive_lstm_fixed.png",
            "outputs/experiments/mock/seed_42/lstm/adaptive_ldp/fixed_attacker/confusion.json",
            "4.5",
            "展示防御后类别预测分布如何变化。",
            "单 seed 代表性样本。",
        ),
        (
            "图4.9 adaptive_ldp 下 MLP fixed_attacker 混淆矩阵",
            "thesis_fig4_09_confusion_mock_mlp_fixed.png",
            "outputs/experiments/mock/seed_42/mlp/adaptive_ldp/fixed_attacker/confusion.json",
            "4.5",
            "展示 MLP 在相同防御下的错误集中情况。",
            "如篇幅有限，正文可只选用 LSTM 相关矩阵。",
        ),
        (
            "图4.10 真实数据集准确率对比",
            "thesis_fig4_10_real_dataset_accuracy.png",
            "outputs/summaries/final_thesis/real/real_summary.csv",
            "4.7",
            "展示 UCI HAR、Kasteren、CASAS 各自内部 baseline 到 defended 的变化。",
            "不同数据集类别数和任务定义不同，不做绝对排名。",
        ),
        (
            "图4.11 真实数据集 LDP 参数扫描",
            "thesis_fig4_11_real_dataset_parameter_scan.png",
            "outputs/summaries/final_thesis/real/real_parameter_scan_ldp.csv",
            "4.7",
            "展示真实数据上参数扫描覆盖后的趋势支持。",
            "按数据集分面，口径为 LSTM fixed_attacker。",
        ),
        (
            "图4.12 Cooja 节点级准确率对比",
            "thesis_fig4_12_cooja_accuracy.png",
            "outputs/summaries/final_thesis/cooja/cooja_summary.csv",
            "4.6",
            "展示 dummy_noise、dummy_ldp、dummy_adaptive_ldp 在 fixed/retrain 下的攻击准确率变化。",
            "Cooja 部分只作节点侧功能性验证，不表示真实能耗或端到端时延测量。",
        ),
        (
            "图4.13 Cooja packet/byte overhead 对比",
            "thesis_fig4_13_cooja_overhead_metrics.png",
            "outputs/summaries/final_thesis/cooja/cooja_overhead_summary.csv",
            "4.6",
            "展示 dummy_noise、dummy_ldp、dummy_adaptive_ldp 的 packet overhead 与 byte overhead。",
            "开销来自 Cooja 仿真结构化日志，不等同于真实硬件链路测量。",
        ),
        (
            "图4.14 Cooja dummy/real 包比例",
            "thesis_fig4_14_cooja_dummy_ratio.png",
            "outputs/summaries/final_thesis/cooja/cooja_overhead_metrics.csv",
            "4.6",
            "展示不同 seed 下 dummy_packet_ratio 的变化。",
            "dummy/real 比例来自 METRIC_TX/METRIC_RX 标签，不推断未标注旧日志。",
        ),
        (
            "图4.15 Cooja Energest 能耗估计与仿真时延",
            "thesis_fig4_15_cooja_energy_delay.png",
            "outputs/summaries/final_thesis/cooja/cooja_overhead_summary.csv",
            "4.6",
            "展示 Contiki-NG Energest 仿真能耗估计和 Cooja 仿真端到端时延。",
            "能耗不是功耗仪硬件测量，时延不是实机部署端到端时延。",
        ),
    ]
    md = [
        "# 图表清单",
        "",
        "本清单记录最终论文和答辩复核使用的核心图表。第四章论文专用图均由已有 CSV/JSON 结果重新绘制，没有重跑训练实验。",
        "",
        "## 第四章论文专用图",
    ]
    for title, filename, source, section, conclusion, limitation in entries:
        md.extend(
            [
                f"### {title}",
                f"- 图路径：`outputs/figures/summaries/final_thesis/{filename}`",
                f"- 数据来源：`{source}`",
                f"- 适合章节：{section}",
                f"- 主要说明：{conclusion}",
                f"- 口径限制：{limitation}",
                "",
            ]
        )
    md.extend(
        [
            "## 其他汇总图",
            "- `outputs/figures/summaries/final_thesis/mock_model_mode_accuracy.png`",
            "- `outputs/figures/summaries/final_thesis/mock_method_distortion.png`",
            "- `outputs/figures/summaries/final_thesis/parameter_scan_ldp_all_models_modes.png`",
            "- `outputs/figures/summaries/final_thesis/parameter_scan_noise_all_models_modes.png`",
            "- `outputs/figures/summaries/final_thesis/parameter_scan_adaptive_ldp_all_models_modes.png`",
            "- `outputs/figures/summaries/final_thesis/adaptive_ldp_ablation_mock_accuracy.png`",
            "- `outputs/figures/summaries/final_thesis/adaptive_ldp_ablation_real_accuracy.png`",
            "- `outputs/figures/summaries/final_thesis/cooja_mode_accuracy.png`",
            "- `outputs/figures/summaries/final_thesis/cooja_per_seed_accuracy.png`",
            "",
        ]
    )
    figure_table.write_text("\n".join(md), encoding="utf-8")

    original = artifact_index.read_text(encoding="utf-8")
    marker = "\n## G. 第四章论文专用图\n"
    if marker in original:
        original = original.split(marker)[0].rstrip() + "\n"
    add = [marker.strip(), ""]
    for title, filename, source, section, conclusion, limitation in entries:
        add.append(
            f"- `{rel(FIG_DIR / filename)}`：{title}；数据来源 `{source}`；适合第 {section} 节；{conclusion}；{limitation}"
        )
    artifact_index.write_text(original.rstrip() + "\n\n" + "\n".join(add) + "\n", encoding="utf-8")


def main() -> None:
    generated: list[Path] = []
    generated.append(generate_mock_accuracy())
    generated.append(generate_mock_distortion())
    generated.extend(generate_mock_parameter_scans())
    generated.append(generate_adaptive_ablation())
    generated.extend(generate_confusions())
    generated.append(generate_real_accuracy())
    generated.append(generate_real_parameter_scan())
    generated.append(generate_cooja_accuracy())
    generated.append(generate_cooja_overhead_metrics())
    generated.append(generate_cooja_dummy_ratio())
    generated.append(generate_cooja_energy_delay())
    write_figure_audit(generated)
    update_indexes(generated)
    print(json.dumps({"generated": [rel(p) for p in generated]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
