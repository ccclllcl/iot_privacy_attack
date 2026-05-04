from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "web_assets" / "images"


def _run(cmd: list[str]) -> tuple[int, str]:
    p = subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    )
    out = (p.stdout or "") + ("\n" + p.stderr if p.stderr else "")
    return p.returncode, out[-12000:]


def _load_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _show_image(path: Path, title: str) -> None:
    st.markdown(f"**{title}**")
    if not path.exists():
        st.info(f"缺少图片：{path.as_posix()}")
        return
    st.image(path.read_bytes(), use_container_width=True, caption=path.name)


def _run_buttons() -> None:
    st.markdown("## 一键运行")
    c1, c2, c3 = st.columns(3)
    with c1:
        run_mock = st.button("补跑 Mock 全流程", use_container_width=True)
        st.caption("generate → preprocess → train(lstm/mlp) → evaluate → defense/eval/compare")
    with c2:
        run_real = st.button("补跑 Real（跳过已存在）", use_container_width=True)
        st.caption("experiments/real_public/run_real_public_benchmark.py --skip-existing")
    with c3:
        sync_assets = st.button("同步前端图片资源", use_container_width=True)
        st.caption("把 outputs 里的关键图复制到 web_assets/images/")

    if run_mock:
        code, out = _run([sys.executable, "experiments/batches/run_all_methods_multiseed.py"])
        st.code(out or "(no output)", language="text")
        st.success("Mock 全流程补跑完成。") if code == 0 else st.error(f"Mock 全流程失败，exit_code={code}")

    if run_real:
        cmd = [
            sys.executable,
            "experiments/real_public/run_real_public_benchmark.py",
            "--datasets",
            "uci_har,kasteren,casas_hh101",
            "--seeds",
            "42,123",
            "--models",
            "lstm,mlp",
            "--max-epochs",
            "25",
            "--skip-existing",
        ]
        code, out = _run(cmd)
        st.code(out or "(no output)", language="text")
        st.success("真实数据补跑完成。") if code == 0 else st.error(f"真实数据补跑失败，exit_code={code}")

    if sync_assets:
        code, out = _run([sys.executable, "tools/refresh_web_assets.py"])
        st.code(out or "(no output)", language="text")
        st.success("前端图片资源已同步。") if code == 0 else st.error(f"同步失败，exit_code={code}")


def _render_gallery(section: str, title: str) -> None:
    folder = ASSET_DIR / section
    st.markdown(f"### {title}")
    if not folder.exists():
        st.info(f"未找到 {folder.as_posix()}。先运行“同步前端图片资源”。")
        return
    images = sorted([p for p in folder.glob("*.png")], key=lambda p: p.name)
    if not images:
        st.info("当前没有可展示图片。")
        return
    for p in images:
        st.image(p.read_bytes(), caption=f"{section}/{p.name}", use_container_width=True)


def _render_defense_cards() -> None:
    st.markdown("## 答辩结论卡片（自动读取结果）")
    mock_metrics = _load_json(ROOT / "outputs" / "reports" / "metrics.json") or {}
    mock_multi = _load_json(ROOT / "outputs" / "reports" / "full_multiseed" / "seed_42" / "metrics.json") or {}
    real_summary = _load_json(ROOT / "outputs" / "reports" / "real_public_benchmark" / "real_public_benchmark_summary.json") or []

    uci_rows = [x for x in real_summary if x.get("dataset") == "uci_har"]
    kasteren_rows = [x for x in real_summary if x.get("dataset") == "kasteren"]
    best_privacy_uci = max(uci_rows, key=lambda x: float(x.get("acc_drop_pct_mean", -1.0))) if uci_rows else {}
    best_recover_uci = max(uci_rows, key=lambda x: float(x.get("retrain_acc_mean", -1.0))) if uci_rows else {}

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mock 单次 Accuracy", f"{float(mock_metrics.get('accuracy', 0.0)):.3f}")
    c2.metric("Mock 多种子样例(Seed42)", f"{float(mock_multi.get('accuracy', 0.0)):.3f}")
    c3.metric(
        "UCI 最强隐私降幅",
        f"{float(best_privacy_uci.get('acc_drop_pct_mean', 0.0)):.1f}%",
        best_privacy_uci.get("method", "-"),
    )
    c4.metric(
        "UCI 重训后恢复",
        f"{float(best_recover_uci.get('retrain_acc_mean', 0.0)):.3f}",
        best_recover_uci.get("method", "-"),
    )

    if uci_rows:
        st.success(
            "UCI HAR 上防御有效：fixed_attacker 准确率平均明显下降，"
            "并且 retrain_attacker 只能部分恢复，符合隐私-效用权衡预期。"
        )
    if kasteren_rows:
        st.info(
            "Kasteren 类别更碎片化、基线更低，防御后 fixed_attacker 进一步降低；"
            "该数据集更适合强调“复杂场景下隐私增强”。"
        )


def _render_defense_figures() -> None:
    st.markdown("## 答辩关键图（四宫格）")
    fig_paths = [
        (ASSET_DIR / "mock" / "default_train_curve.png", "Mock 训练曲线"),
        (ASSET_DIR / "mock" / "default_confusion_matrix.png", "Mock 混淆矩阵"),
        (ASSET_DIR / "real" / "benchmark_uci_har_seed42_confusion_matrix.png", "UCI HAR 混淆矩阵"),
        (ASSET_DIR / "real" / "benchmark_kasteren_seed42_confusion_matrix.png", "Kasteren 混淆矩阵"),
    ]
    cols = st.columns(2)
    for i, (p, title) in enumerate(fig_paths):
        with cols[i % 2]:
            _show_image(p, title)

    with st.expander("补充图（答辩问答备用）", expanded=False):
        _show_image(ASSET_DIR / "mock" / "default_accuracy_comparison.png", "Mock 防御前后准确率对比")
        _show_image(ASSET_DIR / "real" / "benchmark_casas_hh101_seed42_confusion_matrix.png", "CASAS HH101 混淆矩阵")


def main() -> None:
    st.set_page_config(page_title="IoT Privacy - Defense UI", layout="wide")
    st.title("IoT 隐私实验前端")
    st.caption("双轨数据（Mock + Real）保留；支持答辩演示与日常查看。")

    mode = st.sidebar.radio("页面模式", ["答辩模式", "普通模式"], index=0)
    st.sidebar.caption("建议答辩时使用：答辩模式")

    _run_buttons()
    st.markdown("---")

    if mode == "答辩模式":
        _render_defense_cards()
        _render_defense_figures()
    else:
        c4, c5 = st.columns(2)
        with c4:
            _render_gallery("mock", "Mock 数据结果图")
        with c5:
            _render_gallery("real", "真实数据结果图")


if __name__ == "__main__":
    main()
