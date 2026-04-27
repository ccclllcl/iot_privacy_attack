from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import streamlit as st


ROOT = Path(__file__).resolve().parent
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


def main() -> None:
    st.set_page_config(page_title="IoT Privacy - Simple UI", layout="wide")
    st.title("IoT 隐私实验（精简前端）")
    st.caption("保留 mock 与真实数据双轨实验，重点提供最少按钮和结果图展示。")

    st.markdown("## 一键运行")
    c1, c2, c3 = st.columns(3)
    with c1:
        run_mock = st.button("补跑 Mock 全流程", use_container_width=True)
        st.caption("generate → preprocess → train(lstm/mlp) → evaluate → defense/eval/compare")
    with c2:
        run_real = st.button("补跑 Real（跳过已存在）", use_container_width=True)
        st.caption("run_real_public_benchmark.py --skip-existing")
    with c3:
        sync_assets = st.button("同步前端图片资源", use_container_width=True)
        st.caption("把 outputs 里的关键图复制到 web_assets/images/")

    if run_mock:
        cmd = [
            sys.executable,
            "run_all_methods_multiseed.py",
        ]
        code, out = _run(cmd)
        st.code(out or "(no output)", language="text")
        if code == 0:
            st.success("Mock 全流程补跑完成。")
        else:
            st.error(f"Mock 全流程失败，exit_code={code}")

    if run_real:
        cmd = [
            sys.executable,
            "run_real_public_benchmark.py",
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
        if code == 0:
            st.success("真实数据补跑完成。")
        else:
            st.error(f"真实数据补跑失败，exit_code={code}")

    if sync_assets:
        cmd = [sys.executable, "tools/refresh_web_assets.py"]
        code, out = _run(cmd)
        st.code(out or "(no output)", language="text")
        if code == 0:
            st.success("前端图片资源已同步。")
        else:
            st.error(f"同步失败，exit_code={code}")

    st.markdown("---")
    c4, c5 = st.columns(2)
    with c4:
        _render_gallery("mock", "Mock 数据结果图")
    with c5:
        _render_gallery("real", "真实数据结果图")


if __name__ == "__main__":
    main()
