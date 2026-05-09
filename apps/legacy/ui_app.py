"""legacy Streamlit UI 占位入口。

旧式命令 UI 只为历史路径保留。正式 Dashboard 是 `apps/dashboard.py`。
"""

from __future__ import annotations

import streamlit as st


def main() -> None:
    st.set_page_config(page_title="旧版 UI", layout="centered")
    st.title("旧版 UI")
    st.info(
        "该入口仅作为历史说明保留。正式演示请使用 "
        "`python -m streamlit run apps/dashboard.py`。"
    )


if __name__ == "__main__":
    main()
