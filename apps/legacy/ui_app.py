"""Legacy Streamlit UI placeholder.

The old command-style UI was retained only to avoid a surprising missing file
for users who saw the historical path. The supported dashboard is
`apps/dashboard.py`.
"""

from __future__ import annotations

import streamlit as st


def main() -> None:
    st.set_page_config(page_title="Legacy UI", layout="centered")
    st.title("Legacy UI")
    st.info(
        "This legacy entry point is retained for reference only. "
        "Use `python -m streamlit run apps/dashboard.py` for the canonical dashboard."
    )


if __name__ == "__main__":
    main()
