"""
Matplotlib defaults for figures: English-only fonts bundled with matplotlib (no CJK),
avoids missing-glyph boxes in saved PNG/PDF.
"""

from __future__ import annotations

from matplotlib import rcParams


def configure_matplotlib_english() -> None:
    rcParams["font.family"] = "sans-serif"
    rcParams["font.sans-serif"] = [
        "DejaVu Sans",
        "Bitstream Vera Sans",
        "Lucida Grande",
        "Verdana",
        "sans-serif",
    ]
    rcParams["axes.unicode_minus"] = False
