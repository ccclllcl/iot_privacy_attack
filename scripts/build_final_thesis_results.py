#!/usr/bin/env python3
"""Compatibility wrapper for scripts.final_thesis.build_final_thesis_results."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.final_thesis.build_final_thesis_results import main


if __name__ == "__main__":
    main()
