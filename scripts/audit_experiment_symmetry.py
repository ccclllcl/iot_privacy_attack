#!/usr/bin/env python3
"""兼容入口：调用 scripts.audit.audit_experiment_symmetry。"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit.audit_experiment_symmetry import main


if __name__ == "__main__":
    main()
