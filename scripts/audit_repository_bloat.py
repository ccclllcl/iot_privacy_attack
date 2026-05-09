#!/usr/bin/env python3
"""Compatibility wrapper for scripts.audit.audit_repository_bloat."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit.audit_repository_bloat import main


if __name__ == "__main__":
    main()
