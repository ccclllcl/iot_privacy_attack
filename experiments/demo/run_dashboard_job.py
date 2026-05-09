#!/usr/bin/env python3
"""CLI wrapper for one canonical dashboard job."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dashboard_runner import build_arg_parser, run_dashboard_job


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.job.startswith("defense") and not args.method:
        parser.error("--method is required for defense jobs")
    result = run_dashboard_job(args)
    print("RESULT_JSON " + json.dumps(result, ensure_ascii=False), flush=True)
    if result.get("status") != "success":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
