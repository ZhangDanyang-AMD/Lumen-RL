#!/usr/bin/env python3
"""Run the strict-audit Phase 1 production controller."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from multi_tune_agent.production_controller import main


if __name__ == "__main__":
    raise SystemExit(main())
