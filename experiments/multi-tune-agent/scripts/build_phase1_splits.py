#!/usr/bin/env python3
"""Build append-safe Phase 1 lineage splits and collectable requests."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from multi_tune_agent.phase1_splits import main


if __name__ == "__main__":
    raise SystemExit(main())
