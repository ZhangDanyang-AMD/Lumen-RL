#!/usr/bin/env python3
"""Report required coverage dimensions for GEAK Kernel SFT data."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from multi_tune_agent.sft_dataset import coverage_main


if __name__ == "__main__":
    raise SystemExit(coverage_main())
