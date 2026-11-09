#!/usr/bin/env python3
"""Audit GEAK Kernel SFT lineage, family, task, and text leakage."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from multi_tune_agent.sft_dataset import leakage_main


if __name__ == "__main__":
    raise SystemExit(leakage_main())
