#!/usr/bin/env python3
"""Validate processed GEAK Kernel SFT data and its checksums."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from multi_tune_agent.sft_dataset import validate_main


if __name__ == "__main__":
    raise SystemExit(validate_main())
