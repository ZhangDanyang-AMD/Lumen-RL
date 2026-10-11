#!/usr/bin/env python3
"""Build processed GEAK Kernel SFT samples from a pinned raw manifest."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from multi_tune_agent.sft_dataset import build_main


if __name__ == "__main__":
    raise SystemExit(build_main())
