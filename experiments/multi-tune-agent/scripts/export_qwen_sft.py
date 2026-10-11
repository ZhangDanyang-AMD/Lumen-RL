#!/usr/bin/env python3
"""Export validated GEAK Kernel SFT samples for Qwen training."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from multi_tune_agent.sft_training_export import main


if __name__ == "__main__":
    raise SystemExit(main())
