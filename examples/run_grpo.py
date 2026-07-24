"""Unified GRPO launcher for LumenRL.

Usage:
    python examples/run_grpo.py --config configs/grpo_dense_bf16.yaml
    python examples/run_grpo.py --config configs/grpo_dense_fp8.yaml
"""

from __future__ import annotations

import sys

from lumenrl.core.config import LumenRLConfig
from lumenrl.trainer import RLTrainer
from lumenrl.utils.logging import setup_logging


def main() -> None:
    setup_logging()
    config = LumenRLConfig.from_cli()

    if config.algorithm.name != "grpo":
        print(f"Error: expected algorithm 'grpo', got '{config.algorithm.name}'", file=sys.stderr)
        sys.exit(1)

    trainer = RLTrainer(config)
    trainer.setup()
    trainer.train()


if __name__ == "__main__":
    main()
