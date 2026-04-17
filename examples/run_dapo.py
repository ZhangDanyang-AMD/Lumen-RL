"""DAPO launcher for LumenRL.

Usage:
    python examples/run_dapo.py --config configs/dapo_dense_fp8.yaml
    python examples/run_dapo.py --config configs/dapo_moe_fp8_r3.yaml
"""

from __future__ import annotations

import sys

from lumenrl.core.config import LumenRLConfig
from lumenrl.trainer import RLTrainer
from lumenrl.utils.logging import setup_logging


def main() -> None:
    setup_logging()
    config = LumenRLConfig.from_cli()

    if config.algorithm.name != "dapo":
        print(f"Error: expected algorithm 'dapo', got '{config.algorithm.name}'", file=sys.stderr)
        sys.exit(1)

    trainer = RLTrainer(config)
    trainer.setup()
    trainer.train()


if __name__ == "__main__":
    main()
