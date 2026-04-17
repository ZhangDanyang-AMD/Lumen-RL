"""MoE GRPO launcher with R3 support.

Usage:
    python examples/run_grpo_moe.py --config configs/grpo_moe_fp8_r3.yaml
    python examples/run_grpo_moe.py --config configs/grpo_moe_fp8_r3_multinode.yaml \\
        cluster.num_nodes=4 moe.r3.enabled=true
"""

from __future__ import annotations

import logging
import sys

from lumenrl.core.config import LumenRLConfig
from lumenrl.trainer import RLTrainer
from lumenrl.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def main() -> None:
    setup_logging()
    config = LumenRLConfig.from_cli()

    if config.algorithm.name != "grpo":
        print(f"Error: expected algorithm 'grpo', got '{config.algorithm.name}'", file=sys.stderr)
        sys.exit(1)

    if config.moe.r3.enabled:
        logger.info("R3 router alignment is ENABLED")
    else:
        logger.warning(
            "R3 is DISABLED for MoE model -- MoE training may be unstable. "
            "Set moe.r3.enabled=true for production runs."
        )

    trainer = RLTrainer(config)
    trainer.setup()
    trainer.train()


if __name__ == "__main__":
    main()
