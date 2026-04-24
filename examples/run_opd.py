"""On-Policy Distillation (OPD) launcher for LumenRL.

Implements DeepSeek-V4 style on-policy distillation: student rollout,
teacher forward, then KL(student || teacher) training.

Usage:
    python examples/run_opd.py --config configs/opd_dense_bf16.yaml
    python examples/run_opd.py --config examples/DeepSeekV4_OPD_MI300/configs/opd_bf16.yaml
"""

from __future__ import annotations

import sys

from lumenrl.core.config import LumenRLConfig
from lumenrl.trainer import OPDTrainer
from lumenrl.utils.logging import setup_logging


def main() -> None:
    setup_logging()
    config = LumenRLConfig.from_cli()

    if config.algorithm.name != "opd":
        print(f"Error: expected algorithm 'opd', got '{config.algorithm.name}'", file=sys.stderr)
        sys.exit(1)

    trainer = OPDTrainer(config)
    trainer.setup()
    trainer.train()
    trainer.cleanup()


if __name__ == "__main__":
    main()
