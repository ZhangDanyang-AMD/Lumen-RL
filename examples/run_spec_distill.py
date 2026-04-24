"""Speculative Decoding Draft Model Distillation launcher for LumenRL.

Trains Eagle3 or DFlash draft models using teacher hidden states,
following the TorchSpec / KimiV2.5 approach.

Usage:
    python examples/run_spec_distill.py --config configs/spec_distill_eagle3.yaml
    python examples/run_spec_distill.py --config examples/KimiV2.5_Draft_Distill_MI300/configs/eagle3_bf16.yaml
"""

from __future__ import annotations

import sys

from lumenrl.core.config import LumenRLConfig
from lumenrl.trainer import SpecDistillTrainer
from lumenrl.utils.logging import setup_logging


def main() -> None:
    setup_logging()
    config = LumenRLConfig.from_cli()

    if config.algorithm.name != "spec_distill":
        print(f"Error: expected algorithm 'spec_distill', got '{config.algorithm.name}'", file=sys.stderr)
        sys.exit(1)

    trainer = SpecDistillTrainer(config)
    trainer.setup()
    trainer.train()
    trainer.cleanup()


if __name__ == "__main__":
    main()
