"""Model evaluation script for LumenRL.

Usage:
    python examples/run_eval.py --config configs/grpo_dense_bf16.yaml \\
        --checkpoint results/grpo_dense_bf16/step_200
"""

from __future__ import annotations

import argparse
import logging

from lumenrl.core.config import LumenRLConfig
from lumenrl.utils.checkpoint import CheckpointManager
from lumenrl.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def main() -> None:
    setup_logging()

    parser = argparse.ArgumentParser(description="LumenRL Evaluation")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=100)
    args, overrides = parser.parse_known_args()

    config = LumenRLConfig.from_yaml(args.config, overrides=overrides)

    ckpt_mgr = CheckpointManager()
    state_dict = ckpt_mgr.load(args.checkpoint)
    logger.info("Loaded checkpoint from %s", args.checkpoint)

    logger.info("Evaluation with %d samples", args.num_samples)
    logger.info(
        "Model: %s | Backend: %s",
        config.policy.model_name,
        config.policy.generation_backend,
    )

    # TODO: integrate with actual rollout worker for generation + reward eval
    logger.info("Evaluation complete (placeholder -- full eval not yet wired)")


if __name__ == "__main__":
    main()
