"""miles CLI hook functions for LumenRL injection.

Usage — pass each function via its dotted import path::

    python train.py \\
      --custom-megatron-init-path              lumenrl.plugin.miles.hooks.custom_megatron_init \\
      --custom-model-provider-path             lumenrl.plugin.miles.hooks.custom_model_provider \\
      --rollout-function-path                  lumenrl.plugin.miles.hooks.generate_rollout \\
      --custom-megatron-post-save-hook-path    lumenrl.plugin.miles.hooks.custom_megatron_post_save_hook \\
      --custom-megatron-before-train-step-hook-path lumenrl.plugin.miles.hooks.custom_megatron_before_train_step_hook \\
      --custom-megatron-before-log-prob-hook-path   lumenrl.plugin.miles.hooks.custom_megatron_before_log_prob_hook

The three shared hooks are re-exported from lumenrl.plugin.vime.hooks unchanged.
"""

from __future__ import annotations

import logging
from typing import Any

from lumenrl.plugin.vime.hooks import (  # noqa: F401  re-export
    custom_megatron_init,
    custom_model_provider,
    generate_rollout,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# miles-only Hook A: --custom-megatron-post-save-hook-path
# Signature: (args, rollout_id: int, checkpoint_dir: str,
#             hf_checkpoint_dir: str) -> None
# ---------------------------------------------------------------------------

def custom_megatron_post_save_hook(
    args: Any,
    rollout_id: int,
    checkpoint_dir: str,
    hf_checkpoint_dir: str,
) -> None:
    """No-op by default; extend for custom checkpoint export."""
    logger.debug(
        "lumenrl: post_save_hook rollout_id=%d ckpt=%s",
        rollout_id,
        checkpoint_dir,
    )


# ---------------------------------------------------------------------------
# miles-only Hook B: --custom-megatron-before-train-step-hook-path
# Signature: (args, rollout_id, step_id, model, optimizer,
#             opt_param_scheduler) -> None
# ---------------------------------------------------------------------------

def custom_megatron_before_train_step_hook(
    args: Any,
    rollout_id: int,
    step_id: int,
    model: Any,
    optimizer: Any,
    opt_param_scheduler: Any,
) -> None:
    """Reset FP8 state before each training micro-step.

    ``model`` is typically a list of DDP-wrapped modules (one per virtual
    pipeline stage). Each chunk is reset independently.
    """
    fp8_manager = getattr(args, "_lumenrl_fp8_manager", None)
    if fp8_manager is None:
        return

    chunks = model if isinstance(model, (list, tuple)) else [model]
    for chunk in chunks:
        fp8_manager.reset_fp8_state(chunk)


# ---------------------------------------------------------------------------
# miles-only Hook C: --custom-megatron-before-log-prob-hook-path
# Signature: (args, model, store_prefix: str) -> None
# ---------------------------------------------------------------------------

def custom_megatron_before_log_prob_hook(
    args: Any,
    model: Any,
    store_prefix: str,
) -> None:
    """Reset FP8 state before the forward-only log-prob pass."""
    fp8_manager = getattr(args, "_lumenrl_fp8_manager", None)
    if fp8_manager is None:
        return

    chunks = model if isinstance(model, (list, tuple)) else [model]
    for chunk in chunks:
        fp8_manager.reset_fp8_state(chunk)
