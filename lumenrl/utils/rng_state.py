"""Checkpoint the RNG sources used by actor training, including MXFP4 SR seeds."""
from __future__ import annotations

import logging
import random
from typing import Any

import torch

logger = logging.getLogger(__name__)


def capture_actor_rng_state() -> dict[str, Any]:
    """Capture before the next training update consumes random draws."""
    return {
        "cpu": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        "python": random.getstate(),
    }


def restore_actor_rng_state(state: dict[str, Any]) -> None:
    """Restore available streams without inventing legacy checkpoint state."""
    if state.get("cpu") is not None:
        torch.set_rng_state(state["cpu"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state(state["cuda"])
    if state.get("python") is not None:
        random.setstate(state["python"])
    else:
        logger.warning(
            "Checkpoint lacks Python RNG state; keeping the current Python stream. "
            "MXFP4 stochastic-rounding Philox seeds cannot be replayed exactly from this checkpoint."
        )
