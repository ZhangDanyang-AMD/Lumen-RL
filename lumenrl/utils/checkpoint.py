"""Checkpoint save/load utilities."""

from __future__ import annotations

import logging
import os
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)


def capture_rng_state() -> dict[str, Any]:
    """Capture the process-local RNG streams used by training."""
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    """Restore process-local RNG streams captured by :func:`capture_rng_state`."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    cuda_state = state.get("torch_cuda")
    if cuda_state is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("checkpoint contains CUDA RNG state but CUDA is unavailable")
        torch.cuda.set_rng_state(cuda_state)


def capture_distributed_rng_state(distributed: bool) -> dict[str, Any]:
    """Collect one RNG state per draft rank for a rank-0 checkpoint."""
    local_state = capture_rng_state()
    if (
        distributed
        and torch.distributed.is_available()
        and torch.distributed.is_initialized()
    ):
        world_size = torch.distributed.get_world_size()
        states: list[dict[str, Any] | None] = [None] * world_size
        torch.distributed.all_gather_object(states, local_state)
    else:
        world_size = 1
        states = [local_state]
    return {
        "version": 1,
        "world_size": world_size,
        "states": states,
    }


def restore_distributed_rng_state(
    bundle: dict[str, Any],
    *,
    rank: int,
    world_size: int,
) -> None:
    """Restore this rank's RNG streams from a distributed checkpoint."""
    saved_world_size = int(bundle.get("world_size", 0))
    states = bundle.get("states")
    if (
        saved_world_size != world_size
        or not isinstance(states, list)
        or len(states) != world_size
        or any(state is None for state in states)
    ):
        raise RuntimeError(
            "checkpoint RNG topology mismatch: "
            f"saved_world_size={saved_world_size}, current_world_size={world_size}, "
            f"saved_states={len(states) if isinstance(states, list) else 'invalid'}"
        )
    if rank < 0 or rank >= len(states):
        raise RuntimeError(
            f"checkpoint has no RNG state for rank {rank} "
            f"(saved states={len(states)})"
        )

    state = states[rank]
    assert state is not None
    restore_rng_state(state)


class CheckpointManager:
    """Filesystem checkpoint I/O with step-aware filenames."""

    @staticmethod
    def save(state_dict: dict[str, Any], path: str | Path, step: int) -> None:
        """Persist ``state_dict`` together with ``step`` metadata."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"step": int(step), "state_dict": state_dict}
        tmp_path = path.with_suffix(f"{path.suffix}.incomplete")
        try:
            torch.save(payload, tmp_path)
            os.replace(tmp_path, path)
        finally:
            tmp_path.unlink(missing_ok=True)
        logger.info("Saved checkpoint to %s (step=%d)", path, step)

    @staticmethod
    def load(path: str | Path) -> dict[str, Any]:
        """Load a checkpoint payload produced by :meth:`save`."""
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location="cpu")
        if not isinstance(payload, dict) or "state_dict" not in payload:
            raise ValueError(f"Invalid checkpoint format at {path}")
        logger.info("Loaded checkpoint from %s (step=%s)", path, payload.get("step"))
        return payload

    @staticmethod
    def get_latest(checkpoint_dir: str | Path) -> str | None:
        """Return the path to the highest-step ``checkpoint_*.pt`` file, if any."""
        root = Path(checkpoint_dir)
        if not root.is_dir():
            return None
        best: tuple[int, Path] | None = None
        pattern = re.compile(r"checkpoint_(\d+)\.pt$")
        for p in root.iterdir():
            m = pattern.match(p.name)
            if not m:
                continue
            step = int(m.group(1))
            if best is None or step > best[0]:
                best = (step, p)
        return str(best[1]) if best else None
