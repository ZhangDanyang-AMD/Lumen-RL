"""Checkpoint save/load utilities."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Filesystem checkpoint I/O with step-aware filenames."""

    @staticmethod
    def save(state_dict: dict[str, Any], path: str | Path, step: int) -> None:
        """Persist ``state_dict`` together with ``step`` metadata."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"step": int(step), "state_dict": state_dict}
        torch.save(payload, path)
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
