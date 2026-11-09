"""Apply Lumen kernels to embodied-RL components through ``LumenConfig.enable``."""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch.nn as nn

logger = logging.getLogger(__name__)


def apply_lumen(module: nn.Module, lumen_kwargs: Optional[dict[str, Any]], name: str) -> nn.Module:
    """``LumenConfig(**lumen_kwargs).enable(module)``; no-op for empty kwargs."""
    if not lumen_kwargs:
        return module
    from lumen.config import LumenConfig

    _, module = LumenConfig(**dict(lumen_kwargs)).enable(module)
    logger.info("Lumen enabled on %s: %s", name, dict(lumen_kwargs))
    return module
