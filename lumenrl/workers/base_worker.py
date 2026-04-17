"""Abstract base class for Ray-side LumenRL workers."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any


def get_nested_config(config: dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Return ``config[k1][k2]...`` when present, else ``default``."""
    cur: Any = config
    for key in keys:
        if not isinstance(cur, dict):
            return default
        if key not in cur:
            return default
        cur = cur[key]
    return cur


class BaseWorker(ABC):
    """Single-process worker contract used by Ray actors.

    The controller stays in one process; each worker actor inherits from this
    class, owns local devices, and exchanges batches via :class:`DataProto`.
    """

    def __init__(self, rank: int, world_size: int, config: dict[str, Any] | None = None) -> None:
        self.rank = rank
        self.world_size = world_size
        self.config: dict[str, Any] = dict(config or {})
        self._log = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        self._configure_logging()

    def _configure_logging(self) -> None:
        """Ensure worker logs include rank for multi-actor debugging."""
        if not self._log.handlers:
            handler = logging.StreamHandler()
            fmt = logging.Formatter(
                fmt=f"[rank{self.rank}/{self.world_size}] %(name)s: %(levelname)s %(message)s"
            )
            handler.setFormatter(fmt)
            self._log.addHandler(handler)
        self._log.setLevel(logging.INFO)

    @abstractmethod
    def init_model(self) -> None:
        """Allocate models, optimizers, and device state."""

    def cleanup(self) -> None:
        """Release GPU memory and tear down runtime hooks."""
        self._log.info("%s.cleanup: default no-op complete.", self.__class__.__name__)
