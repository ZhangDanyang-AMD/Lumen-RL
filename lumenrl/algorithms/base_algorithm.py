"""Abstract base class for RL optimization algorithms."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from torch import Tensor

from lumenrl.core.config import LumenRLConfig
from lumenrl.core.protocol import DataProto

logger = logging.getLogger(__name__)


class BaseAlgorithm(ABC):
    """Algorithm interface: advantages, loss, and serialized hyperparameters."""

    def __init__(self, config: LumenRLConfig) -> None:
        self._config = config

    @abstractmethod
    def compute_advantages(self, batch: DataProto) -> DataProto:
        """Populate ``advantages`` (and optionally related tensors) on ``batch``.

        Implementations may return the same object with updated tensors or a
        shallow clone; callers should use the returned ``DataProto``.
        """

    @abstractmethod
    def compute_loss(self, batch: DataProto) -> tuple[Tensor, dict[str, Any]]:
        """Return scalar loss and a dict of detached metrics for logging."""

    def get_config(self) -> dict[str, Any]:
        """Return a JSON-serializable snapshot of algorithm-relevant settings."""
        algo = self._config.algorithm
        return {
            "name": algo.name,
            "grpo": vars(algo.grpo),
            "dapo": vars(algo.dapo),
            "ppo": vars(algo.ppo),
            "opd": vars(algo.opd),
            "spec_distill": vars(algo.spec_distill),
        }
