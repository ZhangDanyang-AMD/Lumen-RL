"""RL optimization algorithms and registration."""

from __future__ import annotations

from lumenrl.algorithms.base_algorithm import BaseAlgorithm
from lumenrl.algorithms.dapo import DAPOAlgorithm
from lumenrl.algorithms.grpo import GRPOAlgorithm
from lumenrl.algorithms.opd import OPDAlgorithm
from lumenrl.algorithms.ppo import PPOAlgorithm
from lumenrl.core.registry import ALGORITHM_REGISTRY

__all__ = [
    "ALGORITHM_REGISTRY",
    "BaseAlgorithm",
    "DAPOAlgorithm",
    "GRPOAlgorithm",
    "OPDAlgorithm",
    "PPOAlgorithm",
]
