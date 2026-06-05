"""RL optimization algorithms and registration."""

from __future__ import annotations

import lumenrl.algorithms.advantage_estimators  # populate ADV_ESTIMATOR_REGISTRY
from lumenrl.algorithms.advantage_estimators import ADV_ESTIMATOR_REGISTRY
from lumenrl.algorithms.base_algorithm import BaseAlgorithm
from lumenrl.algorithms.dapo import DAPOAlgorithm
from lumenrl.algorithms.grpo import GRPOAlgorithm
from lumenrl.algorithms.opd import OPDAlgorithm
from lumenrl.algorithms.ppo import PPOAlgorithm
from lumenrl.core.registry import ALGORITHM_REGISTRY

__all__ = [
    "ADV_ESTIMATOR_REGISTRY",
    "ALGORITHM_REGISTRY",
    "BaseAlgorithm",
    "DAPOAlgorithm",
    "GRPOAlgorithm",
    "OPDAlgorithm",
    "PPOAlgorithm",
]
