"""Training entrypoints and orchestration.

Trainers load on first access so that importing one submodule (e.g. ``vec_env_ppo_trainer``) does not pull in
the LLM trainers and their kernel dependencies.
"""

from __future__ import annotations

import importlib
from typing import Any

_TRAINERS = {
    "AsyncRLTrainer": "lumenrl.trainer.async_trainer",
    "EmbodiedTrainer": "lumenrl.trainer.embodied_trainer",
    "OPDTrainer": "lumenrl.trainer.opd_trainer",
    "RLTrainer": "lumenrl.trainer.rl_trainer",
    "SpecDistillTrainer": "lumenrl.trainer.spec_distill_trainer",
    "VecEnvPPOTrainer": "lumenrl.trainer.vec_env_ppo_trainer",
}

__all__ = sorted(_TRAINERS)


def __getattr__(name: str) -> Any:
    module = _TRAINERS.get(name)
    if module is None:
        raise AttributeError(f"module 'lumenrl.trainer' has no attribute {name!r}")
    return getattr(importlib.import_module(module), name)
