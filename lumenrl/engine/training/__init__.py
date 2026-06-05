"""Training backends, engine abstractions, and weight synchronization."""

from __future__ import annotations

from lumenrl.engine.training.base_engine import BaseEngine, BaseEngineCtx, EngineRegistry
from lumenrl.engine.training.fsdp_backend import FSDP2Backend
from lumenrl.engine.training.fsdp_engine import FSDP2Engine, FSDP2EngineWithLMHead
from lumenrl.engine.training.megatron_backend import MegatronBackend
from lumenrl.engine.training.megatron_engine import (
    MegatronEngine,
    MegatronEngineWithLMHead,
    MegatronEngineWithValueHead,
)

try:
    from lumenrl.engine.training.weight_sync import WeightSyncManager
except ModuleNotFoundError:
    WeightSyncManager = None

__all__ = [
    "BaseEngine",
    "BaseEngineCtx",
    "EngineRegistry",
    "FSDP2Backend",
    "FSDP2Engine",
    "FSDP2EngineWithLMHead",
    "MegatronBackend",
    "MegatronEngine",
    "MegatronEngineWithLMHead",
    "MegatronEngineWithValueHead",
    "WeightSyncManager",
]
