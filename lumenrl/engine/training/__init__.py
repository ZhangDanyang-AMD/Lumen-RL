"""Training backends and engine abstractions."""

from __future__ import annotations

from lumenrl.engine.training.base_engine import BaseEngine, BaseEngineCtx, EngineRegistry
from lumenrl.engine.training.fsdp_backend import FSDP2Backend
from lumenrl.engine.training.fsdp_engine import FSDP2Engine, FSDP2EngineWithLMHead
from lumenrl.engine.training.megatron_lumen_dsv4_engine import (
    MegatronLumenDSV4Engine,
    MegatronLumenDSV4EngineWithLMHead,
    MegatronLumenDSV4EngineWithValueHead,
)
from lumenrl.engine.training.megatron_engine import (
    MegatronEngine,
    MegatronEngineWithLMHead,
    MegatronEngineWithValueHead,
)
from lumenrl.engine.training.megatron_native_engine import (
    MegatronNativeEngine,
    MegatronNativeEngineWithLMHead,
    MegatronNativeEngineWithValueHead,
)

__all__ = [
    "BaseEngine",
    "BaseEngineCtx",
    "EngineRegistry",
    "FSDP2Backend",
    "FSDP2Engine",
    "FSDP2EngineWithLMHead",
    "MegatronLumenDSV4Engine",
    "MegatronLumenDSV4EngineWithLMHead",
    "MegatronLumenDSV4EngineWithValueHead",
    "MegatronEngine",
    "MegatronEngineWithLMHead",
    "MegatronEngineWithValueHead",
    "MegatronNativeEngine",
    "MegatronNativeEngineWithLMHead",
    "MegatronNativeEngineWithValueHead",
]
