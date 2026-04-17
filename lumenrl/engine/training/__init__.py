"""Training backends and weight synchronization."""

from __future__ import annotations

from lumenrl.engine.training.fsdp_backend import FSDP2Backend
from lumenrl.engine.training.megatron_backend import MegatronBackend
from lumenrl.engine.training.weight_sync import WeightSyncManager

__all__ = ["FSDP2Backend", "MegatronBackend", "WeightSyncManager"]
