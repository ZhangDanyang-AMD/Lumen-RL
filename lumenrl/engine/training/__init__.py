"""Training backends and weight synchronization."""

from __future__ import annotations

from lumenrl.engine.training.fsdp_backend import FSDP2Backend
from lumenrl.engine.training.megatron_backend import MegatronBackend

try:
    from lumenrl.engine.training.weight_sync import WeightSyncManager
except ModuleNotFoundError:  # Optional in minimal/unit-test environments.
    WeightSyncManager = None

__all__ = ["FSDP2Backend", "MegatronBackend", "WeightSyncManager"]
