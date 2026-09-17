"""Mixture-of-experts utilities: R3 routing, expert parallel, diagnostics."""

from __future__ import annotations

from lumenrl.moe.expert_parallel import ExpertParallelManager
from lumenrl.moe.moe_utils import (
    check_expert_utilization,
    compute_load_balance_loss,
    compute_router_entropy,
    iter_megatron_routers,
    megatron_record_router_logits,
    megatron_replay_router_logits,
)
from lumenrl.moe.r3_manager import R3Manager
from lumenrl.moe.r3_scope import (
    SUPPORTED_R3_GENERATION_BACKEND,
    SUPPORTED_R3_REPLAY_MODE,
    assert_supported_r3,
    r3_fields_from_config,
)
from lumenrl.moe.router_precision import enable_fp32_moe_router, fp32_router_enabled
from lumenrl.moe.router_recorder import RouterRecorder
from lumenrl.moe.router_replayer import RouterReplayer

__all__ = [
    "ExpertParallelManager",
    "R3Manager",
    "SUPPORTED_R3_GENERATION_BACKEND",
    "SUPPORTED_R3_REPLAY_MODE",
    "RouterRecorder",
    "RouterReplayer",
    "assert_supported_r3",
    "r3_fields_from_config",
    "check_expert_utilization",
    "compute_load_balance_loss",
    "compute_router_entropy",
    "enable_fp32_moe_router",
    "fp32_router_enabled",
    "iter_megatron_routers",
    "megatron_record_router_logits",
    "megatron_replay_router_logits",
]
