"""Mixture-of-experts utilities: R3 routing, expert parallel, diagnostics."""

from __future__ import annotations

from lumenrl.moe.expert_parallel import ExpertParallelManager
from lumenrl.moe.moe_utils import (
    check_expert_utilization,
    compute_load_balance_loss,
    compute_router_entropy,
)
from lumenrl.moe.r3_manager import R3Manager
from lumenrl.moe.router_recorder import RouterRecorder
from lumenrl.moe.router_replayer import RouterReplayer

__all__ = [
    "ExpertParallelManager",
    "R3Manager",
    "RouterRecorder",
    "RouterReplayer",
    "check_expert_utilization",
    "compute_load_balance_loss",
    "compute_router_entropy",
]
