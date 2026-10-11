"""Kernel optimization reward — delegates to sandbox package.

Tries sandbox.reward.sandbox_reward_batch first (the pluggable backend),
falls back to the multi-tune-agent implementation if sandbox is not installed.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import torch


def compute_kernel_reward(
    responses: Sequence[str],
    ground_truths: Sequence[str],
) -> torch.Tensor:
    try:
        from sandbox.reward import sandbox_reward_batch
        return sandbox_reward_batch(responses, ground_truths, backend_name="geak")
    except ImportError:
        pass

    _MTA = Path(__file__).resolve().parents[2] / "experiments" / "multi-tune-agent"
    if str(_MTA) not in sys.path:
        sys.path.insert(0, str(_MTA))
    from rewards.kernel_reward import kernel_reward_batch
    return kernel_reward_batch(responses, ground_truths)
