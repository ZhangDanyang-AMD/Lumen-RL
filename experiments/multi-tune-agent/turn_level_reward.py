"""Turn-level reward computation for multi-turn kernel agent episodes.

Assigns per-turn rewards based on sandbox evaluation results at each step,
compatible with the TRLOO advantage estimator.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor

logger = logging.getLogger(__name__)

# Per-action reward signals
_ACTION_REWARDS: dict[str, float] = {
    "list_files": 0.0,
    "read_file": 0.0,
    "write_file": 0.0,
    "state": 0.0,
}

_EVAL_MODE_REWARDS: dict[str, dict[str, float]] = {
    "compile": {"ok": 0.1, "fail": -0.5},
    "correctness": {"ok": 0.3, "fail": -1.0},
    "performance": {"ok": 0.0, "fail": -0.5},
    "full": {"ok": 0.0, "fail": -1.0},
}


def compute_turn_reward(
    action: str,
    tool_result: Mapping[str, Any],
    evaluation: Mapping[str, Any] | None = None,
    previous_best_speedup: float = 1.0,
) -> float:
    """Compute reward for a single turn based on the action and result.

    For non-evaluate actions, returns a small fixed reward.
    For evaluate actions, the reward depends on the evaluation mode and outcome.
    """
    if action in _ACTION_REWARDS:
        return _ACTION_REWARDS[action]

    if action != "evaluate":
        return 0.0

    mode = str(tool_result.get("mode") or "full")
    ok = bool(tool_result.get("ok", False))
    mode_rewards = _EVAL_MODE_REWARDS.get(mode, _EVAL_MODE_REWARDS["full"])

    if not ok:
        return mode_rewards["fail"]

    if mode in ("compile", "correctness"):
        return mode_rewards["ok"]

    # Performance or full evaluation: speedup-based reward
    eval_data = evaluation or tool_result.get("evaluation") or tool_result
    speedup = float(eval_data.get("speedup_geomean") or 0.0)
    correct = bool(eval_data.get("correct"))

    if not correct:
        return -1.0

    import math

    perf = max(-1.0, min(math.log(3.0), math.log(max(speedup, 1e-9))))
    improvement = max(0.0, speedup - previous_best_speedup)
    return 1.0 + perf + 0.5 * improvement


def compute_episode_turn_rewards(
    actions: Sequence[str],
    tool_results: Sequence[Mapping[str, Any]],
    evaluations: Sequence[Mapping[str, Any] | None] | None = None,
    initial_best_speedup: float = 1.0,
) -> list[float]:
    """Compute per-turn rewards for an entire episode."""
    evals = evaluations or [None] * len(actions)
    rewards: list[float] = []
    best_speedup = initial_best_speedup

    for action, result, eval_data in zip(actions, tool_results, evals):
        r = compute_turn_reward(action, result, eval_data, best_speedup)
        rewards.append(r)

        if action == "evaluate" and result.get("ok"):
            e = eval_data or result.get("evaluation") or result
            speedup = float(e.get("speedup_geomean") or 0.0)
            if e.get("correct") and speedup > best_speedup:
                best_speedup = speedup

    return rewards


def build_turn_reward_tensor(
    episode_turn_rewards: Sequence[Sequence[float]],
    max_turns: int = 20,
) -> Tensor:
    """Convert per-episode turn reward lists into a padded tensor [B, max_turns]."""
    B = len(episode_turn_rewards)
    tensor = torch.zeros(B, max_turns, dtype=torch.float32)
    for i, tr in enumerate(episode_turn_rewards):
        n = min(len(tr), max_turns)
        for t in range(n):
            tensor[i, t] = tr[t]
    return tensor
