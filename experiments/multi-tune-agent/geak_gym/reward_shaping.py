"""Configurable reward shaping on top of the base GEAK reward."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass
class RewardShapingConfig:
    correctness_bonus: float = 1.0
    correctness_penalty: float = -1.0
    performance_scale: float = 1.0
    improvement_bonus_weight: float = 0.5
    max_performance_reward: float = 3.0
    compile_fail_penalty: float = -0.5
    step_penalty: float = -0.01
    timeout_penalty: float = -2.0


def shape_reward(
    evaluation: Mapping[str, Any],
    previous_best: float,
    turn_index: int,
    config: RewardShapingConfig | None = None,
) -> float:
    """Compute a shaped scalar reward from a GEAK evaluation result.

    Mirrors ``GEAKToolEnvironment.reward()`` (geak_tool.py:321) but adds
    configurable shaping for RL training.
    """
    cfg = config or RewardShapingConfig()
    compiled = bool(evaluation.get("compiled"))
    correct = bool(evaluation.get("correct"))
    speedup = float(evaluation.get("speedup_geomean") or 0.0)

    if not compiled:
        return cfg.compile_fail_penalty + cfg.step_penalty * turn_index

    if not correct:
        return cfg.correctness_penalty + cfg.step_penalty * turn_index

    perf = min(
        cfg.max_performance_reward,
        cfg.performance_scale * max(-1.0, math.log(max(speedup, 1e-9))),
    )
    improvement = max(0.0, speedup - previous_best)

    total = (
        cfg.correctness_bonus
        + perf
        + cfg.improvement_bonus_weight * improvement
        + cfg.step_penalty * turn_index
    )
    return total


def shape_step_reward(
    tool_result: Mapping[str, Any],
    turn_index: int,
    config: RewardShapingConfig | None = None,
) -> float:
    """Assign a small intermediate reward for non-evaluate tool calls."""
    cfg = config or RewardShapingConfig()
    if not tool_result.get("ok", False):
        return -0.1 + cfg.step_penalty * turn_index
    return cfg.step_penalty * turn_index
