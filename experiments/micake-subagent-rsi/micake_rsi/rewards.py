"""Hierarchical specialist rewards: local turn rewards + lambda * global episode reward.

* local: the child's own step rewards (MICake correctness-gated rewards for
  its submissions, diagnosis validity / probe outcome for the Critic).
* global: clipped log of the Manager episode's final best speedup, shared by
  every specialist that took part in the episode.
* contribution: clipped log of how much this child moved the shared best.

Advantages are turn-level: return-to-go of local rewards with the episode
terms credited at the end, minus a leave-one-out baseline over specialist
episodes of the same role on the same task.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from lumenrl.agentic.core.trajectory import Episode
from lumenrl.agentic.credit import discounted_returns, leave_one_out_baselines


@dataclass(frozen=True)
class HierarchicalRewardConfig:
    """Weights for the episode-level terms (values come from YAML)."""

    global_weight: float
    contribution_weight: float
    log_clip: float
    gamma: float

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "HierarchicalRewardConfig":
        return cls(
            global_weight=float(value["global_weight"]),
            contribution_weight=float(value["contribution_weight"]),
            log_clip=float(value["log_clip"]),
            gamma=float(value["gamma"]),
        )


@dataclass(frozen=True)
class SpecialistCredit:
    """Reward decomposition and per-step advantages of one specialist episode."""

    episode_id: str
    role: str
    local: float
    global_: float
    contribution: float
    total: float
    step_returns: list[float]
    advantages: list[float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "local": self.local,
            "global": self.global_,
            "contribution": self.contribution,
            "total": self.total,
        }


def _clipped_log(ratio: float, clip: float) -> float:
    return max(-clip, min(clip, math.log(ratio))) if ratio > 0 else -clip


def global_reward(manager: Episode, clip: float) -> float:
    """Clipped log best speedup of the Manager episode (0 when the baseline was never beaten).

    The independently re-measured speedup is used when the generation was verified.
    """
    summary = manager.environment.get("final_summary") or {}
    if summary.get("best_source") != "agent":
        return 0.0
    return _clipped_log(float(summary.get("best_speedup_verified", summary["best_speedup"])), clip)


def contribution(child: Episode, clip: float) -> float:
    """Clipped log of the shared-best improvement produced during the child episode."""
    summary = child.environment.get("final_summary") or {}
    before = summary.get("best_speedup_before") or 1.0
    after = summary.get("best_speedup_after") or before
    return _clipped_log(after / before, clip)


def assign_specialist_credit(
    manager_episodes: Sequence[Episode],
    children: Mapping[str, Sequence[Episode]],
    cfg: HierarchicalRewardConfig,
) -> dict[str, SpecialistCredit]:
    """Credit every specialist episode.

    Args:
        manager_episodes: Manager episodes of one generation.
        children: ``manager_episode_id -> child episodes``.
        cfg: Reward weights.

    Returns:
        ``child_episode_id -> SpecialistCredit``.
    """
    rows: list[tuple[Episode, str, float, float, float, list[float]]] = []
    for manager in manager_episodes:
        g = global_reward(manager, cfg.log_clip)
        task_id = manager.task.get("task_id", "")
        for child in children.get(manager.episode_id, ()):
            c = contribution(child, cfg.log_clip)
            terminal = cfg.global_weight * g + cfg.contribution_weight * c
            returns = discounted_returns(child.step_rewards, cfg.gamma, terminal)
            rows.append((child, task_id, g, c, terminal, returns))

    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, (child, task_id, *_rest) in enumerate(rows):
        groups[(child.role or "", task_id)].append(index)

    credits: dict[str, SpecialistCredit] = {}
    for members in groups.values():
        totals = [rows[i][5][0] if rows[i][5] else rows[i][4] for i in members]
        for index, baseline in zip(members, leave_one_out_baselines(totals)):
            child, _task, g, c, terminal, returns = rows[index]
            local = sum(child.step_rewards)
            credits[child.episode_id] = SpecialistCredit(
                episode_id=child.episode_id,
                role=child.role or "",
                local=local,
                global_=g,
                contribution=c,
                total=local + terminal,
                step_returns=returns,
                advantages=[value - baseline for value in returns],
            )
    return credits
