"""Turn-level returns and group leave-one-out advantages (TRLOO-style)."""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

from lumenrl.agentic.core.trajectory import Episode


def discounted_returns(rewards: Sequence[float], gamma: float, terminal_reward: float = 0.0) -> list[float]:
    """Return-to-go for each step; ``terminal_reward`` is credited after the last step."""
    returns = [0.0] * len(rewards)
    running = terminal_reward
    for index in range(len(rewards) - 1, -1, -1):
        running = rewards[index] + gamma * running
        returns[index] = running
    return returns


def leave_one_out_baselines(values: Sequence[float]) -> list[float]:
    """Mean of the other group members' values; 0 for singleton groups."""
    if len(values) < 2:
        return [0.0] * len(values)
    total = sum(values)
    return [(total - value) / (len(values) - 1) for value in values]


def assign_turn_advantages(episodes: Sequence[Episode], gamma: float) -> dict[str, list[float]]:
    """Per-step advantages: step return-to-go minus the group leave-one-out baseline.

    Episodes sharing ``group_id`` were sampled on the same task. The baseline
    for an episode is the mean total return of the other episodes in its group.

    Returns:
        Mapping ``episode_id -> advantages`` aligned with ``episode.steps``.
    """
    groups: dict[str, list[Episode]] = defaultdict(list)
    for episode in episodes:
        groups[episode.group_id].append(episode)

    advantages: dict[str, list[float]] = {}
    for members in groups.values():
        totals = [
            discounted_returns(ep.step_rewards, gamma, ep.terminal_reward)[0] if ep.steps else ep.terminal_reward
            for ep in members
        ]
        for episode, baseline in zip(members, leave_one_out_baselines(totals)):
            step_returns = discounted_returns(episode.step_rewards, gamma, episode.terminal_reward)
            advantages[episode.episode_id] = [value - baseline for value in step_returns]
    return advantages
