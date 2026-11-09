"""Unit tests for turn-level reward computation."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from turn_level_reward import (
    build_turn_reward_tensor,
    compute_episode_turn_rewards,
    compute_turn_reward,
)


class TestComputeTurnReward:
    def test_read_file_zero_reward(self):
        r = compute_turn_reward("read_file", {"ok": True})
        assert r == 0.0

    def test_write_file_zero_reward(self):
        r = compute_turn_reward("write_file", {"ok": True})
        assert r == 0.0

    def test_compile_ok(self):
        r = compute_turn_reward("evaluate", {"ok": True, "mode": "compile"})
        assert r == pytest.approx(0.1)

    def test_compile_fail(self):
        r = compute_turn_reward("evaluate", {"ok": False, "mode": "compile"})
        assert r == pytest.approx(-0.5)

    def test_correctness_ok(self):
        r = compute_turn_reward("evaluate", {"ok": True, "mode": "correctness"})
        assert r == pytest.approx(0.3)

    def test_correctness_fail(self):
        r = compute_turn_reward("evaluate", {"ok": False, "mode": "correctness"})
        assert r == pytest.approx(-1.0)

    def test_full_eval_correct_with_speedup(self):
        r = compute_turn_reward(
            "evaluate",
            {"ok": True, "mode": "full"},
            evaluation={"correct": True, "speedup_geomean": 2.0},
            previous_best_speedup=1.0,
        )
        assert r > 1.0  # correctness + perf + improvement

    def test_full_eval_incorrect(self):
        r = compute_turn_reward(
            "evaluate",
            {"ok": True, "mode": "full"},
            evaluation={"correct": False, "speedup_geomean": 0.0},
        )
        assert r == pytest.approx(-1.0)


class TestComputeEpisodeTurnRewards:
    def test_episode_rewards(self):
        actions = ["read_file", "write_file", "evaluate", "evaluate"]
        results = [
            {"ok": True},
            {"ok": True},
            {"ok": True, "mode": "compile"},
            {"ok": True, "mode": "full"},
        ]
        evals = [None, None, None, {"correct": True, "speedup_geomean": 1.5}]

        rewards = compute_episode_turn_rewards(actions, results, evals)
        assert len(rewards) == 4
        assert rewards[0] == 0.0  # read_file
        assert rewards[1] == 0.0  # write_file
        assert rewards[2] == pytest.approx(0.1)  # compile ok

    def test_best_speedup_tracking(self):
        actions = ["evaluate", "evaluate"]
        results = [
            {"ok": True, "mode": "full"},
            {"ok": True, "mode": "full"},
        ]
        evals = [
            {"correct": True, "speedup_geomean": 1.5},
            {"correct": True, "speedup_geomean": 2.0},
        ]
        rewards = compute_episode_turn_rewards(actions, results, evals, initial_best_speedup=1.0)
        # Second eval should use updated best_speedup=1.5
        assert rewards[1] > 0


class TestBuildTurnRewardTensor:
    def test_shapes(self):
        episode_rewards = [
            [0.1, 0.5, 1.0],
            [0.2, 0.3],
        ]
        tensor = build_turn_reward_tensor(episode_rewards, max_turns=5)
        assert tensor.shape == (2, 5)
        assert tensor[0, 0].item() == pytest.approx(0.1)
        assert tensor[0, 2].item() == pytest.approx(1.0)
        assert tensor[1, 2].item() == pytest.approx(0.0)  # padded
