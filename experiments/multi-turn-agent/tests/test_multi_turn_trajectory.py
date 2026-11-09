"""Unit tests for multi-turn trajectory management."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class TestTurn:
    def test_turn_creation(self):
        from multi_turn_trajectory import Turn

        t = Turn(turn_index=0, action_text='{"action":"read_file"}', observation_text='{"ok":true}', step_reward=0.5)
        assert t.turn_index == 0
        assert t.step_reward == 0.5


class TestEpisode:
    def test_episode_properties(self):
        from multi_turn_trajectory import Episode, Turn

        turns = [
            Turn(0, "a1", "o1", 0.1),
            Turn(1, "a2", "o2", 0.5),
            Turn(2, "a3", "o3", 1.0),
        ]
        ep = Episode("ep1", "task1", turns=turns, final_reward=1.5)
        assert ep.num_turns == 3
        assert ep.turn_rewards == [0.1, 0.5, 1.0]

    def test_to_dict(self):
        from multi_turn_trajectory import Episode, Turn

        ep = Episode("ep1", "task1", turns=[Turn(0, "a", "o", 0.5)], final_reward=1.0)
        d = ep.to_dict()
        assert d["episode_id"] == "ep1"
        assert len(d["turns"]) == 1


class TestTrajectoryCollector:
    def test_collect_and_finalize(self):
        from multi_turn_trajectory import TrajectoryCollector

        collector = TrajectoryCollector("ep1", "task1")
        collector.add_turn("action1", "obs1", step_reward=0.1)
        collector.add_turn("action2", "obs2", step_reward=0.5, done=True)

        ep = collector.finalize(final_reward=1.0)
        assert ep.num_turns == 2
        assert ep.final_reward == 1.0
        assert ep.turns[1].done is True


class TestLoadEpisodes:
    def test_load_from_jsonl(self):
        from multi_turn_trajectory import load_episodes_from_jsonl

        records = [
            {"event": "run_start", "timestamp": 1000, "payload": {"case": {"case_id": "gemm_01"}}},
            {"event": "tool_result", "timestamp": 1001, "payload": {
                "parameters": {"action": "read_file", "path": "a.py"},
                "result": {"ok": True, "content": "x"},
                "reward": {"total": 0.0},
            }},
            {"event": "tool_result", "timestamp": 1002, "payload": {
                "parameters": {"action": "evaluate", "mode": "full"},
                "result": {"ok": True, "evaluation": {"speedup_geomean": 1.5}},
                "reward": {"total": 1.5},
            }},
            {"event": "run_end", "timestamp": 1003, "payload": {
                "status": "success",
                "final_reward": {"total": 1.5},
            }},
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
            path = f.name

        episodes = load_episodes_from_jsonl(path)
        assert len(episodes) == 1
        assert episodes[0].task_id == "gemm_01"
        assert episodes[0].num_turns == 2
        assert episodes[0].final_reward == 1.5
        Path(path).unlink()


class TestEpisodesToTensors:
    def test_tensor_shapes(self):
        from multi_turn_trajectory import Episode, Turn, episodes_to_tensors

        episodes = [
            Episode("ep1", "t1", [Turn(0, "a", "o", 0.1), Turn(1, "a", "o", 0.5)], 1.0),
            Episode("ep2", "t2", [Turn(0, "a", "o", 0.3)], 0.5),
        ]
        tensors = episodes_to_tensors(episodes, max_turns=5)
        assert tensors["rewards"].shape == (2,)
        assert tensors["turn_rewards"].shape == (2, 5)
        assert tensors["num_turns"].tolist() == [2, 1]
        assert tensors["turn_rewards"][0, 0].item() == pytest.approx(0.1)
        assert tensors["turn_rewards"][0, 1].item() == pytest.approx(0.5)
