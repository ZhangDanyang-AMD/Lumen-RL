"""Unit tests for multi-turn rollout manager."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from multi_turn_rollout import (
    MultiTurnRolloutManager,
    RolloutEpisode,
    RolloutTurn,
    concat_turns_for_training,
)


class TestRolloutEpisode:
    def test_total_tokens(self):
        ep = RolloutEpisode(
            episode_id="ep1",
            task_id="task1",
            prompt_token_ids=torch.tensor([1, 2, 3]),
            turns=[
                RolloutTurn(0, torch.tensor([4, 5]), torch.tensor([0.1, 0.2])),
                RolloutTurn(1, torch.tensor([6, 7, 8]), torch.tensor([0.3, 0.4, 0.5])),
            ],
        )
        assert ep.total_tokens() == 8  # 3 prompt + 2 + 3 action tokens
        assert ep.num_turns == 2


class TestConcatTurnsForTraining:
    def test_shapes(self):
        episodes = [
            RolloutEpisode(
                "ep1", "t1",
                prompt_token_ids=torch.tensor([1, 2]),
                turns=[
                    RolloutTurn(0, torch.tensor([3, 4]), torch.tensor([-0.1, -0.2]), reward=0.5),
                    RolloutTurn(1, torch.tensor([5]), torch.tensor([-0.3]), reward=1.0, done=True),
                ],
                final_reward=1.0,
            ),
        ]
        result = concat_turns_for_training(episodes, max_seq_len=16, max_turns=5)

        assert result["input_ids"].shape == (1, 16)
        assert result["log_probs"].shape == (1, 16)
        assert result["response_mask"].shape == (1, 16)
        assert result["turn_ids"].shape == (1, 16)
        assert result["rewards"].shape == (1,)
        assert result["turn_rewards"].shape == (1, 5)

    def test_prompt_not_in_response_mask(self):
        episodes = [
            RolloutEpisode(
                "ep1", "t1",
                prompt_token_ids=torch.tensor([1, 2, 3]),
                turns=[
                    RolloutTurn(0, torch.tensor([4, 5]), torch.tensor([-0.1, -0.2])),
                ],
            ),
        ]
        result = concat_turns_for_training(episodes, max_seq_len=10)
        # Prompt positions (0,1,2) should have mask=0
        assert result["response_mask"][0, 0].item() == 0
        assert result["response_mask"][0, 1].item() == 0
        assert result["response_mask"][0, 2].item() == 0
        # Action positions (3,4) should have mask=1
        assert result["response_mask"][0, 3].item() == 1
        assert result["response_mask"][0, 4].item() == 1

    def test_turn_ids_correct(self):
        episodes = [
            RolloutEpisode(
                "ep1", "t1",
                prompt_token_ids=torch.tensor([1]),
                turns=[
                    RolloutTurn(0, torch.tensor([2, 3]), torch.tensor([0.0, 0.0])),
                    RolloutTurn(1, torch.tensor([4]), torch.tensor([0.0])),
                ],
            ),
        ]
        result = concat_turns_for_training(episodes, max_seq_len=8)
        # pos 0: prompt → turn_id=0
        # pos 1,2: turn 0 → turn_id=1
        # pos 3: turn 1 → turn_id=2
        assert result["turn_ids"][0, 0].item() == 0
        assert result["turn_ids"][0, 1].item() == 1
        assert result["turn_ids"][0, 2].item() == 1
        assert result["turn_ids"][0, 3].item() == 2

    def test_turn_rewards_populated(self):
        episodes = [
            RolloutEpisode(
                "ep1", "t1",
                prompt_token_ids=torch.tensor([1]),
                turns=[
                    RolloutTurn(0, torch.tensor([2]), torch.tensor([0.0]), reward=0.5),
                    RolloutTurn(1, torch.tensor([3]), torch.tensor([0.0]), reward=1.5),
                ],
                final_reward=2.0,
            ),
        ]
        result = concat_turns_for_training(episodes, max_seq_len=8, max_turns=5)
        assert result["turn_rewards"][0, 0].item() == pytest.approx(0.5)
        assert result["turn_rewards"][0, 1].item() == pytest.approx(1.5)
        assert result["rewards"][0].item() == pytest.approx(2.0)


class TestMultiTurnRolloutManager:
    def test_rollout_with_mock(self):
        # Mock tokenizer
        tokenizer = MagicMock()
        tokenizer.encode.return_value = [1, 2, 3]
        tokenizer.decode.return_value = '{"tool_name": "evaluate", "arguments": {"mode": "full"}}'

        # Mock generate_fn: returns 3 action tokens
        def generate_fn(context_ids):
            T = 3
            return torch.tensor([[10, 11, 12]]), torch.tensor([[-0.1, -0.2, -0.3]])

        # Mock env
        from geak_gym.spaces import GEAKObservation
        mock_env = MagicMock()
        mock_obs = GEAKObservation(
            tool_result={"ok": True, "event": "reset", "task_id": "task1"},
            allowed_write_paths=["a.py"],
            baseline_ms={"c": 1.0},
            current_speedup=1.0,
            turn_index=0,
            done=False,
        )
        mock_env.reset.return_value = (mock_obs, {"task_id": "task1"})

        step_obs = GEAKObservation(
            tool_result={"ok": True, "evaluation": {"speedup_geomean": 1.5}},
            allowed_write_paths=["a.py"],
            baseline_ms={"c": 1.0},
            current_speedup=1.5,
            turn_index=1,
            done=True,
        )
        mock_env.step.return_value = (step_obs, 1.5, True, False, {"turn": 1})

        manager = MultiTurnRolloutManager(
            generate_fn=generate_fn,
            tokenizer=tokenizer,
            env=mock_env,
            max_turns=5,
            max_seq_len=64,
        )
        episode = manager.rollout("Optimize this kernel")

        assert episode.task_id == "task1"
        assert episode.num_turns == 1
        assert episode.final_reward == 1.5
