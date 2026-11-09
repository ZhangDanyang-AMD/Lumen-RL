"""Unit tests for the GEAK Gymnasium-compatible environment."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class TestGEAKSpaces:
    """Tests for action/observation parsing."""

    def test_parse_action_from_dict(self):
        from geak_gym.spaces import parse_action

        action = parse_action({"tool_name": "write_file", "arguments": {"path": "a.py", "content": "x"}})
        assert action.tool_name == "write_file"
        assert action.arguments["path"] == "a.py"

    def test_parse_action_from_json_string(self):
        import json
        from geak_gym.spaces import parse_action

        raw = json.dumps({"action": "read_file", "params": {"path": "b.py"}})
        action = parse_action(raw)
        assert action.tool_name == "read_file"

    def test_parse_action_unknown_tool_fallback(self):
        from geak_gym.spaces import parse_action

        action = parse_action({"tool_name": "unknown_tool"})
        assert action.tool_name == "evaluate"

    def test_parse_action_invalid_string_fallback(self):
        from geak_gym.spaces import parse_action

        action = parse_action("not json at all")
        assert action.tool_name == "evaluate"

    def test_to_sandbox_params(self):
        from geak_gym.spaces import GEAKAction

        action = GEAKAction("read_file", {"path": "x.py"})
        params = action.to_sandbox_params()
        assert params["action"] == "read_file"
        assert params["path"] == "x.py"

    def test_observation_to_text(self):
        from geak_gym.spaces import GEAKObservation

        obs = GEAKObservation(
            tool_result={"ok": True},
            allowed_write_paths=["a.py"],
            baseline_ms={"case_0": 1.0},
            current_speedup=1.5,
            turn_index=3,
            done=False,
        )
        text = obs.to_text()
        assert "1.5" in text
        assert "ok" in text


class TestRewardShaping:
    """Tests for reward shaping."""

    def test_correct_gets_positive_reward(self):
        from geak_gym.reward_shaping import shape_reward

        r = shape_reward(
            {"compiled": True, "correct": True, "speedup_geomean": 1.5},
            previous_best=1.0,
            turn_index=1,
        )
        assert r > 0

    def test_compile_fail_gets_penalty(self):
        from geak_gym.reward_shaping import shape_reward

        r = shape_reward(
            {"compiled": False, "correct": False, "speedup_geomean": 0.0},
            previous_best=1.0,
            turn_index=1,
        )
        assert r < 0

    def test_step_penalty_increases_with_turns(self):
        from geak_gym.reward_shaping import shape_step_reward

        r1 = shape_step_reward({"ok": True}, turn_index=1)
        r5 = shape_step_reward({"ok": True}, turn_index=5)
        assert r5 < r1


class TestGEAKGymEnv:
    """Tests for the GEAKGymEnv lifecycle with mocked sandbox."""

    def test_env_step_returns_five_tuple(self):
        from geak_gym.env import GEAKGymEnv

        mock_task = MagicMock()
        mock_task.task_id = "test_task"
        mock_task.task_type = "gemm"
        mock_task.kernel_path = Path("/tmp/test_kernel")

        env = GEAKGymEnv(
            tasks=[mock_task],
            upstream_root="/tmp",
            max_turns=5,
        )

        # Mock the sandbox
        mock_sandbox = MagicMock()
        mock_sandbox.allowed_write_paths = ["kernel_src/main.hip"]
        mock_sandbox.baseline_ms = {"case_0": 1.0}
        mock_sandbox.establish_baseline.return_value = {"per_case_ms": {"case_0": 1.0}}
        mock_sandbox.execute_tool.return_value = {"ok": True, "content": "test"}
        mock_sandbox.prepare.return_value = Path("/tmp/workspace")

        with patch("geak_gym.env.KernelSandbox", return_value=mock_sandbox):
            obs, info = env.reset()
            assert obs.done is False
            assert info["task_id"] == "test_task"

            obs, reward, terminated, truncated, step_info = env.step(
                {"tool_name": "read_file", "arguments": {"path": "a.py"}}
            )
            assert isinstance(reward, float)
            assert isinstance(terminated, bool)
            assert isinstance(truncated, bool)

        env.close()
