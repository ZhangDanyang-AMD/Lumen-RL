from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest

from lumenrl.agentic.core import Episode, ToolCall, TrajectoryStore, build_environment
from lumenrl.agentic.core.trajectory import PolicyTurn
from lumenrl.agentic.credit import assign_turn_advantages, discounted_returns, leave_one_out_baselines
from lumenrl.agentic.envs import mock  # noqa: F401  (registers "mock")
from lumenrl.agentic.rollout import AgenticRolloutEngine, AgentPolicy, RolloutConfig, ScriptedPolicy

AGENTIC_ROOT = Path(__file__).resolve().parents[3] / "lumenrl" / "agentic"
TASK = {"low": 0, "high": 10, "target": 7}


def _guess(value: int) -> dict[str, Any]:
    return {"name": "submit_guess", "arguments": {"value": value}}


def _run(policy: AgentPolicy, max_turns: int = 6, nudges: int = 0, group_id: str = "g") -> Episode:
    env = build_environment("mock", max_submissions=4)
    engine = AgenticRolloutEngine(policy, RolloutConfig(max_turns=max_turns, max_no_tool_nudges=nudges))
    return asyncio.run(engine.run_episode(env, TASK, group_id=group_id))


def test_tool_call_parsing_fails_closed() -> None:
    assert ToolCall.from_raw("c", "t", '{"a": 1}').arguments == {"a": 1}
    assert ToolCall.from_raw("c", "t", "{not json").arguments is None
    assert ToolCall.from_raw("c", "t", "[1, 2]").arguments is None


def test_scripted_episode_reaches_success() -> None:
    episode = _run(ScriptedPolicy([_guess(3), _guess(9), _guess(7)]))
    assert episode.terminal_state == "SUCCESS"
    assert [s.result.execution_result for s in episode.steps] == ["wrong", "wrong", "correct"]
    assert episode.step_rewards == pytest.approx([-0.1, -0.1, 1.0])
    roles = [m["role"] for m in episode.messages]
    assert roles == ["system", "user", "assistant", "tool", "assistant", "tool", "assistant", "tool"]
    assert json.loads(episode.messages[3]["content"])["result"] == "higher"


def test_budget_exhaustion_and_finish() -> None:
    episode = _run(ScriptedPolicy([_guess(1)] * 4))
    assert episode.terminal_state == "BUDGET_EXHAUSTED"
    finished = _run(ScriptedPolicy([{"name": "finish", "arguments": {}}]))
    assert finished.terminal_state == "AGENT_FINISHED"


def test_no_tool_call_nudge_then_stop() -> None:
    episode = _run(ScriptedPolicy([]), nudges=1)
    assert episode.terminal_state == "AGENT_FINISHED"
    assert len(episode.turns) == 2
    assert episode.messages[-2]["role"] == "user"


class _ParallelPolicy(AgentPolicy):
    async def act(self, messages: Sequence[Mapping[str, Any]], tools: Sequence[Mapping[str, Any]], turn_index: int) -> PolicyTurn:
        calls = [ToolCall.from_raw("a", "submit_guess", '{"value": 7}'), ToolCall.from_raw("b", "submit_guess", '{"value": 1}')]
        return PolicyTurn(turn_index, "parallel", "", None, calls, "tool_calls")


def test_parallel_tool_calls_only_first_executed() -> None:
    episode = _run(_ParallelPolicy())
    assert len(episode.steps) == 1
    tool_messages = [m for m in episode.messages if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_messages] == ["a", "b"]
    assert "PARALLEL_TOOL_CALL_REJECTED" in tool_messages[1]["content"]


def test_snapshot_restore_roundtrip() -> None:
    async def scenario() -> tuple[str, str]:
        env = build_environment("mock", max_submissions=4)
        await env.reset(TASK)
        ref = await env.snapshot()
        await env.step(ToolCall.from_raw("x", "submit_guess", '{"value": 7}'))
        await env.restore(ref)
        again = await env.step(ToolCall.from_raw("y", "submit_guess", '{"value": 7}'))
        return again.terminal_state, ref

    terminal, _ = asyncio.run(scenario())
    assert terminal == "SUCCESS"


def test_credit_assignment() -> None:
    assert discounted_returns([1.0, 0.0, 2.0], 1.0, terminal_reward=1.0) == [4.0, 3.0, 3.0]
    assert discounted_returns([1.0, 1.0], 0.5) == [1.5, 1.0]
    assert leave_one_out_baselines([1.0, 3.0]) == [3.0, 1.0]
    assert leave_one_out_baselines([5.0]) == [0.0]

    good = _run(ScriptedPolicy([_guess(7)]), group_id="g1")
    bad = _run(ScriptedPolicy([_guess(1)] * 4), group_id="g1")
    advantages = assign_turn_advantages([good, bad], gamma=1.0)
    assert advantages[good.episode_id] == pytest.approx([1.0 - (-0.4)])
    assert advantages[bad.episode_id][0] == pytest.approx(-0.4 - 1.0)


def test_trajectory_store_roundtrip(tmp_path: Path) -> None:
    episode = _run(ScriptedPolicy([_guess(7)]))
    store = TrajectoryStore(tmp_path / "e.jsonl")
    store.append(episode)
    loaded = store.load()
    assert loaded[0]["schema_version"] == "agent_episode_v1"
    assert loaded[0]["terminal_state"] == "SUCCESS"


def test_core_is_backend_neutral() -> None:
    forbidden = ("micake", "geak", "verltool", "rllm", "sandbox")
    for path in AGENTIC_ROOT.rglob("*.py"):
        for line in path.read_text().splitlines():
            stripped = line.strip()
            if stripped.startswith(("import ", "from ")):
                assert not any(name in stripped.lower() for name in forbidden), f"{path}: {stripped}"
