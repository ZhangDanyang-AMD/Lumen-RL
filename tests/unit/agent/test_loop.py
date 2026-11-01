"""ToolAgentLoop driven by a scripted engine and fake tools."""

from __future__ import annotations

import asyncio

import pytest

from lumenrl.agent.chat import ChatTemplateTokenizer
from lumenrl.agent.loop import ToolAgentLoop, truncate_tool_response
from lumenrl.agent.tools import ToolResult
from tests.unit.agent.fakes import IM_END, ChatMLTokenizer, tool_call_text

PROMPT = [{"role": "user", "content": "What is 6*7?"}]


class EchoTool:
    """Returns its ``code`` argument; ``fail`` makes every call raise."""

    name = "code_interpreter"

    def __init__(self, *, output=None, fail: bool = False) -> None:
        self.calls: list[dict] = []
        self.output = output
        self.fail = fail

    @property
    def schema(self):
        return {"type": "function", "function": {"name": self.name}}

    async def __call__(self, arguments):
        self.calls.append(arguments)
        if self.fail:
            raise RuntimeError("tool exploded")
        return ToolResult(self.output if self.output is not None else str(arguments["code"]))


class ScriptedEngine:
    """Replays one ``(text, finish_reason)`` turn per generate call."""

    def __init__(self, tok: ChatMLTokenizer, turns: list[tuple[str, str]], logprob: float | None = -0.5):
        self.tok = tok
        self.turns = list(turns)
        self.logprob = logprob
        self.requests: list[tuple[list[int], dict]] = []

    async def generate(self, prompt_ids, sampling_params):
        self.requests.append((list(prompt_ids), dict(sampling_params)))
        text, finish_reason = self.turns.pop(0)
        ids = self.tok.encode(text) + ([IM_END] if finish_reason == "stop" else [])
        ids = ids[: sampling_params["max_tokens"]]
        return {
            "token_ids": ids,
            "logprobs": None if self.logprob is None else [self.logprob] * len(ids),
            "finish_reason": finish_reason,
        }


def call(code: str) -> str:
    return "Let me run it.\n" + tool_call_text("code_interpreter", {"code": code})


def run_loop(turns, *, tool=None, max_response_tokens=4096, logprob=-0.5, **loop_kwargs):
    tok = ChatMLTokenizer()
    tool = tool or EchoTool()
    engine = ScriptedEngine(tok, turns, logprob=logprob)
    loop = ToolAgentLoop(ChatTemplateTokenizer(tok), [tool], **loop_kwargs)
    trajectory = asyncio.run(
        loop.run(engine.generate, PROMPT, {"max_tokens": 10_000, "temperature": 1.0}, max_response_tokens),
    )
    return trajectory, engine, tool, tok


def segments(mask: list[int]) -> list[tuple[int, int]]:
    """Run-length encoding of the response mask."""
    runs: list[tuple[int, int]] = []
    for bit in mask:
        if runs and runs[-1][0] == bit:
            runs[-1] = (bit, runs[-1][1] + 1)
        else:
            runs.append((bit, 1))
    return runs


def test_tool_round_trip_masks_the_tool_output():
    trajectory, _, tool, tok = run_loop([(call("42"), "stop"), ("Answer: 42", "stop")])

    first_turn = tok.encode(call("42")) + [IM_END]
    second_turn = tok.encode("Answer: 42") + [IM_END]
    environment_len = len(trajectory.response_ids) - len(first_turn) - len(second_turn)
    assert tool.calls == [{"code": "42"}]
    assert trajectory.stop_reason == "answer"
    assert trajectory.assistant_turns == 2
    assert trajectory.tool_rounds == 1
    assert trajectory.tool_calls == 1
    assert segments(trajectory.response_mask) == [
        (1, len(first_turn)), (0, environment_len), (1, len(second_turn)),
    ]
    assert tok.decode(trajectory.response_ids[len(first_turn):len(first_turn) + environment_len]) == (
        "\n<|im_start|>user\n<tool_response>\n42\n</tool_response><|im_end|>\n<|im_start|>assistant\n"
    )
    assert trajectory.response_logprobs == [
        -0.5 if bit else 0.0 for bit in trajectory.response_mask
    ]


def test_the_engine_sees_exactly_the_tokens_that_are_trained_on():
    trajectory, engine, _, _ = run_loop([(call("1"), "stop"), (call("2"), "stop"), ("done", "stop")])

    mask = trajectory.response_mask
    turn_starts = [i for i, bit in enumerate(mask) if bit and (i == 0 or not mask[i - 1])]
    assert len(turn_starts) == len(engine.requests) == 3
    for start, (prompt_ids, _) in zip(turn_starts, engine.requests):
        assert prompt_ids == trajectory.prompt_ids + trajectory.response_ids[:start]


def test_each_turn_gets_the_rest_of_the_response_budget():
    trajectory, engine, _, _ = run_loop(
        [(call("1"), "stop"), ("done", "stop")], max_response_tokens=500,
    )

    budgets = [params["max_tokens"] for _, params in engine.requests]
    first_turn_and_tool = len(engine.requests[1][0]) - len(trajectory.prompt_ids)
    assert budgets == [500, 500 - first_turn_and_tool]


def test_the_sequence_limit_caps_the_response_budget():
    tok = ChatMLTokenizer()
    engine = ScriptedEngine(tok, [("done", "stop")])
    loop = ToolAgentLoop(ChatTemplateTokenizer(tok), [EchoTool()])

    trajectory = asyncio.run(loop.run(
        engine.generate, PROMPT, {"max_tokens": 10_000}, max_response_tokens=4096,
        max_sequence_tokens=400,
    ))

    [(_, params)] = engine.requests
    assert params["max_tokens"] == 400 - len(trajectory.prompt_ids)


def test_a_truncated_turn_ends_the_rollout_without_running_its_tools():
    trajectory, _, tool, _ = run_loop([(call("1"), "length")])

    assert trajectory.stop_reason == "length"
    assert tool.calls == []


def test_an_aborted_turn_ends_the_rollout():
    trajectory, _, tool, _ = run_loop([(call("1"), "abort")])

    assert trajectory.stop_reason == "abort"
    assert tool.calls == []


def test_the_turn_limit_ends_a_rollout_that_keeps_calling_tools():
    trajectory, _, tool, _ = run_loop(
        [(call("1"), "stop"), (call("2"), "stop"), (call("3"), "stop")],
        max_assistant_turns=2,
    )

    assert trajectory.stop_reason == "max_turns"
    assert trajectory.assistant_turns == 2
    assert tool.calls == [{"code": "1"}]


def test_tool_output_that_does_not_fit_ends_the_rollout_unappended():
    turn_len = len(ChatMLTokenizer().encode(call("1"))) + 1
    trajectory, _, tool, _ = run_loop(
        [(call("1"), "stop")], tool=EchoTool(output="x" * 200), max_response_tokens=turn_len + 50,
        max_tool_response_length=0,
    )

    assert tool.calls == [{"code": "1"}]
    assert trajectory.stop_reason == "length"
    assert trajectory.response_mask == [1] * turn_len


def test_tool_output_is_truncated_before_it_is_appended():
    trajectory, _, _, tok = run_loop(
        [(call("1"), "stop"), ("done", "stop")],
        tool=EchoTool(output="a" * 50 + "b" * 50), max_tool_response_length=10,
    )

    assert "<tool_response>\naaaaa...(truncated)...bbbbb\n</tool_response>" in tok.decode(trajectory.response_ids)


def test_only_the_first_parallel_calls_run():
    two_calls = call("1") + tool_call_text("code_interpreter", {"code": "2"})
    trajectory, _, tool, _ = run_loop([(two_calls, "stop"), ("done", "stop")], max_parallel_calls=1)

    assert tool.calls == [{"code": "1"}]
    assert trajectory.tool_calls == 1


def test_unknown_tools_and_failing_tools_answer_with_an_error():
    unknown = tool_call_text("web_search", {"q": "x"})
    trajectory, _, _, tok = run_loop(
        [(unknown, "stop"), (call("1"), "stop"), ("done", "stop")], tool=EchoTool(fail=True),
    )

    text = tok.decode(trajectory.response_ids)
    assert "Error: unknown tool 'web_search'." in text
    assert "Error when executing tool: tool exploded" in text
    assert trajectory.tool_errors == 2


def test_a_turn_with_only_malformed_calls_ends_the_rollout():
    trajectory, _, tool, _ = run_loop([("<tool_call>{oops}</tool_call>", "stop")])

    assert trajectory.stop_reason == "answer"
    assert trajectory.malformed_tool_calls == 1
    assert tool.calls == []


def test_missing_logprobs_read_as_zero():
    trajectory, _, _, _ = run_loop([("done", "stop")], logprob=None)

    assert trajectory.response_logprobs == [0.0] * len(trajectory.response_ids)


def test_rollout_result_has_the_generate_shape():
    trajectory, _, _, _ = run_loop([(call("1"), "stop"), ("done", "stop")])

    result = trajectory.to_rollout_result()

    assert result["prompt_token_ids"] == trajectory.prompt_ids
    assert result["token_ids"] == trajectory.response_ids
    assert len(result["logprobs"]) == len(result["response_mask"]) == len(result["token_ids"])


@pytest.mark.parametrize(
    ("side", "expected"),
    [
        ("left", "abcd...(truncated)"),
        ("right", "(truncated)...ghij"),
        ("middle", "ab...(truncated)...ij"),
    ],
)
def test_truncation_matches_verl(side, expected):
    assert truncate_tool_response("abcdefghij", 4, side) == expected
    assert truncate_tool_response("abc", 4, side) == "abc"


def test_unknown_truncation_side_is_rejected():
    with pytest.raises(ValueError, match="truncate_side"):
        ToolAgentLoop(ChatTemplateTokenizer(ChatMLTokenizer()), [], tool_response_truncate_side="top")
