"""Rollout logprobs are saved only when they match the sampled completion."""

from __future__ import annotations

from lumenrl.agentic.rollout.openai_policy import OpenAIChatPolicy
from lumenrl.agentic.rollout.token_alignment import (
    ALIGNED,
    LENGTH_MISMATCH,
    MISSING,
    TOOL_TOKENS_MISSING,
    extract_choice_tokens,
    rollout_token_status,
)


def _choice(token_ids, logprobs, cover="completion"):
    return {
        "message": {"content": "", "tool_calls": [{"id": "c", "function": {"name": "submit", "arguments": "{}"}}]},
        "finish_reason": "tool_calls",
        "token_ids": token_ids,
        "token_ids_cover": cover,
        "logprobs": {"token_logprobs": logprobs},
    }


def test_request_asks_for_logprobs(monkeypatch) -> None:
    monkeypatch.setenv("TEST_KEY", "secret")
    policy = OpenAIChatPolicy(
        "http://127.0.0.1:9/v1", "m", "TEST_KEY", {"temperature": 0}, 1, 0, False,
    )
    captured = {}

    async def post(body):
        captured.update(body)
        return {"choices": [{"message": {}, "finish_reason": "stop"}]}

    policy._post = post

    import asyncio
    asyncio.run(policy.act([{"role": "user", "content": "hi"}], [], 0))
    assert captured["logprobs"] is True
    assert captured["top_logprobs"] == 0
    assert captured["return_token_ids"] is True


def test_explicit_logprobs_false_is_kept(monkeypatch) -> None:
    monkeypatch.setenv("TEST_KEY", "secret")
    policy = OpenAIChatPolicy(
        "http://127.0.0.1:9/v1", "m", "TEST_KEY", {}, 1, 0, False, extra_body={"logprobs": False},
    )
    captured = {}

    async def post(body):
        captured.update(body)
        return {"choices": [{"message": {}, "finish_reason": "stop"}]}

    policy._post = post
    import asyncio
    asyncio.run(policy.act([], [], 0))
    assert captured["logprobs"] is False


def test_aligned_completion_is_saved_and_a_short_logprob_list_is_rejected() -> None:
    usage = {"prompt_tokens": 2, "completion_tokens": 3}
    payload = {"prompt_token_ids": [7, 8], "usage": usage, "model": "m"}
    aligned = extract_choice_tokens(_choice([1, 2, 3], [-0.1, -0.2, -0.3]), payload)
    assert rollout_token_status(**aligned, tool_call_count=1, usage=usage) == ALIGNED

    short = extract_choice_tokens(_choice([1, 2, 3], [-0.1]), payload)
    assert rollout_token_status(
        **short, tool_call_count=1, usage={"prompt_tokens": 2, "completion_tokens": 3},
    ) == LENGTH_MISMATCH

    content_only = extract_choice_tokens(
        {"message": {}, "logprobs": {"content": [{"logprob": -0.2}]}, "token_ids": [4]},
        {},
    )
    assert rollout_token_status(**content_only, tool_call_count=1, usage={}) == TOOL_TOKENS_MISSING
    assert rollout_token_status(None, None, None, tool_call_count=0) == MISSING

