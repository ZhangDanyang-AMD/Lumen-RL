"""Align rollout token ids with the logprobs that produced them.

Chat completions often return logprobs only for the visible ``content`` text.
Tool-call tokens then have no logprob, and a re-tokenization of the parsed
call is not the sequence the sampler drew. A turn is usable for AgenticRL
only when the server publishes the full completion (``token_ids_cover`` is
``completion``) and that list is the same length as the logprobs and the
reported usage.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

ALIGNED = "aligned"
MISSING = "missing"
LENGTH_MISMATCH = "length_mismatch"
TOOL_TOKENS_MISSING = "tool_tokens_missing"
COMPLETION_COVER = "completion"


def _ints(value: object) -> list[int] | None:
    if not isinstance(value, list):
        return None
    try:
        return [int(item) for item in value]
    except (TypeError, ValueError):
        return None


def _floats(value: object) -> list[float] | None:
    if not isinstance(value, list) or not value:
        return None
    try:
        parsed = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    if any(not math.isfinite(item) for item in parsed):
        return None
    return parsed


def _usage_count(usage: Mapping[str, Any], key: str) -> int | None:
    value = usage.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def extract_choice_tokens(choice: Mapping[str, Any], payload: Mapping[str, Any]) -> dict[str, Any]:
    """Pull prompt ids, completion ids and logprobs out of one chat choice."""
    logprobs_obj = choice.get("logprobs") or {}
    content = logprobs_obj.get("content") if isinstance(logprobs_obj, dict) else None
    flat = logprobs_obj.get("token_logprobs") if isinstance(logprobs_obj, dict) else None
    if flat is None and isinstance(content, list) and content:
        flat = [item.get("logprob") if isinstance(item, dict) else None for item in content]
    return {
        "prompt_token_ids": _ints(payload.get("prompt_token_ids")),
        "response_token_ids": _ints(choice.get("token_ids")),
        "response_logprobs": _floats(flat),
        "token_ids_cover": choice.get("token_ids_cover"),
    }


def rollout_token_status(
    prompt_token_ids: Sequence[int] | None,
    response_token_ids: Sequence[int] | None,
    response_logprobs: Sequence[float] | None,
    *,
    tool_call_count: int,
    usage: Mapping[str, Any] | None = None,
    token_ids_cover: str | None = None,
) -> str:
    """``aligned`` only when ids, logprobs and usage describe one completion."""
    usage = usage or {}
    if tool_call_count and token_ids_cover != COMPLETION_COVER:
        return TOOL_TOKENS_MISSING
    if not response_token_ids or not response_logprobs:
        return MISSING
    if len(response_token_ids) != len(response_logprobs):
        return LENGTH_MISMATCH
    if any(not math.isfinite(float(value)) for value in response_logprobs):
        return LENGTH_MISMATCH
    completion = _usage_count(usage, "completion_tokens")
    if completion is not None and completion != len(response_token_ids):
        return LENGTH_MISMATCH
    prompt = _usage_count(usage, "prompt_tokens")
    if prompt_token_ids is not None and prompt is not None and prompt != len(prompt_token_ids):
        return LENGTH_MISMATCH
    if prompt_token_ids is not None and not prompt_token_ids:
        return LENGTH_MISMATCH
    return ALIGNED
