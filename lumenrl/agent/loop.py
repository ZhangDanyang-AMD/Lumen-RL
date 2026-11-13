"""The tool-calling agent loop: generate, run the tools it asks for, repeat."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from lumenrl.agent.chat import ChatTemplateTokenizer
from lumenrl.agent.tool_parser import HermesToolParser, ToolCall, ToolCallParser
from lumenrl.agent.tools import StatefulTool, Tool, ToolResult, ToolSession
from lumenrl.agent.trajectory import Trajectory

logger = logging.getLogger(__name__)

# (token ids so far, sampling params) -> a ``VLLMRayServer.generate`` result.
GenerateFn = Callable[[list[int], dict[str, Any]], Awaitable[dict[str, Any]]]

_TRUNCATE_SIDES = ("left", "right", "middle")


def truncate_tool_response(text: str, limit: int, side: str) -> str:
    """Shorten ``text`` to ``limit`` characters the way verl's agent loop does."""
    if limit <= 0 or len(text) <= limit:
        return text
    if side == "left":
        return text[:limit] + "...(truncated)"
    if side == "right":
        return "(truncated)..." + text[-limit:]
    half = limit // 2
    return text[:half] + "...(truncated)..." + text[-half:]


class ToolAgentLoop:
    """Multi-turn rollout of one prompt with tools, as verl's ``ToolAgentLoop``.

    Each assistant turn generates until the model stops. A turn with tool calls
    has them run and their output appended as environment tokens; a turn
    without ends the rollout. The rollout also ends after
    ``max_assistant_turns`` turns, or when the response (model and environment
    tokens together) would exceed its token budget. A stateful tool (one with
    ``open``) gets a session per rollout, opened at the first call and closed
    when the rollout ends. ``scripted_first_turn`` writes the first assistant
    turn instead of generating it, as environment tokens, and runs the tools it
    calls: for plumbing smokes of a model that cannot call tools yet.
    """

    def __init__(
        self,
        chat: ChatTemplateTokenizer,
        tools: Sequence[Tool | StatefulTool],
        *,
        parser: ToolCallParser | None = None,
        max_assistant_turns: int = 8,
        max_parallel_calls: int = 1,
        max_tool_response_length: int = 256,
        tool_response_truncate_side: str = "middle",
        scripted_first_turn: str = "",
    ) -> None:
        if tool_response_truncate_side not in _TRUNCATE_SIDES:
            raise ValueError(
                f"tool_response_truncate_side must be one of {_TRUNCATE_SIDES}, "
                f"got {tool_response_truncate_side!r}"
            )
        self._chat = chat
        self._tools = {tool.name: tool for tool in tools}
        self._parser = parser or HermesToolParser()
        self._max_assistant_turns = int(max_assistant_turns)
        self._max_parallel_calls = max(1, int(max_parallel_calls))
        self._max_tool_response_length = int(max_tool_response_length)
        self._truncate_side = tool_response_truncate_side
        self._scripted_first_turn = scripted_first_turn

    @property
    def tool_schemas(self) -> list[dict[str, Any]]:
        return [tool.schema for tool in self._tools.values()]

    async def run(
        self,
        generate: GenerateFn,
        messages: list[dict[str, Any]],
        sampling_params: dict[str, Any],
        max_response_tokens: int,
        max_sequence_tokens: int | None = None,
    ) -> Trajectory:
        """Roll out ``messages``; the response stays within ``max_response_tokens``
        and, with ``max_sequence_tokens``, prompt plus response within that."""
        trajectory = Trajectory(prompt_ids=self._chat.encode_prompt(messages, self.tool_schemas))
        if max_sequence_tokens is not None:
            max_response_tokens = min(
                max_response_tokens, max_sequence_tokens - len(trajectory.prompt_ids),
            )
        sessions: dict[str, ToolSession] = {}
        try:
            await self._roll_out(
                trajectory, generate, sampling_params, max_response_tokens, sessions,
            )
        finally:
            for name, session in sessions.items():
                try:
                    await session.close()
                except Exception as e:  # noqa: BLE001 - a failed close must not lose the rollout
                    logger.warning("closing the %s session failed: %r", name, e)
        return trajectory

    async def _roll_out(
        self,
        trajectory: Trajectory,
        generate: GenerateFn,
        sampling_params: dict[str, Any],
        max_response_tokens: int,
        sessions: dict[str, ToolSession],
    ) -> None:
        if self._scripted_first_turn:
            turn_ids = self._chat.encode_text(self._scripted_first_turn)
            trajectory.add_scripted_turn(turn_ids)
            calls, malformed = self._parser.parse(self._scripted_first_turn)
            trajectory.malformed_tool_calls += malformed
            if calls and not await self._run_tools(
                trajectory, calls, turn_ids, sessions, max_response_tokens,
            ):
                return
        while True:
            budget = max_response_tokens - len(trajectory.response_ids)
            if budget <= 0:
                trajectory.stop_reason = "length"
                break
            turn_params = dict(sampling_params)
            turn_params["max_tokens"] = min(int(sampling_params.get("max_tokens", budget)), budget)
            if trajectory.routed_experts is not None:
                # The earlier positions keep the routing of the turn that sampled
                # them: on a prefix-cache miss (no cache, eviction, or the reset
                # of a weight sync) this request re-prefills them.
                turn_params["routed_experts_prompt_start"] = trajectory.routed_positions
            context = trajectory.prompt_ids + trajectory.response_ids
            t0 = time.perf_counter()
            out = await generate(context, turn_params)
            trajectory.generate_s += time.perf_counter() - t0
            turn_ids = list(out["token_ids"])
            trajectory.add_model_turn(turn_ids, out.get("logprobs"))
            if out.get("routed_experts") is not None:
                trajectory.add_routes(out["routed_experts"], through=len(context) + len(turn_ids) - 1)

            finish_reason = out.get("finish_reason")
            if finish_reason is None:
                finish_reason = "length" if len(turn_ids) >= turn_params["max_tokens"] else "stop"
            if finish_reason != "stop":
                trajectory.stop_reason = "length" if finish_reason == "length" else "abort"
                break

            calls, malformed = self._parser.parse(self._chat.decode(turn_ids))
            trajectory.malformed_tool_calls += malformed
            if not calls:
                trajectory.stop_reason = "answer"
                break
            if trajectory.assistant_turns >= self._max_assistant_turns:
                trajectory.stop_reason = "max_turns"
                break

            if not await self._run_tools(trajectory, calls, turn_ids, sessions, max_response_tokens):
                break

    async def _run_tools(
        self,
        trajectory: Trajectory,
        calls: list[ToolCall],
        after: list[int],
        sessions: dict[str, ToolSession],
        max_response_tokens: int,
    ) -> bool:
        """Run the turn's calls and append their output after it; False, with
        the rollout ended, when the output does not fit."""
        calls = calls[: self._max_parallel_calls]
        t0 = time.perf_counter()
        results = await asyncio.gather(*(self._call(call, sessions) for call in calls))
        trajectory.tool_s += time.perf_counter() - t0
        trajectory.tool_calls += len(results)
        trajectory.tool_errors += sum(not result.ok for result in results)

        environment = self._chat.encode_tool_responses(
            [
                truncate_tool_response(
                    result.text, self._max_tool_response_length, self._truncate_side,
                )
                for result in results
            ],
            after=after,
        )
        if len(trajectory.response_ids) + len(environment) >= max_response_tokens:
            trajectory.stop_reason = "length"
            return False
        trajectory.add_environment(environment)
        return True

    async def _call(self, call: ToolCall, sessions: dict[str, ToolSession]) -> ToolResult:
        tool = self._tools.get(call.name)
        if tool is None:
            return ToolResult(f"Error: unknown tool {call.name!r}.", ok=False)
        try:
            if hasattr(tool, "open"):
                if call.name not in sessions:
                    sessions[call.name] = tool.open()
                return await sessions[call.name](call.arguments)
            return await tool(call.arguments)
        except Exception as e:  # noqa: BLE001 - a tool's failure is the model's observation
            return ToolResult(f"Error when executing tool: {e}", ok=False)
