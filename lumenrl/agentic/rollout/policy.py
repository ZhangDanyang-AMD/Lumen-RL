"""Policy interface for multi-turn agent rollout, plus a scripted policy."""

from __future__ import annotations

import abc
import json
from typing import Any, Mapping, Sequence

from lumenrl.agentic.core.protocols import ToolCall
from lumenrl.agentic.core.trajectory import PolicyTurn


class PolicyError(RuntimeError):
    """The policy backend failed to produce a response."""


class AgentPolicy(abc.ABC):
    """Produces one assistant turn given the chat history and tool schemas."""

    @abc.abstractmethod
    async def act(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        turn_index: int,
    ) -> PolicyTurn:
        """Return the next assistant turn.

        Raises:
            PolicyError: If the backend cannot produce a response.
        """

    def describe(self) -> dict[str, Any]:
        """Return the policy identity stored in each episode."""
        return {"type": type(self).__name__}

    async def close(self) -> None:
        """Release backend resources."""


class ScriptedPolicy(AgentPolicy):
    """Replays a fixed sequence of tool calls; ends the episode when exhausted.

    Used for Phase-1 scripted baselines and for testing environments without a model.
    """

    def __init__(self, actions: Sequence[Mapping[str, Any]], model: str = "scripted") -> None:
        """Args:
            actions: Items of the form ``{"name": str, "arguments": dict}``.
            model: Identifier recorded as the policy model.
        """
        self._actions = [dict(item) for item in actions]
        self._model = model

    async def act(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        turn_index: int,
    ) -> PolicyTurn:
        if turn_index >= len(self._actions):
            return PolicyTurn(
                turn_index=turn_index, model=self._model, content="done", reasoning=None,
                tool_calls=[], finish_reason="stop",
            )
        item = self._actions[turn_index]
        raw = json.dumps(item["arguments"], sort_keys=True)
        call = ToolCall.from_raw(f"scripted-{turn_index}", item["name"], raw)
        return PolicyTurn(
            turn_index=turn_index, model=self._model, content="", reasoning=None,
            tool_calls=[call], finish_reason="tool_calls",
        )

    def describe(self) -> dict[str, Any]:
        return {"type": type(self).__name__, "model": self._model, "action_count": len(self._actions)}
