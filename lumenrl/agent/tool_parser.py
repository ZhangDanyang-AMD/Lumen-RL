"""Tool calls in a model turn, in the Hermes format Qwen's chat template asks for."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]


class ToolCallParser(Protocol):
    def parse(self, text: str) -> tuple[list[ToolCall], int]:
        """The well-formed calls in a model turn and the number of malformed ones."""


class HermesToolParser:
    """``<tool_call>{"name": ..., "arguments": {...}}</tool_call>`` blocks.

    A block that is not a JSON object with a string ``name`` and an object of
    ``arguments`` (or a JSON string of one) is malformed: it is counted and
    skipped, as verl's hermes parser does.
    """

    _BLOCK = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)

    def parse(self, text: str) -> tuple[list[ToolCall], int]:
        """Return the well-formed calls in ``text`` and the number of malformed ones."""
        calls: list[ToolCall] = []
        malformed = 0
        for block in self._BLOCK.findall(text):
            call = self._parse_block(block)
            if call is None:
                malformed += 1
            else:
                calls.append(call)
        return calls, malformed

    @staticmethod
    def _parse_block(block: str) -> ToolCall | None:
        try:
            payload = json.loads(block.strip())
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, dict) or not isinstance(payload.get("name"), str):
            return None
        arguments = payload.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                return None
        if not isinstance(arguments, dict):
            return None
        return ToolCall(name=payload["name"], arguments=arguments)
