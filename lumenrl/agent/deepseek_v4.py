"""DeepSeek-V4's chat encoding and DSML tool calls, for append-only agent rollouts.

DeepSeek-V4 ships no Jinja chat template: its prompts are built by the
``encoding/encoding_dsv4.py`` reference that comes with the model. This renders
the part of that encoding an agent rollout needs, the prompt and each round of
tool results, so the token ids of the model's own turns are never re-rendered.
With tools on the system message the encoding keeps every turn's reasoning, so
appending reproduces rendering the whole conversation.
"""

from __future__ import annotations

import json
import re
from typing import Any

from lumenrl.agent.tool_parser import ToolCall

BOS = "<｜begin▁of▁sentence｜>"
EOS = "<｜end▁of▁sentence｜>"
USER = "<｜User｜>"
ASSISTANT = "<｜Assistant｜>"
THINK_START = "<think>"
THINK_END = "</think>"
DSML = "｜DSML｜"
THINKING_MODES = ("chat", "thinking")

_TOOLS_TEMPLATE = """## Tools

You have access to a set of tools to help answer the user's question. You can invoke tools by writing a "<{dsml}tool_calls>" block like the following:

<{dsml}tool_calls>
<{dsml}invoke name="$TOOL_NAME">
<{dsml}parameter name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE</{dsml}parameter>
...
</{dsml}invoke>
<{dsml}invoke name="$TOOL_NAME2">
...
</{dsml}invoke>
</{dsml}tool_calls>

String parameters should be specified as is and set `string="true"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string="false"`.

If thinking_mode is enabled (triggered by {think_start}), you MUST output your complete reasoning inside {think_start}...{think_end} BEFORE any tool calls or final response.

Otherwise, output directly after {think_end} with tool calls or final response.

### Available Tool Schemas

{schemas}

You MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.
"""


def render_tools(tools: list[dict[str, Any]]) -> str:
    """The tool section DeepSeek-V4 appends to the system prompt."""
    schemas = "\n".join(
        json.dumps(tool.get("function", tool), ensure_ascii=False) for tool in tools
    )
    return _TOOLS_TEMPLATE.format(
        dsml=DSML, think_start=THINK_START, think_end=THINK_END, schemas=schemas,
    )


class DeepSeekV4ChatTokenizer:
    """``ChatTemplateTokenizer``'s interface for DeepSeek-V4's encoding.

    ``thinking_mode`` is the encoding's: "chat" opens every assistant turn
    with ``</think>``, "thinking" with ``<think>``.
    """

    def __init__(self, tokenizer: Any, thinking_mode: str = "chat") -> None:
        if thinking_mode not in THINKING_MODES:
            raise ValueError(f"thinking_mode must be one of {THINKING_MODES}, got {thinking_mode!r}")
        self._tokenizer = tokenizer
        self._assistant_start = ASSISTANT + (THINK_START if thinking_mode == "thinking" else THINK_END)
        self._eos_ids = self.encode_text(EOS)

    def encode_text(self, text: str) -> list[int]:
        return list(self._tokenizer(text, add_special_tokens=False)["input_ids"])

    def decode(self, token_ids: list[int]) -> str:
        return self._tokenizer.decode(token_ids, skip_special_tokens=False)

    def encode_prompt(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None,
    ) -> list[int]:
        """BOS, the system prompt with the tool section, the user turns, then the
        assistant turn the model writes."""
        system, rest = "", list(messages)
        if rest and rest[0].get("role") == "system":
            system = rest.pop(0).get("content") or ""
        if tools:
            system += "\n\n" + render_tools(tools)
        text = BOS + system
        for message in rest:
            if message.get("role") != "user":
                raise ValueError(
                    f"a DeepSeek-V4 agent prompt holds system and user messages, got {message.get('role')!r}"
                )
            text += USER + (message.get("content") or "")
        return self.encode_text(text + self._assistant_start)

    def encode_tool_responses(self, contents: list[str], after: list[int]) -> list[int]:
        """Environment tokens between the model turn ``after`` and the next one."""
        results = "\n\n".join(f"<tool_result>{content}</tool_result>" for content in contents)
        close = [] if after[-len(self._eos_ids):] == self._eos_ids else list(self._eos_ids)
        return close + self.encode_text(USER + results + self._assistant_start)


_TOOL_CALLS = re.compile(rf"<{DSML}tool_calls>(.*?)</{DSML}tool_calls>", re.DOTALL)
_INVOKE = re.compile(rf'<{DSML}invoke name="([^"]*)">(.*?)</{DSML}invoke>', re.DOTALL)
_PARAMETER = re.compile(
    rf'<{DSML}parameter name="([^"]*)" string="(true|false)">(.*?)</{DSML}parameter>', re.DOTALL,
)


class DsmlToolParser:
    """``<｜DSML｜tool_calls>`` blocks of ``<｜DSML｜invoke>`` calls.

    A parameter with ``string="false"`` holds JSON. An invoke without a name,
    with a parameter that is not JSON where it should be, with a parameter
    given twice, or with text between its parameters is malformed: it is
    counted and skipped, and so is any other text in a block. An unclosed
    block is not a call.
    """

    def parse(self, text: str) -> tuple[list[ToolCall], int]:
        """Return the well-formed calls in ``text`` and the number of malformed ones."""
        calls: list[ToolCall] = []
        malformed = 0
        for block in _TOOL_CALLS.findall(text):
            for name, body in _INVOKE.findall(block):
                call = self._parse_invoke(name, body)
                if call is None:
                    malformed += 1
                else:
                    calls.append(call)
            if _INVOKE.sub("", block).strip():
                malformed += 1
        return calls, malformed

    @staticmethod
    def _parse_invoke(name: str, body: str) -> ToolCall | None:
        if not name:
            return None
        arguments: dict[str, Any] = {}
        if _PARAMETER.sub("", body).strip():
            return None
        for key, is_string, value in _PARAMETER.findall(body):
            if key in arguments:
                return None
            if is_string == "true":
                arguments[key] = value
                continue
            try:
                arguments[key] = json.loads(value)
            except json.JSONDecodeError:
                return None
        return ToolCall(name=name, arguments=arguments)
