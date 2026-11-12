"""DeepSeek-V4's and V4.1's chat encodings and DSML tool calls, for append-only agent rollouts.

DeepSeek-V4 ships no Jinja chat template: its prompts are built by the
``encoding/encoding_dsv4.py`` reference that comes with the model, and V4.1's
by ``encoding/encoding.py``. This renders the part of those encodings an agent
rollout needs, the prompt and each round of tool results, so the token ids of
the model's own turns are never re-rendered. With tools on the system message
both encodings keep every turn's reasoning, so appending reproduces rendering
the whole conversation.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from lumenrl.agent.tool_parser import ToolCall

BOS = "<｜begin▁of▁sentence｜>"
EOS = "<｜end▁of▁sentence｜>"
USER = "<｜User｜>"
ASSISTANT = "<｜Assistant｜>"
THINK_START = "<think>"
THINK_END = "</think>"
SYSTEM = "<｜System｜>"
DSML = "｜DSML｜"
THINKING_MODES = ("chat", "thinking")
REASONING_EFFORTS = {"low": 50, "high": 75, "max": 100}


@dataclass(frozen=True)
class DsmlTags:
    """The tag names after ``｜DSML｜``; V4.1 gives each a leading space."""

    calls: str
    invoke: str
    parameter: str


V4_TAGS = DsmlTags(calls="tool_calls", invoke="invoke", parameter="parameter")
V41_TAGS = DsmlTags(calls=" calls", invoke=" invoke", parameter=" parameter")

_TOOLS_TEMPLATE = """## Tools

You have access to a set of tools to help answer the user's question. You can invoke tools by writing a "<{dsml}{calls}>" block like the following:

<{dsml}{calls}>
<{dsml}{invoke} name="$TOOL_NAME">
<{dsml}{parameter} name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE</{dsml}{parameter}>
...
</{dsml}{invoke}>
<{dsml}{invoke} name="$TOOL_NAME2">
...
</{dsml}{invoke}>
</{dsml}{calls}>

String parameters should be specified as is and set `string="true"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string="false"`.

If thinking_mode is enabled (triggered by {think_start}), you MUST output your complete reasoning inside {think_start}...{think_end} BEFORE any tool calls or final response.

Otherwise, output directly after {think_end} with tool calls or final response.

### Available Tool Schemas

{schemas}

You MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.
"""


def render_tools(tools: list[dict[str, Any]], tags: DsmlTags = V4_TAGS) -> str:
    """The tool section DeepSeek-V4 (or, with ``V41_TAGS``, V4.1) appends to the system prompt."""
    schemas = "\n".join(
        json.dumps(tool.get("function", tool), ensure_ascii=False) for tool in tools
    )
    return _TOOLS_TEMPLATE.format(
        dsml=DSML, calls=tags.calls, invoke=tags.invoke, parameter=tags.parameter,
        think_start=THINK_START, think_end=THINK_END, schemas=schemas,
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


class DeepSeekV41ChatTokenizer(DeepSeekV4ChatTokenizer):
    """``ChatTemplateTokenizer``'s interface for DeepSeek-V4.1's encoding.

    Unlike V4, the first turn opens with ``<｜System｜>``, and in "thinking"
    mode it starts with the reasoning-effort budget: 1-100, or "low" (50),
    "high" (75, the encoding's default) or "max" (100). The tool section sits
    on the system turn, an empty one if the conversation has none, as ATOM's
    and the reference's servers place request-level tools; and consecutive
    user messages share one user turn.
    """

    def __init__(
        self, tokenizer: Any, thinking_mode: str = "chat", reasoning_effort: int | str = "high",
    ) -> None:
        super().__init__(tokenizer, thinking_mode)
        budget = REASONING_EFFORTS.get(reasoning_effort) if isinstance(reasoning_effort, str) else reasoning_effort
        if type(budget) is not int or not 1 <= budget <= 100:
            raise ValueError(
                f"reasoning_effort must be an integer in [1, 100] or one of {list(REASONING_EFFORTS)}, "
                f"got {reasoning_effort!r}"
            )
        self._effort = (
            f"Reasoning Effort: {budget} (range 1-100, the higher the value, the more thorough the reasoning)\n\n"
            if thinking_mode == "thinking" else ""
        )

    def encode_prompt(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None,
    ) -> list[int]:
        """BOS, the system turn with the effort budget and the tool section, the
        user turn, then the assistant turn the model writes."""
        rest = list(messages)
        system = None
        if rest and rest[0].get("role") == "system":
            system = rest.pop(0).get("content") or ""
        if tools:
            system = (system or "") + "\n\n" + render_tools(tools, V41_TAGS)
        users = []
        for message in rest:
            if message.get("role") != "user":
                raise ValueError(
                    f"a DeepSeek-V4.1 agent prompt holds system and user messages, got {message.get('role')!r}"
                )
            users.append(message.get("content") or "")
        text = BOS
        if system is not None or self._effort:
            text += SYSTEM + self._effort + (system or "")
        if users:
            text += USER + "\n\n".join(users)
        return self.encode_text(text + self._assistant_start)


class DsmlToolParser:
    """``<｜DSML｜tool_calls>`` blocks of ``<｜DSML｜invoke>`` calls, or V4.1's
    ``<｜DSML｜ calls>`` and ``<｜DSML｜ invoke>`` with ``V41_TAGS``.

    A parameter with ``string="false"`` holds JSON. An invoke without a name,
    with a parameter that is not JSON where it should be, with a parameter
    given twice, or with text between its parameters is malformed: it is
    counted and skipped, and so is any other text in a block. An unclosed
    block is not a call.
    """

    def __init__(self, tags: DsmlTags = V4_TAGS) -> None:
        self._block = re.compile(rf"<{DSML}{tags.calls}>(.*?)</{DSML}{tags.calls}>", re.DOTALL)
        self._invoke = re.compile(rf'<{DSML}{tags.invoke} name="([^"]*)">(.*?)</{DSML}{tags.invoke}>', re.DOTALL)
        self._parameter = re.compile(
            rf'<{DSML}{tags.parameter} name="([^"]*)" string="(true|false)">(.*?)</{DSML}{tags.parameter}>',
            re.DOTALL,
        )

    def parse(self, text: str) -> tuple[list[ToolCall], int]:
        """Return the well-formed calls in ``text`` and the number of malformed ones."""
        calls: list[ToolCall] = []
        malformed = 0
        for block in self._block.findall(text):
            for name, body in self._invoke.findall(block):
                call = self._parse_invoke(name, body)
                if call is None:
                    malformed += 1
                else:
                    calls.append(call)
            if self._invoke.sub("", block).strip():
                malformed += 1
        return calls, malformed

    def _parse_invoke(self, name: str, body: str) -> ToolCall | None:
        if not name:
            return None
        arguments: dict[str, Any] = {}
        if self._parameter.sub("", body).strip():
            return None
        for key, is_string, value in self._parameter.findall(body):
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
