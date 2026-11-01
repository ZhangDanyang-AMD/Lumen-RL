"""A character-level tokenizer with a Qwen3-style ChatML template."""

from __future__ import annotations

import json
import re
from typing import Any

SPECIALS = (
    "<|im_start|>", "<|im_end|>", "<|endoftext|>",
    "<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>",
)
_SPECIAL_IDS = {token: 100_000 + i for i, token in enumerate(SPECIALS)}
_SPECIAL_TOKENS = {i: token for token, i in _SPECIAL_IDS.items()}
_SPLIT = re.compile("(" + "|".join(re.escape(token) for token in SPECIALS) + ")")


class ChatMLTokenizer:
    """One token per character, one per special token.

    Its template renders consecutive tool messages into one user turn, as
    Qwen3's does. ``rewrite_turn_before_tools`` makes it change a turn once
    tool messages follow it, which append-only tokenization must reject.
    """

    def __init__(self, *, rewrite_turn_before_tools: bool = False) -> None:
        self.rewrite_turn_before_tools = rewrite_turn_before_tools
        self.template_kwargs: list[dict[str, Any]] = []

    def __call__(self, text: str, add_special_tokens: bool = True) -> dict[str, list[int]]:
        assert add_special_tokens is False
        ids: list[int] = []
        for piece in _SPLIT.split(text):
            if piece in _SPECIAL_IDS:
                ids.append(_SPECIAL_IDS[piece])
            else:
                ids.extend(ord(c) for c in piece)
        return {"input_ids": ids}

    def encode(self, text: str) -> list[int]:
        return self(text, add_special_tokens=False)["input_ids"]

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        out = []
        for i in ids:
            if i in _SPECIAL_TOKENS:
                if not skip_special_tokens:
                    out.append(_SPECIAL_TOKENS[i])
            else:
                out.append(chr(i))
        return "".join(out)

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tokenize: bool = False,
        add_generation_prompt: bool = False,
        **kwargs: Any,
    ) -> str:
        assert tokenize is False
        self.template_kwargs.append(kwargs)
        out = ""
        if tools:
            out += "<|im_start|>system\n# Tools\n" + json.dumps(tools) + "<|im_end|>\n"
        for i, message in enumerate(messages):
            role, content = message["role"], message["content"]
            if role == "tool":
                if i == 0 or messages[i - 1]["role"] != "tool":
                    out += "<|im_start|>user"
                out += "\n<tool_response>\n" + content + "\n</tool_response>"
                if i == len(messages) - 1 or messages[i + 1]["role"] != "tool":
                    out += "<|im_end|>\n"
                continue
            next_is_tool = i + 1 < len(messages) and messages[i + 1]["role"] == "tool"
            if self.rewrite_turn_before_tools and next_is_tool:
                content = content.upper()
            out += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        if add_generation_prompt:
            out += "<|im_start|>assistant\n"
            if kwargs.get("enable_thinking") is False:
                out += "<think>\n\n</think>\n\n"
        return out


IM_END = _SPECIAL_IDS["<|im_end|>"]
ENDOFTEXT = _SPECIAL_IDS["<|endoftext|>"]


def tool_call_text(name: str, arguments: dict[str, Any]) -> str:
    return "<tool_call>\n" + json.dumps({"name": name, "arguments": arguments}) + "\n</tool_call>"
