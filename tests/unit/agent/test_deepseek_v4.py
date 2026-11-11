"""DeepSeek-V4's encoding for agent rollouts, and DSML tool-call parsing.

``data/deepseek_v4`` holds case 1 of the tests that come with DeepSeek-V4's
reference encoding (MIT, see the LICENSE there): a thinking-mode conversation
with one tool call.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from lumenrl.agent.deepseek_v4 import (
    ASSISTANT,
    BOS,
    EOS,
    THINK_END,
    THINK_START,
    USER,
    DeepSeekV4ChatTokenizer,
    DsmlToolParser,
)
from lumenrl.agent.tool_parser import ToolCall

DATA = Path(__file__).parent / "data" / "deepseek_v4"
SPECIALS = (BOS, EOS, USER, ASSISTANT, THINK_START, THINK_END)
_IDS = {token: 200_000 + i for i, token in enumerate(SPECIALS)}
_TOKENS = {i: token for token, i in _IDS.items()}
_SPLIT = re.compile("(" + "|".join(re.escape(token) for token in SPECIALS) + ")")
TOOLS = [{"type": "function", "function": {"name": "code_interpreter", "parameters": {}}}]


class CharTokenizer:
    """One token per character, one per DeepSeek-V4 special token."""

    def __call__(self, text: str, add_special_tokens: bool = True) -> dict[str, list[int]]:
        assert add_special_tokens is False
        ids: list[int] = []
        for piece in _SPLIT.split(text):
            ids.extend([_IDS[piece]] if piece in _IDS else (ord(c) for c in piece))
        return {"input_ids": ids}

    def encode(self, text: str) -> list[int]:
        return self(text, add_special_tokens=False)["input_ids"]

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        assert skip_special_tokens is False
        return "".join(_TOKENS.get(i) or chr(i) for i in ids)


def test_a_tool_round_appends_to_what_the_reference_encoding_renders():
    case = json.loads((DATA / "test_input_1.json").read_text())
    gold = (DATA / "test_output_1.txt").read_text()
    system, user, _, tool, _ = case["messages"]
    # The model's two turns, as DeepSeek's own test cuts them out of the rendering.
    marker = ASSISTANT + THINK_START
    first = gold.index(marker) + len(marker)
    second = gold.rindex(marker) + len(marker)
    first_turn = gold[first:gold.index(USER, first)]
    second_turn = gold[second:]

    tok = CharTokenizer()
    chat = DeepSeekV4ChatTokenizer(tok, thinking_mode="thinking")
    prompt = chat.encode_prompt([system, user], case["tools"])
    turn = tok.encode(first_turn)
    environment = chat.encode_tool_responses([tool["content"]], after=turn)

    assert tok.decode(prompt) == gold[:first]
    assert tok.decode(prompt + turn + environment + tok.encode(second_turn)) == gold
    calls, malformed = DsmlToolParser().parse(first_turn)
    assert calls == [ToolCall("get_weather", {"location": "Beijing", "unit": "celsius"})]
    assert malformed == 0


def test_chat_mode_opens_every_assistant_turn_with_the_thinking_closed():
    tok = CharTokenizer()
    chat = DeepSeekV4ChatTokenizer(tok)

    prompt = tok.decode(chat.encode_prompt([{"role": "user", "content": "6*7?"}], TOOLS))
    tool_round = tok.decode(chat.encode_tool_responses(["42", "ok"], after=tok.encode("x" + EOS)))

    assert prompt.startswith(BOS + "\n\n## Tools\n")
    assert '{"name": "code_interpreter", "parameters": {}}' in prompt
    assert prompt.endswith(USER + "6*7?" + ASSISTANT + THINK_END)
    assert tool_round == (
        USER + "<tool_result>42</tool_result>\n\n<tool_result>ok</tool_result>" + ASSISTANT + THINK_END
    )


def test_a_turn_without_its_end_of_sentence_gets_one():
    tok = CharTokenizer()
    chat = DeepSeekV4ChatTokenizer(tok)

    assert tok.decode(chat.encode_tool_responses(["42"], after=tok.encode("x"))).startswith(EOS + USER)


def test_a_prompt_holds_only_system_and_user_messages():
    chat = DeepSeekV4ChatTokenizer(CharTokenizer())

    with pytest.raises(ValueError, match="'assistant'"):
        chat.encode_prompt([{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}])


def test_an_unknown_thinking_mode_is_rejected():
    with pytest.raises(ValueError, match="thinking_mode"):
        DeepSeekV4ChatTokenizer(CharTokenizer(), thinking_mode="think")


def dsml_call(name: str, *params: tuple[str, str, str]) -> str:
    body = "".join(
        f'<｜DSML｜parameter name="{k}" string="{s}">{v}</｜DSML｜parameter>\n' for k, s, v in params
    )
    return f'<｜DSML｜invoke name="{name}">\n{body}</｜DSML｜invoke>\n'


def tool_calls(*calls: str) -> str:
    return "\n\n<｜DSML｜tool_calls>\n" + "".join(calls) + "</｜DSML｜tool_calls>" + EOS


def test_parser_reads_string_and_json_parameters_of_every_call():
    text = "Let me run it." + tool_calls(
        dsml_call("code_interpreter", ("code", "true", 'print("a < b")\nx = 1')),
        dsml_call("search", ("q", "true", "x"), ("n", "false", "3"), ("opts", "false", '{"a": [1]}')),
    )

    calls, malformed = DsmlToolParser().parse(text)

    assert calls == [
        ToolCall("code_interpreter", {"code": 'print("a < b")\nx = 1'}),
        ToolCall("search", {"q": "x", "n": 3, "opts": {"a": [1]}}),
    ]
    assert malformed == 0


@pytest.mark.parametrize(
    "bad",
    [
        dsml_call("", ("code", "true", "1")),
        dsml_call("code_interpreter", ("n", "false", "not json")),
        dsml_call("code_interpreter", ("code", "true", "1"), ("code", "true", "2")),
        '<｜DSML｜invoke name="code_interpreter">\nstray text\n</｜DSML｜invoke>\n',
        "not an invoke\n",
    ],
    ids=["no-name", "bad-json", "repeated-parameter", "text-in-invoke", "text-in-block"],
)
def test_parser_counts_malformed_calls(bad):
    text = tool_calls(bad, dsml_call("ok"))

    calls, malformed = DsmlToolParser().parse(text)

    assert calls == [ToolCall("ok", {})]
    assert malformed == 1


def test_parser_ignores_an_unclosed_block():
    text = "\n\n<｜DSML｜tool_calls>\n" + dsml_call("code_interpreter", ("code", "true", "1"))

    assert DsmlToolParser().parse(text) == ([], 0)
