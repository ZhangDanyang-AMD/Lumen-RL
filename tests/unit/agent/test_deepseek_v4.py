"""DeepSeek-V4's and V4.1's encodings for agent rollouts, and DSML tool-call parsing.

``data/deepseek_v4`` and ``data/deepseek_v41`` hold case 1 of the tests that
come with DeepSeek-V4's and V4.1's reference encodings (MIT, see the LICENSE
in each): a thinking-mode conversation with one tool call.
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
    SYSTEM,
    THINK_END,
    THINK_START,
    USER,
    V4_TAGS,
    V41_TAGS,
    DeepSeekV41ChatTokenizer,
    DeepSeekV4ChatTokenizer,
    DsmlToolParser,
    DsmlTags,
)
from lumenrl.agent.tool_parser import ToolCall

DATA = Path(__file__).parent / "data" / "deepseek_v4"
DATA_V41 = Path(__file__).parent / "data" / "deepseek_v41"
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


@pytest.mark.parametrize(
    ("data", "chat_class", "tags"),
    [(DATA, DeepSeekV4ChatTokenizer, V4_TAGS), (DATA_V41, DeepSeekV41ChatTokenizer, V41_TAGS)],
    ids=["v4", "v41"],
)
def test_a_tool_round_appends_to_what_the_reference_encoding_renders(data, chat_class, tags):
    case = json.loads((data / "test_input_1.json").read_text())
    gold = (data / "test_output_1.txt").read_text()
    system, user, _, tool, _ = case["messages"]
    # The model's two turns, as DeepSeek's own test cuts them out of the rendering.
    marker = ASSISTANT + THINK_START
    first = gold.index(marker) + len(marker)
    second = gold.rindex(marker) + len(marker)
    first_turn = gold[first:gold.index(USER, first)]
    second_turn = gold[second:]

    tok = CharTokenizer()
    chat = chat_class(tok, thinking_mode="thinking")
    prompt = chat.encode_prompt([system, user], case["tools"])
    turn = tok.encode(first_turn)
    environment = chat.encode_tool_responses([tool["content"]], after=turn)

    assert tok.decode(prompt) == gold[:first]
    assert tok.decode(prompt + turn + environment + tok.encode(second_turn)) == gold
    calls, malformed = DsmlToolParser(tags).parse(first_turn)
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


def effort(budget: int) -> str:
    return f"Reasoning Effort: {budget} (range 1-100, the higher the value, the more thorough the reasoning)\n\n"


USER_67 = {"role": "user", "content": "6*7?"}
BRIEF = {"role": "system", "content": "Be brief."}


@pytest.mark.parametrize(
    ("thinking_mode", "reasoning_effort", "messages", "rendered"),
    [
        ("chat", "high", [USER_67], BOS + USER + "6*7?" + ASSISTANT + THINK_END),
        ("chat", "low", [BRIEF, USER_67], BOS + SYSTEM + "Be brief." + USER + "6*7?" + ASSISTANT + THINK_END),
        ("thinking", "max", [USER_67], BOS + SYSTEM + effort(100) + USER + "6*7?" + ASSISTANT + THINK_START),
        ("thinking", 30, [BRIEF, USER_67], BOS + SYSTEM + effort(30) + "Be brief." + USER + "6*7?" + ASSISTANT + THINK_START),
        ("chat", "high", [USER_67, {"role": "user", "content": "And 6*8?"}],
         BOS + USER + "6*7?\n\nAnd 6*8?" + ASSISTANT + THINK_END),
    ],
    ids=["chat", "chat-system", "thinking-max", "thinking-30-system", "two-user-messages"],
)
def test_v41_prompts_render_as_its_reference_encoding_does(thinking_mode, reasoning_effort, messages, rendered):
    tok = CharTokenizer()
    chat = DeepSeekV41ChatTokenizer(tok, thinking_mode=thinking_mode, reasoning_effort=reasoning_effort)

    assert tok.decode(chat.encode_prompt(messages)) == rendered


def test_v41_tools_without_a_system_message_get_an_empty_system_turn():
    tok = CharTokenizer()
    chat = DeepSeekV41ChatTokenizer(tok)

    prompt = tok.decode(chat.encode_prompt([USER_67], TOOLS))

    assert prompt.startswith(BOS + SYSTEM + "\n\n## Tools\n")
    assert '"<｜DSML｜ calls>" block' in prompt and "<｜DSML｜tool_calls>" not in prompt
    assert prompt.endswith('{"name": "code_interpreter", "parameters": {}}\n\n'
                           "You MUST strictly follow the above defined tool name and parameter schemas "
                           "to invoke tool calls.\n" + USER + "6*7?" + ASSISTANT + THINK_END)


@pytest.mark.parametrize("reasoning_effort", [0, 101, "medium", 7.5, True])
def test_v41_rejects_a_reasoning_effort_outside_its_budget(reasoning_effort):
    with pytest.raises(ValueError, match="reasoning_effort"):
        DeepSeekV41ChatTokenizer(CharTokenizer(), thinking_mode="thinking", reasoning_effort=reasoning_effort)


TAGS = pytest.mark.parametrize("tags", [V4_TAGS, V41_TAGS], ids=["v4", "v41"])


def dsml_call(name: str, *params: tuple[str, str, str], tags: DsmlTags = V4_TAGS) -> str:
    body = "".join(
        f'<｜DSML｜{tags.parameter} name="{k}" string="{s}">{v}</｜DSML｜{tags.parameter}>\n' for k, s, v in params
    )
    return f'<｜DSML｜{tags.invoke} name="{name}">\n{body}</｜DSML｜{tags.invoke}>\n'


def tool_calls(*calls: str, tags: DsmlTags = V4_TAGS) -> str:
    return f"\n\n<｜DSML｜{tags.calls}>\n" + "".join(calls) + f"</｜DSML｜{tags.calls}>" + EOS


@TAGS
def test_parser_reads_string_and_json_parameters_of_every_call(tags):
    text = "Let me run it." + tool_calls(
        dsml_call("code_interpreter", ("code", "true", 'print("a < b")\nx = 1'), tags=tags),
        dsml_call("search", ("q", "true", "x"), ("n", "false", "3"), ("opts", "false", '{"a": [1]}'), tags=tags),
        tags=tags,
    )

    calls, malformed = DsmlToolParser(tags).parse(text)

    assert calls == [
        ToolCall("code_interpreter", {"code": 'print("a < b")\nx = 1'}),
        ToolCall("search", {"q": "x", "n": 3, "opts": {"a": [1]}}),
    ]
    assert malformed == 0


BAD_CALLS = {
    "no-name": lambda tags: dsml_call("", ("code", "true", "1"), tags=tags),
    "bad-json": lambda tags: dsml_call("code_interpreter", ("n", "false", "not json"), tags=tags),
    "repeated-parameter": lambda tags: dsml_call(
        "code_interpreter", ("code", "true", "1"), ("code", "true", "2"), tags=tags,
    ),
    "text-in-invoke": lambda tags: (
        f'<｜DSML｜{tags.invoke} name="code_interpreter">\nstray text\n</｜DSML｜{tags.invoke}>\n'
    ),
    "text-in-block": lambda tags: "not an invoke\n",
}


@TAGS
@pytest.mark.parametrize("bad", list(BAD_CALLS))
def test_parser_counts_malformed_calls(tags, bad):
    text = tool_calls(BAD_CALLS[bad](tags), dsml_call("ok", tags=tags), tags=tags)

    calls, malformed = DsmlToolParser(tags).parse(text)

    assert calls == [ToolCall("ok", {})]
    assert malformed == 1


@TAGS
def test_parser_ignores_an_unclosed_block(tags):
    text = f"\n\n<｜DSML｜{tags.calls}>\n" + dsml_call("code_interpreter", ("code", "true", "1"), tags=tags)

    assert DsmlToolParser(tags).parse(text) == ([], 0)


def test_each_parser_reads_only_its_own_version_of_dsml():
    v4 = tool_calls(dsml_call("code_interpreter", ("code", "true", "1")))
    v41 = tool_calls(dsml_call("code_interpreter", ("code", "true", "1"), tags=V41_TAGS), tags=V41_TAGS)

    assert DsmlToolParser(V4_TAGS).parse(v41) == ([], 0)
    assert DsmlToolParser(V41_TAGS).parse(v4) == ([], 0)
