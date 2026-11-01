"""Append-only chat tokenization and Hermes tool-call parsing."""

from __future__ import annotations

import pytest

from lumenrl.agent.chat import ChatTemplateTokenizer
from lumenrl.agent.tool_parser import HermesToolParser, ToolCall
from tests.unit.agent.fakes import ENDOFTEXT, IM_END, ChatMLTokenizer, tool_call_text

TOOLS = [{"type": "function", "function": {"name": "code_interpreter"}}]
PROMPT = [{"role": "user", "content": "What is 6*7?"}]


def test_prompt_renders_the_tool_schemas():
    tok = ChatMLTokenizer()
    chat = ChatTemplateTokenizer(tok)

    text = tok.decode(chat.encode_prompt(PROMPT, TOOLS))

    assert text.startswith("<|im_start|>system\n# Tools\n")
    assert '"code_interpreter"' in text
    assert text.endswith("<|im_start|>user\nWhat is 6*7?<|im_end|>\n<|im_start|>assistant\n")


def test_tool_round_matches_rendering_the_whole_conversation():
    tok = ChatMLTokenizer()
    chat = ChatTemplateTokenizer(tok)
    turn_text = "Let me compute.\n" + tool_call_text("code_interpreter", {"code": "6*7"})
    turn_ids = tok.encode(turn_text) + [IM_END]

    ids = (
        chat.encode_prompt(PROMPT, TOOLS)
        + turn_ids
        + chat.encode_tool_responses(["42\n", "ok"], after=turn_ids)
    )

    whole = tok.apply_chat_template(
        [
            *PROMPT,
            {"role": "assistant", "content": turn_text},
            {"role": "tool", "content": "42\n"},
            {"role": "tool", "content": "ok"},
        ],
        tools=TOOLS,
        add_generation_prompt=True,
    )
    assert ids == tok.encode(whole)


@pytest.mark.parametrize(
    ("turn_end", "closing"),
    [
        ([IM_END], "\n"),
        ([IM_END, ord("\n")], ""),
        ([ENDOFTEXT], "<|im_end|>\n"),
    ],
)
def test_tool_round_closes_only_what_the_model_left_open(turn_end, closing):
    tok = ChatMLTokenizer()
    chat = ChatTemplateTokenizer(tok)

    text = tok.decode(chat.encode_tool_responses(["42"], after=tok.encode("x") + turn_end))

    assert text == (
        closing + "<|im_start|>user\n<tool_response>\n42\n</tool_response><|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def test_template_kwargs_reach_every_rendering():
    tok = ChatMLTokenizer()
    chat = ChatTemplateTokenizer(tok, chat_template_kwargs={"enable_thinking": False})

    prompt = tok.decode(chat.encode_prompt(PROMPT))
    tool_round = tok.decode(chat.encode_tool_responses(["42"], after=[IM_END]))

    assert prompt.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert tool_round.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert all(kwargs == {"enable_thinking": False} for kwargs in tok.template_kwargs)


def test_template_that_rewrites_the_turn_before_tools_is_rejected():
    chat = ChatTemplateTokenizer(ChatMLTokenizer(rewrite_turn_before_tools=True))

    with pytest.raises(ValueError, match="append-only"):
        chat.encode_tool_responses(["42"], after=[IM_END])


def test_parser_reads_hermes_calls():
    text = (
        "think first\n"
        + tool_call_text("code_interpreter", {"code": "print(1)"})
        + "\n"
        + '<tool_call>{"name": "search", "arguments": "{\\"q\\": \\"x\\"}"}</tool_call>'
    )

    calls, malformed = HermesToolParser().parse(text)

    assert calls == [
        ToolCall("code_interpreter", {"code": "print(1)"}),
        ToolCall("search", {"q": "x"}),
    ]
    assert malformed == 0


@pytest.mark.parametrize(
    "block",
    [
        "not json",
        '["code_interpreter"]',
        '{"arguments": {}}',
        '{"name": "code_interpreter", "arguments": [1]}',
        '{"name": "code_interpreter", "arguments": "{broken"}',
    ],
)
def test_parser_counts_malformed_calls(block):
    text = f"<tool_call>{block}</tool_call>" + tool_call_text("ok", {})

    calls, malformed = HermesToolParser().parse(text)

    assert calls == [ToolCall("ok", {})]
    assert malformed == 1


def test_parser_ignores_an_unclosed_call():
    assert HermesToolParser().parse('<tool_call>{"name": "x", "arguments": {}}') == ([], 0)
