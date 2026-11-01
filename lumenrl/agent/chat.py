"""Token ids of a multi-turn conversation, built append-only from a chat template."""

from __future__ import annotations

from typing import Any

# A user turn to render tool messages after. Only the text the template adds
# after it is kept, so its content never reaches the model.
_ANCHOR_CONTENT = "<lumenrl-anchor>"
_ANCHOR = ({"role": "user", "content": _ANCHOR_CONTENT},)


class ChatTemplateTokenizer:
    """Render a rollout's prompt once, then append each round of tool output.

    The token ids the model generated are never re-tokenized. Re-rendering the
    whole conversation would not reproduce them: Qwen3's template, for one,
    drops the thinking of earlier turns. So each round of tool messages is
    rendered on its own, after a placeholder user turn, and only the tokens the
    template adds for the tool messages and the next generation prompt are
    appended.
    """

    def __init__(self, tokenizer: Any, chat_template_kwargs: dict[str, Any] | None = None) -> None:
        self._tokenizer = tokenizer
        self._template_kwargs = dict(chat_template_kwargs or {})
        anchor = self._render(list(_ANCHOR), add_generation_prompt=False)
        self._anchor_text = anchor
        # What the template writes after a turn's content, e.g. "<|im_end|>\n".
        # Generation stops at its first token; the rest is environment.
        self._turn_end_ids = self.encode_text(anchor[anchor.rindex(_ANCHOR_CONTENT) + len(_ANCHOR_CONTENT):])
        if not self._turn_end_ids:
            raise ValueError("chat template writes nothing after a turn; cannot find the turn boundary")

    def _render(
        self,
        messages: list[dict[str, Any]],
        *,
        add_generation_prompt: bool,
        tools: list[dict[str, Any]] | None = None,
    ) -> str:
        return self._tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            **self._template_kwargs,
        )

    def encode_text(self, text: str) -> list[int]:
        return list(self._tokenizer(text, add_special_tokens=False)["input_ids"])

    def decode(self, token_ids: list[int]) -> str:
        return self._tokenizer.decode(token_ids, skip_special_tokens=False)

    def encode_prompt(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None,
    ) -> list[int]:
        return self.encode_text(
            self._render(messages, add_generation_prompt=True, tools=tools or None),
        )

    def encode_tool_responses(self, contents: list[str], after: list[int]) -> list[int]:
        """Environment tokens between the model turn ``after`` and the next one."""
        rendered = self._render(
            [*_ANCHOR, *({"role": "tool", "content": c} for c in contents)],
            add_generation_prompt=True,
        )
        if not rendered.startswith(self._anchor_text):
            raise ValueError(
                "chat template renders the turn before tool messages differently "
                "when tool messages follow it; append-only tokenization needs it unchanged"
            )
        return self._close_turn(after) + self.encode_text(rendered[len(self._anchor_text):])

    def _close_turn(self, after: list[int]) -> list[int]:
        """The part of the turn terminator the model did not generate."""
        end = self._turn_end_ids
        for written in range(min(len(end), len(after)), 0, -1):
            if after[-written:] == end[:written]:
                return end[written:]
        return list(end)
