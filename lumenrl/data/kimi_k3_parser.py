"""Kimi-K3 XTML chat template parser and loss mask computation.

K3 uses a nested XTML markup format with structural tokens:
  <|open|>message role="assistant"<|sep|>
    <|open|>think<|sep|>...<|close|>think<|sep|>
    <|open|>response<|sep|>...<|close|>response<|sep|>
  <|close|>message<|sep|><|end_of_msg|>

This is completely different from K2.5's flat <|im_*|> format.
Formatting delegates to K3's native tokenizer.apply_chat_template()
(encoding_k3.build_chat_segments), which handles all XTML rendering.

The parser extracts <think>...</think> from content into the
reasoning_content field that encoding_k3 expects, then computes
loss mask over the entire assistant message interior (think + response).
"""

import re
from typing import Dict, List, Tuple

import torch
from transformers import PreTrainedTokenizer


def _has_dropped_think_opener(content: str) -> bool:
    return "</think>" in content and not content.lstrip().startswith("<think>")


_THINK_PATTERN = re.compile(r"<think>([\s\S]*?)</think>\s*")

_REASONING_FIELDS = ("thinking", "thinking_content", "reasoning_content", "reasoning")


class KimiK3Parser:
    """Parser for Kimi-K3 with XTML chat format.

    Formatting: delegates to tokenizer.apply_chat_template() which invokes
    encoding_k3.build_chat_segments(). Requires K3's TikTokenTokenizer
    loaded with trust_remote_code=True.

    Loss mask: covers the entire assistant message interior (think + response).

    Interface matches KimiK25Parser: format(conversation) -> str,
    parse(conversation, max_length, last_turn_only) -> (input_ids, loss_mask).
    """

    ASSISTANT_HEADER = '<|open|>message role="assistant"<|sep|>'
    ASSISTANT_END = '<|close|>message<|sep|><|end_of_msg|>'

    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer
        if not hasattr(tokenizer, "apply_chat_template"):
            raise ValueError(
                "KimiK3Parser requires a tokenizer with apply_chat_template(). "
                "Load with AutoTokenizer.from_pretrained(..., trust_remote_code=True)."
            )

    @staticmethod
    def _extract_thinking(content: str) -> Tuple[str, str]:
        """Split <think>...</think> from content into (reasoning, remainder).

        Handles:
        - Normal: <think>reasoning</think>answer -> ("reasoning", "answer")
        - Dropped opener: reasoning</think>answer -> ("reasoning", "answer")
        - Empty thinking: <think></think>answer -> ("", "answer")
        - No thinking: answer -> ("", "answer")
        """
        if _has_dropped_think_opener(content):
            content = "<think>" + content
        match = _THINK_PATTERN.match(content)
        if match:
            reasoning = match.group(1).strip()
            remainder = content[match.end():]
            return reasoning, remainder
        return "", content

    @staticmethod
    def _reasoning_from_field(msg: dict) -> str:
        for field in _REASONING_FIELDS:
            value = msg.get(field)
            if value:
                return value
        return ""

    def _split_reasoning(self, entry: Dict) -> Tuple[str, str]:
        """Reasoning and remaining content for one assistant entry.

        Prefers an explicit reasoning field over an inline ``<think>`` block, but
        strips the inline block either way so it never reaches the renderer as
        literal text (``<think>`` is not a K3 special token — it encodes as three
        ordinary tokens).
        """
        content = entry.get("content", "")
        existing = self._reasoning_from_field(entry)
        if existing:
            if "<think>" in content:
                _, content = self._extract_thinking(content)
            return existing, content
        return self._extract_thinking(content)

    @staticmethod
    def _set_reasoning(entry: Dict, reasoning: str, content: str) -> None:
        """Put reasoning where encoding_k3 looks for it, and nowhere else."""
        entry["content"] = content
        if reasoning:
            entry["reasoning_content"] = reasoning
        else:
            entry.pop("reasoning_content", None)
        for field in _REASONING_FIELDS:
            if field != "reasoning_content" and field in entry:
                del entry[field]

    def _prepare_messages(
        self, conversation: List[Dict], thinking: bool = True
    ) -> List[Dict]:
        """Prepare messages for K3's apply_chat_template.

        The last assistant turn always keeps its reasoning in
        ``reasoning_content``. What happens to *earlier* assistant turns depends
        on the mode, and the distinction is not cosmetic:

        - thinking=True: reasoning is preserved. K3 was trained in preserved
          thinking history mode and its model card requires the full assistant
          message — reasoning_content included — to be replayed. Dropping it
          leaves empty think blocks in the history, which is a context the model
          never saw in training.
        - thinking=False: the think channel is not rendered at all, so reasoning
          has nowhere to go and is stripped.
        """
        last_assistant_idx = max(
            (i for i, msg in enumerate(conversation) if isinstance(msg, dict) and msg.get("role") == "assistant"),
            default=-1,
        )

        messages = []
        for idx, msg in enumerate(conversation):
            entry = dict(msg)
            if entry.get("role") != "assistant":
                messages.append(entry)
                continue

            reasoning, content = self._split_reasoning(entry)
            keep = thinking or idx == last_assistant_idx
            self._set_reasoning(entry, reasoning if keep else "", content)
            messages.append(entry)

        return messages

    def format(
        self,
        conversation: List[Dict],
        add_generation_prompt: bool = False,
        thinking: bool = True,
    ) -> str:
        """Format conversation into XTML string via K3's native tokenizer."""
        messages = self._prepare_messages(conversation, thinking=thinking)
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            thinking=thinking,
            add_generation_prompt=add_generation_prompt,
        )

    def format_generation_prompt(
        self, conversation: List[Dict], thinking: bool = True
    ) -> str:
        """Render the context that precedes the last assistant turn, as a prompt.

        This is what an on-policy teacher is asked to complete: the same bytes a
        serving stack would send, so the response the teacher produces carries no
        train/inference rendering skew.

        Reasoning in the history turns follows the mode, for the same reason as
        in :meth:`_prepare_messages`: preserved under thinking=True because K3
        requires the full assistant message replayed, stripped under
        thinking=False because there is no think channel to put it in.
        """
        last_assistant_idx = max(
            (
                i
                for i, msg in enumerate(conversation)
                if isinstance(msg, dict) and msg.get("role") == "assistant"
            ),
            default=-1,
        )
        if last_assistant_idx < 0:
            return ""

        history = []
        for msg in conversation[:last_assistant_idx]:
            entry = dict(msg)
            if entry.get("role") == "assistant":
                reasoning, content = self._split_reasoning(entry)
                self._set_reasoning(entry, reasoning if thinking else "", content)
            history.append(entry)

        if not history:
            return ""

        return self.tokenizer.apply_chat_template(
            history,
            tokenize=False,
            thinking=thinking,
            add_generation_prompt=True,
        )

    def parse_generation_prompt(
        self, conversation: List[Dict], thinking: bool = True
    ) -> torch.Tensor:
        """Token ids of ``format_generation_prompt``; empty tensor when absent."""
        text = self.format_generation_prompt(conversation, thinking=thinking)
        if not text:
            return torch.zeros(0, dtype=torch.long)
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        return torch.tensor(ids, dtype=torch.long)

    def parse(
        self,
        conversation: List[Dict],
        max_length: int,
        last_turn_only: bool = False,
        thinking: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Format, tokenize, and compute assistant-only loss mask."""
        text = self.format(conversation, thinking=thinking)
        return self._tokenize_with_loss_mask(text, max_length, last_turn_only)

    def _tokenize_with_loss_mask(
        self,
        text: str,
        max_length: int,
        last_turn_only: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize text and compute loss mask via encode-prefix character mapping.

        Only content between ASSISTANT_HEADER and ASSISTANT_END gets loss_mask=1.
        This captures the entire assistant message interior including both
        <|open|>think<|sep|>...<|close|>think<|sep|> and
        <|open|>response<|sep|>...<|close|>response<|sep|>.
        """
        encoding = self.tokenizer(
            text,
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = encoding.input_ids[0]
        loss_mask = torch.zeros(len(input_ids), dtype=torch.long)

        assistant_pattern = (
            re.escape(self.ASSISTANT_HEADER)
            + r"([\s\S]*?)"
            + re.escape(self.ASSISTANT_END)
        )
        matches = list(re.finditer(assistant_pattern, text))
        if last_turn_only and matches:
            matches = matches[-1:]

        for match in matches:
            content_start_char = match.start(1)
            content_end_char = match.end(1)

            prefix_ids = self.tokenizer.encode(
                text[:content_start_char], add_special_tokens=False
            )
            full_ids = self.tokenizer.encode(
                text[:content_end_char], add_special_tokens=False
            )

            start_token_idx = len(prefix_ids)
            end_token_idx = len(full_ids)

            actual_start = min(start_token_idx, len(input_ids))
            actual_end = min(end_token_idx, len(input_ids))

            if actual_start < actual_end:
                loss_mask[actual_start:actual_end] = 1

        return input_ids, loss_mask
