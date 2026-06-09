"""HuggingFace generation-tag loss mask parser.

Uses HF's ``apply_chat_template(return_assistant_tokens_mask=True)`` with
``{% generation %}`` / ``{% endgeneration %}`` Jinja2 tags to produce
assistant-only loss masks.  Mirrors Model-Optimizer's ``LanguageDataCollator``
approach.

When the tokenizer's chat template does NOT include generation tags, falls back
to all-ones loss mask (loss on all non-pad tokens), matching
Model-Optimizer's ``answer_only_loss=false`` behavior.
"""

import logging
import os
from typing import Dict, List, Optional, Tuple

import torch
from transformers import PreTrainedTokenizer

from lumenrl.data.kimi_k25_parser import normalize_conversation

logger = logging.getLogger(__name__)


class HFGenerationParser:
    """Loss mask parser using HF's native chat template mechanism.

    Behavior:
    - Template has ``{% generation %}`` tags → assistant-only loss mask
      via ``return_assistant_tokens_mask=True``.
    - Template exists but no generation tags → tokenize with template,
      all-ones loss mask (equivalent to Model-Optimizer answer_only_loss=false).
    - No template at all → plain concatenation, all-ones loss mask.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        chat_template_override: Optional[str] = None,
    ):
        self.tokenizer = tokenizer

        if chat_template_override:
            if os.path.isfile(chat_template_override):
                with open(chat_template_override) as f:
                    self.tokenizer.chat_template = f.read()
            else:
                self.tokenizer.chat_template = chat_template_override

        template = getattr(self.tokenizer, "chat_template", None) or ""
        self._has_generation_tags = (
            "generation" in template and "endgeneration" in template
        )

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def parse(
        self,
        conversation: List[Dict],
        max_length: int,
        last_turn_only: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize and compute loss mask.

        Returns ``(input_ids, loss_mask)`` — both 1-D ``LongTensor``.
        """
        conversation = normalize_conversation(conversation)

        if not self.tokenizer.chat_template:
            return self._tokenize_plain(conversation, max_length)

        if self._has_generation_tags:
            return self._tokenize_with_generation_tags(
                conversation, max_length, last_turn_only
            )

        return self._tokenize_without_generation_tags(conversation, max_length)

    # ------------------------------------------------------------------
    # Tokenization strategies
    # ------------------------------------------------------------------

    def _tokenize_with_generation_tags(
        self,
        conversation: List[Dict],
        max_length: int,
        last_turn_only: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """``{% generation %}`` tags present — use ``return_assistant_tokens_mask``."""
        result = self.tokenizer.apply_chat_template(
            conversation,
            return_dict=True,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            return_assistant_tokens_mask=True,
            add_generation_prompt=False,
        )
        input_ids = result["input_ids"].squeeze(0)
        assistant_masks = result.get("assistant_masks")

        if assistant_masks is not None:
            if isinstance(assistant_masks, torch.Tensor):
                loss_mask = assistant_masks.squeeze(0).long()
            elif isinstance(assistant_masks, list):
                flat = assistant_masks
                if flat and isinstance(flat[0], list):
                    flat = flat[0]
                loss_mask = torch.tensor(flat, dtype=torch.long)
            else:
                loss_mask = torch.ones(len(input_ids), dtype=torch.long)
        else:
            loss_mask = torch.ones(len(input_ids), dtype=torch.long)

        if len(loss_mask) > len(input_ids):
            loss_mask = loss_mask[: len(input_ids)]
        elif len(loss_mask) < len(input_ids):
            loss_mask = torch.cat(
                [loss_mask, torch.zeros(len(input_ids) - len(loss_mask), dtype=torch.long)]
            )

        if last_turn_only:
            loss_mask = _keep_last_span_only(loss_mask)

        return input_ids, loss_mask

    def _tokenize_without_generation_tags(
        self,
        conversation: List[Dict],
        max_length: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Template exists but no generation tags — all-ones loss mask."""
        text = self.tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=False
        )
        encoding = self.tokenizer(
            text,
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = encoding.input_ids.squeeze(0)
        loss_mask = torch.ones(len(input_ids), dtype=torch.long)
        return input_ids, loss_mask

    def _tokenize_plain(
        self,
        conversation: List[Dict],
        max_length: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """No chat template — concatenate content, all-ones loss mask."""
        text = "\n".join(
            m.get("content", "") for m in conversation if isinstance(m, dict)
        )
        if not text.strip():
            text = " "
        encoding = self.tokenizer(
            text,
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = encoding.input_ids.squeeze(0)
        loss_mask = torch.ones(len(input_ids), dtype=torch.long)
        return input_ids, loss_mask


# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------


def _keep_last_span_only(loss_mask: torch.Tensor) -> torch.Tensor:
    """Zero out all but the last contiguous span of 1s in *loss_mask*."""
    if loss_mask.sum() == 0:
        return loss_mask

    mask_list = loss_mask.tolist()
    last_end = len(mask_list)
    while last_end > 0 and mask_list[last_end - 1] == 0:
        last_end -= 1
    if last_end == 0:
        return loss_mask

    last_start = last_end - 1
    while last_start > 0 and mask_list[last_start - 1] == 1:
        last_start -= 1

    result = torch.zeros_like(loss_mask)
    result[last_start:last_end] = 1
    return result
