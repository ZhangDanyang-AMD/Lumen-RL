"""Eagle3 draft model for speculative decoding.

Adapted from TorchSpec's Eagle3Model.  The model iteratively predicts future
tokens by refining representations using teacher hidden states and a shared
embedding + lm_head from the target model.

Architecture
------------
At each speculative step *i* (0..length-1):

1. Combine the current token embedding with the teacher hidden state via a
   learned fusion layer.
2. Pass through a lightweight Transformer block.
3. Project to vocabulary logits via the shared teacher ``lm_head``.
4. The predicted token embedding feeds back into the next step.
"""

from __future__ import annotations

import logging
import math
import sys
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)

_flex_attn_module = None


def _get_flex_attention():
    global _flex_attn_module
    if _flex_attn_module is not None:
        return _flex_attn_module
    try:
        sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "third_party"))
        from aiter.flex_attention import FlexMultiheadAttention, create_causal_block_mask
        _flex_attn_module = (FlexMultiheadAttention, create_causal_block_mask)
        logger.info("Using Triton FlexAttention for Eagle3 training")
    except ImportError:
        _flex_attn_module = (None, None)
        logger.info("FlexAttention not available, using nn.MultiheadAttention")
    return _flex_attn_module


class Eagle3FusionLayer(nn.Module):
    """Fuses token embedding and teacher hidden state into a single vector."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim, bias=False)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, token_embed: Tensor, teacher_hidden: Tensor) -> Tensor:
        return self.norm(self.fc(torch.cat([token_embed, teacher_hidden], dim=-1)))


class Eagle3TransformerBlock(nn.Module):
    """Single Transformer block with pre-norm and SwiGLU FFN."""

    def __init__(self, hidden_dim: int, num_heads: int, ffn_dim: Optional[int] = None) -> None:
        super().__init__()
        ffn_dim = ffn_dim or hidden_dim * 4
        self.attn_norm = nn.LayerNorm(hidden_dim)
        FlexMHA, _ = _get_flex_attention()
        if FlexMHA is not None:
            self.attn = FlexMHA(hidden_dim, num_heads)
            self._use_flex = True
        else:
            self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
            self._use_flex = False
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.w1 = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.w2 = nn.Linear(ffn_dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, ffn_dim, bias=False)

    def forward(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor] = None,
        block_mask=None,
    ) -> Tensor:
        h = self.attn_norm(x)
        if self._use_flex:
            h, _ = self.attn(h, h, h, block_mask=block_mask)
        else:
            h, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + h
        h = self.ffn_norm(x)
        x = x + self.w2(F.silu(self.w1(h)) * self.w3(h))
        return x


class Eagle3Model(nn.Module):
    """Eagle3 speculative decoding draft model.

    Parameters
    ----------
    hidden_dim : int
        Dimension matching the target model's hidden size.
    num_heads : int
        Number of attention heads in the Transformer blocks.
    num_layers : int
        Number of stacked Transformer blocks.
    length : int
        Number of speculative prediction steps.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 16,
        num_layers: int = 1,
        length: int = 5,
        ffn_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.length = length

        self.fusion = Eagle3FusionLayer(hidden_dim)
        self.blocks = nn.ModuleList(
            [Eagle3TransformerBlock(hidden_dim, num_heads, ffn_dim=ffn_dim)
             for _ in range(num_layers)]
        )
        self.out_norm = nn.LayerNorm(hidden_dim)
        _, self._create_block_mask = _get_flex_attention()
        self._use_flex = self._create_block_mask is not None

    def forward(
        self,
        token_embeds: Tensor,
        teacher_hidden: Tensor,
        lm_head_weight: Tensor,
        loss_mask: Optional[Tensor] = None,
        target_ids: Optional[Tensor] = None,
    ) -> dict[str, Tensor]:
        """Run iterative speculative prediction.

        Args:
            token_embeds: ``[B, T, D]`` embeddings of the input tokens.
            teacher_hidden: ``[B, T, D]`` teacher model hidden states.
            lm_head_weight: ``[V, D]`` weight matrix of the teacher's lm_head.
            loss_mask: ``[B, T]`` mask for valid positions.
            target_ids: ``[B, T + length]`` ground-truth token ids for loss
                computation.  If *None*, only logits are returned.

        Returns:
            Dict with ``"logits_list"`` (list of ``[B, T, V]`` per step),
            and optionally ``"losses"`` and ``"accuracies"`` per step.
        """
        B, T, D = token_embeds.shape
        logits_list: list[Tensor] = []
        losses: list[Tensor] = []
        accuracies: list[Tensor] = []

        block_mask = None
        if self._use_flex:
            block_mask = self._create_block_mask(B, T, token_embeds.device)

        h = self.fusion(token_embeds, teacher_hidden)

        for step in range(self.length):
            for block in self.blocks:
                h = block(h, block_mask=block_mask)

            normed = self.out_norm(h)
            step_logits = F.linear(normed, lm_head_weight)  # [B, T, V]
            logits_list.append(step_logits)

            if target_ids is not None:
                shift = step + 1
                T_ids = target_ids.shape[1]
                if T_ids > T:
                    tgt = target_ids[:, shift : shift + T]
                else:
                    tgt = target_ids[:, shift:]
                    step_logits_for_loss = step_logits[:, : tgt.shape[1]]
                if T_ids > T:
                    step_logits_for_loss = step_logits

                if tgt.shape[1] > 0:
                    ce = F.cross_entropy(
                        step_logits_for_loss.reshape(-1, step_logits_for_loss.size(-1)),
                        tgt.reshape(-1),
                        reduction="none",
                    ).reshape(tgt.shape[0], tgt.shape[1])

                    if loss_mask is not None:
                        mask_f = loss_mask[:, :tgt.shape[1]].to(dtype=ce.dtype)
                        denom = mask_f.sum().clamp(min=1.0)
                        ce = ce.masked_fill(~mask_f.bool(), 0.0)
                        losses.append((ce * mask_f).sum() / denom)
                        preds = step_logits_for_loss.argmax(dim=-1)
                        correct = (preds == tgt).float()
                        accuracies.append((correct * mask_f).sum() / denom)
                    else:
                        losses.append(ce.mean())
                        accuracies.append(
                            (step_logits_for_loss.argmax(dim=-1) == tgt).float().mean()
                        )

            # Feed back: use predicted token embedding for next step
            next_token = step_logits.argmax(dim=-1)  # [B, T]
            # Re-embed using lm_head weight as an embedding matrix (tied weights)
            next_embed = F.embedding(next_token, lm_head_weight)  # [B, T, D]
            h = self.fusion(next_embed, teacher_hidden)

        result: dict[str, Tensor] = {"logits_list": logits_list}
        if losses:
            result["losses"] = losses
            result["accuracies"] = accuracies
        return result
