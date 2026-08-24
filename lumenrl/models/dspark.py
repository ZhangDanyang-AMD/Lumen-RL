"""DSpark draft model for speculative decoding.

Architecture aligned with TorchSpec/vLLM Inferact/Kimi-K3-DSpark:
- 5-layer parallel backbone with MLA dual-source KV attention
- VanillaMarkov head (rank=256) for transition bias
- AcceptRatePredictor confidence head
- Anchor-based block training (block_size=7)

Dual-source KV: draft tokens attend to both context (full-sequence target
hidden states) and draft (self) via shared KV projections, matching TorchSpec's
DFlashAttention architecture.  Intra-block attention is bidirectional.

Loss: L = ce_alpha * CE + l1_alpha * TV + conf_alpha * BCE
Each weighted by position decay w_k = exp(-k / gamma).
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from lumenrl.models.eagle3 import (
    RMSNorm,
    RotaryEmbedding,
    _yarn_get_mscale,
)

logger = logging.getLogger(__name__)


def _rotate_half_interleaved(x: Tensor) -> Tensor:
    """Interleaved-pair rotation: pairs dims (0,1), (2,3), ..."""
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


try:
    from torch.nn.attention.flex_attention import (
        create_block_mask as _create_block_mask,
        flex_attention as _flex_attention,
    )
    # flex_attention only fuses under torch.compile. Called eagerly it warns and
    # materialises the score matrix anyway -- measured at 33.7 GiB against masked
    # SDPA's 24.6 GiB on the shapes this model runs, i.e. strictly worse than what
    # it replaces. There is no useful eager path, so if compilation is unavailable
    # we fall back to SDPA instead.
    #
    # dynamic is left at its default (automatic) on purpose, as TorchSpec does.
    # Batches are padded to their own longest sequence, so KV_LEN changes almost
    # every step; dynamic=False recompiles per shape, blows the dynamo cache
    # limit, and then permanently degrades to the eager path this is meant to
    # avoid. Automatic mode specialises once, then recompiles dynamic and reuses.
    _flex_attention_compiled = torch.compile(_flex_attention)
    _HAS_FLEX = True
except ImportError:  # torch too old
    _create_block_mask = None
    _flex_attention_compiled = None
    _HAS_FLEX = False


class DSparkMLAAttention(nn.Module):
    """Multi-head Latent Attention (MLA) with dual-source KV for DSpark.

    Aligned with TorchSpec's DFlashAttention:
    - Q comes only from draft hidden states
    - K/V come from BOTH context (target hidden states) AND draft (self)
    - Context and draft KV are concatenated along sequence dimension

    Low-rank Q/KV compression matching Inferact/Kimi-K3-DSpark:
    - Q path: hidden -> q_a_proj -> q_a_layernorm -> q_b_proj -> split(nope, rope)
    - KV path: hidden -> kv_a_proj_with_mqa -> split(kv_compressed, k_rope) ->
               kv_a_layernorm(kv_compressed) -> kv_b_proj -> split(k_nope, v)
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        q_lora_rank: int = 1536,
        kv_lora_rank: int = 512,
        qk_nope_head_dim: int = 128,
        qk_rope_head_dim: int = 64,
        v_head_dim: int = 128,
        rms_norm_eps: float = 1e-5,
        rope_theta: float = 50000.0,
        max_position_embeddings: int = 1048576,
        rope_scaling: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim

        # Q path
        self.q_a_proj = nn.Linear(hidden_size, q_lora_rank, bias=False)
        self.q_a_layernorm = RMSNorm(q_lora_rank, eps=rms_norm_eps)
        self.q_b_proj = nn.Linear(
            q_lora_rank, num_heads * self.qk_head_dim, bias=False,
        )

        # KV path (shared between context and draft)
        self.kv_a_proj_with_mqa = nn.Linear(
            hidden_size, kv_lora_rank + qk_rope_head_dim, bias=False,
        )
        self.kv_a_layernorm = RMSNorm(kv_lora_rank, eps=rms_norm_eps)
        self.kv_b_proj = nn.Linear(
            kv_lora_rank,
            num_kv_heads * (qk_nope_head_dim + v_head_dim),
            bias=False,
        )

        # Output
        self.o_proj = nn.Linear(num_heads * v_head_dim, hidden_size, bias=False)

        # RoPE
        rope_cfg = rope_scaling or {}
        self.rotary_emb = RotaryEmbedding(
            dim=qk_rope_head_dim,
            max_position_embeddings=max_position_embeddings,
            base=rope_theta,
            scaling_factor=rope_cfg.get("factor", 1.0),
            original_max_position_embeddings=rope_cfg.get(
                "original_max_position_embeddings", 32768,
            ),
            beta_fast=rope_cfg.get("beta_fast", 32.0),
            beta_slow=rope_cfg.get("beta_slow", 1.0),
            mscale=rope_cfg.get("mscale", 1.0),
            mscale_all_dim=rope_cfg.get("mscale_all_dim", 1.0),
            rope_type=rope_cfg.get("rope_type", "yarn"),
        )

        # YaRN widens the softmax scale by mscale^2, and both consumers of this
        # checkpoint do it: TorchSpec's _compute_softmax_scale and ATOM's
        # kimi_k3_dspark (`self.scaling * mscale * mscale`). Omitting it trains
        # the draft 1.81x colder than it is served at factor=32 /
        # mscale_all_dim=1.0, which no serving flag can undo without also
        # disabling the term everywhere else.
        softmax_scale = 1.0 / math.sqrt(self.qk_head_dim)
        if rope_cfg.get("rope_type", "yarn") == "yarn":
            ms = _yarn_get_mscale(
                float(rope_cfg.get("factor", 1.0)),
                float(rope_cfg.get("mscale_all_dim", 1.0)),
            )
            softmax_scale *= ms * ms
        self._softmax_scale = softmax_scale

    def _project_kv(
        self, hidden: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Project hidden states to K components + V. Does NOT apply RoPE.

        Returns:
            k_nope: [B, num_kv_heads, T, qk_nope_head_dim]
            k_rope_raw: [B, 1, T, qk_rope_head_dim] — raw, pre-RoPE
            v: [B, num_kv_heads, T, v_head_dim]
        """
        B, T, _ = hidden.shape

        kv_combined = self.kv_a_proj_with_mqa(hidden)
        kv_compressed, k_rope_raw = kv_combined.split(
            [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1,
        )
        kv_compressed = self.kv_a_layernorm(kv_compressed)
        kv = self.kv_b_proj(kv_compressed)
        kv = kv.view(B, T, self.num_kv_heads, self.qk_nope_head_dim + self.v_head_dim)
        kv = kv.transpose(1, 2)
        k_nope, v = kv.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        k_rope_raw = k_rope_raw.view(B, T, 1, self.qk_rope_head_dim).transpose(1, 2)

        return k_nope, k_rope_raw, v

    def _apply_rope_by_position(
        self, x: Tensor, position_ids: Tensor,
    ) -> Tensor:
        """Apply RoPE using explicit position IDs (not sequential range).

        Args:
            x: [B, heads, T, rope_dim]
            position_ids: [B, T] — actual position indices

        Returns:
            x with RoPE applied: [B, heads, T, rope_dim]
        """
        max_pos = int(position_ids.max().item()) + 1
        self.rotary_emb._update_cache(max_pos, x.device)
        cos = self.rotary_emb._cos_cached.squeeze(0).squeeze(0).to(x.dtype)
        sin = self.rotary_emb._sin_cached.squeeze(0).squeeze(0).to(x.dtype)
        cos_pos = cos[position_ids].unsqueeze(1)
        sin_pos = sin[position_ids].unsqueeze(1)
        # DeepSeek/Kimi MLA rotates consecutive pairs (0,1), (2,3), ..., not
        # NeoX's first-half/second-half. The cache is built neox-layout
        # ([θ0..θ31, θ0..θ31]), so take one half and duplicate each entry to get
        # the interleaved layout [θ0,θ0, θ1,θ1, ...]. Serving this checkpoint
        # under the other convention costs ~20 points of acceptance rate.
        half = cos_pos.shape[-1] // 2
        cos_pos = cos_pos[..., :half].repeat_interleave(2, dim=-1)
        sin_pos = sin_pos[..., :half].repeat_interleave(2, dim=-1)
        return (x * cos_pos) + (_rotate_half_interleaved(x) * sin_pos)

    def forward(
        self,
        draft_hidden: Tensor,
        context_hidden: Tensor,
        draft_position_ids: Tensor,
        context_position_ids: Tensor,
        attn_mask: Optional[Tensor] = None,
        block_mask=None,
        keep_rows: Optional[Tensor] = None,
    ) -> Tensor:
        """Dual-source KV attention (aligned with TorchSpec DFlashAttention).

        Q comes from draft only. K/V come from both context and draft via
        shared MLA projections. RoPE is applied using position-specific
        indexing into the cos/sin cache.

        Args:
            draft_hidden: [B, draft_len, H] — hidden states of draft tokens
            context_hidden: [B, ctx_len, H] — context features from target
            draft_position_ids: [B, draft_len] — position IDs for draft
            context_position_ids: [B, ctx_len] — position IDs for context
            attn_mask: [B, 1, draft_len, ctx_len+draft_len] — bool True=attend
            block_mask: the same visibility as a flex_attention BlockMask. When
                given it takes precedence and attn_mask is unused.
            keep_rows: [B, 1, draft_len, 1] bool, paired with block_mask. Rows of
                dropped blocks are zeroed after attention, since block_keep_mask
                is left out of the BlockMask to avoid empty rows.
        """
        B, draft_len, _ = draft_hidden.shape
        ctx_len = context_hidden.shape[1]

        # Q only from draft
        q_compressed = self.q_a_layernorm(self.q_a_proj(draft_hidden))
        q = self.q_b_proj(q_compressed)
        q = q.view(B, draft_len, self.num_heads, self.qk_head_dim).transpose(1, 2)
        q_nope, q_rope = q.split(
            [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1,
        )

        # RoPE for Q using position-specific indexing
        q_rope_emb = self._apply_rope_by_position(q_rope, draft_position_ids)

        # K/V from both context and draft (shared projections)
        k_nope_ctx, k_rope_raw_ctx, v_ctx = self._project_kv(context_hidden)
        k_nope_draft, k_rope_raw_draft, v_draft = self._project_kv(draft_hidden)

        # Concatenate K and V along sequence dimension before RoPE
        k_nope = torch.cat([k_nope_ctx, k_nope_draft], dim=2)
        k_rope_raw = torch.cat([k_rope_raw_ctx, k_rope_raw_draft], dim=2)
        v = torch.cat([v_ctx, v_draft], dim=2)

        # RoPE for K using full concatenated position IDs
        full_position_ids = torch.cat(
            [context_position_ids, draft_position_ids], dim=1,
        )
        k_rope_emb = self._apply_rope_by_position(k_rope_raw, full_position_ids)

        # GQA repeat
        num_rep = self.num_heads // self.num_kv_heads
        if num_rep > 1:
            k_nope = k_nope.repeat_interleave(num_rep, dim=1)
            v = v.repeat_interleave(num_rep, dim=1)

        k_rope_emb = k_rope_emb.expand(-1, self.num_heads, -1, -1)

        # Full Q/K by concatenating nope + rope components
        q_full = torch.cat([q_nope, q_rope_emb], dim=-1)
        k_full = torch.cat([k_nope, k_rope_emb], dim=-1)

        # The block mask is the fast path: flex_attention evaluates visibility
        # per block and never forms the [B, H, draft_len, kv_len] score matrix.
        # Measured at anchor_num=512 / ctx=8165 on gfx950: 3.7 ms and 0.72 GiB
        # against masked SDPA's 2379 ms and 24.58 GiB, agreeing to rel L2 2e-03.
        # The dense path below stays for the no-flex case and is what the
        # equivalence test checks against.
        if block_mask is not None:
            attn_output = _flex_attention_compiled(
                q_full, k_full, v.contiguous(),
                block_mask=block_mask,
                scale=self._softmax_scale,
            )
            if keep_rows is not None:
                # Reproduces the 0.0 that masked SDPA returns for a block whose
                # block_keep_mask is False. o_proj has no bias, so zeroing here
                # or after the projection is the same thing.
                attn_output = attn_output * keep_rows
            attn_output = attn_output.transpose(1, 2).reshape(
                B, draft_len, self.num_heads * self.v_head_dim,
            )
            return self.o_proj(attn_output)

        # Dense fallback: SDPA with a boolean mask, which forces the math
        # backend and materialises the full score matrix. Falls back again to
        # chunked per-block computation when that does not fit.
        try:
            attn_output = F.scaled_dot_product_attention(
                q_full, k_full, v,
                attn_mask=attn_mask,
                dropout_p=0.0,
                is_causal=False,
                scale=self._softmax_scale,
            )
        except RuntimeError:
            chunk_size = 64
            chunks = []
            for start in range(0, draft_len, chunk_size):
                end = min(start + chunk_size, draft_len)
                q_slice = q_full[:, :, start:end, :]
                mask_slice = attn_mask[:, :, start:end, :] if attn_mask is not None else None
                w = torch.matmul(q_slice, k_full.transpose(2, 3)) * self._softmax_scale
                if mask_slice is not None:
                    w = w.masked_fill(~mask_slice, torch.finfo(w.dtype).min)
                w = F.softmax(w, dim=-1, dtype=torch.float32).to(q_full.dtype)
                chunks.append(torch.matmul(w, v))
            attn_output = torch.cat(chunks, dim=2)

        attn_output = attn_output.transpose(1, 2).reshape(
            B, draft_len, self.num_heads * self.v_head_dim,
        )

        return self.o_proj(attn_output)


class DSparkMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DSparkDecoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        intermediate_size: int,
        q_lora_rank: int = 1536,
        kv_lora_rank: int = 512,
        qk_nope_head_dim: int = 128,
        qk_rope_head_dim: int = 64,
        v_head_dim: int = 128,
        rms_norm_eps: float = 1e-5,
        rope_theta: float = 50000.0,
        rope_scaling: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.self_attn = DSparkMLAAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            rms_norm_eps=rms_norm_eps,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
        )
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.mlp = DSparkMLP(hidden_size, intermediate_size)

    def forward(
        self,
        draft_hidden: Tensor,
        context_hidden: Tensor,
        draft_position_ids: Tensor,
        context_position_ids: Tensor,
        attn_mask: Optional[Tensor] = None,
        block_mask=None,
        keep_rows: Optional[Tensor] = None,
    ) -> Tensor:
        residual = draft_hidden
        draft_hidden = self.input_layernorm(draft_hidden)
        draft_hidden = self.self_attn(
            draft_hidden=draft_hidden,
            context_hidden=context_hidden,
            draft_position_ids=draft_position_ids,
            context_position_ids=context_position_ids,
            attn_mask=attn_mask,
            block_mask=block_mask,
            keep_rows=keep_rows,
        )
        draft_hidden = residual + draft_hidden

        residual = draft_hidden
        draft_hidden = self.post_attention_layernorm(draft_hidden)
        draft_hidden = self.mlp(draft_hidden)
        draft_hidden = residual + draft_hidden

        return draft_hidden


class VanillaMarkov(nn.Module):
    """Teacher-forced Markov transition bias.

    Computes: markov_bias[i] = markov_w2(markov_w1[prev_token_id[i]])
    """

    def __init__(self, vocab_size: int, rank: int) -> None:
        super().__init__()
        self.markov_w1 = nn.Embedding(vocab_size, rank)
        self.markov_w2 = nn.Linear(rank, vocab_size, bias=False)

    def forward(self, prev_token_ids: Tensor) -> Tensor:
        """prev_token_ids: [B, A, block_size]. Returns: [B, A, block_size, vocab_size]."""
        emb = self.markov_w1(prev_token_ids)
        return self.markov_w2(emb)

    def get_features(self, prev_token_ids: Tensor) -> Tensor:
        """Return intermediate Markov features for confidence head."""
        return self.markov_w1(prev_token_ids)


class AcceptRatePredictor(nn.Module):
    """Predict per-position acceptance probability.

    Target: acceptance_rate = 1 - 0.5 * TV_distance.
    """

    def __init__(self, hidden_size: int, markov_rank: int = 0) -> None:
        super().__init__()
        input_dim = hidden_size + markov_rank if markov_rank > 0 else hidden_size
        self.proj = nn.Linear(input_dim, 1)

    def forward(
        self, hidden_states: Tensor, markov_features: Optional[Tensor] = None,
    ) -> Tensor:
        if markov_features is not None:
            x = torch.cat([hidden_states, markov_features], dim=-1)
        else:
            x = hidden_states
        return self.proj(x).squeeze(-1)


class DSparkModel(nn.Module):
    """DSpark speculative decoding draft model.

    Architecture aligned with TorchSpec/vLLM Inferact/Kimi-K3-DSpark.
    Uses dual-source KV attention: draft tokens attend to both context
    (full-sequence target hidden states) and draft (self) via shared
    MLA KV projections.

    Frozen embed_tokens and lm_head are NOT stored here -- they are
    passed at forward time.
    """

    def __init__(
        self,
        hidden_dim: int,
        vocab_size: int,
        num_heads: int = 64,
        num_kv_heads: int = 64,
        num_layers: int = 5,
        num_target_layers: int = 5,
        block_size: int = 7,
        ffn_dim: int = 14336,
        rms_norm_eps: float = 1e-5,
        rope_theta: float = 50000.0,
        rope_scaling: Optional[dict] = None,
        anchor_num: int = 512,
        markov_rank: int = 256,
        markov_head_type: str = "vanilla",
        enable_confidence_head: bool = True,
        confidence_head_with_markov: bool = True,
        mask_token_id: int = 163837,
        q_lora_rank: int = 1536,
        kv_lora_rank: int = 512,
        qk_nope_head_dim: int = 128,
        qk_rope_head_dim: int = 64,
        v_head_dim: int = 128,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.anchor_num = anchor_num
        self.mask_token_id = mask_token_id
        self.num_target_layers = num_target_layers
        self.markov_rank = markov_rank
        self.enable_confidence_head = enable_confidence_head

        # Fusion: concatenated aux hidden states -> hidden_dim
        self.fc = nn.Linear(hidden_dim * num_target_layers, hidden_dim, bias=False)
        self.hidden_norm = RMSNorm(hidden_dim, eps=rms_norm_eps)

        # Transformer backbone with dual-source KV attention
        self.layers = nn.ModuleList([
            DSparkDecoderLayer(
                hidden_size=hidden_dim,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                intermediate_size=ffn_dim,
                q_lora_rank=q_lora_rank,
                kv_lora_rank=kv_lora_rank,
                qk_nope_head_dim=qk_nope_head_dim,
                qk_rope_head_dim=qk_rope_head_dim,
                v_head_dim=v_head_dim,
                rms_norm_eps=rms_norm_eps,
                rope_theta=rope_theta,
                rope_scaling=rope_scaling,
            )
            for _ in range(num_layers)
        ])

        self.norm = RMSNorm(hidden_dim, eps=rms_norm_eps)

        # Markov head
        if markov_rank > 0 and markov_head_type == "vanilla":
            self.markov_head = VanillaMarkov(vocab_size, markov_rank)
        else:
            self.markov_head = None

        # Confidence head
        if enable_confidence_head:
            mk_rank = markov_rank if confidence_head_with_markov and markov_rank > 0 else 0
            self.confidence_head = AcceptRatePredictor(hidden_dim, mk_rank)
        else:
            self.confidence_head = None

    def _sample_anchors(
        self, seq_len: int, loss_mask: Tensor, num_anchors: int,
    ) -> tuple[Tensor, Tensor]:
        """Sample anchor positions with block_keep_mask (aligned with TorchSpec).

        Returns (anchors [B, A], keep_mask [B, A]).  When fewer than num_anchors
        valid positions exist, excess slots get keep_mask=False.
        """
        B = loss_mask.shape[0]
        device = loss_mask.device
        max_anchor = max(seq_len - self.block_size, 0)

        if max_anchor == 0:
            anchors = torch.zeros(B, num_anchors, dtype=torch.long, device=device)
            keep_mask = torch.zeros(B, num_anchors, dtype=torch.bool, device=device)
            return anchors, keep_mask

        valid = loss_mask[:, : max_anchor + 1] > 0.5
        valid_counts = valid.sum(dim=1)

        indices = torch.arange(max_anchor + 1, device=device).unsqueeze(0).expand(B, -1)
        masked_indices = torch.where(valid, indices, seq_len + 1)

        random_vals = torch.rand(B, max_anchor + 1, device=device)
        random_vals = torch.where(valid, random_vals, 2.0)

        _, sorted_idx = random_vals.sort(dim=1)
        gathered = torch.gather(masked_indices, 1, sorted_idx)

        take_n = min(num_anchors, gathered.shape[1])
        selected = gathered[:, :take_n].sort(dim=1).values
        if take_n < num_anchors:
            pad = torch.zeros(B, num_anchors - take_n, dtype=torch.long, device=device)
            selected = torch.cat([selected, pad], dim=1)
        anchors = selected

        keep_mask = torch.arange(num_anchors, device=device).unsqueeze(0) < valid_counts.unsqueeze(1).clamp(max=num_anchors)
        anchors = torch.where(keep_mask, anchors, torch.zeros_like(anchors))

        return anchors, keep_mask

    def _build_dual_source_mask(
        self,
        anchor_positions: Tensor,
        block_keep_mask: Tensor,
        ctx_len: int,
    ) -> Tensor:
        """Build boolean attention mask for dual-source KV attention.

        KV layout: [Context (ctx_len) | Draft blocks (num_anchors * block_size)]
        Q layout:  [Draft blocks (num_anchors * block_size)]

        Rules (aligned with TorchSpec _create_dflash_mask_mod):
        1. Each block sees context strictly before its anchor (kv_idx < anchor_pos)
        2. Intra-block attention is bidirectional
        3. Different blocks are invisible to each other
        4. Invalid blocks (block_keep_mask=False) see nothing

        Returns: [B, 1, draft_len, kv_len] boolean (True = attend)
        """
        B = anchor_positions.shape[0]
        num_anchors = anchor_positions.shape[1]
        draft_len = num_anchors * self.block_size
        kv_len = ctx_len + draft_len
        device = anchor_positions.device

        q_idx = torch.arange(draft_len, device=device)
        q_block_id = q_idx // self.block_size

        kv_idx = torch.arange(kv_len, device=device)

        masks = []
        for b in range(B):
            anchor_pos_per_q = anchor_positions[b, q_block_id]
            is_context = kv_idx.unsqueeze(0) < ctx_len
            context_visible = is_context & (kv_idx.unsqueeze(0) < anchor_pos_per_q.unsqueeze(1))

            is_draft = kv_idx.unsqueeze(0) >= ctx_len
            kv_draft_offset = kv_idx.unsqueeze(0) - ctx_len
            kv_block_id = kv_draft_offset // self.block_size
            same_block = (q_block_id.unsqueeze(1) == kv_block_id)
            draft_visible = is_draft & same_block

            block_valid = block_keep_mask[b, q_block_id].unsqueeze(1)

            visible = (context_visible | draft_visible) & block_valid
            masks.append(visible)

        return torch.stack(masks).unsqueeze(1)

    def _build_dual_source_block_mask(
        self,
        anchor_positions: Tensor,
        block_keep_mask: Tensor,
        ctx_len: int,
    ):
        """The same visibility as _build_dual_source_mask, as a BlockMask.

        Returns (block_mask, keep_rows) or (None, None) when flex is unavailable.

        NOTE: block_keep_mask is deliberately NOT folded into mask_mod, even though
        _build_dual_source_mask folds it in. An invalid block has anchor 0, so it
        sees no context, and masking its own block too would leave the row
        completely empty. Measured on this build, SDPA answers a fully masked row
        with 0.0 rather than NaN, and the rest of the model is written around
        that; a fused softmax over an empty row is not obliged to agree, and a
        NaN here would survive multiplication by the loss mask and poison the
        gradients.

        So the kernel sees a mask under which every row has at least its own
        block, and the invalid rows are zeroed afterwards instead. That is
        exactly equivalent: block_keep_mask only ever zeroes whole q-blocks, and
        no valid block attends to an invalid one (draft visibility is
        same-block only), so invalid rows cannot influence valid ones either way.
        """
        if not _HAS_FLEX:
            return None, None

        block_size = self.block_size
        # Captured by the closure and lifted into the compiled kernel's inputs.
        anchors = anchor_positions

        def mask_mod(b, h, q_idx, kv_idx):
            q_block = q_idx // block_size
            anchor_pos = anchors[b, q_block]
            context_visible = (kv_idx < ctx_len) & (kv_idx < anchor_pos)
            kv_block = (kv_idx - ctx_len) // block_size
            draft_visible = (kv_idx >= ctx_len) & (q_block == kv_block)
            return context_visible | draft_visible

        B, num_anchors = anchor_positions.shape
        draft_len = num_anchors * block_size

        block_mask = _create_block_mask(
            mask_mod,
            B=B,
            H=None,
            Q_LEN=draft_len,
            KV_LEN=ctx_len + draft_len,
            device=anchor_positions.device,
        )
        # [B, 1, draft_len, 1], broadcasting over heads and the value dim.
        keep_rows = (
            block_keep_mask.repeat_interleave(block_size, dim=1)
            .view(B, 1, draft_len, 1)
        )
        return block_mask, keep_rows

    def forward(
        self,
        input_ids: Tensor,
        token_embeds: Tensor,
        aux_hidden_states: Tensor,
        teacher_lm_head_weight: Tensor,
        embed_weight: Tensor,
        loss_mask: Tensor,
        target_hidden_states: Optional[Tensor] = None,
        ce_alpha: float = 0.1,
        l1_alpha: float = 0.9,
        conf_alpha: float = 1.0,
        decay_gamma: float = 4.0,
    ) -> dict[str, Tensor]:
        """DSpark forward aligned with TorchSpec's DSparkModel.

        Uses dual-source KV attention: draft tokens attend to full-sequence
        context features (from target hidden states) AND self (within block).
        Intra-block attention is bidirectional.

        Label construction follows TorchSpec: slot k predicts input_ids[anchor+k+1].
        Target hidden states are gathered at position anchor+k (hidden at pos p
        predicts token at pos p+1). No pre-shifting of input_ids by caller.

        Args:
            input_ids: [B, T] — raw (unshifted) token IDs
            token_embeds: [B, T, H] — embed(input_ids)
            aux_hidden_states: [B, T, num_target_layers*H] — concatenated aux hidden
            teacher_lm_head_weight: [V, H] — frozen teacher lm_head
            embed_weight: [V, H] — frozen teacher embedding
            loss_mask: [B, T] — 1 for valid positions
            target_hidden_states: [B, T, H] — teacher last hidden (post-norm)
            ce_alpha: Cross-entropy loss weight
            l1_alpha: L1 distribution distillation weight
            conf_alpha: Confidence BCE loss weight
            decay_gamma: Position decay parameter
        """
        B, T = input_ids.shape
        H = self.hidden_dim
        device = input_ids.device
        dtype = token_embeds.dtype

        num_anchors = self.anchor_num

        # 1. Sample anchors with block_keep_mask (aligned with TorchSpec)
        anchors, block_keep_mask = self._sample_anchors(T, loss_mask, num_anchors)

        # 2. Extract context features from full sequence (TorchSpec extract_context_feature)
        context_feature = self.hidden_norm(self.fc(aux_hidden_states))

        # 3. Construct noise embeddings: anchor_token at slot 0, MASK elsewhere
        mask_embed = F.embedding(
            torch.tensor([self.mask_token_id], device=device), embed_weight,
        ).squeeze(0)

        anchor_embeds = torch.gather(
            token_embeds, 1,
            anchors.unsqueeze(-1).expand(-1, -1, H),
        )

        block_input = mask_embed.unsqueeze(0).unsqueeze(0).expand(
            B, num_anchors, self.block_size, -1,
        ).clone()
        block_input[:, :, 0] = torch.where(
            block_keep_mask.unsqueeze(-1), anchor_embeds, mask_embed.unsqueeze(0),
        )

        draft_hidden = block_input.reshape(B, num_anchors * self.block_size, H)

        # 4. Position IDs for draft and context
        offsets = torch.arange(self.block_size, device=device).view(1, 1, -1)
        draft_position_ids = (anchors.unsqueeze(-1) + offsets).reshape(B, -1)
        context_position_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)

        # 5. Dual-source KV attention mask. Built once and shared by all five
        # layers. The BlockMask is preferred; attn_mask is only materialised
        # when flex_attention cannot serve this batch, because at anchor_num=512
        # and an 8192 context it is a ~10 GiB fp32 tensor.
        block_mask, keep_rows = self._build_dual_source_block_mask(
            anchors, block_keep_mask, T,
        )
        attn_mask = None
        if block_mask is None:
            attn_mask = self._build_dual_source_mask(
                anchors, block_keep_mask, T,
            )

        # 6. Transformer backbone with dual-source KV
        for layer in self.layers:
            draft_hidden = layer(
                draft_hidden=draft_hidden,
                context_hidden=context_feature,
                draft_position_ids=draft_position_ids,
                context_position_ids=context_position_ids,
                attn_mask=attn_mask,
                block_mask=block_mask,
                keep_rows=keep_rows,
            )
        draft_hidden = self.norm(draft_hidden)
        hidden_4d = draft_hidden.reshape(B, num_anchors, self.block_size, H)

        # 7. Labels (TorchSpec convention: slot k predicts input_ids[anchor+k+1])
        label_offsets = torch.arange(1, self.block_size + 1, device=device).view(1, 1, -1)
        label_indices = anchors.unsqueeze(-1) + label_offsets
        valid_label_mask = label_indices < T
        safe_label_indices = label_indices.clamp(max=T - 1)
        safe_label_indices = torch.where(
            block_keep_mask.unsqueeze(-1), safe_label_indices,
            torch.zeros_like(safe_label_indices),
        )

        target_ids = torch.gather(
            input_ids.unsqueeze(1).expand(-1, num_anchors, -1),
            2, safe_label_indices,
        )

        # Eval mask: cumprod for contiguous prefix (TorchSpec convention)
        target_loss_mask = torch.gather(
            loss_mask.unsqueeze(1).expand(-1, num_anchors, -1),
            2, safe_label_indices,
        )
        eval_bool = block_keep_mask.unsqueeze(-1) & valid_label_mask & (target_loss_mask > 0.5)
        eval_bool = eval_bool.to(torch.int32).cumprod(dim=-1).bool()
        eval_mask = eval_bool.float()

        # Decay weights
        k_idx = torch.arange(self.block_size, device=device).view(1, 1, -1)
        if decay_gamma > 0:
            decay_w = torch.exp(-k_idx.float() / decay_gamma)
        else:
            decay_w = torch.ones_like(k_idx, dtype=torch.float32)
        decay_weight_mask = eval_mask * decay_w

        # 8. Draft logits via LM head
        base_logits = F.linear(draft_hidden, teacher_lm_head_weight)
        base_logits_4d = base_logits.reshape(B, num_anchors, self.block_size, -1)
        vocab_size = base_logits_4d.size(-1)

        # 9. Markov bias (teacher-forced prev token)
        anchor_token_ids = torch.gather(input_ids, 1, anchors)
        prev_token_ids = torch.cat(
            [anchor_token_ids.unsqueeze(-1), target_ids[:, :, :-1]], dim=-1,
        )

        logits_4d = base_logits_4d
        markov_features = None
        if self.markov_head is not None:
            logits_4d = base_logits_4d + self.markov_head(prev_token_ids)
            markov_features = self.markov_head.get_features(prev_token_ids)

        # 10. Cross-entropy loss
        flat_logits = logits_4d.reshape(-1, vocab_size)
        flat_targets = target_ids.reshape(-1)
        ce_per_token = F.cross_entropy(
            flat_logits.float(), flat_targets, reduction="none",
        ).view(B, num_anchors, self.block_size)
        ce_num = (ce_per_token * decay_weight_mask).sum()

        # 11. L1 distribution distillation (aligned with TorchSpec — no 0.5 factor)
        l1_num = base_logits.new_zeros((), dtype=torch.float32)
        accept_rate = None
        need_target = (l1_alpha > 0) or (
            self.confidence_head is not None and conf_alpha > 0
        )
        if need_target:
            if target_hidden_states is None:
                raise ValueError(
                    "DSpark L1/confidence losses require target_hidden_states."
                )
            # Target hidden at position anchor+k predicts token at anchor+k+1
            tgt_idx = (safe_label_indices - 1).clamp(min=0)
            hdim = target_hidden_states.size(-1)
            gather_idx = tgt_idx.reshape(B, -1, 1).expand(-1, -1, hdim)
            aligned_hidden = torch.gather(target_hidden_states, 1, gather_idx)
            aligned_target_logits = F.linear(aligned_hidden, teacher_lm_head_weight).view(
                B, num_anchors, self.block_size, vocab_size,
            )
            draft_probs = torch.softmax(logits_4d.float(), dim=-1)
            target_probs = torch.softmax(aligned_target_logits.float(), dim=-1)
            l1_per_token = (draft_probs - target_probs).abs().sum(dim=-1)
            if l1_alpha > 0:
                l1_num = (l1_per_token * decay_weight_mask).sum()
            accept_rate = (1.0 - 0.5 * l1_per_token).clamp(0.0, 1.0)

        # 12. Confidence head BCE (binary_cross_entropy_with_logits)
        conf_num = base_logits.new_zeros((), dtype=torch.float32)
        if self.confidence_head is not None and conf_alpha > 0 and accept_rate is not None:
            if markov_features is not None:
                conf_features = torch.cat(
                    [hidden_4d, markov_features.to(hidden_4d.dtype)], dim=-1,
                )
            else:
                conf_features = hidden_4d
            confidence_pred = self.confidence_head(conf_features).float()
            conf_bce = (
                F.binary_cross_entropy_with_logits(
                    confidence_pred, accept_rate.detach(), reduction="none",
                )
                * decay_weight_mask
            )
            conf_num = conf_bce.sum()

        # 13. Pooled global loss (TorchSpec _build_loss: all_reduce denominator,
        # multiply by world_size to cancel FSDP mean reduction)
        local_den = decay_weight_mask.sum()
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        global_den = local_den.detach().clone()
        if world_size > 1:
            dist.all_reduce(global_den, op=dist.ReduceOp.SUM)
        global_den = global_den + 1e-6
        total_loss = (
            ce_alpha * ce_num / global_den
            + l1_alpha * l1_num / global_den
            + conf_alpha * conf_num / global_den
        ) * world_size

        # Per-component logging (local means)
        local_den_eps = local_den + 1e-6
        total_ce = (ce_num / local_den_eps).detach()
        total_tv = (l1_num / local_den_eps).detach()
        total_conf = (conf_num / local_den_eps).detach()

        # 14. Metrics (accuracy per position)
        with torch.no_grad():
            flat_binary = eval_mask.reshape(-1)
            pred_ids = torch.argmax(flat_logits, dim=-1)
            correct = (pred_ids == flat_targets) & (flat_binary > 0.5)

            count_pp = eval_mask.sum(dim=(0, 1)).clamp(min=1.0)
            loss_pp = (ce_per_token * eval_mask).sum(dim=(0, 1)) / count_pp
            acc_pp = (
                correct.view(B, num_anchors, self.block_size).float().sum(dim=(0, 1))
                / count_pp
            )

        losses = [loss_pp[k].detach() for k in range(self.block_size)]
        accuracies = [acc_pp[k].detach() for k in range(self.block_size)]

        return {
            "losses": losses,
            "accuracies": accuracies,
            "total_loss": total_loss,
            "ce_loss": total_ce,
            "tv_loss": total_tv,
            "conf_loss": total_conf,
        }
