"""Tied-embedding (tie_word_embeddings=true) handling in the Qwen3 bridge.

Qwen3-0.6B/1.7B/4B tie lm_head to embed_tokens. Megatron GPTModel built with
``share_embeddings_and_output_weights=True`` at PP=1 allocates no
``output_layer.weight`` parameter, so ``megatron_to_hf`` must not require it.
"""

from __future__ import annotations

import pytest
import torch

from lumenrl.engine.training.qwen3_megatron_bridge import (
    Qwen3Dims,
    hf_to_megatron,
    megatron_to_hf,
)


def _dims() -> Qwen3Dims:
    return Qwen3Dims(
        num_layers=1, hidden=8, num_heads=2, num_kv_groups=1, head_dim=4,
        ffn=16, vocab=10,
    )


def _hf_state(d: Qwen3Dims, with_lm_head: bool) -> dict[str, torch.Tensor]:
    q = d.num_heads * d.head_dim
    kv = d.num_kv_groups * d.head_dim
    hp = "model.layers.0."
    st = {
        "model.embed_tokens.weight": torch.randn(d.vocab, d.hidden),
        "model.norm.weight": torch.ones(d.hidden),
        hp + "input_layernorm.weight": torch.ones(d.hidden),
        hp + "post_attention_layernorm.weight": torch.ones(d.hidden),
        hp + "self_attn.q_proj.weight": torch.randn(q, d.hidden),
        hp + "self_attn.k_proj.weight": torch.randn(kv, d.hidden),
        hp + "self_attn.v_proj.weight": torch.randn(kv, d.hidden),
        hp + "self_attn.o_proj.weight": torch.randn(d.hidden, q),
        hp + "self_attn.q_norm.weight": torch.ones(d.head_dim),
        hp + "self_attn.k_norm.weight": torch.ones(d.head_dim),
        hp + "mlp.gate_proj.weight": torch.randn(d.ffn, d.hidden),
        hp + "mlp.up_proj.weight": torch.randn(d.ffn, d.hidden),
        hp + "mlp.down_proj.weight": torch.randn(d.hidden, d.ffn),
    }
    if with_lm_head:
        st["lm_head.weight"] = st["model.embed_tokens.weight"]
    return st


def test_hf_to_megatron_tied_checkpoint_without_lm_head() -> None:
    d = _dims()
    meg = hf_to_megatron(_hf_state(d, with_lm_head=False), d, te=True)
    assert torch.equal(meg["output_layer.weight"], meg["embedding.word_embeddings.weight"])


@pytest.mark.parametrize("te", [True, False])
def test_megatron_to_hf_without_output_layer_param(te: bool) -> None:
    """PP=1 tied model: no output_layer.weight param -> lm_head aliases embeddings."""
    d = _dims()
    meg = hf_to_megatron(_hf_state(d, with_lm_head=True), d, te=te)
    del meg["output_layer.weight"]  # what GPTModel.named_parameters() yields when tied

    hf = dict(megatron_to_hf(list(meg.items()), d, te=te))
    assert "lm_head.weight" in hf
    assert hf["lm_head.weight"] is hf["model.embed_tokens.weight"]


def test_megatron_to_hf_untied_keeps_real_output_layer() -> None:
    d = _dims()
    st = _hf_state(d, with_lm_head=False)
    st["lm_head.weight"] = torch.randn(d.vocab, d.hidden)  # distinct, untied
    meg = hf_to_megatron(st, d, te=True)
    hf = dict(megatron_to_hf(list(meg.items()), d, te=True))
    assert torch.equal(hf["lm_head.weight"], st["lm_head.weight"])
    assert not torch.equal(hf["lm_head.weight"], hf["model.embed_tokens.weight"])


def test_megatron_to_hf_last_stage_without_embeddings_raises() -> None:
    """PP>1 last stage with neither output_layer.weight nor embeddings is an error."""
    d = _dims()
    meg = hf_to_megatron(_hf_state(d, with_lm_head=True), d, te=True, pp_rank=1, pp_size=2)
    del meg["output_layer.weight"]
    with pytest.raises(KeyError):
        list(megatron_to_hf(list(meg.items()), d, te=True, pp_rank=1, pp_size=2))
