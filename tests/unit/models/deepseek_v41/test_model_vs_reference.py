"""The trainable V4.1 backbone against the released reference implementation, on CPU.

A tiny 8-layer config keeps every V4.1 mechanism: sliding-window-only layers, an encoder KV
source with a reuse layer after it, a decoder KV source that is also the candidate source, a
reindex layer inside its candidate blocks, a reuse layer after that, Engram, and Single-Pass mHC.
Both models get the same random weights by name; logits must agree bit for bit.
"""

import importlib
import sys
from pathlib import Path

import pytest
import torch

from lumenrl.models.deepseek_v41.config import DSV41Config
from lumenrl.models.deepseek_v41.model import DSV41Model, QuantPoints

_REF_DIR = Path(__file__).parent / "reference"
_REF_MODULES = ("model", "engram", "kernel", "image_processor", "vision")

TINY = {
    "vocab_size": 97, "dim": 64, "moe_inter_dim": 32, "n_layers": 8, "n_mtp_layers": 0, "n_heads": 4,
    "n_routed_experts": 8, "n_shared_experts": 1, "n_activated_experts": 2, "score_func": "sqrtsoftplus",
    "route_scale": 1.5, "swiglu_limit": 10.0, "q_lora_rank": 32, "head_dim": 32, "rope_head_dim": 16, "norm_eps": 1e-6,
    "o_groups": 2, "o_lora_rank": 16, "window_size": 8, "compress_ratios": (0, 0, 2, 2, 1, 1, 1, 1),
    "kv_source_layers": (2, 4), "index_source_layers": (2, 4, 6), "compress_rope_theta": 40000.0,
    "original_seq_len": 64, "rope_theta": 10000.0, "rope_factor": 4, "beta_fast": 32, "beta_slow": 1,
    "index_n_heads": 2, "index_head_dim": 32, "index_topk": 4, "candidate_source_layer": 4,
    "candidate_topk_blocks": 2, "candidate_block_size": 2, "hc_mult": 4, "hc_sinkhorn_iters": 20, "hc_eps": 1e-6,
    "engram_layer_ids": (1,), "engram_max_ngram_size": 4, "engram_vocab_size": 50, "engram_n_heads": 2,
    "engram_head_dim": 32, "engram_pad_id": 2, "engram_compressed_vocab_size": 40,
    "max_batch_size": 2, "max_seq_len": 64, "dtype": "bf16", "expert_dtype": None, "vision_n_layers": 0,
}
SEQLEN = 24


@pytest.fixture
def reference(monkeypatch):
    monkeypatch.syspath_prepend(str(_REF_DIR))
    for name in _REF_MODULES:
        monkeypatch.delitem(sys.modules, name, raising=False)
    ref = importlib.import_module("model")
    yield ref
    for name in _REF_MODULES:
        sys.modules.pop(name, None)


def _token_map():
    return [i % TINY["engram_compressed_vocab_size"] for i in range(TINY["vocab_size"])]


def _config(num_embeddings):
    t = TINY
    return DSV41Config(
        vocab_size=t["vocab_size"], dim=t["dim"], moe_inter_dim=t["moe_inter_dim"], n_layers=t["n_layers"],
        n_heads=t["n_heads"], n_routed_experts=t["n_routed_experts"], n_shared_experts=t["n_shared_experts"],
        n_activated_experts=t["n_activated_experts"], score_func=t["score_func"], route_scale=t["route_scale"],
        norm_topk_prob=True, swiglu_limit=t["swiglu_limit"], q_lora_rank=t["q_lora_rank"],
        head_dim=t["head_dim"], rope_head_dim=t["rope_head_dim"], norm_eps=t["norm_eps"],
        o_groups=t["o_groups"], o_lora_rank=t["o_lora_rank"], window_size=t["window_size"],
        compress_ratios=t["compress_ratios"], kv_source_layers=t["kv_source_layers"],
        index_source_layers=t["index_source_layers"], candidate_source_layer=t["candidate_source_layer"],
        candidate_topk_blocks=t["candidate_topk_blocks"], candidate_block_size=t["candidate_block_size"],
        rope_theta=t["rope_theta"], compress_rope_theta=t["compress_rope_theta"],
        original_seq_len=t["original_seq_len"], rope_factor=t["rope_factor"], beta_fast=t["beta_fast"],
        beta_slow=t["beta_slow"], index_n_heads=t["index_n_heads"], index_head_dim=t["index_head_dim"],
        index_topk=t["index_topk"], hc_mult=t["hc_mult"], hc_sinkhorn_iters=t["hc_sinkhorn_iters"],
        hc_eps=t["hc_eps"], engram_layout_layer_ids=t["engram_layer_ids"], engram_num_embeddings=num_embeddings,
        engram_max_ngram_size=t["engram_max_ngram_size"], engram_vocab_size=t["engram_vocab_size"],
        engram_n_heads=t["engram_n_heads"], engram_head_dim=t["engram_head_dim"], engram_pad_id=t["engram_pad_id"],
        engram_compressed_vocab_size=t["engram_compressed_vocab_size"], engram_layers=((1, 0),),
        original_layer_ids=tuple(range(t["n_layers"])),
    )


def _init(model, seed):
    g = torch.Generator().manual_seed(seed)
    for name, p in sorted(model.state_dict().items()):
        if p.dtype == torch.float8_e4m3fn:
            p.copy_((torch.randn(p.shape, generator=g) * 2).to(torch.float8_e4m3fn))
        elif p.dtype == torch.float8_e8m0fnu:
            p.copy_(torch.full(p.shape, 124, dtype=torch.uint8).view(torch.float8_e8m0fnu))
        elif name.endswith(("norm.weight", "q_weight", "k_weight")):
            p.copy_(1 + 0.1 * torch.randn(p.shape, generator=g))
        elif "hc_" in name and name.endswith("_scale"):
            p.copy_(0.5 * torch.rand(p.shape, generator=g))
        elif name == "embed.weight":
            p.copy_(torch.randn(p.shape, generator=g))
        else:
            p.copy_(0.1 * torch.randn(p.shape, generator=g))


@pytest.fixture
def pair(reference, monkeypatch):
    monkeypatch.setattr(reference, "NgramHashState", reference.NgramHashState)
    engram_ref = sys.modules["engram"]
    monkeypatch.setattr(engram_ref, "build_compressed_token_map", lambda tok: (_token_map(), TINY["engram_compressed_vocab_size"]))
    probe = engram_ref.EngramLayout.from_args(reference.ModelArgs(**{**TINY, "engram_num_embeddings": (1,)}))
    rows = (sum(p for per_ngram in probe.primes[0] for p in per_ngram),)
    args = reference.ModelArgs(**{**TINY, "engram_num_embeddings": rows})

    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        ref = reference.Transformer(args)
    finally:
        torch.set_default_dtype(previous)
    with torch.no_grad():
        _init(ref, 0)
    # The reference holds BF16 weights here, so its linear layers take unquantized activations.
    ours = DSV41Model(_config(rows), token_map=torch.tensor(_token_map()), quant=QuantPoints(linear_act_fp8=False))
    for name, t in ref.state_dict().items():
        assert ours.state_dict()[name].dtype == t.dtype, name
    ours.load_state_dict(ref.state_dict(), strict=True)
    monkeypatch.setattr(reference.ParallelHead, "forward", lambda self, x, full_logits=True: torch.nn.functional.linear(x.float(), self.weight))
    return ref, ours


def test_logits_match_the_reference_bit_for_bit(pair):
    ref, ours = pair
    ids = torch.randint(0, TINY["vocab_size"], (2, SEQLEN), generator=torch.Generator().manual_seed(1))

    _, want, _ = ref(ids)
    with torch.no_grad():
        got = ours(ids)

    assert got.shape == (2, SEQLEN, TINY["vocab_size"])
    assert torch.equal(got, want), (got - want).abs().max()


def test_engram_hashes_match_the_reference(pair):
    ref, ours = pair
    ids = torch.randint(0, TINY["vocab_size"], (2, SEQLEN), generator=torch.Generator().manual_seed(2))

    assert torch.equal(ours.hasher(ids), ref.engram_hash(ids, 0))


def test_backward_reaches_every_trainable_parameter_but_the_indexer(pair):
    _, ours = pair
    ids = torch.randint(0, TINY["vocab_size"], (2, SEQLEN), generator=torch.Generator().manual_seed(3))

    ours(ids).logsumexp(-1).mean().backward()

    # The indexer only emits integer top-k positions, so the LM loss cannot reach it; DSA trains
    # it with a separate KL loss (Megatron's dsa_indexer_loss_coeff), which the V4 recipe leaves off.
    trainable = {n: p for n, p in ours.named_parameters() if p.requires_grad}
    indexer = {n for n in trainable if ".indexer." in n}
    missing = [n for n, p in trainable.items() if p.grad is None and n not in indexer and ".experts." not in n]
    assert not missing, missing
    assert indexer and all(trainable[n].grad is None for n in indexer)
    assert all(torch.isfinite(p.grad).all() for p in trainable.values() if p.grad is not None)


def test_quantization_points_can_be_switched_off(pair):
    _, ours = pair
    plain = DSV41Model(ours.cfg, token_map=ours.hasher.token_map, quant=QuantPoints.none())
    plain.load_state_dict(ours.state_dict())
    ids = torch.randint(0, TINY["vocab_size"], (1, SEQLEN), generator=torch.Generator().manual_seed(4))

    with torch.no_grad():
        a, b = ours(ids), plain(ids)

    # random weights make the top-k choices fragile, so only "it changes and stays finite" holds
    assert not torch.equal(a, b)
    assert torch.isfinite(b).all()
