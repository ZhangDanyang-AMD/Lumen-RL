"""Engram for training: the n-gram hash ids of a whole sequence, and the gated residual write.

The hashing is ported from the released `inference/engram.py` without changing a single
arithmetic step -- the table rows a token reads are a pure function of it, so any drift reads
the wrong rows -- but computed for a full sequence at once instead of through a decode cache.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn

from lumenrl.models.deepseek_v41.config import DSV41Config


def find_next_prime(start: int, seen_primes: set[int]) -> int:
    from sympy import isprime

    candidate = start + 1
    while not isprime(candidate) or candidate in seen_primes:
        candidate += 1
    return candidate


def build_compressed_token_map(tokenizer) -> tuple[list[int], int]:
    """Token id -> compressed id, tokens that normalize alike sharing one; and the compressed
    vocab size, which every hash multiplier is derived from."""
    from tokenizers import Regex, normalizers

    sentinel = "\ue000"
    normalizer = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), sentinel),
            normalizers.Strip(),
            normalizers.Replace(sentinel, " "),
        ]
    )
    backend = tokenizer.backend_tokenizer
    key_to_new: dict[str, int] = {}
    lookup = [0] * len(tokenizer)
    for token_id in range(len(tokenizer)):
        text = backend.decode([token_id], skip_special_tokens=False)
        if "\ufffd" in text:
            key = backend.id_to_token(token_id)
        else:
            normalized = normalizer.normalize_str(text)
            key = normalized if normalized else text
        new_id = key_to_new.get(key)
        if new_id is None:
            new_id = len(key_to_new)
            key_to_new[key] = new_id
        lookup[token_id] = new_id
    return lookup, len(key_to_new)


def compute_hash_multipliers(layer_ids, max_ngram_size: int, tokenizer_vocab_size: int) -> Tensor:
    max_long = np.iinfo(np.int64).max
    multiplier_bound = max(1, (max_long // tokenizer_vocab_size) // 2)
    rows = []
    for layer_id in layer_ids:
        generator = np.random.default_rng(10007 * layer_id)
        values = generator.integers(low=0, high=multiplier_bound, size=(max_ngram_size,), dtype=np.int64)
        rows.append(torch.tensor(values * 2 + 1))
    return torch.stack(rows)


@dataclass(frozen=True)
class EngramLayout:
    """Prime-sized bucket ranges per (layer, n-gram size, head), drawn in order over the released
    model's full list of engram layers -- a slice must keep that list to hash the same way."""

    max_ngram_size: int
    layer_ids: tuple[int, ...]
    num_embeddings: tuple[int, ...]
    primes: tuple[tuple[tuple[int, ...], ...], ...]
    n_heads: int
    head_dim: int

    @classmethod
    def from_config(cls, cfg: DSV41Config) -> EngramLayout | None:
        if not cfg.engram_layout_layer_ids:
            return None
        primes, seen = [], set()
        for _ in cfg.engram_layout_layer_ids:
            per_ngram = []
            for _ in range(cfg.engram_max_ngram_size - 1):
                sizes, current = [], cfg.engram_vocab_size - 1
                for _ in range(cfg.engram_n_heads):
                    current = find_next_prime(current, seen)
                    seen.add(current)
                    sizes.append(current)
                per_ngram.append(tuple(sizes))
            primes.append(tuple(per_ngram))
        return cls(
            max_ngram_size=cfg.engram_max_ngram_size,
            layer_ids=cfg.engram_layout_layer_ids,
            num_embeddings=cfg.engram_num_embeddings,
            primes=tuple(primes),
            n_heads=cfg.engram_n_heads,
            head_dim=cfg.engram_head_dim,
        )


class EngramHasher(nn.Module):
    """input_ids [B, L] -> hash ids [B, L, n_layout_layers, (max_ngram - 1) * n_heads], each
    position hashed with the max_ngram - 1 tokens before it; look-back stops at the sequence
    start and at masked-out (image) tokens."""

    DEAD = -1

    def __init__(self, cfg: DSV41Config, layout: EngramLayout, token_map: Tensor):
        super().__init__()
        if token_map.numel() and int(token_map.max()) + 1 > cfg.engram_compressed_vocab_size:
            raise ValueError("token map ids exceed engram_compressed_vocab_size")
        self.layout = layout
        flat = [[p for per_ngram in layer for p in per_ngram] for layer in layout.primes]
        offsets = [np.cumsum([0, *sizes[:-1]]) for sizes in flat]
        multipliers = compute_hash_multipliers(layout.layer_ids, layout.max_ngram_size, cfg.engram_compressed_vocab_size)
        self.register_buffer("primes", torch.tensor(layout.primes), persistent=False)
        self.register_buffer("offsets", torch.tensor(np.array(offsets)), persistent=False)
        self.register_buffer("multipliers", multipliers, persistent=False)
        self.register_buffer("token_map", token_map.long(), persistent=False)
        self.pad_id = int(token_map[cfg.engram_pad_id])

    @torch.no_grad()
    def forward(self, input_ids: Tensor, token_mask: Tensor | None = None) -> Tensor:
        batch, seqlen = input_ids.shape
        compressed = self.token_map[input_ids]
        if token_mask is not None:
            compressed = torch.where(token_mask, compressed, self.DEAD)
        positions = torch.arange(seqlen, device=input_ids.device).expand(batch, seqlen)
        tokens, blocked = [], torch.zeros_like(positions, dtype=torch.bool)
        for shift in range(self.layout.max_ngram_size):
            source = compressed.gather(1, (positions - shift).clamp_min(0))
            blocked = blocked | (positions < shift) | (source == self.DEAD)
            tokens.append(torch.where(blocked, self.pad_id, source))
        tokens = torch.stack(tokens, dim=-1)
        products = tokens.unsqueeze(2) * self.multipliers
        rolling, hashes = products[..., 0], []
        for i in range(1, self.layout.max_ngram_size):
            rolling = torch.bitwise_xor(rolling, products[..., i])
            hashes.append(rolling.unsqueeze(-1) % self.primes[:, i - 1])
        return torch.cat(hashes, dim=-1) + self.offsets


class EngramTable(nn.Module):
    """The frozen FP8 table of one engram layer with its per-32 E8M0 scales; rows dequantize to
    BF16 on lookup. Kept as buffers: RL does not train the table."""

    def __init__(self, num_embeddings: int, dim: int, block_size: int = 32):
        super().__init__()
        self.block_size = block_size
        self.register_buffer("weight", torch.empty(num_embeddings, dim, dtype=torch.float8_e4m3fn))
        self.register_buffer("scale", torch.empty(num_embeddings, dim // block_size, dtype=torch.float8_e8m0fnu))

    def forward(self, indices: Tensor) -> Tensor:
        values = self.weight[indices].float().unflatten(-1, (-1, self.block_size))
        scales = self.scale[indices].float().unsqueeze(-1)
        return (values * scales).flatten(-2).to(torch.bfloat16)


class Engram(nn.Module):
    """h + gate * value, written into every hyper-connection copy; the gate is a signed-sqrt
    sigmoid of the normalized dot product between each copy and its key."""

    def __init__(self, cfg: DSV41Config, layout: EngramLayout, hash_index: int):
        super().__init__()
        self.hash_index = hash_index
        self.dim, self.hc_mult, self.eps = cfg.dim, cfg.hc_mult, cfg.norm_eps
        self.clamp_value = 1e-6
        self.embed = EngramTable(layout.num_embeddings[hash_index], layout.head_dim)
        n_hash_cols = (layout.max_ngram_size - 1) * layout.n_heads
        self.wkv = nn.Linear(n_hash_cols * layout.head_dim, cfg.dim * (cfg.hc_mult + 1), bias=False)
        self.q_weight = nn.Parameter(torch.ones(cfg.hc_mult, cfg.dim))
        self.k_weight = nn.Parameter(torch.ones(cfg.hc_mult, cfg.dim))

    def forward(self, x: Tensor, hash_ids: Tensor, token_mask: Tensor | None = None) -> Tensor:
        kv = self.wkv(self.embed(hash_ids).flatten(-2))
        key, value = kv.split([self.hc_mult * self.dim, self.dim], dim=-1)
        key = key.float().unflatten(-1, (self.hc_mult, self.dim))
        weight = self.q_weight.float() * self.k_weight.float()
        h, eps = x.float(), self.eps
        rstd = torch.rsqrt(h.square().mean(-1) + eps) * torch.rsqrt(key.square().mean(-1) + eps)
        dot = (h * weight * key).sum(-1) * rstd * self.dim**-0.5
        gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(self.clamp_value).sqrt(), dot))
        if token_mask is not None:
            gate = gate.masked_fill(~token_mask.unsqueeze(-1), 0)
        return (h + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(x.dtype)
