"""DeepSeek-V4.1 text backbone configuration, and the per-layer CSA2 roles it implies.

V4.1's causal encoder-decoder is not a separate encoder and decoder: all layers are the same
block, and what differs is where each layer's compressed KV, index keys and top-k selection come
from. A layer whose compress ratio is non-zero reads the compressed KV of the latest preceding
KV-source layer and the top-k of the latest preceding index-source layer; sources publish, the
layers after them reuse. This module turns the HF config into those roles and can cut a slice of
the stack (for development on one node) with every reference renumbered.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from enum import Enum


class AttnMode(str, Enum):
    WINDOW = "window"  # compress ratio 0: sliding window only
    FULL = "full"  # KV source and index source: compresses its own KV, runs its own indexer
    REINDEX = "reindex"  # index source only: its own top-k over the shared index keys
    REUSE = "reuse"  # neither: reuses the shared compressed KV and top-k


@dataclass(frozen=True)
class LayerSpec:
    layer: int  # position in this (possibly sliced) stack
    original_layer: int  # layer id in the released model
    compress_ratio: int
    mode: AttnMode
    kv_source: int | None  # stack position of the layer whose compressed KV this one reads
    index_source: int | None  # stack position of the layer whose top-k this one uses
    candidate_source: int | None  # stack position producing the candidate blocks this layer obeys
    is_candidate_source: bool
    engram_hash_index: int | None  # index into the engram hash layout, None without engram


@dataclass(frozen=True)
class DSV41Config:
    vocab_size: int
    dim: int
    moe_inter_dim: int
    n_layers: int
    n_heads: int
    n_routed_experts: int
    n_shared_experts: int
    n_activated_experts: int
    score_func: str
    route_scale: float
    norm_topk_prob: bool
    swiglu_limit: float
    q_lora_rank: int
    head_dim: int
    rope_head_dim: int
    norm_eps: float
    o_groups: int
    o_lora_rank: int
    window_size: int
    compress_ratios: tuple[int, ...]  # backbone layers only
    kv_source_layers: tuple[int, ...]
    index_source_layers: tuple[int, ...]
    candidate_source_layer: int  # -1: no candidate pre-filtering
    candidate_topk_blocks: int
    candidate_block_size: int
    rope_theta: float
    compress_rope_theta: float
    original_seq_len: int
    rope_factor: float
    beta_fast: int
    beta_slow: int
    index_n_heads: int
    index_head_dim: int
    index_topk: int
    hc_mult: int
    hc_sinkhorn_iters: int
    hc_eps: float
    # Engram. The hash layout (primes, multipliers) is a function of the released model's full list
    # of engram layers, so a slice keeps that list in `engram_layout_layer_ids` and maps its own
    # layers onto it with `engram_layers` (stack position -> layout index).
    engram_layout_layer_ids: tuple[int, ...]
    engram_num_embeddings: tuple[int, ...]
    engram_max_ngram_size: int
    engram_vocab_size: int
    engram_n_heads: int
    engram_head_dim: int
    engram_pad_id: int
    engram_compressed_vocab_size: int
    engram_layers: tuple[tuple[int, int], ...]
    original_layer_ids: tuple[int, ...]

    @classmethod
    def from_hf(cls, config: dict) -> DSV41Config:
        """From the released config.json (top level with `text_config`, or `text_config` itself)."""
        t = config.get("text_config", config)
        n = int(t["num_hidden_layers"])
        rope = t.get("rope_scaling") or {}
        engram_ids = tuple(int(i) for i in t.get("engram_layer_ids", ()))
        cand = t.get("candidate_source_layer_id")
        cfg = cls(
            vocab_size=int(t["vocab_size"]),
            dim=int(t["hidden_size"]),
            moe_inter_dim=int(t["moe_intermediate_size"]),
            n_layers=n,
            n_heads=int(t["num_attention_heads"]),
            n_routed_experts=int(t["n_routed_experts"]),
            n_shared_experts=int(t["n_shared_experts"]),
            n_activated_experts=int(t["num_experts_per_tok"]),
            score_func=str(t["scoring_func"]),
            route_scale=float(t.get("routed_scaling_factor", 1.0)),
            norm_topk_prob=bool(t.get("norm_topk_prob", True)),
            swiglu_limit=float(t.get("swiglu_limit", 0.0)),
            q_lora_rank=int(t["q_lora_rank"]),
            head_dim=int(t["head_dim"]),
            rope_head_dim=int(t["qk_rope_head_dim"]),
            norm_eps=float(t["rms_norm_eps"]),
            o_groups=int(t["o_groups"]),
            o_lora_rank=int(t["o_lora_rank"]),
            window_size=int(t["sliding_window"]),
            compress_ratios=tuple(int(r) for r in t["compress_ratios"][:n]),
            kv_source_layers=tuple(int(i) for i in t.get("kv_source_layer_ids", ())),
            index_source_layers=tuple(int(i) for i in t.get("index_source_layer_ids", ())),
            candidate_source_layer=-1 if cand is None else int(cand),
            candidate_topk_blocks=int(t.get("candidate_topk_blocks", 0)),
            candidate_block_size=int(t.get("candidate_block_size", 0)),
            rope_theta=float(t["rope_theta"]),
            compress_rope_theta=float(t["compress_rope_theta"]),
            original_seq_len=int(rope.get("original_max_position_embeddings", 0)),
            rope_factor=float(rope.get("factor", 1.0)),
            beta_fast=int(rope.get("beta_fast", 32)),
            beta_slow=int(rope.get("beta_slow", 1)),
            index_n_heads=int(t["index_n_heads"]),
            index_head_dim=int(t["index_head_dim"]),
            index_topk=int(t["index_topk"]),
            hc_mult=int(t["hc_mult"]),
            hc_sinkhorn_iters=int(t["hc_sinkhorn_iters"]),
            hc_eps=float(t["hc_eps"]),
            engram_layout_layer_ids=engram_ids,
            engram_num_embeddings=tuple(int(x) for x in t.get("engram_num_embeddings", ())),
            engram_max_ngram_size=int(t.get("engram_max_ngram_size", 1)),
            engram_vocab_size=int(t.get("engram_vocab_size", 0)),
            engram_n_heads=int(t.get("engram_n_heads", 0)),
            engram_head_dim=int(t.get("engram_head_dim", 0)),
            engram_pad_id=int(t.get("engram_pad_token_id", 2)),
            engram_compressed_vocab_size=int(t.get("engram_compressed_vocab_size", 0)),
            engram_layers=tuple((layer, k) for k, layer in enumerate(engram_ids) if layer < n),
            original_layer_ids=tuple(range(n)),
        )
        cfg.validate()
        return cfg

    # ---- roles ----------------------------------------------------------------------------
    def layer_specs(self) -> tuple[LayerSpec, ...]:
        kv_sources, index_sources = set(self.kv_source_layers), set(self.index_source_layers)
        engram = dict(self.engram_layers)
        specs, last_kv, last_index = [], None, None
        for i in range(self.n_layers):
            ratio = self.compress_ratios[i]
            is_kv, is_index = i in kv_sources, i in index_sources
            if is_kv:
                last_kv = i
            if is_index:
                last_index = i
            if ratio == 0:
                mode = AttnMode.WINDOW
            elif is_kv and is_index:
                mode = AttnMode.FULL
            elif is_index:
                mode = AttnMode.REINDEX
            elif is_kv:
                raise ValueError(f"layer {i} is a KV source but not an index source")
            else:
                mode = AttnMode.REUSE
            cand = self.candidate_source_layer
            uses_candidates = mode in (AttnMode.FULL, AttnMode.REINDEX) and 0 <= cand < i
            specs.append(
                LayerSpec(
                    layer=i,
                    original_layer=self.original_layer_ids[i],
                    compress_ratio=ratio,
                    mode=mode,
                    kv_source=last_kv if ratio else None,
                    index_source=last_index if ratio else None,
                    candidate_source=cand if uses_candidates else None,
                    is_candidate_source=i == cand,
                    engram_hash_index=engram.get(i),
                )
            )
        return tuple(specs)

    def validate(self) -> None:
        if len(self.compress_ratios) != self.n_layers:
            raise ValueError(f"{len(self.compress_ratios)} compress ratios for {self.n_layers} layers")
        if any(r not in (0, 1, 2) for r in self.compress_ratios):
            raise ValueError(f"CSA2 compress ratios must be 0, 1 or 2, got {self.compress_ratios}")
        for i in (*self.kv_source_layers, *self.index_source_layers):
            if not 0 <= i < self.n_layers or self.compress_ratios[i] == 0:
                raise ValueError(f"source layer {i} must be a compressing layer of the stack")
        for s in self.layer_specs():
            if not s.compress_ratio:
                continue
            for name, src in (("KV", s.kv_source), ("index", s.index_source)):
                if src is None:
                    raise ValueError(f"layer {s.layer} compresses but no {name} source precedes it")
                if self.compress_ratios[src] != s.compress_ratio:
                    raise ValueError(
                        f"layer {s.layer} (ratio {s.compress_ratio}) shares the {name} source "
                        f"layer {src} with ratio {self.compress_ratios[src]}"
                    )
            if s.mode is AttnMode.REINDEX and s.kv_source is None:
                raise ValueError(f"reindex layer {s.layer} has no index-key owner before it")
        cand = self.candidate_source_layer
        if cand >= 0 and (cand >= self.n_layers or cand not in self.kv_source_layers):
            raise ValueError(f"candidate source {cand} must be a KV source of the stack")

    # ---- slicing ----------------------------------------------------------------------------
    def slice(self, keep: list[int] | tuple[int, ...]) -> DSV41Config:
        """The stack made of `keep` (positions in this config, ascending). Every source a kept
        layer reads must be kept too; engram layers keep their place in the hash layout."""
        keep = tuple(keep)
        if list(keep) != sorted(set(keep)) or not keep or keep[-1] >= self.n_layers:
            raise ValueError(f"slice must be ascending positions of the stack, got {keep}")
        new = {old: k for k, old in enumerate(keep)}
        for s in self.layer_specs():
            if s.layer not in new:
                continue
            for src in (s.kv_source, s.index_source, s.candidate_source):
                if src is not None and src not in new:
                    raise ValueError(f"layer {s.layer} reads layer {src}, which the slice drops")

        def remap(ids):
            return tuple(new[i] for i in ids if i in new)

        cand = self.candidate_source_layer
        cfg = dataclasses.replace(
            self,
            n_layers=len(keep),
            compress_ratios=tuple(self.compress_ratios[i] for i in keep),
            kv_source_layers=remap(self.kv_source_layers),
            index_source_layers=remap(self.index_source_layers),
            candidate_source_layer=new.get(cand, -1) if cand >= 0 else -1,
            engram_layers=tuple((new[layer], k) for layer, k in self.engram_layers if layer in new),
            original_layer_ids=tuple(self.original_layer_ids[i] for i in keep),
        )
        cfg.validate()
        return cfg
