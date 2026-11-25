"""Default-off, fail-closed Qwen3 BF16 operator alignment on clean ATOM main.

The import hook calls :func:`apply` after ``attention_mha`` or ``linear`` loads.
No ATOM file is copied or modified.  A missing capability is an error when
alignment was explicitly requested, never a silent return to native attention.
"""

from __future__ import annotations

import atexit
import copy
import functools
import hashlib
import importlib
import inspect
import json
import logging
import math
import os
import sys
import threading
import weakref
from itertools import pairwise
from pathlib import Path

_LOG = logging.getLogger(__name__)
_SWITCH = "LUMENRL_ATOM_OP_ALIGNMENT"
_TARGETS = frozenset(("atom.model_ops.attention_mha", "atom.model_ops.linear", "atom.model_ops.embed_head"))
_INSTALL_LOCK = threading.RLock()
_INIT_LOCK = threading.RLock()
_ALIGNED_ROPE = weakref.WeakKeyDictionary()
_RUNTIME = {}

_CUSEQLENS_K_CACHE_ENV = "LUMEN_ATOM_CUSEQLENS_CACHE"
_CUSEQLENS_K_CACHE = threading.local()
_CUSEQLENS_K_CACHE_RUN_ID = os.environ.get("LUMEN_ATOM_CUSEQLENS_CACHE_RUN_ID")
_CUSEQLENS_K_CACHE_STATS = {
    "hits": 0,
    "misses": 0,
    "prefill_bypasses": 0,
    "cudagraph_bypasses": 0,
    "invalid_bypasses": 0,
}


def _report_cuseqlens_k_cache_stats():
    if _CUSEQLENS_K_CACHE_RUN_ID:
        stats = _CUSEQLENS_K_CACHE_STATS
        print(
            "CU_SEQLENS_K_CACHE_STATS "
            f"run_id={_CUSEQLENS_K_CACHE_RUN_ID} hits={stats['hits']} "
            f"misses={stats['misses']} prefill_bypasses={stats['prefill_bypasses']} "
            f"cudagraph_bypasses={stats['cudagraph_bypasses']} "
            f"invalid_bypasses={stats['invalid_bypasses']}",
            flush=True,
        )


if (
    os.environ.get(_CUSEQLENS_K_CACHE_ENV, "0") == "1"
    and _CUSEQLENS_K_CACHE_RUN_ID
):
    atexit.register(_report_cuseqlens_k_cache_stats)


def _cu_seqlens_k_tensor(cuk, device, fwd_ctx):
    """Build exact int32 K offsets, reusing only validated decode values."""
    import torch

    if os.environ.get(_CUSEQLENS_K_CACHE_ENV, "0") != "1":
        return torch.tensor(cuk, dtype=torch.int32, device=device)

    context = getattr(fwd_ctx, "context", None)
    if context is None or bool(getattr(context, "is_prefill", True)):
        _CUSEQLENS_K_CACHE_STATS["prefill_bypasses"] += 1
        return torch.tensor(cuk, dtype=torch.int32, device=device)
    try:
        use_cudagraph = bool(context.forward_mode.use_cudagraph)
    except (AttributeError, TypeError):
        use_cudagraph = True
    if use_cudagraph:
        _CUSEQLENS_K_CACHE_STATS["cudagraph_bypasses"] += 1
        return torch.tensor(cuk, dtype=torch.int32, device=device)

    try:
        values = tuple(cuk)
        valid = (
            bool(values)
            and values[0] == 0
            and all(type(value) is int for value in values)
            and all(left <= right for left, right in zip(values, values[1:]))
        )
    except (TypeError, ValueError):
        valid = False
        values = ()
    if not valid:
        _CUSEQLENS_K_CACHE_STATS["invalid_bypasses"] += 1
        return torch.tensor(cuk, dtype=torch.int32, device=device)

    key = (values, str(device))
    entry = getattr(_CUSEQLENS_K_CACHE, "entry", None)
    if entry is not None and entry[0] == key:
        _CUSEQLENS_K_CACHE_STATS["hits"] += 1
        return entry[1]
    result = torch.tensor(values, dtype=torch.int32, device=device)
    _CUSEQLENS_K_CACHE.entry = (key, result)
    _CUSEQLENS_K_CACHE_STATS["misses"] += 1
    return result


class AlignmentError(RuntimeError):
    """The explicitly requested validated operator path cannot be supplied."""


def enabled() -> bool:
    return os.environ.get(_SWITCH, "0") == "1"


def _require_signature(fn, names, *, accepts_extra=False):
    params = tuple(inspect.signature(fn).parameters)
    if params[: len(names)] != tuple(names) or (not accepts_extra and params != tuple(names)):
        raise AlignmentError(f"ATOM main interface drift: {fn}: {params}")


def _policy_key():
    path = os.environ.get("LUMENRL_ATOM_ATTENTION_CONFIG")
    digest = os.environ.get("LUMENRL_ATOM_ATTENTION_CONFIG_SHA256")
    if not path or not digest or len(digest) != 64:
        raise AlignmentError("aligned Attention requires canonical config path and SHA256")
    try:
        int(digest, 16)
    except ValueError as exc:
        raise AlignmentError("invalid aligned Attention config SHA256") from exc
    return str(Path(path).resolve()), digest.lower()


def _load_policy(path, digest):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise AlignmentError("aligned Attention config SHA256 mismatch")
    policy = json.loads(raw)
    if tuple(policy.get(k) for k in ("schema_version", "backend", "arch", "dtype")) != (
        1, "triton", "gfx950", "bfloat16"
    ):
        raise AlignmentError("unsupported aligned Attention manifest")
    if (policy.get("module"), policy.get("function")) != (
        "aiter.ops.triton.attention.mha", "flash_attn_varlen_func"
    ):
        raise AlignmentError("manifest must name the validated public Triton API")
    cfg = policy.get("config", {})
    required = {
        "BLOCK_M", "BLOCK_N", "PRELOAD_V", "waves_per_eu", "num_warps",
        "num_ctas", "num_stages", "enable_fp_fusion",
    }
    if set(cfg) != required or cfg["enable_fp_fusion"] is not False:
        raise AlignmentError("missing or unvalidated fixed Attention configuration")
    return policy


class _InitialLaunchAudit:
    """One-time, owner-thread-only launch observer; removed before serving."""

    def __init__(self, original, config):
        self.original = original
        self.owner = threading.get_ident()
        self.config = config
        self.receipts = []

    def __getattr__(self, name):
        return getattr(self.original, name)

    def __getitem__(self, grid):
        launch = self.original[grid]

        def invoke(*args, **kwargs):
            compiled = launch(*args, **kwargs)
            if threading.get_ident() != self.owner:
                return compiled
            actual = {}
            for key, expected in self.config.items():
                names = ("PRE_LOAD_V", "PRELOAD_V") if key == "PRELOAD_V" else (key,)
                found = next((kwargs[name] for name in names if name in kwargs), None)
                if found != expected or (isinstance(expected, bool) and type(found) is not bool):
                    raise AlignmentError(f"fixed config was not forwarded to launch: {key}")
                actual[key] = found
            metadata = getattr(compiled, "metadata", None)
            if getattr(metadata, "enable_fp_fusion", None) is not False:
                raise AlignmentError("compiled Attention enable_fp_fusion is not False")
            if getattr(metadata, "num_warps", None) != self.config["num_warps"]:
                raise AlignmentError("compiled Attention warp count differs from fixed config")
            kernel_hash = getattr(compiled, "hash", None)
            if not kernel_hash:
                raise AlignmentError("compiled Attention hash receipt unavailable")
            self.receipts.append({
                "launch_config": actual,
                "compiled_enable_fp_fusion": False,
                "compiled_num_warps": metadata.num_warps,
                "compiled_kernel_name": getattr(metadata, "name", None),
                "compiled_hash": str(kernel_hash),
            })
            return compiled

        return invoke


def initialize_aligned_attention(device):
    """Verify pinned public API, source, architecture, compilation, and RNG once."""
    import torch

    device = torch.device(device)
    if device.type != "cuda":
        raise AlignmentError("aligned Attention initialization requires its actual GPU")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    path, digest = _policy_key()
    key = (device.index, path, digest)
    with _INIT_LOCK:
        # Rehash the policy even on a cache hit: replacing the file must fail closed.
        policy = _load_policy(path, digest)
        if key in _RUNTIME:
            return _RUNTIME[key]["receipt"]
        module = importlib.import_module(policy["module"])
        function = getattr(module, policy["function"])
        if str(Path(module.__file__).resolve()) != policy["module_path"]:
            raise AlignmentError("unexpected public Triton module path")
        for entry in policy["sources"].values():
            if hashlib.sha256(Path(entry["path"]).read_bytes()).hexdigest() != entry["sha256"]:
                raise AlignmentError("canonical Triton source identity changed")
        if hashlib.sha256(inspect.getsource(function).encode()).hexdigest() != policy["function_sha256"]:
            raise AlignmentError("public Triton function source changed")
        if str(inspect.signature(function)) != policy["api_signature"]:
            raise AlignmentError("public Triton function signature changed")
        for name, value in policy["runtime_requirements"].items():
            if getattr(module, name, None) != value:
                raise AlignmentError(f"Triton runtime requirement not met: {name}")
        props = torch.cuda.get_device_properties(device)
        if str(props.gcnArchName).split(":")[0] != policy["arch"]:
            raise AlignmentError("unvalidated Attention architecture")
        before_cpu, before_gpu = torch.get_rng_state(), torch.cuda.get_rng_state(device)
        cfg = dict(policy["config"])
        q = torch.zeros(2, 32, 128, device=device, dtype=torch.bfloat16)
        k = torch.zeros(2, 8, 128, device=device, dtype=torch.bfloat16)
        cu = torch.tensor([0, 2], device=device, dtype=torch.int32)
        original = module._attn_fwd
        audit = _InitialLaunchAudit(original, cfg)
        try:
            module._attn_fwd = audit
            output = function(q, k, k, cu, cu, 2, 2, dropout_p=0.0,
                              softmax_scale=policy["input_contract"]["softmax_scale"],
                              causal=True, backend="triton", config=dict(cfg))
        finally:
            module._attn_fwd = original
        if not audit.receipts:
            raise AlignmentError("initial aligned Attention did not expose its compiled launch")
        torch.cuda.synchronize(device)
        if not torch.equal(output, torch.zeros_like(output)):
            raise AlignmentError("initial zero-input Attention sanity check failed")
        if not torch.equal(before_cpu, torch.get_rng_state()) or not torch.equal(
            before_gpu, torch.cuda.get_rng_state(device)
        ):
            raise AlignmentError("initial Attention verification changed RNG state")
        receipt = {
            "kind": "one_time_zero_input_compile_guard", "config_path": path,
            "config_sha256": digest, "module_path": module.__file__,
            "module_sha256": policy["module_sha256"],
            "function_sha256": policy["function_sha256"],
            "arch": str(props.gcnArchName), "pci_bus_id": str(props.pci_bus_id),
            "launches": audit.receipts, "rng_unchanged": True,
            "global_launch_hook_restored": True,
            "per_step_compiler_metadata_checked": False,
        }
        receipt["receipt_sha256"] = hashlib.sha256(json.dumps(receipt, sort_keys=True).encode()).hexdigest()
        _RUNTIME[key] = {
            "module": module, "function": function, "config": cfg,
            "config_items": tuple(sorted(cfg.items())), "policy": policy,
            "receipt": receipt,
        }
        _LOG.info("Aligned Attention initialization receipt: %s", json.dumps(receipt, sort_keys=True))
        return receipt


def _pinned_varlen(q, k, v, *, cu_seqlens_q, cu_seqlens_k,
                   max_seqlen_q, max_seqlen_k, dropout_p, softmax_scale, causal):
    path, digest = _policy_key()
    # Recheck the supplied digest on every call; a changed policy must not reuse state.
    _load_policy(path, digest)
    state = _RUNTIME.get((q.device.index, path, digest))
    if state is None:
        raise AlignmentError("aligned Attention initialization was not verified for this config/device")
    if tuple(sorted(state["config"].items())) != state["config_items"]:
        raise AlignmentError("in-process fixed Attention config changed")
    for name, value in state["policy"]["runtime_requirements"].items():
        if getattr(state["module"], name, None) != value:
            raise AlignmentError("Triton runtime globals changed after initialization")
    if getattr(state["module"], state["policy"]["function"]) is not state["function"]:
        raise AlignmentError("Triton callable binding changed after initialization")
    return state["function"](
        q.contiguous(), k.contiguous(), v.contiguous(), cu_seqlens_q,
        cu_seqlens_k, max_seqlen_q, max_seqlen_k, dropout_p=dropout_p,
        softmax_scale=softmax_scale, causal=causal, backend="triton",
        config=dict(state["config"]),
    )


def _rope_signature(rotary, config, force_triton, fused_qk_norm):
    import torch

    if config is None or getattr(config, "model_type", None) != "qwen3":
        raise AlignmentError("aligned RoPE requires actual Qwen3 HF config")
    params = getattr(config, "rope_parameters", None)
    if not isinstance(params, dict) or params.get("rope_type") != "default":
        raise AlignmentError("aligned RoPE requires explicit default rope_parameters")
    if set(params) - {"rope_type", "rope_theta"}:
        raise AlignmentError("scaled/extended RoPE parameters are not validated")
    if not force_triton or fused_qk_norm or hasattr(rotary, "cos_sin_cache"):
        raise AlignmentError("aligned RoPE requires non-joint external-QKNorm Triton cache writer")
    if (type(rotary).__module__, type(rotary).__name__) != ("aiter.rotary_embedding", "RotaryEmbedding"):
        raise AlignmentError("aligned RoPE requires validated AITER RotaryEmbedding class")
    dim = getattr(config, "head_dim", None)
    if dim != 128 or rotary.head_size != dim or rotary.rotary_dim != dim or not rotary.is_neox_style:
        raise AlignmentError("aligned RoPE supports full-D128 NeoX contract")
    base = params.get("rope_theta")
    if type(base) not in (int, float) or not math.isfinite(base) or base <= 0 or float(rotary.base) != float(base):
        raise AlignmentError("RoPE base differs from actual config")
    length = getattr(config, "max_position_embeddings", None)
    if type(length) is not int or length <= 0 or rotary.max_position_embeddings != length:
        raise AlignmentError("RoPE cache length differs from actual config")
    shape = (length, 1, 1, dim // 2)
    cos, sin = rotary.cos_cache, rotary.sin_cache
    if cos.shape != shape or sin.shape != shape or cos.device != sin.device:
        raise AlignmentError("unexpected actual AITER cache axes/device")
    if cos.dtype not in (torch.bfloat16, torch.float32) or sin.dtype != cos.dtype:
        raise AlignmentError("unexpected actual AITER cache dtype")
    if rotary._parameters or rotary._modules or set(rotary._buffers) != {"cos_cache", "sin_cache"}:
        raise AlignmentError("unsupported rotary module state")
    return float(base), dim, length, cos.device.type, cos.device.index


def validate_aligned_rope(rotary):
    import torch

    signature = getattr(rotary, "_lumenrl_fp32_rope_signature", None)
    if signature is None:
        raise AlignmentError("aligned RoPE cache was not initialized")
    _, dim, length, device_type, device_index = signature
    if tuple(map(id, (rotary.cos_cache, rotary.sin_cache))) != getattr(
        rotary, "_lumenrl_fp32_rope_buffer_ids", None
    ):
        raise AlignmentError("aligned FP32 RoPE buffers were replaced after initialization")
    for cache in (rotary.cos_cache, rotary.sin_cache):
        if cache.dtype != torch.float32 or cache.shape != (length, 1, 1, dim // 2) or (
            cache.device.type, cache.device.index
        ) != (device_type, device_index):
            raise AlignmentError("aligned FP32 RoPE cache dtype/device/shape changed")


def align_rope_cache(rotary, config, *, force_triton, fused_qk_norm=False):
    import torch

    if not enabled():
        return rotary
    if rotary is None:
        raise AlignmentError("aligned RoPE requires an actual rotary module")
    signature = _rope_signature(rotary, config, force_triton, fused_qk_norm)
    existing = _ALIGNED_ROPE.get(rotary)
    if existing is not None:
        previous_signature, aligned = existing
        if previous_signature != signature:
            raise AlignmentError("shared rotary source changed config/device")
        validate_aligned_rope(aligned)
        return aligned
    base, dim, length, _, _ = signature
    inv = 1.0 / (base ** (torch.arange(0, dim, 2, device="cpu", dtype=torch.int64).float() / dim))
    device = rotary.cos_cache.device
    inv = inv.to(device=device, dtype=torch.float32)[None, :, None]
    positions = torch.arange(length, device=device, dtype=torch.int64)[None, :]
    with torch.autocast(device_type=device.type, enabled=False):
        freqs = (inv @ positions[:, None, :].float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = (emb.cos() * 1.0)[0, :, : dim // 2].contiguous()[:, None, None, :]
        sin = (emb.sin() * 1.0)[0, :, : dim // 2].contiguous()[:, None, None, :]
    aligned = copy.copy(rotary)
    aligned._buffers = dict(rotary._buffers)
    aligned._parameters = dict(rotary._parameters)
    aligned._modules = dict(rotary._modules)
    aligned._non_persistent_buffers_set = set(rotary._non_persistent_buffers_set)
    aligned.cos_cache, aligned.sin_cache = cos, sin
    aligned.dtype = torch.float32
    aligned._lumenrl_fp32_rope_signature = signature
    aligned._lumenrl_fp32_rope_buffer_ids = (id(cos), id(sin))
    validate_aligned_rope(aligned)
    _ALIGNED_ROPE[rotary] = (signature, aligned)
    return aligned


def validate_aligned_attention_config(config, impl):
    if config is None or getattr(getattr(config, "hf_config", None), "model_type", None) != "qwen3":
        raise AlignmentError("aligned Attention requires actual Qwen3 config")
    for name in ("tensor_parallel_size", "pipeline_parallel_size", "prefill_context_parallel_size", "decode_context_parallel_size"):
        if getattr(config, name, None) != 1:
            raise AlignmentError(f"aligned Attention requires {name}=1")
    if getattr(config, "enforce_eager", None) is not True:
        raise AlignmentError("aligned Attention requires eager execution")
    if getattr(config, "enable_prefix_caching", None) is not False:
        raise AlignmentError("prefix-cache lifecycle is not validated")
    if (impl.num_heads, impl.num_kv_heads, impl.head_dim, impl.kv_cache_dtype) != (32, 8, 128, "bf16"):
        raise AlignmentError("aligned Attention supports H32/KV8/D128 and BF16 KV")
    if impl.sinks is not None or impl.alibi_slopes is not None or impl.sliding_window != -1:
        raise AlignmentError("aligned Attention supports full causal attention without sinks/ALiBi")
    if impl.logits_soft_cap not in (None, 0, 0.0) or impl.scale != 128 ** -0.5:
        raise AlignmentError("unsupported Attention scaling/soft cap")
    maximum = getattr(config, "max_model_len", None)
    if type(maximum) is not int or not 0 < maximum <= impl.rotary_emb.cos_cache.shape[0]:
        raise AlignmentError("invalid aligned Attention context limit")
    return maximum


def _integer_vector(value, name, minimum):
    import torch

    if not isinstance(value, torch.Tensor) or value.ndim != 1 or value.dtype not in (
        torch.int32, torch.int64
    ) or len(value) < minimum:
        raise AlignmentError(f"invalid {name} integer metadata")
    return value


def _batch(impl, q, key_cache, value_cache, fwd_ctx, positions=None):
    import torch

    context, meta = fwd_ctx.context, fwd_ctx.attn_metadata
    if getattr(context, "is_dummy_run", False) or getattr(fwd_ctx, "spec_decode_metadata", None) is not None or getattr(fwd_ctx, "ubatch_slices", None) is not None:
        raise AlignmentError("dummy/speculative/ubatch Attention is not admitted")
    mode = getattr(context, "forward_mode", None)
    if mode is None or mode.use_cudagraph:
        raise AlignmentError("aligned Attention requires an actual eager forward context")
    nseq = context.scheduled_bs
    if type(nseq) is not int or nseq <= 0 or meta is None:
        raise AlignmentError("invalid scheduled request count")
    if q.ndim != 3 or q.shape[1:] != (32, 128) or q.dtype != torch.bfloat16:
        raise AlignmentError("aligned Attention requires BF16 packed Q[T,32,128]")
    for name in ("cos_cache", "sin_cache"):
        cache = getattr(impl.rotary_emb, name, None)
        if not isinstance(cache, torch.Tensor) or cache.device != q.device:
            raise AlignmentError("RoPE coefficient device differs from this query")
    if getattr(impl, "use_flash_layout", False) or key_cache.ndim != 5 or value_cache.ndim != 5:
        raise AlignmentError("aligned Attention requires native 5D SHUFFLE caches")
    if key_cache.dtype != torch.bfloat16 or value_cache.dtype != torch.bfloat16 or key_cache.device != q.device or value_cache.device != q.device:
        raise AlignmentError("KV cache dtype/device differs from Q")
    pages, heads, dim_pack, page, pack = key_cache.shape
    if pages <= 0 or (heads, dim_pack, pack) != (8, 16, 8) or page <= 0 or page % pack:
        raise AlignmentError("invalid physical K cache")
    if value_cache.shape != (pages, 8, page // pack, 128, pack):
        raise AlignmentError("invalid physical SHUFFLE V cache")
    cuq = _integer_vector(meta.cu_seqlens_q, "cu_seqlens_q", nseq + 1)[:nseq + 1]
    bounds = cuq.detach().cpu().tolist()
    if bounds[0] != 0 or any(b <= a for a, b in pairwise(bounds)) or bounds[-1] > len(q):
        raise AlignmentError("invalid packed query boundaries")
    qlens = [b - a for a, b in pairwise(bounds)]
    lengths = _integer_vector(meta.context_lens, "context_lens", nseq)[:nseq].detach().cpu().tolist()
    if any(n < m or n > impl._aligned_attention_max_len for n, m in zip(lengths, qlens)):
        raise AlignmentError("context length violates query/context capacity")
    slots = _integer_vector(meta.slot_mapping, "slot_mapping", len(q))[:len(q)].detach().cpu().tolist()
    live = slots[:bounds[-1]]
    if any(s < 0 or s >= pages * page for s in live) or len(set(live)) != len(live):
        raise AlignmentError("invalid, duplicate or out-of-capacity live KV write slot")
    if any(s != -1 for s in slots[bounds[-1]:]):
        raise AlignmentError("padded queries must not write KV")
    if positions is not None:
        all_positions = _integer_vector(positions, "positions", len(q))[:len(q)].detach().cpu().tolist()
        if any(p < 0 or p >= impl.rotary_emb.cos_cache.shape[0] for p in all_positions):
            raise AlignmentError("RoPE position exceeds coefficient cache, including padded rows")
        pos = all_positions[:bounds[-1]]
        expected = [i for n, m in zip(lengths, qlens) for i in range(n - m, n)]
        if pos != expected:
            raise AlignmentError("positions do not describe each request's causal suffix")
    if meta.max_seqlen_q != max(qlens):
        raise AlignmentError("max query length does not match actual packed boundaries")
    if context.is_prefill:
        cuk = _integer_vector(meta.cu_seqlens_k, "prefill cu_seqlens_k", nseq + 1)[:nseq + 1].detach().cpu().tolist()
        expected_cuk = [0]
        for length in lengths:
            expected_cuk.append(expected_cuk[-1] + length)
        if cuk != expected_cuk:
            raise AlignmentError("prefill KV boundaries differ from full contexts")
        if not getattr(meta, "has_cached", False):
            if qlens != lengths:
                raise AlignmentError("prefill suffix requires native has_cached gather metadata")
            for r in range(nseq):
                row = live[bounds[r]:bounds[r + 1]]
                for j, slot in enumerate(row):
                    if slot % page != j % page or slot // page != row[j // page * page] // page:
                        raise AlignmentError("pure prefill slot order is not a logical page sequence")
            return cuq, bounds[-1], lengths, None, page, qlens
        starts = _integer_vector(meta.seq_starts, "seq_starts", nseq)[:nseq].detach().cpu().tolist()
        if any(starts) or meta.total_kv != sum(lengths):
            raise AlignmentError("shifted/windowed or inconsistent prefix gather")
        ids_per_token = _integer_vector(meta.batch_id_per_k_token, "batch_id_per_k_token", sum(lengths))[:sum(lengths)].detach().cpu().tolist()
        if ids_per_token != [r for r, length in enumerate(lengths) for _ in range(length)]:
            raise AlignmentError("native prefix gather request mapping is inconsistent")
    table = meta.block_tables
    if not isinstance(table, torch.Tensor) or table.ndim != 2 or table.dtype not in (
        torch.int32, torch.int64
    ) or len(table) < nseq:
        raise AlignmentError("decode requires actual integer block tables")
    table = table[:nseq].detach().cpu()
    for r, length in enumerate(lengths):
        count = math.ceil(length / page)
        if count > table.shape[1]:
            raise AlignmentError("block table does not cover context")
        ids = table[r, :count].tolist()
        if any(i < 0 or i >= pages for i in ids) or len(set(ids)) != len(ids):
            raise AlignmentError("illegal physical page in logical history")
        expected = [ids[p // page] * page + p % page for p in range(length - qlens[r], length)]
        if live[bounds[r]:bounds[r + 1]] != expected:
            raise AlignmentError("write slots do not match each request's causal suffix")
    return cuq, bounds[-1], lengths, table, page, qlens


def validate_aligned_cache_write(impl, q, k, v, positions, fwd_ctx, key_cache, value_cache):
    import torch

    slots = getattr(getattr(fwd_ctx, "attn_metadata", None), "slot_mapping", None)
    for tensor in (k, v, positions, slots):
        if not isinstance(tensor, torch.Tensor) or tensor.device != q.device:
            raise AlignmentError("native writer K/V/positions/slots must share query device")
    if k.shape != (len(q), 8, 128) or v.shape != k.shape or k.dtype != q.dtype or v.dtype != q.dtype:
        raise AlignmentError("native KV writer inputs do not match Q geometry/dtype")
    _batch(impl, q, key_cache, value_cache, fwd_ctx, positions)


def _gather_live_kv(key_cache, value_cache, table, lengths, page, device):
    import torch

    ids = sorted({int(i) for r, length in enumerate(lengths)
                  for i in table[r, :math.ceil(length / page)].tolist()})
    remap = {value: i for i, value in enumerate(ids)}
    select = torch.tensor(ids, dtype=torch.int64, device=device)
    raw_k, raw_v = key_cache[select], value_cache[select]
    keys = raw_k.permute(0, 3, 1, 2, 4).reshape(len(ids), page, 8, 128)
    values = raw_v.permute(0, 2, 4, 1, 3).reshape(len(ids), page, 8, 128)
    packed_k, packed_v, cuk = [], [], [0]
    for r, length in enumerate(lengths):
        row = [remap[int(i)] for i in table[r, :math.ceil(length / page)].tolist()]
        index = torch.tensor(row, dtype=torch.int64, device=device)
        packed_k.append(keys[index].reshape(-1, 8, 128)[:length])
        packed_v.append(values[index].reshape(-1, 8, 128)[:length])
        cuk.append(cuk[-1] + length)
    return torch.cat(packed_k), torch.cat(packed_v), cuk


def aligned_varlen_attention(impl, q, new_key, new_value, key_cache, value_cache, fwd_ctx):
    import torch

    cuq, nq, lengths, table, page, qlens = _batch(impl, q, key_cache, value_cache, fwd_ctx)
    if table is None:
        for tensor in (new_key, new_value):
            if not isinstance(tensor, torch.Tensor) or tensor.shape != (len(q), 8, 128) or tensor.dtype != q.dtype or tensor.device != q.device:
                raise AlignmentError("pure prefill requires actual BF16 new KV tensors")
        k, v = new_key[:nq], new_value[:nq]
        cuk = [0]
        for length in lengths:
            cuk.append(cuk[-1] + length)
    else:
        k, v, cuk = _gather_live_kv(key_cache, value_cache, table, lengths, page, q.device)
    output = _pinned_varlen(q[:nq], k, v,
                            cu_seqlens_q=cuq.to(device=q.device, dtype=torch.int32),
                            cu_seqlens_k=_cu_seqlens_k_tensor(cuk, q.device, fwd_ctx),
                            max_seqlen_q=max(qlens), max_seqlen_k=max(lengths),
                            dropout_p=0.0, softmax_scale=impl.scale, causal=True)
    if not isinstance(output, torch.Tensor) or output.shape != (nq, 32, 128) or output.dtype != q.dtype or output.device != q.device:
        raise AlignmentError("public varlen returned an unexpected output contract")
    if len(q) == nq:
        return output
    padded = torch.zeros_like(q)
    padded[:nq] = output
    return padded


def _install_attention(module):
    cls = getattr(module, "PagedAttentionImpl", None)
    if cls is None or cls.__module__ != module.__name__:
        raise AlignmentError("ATOM main PagedAttentionImpl is unavailable")
    _require_signature(cls.dispatch_backend, ("self", "fwd_ctx", "q", "k", "v"))
    _require_signature(cls.rope_cache, ("self", "q", "k", "v", "qkv", "position", "fwd_ctx"))
    _require_signature(cls.__init__, ("self", "num_heads", "head_dim", "scale", "num_kv_heads", "alibi_slopes"), accepts_extra=True)
    for name in ("forward_impl", "_use_asm_cache_layout"):
        if not callable(getattr(cls, name, None)):
            raise AlignmentError(f"ATOM main {name} call site unavailable")
    original_init, original_rope, original_dispatch = cls.__init__, cls.rope_cache, cls.dispatch_backend
    forward_source = inspect.getsource(cls.forward_impl)
    if "self.rope_cache(" not in forward_source or "self.dispatch_backend(" not in forward_source:
        raise AlignmentError("ATOM main forward no longer calls aligned seams")
    rope_source = inspect.getsource(original_rope)
    if "fused_qk_rope_reshape_and_cache(" not in rope_source or "validate_aligned_cache_write" in rope_source:
        raise AlignmentError("ATOM main native cache writer seam changed")

    @functools.wraps(original_init)
    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        # Only the native Qwen3 dense class enters this path.  Other models
        # continue unchanged; a Qwen3 contract mismatch must fail closed.
        if not enabled():
            return
        config = module.get_current_atom_config()
        if getattr(getattr(config, "hf_config", None), "model_type", None) != "qwen3":
            return
        if type(self) is not cls:
            raise AlignmentError("aligned Qwen3 requires the base MHA implementation")
        self.rotary_emb = align_rope_cache(
            self.rotary_emb, config.hf_config,
            force_triton=module.envs.ATOM_FORCE_ATTN_TRITON,
            fused_qk_norm=self.q_norm is not None or self.k_norm is not None,
        )
        self._aligned_attention_max_len = validate_aligned_attention_config(config, self)
        self._aligned_attention_receipt = initialize_aligned_attention(self.rotary_emb.cos_cache.device)
        self._lumen_alignment_main_active = True

    @functools.wraps(original_rope)
    def rope_cache(self, q, k, v, qkv, position, fwd_ctx):
        if getattr(self, "_lumen_alignment_main_active", False):
            validate_aligned_rope(self.rotary_emb)
            entry = fwd_ctx.kv_cache_data[f"layer_{self.layer_num}"]
            validate_aligned_cache_write(self, q, k, v, position, fwd_ctx,
                                         entry.k_cache, entry.v_cache)
        return original_rope(self, q, k, v, qkv, position, fwd_ctx)

    def aligned_backend(self, q, k, v, k_cache, v_cache, k_scale, v_scale, fwd_ctx):
        if not getattr(self, "_lumen_alignment_main_active", False):
            raise AlignmentError("aligned attention invoked on an inactive instance")
        if self._cache_format != "NHD" or self.use_flash_layout or not self.use_triton_attn:
            raise AlignmentError("native Qwen3 writer did not retain 5D SHUFFLE layout")
        return aligned_varlen_attention(self, q, k, v, k_cache, v_cache, fwd_ctx)

    @functools.wraps(original_dispatch)
    def dispatch(self, fwd_ctx, q, k, v):
        if getattr(self, "_lumen_alignment_main_active", False):
            return self.paged_attention_aligned_varlen
        return original_dispatch(self, fwd_ctx, q, k, v)

    cls.__init__ = init
    cls.rope_cache = rope_cache
    cls.dispatch_backend = dispatch
    cls.paged_attention_aligned_varlen = aligned_backend
    cls._lumen_alignment_main_installed = True


def _install_linear(module):
    import torch

    cls = getattr(module, "LinearBase", None)
    if cls is None or cls.__module__ != module.__name__:
        raise AlignmentError("ATOM main LinearBase is unavailable")
    _require_signature(cls.forward, ("self", "x", "x_scale", "otype", "out"))
    original = cls.forward
    source = inspect.getsource(original)
    if "self.quant_type.value == QuantType.No.value" not in source or "tgemm.mm(x, self.weight, self.bias" not in source:
        raise AlignmentError("ATOM main BF16 No dispatch seam changed")
    native_mm = module.tgemm.mm
    original_tgemm = module.tgemm
    quant_no = module.QuantType.No.value

    @functools.wraps(native_mm)
    def aligned_mm(x, weight, bias=None, *, otype=None, out=None, **kwargs):
        # Main's No branch calls the module-global tgemm.mm with or without
        # ``out``.  Other quantized branches must retain their native GEMM.
        if not getattr(_LINEAR_CALL, "eligible", False):
            if out is None:
                return native_mm(x, weight, bias, otype=otype, **kwargs)
            return native_mm(x, weight, bias, otype=otype, out=out, **kwargs)
        if kwargs or out is not None:
            raise AlignmentError("aligned BF16 dispatch does not support out/scales")
        if otype != torch.bfloat16 or x.dtype != torch.bfloat16 or weight.dtype != x.dtype or x.device != weight.device or not x.is_cuda:
            raise AlignmentError("aligned BF16 dispatch requires same-GPU BF16 operands/output")
        from lumen.ops.atom_bf16_gemm import atom_bf16_gemm

        result = atom_bf16_gemm(x, weight)
        return result if bias is None else result + bias

    # A local proxy leaves the actual AITER module untouched; the thread-local
    # eligibility guard selects only main's unquantized LinearBase.forward arm.
    class _TgemmProxy:
        mm = staticmethod(aligned_mm)

        def __getattr__(self, name):
            return getattr(original_tgemm, name)


    @functools.wraps(original)
    def forward(self, x, x_scale=None, otype=torch.bfloat16, out=None):
        if not enabled() or self.quant_type.value != quant_no:
            return original(self, x, x_scale, otype, out)
        if getattr(_LINEAR_CALL, "eligible", False):
            raise AlignmentError("nested aligned LinearBase.forward is unsupported")
        if out is not None or x_scale is not None:
            raise AlignmentError("aligned BF16 dispatch does not support caller out/scales")
        _LINEAR_CALL.eligible = True
        try:
            return original(self, x, x_scale, otype, out)
        finally:
            _LINEAR_CALL.eligible = False

    # Do not install an AITER-global proxy: main linear's local tgemm binding
    # is sufficient and does not affect MXFP4 or unrelated package consumers.
    module.tgemm = _TgemmProxy()
    cls.forward = forward
    cls._lumen_alignment_main_installed = True


_LINEAR_CALL = threading.local()


def _install_embed_head(module):
    import torch

    head = getattr(module, "ParallelLMHead", None)
    if head is None or head.__module__ != module.__name__:
        raise AlignmentError("ATOM main ParallelLMHead is unavailable")
    if "tgemm.mm(x, self.weight, self.bias)" not in inspect.getsource(head.forward):
        raise AlignmentError("ATOM main LM-head BF16 dispatch seam changed")
    native = module.tgemm
    native_mm = native.mm

    @functools.wraps(native_mm)
    def aligned_mm(x, weight, bias=None, *, otype=None, out=None, **kwargs):
        if not enabled():
            return native_mm(x, weight, bias, **kwargs)
        if out is not None or kwargs or otype not in (None, torch.bfloat16):
            raise AlignmentError("aligned LM-head dispatch has unsupported output args")
        if (
            x.ndim != 2 or weight.ndim != 2 or x.shape[1] != weight.shape[1]
            or x.dtype != torch.bfloat16 or weight.dtype != x.dtype
            or x.device != weight.device or not x.is_cuda
        ):
            raise AlignmentError("aligned LM-head requires same-GPU BF16 operands")
        from lumen.ops.atom_bf16_gemm import atom_bf16_gemm

        output = atom_bf16_gemm(x, weight)
        return output if bias is None else output + bias

    class _TgemmProxy:
        mm = staticmethod(aligned_mm)

        def __getattr__(self, name):
            return getattr(native, name)

    module.tgemm = _TgemmProxy()
    module._lumen_alignment_main_installed = True


def apply(module_name: str) -> bool:
    """Install on a completed ATOM import; return True only when active.

    Wire this call from the shared ``sitecustomize`` hook for every exact
    name in ``_TARGETS``.  A failed explicit opt-in propagates; the shared
    importer must not swallow it or continue with unaligned Qwen3 operators.
    """
    if not enabled() or module_name not in _TARGETS:
        return False
    with _INSTALL_LOCK:
        module = sys.modules.get(module_name)
        if module is None:
            raise AlignmentError(f"ATOM import hook ran without {module_name}")
        cls_name = (
            "PagedAttentionImpl" if module_name.endswith("attention_mha") else
            "ParallelLMHead" if module_name.endswith("embed_head") else "LinearBase"
        )
        cls = getattr(module, cls_name, None)
        if getattr(cls, "_lumen_alignment_main_installed", False):
            return True
        if module_name.endswith("attention_mha"):
            _install_attention(module)
        elif module_name.endswith("embed_head"):
            _install_embed_head(module)
        else:
            _install_linear(module)
        return True
