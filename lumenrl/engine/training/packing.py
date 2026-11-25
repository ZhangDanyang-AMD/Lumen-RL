"""Sequence packing for efficient training with AITER flash_attn_varlen_func.

Eliminates padding waste by concatenating actual tokens from multiple sequences
into a single flat tensor and using variable-length flash attention to maintain
correct cross-sequence isolation.

Usage in training step::

    packed = pack_sequences(input_ids, attention_mask)
    with PackingContext(packed.cu_seqlens, packed.max_seqlen):
        outputs = model(input_ids=packed.input_ids,
                        position_ids=packed.position_ids,
                        attention_mask=None)
        logits = outputs.logits.squeeze(0)
        flat_lp = packed_token_log_probs(logits, packed.input_ids.squeeze(0),
                                         packed.cu_seqlens)
        log_probs = unpack_log_probs(flat_lp, packed.cu_seqlens,
                                      packed.seq_lens, input_ids.shape[1])
"""

from __future__ import annotations

import functools
import hashlib
import importlib
import inspect
import json
import logging
import os
import threading
from pathlib import Path
from typing import NamedTuple, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Flash-Attention's Triton cross-entropy, the same kernel verl reaches through
# `logprobs_from_logits`. It fuses log_softmax + gather and supports an in-place
# backward, so per-token log-probs cost no extra vocab-sized allocation.
try:
    from flash_attn.ops.triton.cross_entropy import cross_entropy_loss as _FUSED_CROSS_ENTROPY
except ImportError:  # fall back to chunked log_softmax + gather
    _FUSED_CROSS_ENTROPY = None

# ---------------------------------------------------------------------------
# Packing context (module-global, survives gradient checkpointing recompute)
# ---------------------------------------------------------------------------
# PyTorch's non-reentrant gradient checkpointing (use_reentrant=False)
# recomputes forward in a context where threading.local() state is NOT
# visible.  We use plain module globals instead — safe because training
# is single-threaded per process (FSDP2 uses one process per GPU).

_cu_seqlens: torch.Tensor | None = None
_max_seqlen: int = 0


def set_packing_context(cu_seqlens: torch.Tensor, max_seqlen: int) -> None:
    """Store packing metadata (used by attention patch during forward)."""
    global _cu_seqlens, _max_seqlen
    _cu_seqlens = cu_seqlens
    _max_seqlen = max_seqlen


def get_packing_context() -> tuple[torch.Tensor, int] | None:
    """Return (cu_seqlens, max_seqlen) or None if not in a packed forward pass."""
    if _cu_seqlens is None:
        return None
    return _cu_seqlens, _max_seqlen


def clear_packing_context() -> None:
    global _cu_seqlens, _max_seqlen
    _cu_seqlens = None
    _max_seqlen = 0


class PackingContext:
    """Context manager that keeps packing metadata alive through backward.

    Must wrap both forward AND backward so that gradient checkpointing
    recomputation can see the packing state.
    """

    def __init__(self, cu_seqlens: torch.Tensor, max_seqlen: int) -> None:
        self.cu_seqlens = cu_seqlens
        self.max_seqlen = max_seqlen

    def __enter__(self) -> "PackingContext":
        set_packing_context(self.cu_seqlens, self.max_seqlen)
        return self

    def __exit__(self, *args: object) -> None:
        clear_packing_context()


# ---------------------------------------------------------------------------
# Pack / unpack utilities
# ---------------------------------------------------------------------------


class PackedBatch(NamedTuple):
    input_ids: torch.Tensor      # (1, total_tokens)
    position_ids: torch.Tensor   # (1, total_tokens)
    cu_seqlens: torch.Tensor     # (B+1,) int32
    seq_lens: torch.Tensor       # (B,) long
    max_seqlen: int


def pack_from_nested(
    input_ids: torch.Tensor,
    cu_seqlens: torch.Tensor,
    seq_lens: torch.Tensor,
) -> PackedBatch:
    """Create a :class:`PackedBatch` from pre-packed (remove-padding) tensors.

    Unlike :func:`pack_sequences` which takes padded ``(B, S)`` inputs,
    this accepts already-concatenated tokens (e.g. from nested tensors or
    an upstream packing pass).

    Args:
        input_ids: ``(1, total_tokens)`` or ``(total_tokens,)`` already-packed tokens.
        cu_seqlens: ``(B+1,)`` int32 cumulative sequence lengths.
        seq_lens: ``(B,)`` long per-sequence actual lengths.

    Returns:
        A :class:`PackedBatch` ready for model forward + ``PackingContext``.
    """
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    total_tokens = input_ids.shape[-1]
    device = input_ids.device
    max_seqlen = int(seq_lens.max().item())

    position_ids = torch.empty(total_tokens, dtype=torch.long, device=device)
    B = seq_lens.shape[0]
    for i in range(B):
        start = int(cu_seqlens[i].item())
        sl = int(seq_lens[i].item())
        position_ids[start:start + sl] = torch.arange(sl, device=device)

    return PackedBatch(
        input_ids=input_ids,
        position_ids=position_ids.unsqueeze(0),
        cu_seqlens=cu_seqlens,
        seq_lens=seq_lens,
        max_seqlen=max_seqlen,
    )


def pack_sequences(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    tp_align: int = 1,
) -> PackedBatch:
    """Pack left-padded sequences into a contiguous flat tensor.

    Args:
        input_ids: ``(B, S)`` left-padded token IDs.
        attention_mask: ``(B, S)`` where 1 = real token, 0 = pad.
        tp_align: Pad total_tokens to a multiple of this value (TP size)
            so that sequence-parallel reduce_scatter doesn't assert.

    Returns:
        A :class:`PackedBatch` with concatenated tokens, cumulative sequence
        lengths, per-sequence position IDs, and per-sequence actual lengths.
    """
    B, S = input_ids.shape
    device = input_ids.device

    seq_lens = attention_mask.sum(dim=1).long()          # (B,)
    total_tokens = int(seq_lens.sum().item())
    max_seqlen = int(seq_lens.max().item())

    pad_len = (tp_align - total_tokens % tp_align) % tp_align
    total_padded = total_tokens + pad_len

    packed_ids = torch.zeros(total_padded, dtype=input_ids.dtype, device=device)
    position_ids = torch.zeros(total_padded, dtype=torch.long, device=device)

    cu_seqlens = torch.zeros(B + 1, dtype=torch.int32, device=device)
    cu_seqlens[1:] = torch.cumsum(seq_lens.int(), dim=0)

    offset = 0
    for i in range(B):
        sl = int(seq_lens[i].item())
        # Left-padded: actual tokens are at the right end [S-sl, S)
        packed_ids[offset:offset + sl] = input_ids[i, S - sl:S]
        position_ids[offset:offset + sl] = torch.arange(sl, device=device)
        offset += sl

    return PackedBatch(
        input_ids=packed_ids.unsqueeze(0),       # (1, total_padded)
        position_ids=position_ids.unsqueeze(0),  # (1, total_padded)
        cu_seqlens=cu_seqlens,
        seq_lens=seq_lens,
        max_seqlen=max_seqlen,
    )


def _atom_true_vocab_size(width: int) -> int:
    value = os.environ.get("LUMENRL_ATOM_TRUE_VOCAB_SIZE")
    if value is None:
        raise RuntimeError("ATOM alignment requires a resolved true vocabulary size")
    true_vocab = int(value)
    if not 0 < true_vocab <= width:
        raise ValueError("ATOM true vocabulary must fit the logits width")
    return true_vocab


def packed_token_log_probs(
    logits: torch.Tensor,
    packed_ids: torch.Tensor,
    cu_seqlens: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Compute per-token log-probs from packed (flat) logits.

    Performs the shifted log_softmax + gather per sequence, vectorized.

    Args:
        logits: ``(total_padded, V)`` model output (may include TP-alignment
            padding beyond ``cu_seqlens[-1]``).
        packed_ids: ``(total_padded,)`` packed token IDs.
        cu_seqlens: ``(B+1,)`` cumulative sequence lengths.
        temperature: logits are divided by this before ``log_softmax``, matching
            verl's ``logits.div_(temperature)`` in ``_forward_micro_batch`` (so the
            training log-probs reflect the sampling distribution). Must be applied
            consistently to old/train/ref log-probs or the importance ratio drifts.

    Returns:
        Flat ``(total_real_tokens - B,)`` float32 tensor of shifted log-probs.
        Each sequence *i* contributes ``sl_i - 1`` values.
    """
    total_real = int(cu_seqlens[-1].item())
    logits = logits[:total_real]
    packed_ids = packed_ids[:total_real]
    total_tokens = total_real
    device = logits.device

    # not_last[j] = True if j is NOT the last token of any sequence
    not_last = torch.ones(total_tokens, dtype=torch.bool, device=device)
    not_last[cu_seqlens[1:].long() - 1] = False

    if os.environ.get("LUMENRL_ATOM_OP_ALIGNMENT", "0") == "1":
        # ATOM reports raw fp32 log-softmax, not temperature-scaled sampling logprobs.
        # Slice rows before gathering to avoid another full-vocabulary logits copy.
        if total_tokens == 0:
            return logits.new_empty(0, dtype=torch.float32)
        true_vocab = _atom_true_vocab_size(logits.shape[-1])
        if bool((packed_ids >= true_vocab).any()):
            raise ValueError("Packed token ID is outside ATOM's true vocabulary")
        padded_columns = torch.arange(logits.shape[-1], device=device) >= true_vocab
        targets = torch.empty_like(packed_ids)
        targets[:-1] = packed_ids[1:]
        targets[-1] = 0
        parts = []
        for start in range(0, total_tokens, 256):
            end = min(start + 256, total_tokens)
            # Match ATOM's padded-width masked reduction without mutating FSDP output.
            z = logits[start:end].float().masked_fill(padded_columns, float("-inf"))
            row_lp = F.log_softmax(z, dim=-1)
            parts.append(row_lp.gather(-1, targets[start:end, None]).squeeze(-1))
        if not parts:
            return logits.new_empty(0, dtype=torch.float32)
        return torch.cat(parts)[not_last]

    if _FUSED_CROSS_ENTROPY is not None and logits.is_cuda:
        # Fused path: one Triton kernel over the full (total_tokens, V) buffer,
        # with the backward writing dlogits in place. Never materializes a
        # log_softmax tensor, and the boundary rows are dropped *after* the
        # kernel so no second vocab-sized copy is made either. Their gradient is
        # zero because the selection below routes no upstream grad to them.
        targets = torch.empty_like(packed_ids)
        targets[:-1] = packed_ids[1:]
        targets[-1] = 0  # position is dropped by not_last
        losses, _ = _FUSED_CROSS_ENTROPY(
            logits if logits.is_contiguous() else logits.contiguous(),
            targets,
            logit_scale=(1.0 / temperature) if temperature != 1.0 else 1.0,
            inplace_backward=True,
        )
        return (-losses)[not_last].float()  # (total - B,)

    # Fallback: chunked log_softmax + gather. Needs the shifted copies.
    not_first = torch.ones(total_tokens, dtype=torch.bool, device=device)
    not_first[cu_seqlens[:-1].long()] = False
    shifted_logits = logits[not_last]                          # (total - B, V)
    shifted_targets = packed_ids[not_first].unsqueeze(-1)      # (total - B, 1)

    chunk_size = 4096
    lp_parts = []
    for start in range(0, shifted_logits.shape[0], chunk_size):
        end = min(start + chunk_size, shifted_logits.shape[0])
        z = shifted_logits[start:end]
        if temperature != 1.0:
            z = z / temperature
        row_lp = F.log_softmax(z, dim=-1)
        lp_parts.append(
            row_lp.gather(-1, shifted_targets[start:end]).squeeze(-1)
        )
    return torch.cat(lp_parts, dim=0).float()  # (total - B,)


@torch.no_grad()
def packed_token_entropy(
    logits: torch.Tensor,
    cu_seqlens: torch.Tensor,
    chunk_size: int = 1024,
    temperature: float = 1.0,
    upcast: bool = True,
) -> torch.Tensor:
    """Per-token predictive entropy for packed (flat) logits, shifted.

    Entropy at each non-last position ``H = logsumexp(z) - sum(softmax(z) * z)``,
    chunked to bound peak memory. Detached (metric only).

    Args:
        temperature: logits are divided by this before the entropy computation,
            matching verl's ``logits.div_(temperature)`` prior to
            ``entropy_from_logits`` (so the metric reflects the sampling
            distribution at the rollout temperature).
        upcast: if True (default) compute in float32; set False to compute in the
            logits' native dtype (e.g. bf16) to match verl's ``actor/entropy``,
            which is computed on the bf16 forward logits without upcasting.

    Returns a flat ``(total_tokens - B,)`` tensor aligned with
    :func:`packed_token_log_probs` output.
    """
    if os.environ.get("LUMENRL_ATOM_OP_ALIGNMENT", "0") == "1":
        # Metrics use the same raw fp32 distribution as aligned token logprobs.
        temperature, upcast = 1.0, True
        true_vocab = _atom_true_vocab_size(logits.shape[-1])
        # A zero-probability padded column must not introduce 0 * -inf NaNs.
        logits = logits[:, :true_vocab]
    total_real = int(cu_seqlens[-1].item())
    logits = logits[:total_real]
    total_tokens = total_real
    device = logits.device
    not_last = torch.ones(total_tokens, dtype=torch.bool, device=device)
    not_last[cu_seqlens[1:].long() - 1] = False

    # Chunk over row slices (views) and drop the boundary rows afterwards, so no
    # vocab-sized copy of the logits is ever materialized.
    parts = []
    for start in range(0, total_tokens, chunk_size):
        end = min(start + chunk_size, total_tokens)
        z = logits[start:end]
        if upcast:
            z = z.float()
        if temperature != 1.0:
            z = z / temperature
        lse = torch.logsumexp(z, dim=-1)
        probs = torch.softmax(z, dim=-1)
        ent = lse - (probs * z).sum(dim=-1)
        parts.append(ent.float())
    if not parts:
        return torch.zeros(0, dtype=torch.float32, device=device)
    return torch.cat(parts, dim=0)[not_last]


def unpack_log_probs(
    flat_lp: torch.Tensor,
    cu_seqlens: torch.Tensor,
    seq_lens: torch.Tensor,
    padded_seq_len: int,
) -> torch.Tensor:
    """Scatter packed log-probs back into left-padded ``[B, S-1]`` format.

    Args:
        flat_lp: ``(total_tokens - B,)`` from :func:`packed_token_log_probs`.
        cu_seqlens: ``(B+1,)`` cumulative sequence lengths.
        seq_lens: ``(B,)`` per-sequence actual lengths.
        padded_seq_len: ``S`` — the original padded sequence dimension.

    Returns:
        ``[B, S-1]`` tensor with log-probs placed at the correct positions
        for left-padded sequences.  Padding positions are zero.
    """
    B = seq_lens.shape[0]
    S_m1 = padded_seq_len - 1
    result = flat_lp.new_zeros(B, S_m1)

    # Shifted cu_seqlens: cumulative (sl - 1) for indexing into flat_lp
    shifted_lens = seq_lens - 1
    shifted_cu = torch.zeros(B + 1, dtype=torch.long, device=flat_lp.device)
    shifted_cu[1:] = torch.cumsum(shifted_lens, dim=0)

    for i in range(B):
        sl = int(seq_lens[i].item())
        if sl <= 1:
            continue
        src_start = int(shifted_cu[i].item())
        src_end = int(shifted_cu[i + 1].item())
        # Left-padded: shifted positions are [S-sl, S-1)
        dst_start = S_m1 - (sl - 1)
        result[i, dst_start:dst_start + (sl - 1)] = flat_lp[src_start:src_end]

    return result


# ---------------------------------------------------------------------------
# Varlen attention patch
# ---------------------------------------------------------------------------

_original_attn_fn = None  # saved reference to the previous sdpa function
_ALIGNED_ATTN_LAUNCH_LOCK = threading.RLock()
_ALIGNED_ATTN_INITIALIZED = {}
_ALIGNED_ATTN_CONFIG_KEYS = frozenset({
    "BLOCK_M", "BLOCK_N", "PRELOAD_V", "waves_per_eu", "num_warps",
    "num_ctas", "num_stages", "enable_fp_fusion",
})


@functools.lru_cache(maxsize=4)
def _load_aligned_attention_contract(path: str, expected_sha256: str):
    """Load the explicit shared candidate once; never derive config from shape."""
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256.lower():
        raise RuntimeError("Shared Attention config SHA256 mismatch")
    contract = json.loads(raw)
    if (contract.get("schema_version") != 1 or contract.get("backend") != "triton"
            or contract.get("arch") != "gfx950" or contract.get("dtype") != "bfloat16"
            or contract.get("module") != "aiter.ops.triton.attention.mha"
            or contract.get("function") != "flash_attn_varlen_func"):
        raise RuntimeError("Unsupported shared Attention contract")
    config = contract["config"]
    if set(config) != _ALIGNED_ATTN_CONFIG_KEYS or config["enable_fp_fusion"] is not False:
        raise RuntimeError("Shared Attention requires a complete explicit no-fusion config")
    if type(config["PRELOAD_V"]) is not bool:
        raise RuntimeError("Invalid shared Attention PRELOAD_V")
    for name in _ALIGNED_ATTN_CONFIG_KEYS - {"PRELOAD_V", "enable_fp_fusion"}:
        if type(config[name]) is not int or config[name] <= 0:
            raise RuntimeError("Invalid shared Attention launch option: " + name)
    required_state = {"_MHA_IMPL": "default", "_USE_FUSED_BWD_KERNEL": False,
                      "_USE_INT64_STRIDES": True}
    if contract.get("runtime_requirements") != required_state:
        raise RuntimeError("Shared Attention runtime requirements differ")
    source_modules = {
        "aiter/ops/triton/attention/mha.py": "aiter.ops.triton.attention.mha",
        "aiter/ops/triton/_triton_kernels/attention/mha.py": "aiter.ops.triton._triton_kernels.attention.mha",
        "aiter/ops/triton/attention/mha_onekernel_bwd.py": "aiter.ops.triton.attention.mha_onekernel_bwd",
        "triton/backends/amd/compiler.py": "triton.backends.amd.compiler",
    }
    if set(contract["sources"]) != set(source_modules):
        raise RuntimeError("Shared Attention dependency identities are incomplete")
    for name, module_name in source_modules.items():
        source = contract["sources"][name]
        actual_module = importlib.import_module(module_name)
        if (Path(actual_module.__file__).resolve() != Path(source["path"]).resolve()
                or hashlib.sha256(Path(actual_module.__file__).read_bytes()).hexdigest() != source["sha256"]):
            raise RuntimeError("Shared Attention dependency source SHA256 mismatch")
    api = importlib.import_module(contract["module"])
    if (Path(api.__file__).resolve() != Path(contract["module_path"]).resolve()
            or hashlib.sha256(Path(api.__file__).read_bytes()).hexdigest() != contract["module_sha256"]):
        raise RuntimeError("Shared Attention public module identity mismatch")
    function = getattr(api, contract["function"])
    if (hashlib.sha256(inspect.getsource(function).encode("utf-8")).hexdigest() != contract["function_sha256"]
            or str(inspect.signature(function)) != contract["api_signature"]):
        raise RuntimeError("Shared Attention public function identity mismatch")
    compiler = importlib.import_module("triton.backends.amd.compiler")
    if "enable_fp_fusion" not in compiler.HIPOptions.__dataclass_fields__:
        raise RuntimeError("Compiler cannot enforce shared Attention no-fusion option")
    logger.info("Shared Attention contract loaded: %s sha256=%s", path, expected_sha256)
    return api, function, contract


def _aligned_triton_attention(module, q, k, v, call_kwargs):
    """Dispatch the supported path after one source-bound initialization check.

    Default one-kernel backward does not use the public deterministic argument as
    a backend selector. Preserve that argument, but refuse global DAO/fused-bwd
    overrides rather than silently changing global library state.
    """
    path = os.environ.get("LUMENRL_ATOM_ATTENTION_CONFIG")
    digest = os.environ.get("LUMENRL_ATOM_ATTENTION_CONFIG_SHA256")
    if not path or not digest:
        raise RuntimeError("Alignment Attention requires shared config path and SHA256")
    api, function, contract = _load_aligned_attention_contract(path, digest)
    for name, expected in contract["runtime_requirements"].items():
        actual = getattr(api, name, None)
        if type(actual) is not type(expected) or actual != expected:
            raise RuntimeError("Shared Attention runtime override is unsupported: " + name)
    if api.get_arch() != contract["arch"]:
        raise RuntimeError("Shared Attention requires the validated GPU architecture")
    supported = contract["input_contract"]
    if (q.dtype != torch.bfloat16 or k.dtype != q.dtype or v.dtype != q.dtype
            or q.shape[1:] != (supported["q_heads"], supported["head_dim"])
            or k.shape[1:] != (supported["kv_heads"], supported["head_dim"])
            or v.shape[1:] != k.shape[1:]
            or call_kwargs["dropout_p"] != supported["dropout_p"]
            or call_kwargs["softmax_scale"] != supported["softmax_scale"]
            or call_kwargs["causal"] is not supported["causal"]):
        raise RuntimeError("Inputs are outside the shared Attention validation contract")
    if torch.compiler.is_compiling() or (q.is_cuda and torch.cuda.is_current_stream_capturing()):
        raise RuntimeError("Checked shared Attention currently requires eager execution")
    original, receipts = _initialize_aligned_attention(api, function, contract, q.device, digest.lower())
    if api._attn_fwd is not original or getattr(api, contract["function"]) is not function:
        raise RuntimeError("Shared Attention runtime binding changed")
    result = function(q, k, v, **call_kwargs, backend="triton", config=dict(contract["config"]))
    module._lumenrl_aligned_attention_receipt = {
        "config_sha256": digest.lower(), "backend": "triton",
        "verification_level": "zero-input initialization launch; explicit config on each real call",
        "initialization_launches": [dict(receipt) for receipt in receipts],
        "return_lse": call_kwargs["return_lse"], "deterministic": call_kwargs["deterministic"],
        "backward": "existing default one-kernel; config controls forward only",
    }
    return result


def _initialize_aligned_attention(api, function, contract, device, digest):
    """Verify compiled options once per stable API/config/device binding, without RNG use."""
    config = dict(contract["config"])
    owner = threading.get_ident()
    key = (id(api), id(function), str(device), digest)
    with _ALIGNED_ATTN_LAUNCH_LOCK:
        original = api._attn_fwd
        cached = _ALIGNED_ATTN_INITIALIZED.get(key)
        if cached is not None and cached[0] is original:
            return cached
        receipts = []

        class CheckedLaunch:
            def __getitem__(self, grid):
                launch = original[grid]

                def invoke(*args, **kwargs):
                    if threading.get_ident() != owner:
                        # Other threads retain the exact original launch behavior.
                        return launch(*args, **kwargs)
                    if any(kwargs.get(key) != value or type(kwargs.get(key)) is not type(value)
                           for key, value in config.items()):
                        raise RuntimeError("Public Attention did not forward the fixed config")
                    if kwargs.get("USE_INT64_STRIDES") is not True:
                        raise RuntimeError("Shared Attention int64 stride option was not effective")
                    compiled = launch(*args, **kwargs)
                    metadata = getattr(compiled, "metadata", None)
                    if (getattr(metadata, "enable_fp_fusion", None) is not False
                            or getattr(metadata, "num_warps", None) != config["num_warps"]):
                        raise RuntimeError("Compiled Attention no-fusion config was not effective")
                    receipts.append({"enable_fp_fusion": False,
                                     "num_warps": metadata.num_warps,
                                     "kernel": getattr(metadata, "name", None)})
                    return compiled

                return invoke

        checked = CheckedLaunch()
        api._attn_fwd = checked
        try:
            cpu_rng = torch.random.get_rng_state()
            cuda_rng = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
            supported = contract["input_contract"]
            with torch.no_grad():
                q0 = torch.zeros((1, supported["q_heads"], supported["head_dim"]), dtype=torch.bfloat16, device=device)
                k0 = torch.zeros((1, supported["kv_heads"], supported["head_dim"]), dtype=torch.bfloat16, device=device)
                v0 = torch.zeros_like(k0)
                cu = torch.tensor([0, 1], dtype=torch.int32, device=device)
                result = function(q0, k0, v0, cu_seqlens_q=cu, cu_seqlens_k=cu,
                                  max_seqlen_q=1, max_seqlen_k=1, dropout_p=0.0,
                                  softmax_scale=supported["softmax_scale"], causal=True,
                                  return_lse=False, deterministic=False,
                                  backend="triton", config=dict(config))
                output = result[0] if isinstance(result, tuple) else result
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                if not torch.isfinite(output).all() or torch.count_nonzero(output):
                    raise RuntimeError("Shared Attention zero-input initialization failed")
            if not torch.equal(cpu_rng, torch.random.get_rng_state()) or (
                    cuda_rng is not None and not torch.equal(cuda_rng, torch.cuda.get_rng_state(device))):
                raise RuntimeError("Shared Attention initialization unexpectedly consumed RNG")
            if not receipts:
                raise RuntimeError("Shared Attention produced no verifiable Triton initialization launch")
            verified = (original, tuple(receipts))
            _ALIGNED_ATTN_INITIALIZED[key] = verified
            return verified
        finally:
            if api._attn_fwd is checked:
                api._attn_fwd = original
            else:
                raise RuntimeError("Shared Attention launch binding changed concurrently")


def configure_packed_attention_determinism(model: torch.nn.Module, enabled: bool) -> None:
    """Configure this model's packed backward without changing global torch modes."""
    if type(enabled) is not bool:
        raise ValueError("Packed attention determinism must be boolean")
    for module in model.modules():
        module._lumenrl_deterministic_attention = enabled


def _varlen_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    is_causal: bool | None = None,
    **kwargs,
) -> Tuple[torch.Tensor, None]:
    """AITER varlen flash attention wrapper for HF's attention interface.

    When a packing context is active, uses ``flash_attn_varlen_func`` with
    ``cu_seqlens`` for correct cross-sequence isolation.  Otherwise delegates
    to the original attention function (Lumen's ``flash_attn_func`` path).
    """
    aligned = os.environ.get("LUMENRL_ATOM_OP_ALIGNMENT", "0") == "1"
    ctx = get_packing_context()
    if ctx is None:
        if aligned:
            raise RuntimeError("Shared alignment Attention requires a packing context")
        # Not in a packed forward — use original attention path
        return _original_attn_fn(
            module, query, key, value, attention_mask,
            dropout=dropout, scaling=scaling, is_causal=is_causal, **kwargs,
        )

    cu_seqlens, max_seqlen = ctx

    if aligned and (attention_mask is not None or kwargs.get("sliding_window") not in (None, -1)):
        raise RuntimeError("Shared alignment Attention supports the unmasked causal packed path only")

    # HF passes QKV as (B, H, S, D) where B=1 for packed input
    # Transpose to (B, S, H, D) then flatten to (total_tokens, H, D)
    B_dim, H_q, S_dim, D = query.shape
    H_k = key.shape[1]

    q = query.transpose(1, 2).reshape(-1, H_q, D)   # (total_tokens, H_q, D)
    k = key.transpose(1, 2).reshape(-1, H_k, D)     # (total_tokens, H_k, D)
    v = value.transpose(1, 2).reshape(-1, H_k, D)   # (total_tokens, H_k, D)

    softmax_scale = scaling if scaling is not None else D ** (-0.5)
    needs_grad = torch.is_grad_enabled() and any(
        t.requires_grad for t in [q, k, v]
    )

    call_kwargs = dict(
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        dropout_p=dropout,
        softmax_scale=softmax_scale,
        causal=True,
        return_lse=needs_grad,
        deterministic=getattr(module, "_lumenrl_deterministic_attention", False),
    )
    if aligned:
        result = _aligned_triton_attention(module, q, k, v, call_kwargs)
    else:
        # Preserve the pre-existing default route; no shared config is read here.
        from aiter import flash_attn_varlen_func

        result = flash_attn_varlen_func(q, k, v, **call_kwargs)

    attn_output = result[0] if isinstance(result, tuple) else result
    # Reshape back to (B, S, H_q, D) for HF
    attn_output = attn_output.view(B_dim, S_dim, H_q, D).contiguous()
    return attn_output, None


def patch_attention_for_packing() -> bool:
    """Install varlen-aware attention as the HF ``sdpa`` backend.

    Must be called AFTER Lumen's ``patch_hf_sdpa()`` so that the original
    Lumen attention function is captured and used as the fallback.

    Returns True if the patch was applied, False otherwise.
    """
    global _original_attn_fn

    try:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    except ImportError:
        logger.warning("Cannot import ALL_ATTENTION_FUNCTIONS; packing attention patch skipped.")
        return False

    # Save the current sdpa function (should be Lumen's _lumen_aiter_attention_forward)
    _original_attn_fn = ALL_ATTENTION_FUNCTIONS.get("sdpa")
    if _original_attn_fn is None:
        logger.warning("No 'sdpa' attention function registered; packing patch skipped.")
        return False

    ALL_ATTENTION_FUNCTIONS["sdpa"] = _varlen_attention_forward
    logger.info("Packing: HF sdpa attention patched -> varlen-aware (AITER flash_attn_varlen_func)")
    return True
