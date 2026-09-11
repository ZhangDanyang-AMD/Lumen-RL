"""Fused-MoE weight sync between a transformers-5.x trainer and ATOM.

The naming half of this is the same gap ``vllm_moe_weight_sync`` closes for
vLLM. transformers 5.x keeps Qwen3-MoE experts as fused 3D tensors, so the
trainer's IPC weight sync emits::

    model.layers.N.mlp.experts.gate_up_proj    (E, 2*I, H)
    model.layers.N.mlp.experts.down_proj       (E, H, I)

ATOM's ``FusedMoE`` calls the same buffers ``experts.w13_weight`` /
``experts.w2_weight``, and ``atom/rollout/weight_updater.py`` resolves incoming
names against nothing but ``named_parameters()`` and the model's
``packed_modules_mapping`` -- which covers ``q/k/v_proj`` and dense
``gate/up_proj`` only. ``make_expert_params_mapping()`` exists but is used at
load time, not by the updater. So a fused name matches nothing, falls into the
``logger.debug("Unmatched parameter")`` branch and is counted in ``skipped``:
no exception, and at default log level not even a line. Measured on
Qwen3-30B-A3B: 96 tensors per replica per sync (48 layers x 2), i.e. every
routed expert weight, silently pinned at whatever the engine loaded from the
checkpoint.

ATOM adds a second half that vLLM does not have. Its unquantized MoE method
ends ``process_weights_after_loading`` with an aiter ``shuffle_weights`` --
unconditional, no ROCm/env gate -- which permutes ``w13_weight`` /
``w2_weight`` in place into the block-interleaved layout
``aiter.fused_moe.fused_moe`` reads. Nothing re-runs it after a weight update:
the updater's only ``shuffle_weights`` call sits behind ``_is_fp8_param``,
which a BF16 MoE parameter can never satisfy. Renaming alone would therefore
land plain row-major bytes in a buffer the kernel reads as shuffled, trading a
silent no-op for silent garbage. So the bytes are pre-shuffled here, in the
receiver's own staging buffer, by calling ATOM's own ``shuffle_weights`` rather
than a local copy of it -- the layout cannot drift from what ATOM expects
without this code moving with it.

Verified in the release container against ATOM's post-load hook: shuffling
through a staging-buffer view is bitwise identical to what the engine holds
after loading the same weights from a checkpoint, for both parameters, and the
shuffle is a real permutation (applying it twice differs from applying it
once), so it must happen exactly once per sync.

``assert_bucket_fully_applied`` is the guard that would have caught the
original bug: ``broadcast_utility_command_sync`` hands back each engine's
``updated`` count and the caller used to drop it on the floor.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Iterable

import torch

logger = logging.getLogger(__name__)

# The trainer's fused expert leaf -> the ATOM FusedMoE parameter it belongs in.
# Both are (experts, out, in) with matching dims, and w13's first half along the
# intermediate dim is the gate projection, the second half the up projection --
# the same convention transformers uses, so no chunk/reorder is needed, only the
# rename. ATOM has no nested container below ``experts`` (vLLM >= 0.22 grew a
# ``routed_experts`` segment; this fork did not), so the prefix is untouched.
_FUSED_TO_ATOM = {
    "gate_up_proj": "w13_weight",
    "down_proj": "w2_weight",
}

_EXPERTS_SUFFIX = ".experts"


_SHIM_ENV = "LUMENRL_ATOM_FUSED_EXPERT_SHIM"


def atom_routes_fused_experts() -> bool:
    """Whether the ATOM in this process handles fused expert names itself.

    ROCm/ATOM#2028 taught the updater the two names above: it resolves them to
    ``w13_weight`` / ``w2_weight``, drives ``FusedMoE.weight_loader`` per shard
    and re-establishes the expert layout per rewritten slice, in place, at the
    end of the sync. When that ATOM is in the process, everything below is not
    merely redundant -- it is the wrong side of the boundary, and it would go
    on encoding ATOM's private kernel layout in a downstream project. So the
    trainer's names go through untouched instead.

    Both halves stay in the tree because the pinned ATOM in the release image
    predates that support, and the same Lumen-RL has to serve both. The probe
    is a method on the mixin rather than a version number: ATOM has no
    published version that brackets this, and the capability is what matters.

    ``LUMENRL_ATOM_FUSED_EXPERT_SHIM`` overrides the probe: ``force`` keeps the
    shim on against an ATOM that would have taken over, ``off`` hands over
    unconditionally. It exists because which side does the expert relayout
    moves ~36 GB of transient device memory between two processes, which on a
    colocated run is the difference between a KV pool and an assertion — so
    when that bites, an operator needs to be able to move it back without a
    code change.
    """
    override = os.environ.get(_SHIM_ENV, "auto").strip().lower()
    if override == "force":
        return False
    if override == "off":
        return True
    if override not in ("", "auto"):
        logger.warning("ignoring %s=%r; expected auto|force|off", _SHIM_ENV, override)
    try:
        from atom.rollout.weight_updater import WeightUpdaterMixin
    except Exception as exc:  # pragma: no cover - ATOM absent in unit tests
        logger.debug("ATOM weight updater unavailable (%s); staying on the shim", exc)
        return False
    return hasattr(WeightUpdaterMixin, "_apply_fused_expert_weight")


def fused_expert_renames(names: Iterable[str]) -> dict[str, str]:
    """``{incoming name: ATOM parameter name}`` for a bucket's fused expert tensors.

    Empty for a dense model, which is what makes every call site below a no-op
    on the 8B ATOM path.
    """
    renames: dict[str, str] = {}
    for name in names:
        prefix, _, leaf = name.rpartition(".")
        atom_leaf = _FUSED_TO_ATOM.get(leaf)
        if atom_leaf is None or not prefix.endswith(_EXPERTS_SUFFIX):
            continue
        renames[name] = f"{prefix}.{atom_leaf}"
    return renames


def rename_bucket_meta(
    bucket_meta: dict[str, dict[str, Any]], renames: dict[str, str]
) -> dict[str, dict[str, Any]]:
    """Re-key a bucket's metadata, preserving order so offsets stay readable."""
    if not renames:
        return bucket_meta
    return {renames.get(name, name): meta for name, meta in bucket_meta.items()}


def require_unsharded_experts(
    tensor_parallel_size: int, enable_expert_parallel: bool
) -> None:
    """Refuse the sharded cases this path cannot serve.

    The rename works because ATOM's parameter has the same shape as the tensor
    the trainer sends, which puts the updater on its ``tensor.shape ==
    param.shape`` fast path and a plain ``copy_``. Under TP the parameter holds
    only ``intermediate_size // tp_size`` of the intermediate dim, and under EP
    only this rank's experts, so the shapes stop matching and the updater falls
    through to ``weight_loader(param, tensor)`` -- a two-argument call that
    ATOM's ``FusedMoE.weight_loader`` rejects with ``shard_id must be
    ['w1','w2','w3']``, caught and counted as ``skipped``. Sharding these
    tensors correctly means driving the loader per shard and per expert, which
    is a different change; failing here keeps it from silently degrading back
    into the bug this module exists to fix.

    Data parallelism is fine: each DP rank holds a full copy of the weights, so
    the shapes still match and every replica gets the same bytes.
    """
    if int(tensor_parallel_size or 1) != 1 or bool(enable_expert_parallel):
        raise RuntimeError(
            "ATOM fused-MoE weight sync supports tensor_parallel_size=1 without "
            f"expert parallelism only, got tensor_parallel_size="
            f"{tensor_parallel_size} enable_expert_parallel={enable_expert_parallel}. "
            "The rollout would serve stale expert weights; see "
            "lumenrl/engine/inference/atom_moe_weight_sync.py."
        )


def relayout_fused_experts(
    buffer: torch.Tensor,
    bucket_meta: dict[str, dict[str, Any]],
    renames: dict[str, str],
) -> None:
    """Apply ATOM's post-load expert shuffle to a staging buffer, in place.

    ``buffer`` is the receiver-owned uint8 staging buffer ATOM's runner will
    read through its IPC handle, and ``bucket_meta`` is keyed by the trainer's
    names (call this before :func:`rename_bucket_meta`). Wrapping the view in a
    throwaway ``Parameter`` is only to satisfy ``shuffle_weights``' type check;
    it shares storage with the buffer, so the shuffle writes through.
    """
    if not renames:
        return

    from atom.model_ops.utils import shuffle_weights

    for name in renames:
        meta = bucket_meta[name]
        dtype = getattr(torch, str(meta["dtype"]).replace("torch.", ""))
        shape = torch.Size(tuple(meta["shape"]))
        offset = int(meta["offset"])
        nbytes = int(meta["nbytes"])
        if len(shape) != 3:
            raise RuntimeError(
                f"fused MoE weight {name} must be 3D (experts, out, in), got "
                f"{tuple(shape)}. The trainer is not emitting the "
                "transformers-5.x fused layout this module expects."
            )
        view = buffer[offset : offset + nbytes].view(dtype=dtype).view(shape)
        shuffle_weights(torch.nn.Parameter(view, requires_grad=False))


def assert_bucket_fully_applied(
    responses: Any,
    bucket_meta: dict[str, dict[str, Any]],
    *,
    context: str = "ipc",
    exact: bool = True,
) -> None:
    """Fail when an engine did not apply every tensor in the bucket.

    ATOM reports ``updated`` per bucket and counts anything it could not place
    in ``skipped``, which it only logs at debug level -- so a naming drift shows
    up as a slowly growing train/rollout divergence rather than a crash. A BF16
    trainer sends no quantization scales, so every tensor in the bucket must be
    accounted for and the counts have to match exactly.

    ``exact=False`` turns a shortfall into a warning, for the one case where
    ``updated`` is legitimately lower than the bucket: an FP8 rollout. There
    ATOM's ``_apply_packed_weight`` accumulates the shards of a fused parameter
    in a float32 buffer and requantizes once the last one lands, so q/k/v_proj
    count as a single update and the two that arrived first count as nothing at
    all -- ``updated`` under-reports by the number of shards still pending when
    the bucket closes. ATOM returns only that one number, so the receiver cannot
    tell those apart from genuinely skipped tensors. Measured on example 4 (8B
    ATOM FP8): 28 of 39, the 11 being q/k/v and gate/up shards of the three
    layers the bucket straddles.

    ``LUMENRL_WEIGHT_SYNC_CHECK`` selects ``error`` (default), ``warn`` or
    ``off``, matching the vLLM path's knob.
    """
    mode = os.environ.get("LUMENRL_WEIGHT_SYNC_CHECK", "error").lower()
    if mode == "off":
        return

    expected = len(bucket_meta)
    applied = _updated_counts(responses)
    if not applied:
        logger.warning(
            "ATOM weight sync (%s): no engine reported an updated count; "
            "coverage unchecked", context,
        )
        return

    short = [count for count in applied if count != expected]
    if not short:
        return

    message = (
        f"ATOM weight sync ({context}) applied {min(short)}/{expected} tensors of a "
        f"bucket on {len(short)} of {len(applied)} engine(s); the rest were skipped "
        "as unrecognised names, so the rollout engine is now serving a mix of "
        "current and stale weights. Fused MoE experts are the usual cause -- see "
        "lumenrl/engine/inference/atom_moe_weight_sync.py. Set "
        "LUMENRL_WEIGHT_SYNC_CHECK=warn to downgrade this to a log line."
    )
    if mode == "warn" or not exact:
        logger.warning(message)
        return
    raise RuntimeError(message)


def _updated_counts(responses: Any) -> list[int]:
    """Pull the ``updated`` counts out of ``broadcast_utility_command_sync``.

    Each engine answers with ``{"cmd": ..., "result": r}``, where ``r`` is what
    ATOM's runner manager returned -- an int for a single runner, a list of them
    when the engine fans out over several.
    """
    counts: list[int] = []
    for response in responses or []:
        result = response.get("result") if isinstance(response, dict) else response
        for value in result if isinstance(result, (list, tuple)) else [result]:
            if isinstance(value, bool) or not isinstance(value, int):
                continue
            counts.append(int(value))
    return counts
