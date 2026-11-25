###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Install Lumen-RL's opt-in ATOM main inference wrappers.

A selected KV pack requires a validated aligned path. Missing hooks fail
closed; no performance run may silently use another backend.

Definitions (all default-off):

``ATOM_ALIGNED_KV_PACK=1``
    ``lumenrl.engine.inference.atom_patch.alignment_main._gather_live_kv`` is replaced by a wrapper
    that calls :func:`lumenrl.engine.inference.atom_patch.aligned_kv_pack.fused_gather_live_kv`
    and fails closed if the selected C1 kernel or wrapper errors.

``ATOM_DECODE_PROFILE=1``
    Host/GPU timers are attached to the same call sites the old in-source
    profiler instrumented (attention batch/gather/attention, MXFP4 quant/GEMM,
    and the model-runner forward).  Because this is a monkeypatch, only
    whole-function timings are available; the in-source profiler's sub-phases
    (``batch_cpu_sync`` and the ``gather_*`` breakdown) are not reproduced.
"""

from __future__ import annotations

import functools
import logging
import os
import threading
import time

logger = logging.getLogger(__name__)

_TRUE = ("1", "true", "yes", "on")

_PATCHED: set[str] = set()
_LOCK = threading.Lock()
_LOCAL = threading.local()

_kv_pack = None
_decode_profile = None


def _phases_enabled() -> tuple[bool, bool]:
    pack = os.environ.get("ATOM_ALIGNED_KV_PACK", "0").strip().lower() in _TRUE
    prof = os.environ.get("ATOM_DECODE_PROFILE", "0").strip().lower() in _TRUE
    return pack, prof


def _load_modules():
    global _kv_pack, _decode_profile
    if _kv_pack is None:
        from . import aligned_kv_pack as kv_pack
        from . import decode_profile

        _kv_pack = kv_pack
        _decode_profile = decode_profile
    return _kv_pack, _decode_profile


def _record_host(name: str, seconds: float, counted_sub: bool = False) -> None:
    step = _decode_profile.current() if _decode_profile is not None else None
    if step is None:
        return
    _decode_profile.add_host(step, name, seconds)
    if counted_sub:
        sub = getattr(_LOCAL, "batch_sub", None)
        if sub is not None:
            sub[0] += seconds


def _measure_gpu(name: str, fn):
    step = _decode_profile.current() if _decode_profile is not None else None
    handle = _decode_profile.event_start(step, name) if step is not None else None
    try:
        return fn()
    finally:
        if handle is not None:
            _decode_profile.event_end(step, handle)


def _patch_aligned_attention(module) -> None:
    original_gather = getattr(module, "_gather_live_kv", None)
    original_attention = getattr(module, "aligned_varlen_attention", None)
    original_batch = getattr(module, "_batch", None)
    if original_gather is None or original_attention is None or original_batch is None:
        raise RuntimeError("explicit aligned KV pack/profile requires Lumen alignment hooks")

    pack_enabled, prof_enabled = _phases_enabled()
    kv_pack, decode_profile = _load_modules()
    dumped = [False]
    counters = {"hit": 0, "fallback": 0}

    def _gather_body(key_cache, value_cache, table, lengths, page, device):
        # Optional one-shot real-KV snapshot for offline GPU comparison tools.
        if not dumped[0] and prof_enabled and getattr(decode_profile, "DUMP_KV", False):
            dumped[0] = True
            try:
                decode_profile.dump_live_kv_once(
                    key_cache, value_cache, table, lengths, page
                )
            except Exception as exc:  # pragma: no cover - diagnostic only
                logger.warning("atom_patch: KV dump failed: %s", exc)

        started = time.perf_counter() if prof_enabled else None
        if pack_enabled:
            try:
                out = _measure_gpu(
                    "gather_gpu",
                    lambda: kv_pack.fused_gather_live_kv(
                        key_cache, value_cache, table, lengths, page, device
                    ),
                )
                counters["hit"] += 1
                if started is not None:
                    _record_host("gather_host", time.perf_counter() - started, True)
                return out
            except Exception as exc:
                counters["fallback"] += 1
                logger.error("atom_patch: selected C1 KV pack failed: %s", exc)
                raise RuntimeError("selected C1 KV pack failed; no silent fallback") from exc

        if started is None:
            return original_gather(
                key_cache, value_cache, table, lengths, page, device
            )
        out = _measure_gpu(
            "gather_gpu",
            lambda: original_gather(
                key_cache, value_cache, table, lengths, page, device
            ),
        )
        _record_host("gather_host", time.perf_counter() - started, True)
        return out

    @functools.wraps(original_gather)
    def gather_live_kv(key_cache, value_cache, table, lengths, page, device):
        # Off-mode diagnostics may fall back; selected C1 must fail closed
        # so a candidate run never inherits baseline timings on kernel error.
        try:
            return _gather_body(
                key_cache, value_cache, table, lengths, page, device
            )
        except Exception as exc:
            if pack_enabled:
                # An explicitly selected candidate must not become a hidden
                # baseline step when its kernel or wrapper fails.
                raise
            logger.warning("atom_patch: gather wrapper failed, using legacy: %s", exc)
            return original_gather(
                key_cache, value_cache, table, lengths, page, device
            )

    @functools.wraps(original_attention)
    def aligned_varlen_attention(*args, **kwargs):
        if not prof_enabled:
            return original_attention(*args, **kwargs)
        started = time.perf_counter()
        out = _measure_gpu("attn_gpu", lambda: original_attention(*args, **kwargs))
        _record_host("attn_host", time.perf_counter() - started, True)
        return out

    @functools.wraps(original_batch)
    def batch(impl, q, *args, **kwargs):
        if not prof_enabled:
            return original_batch(impl, q, *args, **kwargs)
        previous = getattr(_LOCAL, "batch_sub", None)
        _LOCAL.batch_sub = [0.0]
        started = time.perf_counter()
        try:
            out = original_batch(impl, q, *args, **kwargs)
        finally:
            total = time.perf_counter() - started
            sub = _LOCAL.batch_sub[0]
            _LOCAL.batch_sub = previous
            _record_host("batch_host", max(0.0, total - sub), False)
        return out

    module._gather_live_kv = gather_live_kv
    module.aligned_varlen_attention = aligned_varlen_attention
    module._batch = batch
    module._lumen_c1_counters = counters
    module._lumen_patched = True


def _patch_linear(module) -> None:
    original = getattr(module, "gemm_a4w4_quant", None)
    if original is None:
        logger.warning("atom_patch: linear.gemm_a4w4_quant missing; not patched")
        return
    _, prof_enabled = _phases_enabled()
    if not prof_enabled:
        return
    _load_modules()

    @functools.wraps(original)
    def gemm_a4w4_quant(*args, **kwargs):
        if not prof_enabled:
            return original(*args, **kwargs)
        started = time.perf_counter()
        out = _measure_gpu("mxfp4_gemm_gpu", lambda: original(*args, **kwargs))
        _record_host("mxfp4_gemm_host", time.perf_counter() - started, True)
        return out

    module.gemm_a4w4_quant = gemm_a4w4_quant
    module._lumen_patched = True


def _patch_mxfp4_2d(module) -> None:
    original = getattr(module, "quantize_mxfp4_2d_activation", None)
    if original is None:
        return
    _, prof_enabled = _phases_enabled()
    _load_modules()

    @functools.wraps(original)
    def quantize_mxfp4_2d_activation(*args, **kwargs):
        if not prof_enabled:
            return original(*args, **kwargs)
        started = time.perf_counter()
        out = _measure_gpu(
            "mxfp4_quant_gpu", lambda: original(*args, **kwargs)
        )
        _record_host("mxfp4_quant_host", time.perf_counter() - started, True)
        return out

    module.quantize_mxfp4_2d_activation = quantize_mxfp4_2d_activation
    module._lumen_patched = True


def _patch_model_runner(module) -> None:
    runner_cls = getattr(module, "ModelRunner", None)
    original = getattr(runner_cls, "run_model", None) if runner_cls is not None else None
    if original is None:
        logger.warning("atom_patch: ModelRunner.run_model missing; not patched")
        return
    _, prof_enabled = _phases_enabled()
    if not prof_enabled:
        return
    _, decode_profile = _load_modules()

    @functools.wraps(original)
    def run_model(self, *args, **kwargs):
        if not prof_enabled:
            return original(self, *args, **kwargs)
        step = None
        try:
            from atom.utils.forward_context import get_forward_context

            forward_context = get_forward_context()
            context = forward_context.context
            step = decode_profile.begin_step(
                "prefill" if context.is_prefill else "decode",
                decode_profile.meta_from_context(context, forward_context.attn_metadata),
            )
        except Exception:
            step = None
        try:
            return original(self, *args, **kwargs)
        finally:
            if step is not None:
                try:
                    decode_profile.end_step(step)
                except Exception as exc:  # pragma: no cover - diagnostic only
                    logger.warning("atom_patch: profiler end_step failed: %s", exc)

    runner_cls.run_model = run_model
    module._lumen_patched = True


_PATCHERS = {
    "lumenrl.engine.inference.atom_patch.alignment_main": _patch_aligned_attention,
    "atom.model_ops.linear": _patch_linear,
    "atom.model_engine.model_runner": _patch_model_runner,
}


def apply(module_name: str) -> None:
    """Patch one validated call site (idempotent, fail-closed when selected)."""
    patcher = _PATCHERS.get(module_name)
    if patcher is None:
        return
    pack_enabled, prof_enabled = _phases_enabled()
    if not (pack_enabled or prof_enabled):
        return
    with _LOCK:
        if module_name in _PATCHED:
            return
        import sys

        module = sys.modules.get(module_name)
        if module is None:
            raise RuntimeError(f"selected Lumen patch module not loaded: {module_name}")
        patcher(module)
        _PATCHED.add(module_name)
