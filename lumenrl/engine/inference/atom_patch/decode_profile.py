"""Default-off, run-scoped decode profiling for ATOM (diagnostic use only).

Purpose
-------
Localize where decode-step time goes for the RL aligned recipe: host-side
batch validation/sync (``_batch``), live-KV gather/reorder, the varlen
attention call, MXFP4 activation quant + GEMM, and the host/GPU gaps around
them.  This is *diagnostic instrumentation*: a profiled run must never be
used as a speed A/B.

Enable (nothing happens when unset)::

    ATOM_DECODE_PROFILE=1
    ATOM_DECODE_PROFILE_DIR=/dev/shm/.../decode_profile    # default /tmp/...
    ATOM_DECODE_PROFILE_MODE=decode                         # decode|all
    ATOM_DECODE_PROFILE_SKIP_FIRST=5                        # warmup steps
    ATOM_DECODE_PROFILE_EVERY=1                             # sample stride
    ATOM_DECODE_PROFILE_MAX_STEPS=200                       # rows per process

One JSONL row is appended per sampled forward to
``<dir>/<hostname>-<pid>.jsonl`` so the four ATOM replica processes stay
separate.  Each row carries host phase sums, CUDA-event phase sums, the
whole-forward host wall time and the whole-forward GPU span (event pair), so
offline analysis can compare them.

Cost when enabled: one ``torch.cuda.synchronize()`` per sampled forward plus
one file append.  When disabled: a module-level boolean check per call site.

This module must stay import-safe without torch/CUDA (unit tests import it on
CPU); torch is imported lazily and every CUDA call is guarded.
"""

from __future__ import annotations

import json
import os
import socket
import threading
import time

_TRUE = ("1", "true", "yes", "on")


def _flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in _TRUE


ENABLED = _flag("ATOM_DECODE_PROFILE")
_MODE = os.environ.get("ATOM_DECODE_PROFILE_MODE", "decode").strip().lower()
_EVERY = max(1, int(os.environ.get("ATOM_DECODE_PROFILE_EVERY", "1") or "1"))
_MAX_STEPS = max(0, int(os.environ.get("ATOM_DECODE_PROFILE_MAX_STEPS", "200") or "200"))
_SKIP_FIRST = max(0, int(os.environ.get("ATOM_DECODE_PROFILE_SKIP_FIRST", "5") or "5"))
_DIR = os.environ.get("ATOM_DECODE_PROFILE_DIR", "/tmp/atom_decode_profile")
_PATH = os.environ.get("ATOM_DECODE_PROFILE_PATH", "")
_DUMP_KV = _flag("ATOM_DECODE_PROFILE_DUMP_KV")
# Public aliases for the ATOM monkeypatch installer (kept read-only).
DUMP_KV = _DUMP_KV
_KV_DIR = os.environ.get(
    "ATOM_DECODE_PROFILE_KV_DIR", os.path.join(_DIR, "kv_dump")
)
KV_DIR = _KV_DIR

_HOST = socket.gethostname()
_PID = os.getpid()

if ENABLED and not _PATH:
    try:
        os.makedirs(_DIR, exist_ok=True)
        _PATH = os.path.join(_DIR, f"{_HOST}-{_PID}.jsonl")
    except OSError:
        # A read-only /tmp must not break the engine; profiling just turns off.
        ENABLED = False

_lock = threading.Lock()
_seen = 0       # forwards of the sampled kind observed
_profiled = 0   # rows written by this process
_current = None  # type: Step | None


class Step:
    """Per-forward accumulator.  Callers hold an opaque reference."""

    __slots__ = ("kind", "step", "meta", "t0", "host_ms", "count", "gpu_ms", "events", "gpu_start")

    def __init__(self, kind: str, step: int, meta):
        self.kind = kind
        self.step = step
        self.meta = dict(meta or {})
        self.t0 = time.perf_counter()
        self.host_ms: dict[str, float] = {}
        self.count: dict[str, int] = {}
        self.gpu_ms: dict[str, float] = {}
        self.events: list = []
        self.gpu_start = None

    def add_host(self, name: str, seconds: float) -> None:
        self.host_ms[name] = self.host_ms.get(name, 0.0) + seconds * 1000.0
        self.count[name] = self.count.get(name, 0) + 1


class _EventPair:
    __slots__ = ("name", "start", "end")

    def __init__(self, name: str, start, end):
        self.name = name
        self.start = start
        self.end = end


class _SpanTimer:
    __slots__ = ("step", "name", "t0")

    def __init__(self, step, name: str):
        self.step = step
        self.name = name
        self.t0 = 0.0

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc_info):
        if self.step is not None:
            self.step.add_host(self.name, time.perf_counter() - self.t0)
        return False


def current():
    """Current sampled Step, or None (also None when profiling is disabled)."""
    return _current


def span(step, name: str) -> _SpanTimer:
    """Context manager adding host elapsed time of the block to ``name``."""
    return _SpanTimer(step, name)


def add_host(step, name: str, seconds: float) -> None:
    if step is not None:
        step.add_host(name, seconds)


def _new_cuda_events(count: int):
    import torch

    return [torch.cuda.Event(enable_timing=True) for _ in range(count)]


def event_start(step, name: str):
    """Record a start CUDA event; returns a handle for event_end()."""
    if step is None:
        return None
    try:
        start, end = _new_cuda_events(2)
        start.record()
    except Exception:  # no CUDA / no stream: profiling degrades to host only
        return None
    return _EventPair(name, start, end)


def event_end(step, handle) -> None:
    if step is None or handle is None:
        return
    try:
        handle.end.record()
    except Exception:
        return
    step.events.append(handle)


def meta_from_context(context, attn_metadata):
    """Collect batch geometry for one forward (diagnostic rows only)."""
    if not ENABLED:
        return None
    meta: dict = {}
    try:
        meta["scheduled_bs"] = int(getattr(context, "scheduled_bs", -1))
        meta["running_bs"] = int(getattr(context, "running_bs", -1))
        meta["is_prefill"] = bool(getattr(context, "is_prefill", False))
        positions = getattr(context, "positions", None)
        if positions is not None and hasattr(positions, "numel"):
            meta["tokens"] = int(positions.numel())
        for name in ("context_lens", "cu_seqlens_q", "seq_lens"):
            value = getattr(attn_metadata, name, None)
            if value is not None and hasattr(value, "detach"):
                meta[name] = value.detach().cpu().tolist()
        table = getattr(attn_metadata, "block_tables", None)
        if table is not None and hasattr(table, "shape"):
            meta["block_table_shape"] = [int(x) for x in table.shape]
        max_q = getattr(attn_metadata, "max_seqlen_q", None)
        if max_q is not None:
            meta["max_seqlen_q"] = int(max_q)
    except Exception:
        meta["meta_error"] = True
    return meta


def begin_step(kind: str, meta=None):
    """Start a sampled forward.  Returns a Step, or None when not sampled."""
    global _current, _seen
    if not ENABLED:
        return None
    if _MODE == "decode" and kind != "decode":
        return None
    _seen += 1
    index = _seen
    if index <= _SKIP_FIRST or (index - _SKIP_FIRST) % _EVERY != 0:
        return None
    if _MAX_STEPS and _profiled >= _MAX_STEPS:
        return None
    step = Step(kind, index, meta)
    try:
        start = _new_cuda_events(1)[0]
        start.record()
        step.gpu_start = start
    except Exception:
        step.gpu_start = None
    _current = step
    return step


def end_step(step, extra_meta=None) -> None:
    """Finalize a sampled forward: resolve events and append one JSONL row."""
    global _current, _profiled
    if step is None:
        return
    if extra_meta:
        step.meta.update(extra_meta)
    wall_ms = (time.perf_counter() - step.t0) * 1000.0
    gpu_span_ms = None
    if step.gpu_start is not None or step.events:
        try:
            import torch

            end = torch.cuda.Event(enable_timing=True)
            end.record()
            torch.cuda.synchronize()
            if step.gpu_start is not None:
                gpu_span_ms = float(step.gpu_start.elapsed_time(end))
            for handle in step.events:
                try:
                    ms = float(handle.start.elapsed_time(handle.end))
                except Exception:
                    continue
                step.gpu_ms[handle.name] = step.gpu_ms.get(handle.name, 0.0) + ms
        except Exception:
            gpu_span_ms = None
    row = {
        "ts": time.time(),
        "host": _HOST,
        "pid": _PID,
        "kind": step.kind,
        "step": step.step,
        "t_wall_ms": round(wall_ms, 4),
        "gpu_span_ms": None if gpu_span_ms is None else round(gpu_span_ms, 4),
        "host_ms": {k: round(v, 4) for k, v in step.host_ms.items()},
        "counts": dict(step.count),
        "gpu_ms": {k: round(v, 4) for k, v in step.gpu_ms.items()},
        "meta": step.meta,
    }
    try:
        line = json.dumps(row, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        line = None
    if line is not None:
        try:
            with _lock:
                with open(_PATH, "a", encoding="utf-8") as handle:
                    handle.write(line + "\n")
                    handle.flush()
            _profiled += 1
        except OSError:
            pass
    _current = None


def profiled_rows() -> int:
    return _profiled


def enabled() -> bool:
    return ENABLED


def output_path() -> str:
    return _PATH


_kv_dumped = False


def dump_live_kv_once(key_cache, value_cache, table, lengths, page) -> None:
    """Dump one real decode KV snapshot for an offline fused-pack comparison.

    Opt-in with ATOM_DECODE_PROFILE_DUMP_KV=1.  Writes one file per engine
    process: real 5D SHUFFLE caches + CPU page table + lengths + page size.
    Diagnostic only; never used by the inference path.
    """
    global _kv_dumped
    if not (ENABLED and _DUMP_KV) or _kv_dumped:
        return
    try:
        import torch

        os.makedirs(_KV_DIR, exist_ok=True)
        path = os.path.join(_KV_DIR, f"decode-kv-{_HOST}-{_PID}.pt")
        torch.save(
            {
                "key_cache": key_cache.detach().to("cpu"),
                "value_cache": value_cache.detach().to("cpu"),
                "table": table.detach().to("cpu") if hasattr(table, "detach") else table,
                "lengths": [int(x) for x in lengths],
                "page": int(page),
                "key_cache_shape": [int(x) for x in key_cache.shape],
                "value_cache_shape": [int(x) for x in value_cache.shape],
                "dtype": str(key_cache.dtype),
                "host": _HOST,
                "pid": _PID,
            },
            path,
        )
        _kv_dumped = True
    except Exception:
        # A failed diagnostic dump must never affect inference.
        _kv_dumped = True
