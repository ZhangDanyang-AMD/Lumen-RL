"""Align ATOM prefill and decode attention through AITER varlen attention.

ATOM_FORCE_ATTN_TRITON=1 sends decode through Triton RoPE and gluon paged
attention. Prefill, once its RoPE is kept on aiter, uses flash_attn_varlen.
This patch makes decode do the same two things: aiter RoPE plus a gather of
the paged cache into the varlen kernel. It does not run the assembly paged
kernel. Enable with LUMENRL_ALIGN_DECODE=1.

Full-sequence causal varlen and the q_len=1 varlen call disagree on some
prompt lengths (the 354-token step-30 prompt differs by about 0.06–0.18 on the
first response token; a 117-token prompt matches bitwise). Training scores
that token with q_len=1, so alignment keeps prefill on the same call.
``LUMENRL_ALIGN_DECODE=1`` enables it. ``LUMENRL_PREFILL_DECODE_ATTN=1`` enables
it on its own. The prefill path is one varlen launch per query row.
"""

import os
import importlib.machinery
import sys

import torch

# One engine step, published from ModelRunner.prepare_inputs. Layers share it.
# req_ids, new token counts, context length after this step, row offset of
# each sequence in k_out. None while no real batch is active.
_step = None
# Requests whose KV was freed. The scheduler lives in another process, so it
# stamps this set onto the next batch and each worker drops its own copy.
_retired: set[int] = set()
_retired_seen = False
_capacity = 8192

# LUMENRL_ALIGN_BATCHED_DECODE=1: each layer keeps its K/V history in one pool
# of fixed-size slots, so decode attends every sequence in one
# row_prefix_attention launch. Slot numbers and stored lengths are shared by
# all layers; _plan holds this step's indices, built once in prepare_inputs.
_slot_of: dict[int, int] = {}
_slot_len: dict[int, int] = {}
_free_slots: list[int] = []
_num_slots = 0
_plan = None
_check = [0, 0]
# ((step, cu_seqlens_q, cu_seqlens_k, keys), plan) of the current prefill, shared by its layers.
_prefill_plan = None


def _configured_capacity(config) -> int:
    env = int(os.environ.get("LUMENRL_ALIGN_CONTIG_KV_CAPACITY", "0") or 0)
    model = int(getattr(config, "max_model_len", 0) or 0)
    return max(env, model, 1)


def _step_spans(new_tokens, ends):
    """(row_start, row_end, hist_start, hist_end) for each sequence.

    ``ends`` is the context length after this step. The new rows are the tail,
    so a later prefill chunk writes after the rows already stored and a longer
    prompt does not need another code change.
    """
    spans = []
    row = 0
    for new, end in zip(new_tokens, ends):
        new = int(new)
        end = int(end)
        spans.append((row, row + new, end - new, end))
        row += new
    return spans


def install():
    """Patch ATOM attention when this directory is injected into PYTHONPATH."""
    if getattr(sys, "_lumenrl_align_attention_installed", False):
        return
    sys._lumenrl_align_attention_installed = True

    class _Finder:
        def find_spec(self, fullname, path, target=None):
            if fullname != "atom.model_ops.attention_mha":
                return None
            sys.meta_path.remove(self)
            try:
                spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            finally:
                sys.meta_path.insert(0, self)
            if spec is None or spec.loader is None:
                return spec
            orig_exec = spec.loader.exec_module

            def exec_module(module, _orig=orig_exec):
                _orig(module)
                patch_attention(module)

            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _Finder())
    _install_logprob_snapshot()
    _install_o_quant_linear()


def _install_logprob_snapshot():
    """Materialize a sampled logprob on the host before deferred D2H.

    ``tokenIDProcessor`` queues ``gpu_logprobs.to("cpu", non_blocking=True)``
    on a side stream and reads it one step later. The source is a one-element
    gather. Nothing tells the caching allocator about the side-stream read, so
    after a few hundred decode steps the next forward reuses the block before
    the copy runs, and the copy observes that recycled float.
    ``record_stream`` keeps the block until the side stream's queued work is
    done, so the copy stays asynchronous. Enable with ``LUMENRL_ALIGN_DECODE=1``.
    """
    busy = False

    class _Finder:
        def find_spec(self, fullname, path, target=None):
            nonlocal busy
            if busy or fullname != "atom.model_engine.model_runner":
                return None
            busy = True
            try:
                spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            finally:
                busy = False
            if spec is None or spec.loader is None:
                return spec
            orig_exec = spec.loader.exec_module

            def exec_module(module, _orig=orig_exec):
                _orig(module)
                processor = getattr(module, "tokenIDProcessor", None)
                if processor is None or getattr(processor, "_lumenrl_logprob_snapshot", False):
                    return
                orig_init = processor.__init__
                orig_send = processor.send_to_cpu_async

                def __init__(self, *args, **kwargs):
                    orig_init(self, *args, **kwargs)
                    if os.environ.get("LUMENRL_ALIGN_SYNC_OUTPUT", "0") == "1":
                        # Diagnostic alignment path: return this step's token id
                        # and logprob together instead of overwriting the prior
                        # step's scheduler placeholder. This removes deferred
                        # queueing and its one-step state from the comparison.
                        self.is_deferred_out = False

                def send_to_cpu_async(
                    self, gpu_tensor, cpu_tensor_handle, data_ready,
                    copy_done=None, gpu_logprobs=None,
                ):
                    if gpu_logprobs is not None and _flag("LUMENRL_ALIGN_DECODE"):
                        gpu_logprobs.record_stream(self.async_copy_stream)
                    return orig_send(
                        self, gpu_tensor, cpu_tensor_handle, data_ready,
                        copy_done, gpu_logprobs,
                    )

                processor.__init__ = __init__
                processor.send_to_cpu_async = send_to_cpu_async
                processor._lumenrl_logprob_snapshot = True
                print("aligned decode snapshots sampled logprobs", flush=True)
                if os.environ.get("LUMENRL_ALIGN_SYNC_OUTPUT", "0") == "1":
                    print("aligned decode disables deferred output", flush=True)
                _patch_runner_kv_plan(module)

            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _Finder())
    _install_scheduler_retirement()


def _aiter_rope(prefill, align):
    # Read before rope_cache first flips the variable, so this is the launch value.
    if not _flag("ATOM_FORCE_ATTN_TRITON"):
        return False
    return prefill or align


def _history_blocked(fwd_ctx) -> bool:
    """A microbatch sees only part of the step, so its rows are not the plan."""
    return getattr(fwd_ctx, "ubatch_slices", None) is not None


def _publish_step(runner, batch):
    """Record this forward's per-request row layout. Dummy steps clear it."""
    global _step, _capacity
    _capacity = _configured_capacity(getattr(runner, "config", None))
    if os.environ.get("LUMENRL_ALIGN_CONTIG_KV", "0") == "1":
        for req in getattr(batch, "lumenrl_retired", ()) or ():
            _retired.add(int(req))
    if batch is None or getattr(batch, "is_dummy_run", False):
        _step = None
        return
    n = int(getattr(batch, "total_seqs_num", 0) or 0)
    req_ids = tuple(int(x) for x in list(batch.req_ids)[:n])
    new_tokens = tuple(int(x) for x in list(batch.num_scheduled_tokens)[:n])
    ends = tuple(int(x) for x in list(batch.context_lens)[:n])
    if len(req_ids) != n or len(new_tokens) != n or len(ends) != n:
        _step = None
        return
    if any(end < new for end, new in zip(ends, new_tokens)):
        _step = None
        return
    _step = (req_ids, new_tokens, ends, _step_spans(new_tokens, ends))


_flags: dict[str, bool] = {}


def _flag(name: str) -> bool:
    """A LUMENRL_* switch. They are set before the worker starts, so read once."""
    value = _flags.get(name)
    if value is None:
        value = os.environ.get(name, "0") == "1"
        _flags[name] = value
    return value


def _batched() -> bool:
    return (
        _flag("LUMENRL_ALIGN_BATCHED_DECODE")
        and _flag("LUMENRL_ALIGN_CONTIG_KV")
        and _flag("LUMENRL_ALIGN_DECODE")
    )


def _plan_step():
    """Assign pool slots and build this step's write and attention indices."""
    global _plan, _num_slots
    _plan = None
    for req in _retired:
        _paged_write_skipped.discard(req)
        slot = _slot_of.pop(req, None)
        _slot_len.pop(req, None)
        if slot is not None:
            _free_slots.append(slot)
    if _step is None:
        return
    req_ids, _new, ends, spans = _step
    cap = _capacity
    if max(ends, default=0) > cap:
        raise RuntimeError(f"batched decode: context {max(ends)} exceeds slot capacity {cap}")
    slots = []
    for req in req_ids:
        slot = _slot_of.get(req)
        if slot is None:
            if _free_slots:
                slot = _free_slots.pop()
            else:
                slot = _num_slots
                _num_slots += 1
            _slot_of[req] = slot
            _slot_len[req] = 0
        slots.append(slot)

    write_src, write_dst, after = [], [], []
    for req, slot, (row0, row1, start, end) in zip(req_ids, slots, spans):
        length = _slot_len[req]
        # A hole means earlier tokens came from the prefix cache. Leave the
        # length; the next decode gathers the paged cache into the slot.
        if row1 > row0 and 0 <= start <= length:
            write_src.extend(range(row0, row1))
            write_dst.extend(range(slot * cap + start, slot * cap + end))
            after.append(end)
        else:
            after.append(length)

    # Rows sorted by slot keep cu_seqlens_k_padded non-decreasing, the same
    # layout training uses, where all rows of a sequence share one start.
    perm, starts, lens = [], [], []
    for i in sorted(range(len(req_ids)), key=lambda i: slots[i]):
        row0, row1, start, _end = spans[i]
        for j in range(row0, row1):
            perm.append(j)
            starts.append(slots[i] * cap)
            lens.append(start + (j - row0) + 1)

    rows = spans[-1][1] if spans else 0
    _plan = {
        "cap": cap,
        "slots": slots,
        "after": after,
        "complete": all(a >= int(e) for a, e in zip(after, ends)),
        "rows": rows,
        "write_src": None if write_src == list(range(rows)) else write_src,
        "write_dst": write_dst,
        "perm": None if perm == list(range(rows)) else perm,
        "starts": starts,
        "lens": lens,
        "max_len": max(lens, default=0),
        "blocked": False,
        "absorbed": False,
        "dev": None,
        "attn": None,
        "split": None,
    }


def _finish_step():
    """Record each slot's stored length once every layer has written it."""
    global _plan
    plan = _plan
    _plan = None
    if plan is None or plan["blocked"] or _step is None:
        return
    req_ids, _new, ends, _spans = _step
    stored = ends if plan["absorbed"] else plan["after"]
    for req, length in zip(req_ids, stored):
        if req in _slot_of:
            _slot_len[req] = int(length)


def _to_device(parts, dtype, device):
    """Upload host lists as one pinned buffer; return a device view per part.

    The caching host allocator keeps a pinned block until its copy has run,
    so the copy is asynchronous and the host does not wait for the GPU queue.
    """
    flat = [x for part in parts if part is not None for x in part]
    if not flat:
        return [None if part is None else torch.empty(0, dtype=dtype, device=device) for part in parts]
    host = torch.tensor(flat, dtype=dtype, pin_memory=True)
    buf = host.to(device, non_blocking=True)
    views = []
    offset = 0
    for part in parts:
        if part is None:
            views.append(None)
            continue
        views.append(buf[offset : offset + len(part)])
        offset += len(part)
    return views


def _plan_tensors(device):
    dev = _plan["dev"]
    if dev is None:
        write_src, write_dst, perm = _to_device(
            [_plan["write_src"], _plan["write_dst"], _plan["perm"]], torch.long, device
        )
        dev = {"write_src": write_src, "write_dst": write_dst, "perm": perm}
        _plan["dev"] = dev
    return dev


def _absorb_index(device):
    cap = _plan["cap"]
    dst = []
    for slot, end in zip(_plan["slots"], _step[2]):
        dst.extend(range(slot * cap, slot * cap + int(end)))
    (index,) = _to_device([dst], torch.long, device)
    return index


def _attn_tensors(device, rep, pool_rows, hq):
    """cu_seqlens_q, cu_seqlens_k, padded K starts and padded Q starts of this step's rows.

    The same values ``row_prefix_attention`` computes on every call, built on
    the host once per step instead of with five device ops in every layer.
    The Q starts place entry ``e`` at its query row (``grouped_padding``).
    """
    from qlen1_attention import grouped_padding

    attn = _plan["attn"]
    if attn is not None and attn[0] == (rep, pool_rows, hq):
        return attn[1]
    rows = len(_plan["lens"])
    cu_q = list(range(0, rep * rows + 1, rep))
    cu_k = [0]
    for n in _plan["lens"]:
        cu_k.append(cu_k[-1] + int(n))
    padded = list(_plan["starts"]) + [int(pool_rows)]
    src = _plan["perm"] if _plan["perm"] is not None else range(rows)
    tensors = _to_device([cu_q, cu_k, padded, grouped_padding(src, hq, rep)], torch.int32, device)
    _plan["attn"] = ((rep, pool_rows, hq), tensors)
    return tensors


def _pool_slots() -> int:
    return int(os.environ.get("LUMENRL_ALIGN_POOL_SLOTS", "8") or 8)


def _pool_cap() -> int:
    """Cap the indices were built with. A planless capture uses the engine config."""
    if _plan is not None:
        return int(_plan["cap"])
    return _cap()


def _pool_mismatch(pool, like) -> bool:
    return pool[0].shape[1:] != like.shape[1:] or pool[0].dtype != like.dtype


def _pool(module, like):
    """This layer's K/V pool.

    Static decode captures a CUDA graph on the pool's address, so that path
    allocates ``_pool_slots()`` sequence slots plus one scratch slot once and
    never moves it. The eager path grows, as it did before static decode.
    """
    if _flag("LUMENRL_ALIGN_STATIC_DECODE"):
        return _pool_fixed(module, like)
    return _pool_grow(module, like)


def _pool_fixed(module, like):
    cap = _pool_cap()
    slots = _pool_slots()
    if _num_slots > slots:
        raise RuntimeError(
            f"batched decode: {_num_slots} concurrent sequences exceed "
            f"LUMENRL_ALIGN_POOL_SLOTS={slots}"
        )
    need = (slots + 1) * cap
    pool = getattr(module, "_lumenrl_pool", None)
    if pool is not None:
        if _pool_mismatch(pool, like) or pool[0].shape[0] != need:
            raise RuntimeError(
                f"batched decode: K/V {tuple(like.shape)} {like.dtype} does not match "
                f"fixed pool {tuple(pool[0].shape)} {pool[0].dtype} for cap={cap} slots={slots}"
            )
        return pool
    k = torch.zeros((need, *like.shape[1:]), dtype=like.dtype, device=like.device)
    v = torch.zeros_like(k)
    module._lumenrl_pool = (k, v)
    return module._lumenrl_pool


def _pool_grow(module, like):
    cap = _pool_cap()
    pool = getattr(module, "_lumenrl_pool", None)
    if pool is not None and _pool_mismatch(pool, like):
        raise RuntimeError(
            f"batched decode: K/V {tuple(like.shape[1:])} {like.dtype} does not match "
            f"pool {tuple(pool[0].shape[1:])} {pool[0].dtype}"
        )
    if pool is not None and pool[0].shape[0] >= _num_slots * cap:
        return pool
    slots = (max(_num_slots, 1) + 7) // 8 * 8
    k = torch.empty((slots * cap, *like.shape[1:]), dtype=like.dtype, device=like.device)
    v = torch.empty_like(k)
    if pool is not None:
        k[: pool[0].shape[0]].copy_(pool[0])
        v[: pool[1].shape[0]].copy_(pool[1])
    module._lumenrl_pool = (k, v)
    return module._lumenrl_pool


# Index buffers of decode_static_attention, at fixed addresses for CUDA graphs.
_static = None


def _cap() -> int:
    """Slot capacity from the engine config, so capture and later steps agree."""
    global _capacity
    try:
        from atom.config import get_current_atom_config

        _capacity = _configured_capacity(get_current_atom_config())
    except Exception:
        pass
    return _capacity


def _static_rows() -> int:
    """Largest decode batch: every captured graph size is at most max_num_seqs."""
    try:
        from atom.config import get_current_atom_config

        return int(get_current_atom_config().max_num_seqs)
    except Exception:
        return int(os.environ.get("LUMENRL_ALIGN_STATIC_ROWS", "64"))


def _static_buffers(device, rep):
    global _static
    cap = _pool_cap()
    slots = _pool_slots()
    if _static is not None:
        same = (
            _static["rep"] == rep
            and _static["cap"] == cap
            and _static["slots"] == slots
            and _static["gather"].device == device
        )
        if not same:
            raise RuntimeError(
                f"batched decode: static buffers are rep={_static['rep']} cap={_static['cap']} "
                f"slots={_static['slots']} device={_static['gather'].device}, "
                f"this step is rep={rep} cap={cap} slots={slots} device={device}"
            )
        return _static
    qa = _qlen1()
    rows = _static_rows()
    width = -(-cap // qa.SPLIT_KEYS)
    entries = rows * width
    if entries + 1 + rows > cap:
        raise RuntimeError(f"batched decode: scratch slot of {cap} rows cannot hold {entries} entries")
    i32 = dict(dtype=torch.int32, device=device)
    _static = {
        "rows": rows,
        "width": width,
        "rep": rep,
        "cap": cap,
        "slots": slots,
        "scratch": slots * cap,
        "write_dst": torch.zeros(rows, dtype=torch.long, device=device),
        "gather": torch.zeros(entries, dtype=torch.long, device=device),
        "cu_q": torch.arange(0, rep * (entries + 1), rep, **i32),
        "cu_k": torch.zeros(entries + 1, **i32),
        "k_start": torch.zeros(entries + 1, **i32),
        "idx": torch.zeros((rows, width), **i32),
        "npart": torch.zeros(rows, **i32),
        "dst": torch.arange(rows, **i32),
    }
    _fill_static(None)
    return _static


def _static_lists(row_starts, row_lens, order, rows, width, scratch, split, rep):
    """Host contents of the decode_static_attention buffers.

    ``row_starts[i]``/``row_lens[i]`` are pool start and key count of real
    row ``i``; ``order`` lists the real rows by pool start, so entry starts
    stay non-decreasing. Rows past the real ones and entries past the real
    ones read scratch rows. Every padded row merges the first padded entry;
    the entries after it get no query rows and gather ``-1``, so attention
    and the query copy skip them.
    """
    n = len(row_lens)
    entries = rows * width
    write_dst = [row_starts[i] + row_lens[i] - 1 for i in range(n)] + [scratch + i for i in range(n, rows)]
    gather, k_start, k_len = [], [], []
    idx = [[0] * width for _ in range(rows)]
    npart = [1] * rows
    for i in order:
        parts = 0
        for p in range(0, row_lens[i], split):
            idx[i][parts] = len(gather)
            parts += 1
            gather.append(i)
            k_start.append(row_starts[i] + p)
            k_len.append(min(split, row_lens[i] - p))
        if parts > width:
            raise RuntimeError(f"batched decode: row of {row_lens[i]} keys needs {parts} > {width} blocks")
        npart[i] = parts
    real = len(gather)
    for i in range(n, rows):
        idx[i][0] = real
    for e in range(real, entries):
        gather.append(0 if e == real else -1)
        k_start.append(scratch + (e - real))
        k_len.append(1)
    k_start.append(scratch + (entries - real))
    cu_k = [0]
    for x in k_len:
        cu_k.append(cu_k[-1] + x)
    cu_q = [min(e, real + 1) * rep for e in range(entries + 1)]
    flat_idx = [x for row in idx for x in row]
    return write_dst, gather, cu_k, k_start, flat_idx, npart, cu_q


def _fill_static(plan):
    """Upload this step's decode indices (or all-padding) into ``_static``."""
    st = _static
    if st is None:
        return
    if plan is None:
        starts, lens, order = [], [], []
    else:
        lens = [int(x) for x in plan["lens"]]
        starts = [int(x) for x in plan["starts"]]
        perm = plan["perm"] if plan["perm"] is not None else list(range(len(lens)))
        # plan["starts"]/["lens"] are listed in perm order; map them back to rows.
        row_starts = [0] * len(lens)
        row_lens = [0] * len(lens)
        for pos, row in enumerate(perm):
            row_starts[row] = starts[pos]
            row_lens[row] = lens[pos]
        starts, lens = row_starts, row_lens
        order = list(perm)
    write_dst, gather, cu_k, k_start, idx, npart, cu_q = _static_lists(
        starts, lens, order, st["rows"], st["width"], st["scratch"], _qlen1().SPLIT_KEYS, st["rep"]
    )
    device = st["gather"].device
    # dst never changes, but is uploaded too: sleep mode may drop device
    # memory between steps.
    longs = _to_device([write_dst, gather], torch.long, device)
    ints = _to_device([cu_k, k_start, idx, npart, cu_q, list(range(st["rows"]))], torch.int32, device)
    st["write_dst"].copy_(longs[0], non_blocking=True)
    st["gather"].copy_(longs[1], non_blocking=True)
    st["cu_k"].copy_(ints[0], non_blocking=True)
    st["k_start"].copy_(ints[1], non_blocking=True)
    st["idx"].view(-1).copy_(ints[2], non_blocking=True)
    st["npart"].copy_(ints[3], non_blocking=True)
    st["cu_q"].copy_(ints[4], non_blocking=True)
    st["dst"].copy_(ints[5], non_blocking=True)
    if plan is not None:
        plan["static"] = True


def _static_step(plan) -> bool:
    """A decode step that decode_static_attention can run: one new row per sequence."""
    return (
        plan is not None
        and plan["complete"]
        and plan["rows"] == len(plan["slots"])
        and plan["rows"] <= (_static["rows"] if _static is not None else _static_rows())
    )


_qa = None


def _static_decode(impl, fwd_ctx, rows) -> bool:
    """Decode through decode_static_attention, the CUDA-graph-capturable path.

    Without a plan this is a graph capture or a dummy step: every row is padding.
    """
    return (
        _flag("LUMENRL_ALIGN_STATIC_DECODE")
        and not _history_blocked(fwd_ctx)
        and impl.sinks is None
        and impl.sliding_window <= 0
        and rows <= (_static["rows"] if _static is not None else _static_rows())
        and (_plan is None or (_static_step(_plan) and rows >= _plan["rows"]))
    )


def _qlen1():
    global _qa
    if _qa is None:
        import qlen1_attention

        _qa = qlen1_attention
    return _qa


def _pool_write(module, k_out, v_out, defer=False):
    """Write this step's new K/V rows into the pool.

    With ``defer`` a launchable write is left in ``module._lumenrl_pending``
    for this layer's attention, which runs it in the same launch as its own
    query copy (``_flush_pending`` runs it on its own otherwise).
    """
    _flush_pending(module)
    if not _plan["write_dst"]:
        return
    if k_out.shape[0] < _plan["rows"]:
        _plan["blocked"] = True
        return
    k_pool, v_pool = _pool(module, k_out)
    dev = _plan_tensors(k_out.device)
    if dev["write_src"] is None:
        k_new = k_out[: _plan["rows"]]
        v_new = v_out[: _plan["rows"]]
    else:
        k_new = k_out.index_select(0, dev["write_src"])
        v_new = v_out.index_select(0, dev["write_src"])
    dim = k_new.shape[2]
    if k_new.stride()[1:] == (dim, 1) and v_new.stride()[1:] == (dim, 1):
        if defer:
            module._lumenrl_pending = (k_new, v_new, dev["write_dst"])
            return
        _qlen1().pool_write(k_new, v_new, dev["write_dst"], k_pool, v_pool)
        return
    k_pool.index_copy_(0, dev["write_dst"], k_new)
    v_pool.index_copy_(0, dev["write_dst"], v_new)


def _take_pending(module):
    pending = module.__dict__.get("_lumenrl_pending")
    if pending is not None:
        module._lumenrl_pending = None
    return pending


def _flush_pending(module):
    pending = _take_pending(module)
    if pending is not None:
        k_pool, v_pool = module._lumenrl_pool
        _qlen1().pool_write(pending[0], pending[1], pending[2], k_pool, v_pool)


def _gather_step_kv(impl, q, k, v, k_cache, v_cache, k_scale, v_scale, meta):
    """Gather each sequence's paged K/V, back to back, plus host lengths."""
    if _step is not None and _paged_write_skipped.intersection(_step[0]):
        raise RuntimeError(
            "aligned decode: static decode skipped the paged K/V write, "
            "but this step gathers K/V from the paged cache"
        )
    lengths = meta.context_lens.to(dtype=torch.int32)
    nseq = int(lengths.shape[0])
    cu_k = torch.empty(nseq + 1, dtype=torch.int32, device=q.device)
    cu_k[0] = 0
    torch.cumsum(lengths, dim=0, out=cu_k[1:])
    total = int(cu_k[-1].item())
    if meta.cu_seqlens_q is not None and int(meta.cu_seqlens_q.shape[0]) == nseq + 1:
        cu_q = meta.cu_seqlens_q
    else:
        cu_q = torch.arange(nseq + 1, device=q.device, dtype=torch.int32)
    batch_id = torch.repeat_interleave(
        torch.arange(nseq, device=q.device, dtype=torch.int32),
        lengths,
    )
    seq_starts = torch.zeros(nseq, dtype=torch.int32, device=q.device)
    gather_meta = type("GatherMeta", (), {})()
    gather_meta.cu_seqlens_k = cu_k
    gather_meta.total_kv = total
    gather_meta.block_tables = meta.block_tables
    gather_meta.batch_id_per_k_token = batch_id
    gather_meta.seq_starts = seq_starts
    _, k_full, v_full, _, _, _, _ = impl._gather_prefix_and_concat_kv(
        q, k, v, k_cache, v_cache, k_scale, v_scale, gather_meta
    )
    ends = [int(x) for x in cu_k.diff().tolist()]
    q_bounds = [int(x) for x in cu_q.tolist()]
    new_tokens = [q_bounds[i + 1] - q_bounds[i] for i in range(nseq)]
    return k_full, v_full, ends, new_tokens


def _split_plan(device, rep, pool_rows):
    """``decode_split_plan`` of this step's rows, built once and shared by all layers."""
    split = _plan["split"]
    if split is not None and split[0] == (rep, pool_rows):
        return split[1]
    src = _plan["perm"] if _plan["perm"] is not None else range(len(_plan["lens"]))
    plan = _qlen1().decode_split_plan(src, _plan["starts"], _plan["lens"], pool_rows, rep, device)
    _plan["split"] = ((rep, pool_rows), plan)
    return plan


def _pool_attention(module, q):
    qa = _qlen1()
    pool = module.__dict__.get("_lumenrl_pool")
    if pool is None or pool[0].shape[0] < (max(_plan["slots"]) + 1) * _plan["cap"]:
        raise RuntimeError("batched decode: attention before this step's K/V reached the pool")
    k_pool, v_pool = pool
    hq, dim = q.shape[1], q.shape[2]
    rep = hq // k_pool.shape[1]
    split = _split_plan(q.device, rep, k_pool.shape[0])
    if split is not None:
        pending = _take_pending(module)
        if pending is not None and not (q.stride(2) == 1 and q.stride(1) == dim):
            qa.pool_write(pending[0], pending[1], pending[2], k_pool, v_pool)
            pending = None
        return qa.decode_split_attention(q, k_pool, v_pool, split, module.scale, pending)
    _flush_pending(module)
    cu_q, cu_k, padded, q_padded = _attn_tensors(q.device, rep, k_pool.shape[0], hq)
    if q.is_contiguous():
        return qa.row_prefix_attention_inplace(
            q, k_pool, v_pool, cu_q, q_padded, cu_k, padded, _plan["max_len"], module.scale
        )
    dev = _plan_tensors(q.device)
    rows = q if dev["perm"] is None else q.index_select(0, dev["perm"])
    out = qa.row_prefix_attention_planned(
        rows, k_pool, v_pool, cu_q, cu_k, padded, _plan["max_len"], module.scale
    )
    if dev["perm"] is None:
        return out
    result = torch.empty_like(out)
    result.index_copy_(0, dev["perm"], out)
    return result


def _patch_runner_kv_plan(module):
    runner = getattr(module, "ModelRunner", None)
    if runner is None or getattr(runner.prepare_inputs, "_lumenrl_kv_plan", False):
        return
    orig_prepare = runner.prepare_inputs
    orig_run = runner.run_model

    def prepare_inputs(self, batch, *args, **kwargs):
        _publish_step(self, batch)
        if _batched():
            _plan_step()
            # A replayed graph runs no Python, so its indices are uploaded here.
            if _static is not None and _flag("LUMENRL_ALIGN_STATIC_DECODE"):
                if _plan is None:
                    _fill_static(None)
                elif _static_step(_plan):
                    _fill_static(_plan)
        return orig_prepare(self, batch, *args, **kwargs)

    def run_model(self, *args, **kwargs):
        global _retired_seen
        _retired_seen = False
        try:
            return orig_run(self, *args, **kwargs)
        finally:
            if _batched():
                _finish_step()
            # Clear only after a layer has copied the ids. A dummy forward that
            # never reaches attention must leave them for the next real one.
            if _retired_seen:
                _retired.clear()

    prepare_inputs._lumenrl_kv_plan = True
    runner.prepare_inputs = prepare_inputs
    runner.run_model = run_model


def _install_scheduler_retirement():
    """Drop a request's history when the scheduler frees or preempts it."""
    busy = False

    class _Finder:
        def find_spec(self, fullname, path, target=None):
            nonlocal busy
            if busy or fullname != "atom.model_engine.scheduler":
                return None
            busy = True
            try:
                spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            finally:
                busy = False
            if spec is None or spec.loader is None:
                return spec
            orig_exec = spec.loader.exec_module

            def exec_module(module, _orig=orig_exec):
                _orig(module)
                sched = getattr(module, "Scheduler", None)
                if sched is None or getattr(sched.postprocess, "_lumenrl_retire", False):
                    return
                orig_post = sched.postprocess
                orig_preempt = sched.preempt
                orig_schedule = sched.schedule

                def _contig():
                    return os.environ.get("LUMENRL_ALIGN_CONTIG_KV", "0") == "1"

                def postprocess(self, seqs, fwd_output, *args, **kwargs):
                    finished = orig_post(self, seqs, fwd_output, *args, **kwargs)
                    if _contig():
                        for seq in finished or ():
                            _retired.add(int(seq.id))
                    return finished

                def preempt(self, seq, *args, **kwargs):
                    ok = orig_preempt(self, seq, *args, **kwargs)
                    if ok and _contig():
                        _retired.add(int(seq.id))
                    return ok

                def schedule(self, *args, **kwargs):
                    result = orig_schedule(self, *args, **kwargs)
                    batch = result[0] if isinstance(result, tuple) and result else None
                    if batch is not None and _retired and _contig():
                        # Workers pickle this with the batch. Keep the whole set:
                        # an idle tick may build a batch that never reaches them.
                        batch.lumenrl_retired = tuple(_retired)
                    return result

                postprocess._lumenrl_retire = True
                sched.postprocess = postprocess
                sched.preempt = preempt
                sched.schedule = schedule

            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _Finder())


def _drop_retired(module):
    global _retired_seen
    _retired_seen = True
    hist = getattr(module, "_lumenrl_kv", None)
    if not hist or not _retired:
        return
    for req in _retired:
        hist.pop(req, None)


def _ensure_history(module, req, need, like):
    hist = getattr(module, "_lumenrl_kv", None)
    if hist is None:
        hist = {}
        module._lumenrl_kv = hist
    slot = hist.get(req)
    if slot is not None and slot[0].shape[0] >= need and slot[0].shape[1:] == like.shape[1:]:
        return slot
    cap = max(need, _capacity)
    k = torch.empty((cap, *like.shape[1:]), dtype=like.dtype, device=like.device)
    v = torch.empty((cap, *like.shape[1:]), dtype=like.dtype, device=like.device)
    length = 0
    if slot is not None and slot[0].shape[1:] == like.shape[1:]:
        length = min(int(slot[2]), slot[0].shape[0], cap)
        k[:length].copy_(slot[0][:length])
        v[:length].copy_(slot[1][:length])
    slot = [k, v, length]
    hist[req] = slot
    return slot


def _write_history(module, k_out, v_out):
    """Store this step's new K/V at its absolute position in each request."""
    if os.environ.get("LUMENRL_ALIGN_CONTIG_KV", "0") != "1" or _step is None:
        return
    _drop_retired(module)
    req_ids, _new, _ends, spans = _step
    if not spans or k_out.shape[0] < spans[-1][1]:
        return
    for req, (row0, row1, start, end) in zip(req_ids, spans):
        if row1 <= row0 or start < 0:
            continue
        slot = _ensure_history(module, req, end, k_out)
        # A hole means tokens were cached before this history existed. Leave
        # the contiguous length where it is; attention gathers once and fills.
        if start > slot[2]:
            continue
        slot[0][start:end].copy_(k_out[row0:row1])
        slot[1][start:end].copy_(v_out[row0:row1])
        slot[2] = end


def _histories_for_step(module, q):
    """Per-request K/V when every sequence's history covers this step."""
    if os.environ.get("LUMENRL_ALIGN_CONTIG_KV", "0") != "1" or _step is None:
        return None
    req_ids, new_tokens, ends, spans = _step
    if not spans or q.shape[0] != spans[-1][1]:
        return None
    hist = getattr(module, "_lumenrl_kv", None)
    if not hist:
        return None
    slots = []
    for req, end in zip(req_ids, ends):
        slot = hist.get(req)
        if slot is None or int(slot[2]) < int(end) or int(end) < 0:
            return None
        slots.append(slot)
    return slots


def _pack_histories(module, slots, q):
    req_ids, new_tokens, ends, _spans = _step
    if len(slots) == 1:
        end = int(ends[0])
        k_full = slots[0][0][:end]
        v_full = slots[0][1][:end]
    else:
        k_full = torch.cat([slot[0][: int(end)] for slot, end in zip(slots, ends)], 0)
        v_full = torch.cat([slot[1][: int(end)] for slot, end in zip(slots, ends)], 0)
    device = q.device
    lens_k = torch.tensor(ends, dtype=torch.int32, device=device)
    lens_q = torch.tensor(new_tokens, dtype=torch.int32, device=device)
    cu_k = torch.empty(len(ends) + 1, dtype=torch.int32, device=device)
    cu_q = torch.empty(len(new_tokens) + 1, dtype=torch.int32, device=device)
    cu_k[0] = 0
    cu_q[0] = 0
    torch.cumsum(lens_k, dim=0, out=cu_k[1:])
    torch.cumsum(lens_q, dim=0, out=cu_q[1:])
    return k_full, v_full, cu_q, cu_k, int(max(new_tokens)), int(max(ends))


def _absorb_gathered(module, k_full, v_full):
    """Fill any hole from the paged gather so the next step stays contiguous."""
    if _step is None or k_full.shape[0] != sum(int(end) for end in _step[2]):
        return
    req_ids, _new, ends, _spans = _step
    offset = 0
    for req, end in zip(req_ids, ends):
        end = int(end)
        if end > 0:
            slot = _ensure_history(module, req, end, k_full)
            slot[0][:end].copy_(k_full[offset : offset + end])
            slot[1][:end].copy_(v_full[offset : offset + end])
            slot[2] = end
        offset += end


# Requests whose decode tokens were written to the pool and not the paged
# cache. Gathering one of them would read a stale cache. Other requests are
# unaffected, and a request leaves the set when the scheduler frees it.
# A dummy or graph-capture step has no request ids and must not mark anyone.
_paged_write_skipped: set[int] = set()
# (data_ptr, xq, xs) of the last static decode attention output, quantized by
# its merge for the o_proj that reads it next.
_o_quant = None


def _fuse_o_quant() -> bool:
    """Static decode quantizes o_proj's input in the merge; LUMENRL_ALIGN_FUSE_O_QUANT=0 turns it off."""
    value = _flags.get("LUMENRL_ALIGN_FUSE_O_QUANT")
    if value is None:
        value = os.environ.get("LUMENRL_ALIGN_FUSE_O_QUANT", "1") == "1" and _o_quant_linear_ok()
        _flags["LUMENRL_ALIGN_FUSE_O_QUANT"] = value
    return value


def _o_quant_linear_ok() -> bool:
    """ATOM's per_1x128 linear quantizes its input with row-major scales here."""
    try:
        from atom.utils import envs
    except ImportError:
        return False
    return not envs.ATOM_FP8_BLOCKSCALE_WEIGHT_PRESHUFFLE


def _install_o_quant_linear():
    """Let the linear right after a static decode attention take ``_o_quant``.

    It skips its own input quantization only when the input is that attention
    output and the layer would quantize it with aiter's per-1x128 kernel,
    which ``_merge_partials`` reproduces bit for bit. Any other first linear
    drops the stash and runs unchanged.
    """
    busy = False

    def patch(module):
        base = getattr(module, "LinearBase", None)
        if base is None or getattr(base.forward, "_lumenrl_o_quant", False):
            return
        orig = base.forward
        per_1x128 = module.QuantType.per_1x128.value

        def forward(self, x, x_scale=None, otype=module.dtypes.bf16, out=None):
            global _o_quant
            pre = _o_quant
            if pre is not None:
                _o_quant = None
                if (
                    x_scale is None
                    and x.data_ptr() == pre[0]
                    and x.dtype == torch.bfloat16
                    and x.is_contiguous()
                    and tuple(x.shape) == tuple(pre[1].shape)
                    and self.native_a8_group_rows is None
                    and self.quant_type.value == per_1x128
                    and not self.blockscale_e8m0_scale
                    and self.params_dtype == pre[1].dtype
                    and getattr(self, "input_scale", None) is None
                ):
                    return orig(self, pre[1], x_scale=pre[2], otype=otype, out=out)
            return orig(self, x, x_scale, otype, out)

        forward._lumenrl_o_quant = True
        base.forward = forward

    class _Finder:
        def find_spec(self, fullname, path, target=None):
            nonlocal busy
            if busy or fullname != "atom.model_ops.linear":
                return None
            busy = True
            try:
                spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            finally:
                busy = False
            if spec is None or spec.loader is None:
                return spec
            orig_exec = spec.loader.exec_module

            def exec_module(module, _orig=orig_exec):
                _orig(module)
                patch(module)

            spec.loader.exec_module = exec_module
            return spec

    if "atom.model_ops.linear" in sys.modules:
        patch(sys.modules["atom.model_ops.linear"])
    else:
        sys.meta_path.insert(0, _Finder())


def _rope_without_cache(impl, q, k, v, position, fwd_ctx):
    """ATOM's aiter-RoPE ``rope_cache`` branch without its paged K/V write.

    Static decode attends over the pool alone, so nothing reads that write.
    When ``_deferred_rope`` allows it, q and k come back unrotated and
    ``impl._lumenrl_rope`` holds what decode_static_attention needs to rotate
    them in its pool write. ``None`` when this layer would take another ATOM
    branch (fused QK norm, Triton attention or no rotary), which then runs
    unchanged.
    """
    if (
        impl.rotary_emb is None
        or impl.q_norm is not None
        or impl.k_norm is not None
        or impl.sliding_window != -1
        or impl.head_dim != 128
        or position is None
    ):
        return None
    layer = fwd_ctx.kv_cache_data[f"layer_{impl.layer_num}"]
    impl.use_triton_attn = False
    asm_layout = impl._use_asm_cache_layout(layer.k_cache, layer.v_cache, use_triton_attn=False)
    rope = _deferred_rope(impl.rotary_emb, q, k, position)
    if rope is None:
        q, k = impl.rotary_emb(position, q, k)
    impl._lumenrl_rope = rope
    impl._cache_format = "SHUFFLE" if asm_layout else "NHD"
    return q, k, v, layer.k_cache, layer.v_cache, layer.k_scale, layer.v_scale


def _deferred_rope(rotary, q, k, position):
    """``(positions, cos, sin)`` for decode_static_attention to rotate q and k itself.

    Only for a plain neox ``RotaryEmbedding`` over the whole head, ATOM's or
    aiter's on its HIP backend: both end in
    ``aiter.rope_cached_positions_2c_fwd_inplace`` over the per-position
    half-width cache. q/k rows need contiguous heads. ``None`` leaves the
    rotation to ATOM.
    """
    import aiter.rotary_embedding as aiter_rope
    from atom.model_ops.rotary_embedding import RotaryEmbedding

    dim = q.shape[-1]
    if type(rotary) is aiter_rope.RotaryEmbedding:
        if aiter_rope.AITER_ROPE_TRITON_BACKEND or aiter_rope.AITER_ROPE_NATIVE_BACKEND:
            return None
    elif type(rotary) is RotaryEmbedding:
        rotary.cos_cache = rotary.cos_cache.to(q.device, dtype=q.dtype)
        rotary.sin_cache = rotary.sin_cache.to(q.device, dtype=q.dtype)
    else:
        return None
    if (
        not rotary.is_neox_style
        or rotary.rotary_dim != dim
        or q.dim() != 3
        or k.dim() != 3
        or q.stride()[1:] != (dim, 1)
        or k.stride()[1:] != (dim, 1)
        or position.dim() != 1
        or position.shape[0] < q.shape[0]
    ):
        return None
    cos, sin = rotary.cos_cache, rotary.sin_cache
    if (
        cos.dtype != q.dtype
        or sin.dtype != q.dtype
        or cos.device != q.device
        or cos.shape[-1] * 2 != dim
        or not (cos.is_contiguous() and sin.is_contiguous())
    ):
        return None
    return position, cos, sin


def patch_attention(module):
    impl = module.PagedAttentionImpl
    orig_rope = impl.rope_cache
    orig_dispatch = impl.dispatch_backend

    def rope_cache(self, q, k, v, qkv, position, fwd_ctx):
        prefill = bool(getattr(getattr(fwd_ctx, "context", None), "is_prefill", False))
        align = _flag("LUMENRL_ALIGN_DECODE")
        if not _aiter_rope(prefill, align):
            return orig_rope(self, q, k, v, qkv, position, fwd_ctx)
        static = (
            align
            and not prefill
            and _flag("LUMENRL_ALIGN_CONTIG_KV")
            and _batched()
            and _static_decode(self, fwd_ctx, q.shape[0])
        )
        result = _rope_without_cache(self, q, k, v, position, fwd_ctx) if static else None
        if result is not None:
            if _step is not None:
                _paged_write_skipped.update(_step[0])
        else:
            os.environ["ATOM_FORCE_ATTN_TRITON"] = "0"
            try:
                result = orig_rope(self, q, k, v, qkv, position, fwd_ctx)
            finally:
                os.environ["ATOM_FORCE_ATTN_TRITON"] = "1"
        if align and _flag("LUMENRL_ALIGN_CONTIG_KV"):
            _drop_retired(self)
            if _batched():
                if static:
                    _q_out, k_out, v_out, *_ = result
                    self._lumenrl_static_kv = (k_out, v_out)
                elif _plan is not None:
                    if _history_blocked(fwd_ctx):
                        _plan["blocked"] = True
                    else:
                        _q_out, k_out, v_out, *_ = result
                        _pool_write(self, k_out, v_out, defer=not prefill)
            elif not _history_blocked(fwd_ctx):
                _q_out, k_out, v_out, *_ = result
                _write_history(self, k_out, v_out)
        return result

    def paged_attention_varlen(self, q, k, v, k_cache, v_cache, k_scale, v_scale, fwd_ctx):
        static_kv = self.__dict__.get("_lumenrl_static_kv")
        if static_kv is not None:
            self._lumenrl_static_kv = None
            k_pool, v_pool = _pool(self, static_kv[0])
            st = _static_buffers(q.device, q.shape[1] // k_pool.shape[1])
            if _plan is not None and not _plan.get("static"):
                _fill_static(_plan)
            rope = self.__dict__.get("_lumenrl_rope")
            self._lumenrl_rope = None
            if not _fuse_o_quant():
                return _qlen1().decode_static_attention(
                    q, static_kv[0], static_kv[1], k_pool, v_pool, st, self.scale, rope
                )
            from aiter import dtypes

            global _o_quant
            out, xq, xs = _qlen1().decode_static_attention(
                q, static_kv[0], static_kv[1], k_pool, v_pool, st, self.scale, rope, dtypes.fp8
            )
            _o_quant = (out.data_ptr(), xq, xs)
            return out

        meta = fwd_ctx.attn_metadata
        sliding = (self.sliding_window, 0, 0) if self.sliding_window > 0 else (-1, -1, 0)

        def attend_each(ks, vs, ends, new_tokens):
            """Each sequence's new rows over its own K/V, as training scores them.

            Without sinks or a sliding window this is the chunked causal call
            training and prefill use, with the same split of long rows.
            """
            if self.sinks is None and self.sliding_window <= 0:
                from qlen1_attention import causal_chunk_meta, chunked_causal_attention

                q_bounds, k_bounds = [0], [0]
                for end, new in zip(ends, new_tokens):
                    q_bounds.append(q_bounds[-1] + int(new))
                    k_bounds.append(k_bounds[-1] + int(end))
                k_all = ks[0] if len(ks) == 1 else torch.cat(ks, 0)
                v_all = vs[0] if len(vs) == 1 else torch.cat(vs, 0)
                plan = causal_chunk_meta(q_bounds, k_bounds, k_all.shape[0], q.device)
                return chunked_causal_attention(
                    q[: q_bounds[-1]], k_all, v_all, plan, self.scale
                )
            import aiter

            outs = []
            row = 0
            for k_i, v_i, end, new in zip(ks, vs, ends, new_tokens):
                end = int(end)
                new = int(new)
                cu_q_i = torch.tensor([0, new], dtype=torch.int32, device=q.device)
                cu_k_i = torch.tensor([0, end], dtype=torch.int32, device=q.device)
                res = aiter.flash_attn_varlen_func(
                    q[row : row + new],
                    k_i,
                    v_i,
                    cu_seqlens_q=cu_q_i,
                    cu_seqlens_k=cu_k_i,
                    max_seqlen_q=new,
                    max_seqlen_k=end,
                    min_seqlen_q=0,
                    dropout_p=0.0,
                    softmax_scale=self.scale,
                    causal=True,
                    window_size=sliding,
                    sink_ptr=self.sinks,
                    return_lse=True,
                )
                outs.append(res[0])
                row += new
            return outs[0] if len(outs) == 1 else torch.cat(outs, 0)

        if (
            _batched()
            and _plan is not None
            and not _plan["blocked"]
            and not _history_blocked(fwd_ctx)
            and self.sinks is None
            and self.sliding_window <= 0
            and _plan["rows"] > 0
            and q.shape[0] == _plan["rows"]
        ):
            if not _plan["complete"]:
                _flush_pending(self)
                k_full, v_full, ends, new_tokens = _gather_step_kv(
                    self, q, k, v, k_cache, v_cache, k_scale, v_scale, meta
                )
                if k_full.shape[0] != sum(int(e) for e in _step[2]):
                    _plan["blocked"] = True
                    return attend_each(
                        list(torch.split(k_full, ends, dim=0)),
                        list(torch.split(v_full, ends, dim=0)),
                        ends,
                        new_tokens,
                    )
                k_pool, v_pool = _pool(self, k_full)
                index = _absorb_index(q.device)
                k_pool.index_copy_(0, index, k_full)
                v_pool.index_copy_(0, index, v_full)
                _plan["absorbed"] = True
            out = _pool_attention(self, q)
            if _flag("LUMENRL_ALIGN_BATCHED_CHECK"):
                _req, new_tokens, ends, _spans = _step
                k_pool, v_pool = self._lumenrl_pool
                cap = _plan["cap"]
                bases = [slot * cap for slot in _plan["slots"]]
                ref = attend_each(
                    [k_pool[b : b + int(e)] for b, e in zip(bases, ends)],
                    [v_pool[b : b + int(e)] for b, e in zip(bases, ends)],
                    ends,
                    new_tokens,
                )
                _check[0] += 1
                if not torch.equal(out, ref):
                    _check[1] += 1
                    if _check[1] <= 5:
                        diff = (out.float() - ref.float()).abs().max().item()
                        print(
                            f"batched decode CHECK mismatch rows={q.shape[0]} "
                            f"max_end={max(ends)} max_abs={diff}",
                            flush=True,
                        )
                if _check[0] % 2000 == 0:
                    print(f"batched decode CHECK calls={_check[0]} mismatched={_check[1]}", flush=True)
            return out

        _flush_pending(self)
        slots = None if _history_blocked(fwd_ctx) else _histories_for_step(self, q)
        if slots is not None:
            # Host-side lengths, one history per request. No device assert:
            # a mismatch used to abort the GPU queue when several sequences
            # shared one buffer.
            _req, new_tokens, ends, _spans = _step
            return attend_each(
                [slot[0][: int(end)] for slot, end in zip(slots, ends)],
                [slot[1][: int(end)] for slot, end in zip(slots, ends)],
                ends,
                new_tokens,
            )

        k_full, v_full, ends, new_tokens = _gather_step_kv(
            self, q, k, v, k_cache, v_cache, k_scale, v_scale, meta
        )
        if (
            os.environ.get("LUMENRL_ALIGN_CONTIG_KV", "0") == "1"
            and not _batched()
            and not _history_blocked(fwd_ctx)
        ):
            _absorb_gathered(self, k_full, v_full)
        # Gathered K/V is the sequences back to back. Split it by length and
        # attend to each one on its own, as above. return_lse=True selects the
        # same q_len=1 forward variant the trainer's row-wise forward uses.
        k_parts = list(torch.split(k_full, ends, dim=0))
        v_parts = list(torch.split(v_full, ends, dim=0))
        return attend_each(k_parts, v_parts, ends, new_tokens)

    def prefill_attention_decode_varlen(
        self, q, k, v, k_cache, v_cache, k_scale, v_scale, fwd_ctx
    ):
        """Run prefill with the arithmetic of one q_len=1 varlen call per query row.

        For a prefix-cache hit, ``q`` has only the new rows while the gathered
        K/V include cached and new rows.  The bottom-right causal alignment
        means query row ``j`` attends through ``prefix_len + j``.  Supplying
        exactly that K/V prefix lets q_len=1 use the same no-mask attention
        specialization that decode uses.
        """
        import aiter

        meta = fwd_ctx.attn_metadata
        if meta.has_cached:
            q, k, v, _, _, _, _ = self._gather_prefix_and_concat_kv(
                q, k, v, k_cache, v_cache, k_scale, v_scale, meta
            )

        cu_q = meta.cu_seqlens_q
        cu_k = meta.cu_seqlens_k
        if self.sinks is None and self.sliding_window <= 0:
            from qlen1_attention import causal_chunk_meta, chunked_causal_attention

            global _prefill_plan
            # _step is rebuilt by every prepare_inputs, so it pins the plan to
            # this forward even if ATOM refills the same cu_seqlens buffers.
            key = (_step, cu_q, cu_k, k.shape[0])
            cached = _prefill_plan
            if _step is not None and cached is not None and all(
                a is b for a, b in zip(cached[0][:3], key[:3])
            ) and cached[0][3] == key[3]:
                plan = cached[1]
            else:
                plan = causal_chunk_meta(cu_q.tolist(), cu_k.tolist(), k.shape[0], q.device)
                _prefill_plan = (key, plan)
            return chunked_causal_attention(q, k, v, plan, self.scale)

        out = torch.empty_like(q)
        one = torch.tensor([0, 1], dtype=torch.int32, device=q.device)
        sliding = (self.sliding_window, 0, 0) if self.sliding_window > 0 else (-1, -1, 0)

        for seq in range(int(cu_q.numel()) - 1):
            q_start, q_end = (int(x) for x in cu_q[seq : seq + 2].tolist())
            k_start, k_end = (int(x) for x in cu_k[seq : seq + 2].tolist())
            prefix_len = k_end - k_start - (q_end - q_start)
            for row, q_index in enumerate(range(q_start, q_end)):
                key_end = k_start + prefix_len + row + 1
                result = aiter.flash_attn_varlen_func(
                    q[q_index : q_index + 1],
                    k[k_start:key_end],
                    v[k_start:key_end],
                    cu_seqlens_q=one,
                    cu_seqlens_k=torch.tensor(
                        [0, key_end - k_start], dtype=torch.int32, device=q.device
                    ),
                    max_seqlen_q=1,
                    max_seqlen_k=key_end - k_start,
                    min_seqlen_q=0,
                    dropout_p=0.0,
                    softmax_scale=self.scale,
                    causal=True,
                    window_size=sliding,
                    sink_ptr=self.sinks,
                    return_lse=True,
                )
                out[q_index : q_index + 1] = result[0]
        return out

    def _use_qlen1_prefill():
        return _flag("LUMENRL_ALIGN_DECODE") or _flag("LUMENRL_PREFILL_DECODE_ATTN")

    def dispatch_backend(self, fwd_ctx, q, k, v):
        prefill = bool(getattr(getattr(fwd_ctx, "context", None), "is_prefill", False))
        if prefill:
            if _use_qlen1_prefill():
                return self.prefill_attention_decode_varlen
            return orig_dispatch(self, fwd_ctx, q, k, v)
        if not _flag("LUMENRL_ALIGN_DECODE"):
            return orig_dispatch(self, fwd_ctx, q, k, v)
        return self.paged_attention_varlen

    impl.rope_cache = rope_cache
    impl.paged_attention_varlen = paged_attention_varlen
    impl.prefill_attention_decode_varlen = prefill_attention_decode_varlen
    impl.dispatch_backend = dispatch_backend
    print(
        "aligned attention uses aiter RoPE and q_len=1 flash_attn_varlen "
        "for prefill and decode when LUMENRL_ALIGN_DECODE=1",
        flush=True,
    )
    if os.environ.get("LUMENRL_ALIGN_CONTIG_KV", "0") == "1":
        print("aligned contiguous KV is stored per request", flush=True)
    if _batched():
        print("aligned decode attends all sequences in one row_prefix_attention launch", flush=True)
