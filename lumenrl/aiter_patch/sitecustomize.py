"""Patch aiter TunedGemm.mm for deterministic BF16 GEMM across all M values.

Loaded automatically via PYTHONPATH on Python startup.  Applied to every
process in the ATOM rollout worker hierarchy: the Ray actor and all EngineCore
subprocesses it spawns.

Root cause: aiter's getPaddedM maps M=17-48 and M=81-96 to tuning-table rows
with splitK>1 for shapes like o_proj (N=K=4096).  FlyDSL splitK kernels use
atomic BF16 accumulation; wavefront completion order varies between runs, so
the result is non-deterministic.

Fix: wrap TunedGemm.mm so it calls lumen.ops.gemm.stable_tgemm_mm.call_stable_tgemm_mm
on the unpatched method. That function scans upward from the actual M until
it finds an M whose tuning-table entry has splitK<=1, pads the input with
zero rows, calls the unpatched mm, then slices the original rows back. Each
output row depends only on its own input row and the full weight matrix, so
zero-padding does not change results for the original rows. The wrapper is
marked _lumen_stable_tgemm so Lumen's stable_tgemm_mm does not pad a second time.
If that module cannot be loaded, the scan below is the same algorithm.

The GEMM pad is enabled when LUMENRL_STABLE_GEMM=1. The prefill RoPE hook is
always installed: ATOM workers put this directory on PYTHONPATH, and the hook
itself only changes prefill while ATOM_FORCE_ATTN_TRITON=1.
"""

import importlib.machinery
import os
import sys


def _install_prefill_rope():
    """Keep prefill RoPE on aiter while ATOM_FORCE_ATTN_TRITON forces decode onto Triton.

    The flag also selects fused_qk_rope_reshape_and_cache for prefill. That Q/K
    no longer matches Lumen's aiter RoPE, so the varlen prefill diverges. Decode
    still needs Triton because of the assembly paged-attention bug.
    """
    from align_decode import install

    install()


_install_prefill_rope()


def _install_topk_select_shim():
    """Let a newer ATOM import on an aiter build that has no ``topk_select``.

    ``atom.model_ops.rejection_sampler`` imports the symbol at module load.
    Speculative rejection is the only caller. The shim is installed when that
    module is imported, not at process start, so unrelated Ray workers do not
    import aiter.
    """

    class _Finder:
        def find_spec(self, fullname, path, target=None):
            if fullname != "atom.model_ops.rejection_sampler":
                return None
            sys.meta_path.remove(self)
            try:
                import torch

                import aiter

                if not hasattr(aiter, "topk_select"):

                    def topk_select(input, topk, tie="low", indices_type=torch.int32, **_kwargs):
                        # tie="low" is the lowest index among equal scores, which
                        # is what torch.argmax returns. The sampler only keeps
                        # this result on greedy rows.
                        if int(topk) == 1:
                            idx = input.argmax(dim=-1, keepdim=True).to(indices_type)
                            val = input.gather(-1, idx.to(torch.long))
                            return val, idx
                        val, idx = torch.topk(input, int(topk), dim=-1)
                        return val, idx.to(indices_type)

                    aiter.topk_select = topk_select
                spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            finally:
                sys.meta_path.insert(0, self)
            return spec

    sys.meta_path.insert(0, _Finder())


_install_topk_select_shim()


def _install_saved_kv_resume():
    """Wake the KV pool at the size it slept with.

    The image ATOM recomputes ``get_num_blocks`` on wake. Sleep drops the pool
    from the caching allocator without returning those bytes to the driver, so
    the second reading treats the old pool as non-torch memory and the budget
    goes negative. The engine's block pool and the captured graphs are sized
    from the startup count, so the wake allocates that count again.
    """

    def _wrap(module):
        cls = module.MemoryManagerMixin
        current = cls._resume_kv_cache
        if getattr(current, "_lumenrl_saved_blocks", False):
            return
        names = getattr(getattr(current, "__code__", None), "co_names", ())
        if "get_num_blocks" not in names:
            return
        keeps = module.sleep_keeps_memory_resident
        log = module.logger

        def _resume_kv_cache(self) -> None:
            if keeps(self) and getattr(self, "kv_cache", None) is not None:
                return
            saved_blocks = getattr(self, "_kv_cache_num_blocks", None)
            if saved_blocks is None:
                log.warning(f"{self.label}: No KV cache num_blocks to resume from")
                return
            import torch

            torch.cuda.empty_cache()
            free, total = torch.cuda.mem_get_info()
            log.info(
                f"{self.label}: re-allocating {saved_blocks} KV blocks "
                f"({free / (1 << 30):.2f}GB free of {total / (1 << 30):.2f}GB)"
            )
            self.allocate_kv_cache(saved_blocks)
            self._kv_cache_num_blocks = None
            log.info(
                f"{self.label}: KV cache re-allocated and bound ({saved_blocks} blocks)"
            )

        _resume_kv_cache._lumenrl_saved_blocks = True
        cls._resume_kv_cache = _resume_kv_cache

    class _Finder:
        def find_spec(self, fullname, path, target=None):
            if fullname != "atom.rollout.memory_manager":
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
                _wrap(module)

            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _Finder())


_install_saved_kv_resume()


def _install_weight_sync_fence():
    """After the last IPC bucket, wait until the parameter writes are visible.

    ATOM's updater synchronizes before each in-place copy and returns without
    waiting for that copy. The utility RPC can then complete while the write is
    still queued. This hook only runs when LUMENRL_ALIGN_SYNC_WEIGHTS=1.
    """
    import importlib.machinery
    import zlib

    def _fingerprint(updater):
        import torch

        mapping = updater._get_param_to_module_mapping()
        chosen = next(
            (name for name in mapping if name.endswith("layers.0.self_attn.q_proj.weight")),
            None,
        )
        if chosen is None:
            return "missing"
        param = mapping[chosen][2]
        sample = param.detach().contiguous().view(torch.int16).reshape(-1)[:256]
        crc = zlib.crc32(sample.cpu().numpy().tobytes()) & 0xFFFFFFFF
        return f"crc={crc:08x} numel={param.numel()}"

    def _wrap(module):
        cls = module.WeightUpdaterMixin
        current = cls.update_weights_from_ipc
        if getattr(current, "_lumenrl_fenced", False):
            return
        def wrapped(self, ipc_handle, bucket_meta, is_last=True, ipc_handles=None):
            import torch

            updated = current(self, ipc_handle, bucket_meta, is_last, ipc_handles)
            if is_last:
                device = getattr(self, "device", None)
                torch.cuda.synchronize(device if getattr(device, "type", None) == "cuda" else None)
                try:
                    fingerprint = _fingerprint(self)
                except Exception as exc:
                    fingerprint = f"error={exc}"
                print(
                    f"lumenrl weight fingerprint side=atom label={getattr(self, 'label', '')} "
                    f"{fingerprint}",
                    flush=True,
                )
            return updated
        wrapped._lumenrl_fenced = True
        cls.update_weights_from_ipc = wrapped

    class _Finder:
        def find_spec(self, fullname, path, target=None):
            if fullname != "atom.rollout.weight_updater":
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
                _wrap(module)

            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _Finder())


if os.environ.get("LUMENRL_ALIGN_SYNC_WEIGHTS") == "1":
    _install_weight_sync_fence()

if os.environ.get("LUMENRL_PREFILL_RESCORE", "0") == "1":
    from prefill_rescore import install as _install_prefill_rescore

    _install_prefill_rescore()

if os.environ.get("LUMENRL_STABLE_GEMM", "0") != "1":
    # No-op when not explicitly enabled; avoids any side effects in unrelated
    # Python processes that happen to inherit this PYTHONPATH entry.
    pass
else:
    def _load_call_stable_tgemm_mm():
        """Load the Lumen scan without executing ``lumen/__init__.py``.

        That package init imports the quant config. ATOM workers only need
        the one function, and a missing file must leave the local scan below
        in place.
        """
        import importlib.util

        for entry in sys.path:
            path = os.path.join(entry, "lumen", "ops", "gemm", "stable_tgemm_mm.py")
            if not os.path.isfile(path):
                continue
            spec = importlib.util.spec_from_file_location("_lumen_stable_tgemm_mm", path)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module.call_stable_tgemm_mm
        return None

    _call_stable_tgemm_mm = _load_call_stable_tgemm_mm()
    _cache: dict = {}

    def _stable_mm(orig_mm, get_config, self, inp, weight, bias=None, otype=None):
        """Same scan as ``call_stable_tgemm_mm``, used when that module is absent."""
        import torch

        n, k = weight.shape
        m = inp.numel() // inp.shape[-1]
        dtype_str = str(inp.dtype)

        key = (m, n, k)
        stable_m = _cache.get(key)
        if stable_m is None:
            candidate = m
            for _ in range(512):
                cfg = get_config(
                    candidate, n, k, bias is not None, dtype_str, dtype_str
                )
                sk = cfg.get("splitK") if cfg else None
                if sk is None or int(sk) <= 1:
                    break
                candidate += 1
            _cache[key] = candidate
            stable_m = candidate

        if stable_m == m:
            return orig_mm(self, inp, weight, bias, otype=otype)

        orig_shape = inp.shape
        x_2d = inp.reshape(m, k)
        pad = x_2d.new_zeros(stable_m - m, k)
        x_padded = torch.cat([x_2d, pad], dim=0)
        out = orig_mm(self, x_padded, weight, bias, otype=otype)
        return out[:m].reshape(*orig_shape[:-1], n).contiguous()

    def _patch_tuned_gemm(module):
        orig_mm = module.TunedGemm.mm
        if getattr(orig_mm, "_lumen_stable_tgemm", False):
            return
        get_config = module.get_GEMM_A16W16_config

        def _mm(self, inp, weight, bias=None, otype=None):
            if _call_stable_tgemm_mm is not None:
                return _call_stable_tgemm_mm(orig_mm, self, inp, weight, bias, otype)
            return _stable_mm(orig_mm, get_config, self, inp, weight, bias, otype)

        _mm._lumen_stable_tgemm = True
        module.TunedGemm.mm = _mm

    class _Finder:
        def find_spec(self, fullname, path, target=None):
            if fullname != "aiter.tuned_gemm":
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
                _patch_tuned_gemm(module)

            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _Finder())
