"""Compatibility shim for ATOM rollout with mixed aiter installs.

The vLLM ROCm image can provide a system ``aiter`` that is ABI-compatible with
the installed ``flydsl`` but lacks a few ATOM MLA Python modules. Some source
trees contain those modules but their top-level ``aiter`` package expects a
newer flydsl. When ``LUMENRL_ATOM_AITER_SRC`` points at that source tree, extend
only the missing subpackage search paths while keeping the system top-level
``aiter`` package active.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import importlib.util
import os
import sys


class _AiterAtomModuleFinder(importlib.abc.MetaPathFinder):
    """Resolve a small set of ATOM-only aiter modules from a source checkout."""

    _MODULES = {
        "aiter.ops.triton.attention.mla": "aiter/ops/triton/attention/mla.py",
        "aiter.ops.triton._triton_kernels.attention.mla": (
            "aiter/ops/triton/_triton_kernels/attention/mla.py"
        ),
        "aiter.ops.triton._triton_kernels.quant.quant_fp8_blockwise": (
            "aiter/ops/triton/_triton_kernels/quant/quant_fp8_blockwise.py"
        ),
        "aiter.ops.triton._triton_kernels.quant.quant_mxfp8": (
            "aiter/ops/triton/_triton_kernels/quant/quant_mxfp8.py"
        ),
        "aiter.ops.triton.kv_cache": "aiter/ops/triton/kv_cache.py",
        "aiter.ops.triton._triton_kernels.kv_cache": (
            "aiter/ops/triton/_triton_kernels/kv_cache.py"
        ),
        "aiter.ops.flydsl.moe_common": "aiter/ops/flydsl/moe_common.py",
    }

    def __init__(self, source_root: str) -> None:
        self.source_root = source_root

    def find_spec(self, fullname: str, path=None, target=None):  # noqa: ANN001
        rel = self._MODULES.get(fullname)
        if rel is None:
            return None
        file_path = os.path.join(self.source_root, rel)
        if not os.path.isfile(file_path):
            return None
        return importlib.util.spec_from_file_location(fullname, file_path)


class _AiterTopkPatchLoader(importlib.abc.Loader):
    def __init__(self, wrapped, fullname: str) -> None:  # noqa: ANN001
        self.wrapped = wrapped
        self.fullname = fullname

    def create_module(self, spec):  # noqa: ANN001
        if hasattr(self.wrapped, "create_module"):
            return self.wrapped.create_module(spec)
        return None

    def exec_module(self, module) -> None:  # noqa: ANN001
        self.wrapped.exec_module(module)
        if self.fullname == "aiter" and not hasattr(module, "topk_gating"):
            def _topk_gating_unavailable(*args, **kwargs):  # noqa: ANN002, ANN003
                raise RuntimeError(
                    "aiter.topk_gating is unavailable in this container's system aiter; "
                    "this compatibility shim only supports dense-model ATOM rollout."
                )

            module.topk_gating = _topk_gating_unavailable
        if self.fullname == "aiter" and hasattr(module, "init_dist_env"):
            original = module.init_dist_env
            if not getattr(original, "_lumen_atom_compat", False):
                def _init_dist_env_compat(*args, **kwargs):  # noqa: ANN002, ANN003
                    kwargs.pop("prefill_context_model_parallel_size", None)
                    return original(*args, **kwargs)

                _init_dist_env_compat._lumen_atom_compat = True  # type: ignore[attr-defined]
                module.init_dist_env = _init_dist_env_compat
        elif self.fullname == "aiter.ops.shuffle":
            fallback = getattr(module, "shuffle_scale", None)
            if fallback is None:
                fallback = getattr(module, "shuffle_scale_a16w4", None)
            if fallback is not None:
                if not hasattr(module, "shuffle_scale"):
                    module.shuffle_scale = fallback
                if not hasattr(module, "moe_shuffle_scale"):
                    module.moe_shuffle_scale = fallback


class _AiterTopkPatchFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path=None, target=None):  # noqa: ANN001
        if fullname not in {"aiter", "aiter.ops.shuffle"}:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return None
        spec.loader = _AiterTopkPatchLoader(spec.loader, fullname)
        return spec


_SRC = os.environ.get("LUMENRL_ATOM_AITER_SRC")
if _SRC and os.path.isdir(_SRC):
    sys.meta_path.insert(0, _AiterAtomModuleFinder(_SRC))
sys.meta_path.insert(0, _AiterTopkPatchFinder())
