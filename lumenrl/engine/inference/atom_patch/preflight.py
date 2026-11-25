"""Fail-closed startup receipts for the pinned clean ATOM main RL recipe.

The preflight runs before AsyncLLMEngine allocates a model or KV cache.
It verifies that selected Lumen-RL hooks are installed on the expected ATOM
source and preserves the identity of the alignment manifest. A model-level
GPU launch and the KL/full-step gates remain separate later checks.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

_PINNED_ATOM_MAIN_CONFIG_SHA256 = (
    "aabbafb2e19f707a761eec3a1944aae9e936254d042f20db1820da84f035276b"
)
_PINNED_CRITICAL_SOURCES = {
    "atom/model_ops/linear.py": "99d9c261494c29fb993243588bbd2704447e2d246dba9cedea66e2a4e653f468",
    "atom/model_ops/attention_mha.py": "cafa342a29c060a94610b42b4fb87112ea26812b7c1dc019784f0b3088c35358",
    "atom/rollout/weight_updater.py": "b599eabcb1d21befec0b087dbc99cbced966b0d613da027757cb3322a019832c",
    "atom/rollout/model_runner_ext.py": "a98580781d1dd0b079e53a5205c0e01d2d546fcea54aec7d9f4defff4f38e280",
    "atom/model_engine/engine_utility.py": "afbf5f7a6b7bd7af3e061f7ca6733907981c5d09092bfb578fa9d4bb425fac94",
}
_TRUE = frozenset(("1", "true", "yes", "on"))


def _on(name: str) -> bool:
    return os.environ.get(name, "0").strip().lower() in _TRUE


def require_clean_atom_main(*, atom_root: str) -> dict[str, object]:
    """Verify the imported config and the actual marked patch call sites."""
    if not _on("LUMEN_ATOM_RL_PATCH"):
        raise RuntimeError("selected recipe requires LUMEN_ATOM_RL_PATCH=1")
    if not _on("LUMENRL_ATOM_OP_ALIGNMENT"):
        raise RuntimeError("selected recipe requires LUMENRL_ATOM_OP_ALIGNMENT=1")
    if not _on("LUMEN_ATOM_MXFP4_2D"):
        raise RuntimeError("selected recipe requires LUMEN_ATOM_MXFP4_2D=1")
    if not _on("LUMEN_ATOM_MAIN_AITER_COMPAT"):
        raise RuntimeError("selected recipe requires the pinned AITER import adapter")
    if not _on("LUMEN_ATOM_MAIN_LIFECYCLE"):
        raise RuntimeError("selected recipe requires LUMEN_ATOM_MAIN_LIFECYCLE=1")

    import atom.config as config
    import atom.model_ops.attention_mha as mha
    import atom.model_ops.embed_head as embed
    import atom.model_ops.linear as linear
    from atom.quant_spec import QuarkOnlineParser

    root = Path(atom_root).resolve()
    path = Path(config.__file__).resolve()
    if path != root / "atom" / "config.py":
        raise RuntimeError(f"wrong imported ATOM config path: {path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != _PINNED_ATOM_MAIN_CONFIG_SHA256:
        raise RuntimeError("imported ATOM is not pinned clean main 456ba2a")
    for module, relative in (
        (mha, "atom/model_ops/attention_mha.py"),
        (linear, "atom/model_ops/linear.py"),
    ):
        if Path(module.__file__).resolve() != root / relative:
            raise RuntimeError(f"wrong imported ATOM module: {relative}")
    for relative, expected in _PINNED_CRITICAL_SOURCES.items():
        candidate = (root / relative).resolve()
        if not candidate.is_file() or not candidate.is_relative_to(root):
            raise RuntimeError(f"missing or relocated clean-main source: {relative}")
        if hashlib.sha256(candidate.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"clean-main critical source mismatch: {relative}")
    installed = {
        "attention": getattr(mha.PagedAttentionImpl, "_lumen_alignment_main_installed", False),
        "linear": getattr(linear.LinearBase, "_lumen_alignment_main_installed", False),
        "lm_head": getattr(embed, "_lumen_alignment_main_installed", False),
    }
    if any(value is not True for value in installed.values()):
        raise RuntimeError(f"Lumen-RL operator hooks missing: {installed}")
    spec = QuarkOnlineParser().parse(
        {"global_quant_config": "mxfp4_2d", "exclude_layer": ["lm_head"]}
    ).global_spec
    if getattr(spec, "weight_block_2d", False) is not True:
        raise RuntimeError("MXFP4-2D parser hook missing")
    if not linear.use_fp4_non_shuffle_triton_gemm() or not callable(
        getattr(linear, "gemm_afp4wfp4", None)
    ):
        raise RuntimeError("selected AITER non-shuffle MXFP4 GEMM unavailable")
    from aiter.ops.triton.gemm_afp4wfp4 import gemm_afp4wfp4

    # The pinned AITER compatibility alias can materialize a second function
    # object on import; compare immutable code identity and its public origin.
    binding = linear.gemm_afp4wfp4
    if (
        binding.__module__ != gemm_afp4wfp4.__module__
        or binding.__qualname__ != gemm_afp4wfp4.__qualname__
        or binding.__code__.co_code != gemm_afp4wfp4.__code__.co_code
        or binding.__code__.co_consts != gemm_afp4wfp4.__code__.co_consts
        or binding.__code__.co_filename != gemm_afp4wfp4.__code__.co_filename
    ):
        raise RuntimeError("ATOM MXFP4 GEMM binding differs from AITER public backend")
    from lumenrl.engine.inference.atom_patch import alignment_main, lifecycle_main

    policy_path, policy_digest = alignment_main._policy_key()
    policy = alignment_main._load_policy(policy_path, policy_digest)
    if policy.get("backend") != "triton":
        raise RuntimeError("selected varlen attention backend is not Triton")
    if not all(
        callable(getattr(lifecycle_main, name, None))
        for name in ("install_lifecycle_main", "install_lifecycle_main_worker")
    ):
        raise RuntimeError("main lifecycle adapter entry points missing")
    # The EngineCore import hook has not necessarily fired in this actor yet.
    # Install the selected adapter explicitly; spawned cores do this again in
    # their own interpreter, and the worker receipt checks their real state.
    if not lifecycle_main.install_lifecycle_main():
        raise RuntimeError("EngineCore lifecycle adapter was not installed")
    from atom.model_engine.engine_utility import EngineUtilityHandler
    from atom.rollout.weight_updater import WeightUpdaterMixin

    if EngineUtilityHandler._UTILITY_HANDLERS.get("get_lumen_refit_receipts") != (
        "_handle_get_lumen_refit_receipts"
    ):
        raise RuntimeError("ModelRunner refit receipt utility not installed")
    if getattr(WeightUpdaterMixin, "_lumen_mxfp4_2d_refit_installed", False) is not True:
        raise RuntimeError("MXFP4-2D refit hook missing from ModelRunner updater")
    if _on("ATOM_ALIGNED_KV_PACK"):
        from lumenrl.engine.inference.atom_patch import alignment_main

        if getattr(alignment_main._gather_live_kv, "__wrapped__", None) is None:
            raise RuntimeError("selected KV pack hook was not installed")
        counters = getattr(alignment_main, "_lumen_c1_counters", None)
        if not isinstance(counters, dict) or set(counters) != {"hit", "fallback"}:
            raise RuntimeError("selected C1 per-process hit/fallback counters missing")
    return {"atom_config_path": str(path), "atom_config_sha256": digest,
            "installed_hooks": installed, "kv_pack": _on("ATOM_ALIGNED_KV_PACK")}
