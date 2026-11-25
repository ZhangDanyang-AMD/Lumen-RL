###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Default-off bootstrap for Lumen-RL's ATOM main adapters.

Put this directory on ``PYTHONPATH`` and set ``LUMEN_ATOM_RL_PATCH=1`` to let
Python import this module automatically at interpreter startup (``site``
imports ``sitecustomize`` from any ``sys.path`` entry).  It installs an import
hook that applies :mod:`lumenrl.engine.inference.atom_patch.install` after the
target ATOM modules are imported, so the ATOM checkout itself stays unmodified.

No-op unless ``LUMEN_ATOM_RL_PATCH`` is one of 1/true/yes/on. The hook
only matches the exact ATOM main module names below. Explicitly selected
alignment/quantization must fail closed on incompatibility.
"""

from __future__ import annotations

import os
import sys

_TRUE = ("1", "true", "yes", "on")
_ACTIVE = os.environ.get("LUMEN_ATOM_RL_PATCH", "0").strip().lower() in _TRUE
_ALIGNMENT = os.environ.get("LUMENRL_ATOM_OP_ALIGNMENT", "0") == "1"
_QUANT = os.environ.get("LUMEN_ATOM_MXFP4_2D", "0").strip().lower() in _TRUE

_TARGETS = frozenset(
    {
        "atom.model_ops.attention_mha",
        "atom.model_ops.linear",
        "atom.model_ops.embed_head",
        "atom.model_engine.model_runner",
        "atom.model_engine.engine_core",
        "atom.rollout.weight_updater",
    }
)


class _AtomPatchFinder:
    """Wrap the regular file loader and patch the module right after exec."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in _TARGETS:
            return None
        try:
            from importlib.machinery import PathFinder

            spec = PathFinder.find_spec(fullname, path)
        except Exception:
            return None
        if spec is None or spec.loader is None:
            return None
        loader = spec.loader

        class _Loader:
            def create_module(self, module_spec):
                return loader.create_module(module_spec)

            def exec_module(self, module):
                if fullname == "atom.model_ops.attention_mha" and (
                    os.environ.get("LUMEN_ATOM_MAIN_AITER_COMPAT", "0") == "1"
                ):
                    # The spawned ModelRunner imports attention_mha itself;
                    # patch the pinned AITER ABI before ATOM's from-import.
                    from lumenrl.engine.inference.atom_patch.aiter_main_compat import (
                        install as install_compat,
                    )

                    if not install_compat():
                        raise RuntimeError("spawned AITER import adapter missing")
                loader.exec_module(module)
                if _ALIGNMENT and fullname in (
                    "atom.model_ops.attention_mha",
                    "atom.model_ops.linear",
                    "atom.model_ops.embed_head",
                ):
                    # Explicit alignment is a correctness contract: no silent
                    # fallback on an ABI/policy mismatch.
                    from lumenrl.engine.inference.atom_patch.alignment_main import apply as align

                    align(fullname)
                if fullname == "atom.model_ops.linear" and _QUANT:
                    # Parent parser and spawned ModelRunner are separate
                    # interpreters: both need the same online 2D quantizer.
                    from lumenrl.engine.inference.atom_patch.quant_main import (
                        install as install_quant,
                    )

                    if not install_quant():
                        raise RuntimeError("spawned MXFP4-2D quant patch missing")
                if fullname == "atom.model_ops.attention_mha" and (
                    os.environ.get("ATOM_ALIGNED_KV_PACK", "0").strip().lower() in _TRUE
                    or os.environ.get("ATOM_DECODE_PROFILE", "0").strip().lower() in _TRUE
                ):
                    from lumenrl.engine.inference.atom_patch.install import apply

                    apply("lumenrl.engine.inference.atom_patch.alignment_main")
                if fullname in ("atom.model_ops.linear", "atom.model_engine.model_runner"):
                    from lumenrl.engine.inference.atom_patch.install import apply

                    apply(fullname)
                if os.environ.get("LUMEN_ATOM_MAIN_LIFECYCLE", "0") == "1":
                    from lumenrl.engine.inference.atom_patch import lifecycle_main

                    if fullname == "atom.model_engine.engine_core":
                        lifecycle_main.install_lifecycle_main()
                    elif fullname == "atom.model_engine.model_runner":
                        lifecycle_main.install_lifecycle_main_worker()
                    elif fullname == "atom.rollout.weight_updater":
                        lifecycle_main.install_weight_sync_guard()
                        if os.environ.get("LUMEN_ATOM_MXFP4_2D", "0") == "1":
                            from lumenrl.engine.inference.atom_patch.refit_main import (
                                install as install_refit,
                            )

                            if not install_refit():
                                raise RuntimeError("MXFP4-2D refit patch not installed")

            def __getattr__(self, name):
                return getattr(loader, name)

        spec.loader = _Loader()
        return spec


if _ACTIVE and not any(
    isinstance(finder, _AtomPatchFinder) for finder in sys.meta_path
):
    sys.meta_path.insert(0, _AtomPatchFinder())
    # Quant install must run explicitly from the ATOM server before model
    # construction: Python's site machinery swallows sitecustomize errors.
    # A selected but unsupported recipe must instead fail the server startup.
