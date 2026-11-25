"""Process-local AITER API compatibility for pinned ATOM main on gfx950.

The fixed ATOM main 456ba2a expects ``pa_decode_gluon_supports_arch`` while
our pinned AITER wheel has ``get_recommended_splits`` but not that predicate.
The aligned Qwen3 recipe always dispatches to pinned varlen Attention instead
of Gluon paged decode. Install an explicit fail-closed predicate before ATOM
imports its attention wrapper; do not modify ATOM or the AITER wheel.
"""

from __future__ import annotations

import os


def install() -> bool:
    if os.environ.get("LUMEN_ATOM_MAIN_AITER_COMPAT", "0") != "1":
        return False
    if os.environ.get("LUMENRL_ATOM_OP_ALIGNMENT", "0") != "1":
        raise RuntimeError("AITER compatibility requires explicit aligned Qwen3 mode")
    if os.environ.get("ATOM_FORCE_ATTN_TRITON", "0") != "1":
        raise RuntimeError("AITER compatibility requires forced Triton attention")
    from aiter.jit.utils.chip_info import get_gfx
    from aiter.ops.triton.gluon import pa_decode_gluon

    if get_gfx() != "gfx950":
        raise RuntimeError("AITER compatibility is validated only on gfx950")
    if not callable(getattr(pa_decode_gluon, "get_recommended_splits", None)):
        raise RuntimeError("unexpected AITER Gluon wrapper ABI")
    if getattr(pa_decode_gluon, "pa_decode_gluon_supports_arch", None) is None:
        # This predicate is irrelevant to aligned Qwen3 execution but must be
        # present to let pinned ATOM main import. Refuse Gluon on this path.
        pa_decode_gluon.pa_decode_gluon_supports_arch = lambda: False
    elif not callable(pa_decode_gluon.pa_decode_gluon_supports_arch):
        raise RuntimeError("invalid AITER Gluon architecture predicate")
    return True
