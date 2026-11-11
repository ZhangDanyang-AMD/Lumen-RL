"""Regression coverage for repeated actor/ref packing-patch initialization."""

import sys
import types


def test_attention_patch_is_idempotent(monkeypatch):
    registry = {"sdpa": object()}
    modeling_utils = types.ModuleType("transformers.modeling_utils")
    modeling_utils.ALL_ATTENTION_FUNCTIONS = registry
    transformers = types.ModuleType("transformers")

    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "transformers.modeling_utils", modeling_utils)

    from lumenrl.engine.training import packing

    original = registry["sdpa"]
    saved_original = packing._original_attn_fn
    try:
        packing._original_attn_fn = None
        assert packing.patch_attention_for_packing()
        assert registry["sdpa"] is packing._varlen_attention_forward
        assert packing._original_attn_fn is original

        assert packing.patch_attention_for_packing()
        assert registry["sdpa"] is packing._varlen_attention_forward
        assert packing._original_attn_fn is original
    finally:
        packing._original_attn_fn = saved_original
