"""CPU-only install seam and safety regression for clean ATOM main 456ba2a.

Uses stdlib unittest so this contract can run without torch, AITER or pytest.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import hashlib
import types
import unittest
from pathlib import Path
from unittest.mock import patch

MODULE_PATH = Path(__file__).resolve().parents[3] / "lumenrl/engine/inference/atom_patch/alignment_main.py"
spec = importlib.util.spec_from_file_location("_alignment_main_test", MODULE_PATH)
alignment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(alignment)


class _ForwardContext:
    pass


class FakeAttention:
    __module__ = "atom.model_ops.attention_mha"

    def __init__(self, num_heads, head_dim, scale, num_kv_heads, alibi_slopes, **kwargs):
        self.num_heads = num_heads
        self.rotary_emb = None
        self.q_norm = self.k_norm = None

    def forward_impl(self):
        self.rope_cache(None, None, None, None, None, None)
        self.dispatch_backend(None, None, None, None)

    def rope_cache(self, q, k, v, qkv, position, fwd_ctx):
        fused_qk_rope_reshape_and_cache(q, k)
        return "native"

    def dispatch_backend(self, fwd_ctx, q, k, v):
        return "native dispatch"

    def _use_asm_cache_layout(self):
        return True


class _QuantValue:
    value = 0


class FakeLinear:
    __module__ = "atom.model_ops.linear"

    def __init__(self, quant):
        self.quant_type = types.SimpleNamespace(value=quant)
        self.weight = "weight"
        self.bias = None

    def forward(self, x, x_scale=None, otype=None, out=None):
        if self.quant_type.value == QuantType.No.value:
            if out is None:
                y = tgemm.mm(x, self.weight, self.bias, otype=otype)
            else:
                y = tgemm.mm(x, self.weight, self.bias, otype=otype, out=out)
        else:
            y = tgemm.mm(x, self.weight, self.bias, otype=otype, scale_a=x_scale)
        return y


class TestInstall(unittest.TestCase):
    def setUp(self):
        alignment._RUNTIME.clear()
        self.env = patch.dict(os.environ, {}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_default_off_no_import_or_module_mutation(self):
        module = types.ModuleType("atom.model_ops.attention_mha")
        module.PagedAttentionImpl = FakeAttention
        with patch.dict(sys.modules, {module.__name__: module}):
            before = FakeAttention.dispatch_backend
            self.assertFalse(alignment.apply(module.__name__))
            self.assertIs(FakeAttention.dispatch_backend, before)
            self.assertFalse(alignment.apply("atom.model_ops.aligned_attention"))

    def test_attention_main_install_and_no_redirect_for_non_qwen(self):
        os.environ["LUMENRL_ATOM_OP_ALIGNMENT"] = "1"
        module = types.ModuleType("atom.model_ops.attention_mha")
        cls = type("PagedAttentionImpl", (FakeAttention,), {"__module__": module.__name__})
        # Main's class defines its own methods, unlike inheritance-only fixtures.
        for name in ("__init__", "rope_cache", "dispatch_backend", "forward_impl", "_use_asm_cache_layout"):
            setattr(cls, name, getattr(FakeAttention, name))
        module.PagedAttentionImpl = cls
        module.get_current_atom_config = lambda: types.SimpleNamespace(hf_config=types.SimpleNamespace(model_type="other"))
        module.envs = types.SimpleNamespace(ATOM_FORCE_ATTN_TRITON=True)
        with patch.dict(sys.modules, {module.__name__: module}):
            self.assertTrue(alignment.apply(module.__name__))
            self.assertTrue(alignment.apply(module.__name__))
            inst = cls(32, 128, 128 ** -0.5, 8, None)
            self.assertEqual(inst.dispatch_backend(None, None, None, None), "native dispatch")
            self.assertFalse(getattr(inst, "_lumen_alignment_main_active", False))
            inst._lumen_alignment_main_active = True
            self.assertIs(inst.dispatch_backend(None, None, None, None).__func__, cls.paged_attention_aligned_varlen)

    def test_attention_explicit_qwen_contract_refuses_unsupported_rope(self):
        os.environ["LUMENRL_ATOM_OP_ALIGNMENT"] = "1"
        module = types.ModuleType("atom.model_ops.attention_mha")
        cls = type("PagedAttentionImpl", (FakeAttention,), {"__module__": module.__name__})
        for name in ("__init__", "rope_cache", "dispatch_backend", "forward_impl", "_use_asm_cache_layout"):
            setattr(cls, name, getattr(FakeAttention, name))
        module.PagedAttentionImpl = cls
        module.get_current_atom_config = lambda: types.SimpleNamespace(hf_config=types.SimpleNamespace(model_type="qwen3"))
        module.envs = types.SimpleNamespace(ATOM_FORCE_ATTN_TRITON=True)
        with patch.dict(sys.modules, {module.__name__: module}):
            alignment.apply(module.__name__)
            with patch.object(alignment, "align_rope_cache", side_effect=alignment.AlignmentError("unsupported rotary")):
                with self.assertRaisesRegex(alignment.AlignmentError, "unsupported rotary"):
                    cls(32, 128, 128 ** -0.5, 8, None)

    def test_main_interface_drift_refuses_to_install(self):
        os.environ["LUMENRL_ATOM_OP_ALIGNMENT"] = "1"
        module = types.ModuleType("atom.model_ops.attention_mha")
        cls = type("PagedAttentionImpl", (FakeAttention,), {"__module__": module.__name__})
        for name in ("__init__", "rope_cache", "forward_impl", "_use_asm_cache_layout"):
            setattr(cls, name, getattr(FakeAttention, name))
        cls.dispatch_backend = lambda self, q: None
        module.PagedAttentionImpl = cls
        with patch.dict(sys.modules, {module.__name__: module}):
            with self.assertRaisesRegex(alignment.AlignmentError, "interface drift"):
                alignment.apply(module.__name__)
            self.assertFalse(getattr(cls, "_lumen_alignment_main_installed", False))

    def test_policy_digest_and_schema_fail_closed(self):
        policy = b'{"schema_version": 1}'
        with patch.object(alignment.Path, "read_bytes", return_value=policy):
            with self.assertRaisesRegex(alignment.AlignmentError, "SHA256 mismatch"):
                alignment._load_policy("/virtual/policy.json", "0" * 64)
            digest = hashlib.sha256(policy).hexdigest()
            with self.assertRaisesRegex(alignment.AlignmentError, "unsupported"):
                alignment._load_policy("/virtual/policy.json", digest)

    def test_linear_main_proxy_respects_quant_and_fails_on_out(self):
        os.environ["LUMENRL_ATOM_OP_ALIGNMENT"] = "1"
        module = types.ModuleType("atom.model_ops.linear")
        cls = type("LinearBase", (FakeLinear,), {"__module__": module.__name__})
        cls.forward = FakeLinear.forward
        module.LinearBase = cls
        module.QuantType = types.SimpleNamespace(No=_QuantValue)
        calls = []
        module.tgemm = types.SimpleNamespace(mm=lambda *args, **kwargs: calls.append(kwargs) or "native")
        # The source fixture names are looked up inside its own global scope.
        global QuantType, tgemm
        QuantType, tgemm = module.QuantType, module.tgemm
        fake_torch = types.SimpleNamespace(bfloat16="bf16")
        with patch.dict(sys.modules, {module.__name__: module, "torch": fake_torch}):
            self.assertTrue(alignment.apply(module.__name__))
            tgemm = module.tgemm
            self.assertEqual(cls(1).forward("x", x_scale="scale"), "native")
            self.assertEqual(calls[-1]["scale_a"], "scale")
            with self.assertRaisesRegex(alignment.AlignmentError, "out/scales"):
                cls(0).forward("x", out="out")
            with self.assertRaisesRegex(alignment.AlignmentError, "out/scales"):
                cls(0).forward("x", x_scale="scale")
            os.environ["LUMENRL_ATOM_OP_ALIGNMENT"] = "0"
            self.assertEqual(cls(0).forward("x"), "native")


if __name__ == "__main__":
    unittest.main()
