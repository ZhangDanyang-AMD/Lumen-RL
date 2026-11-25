"""CPU-only behavior tests for the opt-in aligned decode metadata cache."""

from __future__ import annotations

import importlib.util
import os
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

MODULE_PATH = Path(__file__).resolve().parents[3] / "lumenrl/engine/inference/atom_patch/alignment_main.py"
spec = importlib.util.spec_from_file_location("_alignment_main_cuseqlens_test", MODULE_PATH)
alignment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(alignment)


class TestCuSeqlensKCache(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)
        cache = getattr(alignment, "_CUSEQLENS_K_CACHE", None)
        if cache is not None:
            cache.__dict__.clear()

    @staticmethod
    def context(*, prefill=False, cudagraph=False):
        return types.SimpleNamespace(
            context=types.SimpleNamespace(
                is_prefill=prefill,
                forward_mode=types.SimpleNamespace(use_cudagraph=cudagraph),
            )
        )

    def make(self, values, *, ctx=None):
        return alignment._cu_seqlens_k_tensor(values, torch.device("cpu"), ctx)

    def test_default_off_allocates_fresh_equal_tensor(self):
        ctx = self.context()
        first = self.make([0, 4, 9], ctx=ctx)
        second = self.make([0, 4, 9], ctx=ctx)
        self.assertIsNot(first, second)
        self.assertEqual(first.dtype, torch.int32)
        self.assertTrue(torch.equal(first, torch.tensor([0, 4, 9], dtype=torch.int32)))

    def test_decode_reuses_equal_values_and_invalidates_changed_lengths(self):
        os.environ["LUMEN_ATOM_CUSEQLENS_CACHE"] = "1"
        ctx = self.context()
        first = self.make([0, 4, 9], ctx=ctx)
        same = self.make([0, 4, 9], ctx=ctx)
        changed = self.make([0, 5, 9], ctx=ctx)
        self.assertIs(first, same)
        self.assertIsNot(first, changed)
        self.assertTrue(torch.equal(changed, torch.tensor([0, 5, 9], dtype=torch.int32)))

    def test_prefill_and_cudagraph_bypass_cache(self):
        os.environ["LUMEN_ATOM_CUSEQLENS_CACHE"] = "1"
        for ctx in (self.context(prefill=True), self.context(cudagraph=True)):
            first = self.make([0, 4, 9], ctx=ctx)
            second = self.make([0, 4, 9], ctx=ctx)
            self.assertIsNot(first, second)
            self.assertTrue(torch.equal(first, second))

    def test_nonmonotonic_lengths_bypass_cache(self):
        os.environ["LUMEN_ATOM_CUSEQLENS_CACHE"] = "1"
        ctx = self.context()
        first = self.make([0, 4, 9], ctx=ctx)
        invalid = self.make([0, 7, 6], ctx=ctx)
        self.assertIsNot(first, invalid)
        self.assertTrue(torch.equal(invalid, torch.tensor([0, 7, 6], dtype=torch.int32)))


if __name__ == "__main__":
    unittest.main()
