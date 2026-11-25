"""CPU-only: the C4 switch is the only thing that relaxes aligned Attention's eager requirement."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

MODULE_PATH = Path(__file__).resolve().parents[3] / "lumenrl/engine/inference/atom_patch/alignment_main.py"
EAGER = "aligned Attention requires eager execution"


def _load(env):
    with patch.dict(os.environ, env, clear=True):
        spec = importlib.util.spec_from_file_location("_alignment_main_eager_gate", MODULE_PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


def _config(enforce_eager):
    # enable_prefix_caching=True makes the check right after the eager gate fail, so the
    # error message shows whether the eager gate passed without needing a real Attention impl.
    return SimpleNamespace(
        hf_config=SimpleNamespace(model_type="qwen3"),
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        prefill_context_parallel_size=1,
        decode_context_parallel_size=1,
        enforce_eager=enforce_eager,
        enable_prefix_caching=True,
    )


def _error(module, enforce_eager):
    with pytest.raises(module.AlignmentError) as info:
        module.validate_aligned_attention_config(_config(enforce_eager), None)
    return str(info.value)


@pytest.mark.parametrize("env", [{}, {"LUMEN_ATOM_ALIGNED_GRAPH": "0"}, {"LUMEN_ATOM_ALIGNED_GRAPH_PAGED": "1"}])
@pytest.mark.parametrize("enforce_eager", [False, None, 1, 0])
def test_switch_off_still_requires_enforce_eager_true(env, enforce_eager):
    module = _load(env)
    assert module._GRAPH is False
    assert _error(module, enforce_eager) == EAGER


@pytest.mark.parametrize("env", [{}, {"LUMEN_ATOM_ALIGNED_GRAPH": "1"}])
def test_enforce_eager_true_passes_the_gate(env):
    assert _error(_load(env), True) == "prefix-cache lifecycle is not validated"


def test_switch_on_admits_only_explicit_false():
    module = _load({"LUMEN_ATOM_ALIGNED_GRAPH": "1"})
    assert _error(module, False) == "prefix-cache lifecycle is not validated"
    for eager in (None, 0, 1):
        assert _error(module, eager) == EAGER
