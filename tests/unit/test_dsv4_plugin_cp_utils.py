import importlib.util
from pathlib import Path

import pytest
import torch

_CP_UTILS = (
    Path(__file__).resolve().parents[2]
    / "examples/DSv4_DAPO/plugins/vendored/miles_plugins/models/deepseek_v4/ops/cp_utils.py"
)


@pytest.fixture
def cp_utils(monkeypatch):
    spec = importlib.util.spec_from_file_location("dsv4_cp_utils_under_test", _CP_UTILS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # The reference builders end in ``.cuda()``; keep them on the CPU here.
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *args, **kwargs: self)
    return module


@pytest.mark.parametrize("seqlen,window,bsz", [(1, 128, 1), (100, 128, 2), (128, 128, 1), (513, 128, 3)])
def test_window_topk_idxs_match_the_reference(cp_utils, seqlen, window, bsz):
    q_positions = torch.arange(seqlen)

    got = cp_utils.get_window_topk_idxs_cp(q_positions, window_size=window, cp_size=1, bsz=bsz)

    assert torch.equal(got, cp_utils._get_window_topk_idxs_ref(window, bsz, seqlen, start_pos=0))


@pytest.mark.parametrize("seqlen,ratio,bsz", [(4, 4, 1), (130, 4, 2), (512, 128, 1), (1000, 128, 2)])
def test_compress_topk_idxs_match_the_reference(cp_utils, seqlen, ratio, bsz):
    q_positions = torch.arange(seqlen)

    got = cp_utils.get_compress_topk_idxs_cp(q_positions, ratio=ratio, cp_size=1, bsz=bsz)

    want = cp_utils._get_compress_topk_idxs_ref(ratio, bsz, seqlen, start_pos=0, offset=seqlen)
    assert torch.equal(got, want)


def test_index_builders_do_not_consult_the_reference_per_call(cp_utils, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("reference builder called on the forward path")

    monkeypatch.setattr(cp_utils, "_get_window_topk_idxs_ref", forbidden)
    monkeypatch.setattr(cp_utils, "_get_compress_topk_idxs_ref", forbidden)
    q_positions = torch.arange(256)

    cp_utils.get_window_topk_idxs_cp(q_positions, window_size=128, cp_size=1, bsz=2)
    cp_utils.get_compress_topk_idxs_cp(q_positions, ratio=4, cp_size=1, bsz=2)
