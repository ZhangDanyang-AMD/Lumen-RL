"""Trainer op for score centering: log-probs at selected ids with one LSE per row."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from lumenrl.engine.training.packing import (
    pack_shifted,
    packed_selected_log_probs,
    packed_token_log_probs,
    selected_log_probs,
    unpack_log_probs,
)


def _ref(logits, ids, temperature=1.0):
    return F.log_softmax(logits.float() / temperature, -1).gather(-1, ids)


@pytest.mark.parametrize("chunk_rows", [1, 3, 4096])
@pytest.mark.parametrize("temperature", [1.0, 0.7])
def test_forward_matches_log_softmax_gather(chunk_rows, temperature) -> None:
    torch.manual_seed(0)
    logits = torch.randn(7, 50)
    ids = torch.randint(0, 50, (7, 5))
    out = selected_log_probs(logits, ids, temperature=temperature, chunk_rows=chunk_rows)
    torch.testing.assert_close(out, _ref(logits, ids, temperature))


def test_gradcheck_fp64_with_duplicate_ids() -> None:
    torch.manual_seed(0)
    logits = torch.randn(4, 9, dtype=torch.float64, requires_grad=True)
    ids = torch.tensor([[1, 1, 3], [0, 8, 8], [2, 5, 2], [7, 6, 4]])
    assert torch.autograd.gradcheck(
        lambda z: selected_log_probs(z, ids, temperature=0.8, chunk_rows=3), (logits,)
    )


@pytest.mark.parametrize("chunk_rows", [2, 4096])
def test_backward_matches_autograd_reference(chunk_rows) -> None:
    torch.manual_seed(1)
    base = torch.randn(6, 40)
    ids = torch.randint(0, 40, (6, 4))
    ids[:, 1] = ids[:, 0]  # sampled token also in the head
    g_out = torch.randn(6, 4)

    z1 = base.clone().requires_grad_(True)
    (selected_log_probs(z1, ids, temperature=0.9, chunk_rows=chunk_rows) * g_out).sum().backward()
    z2 = base.clone().requires_grad_(True)
    (_ref(z2, ids, 0.9) * g_out).sum().backward()

    torch.testing.assert_close(z1.grad, z2.grad)


def test_bf16_logits_give_fp32_outputs_and_bf16_grads() -> None:
    torch.manual_seed(2)
    logits = torch.randn(5, 64).bfloat16().requires_grad_(True)
    ids = torch.randint(0, 64, (5, 3))
    out = selected_log_probs(logits, ids)
    assert out.dtype == torch.float32
    torch.testing.assert_close(out, _ref(logits.detach(), ids))
    out.sum().backward()
    assert logits.grad.dtype == torch.bfloat16


def _packed_case(seed=3, v=30, k=4, tp_pad=2):
    """Two left-padded sequences packed flat, plus per-position head ids ``[B, S-1, k]``."""
    torch.manual_seed(seed)
    seq_lens = torch.tensor([5, 3])
    s = 6
    cu = torch.tensor([0, 5, 8], dtype=torch.int32)
    total = int(cu[-1]) + tp_pad
    logits = torch.randn(total, v, requires_grad=True)
    packed_ids = torch.randint(0, v, (total,))
    head = torch.randint(0, v, (2, s - 1, k))
    return logits, packed_ids, cu, seq_lens, s, head


def test_pack_shifted_inverts_unpack_with_trailing_dim() -> None:
    _, _, cu, seq_lens, s, head = _packed_case()
    flat = pack_shifted(head, seq_lens)
    assert flat.shape == (int(seq_lens.sum()) - 2, head.shape[-1])
    back = unpack_log_probs(flat, cu, seq_lens, s)
    for b, sl in enumerate(seq_lens.tolist()):
        torch.testing.assert_close(back[b, s - sl:], head[b, s - sl:])
        assert (back[b, : s - sl] == 0).all()


def test_packed_selected_log_probs_matches_token_and_head_reference() -> None:
    logits, packed_ids, cu, seq_lens, s, head = _packed_case()
    flat_head = pack_shifted(head, seq_lens)

    tok_lp, head_lp = packed_selected_log_probs(
        logits, packed_ids, cu, flat_head, temperature=0.8, chunk_rows=3
    )

    ref_tok = packed_token_log_probs(logits.detach(), packed_ids, cu, temperature=0.8)
    torch.testing.assert_close(tok_lp, ref_tok)
    # Row j of the flat output predicts token j+1 within its sequence.
    rows = torch.tensor([0, 1, 2, 3, 5, 6])  # packed rows that are not a sequence's last token
    ref_head = _ref(logits.detach()[rows], flat_head, 0.8)
    torch.testing.assert_close(head_lp, ref_head)


def test_packed_selected_log_probs_leaves_boundary_and_pad_rows_without_grad() -> None:
    logits, packed_ids, cu, seq_lens, _, head = _packed_case()
    tok_lp, head_lp = packed_selected_log_probs(logits, packed_ids, cu, pack_shifted(head, seq_lens))
    (tok_lp.sum() + head_lp.sum()).backward()
    for row in (4, 7, 8, 9):  # last token of each sequence + TP pad rows
        assert (logits.grad[row] == 0).all()
    assert logits.grad[0].abs().sum() > 0
