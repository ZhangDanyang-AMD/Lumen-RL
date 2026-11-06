"""Megatron training log-probs take the model's bf16 logits as-is.

The packed training forward used to upcast the whole [T, V] logits to fp32 and
divide by temperature before the log-prob op (two full fp32 copies, ~12 GiB each
at T=21.5K, V=152K). The op upcasts per row chunk and applies temperature itself,
so passing bf16 logits plus the temperature must give bit-identical results.
"""

from __future__ import annotations

import pytest
import torch

from lumenrl.engine.training.megatron_base_engine import MegatronBaseEngine
from lumenrl.engine.training.packing import selected_log_probs


def _engine(chunk=3):
    e = object.__new__(MegatronBaseEngine)
    e._logprob_chunk_size = chunk
    return e


def _inputs(seed=0, rows=9, v=50, k=4):
    g = torch.Generator().manual_seed(seed)
    logits = (torch.randn(rows, v, generator=g) * 4).to(torch.bfloat16)
    targets = torch.randint(0, v, (rows,), generator=g)
    head = torch.randint(0, v, (rows, k), generator=g)
    grad_tok = torch.randn(rows, generator=g)
    grad_head = torch.randn(rows, k, generator=g)
    return logits, targets, head, grad_tok, grad_head


def _saved_bytes(fn):
    """Bytes autograd keeps for backward while ``fn`` runs."""
    total = [0]

    def pack(t):
        total[0] += t.numel() * t.element_size()
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        out = fn()
    return out, total[0]


@pytest.mark.parametrize("temperature", [1.0, 0.7])
def test_head_logprobs_on_bf16_match_the_upcast_then_divide_path_bitwise(temperature) -> None:
    logits, targets, head, g_tok, g_head = _inputs()

    old_in = logits.clone().requires_grad_(True)
    z = old_in.float() / temperature
    ids = torch.cat([targets[:, None], head], dim=1)
    old = selected_log_probs(z, ids, chunk_rows=2)
    (old[:, 0] * g_tok + (old[:, 1:] * g_head).sum(-1)).sum().backward()

    new_in = logits.clone().requires_grad_(True)
    tok, hd = _engine()._token_head_logprob_train(
        new_in, targets, head, chunk_rows=2, temperature=temperature,
    )
    ((tok * g_tok) + (hd * g_head).sum(-1)).sum().backward()

    assert torch.equal(tok, old[:, 0].detach())
    assert torch.equal(hd, old[:, 1:].detach())
    assert new_in.grad.dtype == torch.bfloat16
    assert torch.equal(new_in.grad, old_in.grad)


@pytest.mark.parametrize("chunk", [3, 0])
@pytest.mark.parametrize("temperature", [1.0, 0.7])
def test_token_logprobs_on_bf16_match_fp32_log_softmax(chunk, temperature) -> None:
    logits, targets, _, g_tok, _ = _inputs(seed=1)

    ref_in = logits.clone().requires_grad_(True)
    ref = torch.log_softmax(ref_in.float() / temperature, -1).gather(-1, targets[:, None])[:, 0]
    (ref * g_tok).sum().backward()

    new_in = logits.clone().requires_grad_(True)
    out = _engine(chunk)._token_logprob_train(new_in, targets, temperature=temperature)
    (out * g_tok).sum().backward()

    assert out.dtype == torch.float32
    torch.testing.assert_close(out, ref.detach())
    torch.testing.assert_close(new_in.grad.float(), ref_in.grad.float())


def test_token_logprobs_keep_only_the_input_logits_for_backward() -> None:
    logits, targets, _, _, _ = _inputs(rows=64, v=512)
    x = logits.clone().requires_grad_(True)
    _, saved = _saved_bytes(lambda: _engine(16)._token_logprob_train(x, targets))
    # bf16 logits + per-row LSE + ids; no fp32 [rows, V] softmax.
    assert saved < x.numel() * x.element_size() * 1.1


def test_packed_forward_can_return_the_model_dtype() -> None:
    e = _engine()
    out = torch.randn(1, 5, 7).to(torch.bfloat16)
    model = lambda **kw: out  # noqa: E731
    ids = [torch.arange(2), torch.arange(3)]

    logits, offsets = e._forward_logits_packed(ids, model=model, upcast=False)
    assert logits.dtype == torch.bfloat16 and offsets == [0, 2, 5]
    assert e._forward_logits_packed(ids, model=model)[0].dtype == torch.float32
