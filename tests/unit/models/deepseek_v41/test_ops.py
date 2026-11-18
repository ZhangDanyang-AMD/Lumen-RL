import pytest
import torch

from lumenrl.models.deepseek_v41 import ops


def test_e2m1_grid_values_are_fixed_points():
    grid = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    assert torch.equal(ops.round_e2m1(grid), grid)
    assert torch.equal(ops.round_e2m1(-grid), -grid)


@pytest.mark.parametrize(
    "x,want",
    [(0.25, 0.0), (0.75, 1.0), (1.25, 1.0), (1.75, 2.0), (2.5, 2.0), (3.5, 4.0), (5.0, 4.0), (-2.5, -2.0)],
)
def test_e2m1_ties_go_to_the_even_mantissa(x, want):
    assert ops.round_e2m1(torch.tensor([x])).item() == want


def test_e2m1_rounds_to_the_nearest_value_off_ties():
    x = torch.tensor([0.26, 0.74, 1.3, 2.6, 4.9, 5.1])
    assert ops.round_e2m1(x).tolist() == [0.5, 0.5, 1.5, 3.0, 4.0, 6.0]


def test_pow2_ceil_keeps_powers_of_two_and_rounds_others_up():
    x = torch.tensor([0.25, 1.0, 1.5, 3.0, 2.0**-20, 2.0**-20 * 1.01])
    assert ops.pow2_ceil(x).tolist() == [0.25, 1.0, 2.0, 4.0, 2.0**-20, 2.0**-19]


@pytest.mark.parametrize("pow2", [True, False])
def test_fp8_fake_quant_is_idempotent_and_bounded(pow2):
    torch.manual_seed(0)
    x = torch.randn(4, 64) * 3
    once = ops.fp8_fake_quant(x, 32, pow2)

    assert torch.equal(ops.fp8_fake_quant(once, 32, pow2), once)
    amax = x.unflatten(-1, (-1, 32)).abs().amax(-1, keepdim=True)
    err = (once - x).unflatten(-1, (-1, 32)).abs()
    assert (err <= amax / 448 * 2 * 16).all()  # E4M3 keeps 3 mantissa bits; generous bound


@pytest.mark.parametrize("fmt,block", [("e8m0", 32), ("e4m3", 16)])
def test_fp4_fake_quant_lands_on_scaled_grid_and_is_idempotent(fmt, block):
    torch.manual_seed(0)
    x = torch.randn(3, 64, dtype=torch.float32)
    once = ops.fp4_fake_quant(x, block, fmt)

    assert torch.equal(ops.fp4_fake_quant(once, block, fmt), once)
    assert once.unflatten(-1, (-1, block)).unique(dim=-1).numel() > 0
    # every group holds at most 15 distinct values (signed E2M1 grid)
    for g in once.unflatten(-1, (-1, block)).reshape(-1, block):
        assert g.unique().numel() <= 15


def _dense_reference(q, kv, sink, idxs, scale):
    b, s, h, d = q.shape
    out = torch.zeros(b, s, h, d)
    for bi in range(b):
        for si in range(s):
            valid = [int(i) for i in idxs[bi, si] if i >= 0]
            rows = kv[bi, valid].float()
            for hi in range(h):
                logits = torch.cat([(rows @ q[bi, si, hi].float()) * scale, sink[hi : hi + 1].float()])
                p = logits.softmax(-1)[:-1]
                out[bi, si, hi] = p @ rows if valid else 0
    return out


def test_sparse_attention_matches_a_dense_softmax_with_a_sink():
    torch.manual_seed(0)
    b, s, h, d, n, t = 2, 5, 3, 8, 11, 4
    q, kv, sink = torch.randn(b, s, h, d), torch.randn(b, n, d), torch.randn(h)
    idxs = torch.stack([torch.stack([torch.randperm(n)[:t] for _ in range(s)]) for _ in range(b)]).int()
    idxs[0, 1, 2:] = -1
    idxs[1, 3, :] = -1

    got = ops.sparse_attention(q, kv, sink, idxs, 0.3)

    assert torch.allclose(got, _dense_reference(q, kv, sink, idxs, 0.3), atol=1e-6)
    assert torch.equal(got[1, 3], torch.zeros(h, d))


def test_sparse_attention_over_query_chunks_is_the_same_and_so_is_its_gradient(monkeypatch):
    g = torch.Generator().manual_seed(3)
    q = torch.randn(2, 13, 4, 16, generator=g).requires_grad_()
    kv = torch.randn(2, 20, 16, generator=g).requires_grad_()
    sink = torch.randn(4, generator=g).requires_grad_()
    idxs = torch.randint(-1, 20, (2, 13, 7), generator=g).int()
    w = torch.randn(2, 13, 4, 16, generator=g)

    want = ops.sparse_attention(q, kv, sink, idxs, 0.25)
    grads = torch.autograd.grad((want * w).sum(), (q, kv, sink))
    monkeypatch.setattr(ops, "SPARSE_ATTENTION_QUERY_CHUNK", 4)
    got = ops.sparse_attention(q, kv, sink, idxs, 0.25)
    chunked = torch.autograd.grad((got * w).sum(), (q, kv, sink))

    assert torch.equal(got, want)
    for a, b in zip(chunked, grads):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)


def test_hc_comb_is_doubly_stochastic_and_pre_post_in_range():
    torch.manual_seed(0)
    hc = 4
    mixes = torch.randn(2, 3, (2 + hc) * hc)
    pre, post, comb = ops.hc_split_sinkhorn(mixes, torch.rand(3), torch.randn((2 + hc) * hc), hc, 20, 1e-6)

    assert ((pre > 0) & (pre <= 1 + 1e-6)).all() and ((post >= 0) & (post <= 2)).all()
    # the last of the 20 iterations normalizes columns, so those are exact and rows converge
    assert torch.allclose(comb.sum(-2), torch.ones(2, 3, hc), atol=1e-5)
    assert torch.allclose(comb.sum(-1), torch.ones(2, 3, hc), atol=1e-3)


def test_straight_through_passes_the_gradient_unchanged():
    x = torch.randn(4, 32, requires_grad=True)
    y = ops.straight_through(x, ops.fp8_fake_quant(x, 32))
    y.sum().backward()

    assert torch.equal(x.grad, torch.ones_like(x))
    assert torch.equal(y.detach(), ops.fp8_fake_quant(x.detach(), 32))


def test_qlinear_quantizes_its_input_and_passes_the_gradient_straight_through():
    torch.manual_seed(0)
    layer = ops.QLinear(64, 16, act_fp8=True)
    x = (torch.randn(3, 64) * 5).requires_grad_()

    y = layer(x)
    y.sum().backward()

    want = torch.nn.functional.linear(ops.fp8_fake_quant(x.detach(), 32, pow2_scale=True), layer.weight)
    assert torch.equal(y, want) and not torch.equal(y, torch.nn.functional.linear(x, layer.weight))
    assert torch.equal(x.grad, layer.weight.sum(0).expand(3, -1))


def test_qlinear_without_act_fp8_is_a_plain_linear():
    layer = ops.QLinear(64, 16)
    x = torch.randn(3, 64)

    assert layer.bias is None and torch.equal(layer(x), torch.nn.functional.linear(x, layer.weight))
