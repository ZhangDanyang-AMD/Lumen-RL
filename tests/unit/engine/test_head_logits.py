"""Log-probs from the last hidden states a chunk of positions at a time (`HeadLogits`)."""

from types import SimpleNamespace

import torch

from lumenrl.engine.training.megatron_base_engine import HeadLogits, MegatronBaseEngine


def _case(seed=0, n=11, d=8, v=37):
    g = torch.Generator().manual_seed(seed)
    hidden = torch.randn(n, d, generator=g, dtype=torch.float64).float().requires_grad_()
    weight = torch.randn(v, d, generator=g, dtype=torch.float64).float().requires_grad_()
    targets = torch.randint(0, v, (n,), generator=g)
    return hidden, weight, targets


def _reference(hidden, weight, targets, temperature):
    lsm = torch.log_softmax(hidden @ weight.t() / temperature, dim=-1)
    return lsm.gather(-1, targets.view(-1, 1)).squeeze(-1), -(lsm.exp() * lsm).sum(-1)


def test_values_and_gradients_are_the_full_logits_ones():
    hidden, weight, targets = _case()
    w = torch.randn(hidden.shape[0], generator=torch.Generator().manual_seed(9))

    want, _ = _reference(hidden, weight, targets, 0.7)
    want_grads = torch.autograd.grad((want * w).sum(), (hidden, weight))
    got = HeadLogits(hidden, weight, 0.7).token_logprob(targets, chunk=4)
    got_grads = torch.autograd.grad((got * w).sum(), (hidden, weight))

    torch.testing.assert_close(got, want, rtol=1e-6, atol=1e-6)
    for a, b in zip(got_grads, want_grads):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)


def test_entropy_without_grad():
    hidden, weight, targets = _case(1)
    lp, ent = HeadLogits(hidden, weight, 1.0).logprob_entropy(targets, chunk=3, want_entropy=True)
    want_lp, want_ent = _reference(hidden, weight, targets, 1.0)
    torch.testing.assert_close(lp, want_lp.detach(), rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(ent, want_ent.detach(), rtol=1e-6, atol=1e-6)


def test_it_slices_positions_like_the_tensor_it_stands_for():
    hidden, weight, targets = _case(2)
    logits = HeadLogits(hidden, weight, 1.0)

    part = logits[2:7][:-1]

    assert logits.shape == (11, 37) and part.shape == (4, 37)
    torch.testing.assert_close(part.token_logprob(targets[2:6], 2), _reference(hidden, weight, targets, 1.0)[0][2:6])


def test_the_engine_helpers_take_it_in_place_of_logits():
    hidden, weight, targets = _case(3)
    engine = SimpleNamespace(_logprob_chunk_size=0)
    want = _reference(hidden, weight, targets, 1.0)[0]

    train = MegatronBaseEngine._token_logprob_train(engine, HeadLogits(hidden, weight, 1.0), targets)
    nograd, _ = MegatronBaseEngine._logprob_entropy_nograd(engine, HeadLogits(hidden, weight, 1.0), targets, False)

    torch.testing.assert_close(train, want)
    torch.testing.assert_close(nograd, want.detach())
