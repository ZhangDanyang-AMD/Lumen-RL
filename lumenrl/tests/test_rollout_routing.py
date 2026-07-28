"""R3 packing alignment and replay semantics.

The alignment here is the one thing in R3 that fails silently: feed a token the
wrong row and training still runs, it just optimizes against experts that were
never used. CPU-only. Run: python -m lumenrl.tests.test_rollout_routing
"""

import numpy as np
import torch

from lumenrl.moe.rollout_routing import (
    RoutingReplayContext,
    apply_replay,
    layer_index_of,
    pack_rollout_routing,
)

N_LAYERS, TOP_K = 3, 2


def _rows(lens):
    """Row i gets routing whose every entry encodes (row, position)."""
    out = []
    for i, sl in enumerate(lens):
        a = np.zeros((sl - 1, N_LAYERS, TOP_K), dtype=np.uint8)
        for t in range(sl - 1):
            a[t, :, :] = i * 10 + t
        out.append(a)
    return out


def test_packed_layout_matches_pack_sequences_order():
    """pack_sequences concatenates each row's real tokens in row order."""
    lens = [4, 2, 5]
    idx, valid = pack_rollout_routing(
        _rows(lens), torch.tensor(lens), N_LAYERS, TOP_K, "cpu"
    )
    assert idx.shape == (sum(lens), N_LAYERS, TOP_K)

    off = 0
    for i, sl in enumerate(lens):
        for t in range(sl - 1):
            assert idx[off + t, 0, 0].item() == i * 10 + t, (i, t)
            assert bool(valid[off + t])
        # the engine never ran a forward at the final position, so no routing
        assert not bool(valid[off + sl - 1])
        off += sl
    assert int(valid.sum()) == sum(lens) - len(lens)


def test_missing_and_short_rows_are_tolerated():
    lens = [3, 3]
    rows = _rows(lens)
    rows[0] = None                      # engine returned nothing for this row
    rows[1] = rows[1][:1]               # shorter than L-1
    idx, valid = pack_rollout_routing(rows, torch.tensor(lens), N_LAYERS, TOP_K, "cpu")
    assert not valid[0:3].any()
    assert bool(valid[3]) and not bool(valid[4]) and not bool(valid[5])


def test_shape_mismatch_is_loud():
    bad = [np.zeros((2, N_LAYERS + 1, TOP_K), dtype=np.uint8)]
    try:
        pack_rollout_routing(bad, torch.tensor([3]), N_LAYERS, TOP_K, "cpu")
    except RuntimeError as exc:
        assert "expected" in str(exc)
    else:
        raise AssertionError("wrong-shaped routing was accepted")


def test_replay_swaps_selection_and_regathers_weights():
    T, E = 5, 8
    probs = torch.rand(T, E, requires_grad=True)
    computed = torch.topk(probs, TOP_K, dim=-1)[1]
    injected = torch.zeros(T, TOP_K, dtype=torch.long)
    injected[:, 0], injected[:, 1] = 7, 6
    valid = torch.tensor([True, True, False, True, False])

    idx3 = injected.unsqueeze(1).expand(T, N_LAYERS, TOP_K).contiguous()
    with RoutingReplayContext(idx3, valid):
        out = apply_replay(probs, computed, layer_idx=1)
    assert out is not None
    top_value, merged = out

    # valid rows take the injected experts, invalid rows keep the computed ones
    assert torch.equal(merged[valid], injected[valid])
    assert torch.equal(merged[~valid], computed[~valid])
    # weights come from the live probabilities at the replayed positions
    assert torch.allclose(top_value, probs.gather(1, merged))

    # and the router stays differentiable -- overwriting logits would break this
    top_value.sum().backward()
    assert probs.grad is not None and probs.grad.abs().sum() > 0


def test_no_context_is_a_noop():
    probs = torch.rand(4, 8)
    computed = torch.topk(probs, TOP_K, dim=-1)[1]
    assert apply_replay(probs, computed, layer_idx=0) is None


def test_token_count_mismatch_raises():
    idx = torch.zeros(5, N_LAYERS, TOP_K, dtype=torch.long)
    valid = torch.ones(5, dtype=torch.bool)
    probs = torch.rand(6, 8)  # router saw 6 tokens, routing has 5 rows
    computed = torch.topk(probs, TOP_K, dim=-1)[1]
    with RoutingReplayContext(idx, valid):
        try:
            apply_replay(probs, computed, layer_idx=0)
        except RuntimeError as exc:
            assert "mismatch" in str(exc)
        else:
            raise AssertionError("misaligned routing was silently accepted")


def test_context_restores_previous_state():
    idx = torch.zeros(2, N_LAYERS, TOP_K, dtype=torch.long)
    valid = torch.ones(2, dtype=torch.bool)
    probs = torch.rand(2, 8)
    computed = torch.topk(probs, TOP_K, dim=-1)[1]
    with RoutingReplayContext(idx, valid):
        with RoutingReplayContext(None, None):
            assert apply_replay(probs, computed, 0) is None
        assert apply_replay(probs, computed, 0) is not None
    assert apply_replay(probs, computed, 0) is None


def test_layer_index_parsing():
    assert layer_index_of("model.layers.7.mlp.gate") == 7
    assert layer_index_of("_fsdp_wrapped.model.layers.41.mlp.gate") == 41
    assert layer_index_of("model.mlp.gate") is None


def test_layer_lookup_is_order_independent():
    """Backward recompute visits layers in reverse; lookup must not care."""
    T = 3
    idx = torch.zeros(T, N_LAYERS, TOP_K, dtype=torch.long)
    for layer in range(N_LAYERS):
        idx[:, layer, :] = layer + 1
    valid = torch.ones(T, dtype=torch.bool)
    probs = torch.rand(T, 8)
    computed = torch.topk(probs, TOP_K, dim=-1)[1]
    with RoutingReplayContext(idx, valid):
        forward = [apply_replay(probs, computed, l)[1][0, 0].item() for l in range(N_LAYERS)]
        backward = [apply_replay(probs, computed, l)[1][0, 0].item()
                    for l in reversed(range(N_LAYERS))]
    assert forward == [1, 2, 3] and backward == [3, 2, 1]


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name} ok")
    print("all R3 routing tests passed")
