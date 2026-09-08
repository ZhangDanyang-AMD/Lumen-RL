"""ATOM fused-MoE weight sync: name rewriting, staging-buffer relayout, coverage.

CPU-only, no ATOM and no aiter needed. ``relayout_fused_experts`` imports
``atom.model_ops.utils.shuffle_weights`` lazily, so a stub stands in for it --
the point of that test is the offset/shape/dtype arithmetic on the staging
buffer, not the permutation itself. That the real shuffle is bitwise equal to
what ATOM's post-load hook produces was checked on GPU separately; the stub
only has to share ATOM's contract: shape-preserving and written in place.

Run: python -m lumenrl.tests.test_atom_moe_weight_sync
"""

import os
import sys
import types

import torch

from lumenrl.engine.inference.atom_moe_weight_sync import (
    assert_bucket_fully_applied,
    fused_expert_renames,
    relayout_fused_experts,
    rename_bucket_meta,
    require_unsharded_experts,
)

E, I, H = 4, 6, 8


def _meta(shape, offset, dtype=torch.bfloat16):
    numel = torch.Size(shape).numel()
    return {
        "shape": tuple(shape),
        "dtype": str(dtype),
        "offset": offset,
        "nbytes": numel * dtype.itemsize,
    }


def _moe_bucket():
    """One layer's fused experts plus the neighbours that must be left alone."""
    w13 = _meta((E, 2 * I, H), 128)
    w2 = _meta((E, H, I), 128 + w13["nbytes"])
    return {
        "model.layers.0.mlp.experts.gate_up_proj": w13,
        "model.layers.0.mlp.experts.down_proj": w2,
        "model.layers.0.mlp.gate.weight": _meta((E, H), 0),
        "model.layers.0.self_attn.q_proj.weight": _meta((H, H), 64),
    }


def test_renames_only_fused_expert_leaves():
    renames = fused_expert_renames(_moe_bucket())
    assert renames == {
        "model.layers.0.mlp.experts.gate_up_proj": "model.layers.0.mlp.experts.w13_weight",
        "model.layers.0.mlp.experts.down_proj": "model.layers.0.mlp.experts.w2_weight",
    }

    # A dense MLP has the same leaf one level up, and ATOM resolves that one
    # itself through packed_modules_mapping. Rewriting it would break it.
    dense = [
        "model.layers.0.mlp.gate_up_proj",
        "model.layers.0.mlp.down_proj",
        "model.layers.0.self_attn.qkv_proj.weight",
        "lm_head.weight",
    ]
    assert fused_expert_renames(dense) == {}


def test_rename_bucket_meta_keeps_everything_else_intact():
    bucket = _moe_bucket()
    renamed = rename_bucket_meta(bucket, fused_expert_renames(bucket))

    assert list(renamed) == [
        "model.layers.0.mlp.experts.w13_weight",
        "model.layers.0.mlp.experts.w2_weight",
        "model.layers.0.mlp.gate.weight",
        "model.layers.0.self_attn.q_proj.weight",
    ]
    # Same metadata objects: only the keys move, offsets must not shift.
    assert renamed["model.layers.0.mlp.experts.w13_weight"] is bucket[
        "model.layers.0.mlp.experts.gate_up_proj"
    ]
    # A dense bucket is handed back untouched rather than rebuilt.
    dense = {"lm_head.weight": _meta((H, H), 0)}
    assert rename_bucket_meta(dense, {}) is dense


def test_require_unsharded_experts():
    require_unsharded_experts(1, False)
    require_unsharded_experts(1, 0)
    require_unsharded_experts(None, None)  # unset keys mean "1" / "off"

    for tp, ep in ((2, False), (8, False), (1, True), (1, 1)):
        try:
            require_unsharded_experts(tp, ep)
        except RuntimeError as exc:
            assert "stale expert weights" in str(exc)
        else:
            raise AssertionError(f"tp={tp} ep={ep} must be refused, not silently served")


class _ShuffleStub:
    """Stands in for ATOM's shuffle_weights: in place, shape-preserving."""

    def __init__(self):
        self.seen = []

    def __call__(self, *tensors, layout=(16, 16)):
        for tensor in tensors:
            assert isinstance(tensor, torch.nn.Parameter), type(tensor)
            assert tensor.dim() == 3, tensor.shape
            self.seen.append((tuple(tensor.shape), tensor.dtype))
            # Per-expert, like ATOM's 3D branch, and reversing the rows is a
            # real permutation so a missing or doubled call is detectable.
            for i in range(tensor.shape[0]):
                tensor.data[i].copy_(tensor.data[i].flip(0))


def _install_shuffle_stub():
    stub = _ShuffleStub()
    atom = types.ModuleType("atom")
    model_ops = types.ModuleType("atom.model_ops")
    utils = types.ModuleType("atom.model_ops.utils")
    utils.shuffle_weights = stub
    atom.model_ops = model_ops
    model_ops.utils = utils
    sys.modules.update(
        {"atom": atom, "atom.model_ops": model_ops, "atom.model_ops.utils": utils}
    )
    return stub


def test_relayout_writes_through_the_staging_buffer_at_the_right_offsets():
    saved = {k: sys.modules.get(k) for k in ("atom", "atom.model_ops", "atom.model_ops.utils")}
    stub = _install_shuffle_stub()
    try:
        bucket = _moe_bucket()
        renames = fused_expert_renames(bucket)
        w13, w2 = (
            bucket["model.layers.0.mlp.experts.gate_up_proj"],
            bucket["model.layers.0.mlp.experts.down_proj"],
        )
        size = w2["offset"] + w2["nbytes"] + 256
        buffer = torch.zeros(size, dtype=torch.uint8)

        plain = {}
        for meta in (w13, w2):
            tensor = torch.randn(meta["shape"], dtype=torch.bfloat16)
            plain[meta["offset"]] = tensor
            flat = tensor.contiguous().view(-1).view(torch.uint8)
            buffer[meta["offset"] : meta["offset"] + meta["nbytes"]].copy_(flat)
        untouched_tail = buffer[w2["offset"] + w2["nbytes"] :].clone()
        head = buffer[: w13["offset"]].clone()

        relayout_fused_experts(buffer, bucket, renames)

        # Both fused tensors, with the shapes and dtypes their metadata declares.
        assert stub.seen == [
            ((E, 2 * I, H), torch.bfloat16),
            ((E, H, I), torch.bfloat16),
        ]
        for meta in (w13, w2):
            view = (
                buffer[meta["offset"] : meta["offset"] + meta["nbytes"]]
                .view(dtype=torch.bfloat16)
                .view(meta["shape"])
            )
            expected = plain[meta["offset"]].flip(1)  # per-expert row reversal
            assert torch.equal(view, expected), meta["shape"]

        # Bytes outside the fused tensors -- the dense weights parked before them
        # and the slack after -- must not move.
        assert torch.equal(buffer[: w13["offset"]], head)
        assert torch.equal(buffer[w2["offset"] + w2["nbytes"] :], untouched_tail)

        # A dense bucket must not even import ATOM.
        stub.seen.clear()
        relayout_fused_experts(buffer, {"lm_head.weight": _meta((H, H), 0)}, {})
        assert stub.seen == []
    finally:
        for key, module in saved.items():
            if module is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = module


def test_relayout_rejects_a_non_fused_layout():
    saved = {k: sys.modules.get(k) for k in ("atom", "atom.model_ops", "atom.model_ops.utils")}
    _install_shuffle_stub()
    try:
        name = "model.layers.0.mlp.experts.gate_up_proj"
        bucket = {name: _meta((2 * I, H), 0)}  # 2D: per-expert, not fused
        buffer = torch.zeros(bucket[name]["nbytes"] + 64, dtype=torch.uint8)
        try:
            relayout_fused_experts(buffer, bucket, fused_expert_renames(bucket))
        except RuntimeError as exc:
            assert "must be 3D" in str(exc)
        else:
            raise AssertionError("a 2D expert tensor must not be shuffled as fused")
    finally:
        for key, module in saved.items():
            if module is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = module


def test_coverage_accepts_a_fully_applied_bucket():
    bucket = _moe_bucket()
    full = len(bucket)
    # Scalar result (one runner) and list result (several) both count.
    assert_bucket_fully_applied([{"cmd": "x", "result": full}] * 8, bucket)
    assert_bucket_fully_applied([{"cmd": "x", "result": [full, full]}], bucket)
    # No usable counts is a warning, not a failure: nothing was observed.
    assert_bucket_fully_applied([], bucket)
    assert_bucket_fully_applied([{"cmd": "x", "result": None}], bucket)


def test_coverage_rejects_a_partially_applied_bucket():
    bucket = _moe_bucket()
    full = len(bucket)
    responses = [{"cmd": "x", "result": full}] * 7 + [{"cmd": "x", "result": full - 2}]
    try:
        assert_bucket_fully_applied(responses, bucket)
    except RuntimeError as exc:
        message = str(exc)
        assert f"{full - 2}/{full}" in message
        assert "1 of 8 engine(s)" in message
    else:
        raise AssertionError(
            "a bucket ATOM only partly applied is the silent-stale-weights bug"
        )

    # True is an int in Python; a boolean ack must not be read as a count of 1.
    assert_bucket_fully_applied([{"cmd": "x", "result": True}], bucket)


def test_coverage_tolerates_a_short_count_when_shards_accumulate():
    """An FP8 rollout under-reports and it is not a fault.

    ATOM accumulates the shards of a fused parameter and requantizes on the
    last one, so q/k/v_proj count as one update and the two before it as none.
    A bucket ending mid-group then reports fewer updates than it holds. This is
    what deadlocked example 4 -- the receiver raised, its socket closed, and the
    sender sat in recv() forever.
    """
    bucket = _moe_bucket()
    short = [{"cmd": "x", "result": len(bucket) - 11}] * 8

    assert_bucket_fully_applied(short, bucket, exact=False)

    try:
        assert_bucket_fully_applied(short, bucket, exact=True)
    except RuntimeError:
        pass
    else:
        raise AssertionError("a BF16 rollout must still be held to exact counts")


def test_coverage_modes_are_configurable():
    bucket = _moe_bucket()
    short = [{"cmd": "x", "result": 0}]
    previous = os.environ.get("LUMENRL_WEIGHT_SYNC_CHECK")
    try:
        for mode in ("warn", "off", "OFF"):
            os.environ["LUMENRL_WEIGHT_SYNC_CHECK"] = mode
            assert_bucket_fully_applied(short, bucket)
    finally:
        if previous is None:
            os.environ.pop("LUMENRL_WEIGHT_SYNC_CHECK", None)
        else:
            os.environ["LUMENRL_WEIGHT_SYNC_CHECK"] = previous


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name} ok")
    print("all ATOM fused-MoE weight sync tests passed")
