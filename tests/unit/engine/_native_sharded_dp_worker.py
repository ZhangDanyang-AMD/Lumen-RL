"""torchrun worker: a sharded ``data_parallel`` against Megatron DDP on one checkpoint.

Not collected by pytest. Env: ``LUMENRL_ROOT``, ``SHARDED_DP`` (``torch_fsdp2`` or
``megatron_fsdp``), ``SHARDED_DP_MODEL_DIR`` (a directory every rank can see; rank 0
writes a tiny random Qwen3 checkpoint there), ``SHARDED_DP_TP``, ``SHARDED_DP_STRATEGY``
(Megatron FSDP's ``data_parallel_sharding_strategy``, default ``optim_grads_params``).

Both engines load the same weights and see the same rows; data-parallel rank ``r``
gets ``r + 2`` rows, so the sharded engine has to pad microbatch counts into lockstep. Prints
``SHARDED_DP_OK``.
"""

from __future__ import annotations

import os
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, os.environ["LUMENRL_ROOT"])

from megatron.core import parallel_state as mpu  # noqa: E402
from torch.distributed.tensor import DTensor, Shard  # noqa: E402

from lumenrl.core.protocol import DataProto  # noqa: E402
from lumenrl.engine.training.megatron_native_engine import MegatronNativeEngine  # noqa: E402

VOCAB = 512
TP = int(os.environ.get("SHARDED_DP_TP", "1"))
MODE = os.environ.get("SHARDED_DP", "torch_fsdp2")
STRATEGY = os.environ.get("SHARDED_DP_STRATEGY", "optim_grads_params")
LR = 1e-3


def _write_checkpoint(path: str) -> None:
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(0)
    cfg = Qwen3Config(
        vocab_size=VOCAB, hidden_size=128, intermediate_size=256, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=32,
        max_position_embeddings=256, tie_word_embeddings=False, torch_dtype="bfloat16",
    )
    Qwen3ForCausalLM(cfg).to(torch.bfloat16).save_pretrained(path, safe_serialization=True)


def _engine(model_dir: str, data_parallel: str) -> MegatronNativeEngine:
    engine = MegatronNativeEngine(
        model_config={"local_path": model_dir},
        engine_config={"data_parallel": data_parallel, "seed": 0,
                       "tensor_model_parallel_size": TP,
                       "data_parallel_sharding_strategy": STRATEGY},
        optimizer_config={"lr": LR, "lr_warmup_steps": 0, "clip_grad": 1.0,
                          "weight_decay": 0.0},
        model_name=model_dir,
    )
    engine.initialize()
    return engine


def _batch(dp_rank: int, dp: int) -> tuple[DataProto, DataProto]:
    """This data-parallel rank's rows, with every rank's response token count."""
    g = torch.Generator().manual_seed(100 + dp_rank)
    rows, S = dp_rank + 2, 24
    ids = torch.randint(0, VOCAB, (rows, S), generator=g)
    am = torch.zeros(rows, S, dtype=torch.long)
    for r in range(rows):
        am[r, : int(torch.randint(8, S + 1, (1,), generator=g))] = 1
    resp = am[:, 1:].clone()
    tokens = torch.tensor(float(resp.sum()), device="cuda")
    dist.all_reduce(tokens, group=mpu.get_data_parallel_group())
    adv = torch.randn(rows, generator=g)
    lp_batch = DataProto(tensors={"input_ids": ids, "attention_mask": am}, meta={})
    update = DataProto(
        tensors={"input_ids": ids, "attention_mask": am, "response_mask": resp,
                 "advantages": adv},
        meta={"algorithm": "dapo", "batch_num_tokens": float(tokens), "dp_size": dp,
              "global_batch_size": sum(r + 2 for r in range(dp))},
    )
    return lp_batch, update


def _export(engine) -> list[tuple[str, torch.Tensor]]:
    return [(n, t.detach().float().cpu()) for n, t in engine._full_megatron_named_params()]


def main() -> None:
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank % torch.cuda.device_count())
    dist.init_process_group(backend="nccl")
    model_dir = os.environ["SHARDED_DP_MODEL_DIR"]
    if rank == 0:
        _write_checkpoint(model_dir)
    dist.barrier()

    ddp = _engine(model_dir, "ddp")
    fsdp = _engine(model_dir, MODE)
    assert fsdp._sharded_dp and not ddp._sharded_dp
    dp_rank, dp = mpu.get_data_parallel_rank(), mpu.get_data_parallel_world_size()

    # FSDP2: every parameter is a dim-0 slice of this rank's TP shard, the one
    # torch.chunk gives its data-parallel rank.
    whole = dict(ddp.module.named_parameters())
    for name, p in (fsdp.module.named_parameters() if MODE == "torch_fsdp2" else ()):
        assert isinstance(p, DTensor) and tuple(p.placements) == (Shard(0),), name
        chunks = torch.chunk(whole[name].detach(), dp, dim=0)
        want = chunks[dp_rank] if dp_rank < len(chunks) else whole[name].new_empty(0)
        assert torch.equal(p.to_local().reshape(want.shape), want), f"{name} shard"

    before = _export(fsdp)
    assert [n for n, _ in before] == [n for n, _ in _export(ddp)]

    lp_batch, update = _batch(dp_rank, dp)
    lp_ddp = ddp._pp_compute_log_probs(lp_batch)["log_probs"]
    lp_fsdp = fsdp.engine_compute_log_probs(lp_batch)["log_probs"]
    # Same weights gathered back, same kernels: forwards agree bit for bit.
    assert torch.equal(lp_ddp, lp_fsdp), (
        f"log-probs differ by {(lp_ddp - lp_fsdp).abs().max():.3g}")

    update.tensors["old_log_probs"] = lp_ddp.detach().cpu()
    m_ddp = ddp._pp_update_policy(update)
    m_fsdp = fsdp.engine_update_policy(update)
    gn_ddp, gn_fsdp = m_ddp["grad_norm"], m_fsdp["grad_norm"]
    assert gn_ddp > 0 and abs(gn_ddp - gn_fsdp) <= 1e-2 * gn_ddp, (gn_ddp, gn_fsdp)
    if "loss" in m_ddp:
        assert abs(m_ddp["loss"] - m_fsdp["loss"]) <= 1e-3 * max(1.0, abs(m_ddp["loss"]))

    # One Adam step moves each weight by about LR, so the two must agree to well
    # within that, while differing from the weights before the step.
    after_ddp, after_fsdp = _export(ddp), _export(fsdp)
    moved = 0.0
    for (n, a), (_, b), (_, c) in zip(after_ddp, after_fsdp, before, strict=True):
        diff = (a - b).abs()
        assert diff.max() <= 0.5 * LR + 1e-2 * a.abs().max(), f"{n}: {diff.max():.3g}"
        assert (diff > 1e-6).float().mean() < 0.01, f"{n}: {(diff > 1e-6).float().mean():.3g}"
        moved = max(moved, float((a - c).abs().max()))
    assert moved > 0.5 * LR, moved

    # The rollout stream (HF names) drains on every rank.
    stream, _ = fsdp.get_per_tensor_param()
    n_stream = sum(1 for _ in stream)
    assert n_stream > 0

    dist.barrier()
    if rank == 0:
        print(f"SHARDED_DP_OK {MODE} {STRATEGY} world={world} tp={TP} grad_norm ddp={gn_ddp:.6g} "
              f"sharded={gn_fsdp:.6g} "
              f"tensors={len(after_fsdp)} stream={n_stream}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
