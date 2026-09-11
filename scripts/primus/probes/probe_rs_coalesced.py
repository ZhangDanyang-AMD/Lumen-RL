"""Does coalesced reduce-scatter survive on this image?

Megatron's distributed optimizer reduces gradients as
``_coalescing_manager(group) { reduce_scatter_tensor(bucket) for bucket in ... }``
(param_and_grad_buffer.py start_grad_sync). On primus every actor SIGSEGVs there,
inside ProcessGroupNCCL::reduce_scatter_tensor_coalesced -- with no Megatron, Ray
or vLLM needed to explain it, if this probe reproduces.

Runs plain uncoalesced reduce-scatter first as a control, then the coalesced form
with an increasing number of buckets, so the output says which one breaks.

    torchrun --nproc_per_node=8 probe_rs_coalesced.py
    NBUCKETS=4 torchrun --nproc_per_node=8 probe_rs_coalesced.py
"""

import os

import torch
import torch.distributed as dist
from torch.distributed.distributed_c10d import _coalescing_manager


def log(rank: str, msg: str) -> None:
    print(f"[{rank}] {msg}", flush=True)


def main() -> None:
    dist.init_process_group(backend="nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    tag = f"rank{rank}"
    if rank == 0:
        log(tag, f"torch {torch.__version__} world {world}")

    nbuckets = int(os.getenv("NBUCKETS", "4"))
    elems = int(os.getenv("ELEMS", str(4 << 20)))       # per bucket, must divide by world

    def buckets():
        ins = [torch.ones(elems, dtype=torch.bfloat16, device="cuda") for _ in range(nbuckets)]
        outs = [torch.empty(elems // world, dtype=torch.bfloat16, device="cuda") for _ in ins]
        return ins, outs

    stages = os.getenv("STAGES", "plain,coalesced").split(",")

    # ---- control: the same collectives, one at a time ----
    if "plain" in stages:
        ins, outs = buckets()
        for i, o in zip(ins, outs):
            dist.reduce_scatter_tensor(o, i, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
        log(tag, f"plain reduce_scatter x{nbuckets}: OK (sum {outs[0][0].item()})")

    if "coalesced" not in stages:
        dist.barrier()
        dist.destroy_process_group()
        return

    # ---- the shape Megatron actually issues ----
    for n in (1, nbuckets):
        ins = [torch.ones(elems, dtype=torch.bfloat16, device="cuda") for _ in range(n)]
        outs = [torch.empty(elems // world, dtype=torch.bfloat16, device="cuda") for _ in ins]
        with _coalescing_manager(dist.group.WORLD, async_ops=False):
            for i, o in zip(ins, outs):
                dist.reduce_scatter_tensor(o, i, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
        log(tag, f"coalesced reduce_scatter x{n}: OK (sum {outs[0][0].item()})")

    dist.barrier()
    if rank == 0:
        log(tag, "VERDICT: coalesced reduce-scatter works")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
