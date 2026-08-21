"""Can one process open another's CUDA IPC handle on this image?

LumenRL's rollout weight sync (BucketedWeightSender/Receiver) shares a 512 MB
GPU buffer with the colocated vLLM worker through
``torch.multiprocessing.reductions.reduce_tensor``. On primus that import fails
with HSA's "IPC Client Import: Invalid IPC handle! expected N, got 0" followed
by "HIP failure: 'invalid device pointer'", and the run dies several frames
later in an unrelated ``torch.zeros``. This reproduces just the handle exchange,
with no Ray, no vLLM and no Megatron in the picture.

    python3 probe_ipc_primus.py                 # inherit the caller's alloc conf
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False python3 probe_ipc_primus.py
"""

import os
import sys

import torch
import torch.multiprocessing as mp
from torch.multiprocessing.reductions import reduce_tensor


def child(handle, expected, q):
    try:
        func, args = handle
        list_args = list(args)
        list_args[6] = torch.cuda.current_device()   # what LumenRL's rebuild_ipc does
        t = func(*list_args)
        q.put(("ok", float(t[:4].sum().item()), float(expected)))
    except Exception as exc:  # noqa: BLE001
        q.put(("fail", f"{type(exc).__name__}: {exc}", None))


def main() -> None:
    print("torch", torch.__version__)
    print("alloc backend", torch.cuda.get_allocator_backend())
    print("PYTORCH_CUDA_ALLOC_CONF", repr(os.environ.get("PYTORCH_CUDA_ALLOC_CONF")))

    t = torch.ones(1 << 20, dtype=torch.float32, device="cuda")
    handle = reduce_tensor(t)
    torch.cuda.synchronize()

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=child, args=(handle, t[:4].sum().item(), q))
    p.start()
    status, a, b = q.get(timeout=180)
    p.join(60)

    if status == "ok":
        print(f"VERDICT: IPC works (child read {a}, parent has {b})")
        sys.exit(0 if a == b else 1)
    print(f"VERDICT: IPC BROKEN -- {a}")
    sys.exit(1)


if __name__ == "__main__":
    main()
