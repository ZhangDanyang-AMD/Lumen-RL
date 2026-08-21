"""Does the 302 MB/rank expert all-to-all put ANY bytes on the wire?

bringup handoff §4: a 4-node run sat 30 minutes on one collective
(ALLTOALL_BASE, NumelOut=150,994,944 bf16 = 302 MB/rank) and then aborted every
rank, while §4.1 showed the same per-rank volume finishing in 2.5 minutes inside
a single node. That narrowed it to the cross-node path -- NCCL_IB_DISABLE=1
forces TCP over ens3, because RCCL's IB transport cannot drive these ionic RoCE
HCAs -- but left the decisive question open:

    is it DEADLOCKED, or merely catastrophically slow?

Only bytes on the wire can tell those apart, so this samples
/sys/class/net/ens3/statistics/{tx,rx}_bytes throughout:

    rate pinned at 0        -> deadlock. Nothing was ever handed to the NIC.
    rate low but non-zero   -> a throughput problem, and now it has a number.

It loads no model and needs no checkpoint, so it runs on any allocation in
minutes -- which is why §9 put it first.

Sizes are given as ``--seq`` and converted the way the real dispatch does,
``seq x 6 experts x 4096 hidden`` in bf16, so they are directly comparable to the
handoff: seq 1280 is the 63 MB/rank that WORKED on 4 nodes, seq 6144 is the
302 MB/rank that stalled.

    torchrun --nnodes=2 --node_rank=$R --master_addr=$HEAD_IP --master_port=29540 \
        --nproc_per_node=8 probe_a2a_ens3.py --seqs 1280,2048,3072,4096,6144
"""

import argparse
import datetime
import os
import socket
import threading
import time

import torch
import torch.distributed as dist

HIDDEN = 4096
EXPERTS_PER_TOKEN = 6
NIC = os.environ.get("NET_IF", "ens3")


def p0(*a):
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(*a, flush=True)


def nic_bytes(nic: str = NIC):
    """(tx, rx) for the interface, or (None, None) if it is not readable.

    /sys is the host's here because the container runs --network=host, which is
    also why this measures the real wire and not a veth.
    """
    try:
        base = f"/sys/class/net/{nic}/statistics"
        with open(f"{base}/tx_bytes") as f:
            tx = int(f.read())
        with open(f"{base}/rx_bytes") as f:
            rx = int(f.read())
        return tx, rx
    except OSError:
        return None, None


class NicSampler:
    """Print ens3 throughput while a collective is in flight.

    One sampler per NODE (local_rank 0), not per rank: the counter is the node's,
    so eight samplers would print the same number eight times.
    """

    def __init__(self, host: str, period: float = 2.0):
        self.host = host
        self.period = period
        self._stop = threading.Event()
        self._thread = None
        self.samples = []

    def _loop(self):
        last = nic_bytes()
        last_t = time.time()
        while not self._stop.wait(self.period):
            cur = nic_bytes()
            now = time.time()
            if cur[0] is None or last[0] is None:
                continue
            dt = now - last_t
            dtx = (cur[0] - last[0]) / dt / 2**20
            drx = (cur[1] - last[1]) / dt / 2**20
            self.samples.append((dtx, drx))
            print(
                f"NIC[{self.host}] {NIC} tx {dtx:9.1f} MiB/s   rx {drx:9.1f} MiB/s",
                flush=True,
            )
            last, last_t = cur, now

    def start(self):
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)


def run_size(group, world: int, seq: int, iters: int, uneven: bool) -> None:
    """One all_to_all_single at the volume ``seq`` implies, timed."""
    numel = seq * EXPERTS_PER_TOKEN * HIDDEN
    if numel % world:
        numel -= numel % world
    mb = numel * 2 / 10**6

    out = torch.empty(numel, dtype=torch.bfloat16, device="cuda")
    inp = torch.full((numel,), float(dist.get_rank() + 1), dtype=torch.bfloat16,
                     device="cuda")

    in_splits = out_splits = None
    if uneven:
        # MoE dispatch is unbalanced: the stalled run had NumelIn spread over
        # 109M-282M while NumelOut was identical on every rank. Build the whole
        # send matrix from a function of (sender, receiver) so the skew is
        # ASYMMETRIC like real routing, then read row r as what rank r sends and
        # column r as what it receives. That is matched pairwise BY CONSTRUCTION:
        # a genuine split mismatch deadlocks for a reason that has nothing to do
        # with the fabric, which is not what this probe is about.
        base = numel // world
        rank = dist.get_rank()

        def send(src: int, dst: int) -> int:
            # 0.5x..1.5x, i.e. a 3x spread between the lightest and heaviest
            # pair, matching the observed NumelIn range of 109M-282M.
            skew = 0.5 + ((src * 7 + dst * 13) % 5) / 4.0
            return int(base * skew) // 8 * 8

        in_splits = [send(rank, p) for p in range(world)]
        out_splits = [send(p, rank) for p in range(world)]
        inp = torch.full((sum(in_splits),), float(rank + 1),
                         dtype=torch.bfloat16, device="cuda")
        out = torch.empty(sum(out_splits), dtype=torch.bfloat16, device="cuda")
        p0(f"    uneven: this rank sends {sum(in_splits) * 2 / 10**6:.0f} MB, "
           f"receives {sum(out_splits) * 2 / 10**6:.0f} MB")

    dist.barrier(group=group)
    p0(f"\n--- seq {seq}: {mb:.0f} MB/rank ({numel} bf16 elem), "
       f"{iters} iters, {'uneven' if uneven else 'even'} split ---")

    t_all = time.time()
    for i in range(iters):
        t0 = time.time()
        if uneven:
            dist.all_to_all_single(out, inp, out_splits, in_splits, group=group)
        else:
            dist.all_to_all_single(out, inp, group=group)
        torch.cuda.synchronize()
        dt = time.time() - t0
        p0(f"    iter {i}: {dt:7.3f} s   {mb / dt / 1000:6.2f} GB/s per rank")
    dist.barrier(group=group)
    p0(f"    seq {seq} TOTAL {time.time() - t_all:.3f} s -> A2A OK")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", default="1280,6144",
                    help="token counts to try, ascending; 1280 = the 63 MB/rank "
                         "that worked on 4 nodes, 6144 = the 302 MB that stalled")
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--uneven", action="store_true",
                    help="skew send sizes like a real MoE dispatch (§4)")
    ap.add_argument("--intra-node", action="store_true",
                    help="restrict the group to one node -- the §4.1 control, "
                         "which is expected to PASS and proves the probe itself "
                         "and the GPUs are fine")
    ap.add_argument("--sample-period", type=float, default=2.0)
    ap.add_argument("--timeout", type=int, default=180,
                    help="watchdog seconds. Deliberately far shorter than the "
                         "real run's 1800: 302 MB/rank across 2 nodes is ~1.2 GB "
                         "off each node, which even a 1 GB/s TCP path would "
                         "finish in about a second. Declaring a stall at 180 s "
                         "costs nothing, and the byte counters -- not the "
                         "timeout -- are what say whether it was a deadlock.")
    cli = ap.parse_args()

    # local_rank may exceed the GPU count on purpose: running 16 procs on 8 GPUs
    # is how the rank count gets to the real run's 32 without a 4-node
    # allocation, and this collective is network-bound, so sharing a GPU does not
    # change what is being measured.
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank % max(1, torch.cuda.device_count()))
    dist.init_process_group(
        "nccl", timeout=datetime.timedelta(seconds=cli.timeout),
    )
    rank, world = dist.get_rank(), dist.get_world_size()
    host = socket.gethostname()

    tx, _ = nic_bytes()
    if rank == 0:
        p0(f"=== a2a probe: world={world} nic={NIC} "
           f"readable={'yes' if tx is not None else 'NO'} ===")
    print(f"  rank {rank} on {host} (local_rank {local_rank})", flush=True)

    group = None
    if cli.intra_node:
        # 8 ranks per node, contiguous by node_rank -- matches torchrun's layout.
        # Every rank must call new_group for every group, even ones it is not in.
        per_node = int(os.environ.get("LOCAL_WORLD_SIZE", "8"))
        for start in range(0, world, per_node):
            g = dist.new_group(list(range(start, min(start + per_node, world))))
            if start <= rank < start + per_node:
                group = g
        eff_world = per_node
        p0(f"    (intra-node control: groups of {per_node})")
    else:
        eff_world = world

    sampler = None
    if local_rank == 0:
        sampler = NicSampler(host, cli.sample_period)
        sampler.start()

    # Warmup at a trivial size, timed separately: it is where the communicator
    # gets built, so a hang HERE is a rendezvous/transport-setup problem rather
    # than a volume problem, and the two want different fixes.
    t0 = time.time()
    warm = torch.empty(eff_world * 1024, dtype=torch.bfloat16, device="cuda")
    dist.all_to_all_single(warm, warm.clone(), group=group)
    torch.cuda.synchronize()
    p0(f"  warmup (2 MB) took {time.time() - t0:.3f} s -> communicator is up")

    try:
        for s in cli.seqs.split(","):
            s = s.strip()
            if s:
                run_size(group, eff_world, int(s), cli.iters, cli.uneven)
        p0("\nA2A PROBE: ALL SIZES PASSED")
    finally:
        if sampler:
            sampler.stop()
            if sampler.samples:
                peak_tx = max(s[0] for s in sampler.samples)
                peak_rx = max(s[1] for s in sampler.samples)
                print(f"NIC[{host}] peak tx {peak_tx:.1f} MiB/s "
                      f"peak rx {peak_rx:.1f} MiB/s over {len(sampler.samples)} samples",
                      flush=True)

    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
