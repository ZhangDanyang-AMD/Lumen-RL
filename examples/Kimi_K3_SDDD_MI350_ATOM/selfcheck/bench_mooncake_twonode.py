#!/usr/bin/env python3
"""Two-node Mooncake put/get bandwidth for the disaggregated teacher path.

Why this exists: the shipped topology asks one teacher process for a pool of 128
independently registered 2 GiB RDMA segments. On this cluster's Ionic HCAs a
process can hold exactly one RDMA client per HCA (the second registration on the
same device fails with EINVAL), so the reachable ceiling is 7 x 2 GiB = 14 GiB --
far below the ~25 GiB an average batch of 128 sequences publishes, and the
producer therefore stalls until the ``extract_hidden`` command times out.

TCP has no memory-region limit, so the question becomes purely one of bandwidth:
can a teacher publish a batch, and can the draft node fetch it, inside one
optimizer step? This measures exactly that between two hosts.

Producer side (node A):
    bench_mooncake_twonode.py --role producer --coord DIR --payloads 16
Consumer side (node B):
    bench_mooncake_twonode.py --role consumer --coord DIR --payloads 16 --workers 8
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

AUX_LAYERS = 5
HIDDEN_DIM = 7168


def _config(cls, master_addr: str, metadata_server: str, local_hostname: str,
            segment_size: str, local_buffer_size: str):
    return cls(
        master_server_address=master_addr,
        metadata_server=metadata_server,
        local_hostname=local_hostname,
        protocol="tcp",
        device_name="",
        global_segment_size=segment_size,
        local_buffer_size=local_buffer_size,
        max_seq_len=8192,
        hidden_dim=HIDDEN_DIM,
        async_put_pool_size=2,
        enable_hard_pin=True,
        kv_lease_ttl_s=3600.0,
        get_retry_max_wait_seconds=300.0,
    )


def _routable_ip() -> str:
    """Address other hosts can reach.

    ``gethostbyname(gethostname())`` returns 127.0.0.1 inside these containers,
    which is why the launcher takes Ray's node IP instead. Ask the routing table
    the same question without needing Ray.
    """
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(("8.8.8.8", 53))
        return probe.getsockname()[0]
    finally:
        probe.close()


def _shapes(tokens: int):
    return (
        {
            "hidden_states": (tokens, AUX_LAYERS * HIDDEN_DIM),
            "input_ids": (tokens,),
            "last_hidden_states": (tokens, HIDDEN_DIM),
        },
        {
            "hidden_states": torch.bfloat16,
            "input_ids": torch.int64,
            "last_hidden_states": torch.bfloat16,
        },
    )


def _payload_bytes(tokens: int) -> int:
    return tokens * AUX_LAYERS * HIDDEN_DIM * 2 + tokens * HIDDEN_DIM * 2 + tokens * 8


def run_producer(a) -> int:
    from lumenrl.transfer.eagle_mooncake_store import EagleMooncakeStore
    from lumenrl.transfer.mooncake_config import MooncakeConfig
    from lumenrl.transfer.mooncake_master import MooncakeMaster

    coord = Path(a.coord)
    coord.mkdir(parents=True, exist_ok=True)
    for stale in coord.glob("*"):
        stale.unlink()

    master = MooncakeMaster()
    info = master.start(kv_lease_ttl_s=3600.0)
    host_ip = _routable_ip()
    master_addr = f"{host_ip}:{info['master_addr'].rsplit(':', 1)[1]}"
    metadata_server = f"http://{host_ip}:{info['http_port']}/metadata"

    store = EagleMooncakeStore(
        _config(MooncakeConfig, master_addr, metadata_server, host_ip,
                a.segment_size, a.local_buffer_size)
    )
    store.setup()
    try:
        hidden = torch.randn(
            (a.tokens, AUX_LAYERS * HIDDEN_DIM), dtype=torch.bfloat16
        )
        last = torch.randn((a.tokens, HIDDEN_DIM), dtype=torch.bfloat16)
        ids = torch.arange(a.tokens, dtype=torch.int64)
        payload = _payload_bytes(a.tokens)

        keys = [f"{a.prefix}-{i}" for i in range(a.payloads)]
        t0 = time.time()
        for key in keys:
            store.put(key, hidden, ids, last)
        store.flush()
        put_seconds = time.time() - t0
        total_gib = a.payloads * payload / 1024**3
        print(f"put {a.payloads} x {payload / 1024**2:.0f} MiB = "
              f"{total_gib:.1f} GiB in {put_seconds:.1f}s "
              f"= {total_gib / put_seconds:.2f} GiB/s", flush=True)

        (coord / "manifest.json").write_text(json.dumps({
            "master_addr": master_addr,
            "metadata_server": metadata_server,
            "keys": keys,
            "tokens": a.tokens,
            "put_seconds": put_seconds,
            "producer": socket.gethostname(),
        }))

        deadline = time.time() + a.wait_seconds
        while time.time() < deadline:
            if (coord / "consumer-done").exists():
                print("consumer finished", flush=True)
                break
            time.sleep(1)
        else:
            print("timed out waiting for the consumer", flush=True)
            return 1
        return 0
    finally:
        store.close()
        master.shutdown()


def run_consumer(a) -> int:
    from lumenrl.transfer.eagle_mooncake_store import EagleMooncakeStore
    from lumenrl.transfer.mooncake_config import MooncakeConfig

    coord = Path(a.coord)
    manifest_path = coord / "manifest.json"
    deadline = time.time() + a.wait_seconds
    while not manifest_path.exists():
        if time.time() > deadline:
            print("no manifest appeared", flush=True)
            return 1
        time.sleep(1)
    manifest = json.loads(manifest_path.read_text())

    host_ip = _routable_ip()
    shapes, dtypes = _shapes(manifest["tokens"])
    payload = _payload_bytes(manifest["tokens"])

    stores = [
        EagleMooncakeStore(
            _config(MooncakeConfig, manifest["master_addr"],
                    manifest["metadata_server"], host_ip,
                    a.segment_size, a.local_buffer_size)
        )
        for _ in range(a.workers)
    ]
    for store in stores:
        store.setup()
    try:
        keys = manifest["keys"]

        def fetch(worker: int) -> int:
            got = 0
            for index in range(worker, len(keys), a.workers):
                output = stores[worker].get(
                    keys[index], shapes, dtypes, device=torch.device("cpu"),
                )
                got += int(output.hidden_states.shape[0])
            return got

        t0 = time.time()
        with ThreadPoolExecutor(max_workers=a.workers) as pool:
            rows = sum(pool.map(fetch, range(a.workers)))
        get_seconds = time.time() - t0
        total_gib = len(keys) * payload / 1024**3
        print(f"get {len(keys)} x {payload / 1024**2:.0f} MiB = "
              f"{total_gib:.1f} GiB in {get_seconds:.1f}s "
              f"= {total_gib / get_seconds:.2f} GiB/s "
              f"({a.workers} workers, {rows} rows)", flush=True)
        print(f"producer put was {manifest['put_seconds']:.1f}s on "
              f"{manifest['producer']}", flush=True)
        (coord / "consumer-done").write_text("ok")
        return 0
    finally:
        for store in stores:
            store.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", required=True, choices=["producer", "consumer"])
    ap.add_argument("--coord", required=True)
    ap.add_argument("--payloads", type=int, default=16)
    ap.add_argument("--tokens", type=int, default=2338,
                    help="mean prompt+response length of the nine-category set")
    ap.add_argument("--workers", type=int, default=8,
                    help="consumer-side parallel fetchers; the draft node has 8 ranks")
    ap.add_argument("--segment-size", default="64GB")
    ap.add_argument("--local-buffer-size", default="8GB")
    ap.add_argument("--prefix", default="twonode-bench")
    ap.add_argument("--wait-seconds", type=float, default=900.0)
    a = ap.parse_args()
    os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    return run_producer(a) if a.role == "producer" else run_consumer(a)


if __name__ == "__main__":
    raise SystemExit(main())
