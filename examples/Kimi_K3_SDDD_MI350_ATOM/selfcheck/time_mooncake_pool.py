#!/usr/bin/env python3
"""Time segmented Mooncake producer store creation.

The disaggregated teacher creates its pool lazily: one store per ``put``, and a
batch of 128 sequences is scheduled as 128 single-sequence prefill steps, so the
first batch pays 128 store registrations inside one ``extract_hidden`` command.
That command's budget is 600 s. This script measures the per-store cost so the
pool can either be pre-warmed or resized against a real number.

Usage (inside the training image, with /dev/infiniband and memlock unlimited):
    python3 selfcheck/time_mooncake_pool.py [--stores 32] [--protocol rdma]
"""

from __future__ import annotations

import argparse
import os
import resource
import socket
import time

import torch


def _rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stores", type=int, default=32)
    ap.add_argument("--segment-size", default="2GB")
    ap.add_argument("--local-buffer-size", default="1GB")
    ap.add_argument("--protocol", default="rdma", choices=["rdma", "tcp"])
    ap.add_argument("--payload-tokens", type=int, default=8192,
                    help="rows in the fake per-sequence payload; 8192 is the "
                         "worst case for max_model_len=8192")
    ap.add_argument("--devices", default="",
                    help="comma-separated HCA list; default is ionic_0..6, "
                         "which is what the host actually exposes")
    ap.add_argument("--hard-pin", default="1", choices=["0", "1"])
    a = ap.parse_args()

    devices = a.devices or ",".join(f"ionic_{rank}" for rank in range(7))
    os.environ["MOONCAKE_DEVICE_NAME"] = devices if a.protocol == "rdma" else ""
    os.environ["MOONCAKE_LOCAL_BUFFER_SIZE"] = a.local_buffer_size
    os.environ["LUMENRL_TEACHER_MOONCAKE_SEGMENT_SIZE"] = a.segment_size
    os.environ["LUMENRL_TEACHER_MOONCAKE_SEGMENT_POOL_SIZE"] = str(a.stores)
    os.environ["LUMENRL_TEACHER_MOONCAKE_POOL_WAIT_SECONDS"] = "30"

    from lumenrl.transfer.eagle_mooncake_store import SegmentedEagleMooncakeStore
    from lumenrl.transfer.mooncake_config import MooncakeConfig
    from lumenrl.transfer.mooncake_master import MooncakeMaster

    master = MooncakeMaster()
    info = master.start()
    local_hostname = socket.gethostbyname(socket.gethostname())
    metadata_server = f"http://{local_hostname}:{info['http_port']}/metadata"

    config = MooncakeConfig(
        master_server_address=info["master_addr"],
        metadata_server=metadata_server,
        local_hostname=local_hostname,
        protocol=a.protocol,
        device_name=devices if a.protocol == "rdma" else "",
        global_segment_size=a.segment_size,
        local_buffer_size=a.local_buffer_size,
        # Leave host_buffer_size to __post_init__ so it is sized off max_seq_len
        # and hidden_dim exactly as the teacher path does. Hardcoding 16 MiB (as
        # verify_mooncake_multirank.py does for its 4x8 toy payloads) makes any
        # realistic put fail with "Buffer overflow".
        max_seq_len=8192,
        hidden_dim=7168,
        async_put_pool_size=1,
        enable_hard_pin=a.hard_pin == "1",
        get_retry_max_wait_seconds=30,
    )
    print(f"devices={devices!r} hard_pin={a.hard_pin} segment={a.segment_size} "
          f"local_buffer={a.local_buffer_size} protocol={a.protocol}", flush=True)

    producer = SegmentedEagleMooncakeStore(config)
    per_store: list[float] = []
    try:
        t_pool = time.time()
        for index in range(a.stores):
            t0 = time.time()
            producer._get_store(index)
            dt = time.time() - t0
            per_store.append(dt)
            print(f"store {index + 1:3d}/{a.stores}  {dt:6.2f}s  "
                  f"cum {time.time() - t_pool:7.1f}s  rss {_rss_gib():6.1f} GiB",
                  flush=True)
        pool_seconds = time.time() - t_pool

        width = 5 * 7168
        hidden = torch.zeros((a.payload_tokens, width), dtype=torch.bfloat16)
        last = torch.zeros((a.payload_tokens, 7168), dtype=torch.bfloat16)
        ids = torch.zeros((a.payload_tokens,), dtype=torch.int64)
        payload_mib = (hidden.numel() * 2 + last.numel() * 2 + ids.numel() * 8) / 1024**2

        t_put = time.time()
        puts = min(a.stores, 8)
        for i in range(puts):
            producer.put(f"pool-timing-{i}", hidden, ids, last)
        put_seconds = time.time() - t_put
        for i in range(puts):
            producer._get_store(0).remove_eagle3_tensors(
                f"pool-timing-{i}", has_last_hidden_states=True, has_target=False,
            )

        print("\n=== summary ===")
        print(f"stores            : {a.stores} x {a.segment_size} segment "
              f"+ {a.local_buffer_size} local buffer")
        print(f"pool creation     : {pool_seconds:.1f}s total, "
              f"{pool_seconds / a.stores:.2f}s/store "
              f"(min {min(per_store):.2f} max {max(per_store):.2f})")
        print(f"extrapolated 128  : {128 * pool_seconds / a.stores:.0f}s "
              f"(the extract_hidden budget is 600s)")
        print(f"payload           : {payload_mib:.0f} MiB per sequence "
              f"at {a.payload_tokens} tokens")
        print(f"put               : {puts} payloads in {put_seconds:.1f}s "
              f"= {puts * payload_mib / 1024 / max(put_seconds, 1e-9):.2f} GiB/s")
        print(f"peak rss          : {_rss_gib():.1f} GiB")
        return 0
    finally:
        try:
            producer.close()
        finally:
            master.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
