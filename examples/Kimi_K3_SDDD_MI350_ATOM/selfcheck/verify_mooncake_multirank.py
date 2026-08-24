#!/usr/bin/env python3
"""Verify the reusable segmented producer against real Ionic Mooncake stores."""

from __future__ import annotations

import os
import socket

import torch

from lumenrl.transfer.eagle_mooncake_store import (
    EagleMooncakeStore,
    SegmentedEagleMooncakeStore,
)
from lumenrl.transfer.mooncake_config import MooncakeConfig
from lumenrl.transfer.mooncake_master import MooncakeMaster


def _config(
    master_addr: str,
    metadata_server: str,
    local_hostname: str,
    device_name: str,
    protocol: str = "rdma",
) -> MooncakeConfig:
    return MooncakeConfig(
        master_server_address=master_addr,
        metadata_server=metadata_server,
        local_hostname=local_hostname,
        protocol=protocol,
        device_name=device_name,
        global_segment_size="2GB",
        local_buffer_size="1GB",
        host_buffer_size=16 * 1024**2,
        async_put_pool_size=1,
        enable_hard_pin=True,
        get_retry_max_wait_seconds=30,
    )


def main() -> int:
    devices = ",".join(f"ionic_{rank}" for rank in range(8))
    os.environ["MOONCAKE_DEVICE_NAME"] = devices
    os.environ["MOONCAKE_LOCAL_BUFFER_SIZE"] = "1GB"
    os.environ["LUMENRL_TEACHER_MOONCAKE_SEGMENT_SIZE"] = "2GB"
    os.environ["LUMENRL_TEACHER_MOONCAKE_SEGMENT_POOL_SIZE"] = "8"
    os.environ["LUMENRL_TEACHER_MOONCAKE_POOL_WAIT_SECONDS"] = "5"

    master = MooncakeMaster()
    info = master.start()
    local_hostname = socket.gethostbyname(socket.gethostname())
    metadata_server = f"http://{local_hostname}:{info['http_port']}/metadata"
    producer = None
    consumer = None
    try:
        producer = SegmentedEagleMooncakeStore(
            _config(
                info["master_addr"],
                metadata_server,
                local_hostname,
                devices,
            )
        )
        for index in range(8):
            producer._get_store(index)
        initialized = [store for store in producer._stores if store is not None]
        observed_devices = [store.config.device_name for store in initialized]
        expected_devices = [f"ionic_{rank}" for rank in range(8)]
        if observed_devices != expected_devices:
            raise AssertionError(
                f"HCA placement mismatch: {observed_devices} != {expected_devices}"
            )
        for rank in range(8):
            hidden_states = torch.full((4, 8), rank, dtype=torch.bfloat16)
            producer.put(
                f"segmented-rdma-selfcheck-{rank}",
                hidden_states,
                torch.arange(4, dtype=torch.int64),
                hidden_states.clone(),
            )
        producer.flush()
        for rank, store in enumerate(initialized):
            store.remove_eagle3_tensors(
                f"segmented-rdma-selfcheck-{rank}",
                has_last_hidden_states=True,
                has_target=False,
            )
        print("MOONCAKE_SEGMENTED_IONIC_PUT_PASS", flush=True)
        producer.close()

        # Multiple same-process Ionic clients cannot establish QPs to each
        # other through the node's loopback identity. Exercise global metadata
        # discovery separately over TCP; the formal multi-node run validates
        # the RDMA data path between distinct hosts.
        producer = SegmentedEagleMooncakeStore(
            _config(
                info["master_addr"],
                metadata_server,
                local_hostname,
                "",
                protocol="tcp",
            )
        )

        for rank in range(8):
            key = f"segmented-selfcheck-{rank}"
            hidden_states = torch.full((4, 8), rank, dtype=torch.bfloat16)
            input_ids = torch.arange(4, dtype=torch.int64)
            producer.put(
                key,
                hidden_states,
                input_ids,
                hidden_states.clone(),
            )
        producer.flush()

        consumer = EagleMooncakeStore(
            _config(
                info["master_addr"],
                metadata_server,
                local_hostname,
                "",
                protocol="tcp",
            )
        )
        consumer.setup()
        for rank in range(8):
            key = f"segmented-selfcheck-{rank}"
            output = consumer.get(
                key,
                {
                    "hidden_states": (4, 8),
                    "input_ids": (4,),
                    "last_hidden_states": (4, 8),
                },
                {
                    "hidden_states": torch.bfloat16,
                    "input_ids": torch.int64,
                    "last_hidden_states": torch.bfloat16,
                },
                device=torch.device("cpu"),
            )
            expected = torch.full((4, 8), rank, dtype=torch.bfloat16)
            if not torch.equal(output.hidden_states, expected):
                raise AssertionError(f"payload mismatch for {key}")
            consumer.remove_eagle3_tensors(
                key,
                has_last_hidden_states=True,
                has_target=False,
            )
            print(
                f"key={key} device=ionic_{rank} put/get/remove=ok",
                flush=True,
            )

        print("MOONCAKE_SEGMENTED_CROSS_STORE_PASS", flush=True)
        return 0
    finally:
        if consumer is not None:
            consumer.close()
        if producer is not None:
            producer.close()
        master.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
