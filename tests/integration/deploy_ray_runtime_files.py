from __future__ import annotations

import hashlib
import os
from pathlib import Path

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy


ROOT = Path("/workspace/Lumen-RL")
FILES = (
    "lumenrl/core/config.py",
    "lumenrl/controller/ray_worker_group.py",
    "lumenrl/trainer/rl_trainer.py",
    "lumenrl/engine/inference/fp8_weight_quantizer.py",
    "lumenrl/engine/inference/rdma_protocol.py",
    "lumenrl/engine/inference/rdma_weight_transfer.py",
    "lumenrl/engine/inference/vllm_colocate_worker_ext.py",
    "lumenrl/engine/inference/vllm_fp8_utils.py",
    "lumenrl/engine/inference/weight_integrity.py",
    "tests/integration/run_dsv4_weight_sync_integrity.py",
)


@ray.remote(num_cpus=1)
def install(files: dict[str, bytes]) -> dict[str, object]:
    installed: dict[str, str] = {}
    for relative, content in files.items():
        destination = ROOT / relative
        temporary = destination.with_suffix(f"{destination.suffix}.tmp.{os.getpid()}")
        temporary.write_bytes(content)
        temporary.replace(destination)
        installed[relative] = hashlib.sha256(destination.read_bytes()).hexdigest()
    return {
        "node_ip": ray.util.get_node_ip_address(),
        "files": installed,
    }


def main() -> None:
    ray.init(address="auto")
    payload = {relative: (ROOT / relative).read_bytes() for relative in FILES}
    expected = {
        relative: hashlib.sha256(content).hexdigest()
        for relative, content in payload.items()
    }
    nodes = [node for node in ray.nodes() if node["Alive"]]
    results = ray.get(
        [
            install.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=node["NodeID"],
                    soft=False,
                )
            ).remote(payload)
            for node in nodes
        ]
    )
    for result in results:
        if result["files"] != expected:
            raise RuntimeError(f"runtime file hash mismatch: {result}")
    print({"expected": expected, "nodes": results}, flush=True)


if __name__ == "__main__":
    main()
