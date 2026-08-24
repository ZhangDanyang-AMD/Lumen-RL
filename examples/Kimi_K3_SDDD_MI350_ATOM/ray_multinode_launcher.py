#!/usr/bin/env python3
"""Launch the K3 SDDD torch ranks through one Ray actor per node.

Ray is the control plane (placement, process launch, health, and exit
propagation).  FSDP2 and ATOM tensor communication still use NCCL/RCCL.
"""

from __future__ import annotations

import os
import shlex
import socket
import subprocess
import sys
import uuid
from pathlib import Path

import ray


def _env_int(name: str, default: int) -> int:
    value = int(os.environ.get(name, default))
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


class NodeLauncher:
    """Own all GPUs on one Ray node and run that node's torch ranks."""

    def identity(self) -> dict[str, object]:
        return {
            "hostname": socket.gethostname(),
            "ip": ray.util.get_node_ip_address(),
            "gpu_ids": list(ray.get_gpu_ids()),
        }

    def check_shared_file(self, path: str, token: str) -> dict[str, str]:
        probe = Path(path)
        observed = probe.read_text().strip() if probe.is_file() else ""
        if observed != token:
            raise RuntimeError(
                f"{socket.gethostname()} cannot see shared probe {path!r}: "
                f"expected {token!r}, got {observed!r}"
            )
        return {"hostname": socket.gethostname(), "path": str(probe)}

    def run(
        self,
        *,
        node_rank: int,
        num_nodes: int,
        gpus_per_node: int,
        master_addr: str,
        master_port: int,
        config: str,
        overrides: list[str],
        repo_root: str,
        log_dir: str,
    ) -> int:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        log_path = Path(log_dir) / f"node-{node_rank:02d}-{socket.gethostname()}.log"
        command = [
            "torchrun",
            f"--nnodes={num_nodes}",
            f"--nproc-per-node={gpus_per_node}",
            f"--node-rank={node_rank}",
            f"--master-addr={master_addr}",
            f"--master-port={master_port}",
            "-m",
            "lumenrl.trainer.main",
            "--config",
            config,
            *overrides,
        ]

        env = os.environ.copy()
        env.update(
            {
                "PYTHONUNBUFFERED": "1",
                "LUMENRL_LOG_LEVEL": env.get("LUMENRL_LOG_LEVEL", "INFO"),
                "NCCL_TIMEOUT": env.get("NCCL_TIMEOUT", "7200"),
                "RAY_DEDUP_LOGS": "0",
            }
        )
        env.pop("PYTORCH_CUDA_ALLOC_CONF", None)

        with log_path.open("a", buffering=1) as log:
            log.write(
                f"\n=== node_rank={node_rank} host={socket.gethostname()} "
                f"command={shlex.join(command)} ===\n"
            )
            process = subprocess.Popen(
                command,
                cwd=repo_root,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            return process.wait()


def _eligible_nodes(gpus_per_node: int) -> list[dict]:
    nodes = [
        node
        for node in ray.nodes()
        if node.get("Alive")
        and float(node.get("Resources", {}).get("GPU", 0)) >= gpus_per_node
    ]
    head_ip = ray.util.get_node_ip_address()
    nodes.sort(
        key=lambda node: (
            node.get("NodeManagerAddress") != head_ip,
            node.get("NodeManagerAddress", ""),
        )
    )
    return nodes


def main() -> int:
    num_nodes = _env_int("LUMENRL_NUM_NODES", 6)
    gpus_per_node = _env_int("LUMENRL_GPUS_PER_NODE", 8)
    master_port = _env_int("LUMENRL_TORCH_MASTER_PORT", 29500)
    repo_root = os.environ.get("LUMENRL_REPO_ROOT", "/root/lumenrl")
    config = os.environ.get(
        "LUMENRL_CONFIG",
        f"{repo_root}/examples/Kimi_K3_SDDD_MI350_ATOM/configs/train.yaml",
    )
    log_dir = os.environ["LUMENRL_LOG_DIR"]
    shared_dir = Path(os.environ["LUMENRL_SHARED_CACHE_DIR"])
    overrides = shlex.split(os.environ.get("LUMENRL_OVERRIDES", ""))

    ray.init(address="auto", ignore_reinit_error=True)
    nodes = _eligible_nodes(gpus_per_node)
    if len(nodes) != num_nodes:
        details = [
            (node.get("NodeManagerAddress"), node.get("Resources", {}).get("GPU", 0))
            for node in nodes
        ]
        raise RuntimeError(
            f"expected exactly {num_nodes} Ray nodes with >= {gpus_per_node} GPUs, "
            f"found {len(nodes)}: {details}"
        )

    actor_cls = ray.remote(NodeLauncher)
    actors = []
    for index, node in enumerate(nodes):
        ip = node["NodeManagerAddress"]
        actor = actor_cls.options(
            num_gpus=gpus_per_node,
            num_cpus=1,
            resources={f"node:{ip}": 0.001},
            name=f"kimi-k3-sddd-node-{index}",
        ).remote()
        actors.append(actor)

    identities = ray.get([actor.identity.remote() for actor in actors])
    for index, identity in enumerate(identities):
        print(f"Ray node {index}: {identity}", flush=True)

    shared_dir.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    probe = shared_dir / ".ray-shared-filesystem-probe"
    probe.write_text(token)
    ray.get([actor.check_shared_file.remote(str(probe), token) for actor in actors])
    probe.unlink(missing_ok=True)
    print(f"Shared filesystem verified on {num_nodes} nodes: {shared_dir}", flush=True)

    master_addr = str(identities[0]["ip"])
    refs = [
        actor.run.remote(
            node_rank=index,
            num_nodes=num_nodes,
            gpus_per_node=gpus_per_node,
            master_addr=master_addr,
            master_port=master_port,
            config=config,
            overrides=overrides,
            repo_root=repo_root,
            log_dir=log_dir,
        )
        for index, actor in enumerate(actors)
    ]
    exit_codes = ray.get(refs)
    failures = [index for index, code in enumerate(exit_codes) if code != 0]
    if failures:
        print(f"torchrun failed on node ranks {failures}: {exit_codes}", file=sys.stderr)
        return 1
    print(f"K3 SDDD completed on {num_nodes} nodes: {exit_codes}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
