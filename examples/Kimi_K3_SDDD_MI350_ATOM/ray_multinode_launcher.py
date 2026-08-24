#!/usr/bin/env python3
"""Launch four TP=8 ATOM teachers and one 8-rank draft training node."""

from __future__ import annotations

import os
import shlex
import socket
import subprocess
import sys
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
    num_nodes = _env_int("LUMENRL_NUM_NODES", 5)
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
    mooncake_devices = os.environ.get("MOONCAKE_DEVICE_NAME", "").strip()
    if not mooncake_devices:
        raise ValueError(
            "MOONCAKE_DEVICE_NAME is required for the RDMA topology"
        )

    if num_nodes != 5:
        raise ValueError(
            f"K3 disaggregated topology requires exactly 5 nodes, got {num_nodes}"
        )

    ray.init(
        address="auto",
        namespace="lumenrl-sddd",
        ignore_reinit_error=True,
    )
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

    shared_dir.mkdir(parents=True, exist_ok=True)
    from lumenrl.transfer.mooncake_master import MooncakeMaster

    mooncake_master = MooncakeMaster()
    mooncake_lease_ttl = float(
        os.environ.get("MOONCAKE_KV_LEASE_TTL_S", "3600")
    )
    master_info = mooncake_master.start(
        kv_lease_ttl_s=mooncake_lease_ttl
    )
    # MooncakeMaster resolves the local hostname, which can point at a
    # container-only address. Ray's node IP is routable from all five nodes.
    master_host = ray.util.get_node_ip_address()
    master_port_value = master_info["master_addr"].rsplit(":", 1)[1]
    master_info["master_addr"] = f"{master_host}:{master_port_value}"
    master_info["metadata_server"] = (
        f"http://{master_host}:{master_info['http_port']}/metadata"
    )
    actor_prefix = os.environ.get(
        "LUMENRL_TEACHER_ACTOR_PREFIX", "kimi-k3-sddd-teacher"
    )
    weights_path = str(shared_dir / "teacher-static-weights.pt")
    runtime_overrides = [
        *overrides,
        "cluster.num_nodes=1",
        f"cluster.gpus_per_node={gpus_per_node}",
        "algorithm.spec_distill.sequential_mode=streaming_disaggregated",
        "algorithm.spec_distill.teacher_replicas=4",
        f"algorithm.spec_distill.teacher_actor_prefix={actor_prefix}",
        f"algorithm.spec_distill.teacher_weights_path={weights_path}",
        f"mooncake.master_server_address={master_info['master_addr']}",
        f"mooncake.metadata_server={master_info['metadata_server']}",
        "mooncake.protocol=rdma",
        f"mooncake.device_name={mooncake_devices}",
        f"mooncake.global_segment_size={os.environ.get('MOONCAKE_GLOBAL_SEGMENT_SIZE', '512GB')}",
        f"mooncake.local_buffer_size={os.environ.get('MOONCAKE_LOCAL_BUFFER_SIZE', '1GB')}",
        "mooncake.enable_gpu_direct=false",
        "mooncake.enable_hard_pin=true",
        f"mooncake.kv_lease_ttl_s={mooncake_lease_ttl}",
        "eval.enabled=false",
    ]

    from lumenrl.engine.inference.atom_teacher_ray import AtomTeacherRayActor

    teacher_cls = ray.remote(AtomTeacherRayActor)
    teacher_actors = []
    draft_actor = None
    try:
        for index, node in enumerate(nodes[:4]):
            ip = node["NodeManagerAddress"]
            actor = teacher_cls.options(
                num_gpus=gpus_per_node,
                num_cpus=1,
                max_concurrency=1,
                resources={f"node:{ip}": 0.001},
                name=f"{actor_prefix}-{index}",
            ).remote(config, runtime_overrides, index)
            teacher_actors.append(actor)

        identities = ray.get([actor.identity.remote() for actor in teacher_actors])
        for identity in identities:
            print(f"Teacher replica: {identity}", flush=True)

        exported = ray.get(
            teacher_actors[0].export_static_weights.remote(weights_path)
        )
        print(f"Teacher static weights exported: {exported}", flush=True)

        draft_node = nodes[4]
        draft_ip = draft_node["NodeManagerAddress"]
        launcher_cls = ray.remote(NodeLauncher)
        draft_actor = launcher_cls.options(
            num_gpus=gpus_per_node,
            num_cpus=1,
            resources={f"node:{draft_ip}": 0.001},
            name="kimi-k3-sddd-draft",
        ).remote()
        identity = ray.get(draft_actor.identity.remote())
        print(f"Draft node: {identity}", flush=True)

        exit_code = ray.get(
            draft_actor.run.remote(
                node_rank=0,
                num_nodes=1,
                gpus_per_node=gpus_per_node,
                master_addr=str(identity["ip"]),
                master_port=master_port,
                config=config,
                overrides=runtime_overrides,
                repo_root=repo_root,
                log_dir=log_dir,
            )
        )
        if exit_code != 0:
            print(f"draft torchrun failed: exit={exit_code}", file=sys.stderr)
            return 1
        print("K3 SDDD 4-teacher + 1-draft job completed", flush=True)
        return 0
    finally:
        shutdown_refs = [actor.shutdown.remote() for actor in teacher_actors]
        if shutdown_refs:
            try:
                ray.get(shutdown_refs, timeout=60)
            except Exception as exc:
                print(f"teacher shutdown warning: {exc}", file=sys.stderr)
                # A failed draft may leave long A1/A2 calls ahead of shutdown
                # in each actor's single-threaded mailbox. Kill the actor
                # processes (their ATOM children use PDEATHSIG) before stopping
                # Mooncake so producer clients cannot outlive the master.
                for actor in teacher_actors:
                    try:
                        ray.kill(actor, no_restart=True)
                    except Exception:
                        pass
        mooncake_master.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
