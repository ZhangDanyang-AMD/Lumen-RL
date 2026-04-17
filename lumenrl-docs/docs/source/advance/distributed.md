# Distributed training

LumenRL scales along two axes: **training backend parallelism** (FSDP2 vs Megatron-Core) and **cluster orchestration** (Ray placement, multi-node SLURM). This page summarizes how the pieces fit together.

## FSDP2 backend

`policy.training_backend: fsdp2` selects the PyTorch FSDP2 integration under `lumenrl/engine/training/fsdp_backend.py`. It is the default for dense models and smaller MoE runs where expert parallel is not required.

Strengths:

- Native integration with Hugging Face–style module trees
- Works well for single-node and moderate multi-node jobs
- Compatible with Lumen FP8 hooks via `FP8TrainingManager`

## Megatron-Core backend

`policy.training_backend: megatron` enables the Megatron path for large sharded models. Configure tensor, pipeline, and expert parallel sizes via `policy.training.megatron_cfg` (`MegatronConfig` in {doc}`/api/config`).

MoE RL recipes typically need Megatron for **MORI-backed expert parallel** and grouped GEMM flags (`moe_grouped_gemm`).

## Backend support matrix

| Capability | FSDP2 | Megatron-Core |
| --- | --- | --- |
| Dense FP8 training | Yes | Yes |
| MoE training | Limited | Yes (EP via MORI) |
| FP8 rollout (ATOM) | Yes | Yes |
| R3 router replay | Yes | Yes |
| LoRA / adapters | Yes | Yes |
| Multi-node | Yes | Yes |
| Expert parallel | No | Yes |
| Sequence parallelism | Yes | Yes |

```{note}
If your model requires expert parallel (`expert_parallel_size > 1`), start from a Megatron-based recipe such as `configs/grpo_moe_fp8_r3.yaml` rather than forcing FSDP2.
```

## Multi-node launch

At minimum, align YAML cluster fields with the physical allocation:

```yaml
cluster:
  num_nodes: 2
  gpus_per_node: 8
  ray_address: null    # or "auto" / explicit head address
```

For SLURM, keep the driver command in a single variable so `sbatch` can reuse it:

```bash
NUM_NODES=4
COMMAND="python examples/run_grpo_moe.py \
  --config configs/grpo_moe_fp8_r3_multinode.yaml \
  cluster.num_nodes=$NUM_NODES"

sbatch --nodes=$NUM_NODES --gres=gpu:8 scripts/ray.sub
```

Adapt `scripts/ray.sub` to export `RAY_ADDRESS`, NCCL/MORI environment variables, and conda/module initialization for your site.

## Ray cluster setup

Ray options:

- **Single-node local** — default `ray.init()` inside controller utilities when `cluster.ray_address` is unset.
- **Head + workers** — set `cluster.ray_address` to `ray://<head-ip>:10001` (port depends on your Ray version and firewall rules).

Ensure object store memory (`--object-store-memory`) and `/dev/shm` size are adequate for `DataProto` payloads containing long responses and MoE router tensors.

## SLURM integration

Typical pattern:

1. `sbatch` allocates nodes and GPUs.
2. Prolog starts Ray head on rank 0 and Ray workers on other ranks.
3. Rank 0 executes the `COMMAND` Python entrypoint with `cluster.ray_address` pointing at the head.

Validate networking (`NCCL_SOCKET_IFNAME`, RoCE GIDs) independently of LumenRL—distributed hangs here manifest as “stuck rollout” rather than Python tracebacks.

Related pages: {doc}`/quickstart/quick_start`, {doc}`/architecture`, {doc}`/api/config`.
