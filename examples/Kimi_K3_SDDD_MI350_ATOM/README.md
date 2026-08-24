# Kimi K3 DSpark Draft Distillation (ATOM + FSDP2) — MI350

Off-policy speculative distillation of the DSpark draft against a Kimi K3
teacher. Responses come from `slippedJim/ATOM_regen_seeklight_kimi_mtp`;
ATOM only prefills those sequences and extracts hidden states.

## Why ATOM

The regenerated dataset caches Kimi-K3 responses once, removing the expensive
A1 teacher decode from every training round. ATOM captures auxiliary states
through forward hooks during the remaining prefill pass.

## Architecture

The production topology is five nodes: four resident ATOM TP=8 replicas and
one 8-rank FSDP2 draft node. Whole batches are assigned round-robin
(`batch_id % 4`). Each teacher actor prefills complete dataset sequences before
publishing per-sequence keys. Ray carries only tokens, masks, and keys;
hidden states move through pinned-host Mooncake RDMA.

```
teacher0 TP=8 ─┐
teacher1 TP=8 ─┼─ Mooncake RDMA ─> draft ranks 0..7 (FSDP2 replicate)
teacher2 TP=8 ─┤                     each rank fetches 8/64 sequence keys
teacher3 TP=8 ─┘
```

The bounded prefetch window keeps two batches queued per teacher. There is no
round cache and the draft starts as soon as the oldest complete batch is
available.

`separate_last_hidden: false` — this is the one config line that is not a
verbatim copy of the vLLM variant. ATOM's `hidden_states` already holds all
five aux layers (5 × 7168), and `last_hidden_states` is the model's final
output with the norm already applied. Setting it `true` makes the trainer
concatenate once more, so `fc` receives 6 × 7168 and RMSNorm is applied twice.

## Quick start

```bash
# 1. Build the image, from the repository root (the Dockerfile copies the tree).
docker build -f examples/Kimi_K3_SDDD_MI350_ATOM/docker/Dockerfile \
    -t kimi_k3_dspark_atom:latest .

# 2. Smoke test (5 batches, exercises prefill extraction end to end).
DATA_ROOT=/your/data bash examples/Kimi_K3_SDDD_MI350_ATOM/run_docker.sh --smoke-test

# 3. Full runs use the five-node SLURM launcher below.
```

`DATA_ROOT` is where the ~1.5 TB of teacher weights and the checkpoints live;
it must be a real filesystem, since tmpfs is wiped when the allocation ends.
`MODEL_PATH`, `DATASET_SRC`, `CKPT_DIR`, `CACHE_DIR`, `DOCKER_IMAGE` and
`EXTRA_MOUNTS` all override individually if your layout differs.

`run_kimi_k3.sh` runs `selfcheck/preflight.py` first. It needs no GPU, takes
seconds, and fails on the first problem — the point is to catch ATOM API drift
and config typos before a 20-minute weight load rather than after.

## Multi-node Ray + SLURM

`run_multinode_slurm.sh` starts one Ray daemon container on each of five
allocated nodes. The launcher pins four named ATOM actors to the first four
GPU nodes, starts Mooncake master on the head node, and runs one local
`torchrun --nproc-per-node=8` on the final node. Ray is the control plane,
Mooncake RDMA is the hidden-state data plane, and draft gradient synchronization
uses RCCL within the draft node.

```bash
# Smoke test first. The listed nodes must be idle/allocated to this job.
SMOKE_TEST=1 \
sbatch --nodelist=crsuse2-m2m-v2-[030,035,037-039] \
  examples/Kimi_K3_SDDD_MI350_ATOM/run_multinode_slurm.sh
```

Use exactly five hosts. Model and dataset files live under each node's local
`/mnt/m2m_nobackup/danyzhan`; checkpoints, logs, and the one-time static
teacher-weight export use `SHARED_ROOT` (default `/shared_nfs/danyzhan/lumenrl`).

Set `MOONCAKE_DEVICE_NAME` to the RoCE HCA list visible in the container.
The recipe now follows Jim's TorchSpec-aligned cache-on-policy branch: batch
128, lr `5e-5`, an 8192-token window, and final-assistant-turn supervision.

## Configuration

```yaml
algorithm:
  spec_distill:
    sequential_mode: streaming_disaggregated
    teacher_replicas: 4
    stream_prefetch_batches: 8
    aux_hidden_state_layer_ids: [2, 23, 47, 71, 89]
    separate_last_hidden: false              # inverse of the vLLM variant
    anchor_num: 512
  teacher:
    inference_backend: atom
    tensor_parallel_size: 8
    generate_mode: prefill                   # response comes from the dataset
    transport: mooncake
    atom:
      max_model_len: 8192
      max_num_seqs: 32
      max_num_batched_tokens: 8192
      gpu_memory_utilization: 0.90
      kv_cache_dtype: fp8
      index_cache_dtype: fp8
      enforce_eager: true

policy:
  max_total_sequence_length: 8192
  train_global_batch_size: 128
  learning_rate: 5.0e-5
num_training_steps: 1658
```

Three of these will bite if copied carelessly, and `configs/train.yaml` carries
the full reasoning inline:

- **`kv_cache_dtype/index_cache_dtype: fp8`.** This matches the ATOM serving
  path, avoiding a train/serve context-feature mismatch.
- **Batch 128, lr `5e-5`, max grad norm 1.0.** These are the TorchSpec reference
  optimizer settings and should be changed as a unit.
- **`max_num_batched_tokens >= max_model_len`, chunked prefill and prefix
  caching both off.** ATOM rewrites the same Mooncake key every scheduler step
  without checking `is_final_chunk`, so anything that splits a prefill silently
  keeps only the last chunk. Preflight enforces this.

## Layout

```
configs/
  train.yaml            # full run; every non-obvious value is explained inline
  smoke_test.yaml       # 5 streamed optimizer steps
docker/
  Dockerfile            # rocm/atom-dev + LumenRL + Lumen + torchspec shim
  torchspec/            # shim mapping ATOM's torchspec imports to lumenrl.transfer
run_docker.sh           # host entry point: stages the dataset, launches detached
run_kimi_k3.sh          # in-container entry point: preflight, then torchrun
al_monitor.py           # train and eval acceptance length side by side
selfcheck/
  preflight.py          # ATOM API contract, config, worker script — no GPU needed
  preprocess_dataset.py # build the cache on CPU and report num_training_steps
```

`docker/torchspec/` exists because ATOM's runner imports `torchspec.transfer.
mooncake.eagle_store`. The shim points those imports at `lumenrl.transfer`, so
both sides of the transfer agree on the format without patching ATOM.

## Output

| Path | Contents |
|---|---|
| `$SHARED_ROOT/checkpoints/...` | Checkpoints |
| `$SHARED_ROOT/cache/.../teacher-static-weights.pt` | One-time static teacher weights shared with draft ranks |
| `/dev/shm/lumenrl_teacher_hidden/atom_teacher_worker.log` | Per-teacher ATOM worker log |
| `output/Kimi_K3_SDDD/LumenRL/` | Training log |

## Reference

- [Inferact/Kimi-K3-DSpark](https://huggingface.co/Inferact/Kimi-K3-DSpark) — draft model config
- [moonshotai/Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3) — target model
- [DSpark paper (arXiv:2607.05147)](https://arxiv.org/abs/2607.05147)
- [slippedJim/ATOM_regen_seeklight_kimi_mtp](https://huggingface.co/datasets/slippedJim/ATOM_regen_seeklight_kimi_mtp)
