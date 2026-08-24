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
teacher2 TP=8 ─┤                     each rank fetches 16/128 sequence keys
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

# 3. Full runs use the five-node SLURM solution below.
```

`DATA_ROOT` is where the ~1.5 TB of teacher weights and the checkpoints live;
it must be a real filesystem, since tmpfs is wiped when the allocation ends.
`MODEL_PATH`, `DATASET_SRC`, `CKPT_DIR`, `CACHE_DIR`, `DOCKER_IMAGE` and
`EXTRA_MOUNTS` all override individually if your layout differs.

`run_kimi_k3.sh` runs `selfcheck/preflight.py` first. It needs no GPU, takes
seconds, and fails on the first problem — the point is to catch ATOM API drift
and config typos before a 20-minute weight load rather than after.

## Five-node Ray + SLURM solution

This section is an additional deployment option. It does not replace the
single-node `run_docker.sh` workflow above.

### Topology and prerequisites

`run_multinode_rank.sh` is the recommended launcher. The top-level `srun`
starts exactly one copy on each node; rank 0 forms Ray and launches training
after all five containers are ready. Ray pins four TP=8 ATOM actors to four
nodes and starts one `torchrun --nproc-per-node=8` FSDP2 draft on the remaining
node. Ray is the control plane, Mooncake RDMA is the hidden-state data plane,
and draft gradient synchronization uses RCCL within the draft node.

Before starting:

1. Reserve exactly five idle MI350 nodes with eight GPUs each.
2. Make the repository available at the same absolute path on all nodes.
3. Put Kimi-K3 and the JSONL dataset on each node's local
   `/mnt/m2m_nobackup` filesystem, or override their paths.
4. Mount the same `SHARED_ROOT` on every node. It stores coordination files,
   logs, checkpoints, the token cache, and the one-time static-weight export.
5. Ensure `/dev/infiniband`, the host Ionic userspace provider, and unlimited
   memlock are available to Docker. The launcher passes all three through.
6. Build the same Docker image on every selected node. Docker images are
   node-local and rebuilding only the login/head node is insufficient.

### 1. Select nodes and configure paths

Run from the repository root:

```bash
export K3_NODES='crsuse2-m2m-v2-[030,035,037-039]'
export DATA_ROOT=/mnt/m2m_nobackup/danyzhan
export SHARED_ROOT=/shared_nfs/danyzhan/lumenrl
export MODEL_PATH="${DATA_ROOT}/models/Kimi-K3"
export DATASET_PATH="${DATA_ROOT}/datasets/ATOM_regen_seeklight_kimi_mtp/data/train.jsonl"
export DOCKER_IMAGE=kimi_k3_dspark_atom:latest
```

The defaults already use these values. Other optional overrides are
`CKPT_DIR`, `CACHE_DIR`, `TOKEN_CACHE_DIR`, `RAY_PORT`, and
`EXTRA_OVERRIDES`. The tokenized dataset cache is persistent and reused after
a restart.

Check that every node sees the repository, model, dataset, and shared storage:

```bash
srun --partition=default --nodes=5 --ntasks=5 --nodelist="${K3_NODES}" \
  bash -lc 'test -d "$MODEL_PATH" &&
            test -f "$DATASET_PATH" &&
            test -d "$SHARED_ROOT" &&
            test -d "$PWD" &&
            echo "$(hostname): inputs ready"'
```

### 2. Build and verify the image on all nodes

```bash
srun --partition=default --nodes=5 --ntasks=5 --nodelist="${K3_NODES}" \
  bash -lc 'docker build \
    -f examples/Kimi_K3_SDDD_MI350_ATOM/docker/Dockerfile \
    -t "$DOCKER_IMAGE" .'

# Verify that every image contains the segmented ATOM adapter.
srun --partition=default --nodes=5 --ntasks=5 --nodelist="${K3_NODES}" \
  docker run --rm "$DOCKER_IMAGE" python3 -c \
  'from torchspec.transfer.mooncake.eagle_store import EagleMooncakeStore
assert hasattr(EagleMooncakeStore, "_create_store")
print(EagleMooncakeStore.__mro__[1].__name__)'
```

All five verification tasks must print `SegmentedEagleMooncakeStore`. This
check is important after changing `docker/torchspec`: that shim is copied into
`site-packages` while the image is built.

### 3. Start training

The optional smoke command uses `configs/smoke_test.yaml` and stops after five
optimizer steps:

```bash
LUMENRL_RUN_ID="smoke-$(date +%s)" SMOKE_TEST=1 \
srun --partition=default --nodes=5 --ntasks=5 --nodelist="${K3_NODES}" \
  bash examples/Kimi_K3_SDDD_MI350_ATOM/run_multinode_rank.sh
```

The formal command uses `configs/train.yaml` and reuses an existing token
cache:

```bash
LUMENRL_RUN_ID="train-$(date +%s)" \
srun --partition=default --nodes=5 --ntasks=5 --nodelist="${K3_NODES}" \
  bash examples/Kimi_K3_SDDD_MI350_ATOM/run_multinode_rank.sh
```

`LUMENRL_RUN_ID` must be unique. It names the containers, coordination
directory, and shared log directory, preventing stale state from an earlier
run from being mistaken for the current one.

For a queued workflow, submit the alternative launcher:

```bash
sbatch --nodelist="${K3_NODES}" \
  examples/Kimi_K3_SDDD_MI350_ATOM/run_multinode_slurm.sh
```

### 4. Mooncake capacity and RDMA settings

For batch 128, the teacher uses a lazy pool of 128 independently registered
2 GiB Mooncake segments. Stores bind round-robin to `ionic_0..7`, use a 1 GiB
local buffer, and synchronously flush each sequence before publishing its
manifest. Do not raise either segment above 2 GiB on Ionic hardware.

The launcher supplies the validated RDMA defaults:

```bash
MOONCAKE_DEVICE_NAME=ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7
MOONCAKE_GLOBAL_SEGMENT_SIZE=2GB
MOONCAKE_LOCAL_BUFFER_SIZE=1GB
LUMENRL_TEACHER_MOONCAKE_SEGMENT_POOL_SIZE=128
LUMENRL_TEACHER_MOONCAKE_SEGMENT_SIZE=2GB
LUMENRL_DRAFT_MOONCAKE_SEGMENT_SIZE=2GB
```

The producer pins each write to its own preferred Mooncake segment. Without
that placement constraint Mooncake may select a draft or another teacher
segment, which creates unnecessary cross-node writes and can fail with
`TRANSFER_FAIL`. Do not raise an individual Ionic segment or local buffer above
2 GiB.

### 5. Monitor, stop, and resume

The draft log is written to:

```text
$SHARED_ROOT/logs/kimi_k3_dspark_atom_$LUMENRL_RUN_ID/
```

Each teacher writes its subprocess log on its own host:

```text
/dev/shm/lumenrl_teacher_hidden/atom_teacher_worker.*.log
```

Useful checks:

```bash
squeue -u "$USER" -o '%i %t %R %N'

srun --partition=default --nodes=1 --ntasks=1 --nodelist=<teacher-node> \
  ls -lt /dev/shm/lumenrl_teacher_hidden
```

Stop the full allocation with `scancel <job-id>`. The launcher removes its Ray
containers on exit. A later launch with a new `LUMENRL_RUN_ID` reuses the
token cache and static-weight export; checkpoint resume behavior follows
`configs/train.yaml`.

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
run_multinode_rank.sh   # recommended: one task per node under a five-task srun
run_multinode_slurm.sh  # alternative queued sbatch launcher
al_monitor.py           # train and eval acceptance length side by side
selfcheck/
  preflight.py          # ATOM API contract, config, worker script — no GPU needed
  preprocess_dataset.py # build the cache on CPU and report num_training_steps
  verify_mooncake_multirank.py # real Ionic segmented-store verification
```

`docker/torchspec/` exists because ATOM's runner imports `torchspec.transfer.
mooncake.eagle_store`. The shim points those imports at `lumenrl.transfer`, so
both sides of the transfer agree on the format without patching ATOM.

## Output

| Path | Contents |
|---|---|
| `$SHARED_ROOT/checkpoints/...` | Checkpoints |
| `$SHARED_ROOT/cache/.../teacher-static-weights.pt` | One-time static teacher weights shared with draft ranks |
| `$SHARED_ROOT/logs/kimi_k3_dspark_atom_$LUMENRL_RUN_ID/` | Draft torchrun log |
| `/dev/shm/lumenrl_teacher_hidden/atom_teacher_worker.*.log` | Per-teacher ATOM worker logs |
| `output/Kimi_K3_SDDD/LumenRL/` | Training log |

## Reference

- [Inferact/Kimi-K3-DSpark](https://huggingface.co/Inferact/Kimi-K3-DSpark) — draft model config
- [moonshotai/Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3) — target model
- [DSpark paper (arXiv:2607.05147)](https://arxiv.org/abs/2607.05147)
- [slippedJim/ATOM_regen_seeklight_kimi_mtp](https://huggingface.co/datasets/slippedJim/ATOM_regen_seeklight_kimi_mtp)
