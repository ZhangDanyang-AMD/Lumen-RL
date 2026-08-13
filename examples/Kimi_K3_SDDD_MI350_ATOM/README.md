# Kimi K3 DSpark Draft Distillation (ATOM + FSDP2) — MI350

On-policy speculative distillation of the DSpark draft against a Kimi K3
teacher, on 8× MI350. Same recipe as the sibling `Kimi_K3_SDDD_MI350_vllm`
example; the teacher inference backend is ATOM instead of vLLM.

## Why ATOM

On-policy training needs two passes over every batch: the teacher decodes its
own continuation (A1), then the prompt and that continuation are prefilled
together so the aux hidden states can be captured (A2).

ATOM captures through forward hooks, so capture does not depend on how the
model decodes and one loaded engine serves both passes. Which pass a request
belongs to is decided per request: the generate sweep withholds the external
request id, the extract sweep supplies it, and ATOM only writes to Mooncake
when an id is present. On vLLM the two passes need different engine configs,
so every switch reloads K3's 1.5 TB of weights.

## Architecture

```
Round = Phase A (all 8 GPUs)  ──────────────>  Phase B (all 8 GPUs)

  A1  ATOM TP=8, decode 50 batches            teacher released,
      no external id -> no capture            draft trained on the 50
        |                                     cached batches
  A2  same engine, prefill prompt+response      ^
      external id -> hooks fire at             |
      layers [2,23,47,71,89]                   |
        |                                      |
      Mooncake TCP ─> /dev/shm/teacher_cache_atom
```

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

# 2. Smoke test (5 cached batches, exercises both sweeps end to end).
DATA_ROOT=/your/data bash examples/Kimi_K3_SDDD_MI350_ATOM/run_docker.sh --smoke-test

# 3. Full run, detached so it survives the launching shell.
DATA_ROOT=/your/data DETACH=1 bash examples/Kimi_K3_SDDD_MI350_ATOM/run_docker.sh

# Resume after a crash or a lost node:
DATA_ROOT=/your/data DETACH=1 EXTRA_OVERRIDES="checkpointing.resume=true" \
    bash examples/Kimi_K3_SDDD_MI350_ATOM/run_docker.sh
```

`DATA_ROOT` is where the ~1.5 TB of teacher weights and the checkpoints live;
it must be a real filesystem, since tmpfs is wiped when the allocation ends.
`MODEL_PATH`, `DATASET_SRC`, `CKPT_DIR`, `CACHE_DIR`, `DOCKER_IMAGE` and
`EXTRA_MOUNTS` all override individually if your layout differs.

`run_kimi_k3.sh` runs `selfcheck/preflight.py` first. It needs no GPU, takes
seconds, and fails on the first problem — the point is to catch ATOM API drift
and config typos before a 20-minute weight load rather than after.

## Configuration

```yaml
algorithm:
  spec_distill:
    sequential_mode: batch_alternating
    cache_batches: 50                        # 11.3 GB each at bs=64
    aux_hidden_state_layer_ids: [2, 23, 47, 71, 89]
    separate_last_hidden: false              # inverse of the vLLM variant
    anchor_num: 512
  teacher:
    inference_backend: atom
    tensor_parallel_size: 8
    generate_mode: generate                  # on-policy
    generate_max_tokens: 992
    transport: mooncake
    atom:
      max_model_len: 2048
      max_num_seqs: 256
      gpu_memory_utilization: 0.90
      kv_cache_dtype: bf16
      enforce_eager: false

training:
  train_global_batch_size: 64
  learning_rate: 7.5e-5
num_training_steps: 5413
```

Four of these will bite if copied carelessly, and `configs/train.yaml` carries
the full reasoning inline:

- **`generate_max_tokens: 992`, not 1024.** A row whose length lands exactly on
  `max_model_len` asks the block table for `ceil(2049/16) = 129` blocks when it
  was allocated `2048/16 = 128`.
- **`gpu_memory_utilization: 0.90`, not 0.85.** The trainer still holds ~23 GB
  when the teacher restarts for the next round, and 0.85 leaves the teacher
  short from round 2 onward.
- **`kv_cache_dtype: bf16`, not fp8.** Quantization noise would propagate into
  the aux-layer labels the draft is trained against.
- **`max_num_batched_tokens >= max_model_len`, chunked prefill and prefix
  caching both off.** ATOM rewrites the same Mooncake key every scheduler step
  without checking `is_final_chunk`, so anything that splits a prefill silently
  keeps only the last chunk. Preflight enforces this.

## Layout

```
configs/
  train.yaml            # full run; every non-obvious value is explained inline
  smoke_test.yaml       # 5 cached batches
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
| `$DATA_ROOT/checkpoints/kimi_k3_dspark_atom/` | Checkpoints (not tmpfs, which is wiped when the allocation ends) |
| `/dev/shm/teacher_cache_atom/round_N/` | One round of cached hidden states |
| `/dev/shm/lumenrl_teacher_hidden/atom_teacher_worker.log` | Teacher worker log |
| `output/Kimi_K3_SDDD/LumenRL/` | Training log |

## Reference

- [Inferact/Kimi-K3-DSpark](https://huggingface.co/Inferact/Kimi-K3-DSpark) — draft model config
- [moonshotai/Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3) — target model
- [DSpark paper (arXiv:2607.05147)](https://arxiv.org/abs/2607.05147)
- [lightseekorg/kimi-mtp-dataset](https://huggingface.co/datasets/lightseekorg/kimi-mtp-dataset)
