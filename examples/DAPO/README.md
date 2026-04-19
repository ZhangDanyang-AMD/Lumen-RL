# DAPO RL Training Experiments

Reproduce VERL-style DAPO math RL training using LumenRL on AMD MI350X GPUs.
Tested with Qwen3-8B-Base (dense) on 8x MI350X (252 GiB VRAM per GPU).

## Table of Contents

- [Hardware Requirements](#hardware-requirements)
- [Environment Setup](#environment-setup)
- [Download Model and Data](#download-model-and-data)
- [Run Training](#run-training)
- [Monitor Training](#monitor-training)
- [Experiments](#experiments)
- [Configuration Reference](#configuration-reference)
- [ROCm Known Issues](#rocm-known-issues)
- [Dashboards](#dashboards)

## Hardware Requirements

| Resource | Minimum | Tested |
|----------|---------|--------|
| GPUs | 8x AMD MI300X or MI350X | 8x MI350X (252 GiB each) |
| CPU RAM | 256 GiB | 512 GiB |
| `/dev/shm` | 128 GiB | 256 GiB (used for model, data, checkpoints, optimizer offload) |
| Disk | 50 GiB | 100 GiB |
| ROCm | 6.4+ | 7.12 |

## Environment Setup

### Option A: Docker (recommended)

```bash
cd /path/to/Lumen-RL

# Build the container (includes all dependencies)
docker build -t lumenrl -f docker/Dockerfile .

# Launch with GPU access and shared memory
docker run -it --rm \
    --device /dev/kfd --device /dev/dri \
    --group-add video \
    -v /dev/shm:/dev/shm \
    --shm-size=256g \
    --network=host \
    --name lumenrl-train \
    lumenrl bash
```

### Option B: Existing container / bare metal

If you already have a ROCm + PyTorch environment, install LumenRL and its
submodules:

```bash
cd /path/to/Lumen-RL

# Install submodules
cd third_party/aiter && PREBUILD_KERNELS=1 pip install -e . && cd ../..
cd third_party/mori && pip install -e . && cd ../..
cd third_party/Lumen && pip install -e ".[dev]" && cd ../..
cd third_party/ATOM && pip install -e . && cd ../..

# Install LumenRL itself
pip install -e ".[test]"

# Additional RL dependencies
pip install datasets "math_verify[antlr4_13_2]"
```

Verify the installation:

```bash
python -c "import lumenrl; import torch; print(f'LumenRL OK, GPUs: {torch.cuda.device_count()}')"
```

## Download Model and Data

All assets must be placed on `/dev/shm` for fast I/O.

### 1. Model: Qwen3-8B-Base

```bash
# Install huggingface_hub if not present
pip install huggingface_hub

# Download to /dev/shm/model/qwen3-8b-base
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'Qwen/Qwen3-8B-Base',
    local_dir='/dev/shm/model/qwen3-8b-base',
    local_dir_use_symlinks=False,
)
print('Model downloaded to /dev/shm/model/qwen3-8b-base')
"
```

Alternatively, if the model is already on a network share or local disk:

```bash
mkdir -p /dev/shm/model
cp -r /path/to/qwen3-8b-base /dev/shm/model/qwen3-8b-base
```

### 2. Training Data: DAPO-Math-17k

```bash
mkdir -p /dev/shm/data

python -c "
from datasets import load_dataset
ds = load_dataset('BytedTsinghua/DAPO-Math-17k', split='train')
ds.to_parquet('/dev/shm/data/dapo-math-17k.parquet')
print(f'Saved {len(ds)} samples to /dev/shm/data/dapo-math-17k.parquet')
"
```

### 3. Validation Data: AIME-2024 (optional, for eval callback)

```bash
python -c "
from datasets import load_dataset
ds = load_dataset('HuggingFaceH4/aime-2024', split='train')
ds.to_parquet('/dev/shm/data/aime-2024.parquet')
print(f'Saved {len(ds)} samples to /dev/shm/data/aime-2024.parquet')
"
```

### 4. Verify all assets

```bash
ls -lh /dev/shm/model/qwen3-8b-base/config.json
ls -lh /dev/shm/data/dapo-math-17k.parquet
# Expected: config.json exists, parquet ~50-100 MB
```

## Run Training

### Experiment 1A-Sync: BF16 Sync On-Policy (recommended starting point)

```bash
cd /path/to/Lumen-RL

# Full 275-step run (~8-12 hours depending on crash frequency)
bash examples/DAPO/run_1a_bf16_sync.sh

# Quick smoke test (5 steps, ~15 min)
TOTAL_STEPS=5 bash examples/DAPO/run_1a_bf16_sync.sh
```

The script:
1. Validates model and data paths exist
2. Sets ROCm-specific environment variables
3. Launches `torchrun` with 8 GPUs
4. Automatically restarts from checkpoint on crash (up to 50 retries)
5. Logs to `output/DAPO/1a-bf16-sync/1a-bf16-sync.log`

### Experiment 1A-Async: BF16 Async Off-Policy

```bash
bash examples/DAPO/run_1a_bf16_async.sh
```

### Overridable Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `NUM_GPUS` | `8` | GPUs per node |
| `TOTAL_STEPS` | `275` | Training steps |
| `MASTER_PORT` | `29500` | Distributed port |
| `MAX_RETRIES` | `50` | Auto-restart retry limit (sync only) |
| `RETRY_DELAY` | `10` | Seconds between retries |

Example:

```bash
NUM_GPUS=4 TOTAL_STEPS=50 bash examples/DAPO/run_1a_bf16_sync.sh
```

## Monitor Training

### Log file

```bash
# Follow live output
tail -f output/DAPO/1a-bf16-sync/1a-bf16-sync.log

# Search for step completions
grep "step=" output/DAPO/1a-bf16-sync/1a-bf16-sync.log

# Check for crashes
grep -i "crash\|fault\|error\|Traceback" output/DAPO/1a-bf16-sync/1a-bf16-sync.log
```

### GPU utilization

```bash
watch -n 2 rocm-smi
```

### Checkpoints

```bash
ls -lht /dev/shm/ckpts/lumenrl-dapo/1a-bf16-sync/
# Shows checkpoint_0.pt, checkpoint_2.pt, etc.
# Only the 3 most recent are kept (save_total_limit=3)
```

## Experiments

### LumenRL Native Trainer (current)

| Script | ID | Mode | Precision | Status |
|--------|----|------|-----------|--------|
| `run_1a_bf16_sync.sh` | 1A-Sync | Sync on-policy | BF16 | tested |
| `run_1a_bf16_async.sh` | 1A-Async | Async off-policy | BF16 | tested |

### Legacy VERL-based (via `common.sh`)

These scripts use the VERL entry point and are preserved for reference.

| Script | ID | Training | Rollout | TIS | Status |
|--------|----|----------|---------|-----|--------|
| `run_1a_bf16_baseline.sh` | 1A | BF16 (FSDP2) | BF16 | -- | ready |
| `run_1b_fp8_rollout_tis.sh` | 1B | BF16 (FSDP2) | FP8 + TIS | token, C=2 | ready |
| `run_1d_fp8_e2e_blockwise.sh` | 1D | FP8 E2E (blockwise 128) | FP8 + TIS | token, C=2 | ready |
| `run_1f_fp8_e2e_per_tensor.sh` | 1F | FP8 E2E (per-tensor delayed) | FP8 + TIS | token, C=2 | ready |

See [FP8_TRAINING_ALIGNMENT_PLAN.md](FP8_TRAINING_ALIGNMENT_PLAN.md) for the
full experiment matrix and FP8 alignment methodology.

## Configuration Reference

### Sync Training Config (`configs/1a_bf16_sync.yaml`)

| Parameter | Value | Notes |
|-----------|-------|-------|
| `train_global_batch_size` | 128 | 8 prompts x 16 generations |
| `train_micro_batch_size` | 1 | Sequential forward/backward per micro-batch |
| `max_total_sequence_length` | 4096 | Prompt + response budget |
| `max_response_length` | 3072 | Max generated tokens |
| `gpu_memory_utilization` | 0.20 | ATOM/vLLM KV cache reservation |
| `max_model_len` | 4096 | vLLM model context length |
| `num_generations` | 16 | DAPO group size |
| `kl_coeff` | 0.0 | No KL penalty (DAPO default) |
| `clip_ratio_low` | 0.2 | Asymmetric clip lower bound |
| `clip_ratio_high` | 0.28 | Asymmetric clip upper bound |
| `save_steps` | 2 | Frequent checkpoints for crash recovery |
| `save_total_limit` | 3 | Prune old checkpoints |
| `resume` | true | Auto-resume from latest checkpoint |

### ROCm Environment Variables

Set automatically by the launch scripts:

| Variable | Value | Purpose |
|----------|-------|---------|
| `PYTHONUNBUFFERED` | `1` | Real-time log output |
| `TORCHDYNAMO_DISABLE` | `1` | Disable torch.compile (unstable on ROCm) |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | Reduce fragmentation |
| `HSA_DISABLE_FRAGMENT_ALLOCATOR` | `1` | Workaround for ROCm memory allocator bugs |
| `VLLM_USE_V1` | `1` | Use vLLM V1 engine |
| `VLLM_ENABLE_V1_MULTIPROCESSING` | `0` | Single-process vLLM (required for subprocess isolation) |
| `NCCL_TIMEOUT` | `7200` | 2-hour NCCL timeout for slow steps |

## ROCm Known Issues

### `hipMemSetAccess` Page Fault (MI350X / ROCm 7.12)

**Symptom**: Training crashes with `Memory access fault by GPU node-N ...
Write access to a read-only page`. Non-deterministic, typically occurs after
3-21 training steps.

**Root cause**: `hipMemSetAccess` is broken on ROCm 7.12 / MI350X
([rocm-systems#2516](https://github.com/ROCm/rocm-systems/issues/2516)).
Any GPU memory management operation touching the VMM layer can trigger the
fault. This affects vLLM model loading, `LLM.sleep()`/`wake_up()`, and even
normal generation after cumulative memory churn.

**Mitigation** (already implemented in the launch script):
- Auto-restart loop (`MAX_RETRIES=50`) with process cleanup between attempts
- Frequent checkpoints (`save_steps=2`) so each crash loses at most 1 step
- Persistent vLLM subprocess (no sleep/wake between steps)
- `HSA_DISABLE_FRAGMENT_ALLOCATOR=1` to reduce memory allocator contention

**If training gets permanently stuck** (crashing immediately on every retry):
```bash
# GPU state is corrupted; restart the container
docker restart lumenrl-train
# Then re-launch the training script (it auto-resumes from checkpoint)
bash examples/DAPO/run_1a_bf16_sync.sh
```

## Dashboards

Training dashboards are saved to `dashboards/DAPO/`:

| File | Description |
|------|-------------|
| `Qwen3-8B-Base_BF16_DAPO-Sync.html` | Sync training metrics (reward, loss, timing) |
| `Qwen3-8B-Base_BF16_DAPO-Async.html` | Async training metrics |

Open in a browser:
```bash
# From the host machine (if using Docker, copy first)
docker cp lumenrl-train:/workspace/Lumen-RL/dashboards/DAPO/ ./dashboards/
open dashboards/Qwen3-8B-Base_BF16_DAPO-Sync.html
```

## Directory Structure

```
examples/DAPO/
├── README.md                          # This file
├── FP8_TRAINING_ALIGNMENT_PLAN.md     # Full experiment design
├── configs/
│   ├── 1a_bf16_sync.yaml              # Sync on-policy (recommended)
│   ├── 1a_bf16_async.yaml             # Async off-policy
│   ├── 1a_bf16_baseline.yaml          # Legacy VERL baseline
│   └── smoke_test_native.yaml         # Quick smoke test
├── run_1a_bf16_sync.sh                # Launch sync training (auto-restart)
├── run_1a_bf16_async.sh               # Launch async training
├── run_1a_bf16_baseline.sh            # Legacy VERL baseline
├── common.sh                          # Shared config for legacy scripts
├── smoke_test.sh                      # Quick BF16 smoke test
└── smoke_test_fp8.sh                  # Quick FP8 smoke test
```

## Quick Reproduction Checklist

1. Build or enter the Docker container
2. Download Qwen3-8B-Base to `/dev/shm/model/qwen3-8b-base`
3. Download DAPO-Math-17k to `/dev/shm/data/dapo-math-17k.parquet`
4. Run `bash examples/DAPO/run_1a_bf16_sync.sh`
5. Monitor with `tail -f output/DAPO/1a-bf16-sync/1a-bf16-sync.log`
6. Expect reward accuracy to climb from ~10% (step 0) toward ~25-30% by step 275
7. If ROCm page faults occur, the script auto-restarts from checkpoint
8. If restarts loop forever, `docker restart` the container and re-run
