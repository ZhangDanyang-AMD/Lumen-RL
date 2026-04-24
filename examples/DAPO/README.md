# DAPO RL Training Experiments

Reproduce VERL-style DAPO math RL training using LumenRL on AMD MI300X / MI350X GPUs.
Tested with Qwen3-8B-Base (dense) on 8x MI300X (192 GiB VRAM per GPU) and 8x MI350X (252 GiB each).

## Table of Contents

- [Hardware Requirements](#hardware-requirements)
- [Environment Setup](#environment-setup)
- [Download Model and Data](#download-model-and-data)
- [Run Training](#run-training)
- [Monitor Training](#monitor-training)
- [Experiments](#experiments)
- [Configuration Reference](#configuration-reference)
- [Attention Backend (AITER)](#attention-backend-aiter)
- [ROCm Known Issues](#rocm-known-issues)
- [Dashboards](#dashboards)

## Hardware Requirements

| Resource | Minimum | Tested (MI300X) | Tested (MI350X) |
|----------|---------|-----------------|-----------------|
| GPUs | 8x AMD MI300X or MI350X | 8x MI300X (192 GiB each) | 8x MI350X (252 GiB each) |
| CPU RAM | 256 GiB | 512 GiB | 512 GiB |
| Disk | 100 GiB | 500 GiB | 256 GiB (or `/dev/shm` for tmpfs) |
| ROCm | 6.4+ | 6.4 | 7.12 |

## Environment Setup

### Option A: Docker (recommended)

```bash
cd /path/to/Lumen-RL

# Initialize submodules (required before build)
git submodule update --init --recursive

# Build the container (includes all dependencies)
docker build -t lumenrl -f docker/Dockerfile .

# Launch with GPU access — bind-mount your data directory
docker run -it --rm \
    --device /dev/kfd --device /dev/dri \
    --group-add video \
    -v /home/danyzhan:/home/danyzhan \
    -v /dev/shm:/dev/shm \
    --shm-size=256g \
    --network=host \
    --name lumenrl-train \
    lumenrl bash
```

### Option B: Existing container / bare metal

If you already have a ROCm + PyTorch environment, install LumenRL and its
submodules.  **Important**: ATOM and Lumen must share the same AITER install.

```bash
cd /path/to/Lumen-RL

# Initialize submodules
git submodule update --init --recursive

# Install AITER once (shared by both Lumen and ATOM)
cd third_party/aiter && PREBUILD_KERNELS=1 pip install -e . && cd ../..

# Symlink Lumen's third_party/aiter → the shared copy
rm -rf third_party/Lumen/third_party/aiter
ln -s "$(pwd)/third_party/aiter" third_party/Lumen/third_party/aiter

# Install remaining submodules
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

# Verify ATOM and Lumen share the same aiter
python -c "
import aiter, importlib.util
spec = importlib.util.find_spec('aiter')
print(f'aiter location: {spec.origin}')
"
```

## Download Model and Data

Default data path is `/home/danyzhan/` (override with `MODEL_DIR` / `DATA_DIR`
env vars in the launch script).  For MI350X setups with large `/dev/shm`, you
can place assets there for faster I/O.

### 1. Model: Qwen3-8B-Base

```bash
pip install huggingface_hub

# Download to /home/danyzhan/model/qwen3-8b-base
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'Qwen/Qwen3-8B-Base',
    local_dir='/home/danyzhan/model/qwen3-8b-base',
    local_dir_use_symlinks=False,
)
print('Model downloaded to /home/danyzhan/model/qwen3-8b-base')
"
```

Alternatively, if the model is already available locally:

```bash
mkdir -p /home/danyzhan/model
cp -r /path/to/qwen3-8b-base /home/danyzhan/model/qwen3-8b-base
```

### 2. Training Data: DAPO-Math-17k

```bash
mkdir -p /home/danyzhan/data

python -c "
from datasets import load_dataset
ds = load_dataset('BytedTsinghua/DAPO-Math-17k', split='train')
ds.to_parquet('/home/danyzhan/data/dapo-math-17k.parquet')
print(f'Saved {len(ds)} samples to /home/danyzhan/data/dapo-math-17k.parquet')
"
```

### 3. Validation Data: AIME-2024 (optional, for eval callback)

```bash
python -c "
from datasets import load_dataset
ds = load_dataset('HuggingFaceH4/aime-2024', split='train')
ds.to_parquet('/home/danyzhan/data/aime-2024.parquet')
print(f'Saved {len(ds)} samples to /home/danyzhan/data/aime-2024.parquet')
"
```

### 4. Verify all assets

```bash
ls -lh /home/danyzhan/model/qwen3-8b-base/config.json
ls -lh /home/danyzhan/data/dapo-math-17k.parquet
```

## Run Training

### Experiment 1A-Sync: BF16 Sync On-Policy (recommended starting point)

```bash
cd /path/to/Lumen-RL

# Full 275-step run
bash examples/DAPO/run_1a_bf16_sync.sh

# Quick smoke test (2 steps, tiny batch, ~5-10 min)
bash examples/DAPO/run_1a_bf16_sync.sh --smoke-test

# Dry run (mock ATOM generation, trains on synthetic data — for debugging)
bash examples/DAPO/run_1a_bf16_sync.sh --dry-run

# Override total steps from environment
TOTAL_STEPS=50 bash examples/DAPO/run_1a_bf16_sync.sh
```

The script:
1. Validates model and data paths exist
2. Sets ROCm-specific environment variables
3. Sets ATOM attention backend to `ROCM_AITER_FA` with automatic fallback to
   `ROCM_AITER_UNIFIED_ATTN` if the FA build is broken
4. Launches `torchrun` with 8 GPUs
5. Automatically restarts from checkpoint on crash (up to 50 retries)
6. Logs to `output/DAPO/1a-bf16-sync/1a-bf16-sync.log`

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
| `MODEL_DIR` | `/home/danyzhan/model` | Directory containing model checkpoints |
| `DATA_DIR` | `/home/danyzhan/data` | Directory containing training data |
| `CKPT_DIR` | `/home/danyzhan/ckpts/...` | Checkpoint save directory |
| `VLLM_ROCM_ATTN_BACKEND` | `ROCM_AITER_FA` | ATOM/vLLM attention backend |
| `VLLM_ROCM_ATTN_BACKEND_FALLBACK` | `ROCM_AITER_UNIFIED_ATTN` | Fallback backend |

Example:

```bash
NUM_GPUS=4 TOTAL_STEPS=50 bash examples/DAPO/run_1a_bf16_sync.sh

# Use /dev/shm for faster I/O (MI350X with large tmpfs)
MODEL_DIR=/dev/shm/model DATA_DIR=/dev/shm/data bash examples/DAPO/run_1a_bf16_sync.sh

# Force a specific attention backend
VLLM_ROCM_ATTN_BACKEND=ROCM_AITER_UNIFIED_ATTN bash examples/DAPO/run_1a_bf16_sync.sh
```

## Monitor Training

### Log file

```bash
# Follow live output
tail -f output/DAPO/1a-bf16-sync/1a-bf16-sync.log

# Step completions with all metrics
grep "callbacks" output/DAPO/1a-bf16-sync/1a-bf16-sync.log

# GPU memory at each phase
grep "GPU-MEM" output/DAPO/1a-bf16-sync/1a-bf16-sync.log

# Check for crashes
grep -i "crash\|fault\|error\|Traceback\|OOM" output/DAPO/1a-bf16-sync/1a-bf16-sync.log
```

### GPU utilization

```bash
watch -n 2 rocm-smi
# Expect ~99% GPU utilization during training phase
# Expect ~46% VRAM usage during training (with param_offload)
```

### Checkpoints

```bash
ls -lht /home/danyzhan/ckpts/lumenrl-dapo/1a-bf16-sync/
# Shows checkpoint_0.pt, checkpoint_5.pt, etc.
# Only the 3 most recent are kept (save_total_limit=3)
```

### Expected Timeline

With the Verl-aligned config (1536 batch, 20K response) on 8×MI300X:

| Phase | Duration | Notes |
|-------|----------|-------|
| Generation | ~25 min | ATOM generates 1536 sequences on GPU 0 |
| ATOM sleep + log-probs | ~10 min | Free GPU memory, compute reference log-probs |
| Training | ~56 min | 192 sequences/rank × 20K tokens, mini-batch=1 |
| Weight sync | ~1 min | Sync updated weights to ATOM |
| **Total per step** | **~93 min** | 275 steps ≈ 17 days |

For faster iteration, reduce `train_global_batch_size` (e.g. 512 = 32p×16g gives
~27 min/step) or use `--smoke-test` / `--dry-run` flags.

## Experiments

### LumenRL Native Trainer (current)

| Script | ID | Mode | Precision | Batch | max_resp | Status |
|--------|----|------|-----------|-------|----------|--------|
| `run_1a_bf16_sync.sh` | 1A-Sync | Sync on-policy | BF16 | 1536 (96p×16g) | 20K | tested (MI300X) |
| `run_1a_bf16_async.sh` | 1A-Async | Async off-policy | BF16 | varies | varies | tested |

### Training Results (1A-Sync, Qwen3-8B-Base, 8×MI300X)

**Run with Verl-aligned rollout batch (1536 sequences, 20K response):**

| Step | Accuracy | Reward | Grad Norm | Resp Len | Gen tok/s | Gen (s) | Train (s) | Step (s) |
|------|----------|--------|-----------|----------|-----------|---------|-----------|----------|
| 0 | 12.1% | -0.758 | 1.204 | 1198 | 1139 | 1414 | 3342 | 5400 |
| 1 | 13.3% | -0.734 | 1.578 | 1146 | 1238 | 1236 | 3378 | 5254 |
| 2 | 9.5% | -0.810 | 1.402 | 1300 | 1183 | 1495 | 3341 | 5480 |
| 3 | 10.9% | -0.781 | 1.102 | 1309 | 1098 | 1617 | 3334 | 5597 |
| 4 | 9.6% | -0.809 | 0.913 | 1291 | 1120 | 1575 | 3390 | 5610 |

- 0 NaN parameters, 0 errors/OOMs across all steps
- GPU memory stable: post_train free ≥110 GiB / 206 GiB
- Step time: ~93 min (gen ~25 min + train ~56 min + sync ~1 min)
- Training backend: FSDP2 + AITER Flash Attention (fwd: `fmha_v3_fwd`, bwd: `mha_bwd`)
- Generation backend: ATOM with `ROCM_AITER_FA`

### Comparison with Verl Reference

The Verl [FP8 RL documentation](https://github.com/verl-project/verl/blob/4b9c14f2c306d43f84eccf47315323e3f1ce1270/docs/advance/fp8.md#qwen3-8b-base-dense-model)
reports accuracy climbing from ~15% to ~40% in the first 50 steps on 8×H100.
Our configuration aligns the rollout batch size (1536) and response length (20K)
but differs in two areas:

| Parameter | Verl | LumenRL | Impact |
|-----------|------|---------|--------|
| GPU | 8×H100 | 8×MI300X | Different hardware, different kernel performance |
| Token-level TIS | C=2 | not implemented | Verl shows accuracy drops without TIS |
| Training backend | FSDP (PyTorch) | FSDP2 + Lumen AITER | Different attention kernels |
| Rollout backend | vLLM | ATOM (vLLM fork) | Compatible, AITER-based |

The remaining accuracy gap is primarily attributed to the absence of
**Token-level Truncated Importance Sampling (TIS)**, which Verl uses with C=2 to
correct distribution shift between the rollout and training policies. Verl's own
results show that removing TIS causes a measurable accuracy drop.

### Memory Optimizations for 20K Sequences on MI300X

Running 20K-token sequences on 192 GiB MI300X GPUs required several fixes:

1. **FSDP2 param_offload**: Offloads model parameters and optimizer states to CPU.
   Required `foreach=False` in AdamW to avoid DTensor device mismatch.
2. **ATOM sleep after generation**: Frees ~55 GiB of ATOM's GPU memory before training.
3. **Per-rank data sharding**: Each rank processes only `batch_size / world_size`
   sequences (192 of 1536), not the full batch.
4. **Padded-length mini-batching**: `_dynamic_mini_batches` uses padded sequence
   length (not actual token count) to prevent OOMs from multi-sequence mini-batches.
5. **AITER causal mask bypass**: Bypasses HuggingFace's redundant 4D causal
   attention mask to use AITER's native causal flag, avoiding SDPA fallback that
   fails with GQA.

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
| `train_global_batch_size` | 1536 | 96 prompts × 16 generations (matches Verl 32×3×16) |
| `train_micro_batch_size` | 1 | Sequential forward/backward per micro-batch |
| `max_total_sequence_length` | 20480 | Prompt + response budget (20K tokens) |
| `max_response_length` | 20480 | Max generated tokens (matches Verl 20K) |
| `max_token_len_per_gpu` | 20480 | Token budget for dynamic mini-batching |
| `gpu_memory_utilization` | 0.25 | ATOM/vLLM KV cache reservation |
| `max_model_len` | 20480 | vLLM model context length |
| `param_offload` | true | FSDP2 CPU offload for 20K-token training |
| `num_generations` | 16 | DAPO group size |
| `kl_coeff` | 0.0 | No KL penalty (DAPO default) |
| `clip_ratio_low` | 0.2 | Asymmetric clip lower bound |
| `clip_ratio_high` | 0.28 | Asymmetric clip upper bound |
| `dynamic_sampling` | true | DAPO dynamic sampling |
| `token_level_pg` | true | Token-level policy gradient |
| `overlong_reward_shaping` | true | Penalize overlong responses |
| `save_steps` | 5 | Checkpoint frequency |
| `save_total_limit` | 3 | Prune old checkpoints |
| `resume` | false | Set true to resume from checkpoint |

#### Rollout Batch Size Alignment with Verl

The Verl reference uses `Rollout batch size: 32×3×16 = 1536`, meaning 32 prompts
accumulated over 3 rounds, each with 16 generations. LumenRL does not have a
"rounds" multiplier; instead, `train_global_batch_size` directly controls the
total rollout count:

```
train_global_batch_size = num_prompts × num_generations
1536                    = 96          × 16
```

This produces the same 1536 rollout sequences per step as Verl.

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

## Attention Backend (AITER)

Both ATOM (generation) and Lumen (training) use the AITER library for attention
on ROCm. ATOM's backend is selected via `VLLM_ROCM_ATTN_BACKEND`; Lumen's
training backend is patched automatically via `lumen.ops.attention.hf_patch`.

### Generation Backends (ATOM/vLLM)

| Backend | Env Value | Description | GPU Support |
|---------|-----------|-------------|-------------|
| **AITER FA** | `ROCM_AITER_FA` | CK/ASM Flash Attention v3. Highest throughput. | MI300X, MI350X |
| **AITER Unified** | `ROCM_AITER_UNIFIED_ATTN` | AITER's unified Triton attention kernel. Stable fallback. | MI300X, MI350X |
| ROCm Attn | `ROCM_ATTN` | Triton paged attention (default ROCm path). | MI300X, MI350X |

### Training Backend (Lumen AITER Patch)

During training, HuggingFace's default SDPA attention is replaced by AITER
kernels via Lumen's HF attention patch (`lumen.ops.attention.hf_patch`):

| Kernel | Operation | AITER Function |
|--------|-----------|----------------|
| Forward | Flash Attention forward | `fmha_v3_fwd` (CK/ASM) |
| Backward | Flash Attention backward | `mha_bwd` (CK/ASM) |

The patch is applied automatically when the FSDP2 backend loads the model.
It handles GQA (Grouped Query Attention) natively — no `repeat_kv` needed.

### Automatic Fallback

The launch scripts (`run_1a_bf16_sync.sh`) implement automatic fallback:

1. Start with `ROCM_AITER_FA` (best performance)
2. If the process crashes with `fmha_fwd_v3` / `undefined symbol` errors
   (FA build broken due to missing gfx950 kernels), automatically retry
   with `ROCM_AITER_UNIFIED_ATTN`

Override manually:

```bash
VLLM_ROCM_ATTN_BACKEND=ROCM_AITER_UNIFIED_ATTN bash examples/DAPO/run_1a_bf16_sync.sh
```

### Shared AITER Build

Both ATOM (inference) and Lumen (training) depend on AITER.  The Docker build
and bare-metal install ensure they share the **same** AITER installation from
`third_party/aiter`.  Lumen's own `third_party/aiter` is symlinked to the
shared copy to prevent version conflicts.

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

Training dashboards are saved to `dashboards/` at the repo root, organized by
run configuration (see the [lumenrl-training skill](../../.cursor/skills-cursor/lumenrl-training/SKILL.md)
for the naming convention):

| Directory | Description |
|-----------|-------------|
| `dashboards/1a-bf16-sync/` | 1A BF16 sync training — live metrics (accuracy, reward, timing, throughput) |

Open in a browser:
```bash
# Serve locally
cd dashboards/1a-bf16-sync && python3 -m http.server 8765
# Open http://localhost:8765/dashboard.html
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

1. Build or enter the Docker container (submodules must be initialized first)
2. Download Qwen3-8B-Base to `/home/danyzhan/model/qwen3-8b-base`
3. Download DAPO-Math-17k to `/home/danyzhan/data/dapo-math-17k.parquet`
4. Validate setup: `bash examples/DAPO/run_1a_bf16_sync.sh --smoke-test`
5. Launch full run: `bash examples/DAPO/run_1a_bf16_sync.sh`
6. Monitor with `grep "callbacks" output/DAPO/1a-bf16-sync/1a-bf16-sync.log`
7. Expect reward accuracy starting at ~10-13% (step 0), climbing over subsequent steps
8. Script tries `ROCM_AITER_FA` first; falls back to `ROCM_AITER_UNIFIED_ATTN` on MI300X
9. If ROCm page faults occur, the script auto-restarts from checkpoint
10. If restarts loop forever, `docker restart` the container and re-run

### Debugging Aids

```bash
# Dry run — no ATOM, mock generation, tests training loop only (~5 min/step)
bash examples/DAPO/run_1a_bf16_sync.sh --dry-run

# Smoke test — real ATOM but tiny batch (2 steps, ~10 min total)
bash examples/DAPO/run_1a_bf16_sync.sh --smoke-test

# Smaller batch for faster iteration (~27 min/step instead of ~93 min)
# Edit configs/1a_bf16_sync.yaml:
#   train_global_batch_size: 512    # 32 prompts × 16 gens
```
