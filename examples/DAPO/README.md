# DAPO RL Training — FP8 Alignment Experiments

Reproduces VERL's FP8 training alignment benchmarks using LumenRL on AMD MI350X GPUs.

**Reference**: [FP8_TRAINING_ALIGNMENT_PLAN.md](../../third_party/Lumen/outputs/fp8_training_alignment/FP8_TRAINING_ALIGNMENT_PLAN.md) (in the Lumen submodule)

## Prerequisites

```bash
# Build and enter the LumenRL container
cd /path/to/Lumen-RL
docker build -t lumenrl -f docker/Dockerfile .
docker run --rm -it --device /dev/kfd --device /dev/dri \
    -v /dev/shm:/dev/shm \
    --shm-size=256g \
    lumenrl bash

# Or use an existing container
docker exec -it lumen_verl_test bash
cd /workspace/Lumen-RL/examples/DAPO
```

**Data & model** (must be on host `/dev/shm`):

| Asset | Path |
|-------|------|
| Qwen3-8B-Base | `/dev/shm/model/qwen3-8b-base` |
| DAPO-Math-17k (train) | `/dev/shm/data/dapo-math-17k.parquet` |
| AIME-2024 (val) | `/dev/shm/data/aime-2024.parquet` |

## Quick Start

```bash
# 1. Smoke test (2 steps, ~5 min)
bash smoke_test.sh

# 2. Smoke test FP8 (2 steps, verifies full Lumen FP8 stack)
bash smoke_test_fp8.sh

# 3. Run Experiment 1A baseline
bash run_1a_bf16_baseline.sh
```

## Experiments (Experiment 1: Qwen3-8B-Base Dense)

| Script | Run ID | Training | Rollout | TIS |
|--------|--------|----------|---------|-----|
| `run_1a_bf16_baseline.sh` | 1A | BF16 (FSDP2) | BF16 | — |
| `run_1b_fp8_rollout_tis.sh` | 1B | BF16 (FSDP2) | FP8 + TIS | token, C=2 |
| `run_1d_fp8_e2e_blockwise.sh` | 1D | FP8 E2E (blockwise 128) | FP8 + TIS | token, C=2 |
| `run_1f_fp8_e2e_per_tensor.sh` | 1F | FP8 E2E (per-tensor delayed) | FP8 + TIS | token, C=2 |

## Configuration

All scripts source `common.sh`. Key env vars you can override:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `/dev/shm/model/qwen3-8b-base` | HuggingFace model path |
| `GPU_MEM_UTIL` | `0.3` | vLLM `gpu_memory_utilization` |
| `TOTAL_STEPS` | `275` | Training steps |
| `TEST_FREQ` | `5` | Validation every N steps |
| `SAVE_FREQ` | `20` | Checkpoint save frequency |

## Output

Logs and results are written to `output/DAPO/`:

```
output/DAPO/
├── <EXP_NAME>.log          # Training log (tee'd from stdout)
├── metrics/                 # Parsed metrics for dashboard
└── dashboard.html           # Comparison dashboard (generated after runs)
```

Checkpoints: `/root/ckpts/${PROJECT_NAME}/${EXP_NAME}/`

## ROCm Adaptations

These are pre-configured in the scripts:

- **`free_cache_engine=True`**: Explicitly frees vLLM KV cache between rollout and training
- **`enforce_eager=True`**: CUDA graph capture disabled on ROCm
- **`gpu_memory_utilization=0.3`**: Low reservation since vLLM `sleep()` is a no-op on ROCm
- **`VLLM_USE_V1=1`**: Uses vLLM V1 engine architecture

## Patches

The `patches/` directory contains drop-in replacements for upstream VERL/vLLM modules
that add ROCm support, debug logging, and dynamic sampling. They are automatically
applied by `common.sh` at launch time.
