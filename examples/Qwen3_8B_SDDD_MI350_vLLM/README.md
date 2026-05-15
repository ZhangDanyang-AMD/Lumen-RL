# Qwen3-8B Eagle3 Draft Distillation (vLLM + Mooncake TCP) — MI350

Train Eagle3 speculative decoding draft model using Qwen3-8B teacher hidden states on **8x MI350 GPUs** with vLLM inference and Mooncake TCP transfer.

- **GPUs 0-3**: torchrun FSDP2 -- Eagle3 draft model training (BF16, LumenRL + aiter)
- **GPUs 4-7**: vLLM -- Qwen3-8B teacher (TP=4)

## Architecture

```
Training GPUs (0-3)                    Inference GPUs (4-7)
LumenRL FSDP2 + aiter          <---   vLLM (extract_hidden_states)
  Eagle3 draft model, BF16              TP=4, BF16
       ^                                       |
  Mooncake TCP  <---------------------  hidden_states via
  EagleMooncakeStore                   MooncakeHiddenStatesConnector
```

## Quick Start

### 1. Download Model to /dev/shm

```bash
huggingface-cli download Qwen/Qwen3-8B --local-dir /dev/shm/Qwen3-8B
```

### 2. Build Docker Image

```bash
bash examples/Qwen3_8B_SDDD_MI350_vLLM/docker/build.sh
```

### 3. Smoke Test (2 steps)

```bash
# Docker (logs saved to output/Qwen3_8B_SDDD/LumenRL/)
bash examples/Qwen3_8B_SDDD_MI350_vLLM/run_docker.sh --smoke-test

# Or bare-metal (if dependencies installed)
bash examples/Qwen3_8B_SDDD_MI350_vLLM/run_qwen3_8b.sh --smoke-test
```

### 4. Full Training (500 steps)

```bash
bash examples/Qwen3_8B_SDDD_MI350_vLLM/run_docker.sh
```

## Output

Logs are saved to `output/Qwen3_8B_SDDD/LumenRL/`.

## Environment Variables

| Variable | Value | Purpose |
|----------|-------|---------|
| `CUDA_VISIBLE_DEVICES` | `0,1,2,3,4,5,6,7` | All 8 GPUs |
| `PYTORCH_ROCM_ARCH` | `gfx950` | MI350 GPU architecture |
| `HIP_FORCE_DEV_KERNARG` | `1` | Required for MI350 kernel args |
| `WANDB_MODE` | `disabled` | Disable W&B logging for smoke tests |
