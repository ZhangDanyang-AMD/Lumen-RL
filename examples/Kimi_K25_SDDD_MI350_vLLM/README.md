# Kimi K2.5 Eagle3 Draft Distillation (vLLM + ATOM + MXFP4) — MI350

Train Eagle3 speculative decoding draft model using Kimi K2.5 teacher hidden states on **8x MI350 GPUs** with vLLM + ATOM plugin inference (MXFP4 online quantization via AITER on ROCm) and Mooncake TCP transfer.

- **GPUs 0-3**: torchrun FSDP2 -- Eagle3 draft model training (BF16, LumenRL + aiter)
- **GPUs 4-7**: vLLM + ATOM plugin -- Kimi K2.5 teacher (TP=4, BF16->MXFP4 via aiter)

Both inference and training share the **same AITER version** from `third_party/aiter`.

## Architecture

```
Training GPUs (0-3)                    Inference GPUs (4-7)
LumenRL FSDP2 + aiter          <---   vLLM + ATOM plugin + aiter
  Eagle3 draft model, BF16              TP=4, BF16->MXFP4 (AITER)
       ^                                       |
  Mooncake TCP  <---------------------  hidden_states via
  EagleMooncakeStore                   MooncakeHiddenStatesConnector
```

## Software Stack (all from third_party/)

| Component | Source | Purpose |
|-----------|--------|---------|
| **aiter** | `third_party/aiter` | GPU compute kernels (shared by ATOM + LumenRL) |
| **ATOM** | `third_party/ATOM` | vLLM plugin, MXFP4 online quantization |
| **Lumen** | `third_party/Lumen` | Quantized training engine, aiter-patched FSDP2 |
| **mori** | `third_party/mori` | MoE expert-parallel all-to-all communication |
| vLLM | 0.19.1 + TorchSpec patches | `extract_hidden_states` + `MooncakeHiddenStatesConnector` |
| Mooncake | `f2853a80` (source-built HIP) | Hidden state transfer (TCP) |
| TorchSpec | `/home/danyzhan/TorchSpec` | vLLM patches + mooncake connectors |
| LumenRL | repo root | Training framework (`SpecDistillTrainer`) |

## Models

| Role | Model | Notes |
|------|-------|-------|
| **Teacher** | [moonshotai/Kimi-K2.5](https://huggingface.co/moonshotai/Kimi-K2.5) | 1T MoE, BF16, online MXFP4 via ATOM/AITER |
| **Draft** | Eagle3 (from scratch) | hidden=7168, 1 layer, 64 heads |

## Two-Phase Training (lightseekorg recipe)

| Phase | Steps | Dataset | Description |
|-------|-------|---------|-------------|
| Phase 1 | 0->20K | perfectblend (296K samples) | Foundation training |
| Phase 2 | 20K->40K | Mixed (181K: VL, CN, tool-call, agent, writing) | Domain diversity |

Training data: [lightseekorg/kimi-mtp-dataset](https://huggingface.co/datasets/lightseekorg/kimi-mtp-dataset)

## Quick Start

### 1. Download Models & Data to /dev/shm

```bash
# Qwen3-8B for smoke test
huggingface-cli download Qwen/Qwen3-8B --local-dir /dev/shm/Qwen3-8B

# Kimi K2.5 for full training
huggingface-cli download moonshotai/Kimi-K2.5 --local-dir /dev/shm/Kimi-K2.5-BF16

# Training dataset
huggingface-cli download lightseekorg/kimi-mtp-dataset --local-dir /dev/shm/kimi-mtp-dataset
```

### 2. Build Docker Image

```bash
bash examples/Kimi_K25_SDDD_MI350_vLLM/docker/build.sh
```

### 3. Smoke Test (Qwen3-8B, 2 steps)

```bash
# Docker
bash examples/Kimi_K25_SDDD_MI350_vLLM/run_full_training_docker.sh --smoke-test

# Or bare-metal (if dependencies installed)
bash examples/Kimi_K25_SDDD_MI350_vLLM/run_kimi_k25.sh --smoke-test
```

### 4. Full Two-Phase Training

```bash
bash examples/Kimi_K25_SDDD_MI350_vLLM/run_full_training_docker.sh
```

## Key Differences from SGLang Version

| Aspect | SGLang (`Kimi_K25_SDDD_MI350/`) | vLLM (`Kimi_K25_SDDD_MI350_vLLM/`) |
|--------|----------------------------------|-------------------------------------|
| Inference engine | SGLang + ATOM plugin | vLLM + ATOM plugin |
| ATOM registration | `SGLANG_EXTERNAL_MODEL_PACKAGE` env var | pip entry points (automatic) |
| Hidden state mode | `spec_training_mooncake` | `extract_hidden_states` |
| Transfer protocol | RDMA (mlx5_8) | TCP (no RDMA hardware) |
| Docker base | `rocm/sgl-dev:...` | `vllm/vllm-openai-rocm:v0.19.1` |

## Environment Variables

| Variable | Value | Purpose |
|----------|-------|---------|
| `CUDA_VISIBLE_DEVICES` | `0,1,2,3,4,5,6,7` | All 8 GPUs (do NOT set `HIP_VISIBLE_DEVICES`) |
| `PYTORCH_ROCM_ARCH` | `gfx950` | MI350 GPU architecture |
| `HIP_FORCE_DEV_KERNARG` | `1` | Required for MI350 kernel args |
| `WANDB_MODE` | `disabled` | Disable W&B logging for smoke tests |

## Reference

- [lightseekorg/kimi-k2.5-eagle3](https://huggingface.co/lightseekorg/kimi-k2.5-eagle3) -- training recipe
- [TorchSpec](https://github.com/lightseekorg/TorchSpec)
- [KimiV2.5 Technical Report](https://arxiv.org/abs/2505.01759)
