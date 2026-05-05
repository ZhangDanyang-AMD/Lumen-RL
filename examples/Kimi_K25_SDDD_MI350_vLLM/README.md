# Kimi K2.5 Eagle3 Draft Distillation (vLLM + BF16) — MI350

Train Eagle3 speculative decoding draft model using Kimi K2.5 teacher hidden states on **8x MI350 GPUs** with vLLM inference (BF16) and Mooncake TCP transfer.

- **GPUs 0-3**: torchrun FSDP2 -- Eagle3 draft model training (BF16, LumenRL + aiter)
- **GPUs 4-7**: vLLM -- Kimi K2.5 teacher (TP=4, BF16)

## Architecture

```
Training GPUs (0-3)                    Inference GPUs (4-7)
LumenRL FSDP2 + aiter          <---   vLLM (BF16)
  Eagle3 draft model, BF16              TP=4, BF16
       ^                                       |
  Mooncake TCP  <---------------------  hidden_states via
  EagleMooncakeStore                   MooncakeHiddenStatesConnector
```

## Software Stack (all from third_party/)

| Component | Source | Purpose |
|-----------|--------|---------|
| **Lumen** | `third_party/Lumen` | FSDP2 + aiter integration for training |
| **aiter** | `third_party/aiter` | GPU compute kernels (used by Lumen) |
| vLLM | 0.19.1 + TorchSpec patches | `extract_hidden_states` + `MooncakeHiddenStatesConnector` |
| Mooncake | `f2853a80` (source-built HIP) | Hidden state transfer (TCP) |
| TorchSpec | `/home/danyzhan/TorchSpec` | vLLM patches + mooncake connectors |
| LumenRL | repo root | Training framework (`SpecDistillTrainer`) |

## Models

| Role | Model | Notes |
|------|-------|-------|
| **Teacher** | [moonshotai/Kimi-K2.5](https://huggingface.co/moonshotai/Kimi-K2.5) | 1T MoE, BF16 |
| **Draft** | Eagle3 (from scratch) | hidden=7168, 1 layer, 64 heads |

## Two-Phase Training (lightseekorg recipe)

| Phase | Steps | Dataset | Description | Status |
|-------|-------|---------|-------------|--------|
| Phase 1 | 0→111K | perfectblend (296K samples, 3 epochs) | Foundation training | **Completed** |
| Phase 2 | 0→67,826 | Mixed (181K: VL, CN, tool-call, agent, writing, 3 epochs) | Domain diversity | **Completed** |

Training data: [lightseekorg/kimi-mtp-dataset](https://huggingface.co/datasets/lightseekorg/kimi-mtp-dataset)

### Training Results

- **Total training time**: ~18h 15m (Phase 2)
- **Final loss**: 1.5e-6
- **Final accuracy**: 100% at all 4 speculative positions
- **Avg step time**: ~420 ms
- **Exported model**: `output/Kimi_K25_eagle3_HF/` (2.92B params, BF16, safetensors)
- **Dashboard**: `dashboards/SDDD/Kimi_K25_SDDD_MI350/phase2.html`

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

## Environment Variables

| Variable | Value | Purpose |
|----------|-------|---------|
| `CUDA_VISIBLE_DEVICES` | `0,1,2,3,4,5,6,7` | All 8 GPUs (do NOT set `HIP_VISIBLE_DEVICES`) |
| `PYTORCH_ROCM_ARCH` | `gfx950` | MI350 GPU architecture |
| `HIP_FORCE_DEV_KERNARG` | `1` | Required for MI350 kernel args |
| `WANDB_MODE` | `disabled` | Disable W&B logging for smoke tests |

## Output Files

| Path | Description |
|------|-------------|
| `output/Kimi_K25_eagle3_HF/` | Exported Eagle3 model (HuggingFace safetensors format) |
| `dashboards/SDDD/Kimi_K25_SDDD_MI350/phase1.html` | Phase 1 training dashboard |
| `dashboards/SDDD/Kimi_K25_SDDD_MI350/phase2.html` | Phase 2 training dashboard |

### Exported Model Architecture

The exported model at `output/Kimi_K25_eagle3_HF/` differs from [lightseekorg/kimi-k2.5-eagle3](https://huggingface.co/lightseekorg/kimi-k2.5-eagle3):

| Config | Ours | lightseekorg |
|--------|------|-------------|
| `attention_bias` | `true` | `false` |
| `norm_bias` | `true` | `false` |
| `rope_theta` | 50000.0 | 1000000 |
| `rope_scaling` | yarn (matching Kimi-K2.5 base) | none |
| `rms_norm_eps` | 1e-5 | 1e-6 |

## Reference

- [lightseekorg/kimi-k2.5-eagle3](https://huggingface.co/lightseekorg/kimi-k2.5-eagle3) -- training recipe
- [TorchSpec](https://github.com/lightseekorg/TorchSpec)
- [KimiV2.5 Technical Report](https://arxiv.org/abs/2505.01759)
