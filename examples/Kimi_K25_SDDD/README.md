# Kimi K2.5 Eagle3 Draft Distillation (SGLang + ATOM + MXFP4)

Train Eagle3 speculative decoding draft model using Kimi K2.5 teacher hidden states on **8 GPUs** with SGLang + ATOM plugin inference (MXFP4 online quantization on ROCm) and Mooncake RDMA transfer.

- **GPUs 0-3**: torchrun FSDP2 -- Eagle3 draft model training (BF16)
- **GPUs 4-7**: SGLang Engine + ATOM plugin -- Kimi K2.5 teacher (TP=4, BF16→MXFP4)

## Architecture

```
Training GPUs (0-3)                    Inference GPUs (4-7)
FSDP2 Eagle3 draft model       <---   SGLang Engine + ATOM plugin
  4 GPUs, BF16 training                  TP=4, BF16→MXFP4 (AITER)
       ^                                       |
  Mooncake RDMA  <--------------------  hidden_states to Mooncake
```

Key difference from the `KimiV2.5_Draft_Distill_MI300` example: this uses **SGLang + ATOM plugin** (batched scheduling, KV cache management) instead of direct ATOM engine + MORI-IO RDMA.

## Models

| Role | Model | Notes |
|------|-------|-------|
| **Teacher** | [moonshotai/Kimi-K2.5](https://huggingface.co/moonshotai/Kimi-K2.5) | 1T MoE, BF16, online MXFP4 via ATOM/AITER |
| **Draft** | Eagle3 (from scratch) | hidden=7168, 1 layer, 64 heads |

## Two-Phase Training (lightseekorg recipe)

| Phase | Steps | Dataset | Description |
|-------|-------|---------|-------------|
| Phase 1 | 0→20K | perfectblend (296K samples) | Foundation training |
| Phase 2 | 20K→40K | Mixed (181K: VL, CN, tool-call, agent, writing) | Domain diversity |

Training data: [lightseekorg/kimi-mtp-dataset](https://huggingface.co/datasets/lightseekorg/kimi-mtp-dataset)

## Requirements

- SGLang with TorchSpec spec_training patches
- ATOM (SGLang model plugin) with AITER MXFP4 kernels
- Mooncake Transfer Engine (RDMA)
- 8 AMD GPUs with ROCm (MI300X 192 GiB recommended)

## Quick Start

### Smoke Test (no external deps)

```bash
bash examples/Kimi_K25_SDDD/run_kimi_k25.sh --smoke-test
```

### Full Two-Phase Training (Docker)

```bash
bash examples/Kimi_K25_SDDD/run_full_training_docker.sh
```

### Custom Model Path

```bash
MODEL_PATH=/path/to/kimi-k2.5 bash examples/Kimi_K25_SDDD/run_kimi_k25.sh
```

## Monitoring

```bash
tail -f /datasets/checkpoints/kimi_k25_eagle3_phase1/phase1.log
```

## Reference

- [lightseekorg/kimi-k2.5-eagle3](https://huggingface.co/lightseekorg/kimi-k2.5-eagle3) -- training recipe
- [TorchSpec](https://github.com/lightseekorg/TorchSpec)
- [KimiV2.5 Technical Report](https://arxiv.org/abs/2505.01759)
