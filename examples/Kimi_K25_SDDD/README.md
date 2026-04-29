# Kimi K2.5 Eagle3 Draft Distillation (SGLang + ATOM + NVFP4)

Train Eagle3 speculative decoding draft model using Kimi K2.5 teacher hidden states on **8 GPUs** with SGLang + ATOM plugin inference and Mooncake RDMA transfer.

- **GPUs 0-3**: torchrun FSDP2 -- Eagle3 draft model training (BF16)
- **GPUs 4-7**: SGLang Engine + ATOM plugin -- Kimi K2.5 teacher (TP=4, NVFP4)

## Architecture

```
Training GPUs (0-3)                    Inference GPUs (4-7)
FSDP2 Eagle3 draft model       <---   SGLang Engine + ATOM plugin
  4 GPUs, BF16 training                  TP=4, NVFP4 quantization
       ^                                       |
  Mooncake RDMA  <--------------------  hidden_states to Mooncake
```

Key difference from the `KimiV2.5_Draft_Distill_MI300` example: this uses **SGLang + ATOM plugin** (batched scheduling, KV cache management) instead of direct ATOM engine + MORI-IO RDMA.

## Models

| Role | Model | Notes |
|------|-------|-------|
| **Teacher** | [nvidia/Kimi-K2.5-NVFP4](https://huggingface.co/nvidia/Kimi-K2.5-NVFP4) | 1T MoE, NVFP4 quantized, ~125 GB/GPU with TP=4 |
| **Draft** | Eagle3 (from scratch) | hidden=7168, 1 layer, 64 heads |

## Requirements

- SGLang with TorchSpec spec_training patches
- ATOM (SGLang model plugin)
- Mooncake Transfer Engine (RDMA)
- 8 GPUs with sufficient memory (MI300X 192 GiB recommended)

## Quick Start

### Smoke Test (no external deps)

```bash
bash examples/Kimi_K25_SDDD/run_kimi_k25.sh --smoke-test
```

### Full Training

```bash
bash examples/Kimi_K25_SDDD/run_kimi_k25.sh
```

### Custom Model Path

```bash
MODEL_PATH=/path/to/kimi-k2.5-nvfp4 bash examples/Kimi_K25_SDDD/run_kimi_k25.sh
```

## Monitoring

```bash
tail -f output/Kimi_K25_SDDD/kimi-k25-eagle3-sglang-nvfp4/kimi-k25-eagle3-sglang-nvfp4.log
```

## Reference

- [TorchSpec](https://github.com/lightseekorg/TorchSpec) -- config: `sglang_kimi_k25_nvfp4.yaml`
- [KimiV2.5 Technical Report](https://arxiv.org/abs/2505.01759)
