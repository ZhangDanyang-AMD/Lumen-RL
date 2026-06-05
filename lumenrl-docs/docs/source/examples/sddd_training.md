# SDDD Training

This guide walks through training Eagle3 speculative decoding draft models using LumenRL's SDDD pipeline.

## Kimi K2.5 Eagle3 (SGLang + Mooncake)

Train an Eagle3 draft model for Kimi K2.5 on 8 GPUs with SGLang + ATOM inference and Mooncake RDMA transfer.

### GPU Layout

- **GPUs 0-3**: FSDP2 Eagle3 draft model training (BF16)
- **GPUs 4-7**: SGLang + ATOM plugin — Kimi K2.5 teacher (TP=4, BF16 to MXFP4)

### Quick Start

```bash
# Smoke test
bash examples/Kimi_K25_SDDD/run_kimi_k25.sh --smoke-test

# Full two-phase training (Docker)
bash examples/Kimi_K25_SDDD/run_full_training_docker.sh

# Custom model path
MODEL_PATH=/path/to/kimi-k2.5 bash examples/Kimi_K25_SDDD/run_kimi_k25.sh
```

### Models

| Role | Model | Notes |
|------|-------|-------|
| **Teacher** | [moonshotai/Kimi-K2.5](https://huggingface.co/moonshotai/Kimi-K2.5) | 1T MoE, BF16, online MXFP4 via ATOM/AITER |
| **Draft** | Eagle3 (from scratch) | hidden=7168, 1 layer, 64 heads |

## Qwen3-8B Eagle3 (vLLM + MORI)

Train an Eagle3 draft model for Qwen3-8B using vLLM inference backend with MORI RDMA transfer on MI350.

### Quick Start

```bash
# See the example directory for full scripts
ls examples/Qwen3_8B_SDDD_MI350_vLLM/
```

## MI350-Specific Examples

LumenRL includes MI350-optimized SDDD configurations:

- `examples/Kimi_K25_SDDD_MI350/` — Kimi K2.5 on MI350 with SGLang
- `examples/Kimi_K25_SDDD_MI350_ATOM/` — Kimi K2.5 on MI350 with ATOM
- `examples/Qwen3_8B_SDDD_MI350_vLLM/` — Qwen3-8B on MI350 with vLLM

## Monitoring

```bash
# Watch training logs
tail -f /datasets/checkpoints/kimi_k25_eagle3_phase1/phase1.log
```

## References

- [lightseekorg/kimi-k2.5-eagle3](https://huggingface.co/lightseekorg/kimi-k2.5-eagle3) — Training recipe
- [KimiV2.5 Technical Report](https://arxiv.org/abs/2505.01759)

For architecture details and configuration options, see {doc}`/advance/sddd`.
