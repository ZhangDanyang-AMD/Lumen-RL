# Kimi K2.5 Eagle3 Draft Distillation (vLLM + BF16) — MI350

Train Eagle3 speculative decoding draft model using Kimi K2.5 teacher hidden states on **8x MI350 GPUs** with vLLM inference (BF16) and Mooncake TCP transfer.

- **GPUs 0-3**: torchrun FSDP2 — Eagle3 draft model training (BF16)
- **GPUs 4-7**: vLLM — Kimi K2.5 teacher (TP=4, BF16)

## Architecture

```
Training GPUs (0-3)                    Inference GPUs (4-7)
LumenRL FSDP2                  <---   vLLM (BF16, TP=4)
  Eagle3 draft model, BF16                    |
       ^                                      |
  Mooncake TCP  <--------------------  hidden_states via
  EagleMooncakeStore                  MooncakeHiddenStatesConnector
```

## Lumen + AITER Acceleration

[Lumen](../../third_party/Lumen) integrates [AITER](../../third_party/aiter) GPU kernels to replace PyTorch's default operators with hardware-optimized ROCm implementations. Controlled via `quantization.training` in YAML configs:

| Feature | Config Flag | Replaces | AITER Kernel | Status |
|---------|-------------|----------|--------------|--------|
| **lumen_linear** | `lumen_linear: true` | `nn.Linear` forward (BF16 GEMM) | `tuned_gemm.gemm_a16w16` — ASM/hipBLASLt/Triton dispatch chain with auto-tuning via `GemmTuner` | Disabled (MI350 backward race condition) |
| **lumen_norm** | `lumen_norm: true` | `RMSNorm`, `LayerNorm` | `aiter.ops.rmsnorm` / `aiter.ops.norm` — CK kernels with Triton fallback, supports fused FP8 quantization | Disabled (MI350 backward race condition) |
| **hf_attn_patch** | `hf_attn_patch: true` | `F.scaled_dot_product_attention` | `aiter.ops.attention` — CK attention kernels with Triton fallback | Disabled (MI350 backward race condition) |

All three features are currently disabled due to `HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION` crashes caused by ROCm async kernel race conditions on MI350 (gfx950). The workaround is `torch.cuda.synchronize()` between training phases; re-enabling AITER with sync points is planned.

## Two-Phase Training (lightseekorg recipe)

| Phase | Steps | Dataset | Description | Status |
|-------|-------|---------|-------------|--------|
| Phase 1 | 111,012 | perfectblend (296K samples, 3 epochs) | Foundation training | **In Progress** |
| Phase 2 | 67,826 | Mixed (181K: VL, CN, tool-call, agent, writing, 3 epochs) | Domain diversity | Not started |

Training data: [lightseekorg/kimi-mtp-dataset](https://huggingface.co/datasets/lightseekorg/kimi-mtp-dataset)

## Quick Start

### 1. Download Models & Data to /dev/shm

```bash
huggingface-cli download moonshotai/Kimi-K2.5 --local-dir /dev/shm/Kimi-K2.5-BF16
huggingface-cli download lightseekorg/kimi-mtp-dataset --local-dir /dev/shm/kimi-mtp-dataset
```

### 2. Build Docker Image

```bash
bash examples/Kimi_K25_SDDD_MI350_vLLM/docker/build.sh
```

### 3. Full Two-Phase Training

```bash
# Both phases (Phase 1 + Phase 2)
bash examples/Kimi_K25_SDDD_MI350_vLLM/run_full_training_docker.sh

# Phase 2 only (resume from Phase 1 checkpoint)
bash examples/Kimi_K25_SDDD_MI350_vLLM/run_full_training_docker.sh --phase2-only

# Smoke test (Qwen3-8B, single GPU, 2 steps, no vLLM/Mooncake)
CUDA_VISIBLE_DEVICES=0 python -m lumenrl.trainer.main \
    --config examples/Kimi_K25_SDDD_MI350_vLLM/configs/smoke_test_hf.yaml
```

### 4. Benchmark (after training)

```bash
bash examples/Kimi_K25_SDDD_MI350_vLLM/run_benchmark_vllm_docker.sh
```

## File Structure

```
configs/
  phase1_foundation.yaml   # Phase 1: perfectblend, 111K steps, from scratch
  phase2_mixed.yaml        # Phase 2: mixed domain, 68K steps, resume from Phase 1
  smoke_test_hf.yaml       # Quick validation: single GPU, HF backend, 2 steps
docker/
  Dockerfile               # ROCm + vLLM 0.19.1 + Mooncake + LumenRL
  build.sh                 # Docker build script
  patches/vllm/v0.19.1/    # vLLM extract_hidden_states patches
run_full_training.sh       # Bare-metal two-phase training entry point
run_full_training_docker.sh # Docker two-phase training entry point
run_benchmark_vllm_docker.sh # vLLM speculative decoding benchmark
bench_eagle3_vllm.py       # Benchmark script (accept_length measurement)
split_dataset.py           # Split kimi-mtp-dataset into Phase 1/2 subsets
```

## Output

| Path | Description |
|------|-------------|
| `/dev/shm/checkpoints/kimi_k25_eagle3_v2_phase1` | Phase 1 checkpoint |
| `/dev/shm/checkpoints/kimi_k25_eagle3_v2_phase2` | Phase 2 checkpoint |
| `dashboards/SDDD/Kimi_K25_SDDD_MI350/phase1.html` | Phase 1 training dashboard |
| `dashboards/SDDD/Kimi_K25_SDDD_MI350/phase2.html` | Phase 2 training dashboard |

## Reference

- [lightseekorg/kimi-k2.5-eagle3](https://huggingface.co/lightseekorg/kimi-k2.5-eagle3) — training recipe
- [KimiV2.5 Technical Report](https://arxiv.org/abs/2505.01759)
