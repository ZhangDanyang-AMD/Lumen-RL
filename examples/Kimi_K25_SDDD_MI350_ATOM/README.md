# Kimi K2.5 Eagle3 Draft Distillation (ATOM MXFP4 + FSDP2) — MI350

Train Eagle3 speculative decoding draft model using Kimi K2.5 teacher hidden states on **8x MI350 GPUs** with ATOM inference (MXFP4 online quantization) and Mooncake TCP transfer.

- **GPUs 0-3**: torchrun FSDP2 — Eagle3 draft model training (BF16)
- **GPUs 4-7**: ATOM (vLLM plugin) — Kimi K2.5 teacher (TP=4, MXFP4, AITER kernels)

## Architecture

```
Training GPUs (0-3)                    Inference GPUs (4-7)
LumenRL FSDP2                  <---   ATOM + vLLM (MXFP4, TP=4)
  Eagle3 draft model, BF16              AITER gfx950 kernels
       ^                                      |
  Mooncake TCP  <--------------------  hidden_states via
  EagleMooncakeStore                  MooncakeHiddenStatesConnector
```

### Inference Stack (ATOM MXFP4)

ATOM is an AMD-optimized vLLM plugin that performs online MXFP4 quantization at inference time. It uses AITER's `tuned_gemm` dispatch chain (ASM → hipBLASLt → Triton) with gfx950-specific tuned configs for W4A16 MOE and dense GEMM. The teacher model is loaded in BF16 and quantized on-the-fly — no pre-quantized weights needed.

| Component | Role |
|-----------|------|
| **ATOM** (`third_party/ATOM`) | vLLM AsyncLLMEngine plugin: online MXFP4 quantization, FP8 KV cache, PagedAttention |
| **AITER** (`third_party/aiter`) | GPU kernel library: tuned GEMM (gfx950 ASM), CK attention, Triton MoE, RMSNorm |
| **Mooncake** | TCP-based hidden state transfer (trainer ↔ teacher), source-built for HIP |

### Training Stack (LumenRL FSDP2)

| Component | Role |
|-----------|------|
| **LumenRL** | FSDP2 distributed training with gradient accumulation, checkpoint management |
| **Lumen** (`third_party/Lumen`) | AITER-patched HuggingFace model: replaces `nn.Linear`, `RMSNorm`, `SDPA` with AITER kernels |
| **Eagle3** | 1-layer Transformer draft model with 3 auxiliary hidden state inputs from teacher layers [1, N/2-1, N-4] |

## Data Processing Pipeline

Aligned with [TorchSpec](https://github.com/LightSeek/TorchSpec) for identical tokenization and loss masking:

| Feature | Description |
|---------|-------------|
| **KimiK25Parser** | Manual Kimi-K2.5 chat template with special tokens (`<\|im_assistant\|>`, `<\|im_middle\|>`, `<\|im_end\|>`) |
| **Assistant-only loss mask** | Loss computed only on assistant response tokens via encode-prefix character mapping |
| **`<think>` handling** | Strips `<think>` from non-last assistant turns; injects `<think></think>` if missing |
| **`last_turn_loss_only=auto`** | Auto-detects thinking content and restricts loss to last assistant turn |
| **Packed loss mask** | Run-length encoded for efficient storage/transfer through SHM pipeline |
| **Multiprocess preprocessing** | 16-worker parallel tokenization with `.pt` caching |

### Loss Function

Forward KL divergence with position decay:

```
L = Σ_i (0.8^i) · KL(teacher_i || draft_i)    i ∈ {0, 1, 2, 3}
```

where `teacher_i` and `draft_i` are next-token distributions at speculative position `i`, and the loss mask restricts the sum to assistant-only tokens.

## Lumen + AITER Acceleration

[Lumen](../../third_party/Lumen) integrates [AITER](../../third_party/aiter) GPU kernels to replace PyTorch's default operators with hardware-optimized ROCm implementations. Controlled via `quantization.training` in YAML configs:

| Feature | Config Flag | Replaces | AITER Kernel |
|---------|-------------|----------|--------------|
| **lumen_linear** | `lumen_linear: true` | `nn.Linear` forward (BF16 GEMM) | `tuned_gemm.gemm_a16w16` — ASM/hipBLASLt/Triton dispatch chain with auto-tuning via `GemmTuner` |
| **lumen_norm** | `lumen_norm: true` | `RMSNorm`, `LayerNorm` | `aiter.ops.rmsnorm` / `aiter.ops.norm` — CK kernels with Triton fallback |
| **hf_attn_patch** | `hf_attn_patch: true` | `F.scaled_dot_product_attention` | `aiter.ops.attention` — CK attention kernels with Triton fallback |

## Two-Phase Training (lightseekorg recipe)

| Phase | Steps | Dataset | Description |
|-------|-------|---------|-------------|
| Phase 1 | 20,000 | perfectblend (296K samples) | Foundation training, from scratch |
| Phase 2 | 20,000 | Mixed (181K: VL, CN, tool-call, agent, writing) | Domain diversity, resume from Phase 1 |

Training data: [lightseekorg/kimi-mtp-dataset](https://huggingface.co/datasets/lightseekorg/kimi-mtp-dataset)

## Quick Start

### 1. Download Models & Data to /dev/shm

```bash
# Teacher model (BF16 weights, tokenized on-the-fly to MXFP4 by ATOM)
huggingface-cli download moonshotai/Kimi-K2.5 --local-dir /dev/shm/Kimi-K2.5-BF16

# Training dataset
huggingface-cli download lightseekorg/kimi-mtp-dataset --local-dir /dev/shm/kimi-mtp-dataset
```

### 2. Build Docker Image

```bash
bash examples/Kimi_K25_SDDD_MI350_ATOM/docker/build.sh
```

The Docker image bundles: vLLM 0.19.1 (patched for `extract_hidden_states`), ATOM, AITER, LumenRL, Lumen, Mooncake (source-built HIP), MORI (MoE all-to-all).

### 3. Full Two-Phase Training

```bash
# Both phases (Phase 1 + Phase 2)
bash examples/Kimi_K25_SDDD_MI350_ATOM/run_full_training_docker.sh

# Phase 2 only (resume from Phase 1 checkpoint)
bash examples/Kimi_K25_SDDD_MI350_ATOM/run_full_training_docker.sh --phase2-only

# Smoke test (Qwen3-8B, single GPU, 2 steps, no vLLM/Mooncake)
CUDA_VISIBLE_DEVICES=0 python -m lumenrl.trainer.main \
    --config examples/Kimi_K25_SDDD_MI350_ATOM/configs/smoke_test_hf.yaml
```

### 4. Benchmark (after training)

```bash
bash examples/Kimi_K25_SDDD_MI350_ATOM/run_benchmark_vllm_docker.sh
```

## Configuration

Key YAML config sections (`configs/phase1_foundation.yaml`):

```yaml
# Data processing (aligned with TorchSpec)
dataset:
  chat_template: kimi-k25           # KimiK25Parser for tokenization + loss mask
  last_turn_loss_only: false         # "true", "false", or "auto"
  min_loss_tokens: 0                 # Skip samples with too few supervised tokens
  num_preprocess_workers: 16
  cache_dir: /dev/shm/lumenrl_cache

# ATOM teacher inference
algorithm:
  teacher:
    inference_backend: atom
    tensor_parallel_size: 4
    gpu_ids: [4, 5, 6, 7]
    transport: mooncake
    atom:
      kv_cache_dtype: fp8
      gpu_memory_utilization: 0.9

# Eagle3 draft model
  spec_distill:
    draft_type: eagle3
    loss_type: forward_kl
    position_decay: 0.8
    spec_length: 4
```

## File Structure

```
configs/
  phase1_foundation.yaml   # Phase 1: perfectblend, 20K steps, from scratch
  phase2_mixed.yaml        # Phase 2: mixed domain, 20K steps, resume from Phase 1
  smoke_test_hf.yaml       # Quick validation: single GPU, HF backend, 2 steps
docker/
  Dockerfile               # ROCm + vLLM 0.19.1 + ATOM + AITER + Mooncake + LumenRL
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
- [TorchSpec](https://github.com/LightSeek/TorchSpec) — data processing reference implementation
