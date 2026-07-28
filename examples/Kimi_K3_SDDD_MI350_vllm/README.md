# Kimi K3 DSpark Draft Distillation (vLLM + FSDP2) — MI350

Train DSpark speculative decoding draft model using Kimi K3 teacher hidden states on **8× MI350 GPUs** with vLLM inference and Mooncake TCP transfer.

**Key difference from K2.5/GPT-OSS recipes:** K3's ~1TB MoE model (93 layers, 896 experts) requires TP=8 for inference. This forces **8-GPU sequential mode** (all GPUs for inference, then all GPUs for training) instead of the 4+4 disaggregated split.

- **Phase A**: GPUs 0-7 — vLLM teacher prefill (TP=8, MXFP4 MoE, FP8 KV cache)
- **Phase B**: GPUs 0-7 — torchrun replicate DSpark draft training (BF16)

## Architecture

```
Phase A: Prefill (all 8 GPUs)       Phase B: Training (all 8 GPUs)
vLLM TP=8 (MXFP4 MoE)              LumenRL replicate (BF16)
  Kimi-K3 teacher                     DSpark draft model
       |                                   ^
  extract_hidden_states              Load from /dev/shm
  at layers [2,23,47,71,89]         (mmap SHM files)
       |                                   |
  Mooncake TCP → /dev/shm  ────>    Train: CE + TV + BCE loss
  (SHM file cache)                  Markov head + confidence head
```

### DSpark Draft Model (vs Eagle3)

| Feature | DSpark (K3) | Eagle3 (GPT-OSS/K2.5) |
|---------|-------------|----------------------|
| Backbone | 5 layers, parallel | 1 layer, autoregressive |
| Target layers | 5: [2,23,47,71,89] | 3: [early,mid,late] |
| Block size | 7 tokens | N/A (TTT) |
| Markov head | VanillaMarkov (rank=256) | None |
| Confidence head | AcceptRatePredictor | None |
| Loss | CE(0.1) + TV(0.9) + BCE(1.0) | Forward KL |
| Anchors per seq | 512 | N/A |

## Quick Start

### 1. Download Models & Data

```bash
# Teacher model (~500GB with MXFP4 quantization)
huggingface-cli download moonshotai/Kimi-K3 --local-dir /dev/shm/Kimi-K3

# Training dataset
huggingface-cli download lightseekorg/kimi-mtp-dataset --local-dir /dev/shm/kimi-mtp-dataset
```

### 2. Build Docker Image

```bash
bash examples/Kimi_K3_SDDD_MI350_vllm/docker/build.sh
```

### 3. Training

```bash
# Full training (Docker)
bash examples/Kimi_K3_SDDD_MI350_vllm/run_docker.sh

# With auto-retry (absorbs ROCm failures)
bash examples/Kimi_K3_SDDD_MI350_vllm/run_with_retry.sh

# Smoke test (5 steps, validates full pipeline)
bash examples/Kimi_K3_SDDD_MI350_vllm/run_docker.sh --smoke-test
```

## Configuration

Key YAML config sections (`configs/train.yaml`):

```yaml
# Batch-alternating 8-GPU sequential mode
cluster:
  gpus_per_node: 8

# DSpark-specific algorithm config
algorithm:
  spec_distill:
    draft_type: dspark
    loss_type: dspark
    sequential_mode: batch_alternating
    cache_batches: 50
    loss_decay_gamma: 4.0
    num_target_layers: 5
    spec_length: 7
    aux_hidden_state_layer_ids: [2, 23, 47, 71, 89]
    anchor_num: 512

  teacher:
    inference_backend: vllm
    tensor_parallel_size: 8
    gpu_ids: [0, 1, 2, 3, 4, 5, 6, 7]
    generate_mode: prefill
    transport: mooncake

  draft:
    num_layers: 5
    num_heads: 64
    ffn_dim: 14336
    rope_theta: 50000.0
    rope_scaling_type: yarn
    markov_rank: 256
    enable_confidence_head: true
```

## File Structure

```
configs/
  train.yaml          # Full training: 8-GPU sequential, 2 epochs
  smoke_test.yaml     # Quick validation: 5 steps
docker/
  build.sh            # Docker build script
  patches/vllm/v0.19.1/  # vLLM extract_hidden_states + KDA patches
Dockerfile.train      # ROCm + vLLM + ATOM + AITER + Mooncake + LumenRL
run_kimi_k3.sh        # Bare-metal training entry point
run_docker.sh         # Docker training entry point
run_with_retry.sh     # Auto-restart wrapper with watchdog
split_dataset.py      # Split kimi-mtp-dataset into Phase 1/2
```

## Output

| Path | Description |
|------|-------------|
| `/dev/shm/checkpoints/kimi_k3_dspark_vllm/` | Training checkpoints |
| `output/Kimi_K3_SDDD/LumenRL/` | Training logs |

## Reference

- [Inferact/Kimi-K3-DSpark](https://huggingface.co/Inferact/Kimi-K3-DSpark) — Draft model config
- [moonshotai/Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3) — Target model
- [DSpark paper (arXiv:2607.05147)](https://arxiv.org/abs/2607.05147)
- [vLLM K3 Day-0 Support](https://vllm.ai/blog/2026-07-27-k3)
- [lightseekorg/kimi-mtp-dataset](https://huggingface.co/datasets/lightseekorg/kimi-mtp-dataset)
