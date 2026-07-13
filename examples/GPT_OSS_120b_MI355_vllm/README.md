# GPT-OSS-120B Eagle3 SDDD — vLLM Teacher (MI355)

On-policy speculative decoding draft distillation for GPT-OSS-120B (117B MoE)
using **vLLM** for teacher inference. Replaces ATOM teacher to eliminate the
double-normalization problem caused by ATOM's fused kernel residual scale.

## Architecture

```
GPUs 0-3: FSDP2 Eagle3 draft model training (BF16, LumenRL + aiter)
GPUs 4-7: vLLM teacher inference (TP=4, BF16, FP8 KV cache)
Transfer: Mooncake TCP (hidden states)
aux_hidden_state_layer_ids: [2, 18, 33, 36]
```

## Why vLLM instead of ATOM

vLLM captures raw `hidden_states + residual` at natural scale (~1-10),
matching NVIDIA's approach. Eagle3's `hidden_norm` provides the single
learned normalization — no double-norm.

ATOM's MXFP4 fused kernels accumulate residuals to 10^35+ scale, forcing
capture normalization that conflicts with Eagle3's `hidden_norm`.

## Quick Start

```bash
# Build Docker image
docker buildx build \
    -f examples/GPT_OSS_120b_MI355_vllm/Dockerfile.train \
    -t gpt_oss_eagle3_vllm_train:latest .

# Smoke test (5 steps)
bash examples/GPT_OSS_120b_MI355_vllm/run_docker.sh --smoke-test

# Full training with auto-retry
bash examples/GPT_OSS_120b_MI355_vllm/run_with_retry.sh
```

## vLLM Patches

Patches for vLLM 0.19.1+rocm721 are in `patch/`. They are applied
automatically at container startup by `run_gpt_oss_120b.sh` (idempotent).

| Patch | Description |
|-------|-------------|
| `patch_vllm_kv_cache_grouping.py` | Fix `HiddenStatesCacheSpec` being merged with normal attention layers in KV cache group init. `HiddenStatesCacheSpec` inherits `FullAttentionSpec`, so `UniformTypeKVCacheSpecs.from_specs()` incorrectly treats it as uniform. The patch strips hidden-state specs before uniformity checks and reattaches them as singleton groups. |

To apply manually (e.g. in a running container):

```bash
docker exec <container> python /root/lumenrl/examples/GPT_OSS_120b_MI355_vllm/patch/patch_vllm_kv_cache_grouping.py
```

## Training Config

- Forward KL loss with `position_decay=0.9` across 3 TTT steps (`spec_length=3`)
- lr=1e-4, linear warmup ~1000 steps, linear decay, total 15,871 steps
- `global_batch_size=32`, `save_steps=100`, `save_total_limit=3`
- See `configs/train.yaml` for full config

## Benchmark: MT-Bench by Category

Evaluation uses vLLM speculative decoding with `draft_length=3`, matching
[NVIDIA's evaluation methodology](https://huggingface.co/nvidia/gpt-oss-120b-Eagle3-long-context#evaluation).

### Method

1. Export checkpoint to HuggingFace safetensors format (`output/export_eagle3_hf_gpt_oss.py`)
2. Serve with vLLM: `--speculative-model <exported_model> --num-speculative-tokens 3 --speculative-draft-tensor-parallel-size 1`
3. Run all 80 MT-Bench questions grouped by category, measure acceptance length via vLLM prometheus metrics (`spec_decode_num_accepted_tokens_total / spec_decode_num_drafts_total + 1`)
4. Benchmark script: `bench_mtbench_category.py`

### Results — checkpoint_5500 (step 5500 / 15,871, 34.7% training)

| Category | Accept Length | NVIDIA Ref | Gap |
|------------|:---:|:---:|:---:|
| coding | 1.92 | 2.51 | -0.59 |
| extraction | 1.72 | 2.53 | -0.81 |
| humanities | 1.68 | 1.95 | -0.27 |
| math | 2.09 | 2.83 | -0.74 |
| reasoning | 1.70 | 2.47 | -0.77 |
| roleplay | 1.74 | 2.25 | -0.51 |
| stem | 1.77 | 2.17 | -0.40 |
| writing | 1.64 | 2.24 | -0.60 |
| **OVERALL** | **1.77** | **2.37** | **-0.59** |

NVIDIA reference: [nvidia/gpt-oss-120b-Eagle3-long-context](https://huggingface.co/nvidia/gpt-oss-120b-Eagle3-long-context) (fully trained).

Our checkpoint at 34.7% training reaches 74.7% of NVIDIA's overall acceptance length.

### Exported Models

| Step | HuggingFace |
|------|-------------|
| 5500 | [Zhangdanyang/gpt-oss-120b-Eagle3-step5500-Lumen](https://huggingface.co/Zhangdanyang/gpt-oss-120b-Eagle3-step5500-Lumen) |

## Known Issues

- **Mooncake TCP crashes**: `batch_put_from` / `batch_get_buffer` timeout every ~4-6 hours. Recovery: stop container → `rm /dev/shm/nccl-* /dev/shm/lumenrl_vllm_*` → restart. Watchdog (`watchdog.sh`) monitors and auto-restarts.

## Prerequisites

- 8x MI355 GPUs (288GB each)
- Model weights at `/dev/shm/gpt-oss-120b`
- Dataset at `/dev/shm/gpt_oss_120b_dataset/train.jsonl`
