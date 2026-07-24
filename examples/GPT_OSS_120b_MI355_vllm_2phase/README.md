# GPT-OSS-120B Eagle3 SDDD v3 — Two-Phase Training (MI355)

Two-phase on-policy speculative decoding draft distillation for GPT-OSS-120B
(117B MoE), aligned with NVIDIA's Eagle3-v3 approach using **Nemotron
Post-Training V3** datasets.

## Motivation

Single-phase training on UltraChat+Magpie (~503K samples) reached 1.88 accept
length vs NVIDIA's 2.37 (79.3%). NVIDIA's v3 approach uses ~2.9M samples from
the Nemotron PTv3 collection in two phases to achieve higher acceptance rates.

| Approach | Samples | Accept Length | % of NVIDIA |
|:---------|--------:|:---:|:---:|
| v1 single-phase (UltraChat+Magpie) | 503K | 1.88 | 79.3% |
| **v3 two-phase (Nemotron PTv3)** | **~2.9M** | **TBD** | **TBD** |
| NVIDIA reference | ~2.9M | 2.37 | 100% |

## Architecture

```
GPUs 0-3: FSDP2 Eagle3 draft model training (BF16, LumenRL + aiter)
GPUs 4-7: vLLM teacher inference (TP=4, BF16, FP8 KV cache)
Transfer: Mooncake TCP (hidden states)
aux_hidden_state_layer_ids: [2, 18, 33, 36]
```

## Two-Phase Training

### Phase 1: Short Context (≤4096 tokens)

- **Dataset**: Nemotron PTv3 prompts with tokenized length ≤4096 (~2.7M samples)
- **Sequence length**: 4096 tokens
- **Steps**: ~84,375 (2.7M / batch_size 32)
- **Warmup**: 1000 steps
- **Config**: `configs/train_phase1.yaml`

### Phase 2: Long Context (>4096 tokens)

- **Dataset**: Nemotron PTv3 prompts with tokenized length >4096 (~200K samples)
- **Sequence length**: 8192 tokens
- **Steps**: ~6,250 (200K / batch_size 32)
- **Warmup**: 200 steps
- **Resumes from**: Phase 1 final checkpoint
- **Config**: `configs/train_phase2.yaml`

### Shared Training Parameters

- Forward KL loss with `position_decay=0.9` across 3 TTT steps (`spec_length=3`)
- lr=1e-4, linear warmup, linear decay
- `global_batch_size=32`, `max_grad_norm=1.0`

## Dataset: Nemotron Post-Training V3

Prompts extracted from the [Nemotron Post-Training V3 collection](https://huggingface.co/collections/nvidia/nemotron-post-training-v3):

| Dataset | Splits | Cap/Split | Est. Samples |
|:--------|:-------|:---------:|-------------:|
| Nemotron-Math-v2 | 5 | 40K | 200K |
| Nemotron-SFT-Math-v3 | 1 | 200K | 200K |
| Nemotron-Math-Proofs-v1 | 1 | 50K | 50K |
| Nemotron-SWE-v1 | 1 | — | ~51K |
| Nemotron-SFT-SWE-v2 | 2 | — | ~256K |
| Nemotron-Competitive-Programming-v1 | 6 | 50K | 300K |
| Nemotron-SFT-Competitive-Programming-v2 | 4 | 50K | 200K |
| Nemotron-Science-v1 | 2 | — | ~226K |
| Nemotron-Instruction-Following-Chat-v1 | 2 | — | ~288K |
| Nemotron-SFT-Instruction-Following-Chat-v2 | 2 | — | varies |
| Nemotron-Agentic-v1 | 2 | — | ~335K |
| Nemotron-SFT-Agentic-v2 | 3 | — | ~992K |
| Nemotron-SFT-Safety-v1 | 1 | — | ~45K |
| Nemotron-SpecializedDomains-Finance-v1 | 1 | 100K | 100K |
| Nemotron-SFT-Multilingual-v1 | 18 | 10K | 180K |

Only prompts (user messages) are used — assistant turns are stripped. The teacher
model generates responses on-policy during training.

## Quick Start

```bash
# Uses the same Docker image as v1
docker buildx build \
    -f examples/GPT_OSS_120b_MI355_vllm/Dockerfile.train \
    -t gpt_oss_eagle3_vllm_train:latest .

# Step 0: Prepare dataset (auto-downloads from HuggingFace)
# Step 1: Run Phase 1 (short context)
# Step 2: Run Phase 2 (long context, resumes from Phase 1)
bash examples/GPT_OSS_120b_MI355_vllm_2phase/run_with_retry.sh

# Or run phases individually:
PHASE=1 bash examples/GPT_OSS_120b_MI355_vllm_2phase/run_with_retry.sh
PHASE=2 bash examples/GPT_OSS_120b_MI355_vllm_2phase/run_with_retry.sh
```

### Dataset Preparation Only

```bash
docker run --rm \
    -v /dev/shm:/dev/shm \
    -v /home/danyzhan/Lumen-RL:/root/lumenrl \
    -w /root/lumenrl \
    -e HF_TOKEN="${HF_TOKEN}" \
    gpt_oss_eagle3_vllm_train:latest \
    python3 examples/GPT_OSS_120b_MI355_vllm_2phase/make_dataset_nemotron_ptv3.py \
        --output-dir /dev/shm/gpt_oss_120b_dataset_v3 \
        --tokenizer /dev/shm/gpt-oss-120b
```

## File Structure

```
GPT_OSS_120b_MI355_vllm_2phase/
├── configs/
│   ├── train_phase1.yaml          # Phase 1: short context ≤4096
│   └── train_phase2.yaml          # Phase 2: long context >4096
├── make_dataset_nemotron_ptv3.py  # Dataset preparation script
├── run_gpt_oss_120b.sh            # Inner training runner (per-phase)
├── run_docker.sh                  # Docker wrapper (data prep + training)
├── run_with_retry.sh              # Auto-retry wrapper (both phases)
└── README.md
```

## Checkpoints

- Phase 1: `/dev/shm/checkpoints/gpt_oss_120b_eagle3_v3_phase1/`
- Phase 2: `/dev/shm/checkpoints/gpt_oss_120b_eagle3_v3_phase2/`
- Logs: `output/GPT_OSS_120b_SDDD_v3/LumenRL/`

## vLLM Patches

Reuses patches from the v1 setup (`../GPT_OSS_120b_MI355_vllm/patch/`).

## Known Issues

- **Mooncake TCP crashes**: `batch_put_from` / `batch_get_buffer` timeout every
  ~4-6 hours. `run_with_retry.sh` handles automatic recovery.
- **Dataset download**: First run downloads ~50GB+ of Nemotron datasets from
  HuggingFace. Ensure sufficient disk space and a valid `HF_TOKEN`.

## Prerequisites

- 8x MI355 GPUs (288GB each)
- Model weights at `/dev/shm/gpt-oss-120b`
- Docker image: `gpt_oss_eagle3_vllm_train:latest`
- HuggingFace token with access to Nemotron datasets
