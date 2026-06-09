# GPT-OSS-120B Eagle3 Draft Distillation (vLLM + Mooncake TCP) — MI308

Train Eagle3 speculative decoding draft model using OpenAI's `gpt-oss-120b` (117B-param MoE, 5.1B active) teacher hidden states on **8x MI308 GPUs** with vLLM inference and Mooncake TCP transfer.

- **GPUs 0-3**: torchrun FSDP2 -- Eagle3 draft model training (BF16, LumenRL + aiter)
- **GPUs 4-7**: vLLM -- gpt-oss-120b teacher (TP=4, native MXFP4 MoE)

## Architecture

```
Training GPUs (0-3)                    Inference GPUs (4-7)
LumenRL FSDP2 + aiter          <---   vLLM 0.19.1 (extract_hidden_states)
  Eagle3 draft model, BF16              TP=4, native MXFP4 MoE
       ^                                       |
  Mooncake TCP  <---------------------  hidden_states via
  EagleMooncakeStore                   MooncakeHiddenStatesConnector
```

## Docker image

**Reuses `lumenrl-vllm-mi308:latest`** built by the sibling Qwen3-8B example. No
separate Dockerfile is needed here because:

- vLLM 0.19.1 (in the existing image) already registers `GptOssForCausalLM` and
  loads MXFP4 weights natively.
- `vllm/model_executor/models/gpt_oss.py` already inherits from `EagleModelMixin`
  and implements `_maybe_add_hidden_state`, so the existing
  `examples/Qwen3_8B_SDDD_MI350_vLLM/docker/patches/vllm/v0.19.1/*.patch`
  cover the extract_hidden_states pipeline without any gpt-oss-specific addition.

If the image isn't built yet:
```bash
bash examples/Qwen3_8B_SDDD_MI350_vLLM/docker/build.sh
```

## Quick Start

### 1. Download model

```bash
# huggingface-cli download openai/gpt-oss-120b --local-dir /dev/shm/gpt-oss-120b
```
(~196 GB MXFP4 distribution; large download.)

### 2. Smoke test (5 steps, synthetic prompts)

```bash
bash examples/GPT_OSS_120b_MI308_ATOM/run_docker.sh --smoke-test
# logs → output/GPT_OSS_120b_SDDD/LumenRL/
```

### 3. Phase 1 — Foundation on ultrachat_200k (19,488 steps ≈ 3 epochs)

```bash
# HF_TOKEN strongly recommended — anonymous Hub API gets 429-rate-limited
# after a few container restarts and the trainer dies on dataset load.
HF_TOKEN=hf_xxx bash examples/GPT_OSS_120b_MI308_ATOM/run_docker.sh
```

### 4. Phase 2 — Specialize on Magpie-300K (28,125 steps ≈ 3 epochs, resumes phase 1)

```bash
HF_TOKEN=hf_xxx bash examples/GPT_OSS_120b_MI308_ATOM/run_docker.sh --phase2
```

Loads the latest `checkpoint_*.pt` from phase 1's directory (`/dev/shm/checkpoints/gpt_oss_120b_eagle3_vllm/`) and continues into a separate `..._phase2/` dir so phase 1 finals are preserved. Same draft arch + aux layer IDs as phase 1 — only dataset, LR (2e-5), and warmup change. Aborts if phase 1 ckpts are missing.

### 5. Auto-restart wrapper (long runs)

```bash
LUMENRL_DRAFT_FLASH_BACKEND=matmul \
    bash examples/GPT_OSS_120b_MI308_ATOM/run_with_retry.sh
```

`run_with_retry.sh` wraps `run_docker.sh` with two safety nets specific to MI308/ROCm:
- **HSA aperture fault loop** — auto-restarts the container; `resume:true` picks up the latest ckpt so worst-case redo ≈ `save_steps` (100).
- **Idle-log watchdog** — kills + restarts if the trainer log goes silent for `HANG_IDLE_SEC` (default 600s), to escape silent step-700-class hangs where NCCL watchdog cannot detect rank 0 stuck in CUDA sync.

Knobs: `MAX_ATTEMPTS=50 HANG_IDLE_SEC=900 HF_TOKEN=hf_xxx ...`. Passes `--phase2` through if present.

### Recipe alignment with [nvidia/gpt-oss-120b-Eagle3-long-context](https://huggingface.co/nvidia/gpt-oss-120b-Eagle3-long-context)

| | This config | NVIDIA recipe |
|---|---|---|
| Draft arch | 1-layer Llama-style, head_dim=64, ffn=17280, GQA 8:1, llama3 RoPE (factor=8) | identical |
| `eagle_aux_hidden_state_layer_ids` | `[1, 17, 32]` | `[1, 17, 32]` |
| Max seq len (train) | 8192 | 8192 |
| Phase 1 prompts | ultrachat_200k `train_sft` (loader uses `prompt` field only) | ultrachat_200k, prompts only |
| Phase 2 prompts | Magpie-Llama-3.1-Pro-300K-Filtered `train` (loader applies chat template to `conversations`) | Magpie, prompts only |
| Phase 1 steps | 19,488 (3 epochs @ global_bs=32) | similar |
| Phase 2 steps | 28,125 (3 epochs @ global_bs=32) | similar |

Behavioral note: ultrachat doesn't carry a `conversations` field, so the loader falls back to `s["prompt"]` (user-only). Magpie does carry `conversations`, so the loader templates the full user+assistant turn. Strict NVIDIA "prompts only" parity for phase 2 would require strip-assistant preprocessing in `_load_dataset`.

Override env vars as needed:
```bash
MODEL_PATH=/some/other/path \
CKPT_DIR=/some/checkpoint/dir \
DOCKER_IMAGE=lumenrl-vllm-mi308:my-tag \
HF_TOKEN=hf_xxx \
bash examples/GPT_OSS_120b_MI308_ATOM/run_docker.sh
```

## Model facts (gpt-oss-120b)

| | |
|---|---|
| Total params | 117B |
| Active params / token | 5.1B (MoE: 128 experts, 4 per token) |
| Layers | 36 |
| Hidden | 2880 |
| Heads / KV heads | 64 / 8 (GQA) |
| Vocab | 201088 (`o200k_harmony` tokenizer) |
| RoPE | θ=150000 + YaRN(factor=32, original_max=4096) |
| Native quantization | MXFP4 for MoE weights (~196 GB on disk) |
| Eagle3 aux layers (auto, N=36) | `[2, 18, 33]` |

## File structure

```
configs/
  phase1_ultrachat.yaml         # Phase 1: ~3 epochs ultrachat_200k from scratch (19,488 steps)
  phase2_magpie.yaml            # Phase 2: 3 epochs Magpie-300K, resumes from phase 1 final ckpt
  smoke_test.yaml               # 5-step e2e pipeline validation (synthetic prompts, dataset-free)
run_gpt_oss_120b.sh             # In-container entrypoint (torchrun + overrides)
run_docker.sh                   # Host-side wrapper, launches container
run_with_retry.sh               # Auto-restart wrapper around run_docker.sh (HSA-fault loop + hang watchdog)
run_benchmark_vllm_docker.sh    # vLLM-serve + bench_eagle3_vllm.py runner against a trained draft
benchmark_results/              # JSON dumps + HTML dashboards
```

No `docker/` subdir: the image is shared with `Qwen3_8B_SDDD_MI350_vLLM`.

## Environment variables

| Variable | Value | Purpose |
|----------|-------|---------|
| `CUDA_VISIBLE_DEVICES` | `0,1,2,3,4,5,6,7` | All 8 GPUs |
| `PYTORCH_ROCM_ARCH` | `gfx942` | MI308 GPU architecture |
| `HIP_FORCE_DEV_KERNARG` | `1` | Required for MI308 kernel args |
| `WANDB_MODE` | `disabled` | Disable W&B logging for smoke tests |
| `MODEL_PATH` | `/dev/shm/gpt-oss-120b` (override) | Teacher / policy model path |
| `DOCKER_IMAGE` | `lumenrl-vllm-mi308:latest` (override) | Container image tag |

## Config notes

- `algorithm.teacher.quantization: ""` — gpt-oss-120b ships natively as MXFP4
  (see HF config `quantization_config.quant_method=mxfp4`); vLLM loads it
  as-is from the checkpoint directory. We do **not** pass `quantization=mxfp4`
  to vLLM args, which would re-route through the ATOM online-quantization
  plugin (built for BF16-shipped models and trips a `list(None)` bug on
  non-quark configs).
- `policy.max_total_sequence_length: 16384` — gpt-oss native is 131K (YaRN
  factor 32) but Eagle3 draft training doesn't need full context; capping
  reduces mooncake hidden-state buffer pressure.
- `policy.learning_rate: 5e-5` — lower than Qwen3-8B's `1e-4`, matching Kimi
  K25 (larger model → smaller LR).
- AITER acceleration features (`lumen_norm`, `lumen_linear`, `hf_attn_patch`)
  intentionally left unset → defaults to disabled, same race-condition
  workaround as the other MI308/MI350 examples.

## Known issues

1. **MI308 (gfx942) not officially validated upstream.** vLLM/AMD only validate
   MI300X/325X/355X. MI308 shares the gfx942 ISA so the same code paths
   should run, but no upstream report confirms gpt-oss-120b on MI308.
   First run is the validation.
2. **Eagle3 aux layer indices `[2, 18, 33]` correspond to gpt-oss layer types
   `sliding/full/full`** per the model's alternating `layer_types` config.
   If draft acceptance is unexpectedly low, sampling only from full-attention
   layers might help — would require a code change in
   `lumenrl/engine/inference/vllm_teacher_engine.py:104`.

## Reference

- [openai/gpt-oss-120b on Hugging Face](https://huggingface.co/openai/gpt-oss-120b)
- [vLLM gpt-oss recipe](https://docs.vllm.ai/projects/recipes/en/latest/OpenAI/GPT-OSS.html)
- Sibling examples: `examples/Qwen3_8B_SDDD_MI350_vLLM/`, `examples/Kimi_K25_SDDD_MI350_ATOM/`
