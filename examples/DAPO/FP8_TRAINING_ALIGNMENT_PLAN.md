# FP8 Training Alignment — LumenRL Reproduction Plan

**Date**: 2026-04-16
**Reference**: [Lumen FP8 Training Alignment Plan](../../third_party/Lumen/outputs/fp8_training_alignment/FP8_TRAINING_ALIGNMENT_PLAN.md) | [FP8-RL paper (arXiv:2601.18150)](https://arxiv.org/abs/2601.18150)
**Goal**: Demonstrate FP8 training alignment with BF16 using the **LumenRL** framework (Lumen + ATOM software stack) on AMD Instinct MI350X GPUs. Test blockwise, MXFP8 (block-size sweep), and per-tensor delayed FP8 scaling, FP8 attention, and low-precision optimizers.

---

## Hardware & Environment

| Item | Spec |
|------|------|
| GPUs | 8x AMD Instinct MI350X |
| Container | `rocm/pytorch:rocm6.4_ubuntu22.04_py3.10_pytorch_release_2.6.0` (default) or `--build-arg BASE_IMAGE=...` |
| LumenRL | 0.1.0 (this repo) |
| Lumen | dev/RL branch (`third_party/Lumen`) — FP8 quantized training |
| ATOM | Lumen/RL branch (`third_party/ATOM`) — optimized inference engine |
| AITER | lumen/triton_kernels branch (`third_party/aiter`) — GPU kernels |
| MORI | sdma-new branch (`third_party/mori`) — RDMA + GPU communication |
| PyTorch | 2.9.0+rocm7.0 |

---

## Models & Data

| Asset | Location | Size |
|-------|----------|------|
| Qwen3-8B-Base | `/dev/shm/model/qwen3-8b-base` | 16 GB |
| Qwen3-30B-A3B-Base (MoE) | `/dev/shm/model/qwen3-30b-a3b-base` | 57 GB |
| Qwen3-30B-A3B (MoE) | `/dev/shm/model/qwen3-30b-a3b` | 57 GB |
| DAPO-Math-17k (train) | `/dev/shm/data/dapo-math-17k.parquet` | 286 MB |
| AIME-2024 (val) | `/dev/shm/data/aime-2024.parquet` | 29 KB |

---

## LumenRL Scripts

All experiment scripts live in `examples/DAPO/` and source `common.sh`, which handles prerequisite validation, patch application, environment setup, and the `launch_training` function.

### Existing Scripts

| Script | Experiment | Description |
|--------|-----------|-------------|
| `smoke_test.sh` | — | 2-step BF16 smoke test |
| `smoke_test_fp8.sh` | — | 2-step FP8 E2E (blockwise) smoke test |
| `run_1a_bf16_baseline.sh` | 1A | BF16 training + BF16 rollout (baseline) |
| `run_1b_fp8_rollout_tis.sh` | 1B | BF16 training + FP8 rollout + TIS |
| `run_1d_fp8_e2e_blockwise.sh` | 1D | FP8 E2E (blockwise 128) + FP8 rollout + TIS |
| `run_1f_fp8_e2e_per_tensor.sh` | 1F | FP8 E2E (per-tensor delayed) + FP8 rollout + TIS |

### Scripts to Create

| Script | Experiment | Description |
|--------|-----------|-------------|
| `run_1c_fp8_rollout_no_tis.sh` | 1C | BF16 training + FP8 rollout (no TIS — ablation) |
| `run_1e_fp8_e2e_mxfp8_b32.sh` | 1E | FP8 E2E (MXFP8 block=32) + FP8 rollout + TIS |
| `run_1e64_fp8_e2e_mxfp8_b64.sh` | 1E-64 | FP8 E2E (MXFP8 block=64) + FP8 rollout + TIS |
| `run_1e128_fp8_e2e_mxfp8_b128.sh` | 1E-128 | FP8 E2E (MXFP8 block=128) + FP8 rollout + TIS |
| `run_4a_fp8_e2e_attn_dpa.sh` | 4A | blockwise linear + DPA blockwise attention |
| `run_4b_fp8_e2e_attn_dpa_mxfp8.sh` | 4B | blockwise linear + DPA MXFP8 attention |
| `run_4c_fp8_e2e_attn_dpa_dynamic.sh` | 4C | blockwise linear + DPA dynamic attention |
| `run_4d_fp8_e2e_attn_mha.sh` | 4D | blockwise linear + MHA blockwise2d attention |
| `run_5a_fp8_e2e_optim_bnb8bit.sh` | 5A | blockwise + BNB AdamW8bit |
| `run_5b_fp8_e2e_optim_torchao.sh` | 5B | blockwise + TorchAO _AdamW bf16 stochastic round |

---

## Quantization Methods

### Lumen FP8 Scaling Strategies

LumenRL delegates FP8 training to the Lumen submodule (`third_party/Lumen`). Three scaling methods are tested, controlled by environment variables:

| Scaling | Env Vars | Description | Experiments |
|---------|----------|-------------|-------------|
| **Blockwise** | `LUMEN_FP8_SCALING=blockwise`, `LUMEN_FP8_BLOCK_SIZE=128` | One FP32 scale per 128-element block. Matches the FP8-RL paper and TE reference. | 1D, 2C, 3C (priority 1) |
| **MXFP8** | `LUMEN_FP8_FORMAT=mxfp8`, `LUMEN_FP8_BLOCK_SIZE=32/64/128` | E8M0 (exponent-only) scales per B-element block. Power-of-two scales. OCP MX standard. | 1E/1E-64/1E-128 (priority 2) |
| **Per-tensor (delayed)** | `LUMEN_FP8_SCALING=delayed` | One FP32 scale per entire matrix, derived from amax history. Coarsest granularity. | 1F, 2E, 3E (priority 3) |

### Scaling Comparison

| Property | Blockwise (128) | MXFP8-32 | MXFP8-64 | MXFP8-128 | Per-tensor delayed | H100 Reference (TE) |
|----------|-----------------|----------|----------|-----------|-------------------|---------------------|
| Granularity | 1 per 128 | 1 per 32 | 1 per 64 | 1 per 128 | 1 per matrix | 1 per 128 |
| Scale dtype | FP32 | E8M0 (uint8) | E8M0 | E8M0 | FP32 | FP32 |
| Quant noise | medium | lowest | low | medium | highest | medium |
| GEMM backend | CK/Triton | Triton MXFP8 | same | same | hipBLASLt/CK/Triton | cuBLAS/cuDNN |

### Lumen FP8 E2E Stack — Environment Variables

| Feature | Env Var | Effect |
|---------|---------|--------|
| FP8 Param Manager | `FP8_PARAM_MANAGER=1` | Stores weights as FP8 (memory savings) |
| FP8 Linear GEMM | `LUMEN_FP8=1` | Replaces `nn.Linear` forward with FP8 GEMM via AITER |
| FP8 Scaling Method | `LUMEN_FP8_SCALING=delayed\|blockwise` | Per-tensor delayed or blockwise scaling |
| FP8 Format | `LUMEN_FP8_FORMAT=fp8_e4m3\|mxfp8` | Standard FP8 or MXFP8 microscaling |
| FP8 Block Size | `LUMEN_FP8_BLOCK_SIZE=128` | Block size for blockwise/MXFP8 scaling |
| FP8 Attention | `LUMEN_FP8_ATTN=none\|dpa\|mha` | Quantizes Q/K/V for attention compute |
| Attention Quant Type | `LUMEN_FP8_QUANT_TYPE=blockwise` | Scaling method for attention Q/K/V |
| Attention Kernel | `LUMEN_ATTN_KERNEL_BACKEND=auto` | Attention kernel backend |
| Lumen Norm | `LUMEN_NORM=1` | Optimized RMSNorm/LayerNorm kernels |
| FP8 Activation Store | `LUMEN_FP8_ACTIVATION_STORE=1` | MLP activations stored in FP8 (MoE only) |
| Entry point | `lumenrl.trainer.main` | LumenRL trainer with Lumen FP8 |

### Attention FP8 Modes

| Mode | Env `LUMEN_FP8_ATTN` | Effect |
|------|----------------------|--------|
| **none** (default) | `none` | BF16 attention |
| **dpa** | `dpa` | Q/K/V quantized to FP8 after linear projections; fused FP8 attention kernel |
| **mha** | `mha` | Full MHA FP8: QKV + attention + output projection all in FP8 |

### Attention Quant Types

| Type | Env `LUMEN_FP8_QUANT_TYPE` | Description |
|------|---------------------------|-------------|
| **blockwise** | `blockwise` | Blockwise FP8 scaling for Q/K/V |
| **blockwise2d** | `blockwise2d` | 2D blockwise with shared scale manager (used with `mha`) |
| **mxfp8** | `mxfp8` | MXFP8 microscaling for attention tensors |
| **dynamic** | `dynamic` | Dynamic per-tensor scaling |

---

## Experiment Matrix

### Experiment 1: Qwen3-8B-Base Dense — FP8 Rollout + FP8 E2E

| Run | Training | Rollout | TIS | Script |
|-----|----------|---------|-----|--------|
| **1A** | BF16 (FSDP2) | BF16 | — | `run_1a_bf16_baseline.sh` |
| **1B** | BF16 (FSDP2) | FP8 + TIS | token, C=2 | `run_1b_fp8_rollout_tis.sh` |
| 1C | BF16 (FSDP2) | FP8 | — | `run_1c_fp8_rollout_no_tis.sh` |
| **1D** | FP8 E2E (blockwise 128) | FP8 + TIS | token, C=2 | `run_1d_fp8_e2e_blockwise.sh` |
| 1E | FP8 E2E (MXFP8 block=32) | FP8 + TIS | token, C=2 | `run_1e_fp8_e2e_mxfp8_b32.sh` |
| 1E-64 | FP8 E2E (MXFP8 block=64) | FP8 + TIS | token, C=2 | `run_1e64_fp8_e2e_mxfp8_b64.sh` |
| 1E-128 | FP8 E2E (MXFP8 block=128) | FP8 + TIS | token, C=2 | `run_1e128_fp8_e2e_mxfp8_b128.sh` |
| **1F** | FP8 E2E (per-tensor delayed) | FP8 + TIS | token, C=2 | `run_1f_fp8_e2e_per_tensor.sh` |
| 4A | FP8 E2E blockwise + DPA blockwise attn | FP8 + TIS | token, C=2 | `run_4a_fp8_e2e_attn_dpa.sh` |
| 4B | FP8 E2E blockwise + DPA MXFP8 attn | FP8 + TIS | token, C=2 | `run_4b_fp8_e2e_attn_dpa_mxfp8.sh` |
| 4C | FP8 E2E blockwise + DPA dynamic attn | FP8 + TIS | token, C=2 | `run_4c_fp8_e2e_attn_dpa_dynamic.sh` |
| 4D | FP8 E2E blockwise + MHA blockwise2d attn | FP8 + TIS | token, C=2 | `run_4d_fp8_e2e_attn_mha.sh` |
| 5A | FP8 E2E blockwise + BNB AdamW8bit | FP8 + TIS | token, C=2 | `run_5a_fp8_e2e_optim_bnb8bit.sh` |
| 5B | FP8 E2E blockwise + TorchAO _AdamW bf16 | FP8 + TIS | token, C=2 | `run_5b_fp8_e2e_optim_torchao.sh` |

**Expected outcome**: 1A ≈ 1B ≈ 1D ≈ 1E ≈ 1E-64 ≈ 1E-128 ≈ 1F ≈ 4A-4D ≈ 5A-5B. 1C shows accuracy drop (no TIS ablation).

### Experiment 1 Config (8x MI350X)

| Parameter | Value | Ref (H100) | Notes |
|-----------|-------|------------|-------|
| Model | Qwen/Qwen3-8B-Base | Same | Dense, 8B params |
| GPUs | 8 | 8 | Same |
| Prompt batch size | 32 | 32 | Same |
| Responses per prompt (n) | 16 | 16 | Same |
| Train batch size | 32 | 32 | Same |
| Gen batch size | 32 | 96 | Reduced for MI350 memory |
| PPO mini batch size | 32 | 32 | Same |
| Max prompt length | 1024 | 1024 | Same |
| Max response length | 20480 | 20K | Same |
| LR | 1e-6 | 1e-6 | Same |
| Clip (low/high) | 0.2 / 0.28 | Same | DAPO decoupled clip |
| Loss aggregation | token-mean | Same | DAPO token-level loss |
| Total steps | **275** | 500 | Reduced (~5 min/step on MI350) |
| Val frequency | 5 steps | 5 | Same |
| gpu_memory_util | **0.3** | 0.9 | ATOM sleep() ROCm adaptation |
| free_cache_engine | **True** | N/A | Explicitly frees KV cache |
| enforce_eager | True | False | Required for ROCm |
| FSDP2 offload | param + optimizer | Same | Memory savings |

### Experiment 2: Qwen3-30B-A3B-Base MoE

| Run | Training | Rollout | TIS | Notes |
|-----|----------|---------|-----|-------|
| 2A | BF16 (FSDP2) | BF16 + TIS | token, C=2 | MoE BF16 baseline |
| 2B | BF16 (FSDP2) | FP8 + TIS | token, C=2 | FP8 rollout only |
| **2C** | FP8 E2E (blockwise 128) | FP8 + TIS | token, C=2 | Priority 1 |
| 2D | FP8 E2E (MXFP8 block=32) | FP8 + TIS | token, C=2 | Priority 2 |
| 2D-64 | FP8 E2E (MXFP8 block=64) | FP8 + TIS | token, C=2 | |
| 2D-128 | FP8 E2E (MXFP8 block=128) | FP8 + TIS | token, C=2 | |
| 2E | FP8 E2E (per-tensor delayed) | FP8 + TIS | token, C=2 | Priority 3 |

**Config adaptation for 8 GPUs** (reference used 16): batch sizes halved (train=16, gen=16, mini=16), SP_SIZE=4.

### Experiment 3: Qwen3-30B-A3B MoE (Instruct)

Same matrix as Experiment 2 (runs 3A-3E), using the instruct model variant. Independent confirmation of FP8 alignment on instruction-tuned starting point.

### Experiment 4: FP8 Attention Sweep (Qwen3-8B-Base)

Linear is **fixed** to blockwise 128 (same as 1D). Only attention env vars change.

| Run | Attention Mode | Attn Quant Type |
|-----|----------------|-----------------|
| 1D (baseline) | none (BF16) | — |
| 4A | dpa | blockwise |
| 4B | dpa | mxfp8 |
| 4C | dpa | dynamic |
| 4D | mha | blockwise2d |

### Experiment 5: Low-Precision Optimizer (Qwen3-8B-Base)

Linear is **fixed** to blockwise 128, attention is BF16 (same as 1D). Only the optimizer changes.

| Run | Optimizer |
|-----|-----------|
| 1D (baseline) | PyTorch AdamW (FP32 states) |
| 5A | BNB AdamW8bit |
| 5B | TorchAO _AdamW (bf16 stochastic round) |

---

## Lumen FP8 E2E Config per Experiment

| Parameter | Blockwise (1D/2C/3C) | MXFP8-32 (1E/2D/3D) | MXFP8-64 | MXFP8-128 | Per-tensor (1F/2E/3E) |
|-----------|----------------------|----------------------|----------|-----------|----------------------|
| `FP8_PARAM_MANAGER` | 1 | 1 | 1 | 1 | 1 |
| `LUMEN_FP8` | 1 | 1 | 1 | 1 | 1 |
| `LUMEN_FP8_SCALING` | `blockwise` | `blockwise` | `blockwise` | `blockwise` | `delayed` |
| `LUMEN_FP8_FORMAT` | `fp8_e4m3` | `mxfp8` | `mxfp8` | `mxfp8` | `fp8_e4m3` |
| `LUMEN_FP8_BLOCK_SIZE` | `128` | `32` | `64` | `128` | — |
| `LUMEN_NORM` | 1 | 1 | 1 | 1 | 1 |
| `LUMEN_FP8_ACTIVATION_STORE` | 1 (MoE) | 1 (MoE) | 1 (MoE) | 1 (MoE) | 1 (MoE) |
| Entry point | `lumenrl.trainer.main` | same | same | same | same |

---

## Metrics to Track

### Accuracy Metrics (alignment — primary result)

| Metric | Log Key | Description |
|--------|---------|-------------|
| **Validation accuracy** | `val-core/math_dapo/acc/mean@1` | AIME-2024 accuracy |
| **Reward** | `critic/rewards/mean` | DAPO reward signal |
| **Score** | `critic/score/mean` | DAPO math accuracy score |
| **Mismatch KL** | `rollout_corr/kl` | Rollout/training distribution KL |
| **Response length** | `response_length/mean` | Average generation length |

### Performance Metrics (throughput)

| Metric | Log Key | Description |
|--------|---------|-------------|
| **Throughput** | `perf/throughput` | Tokens/second |
| **Step time** | `timing_s/step` | Seconds per training step |
| **Rollout time** | `timing_s/gen` | Seconds for generation phase |
| **Update time** | `timing_s/update_actor` | Seconds for actor gradient update |

---

## Execution Order

### Step 1: Environment Setup

```bash
# Enter the container
docker exec -it lumenrl bash
cd /workspace/Lumen-RL/examples/DAPO

# Verify prerequisites
ls /dev/shm/model/qwen3-8b-base/
ls /dev/shm/data/dapo-math-17k.parquet
ls /dev/shm/data/aime-2024.parquet
python3 -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

### Step 2: Smoke Tests

```bash
# BF16 pipeline sanity check (2 steps, ~5 min)
bash smoke_test.sh

# FP8 E2E sanity check (2 steps, verifies Lumen FP8 stack)
bash smoke_test_fp8.sh
```

Both smoke tests use reduced batch sizes (train=16, n=2, max_resp=512) for fast iteration. They must pass before launching full experiments.

### Step 3: BF16 Baseline (Experiment 1A)

```bash
bash run_1a_bf16_baseline.sh
```

**Verify**:
- Training converges (reward increases over steps)
- `val-core/math_dapo/acc/mean@1` shows improving validation accuracy
- Logs appear in `output/DAPO/1A-qwen3-8b-bf16-baseline.log`
- Checkpoints write to `/root/ckpts/LUMENRL-FP8-ALIGN/1A-qwen3-8b-bf16-baseline/`

This is the reference curve for all FP8 comparisons.

### Step 4: FP8 Rollout Tests

```bash
# FP8 rollout with TIS (expected: 1B ≈ 1A)
bash run_1b_fp8_rollout_tis.sh

# FP8 rollout without TIS (ablation: expected accuracy drop vs 1A)
bash run_1c_fp8_rollout_no_tis.sh
```

### Step 5: FP8 E2E Tests (priority order)

```bash
# Priority 1: Blockwise (closest match to H100 TE reference)
bash run_1d_fp8_e2e_blockwise.sh

# Priority 2: MXFP8 block-size sweep
bash run_1e_fp8_e2e_mxfp8_b32.sh
bash run_1e64_fp8_e2e_mxfp8_b64.sh
bash run_1e128_fp8_e2e_mxfp8_b128.sh

# Priority 3: Per-tensor delayed (coarsest, lower bound)
bash run_1f_fp8_e2e_per_tensor.sh
```

### Step 6: Attention FP8 Sweep (Experiment 4)

After linear FP8 is validated, fix linear to blockwise 128 and sweep attention:

```bash
bash run_4a_fp8_e2e_attn_dpa.sh
bash run_4b_fp8_e2e_attn_dpa_mxfp8.sh
bash run_4c_fp8_e2e_attn_dpa_dynamic.sh
bash run_4d_fp8_e2e_attn_mha.sh
```

### Step 7: Optimizer Sweep (Experiment 5)

After FP8 linear + attention are validated:

```bash
bash run_5a_fp8_e2e_optim_bnb8bit.sh
bash run_5b_fp8_e2e_optim_torchao.sh
```

### Step 8: MoE Experiments (Experiments 2 & 3)

Run after dense 8B experiments confirm the stack works:

```bash
# MoE Base model (halved batch sizes, SP=4)
# Scripts to be created following the same pattern
```

### Step 9: Compare and Report

- Experiment 1 chart: overlay 1A vs 1B vs 1C vs 1D vs 1E vs 1E-64 vs 1E-128 vs 1F vs 4A-4D vs 5A-5B
- MXFP8 block-size sweep chart: overlay MXFP8-32 vs MXFP8-64 vs MXFP8-128
- Attention FP8 chart: 1D vs 4A vs 4B vs 4C vs 4D
- Optimizer chart: 1D vs 5A vs 5B (alignment + peak GPU memory)
- Write results to `output/DAPO/FP8_TRAINING_ALIGNMENT_RESULTS.md`

---

## Code Changes Required (in Lumen submodule)

These changes are required in `third_party/Lumen` before running certain experiments:

### 1. Wire blockwise/MXFP8 env vars into trainer config

The LumenRL trainer config currently defaults to `linear_fp8_scaling="delayed"`. Add:

```python
linear_fp8_scaling = os.environ.get("LUMEN_FP8_SCALING", "delayed")
linear_fp8_format = os.environ.get("LUMEN_FP8_FORMAT", "fp8_e4m3")
linear_fp8_block_size = int(os.environ.get("LUMEN_FP8_BLOCK_SIZE", "128"))
```

**Needed before**: Experiments 1D, 1E, 1E-64, 1E-128 (blockwise and MXFP8 runs).

### 2. Remove MXFP8 block_size clamp

`lumen/ops/quantize/linear.py:195` has `mxfp8_block = 32 if block_size > 64 else block_size` which silently reduces block_size=128 to 32. Change to `mxfp8_block = block_size`.

**Needed before**: Experiments 1E-64, 1E-128 (MXFP8 block=64 and 128).

### 3. Wire attention FP8 env vars into trainer config

Add `LUMEN_FP8_QUANT_TYPE` and `LUMEN_ATTN_KERNEL_BACKEND` env var reads. Add `lumen_fp8_quant_type` and `lumen_attn_backend` fields to `LumenRLConfig`.

**Needed before**: Experiment 4 (all attention FP8 runs).

### 4. Wire low-precision optimizer

Modify LumenRL trainer config to inject `optimizer_impl` overrides when `USE_8BIT_ADAM=1`.

**Needed before**: Experiment 5 (optimizer sweep).

---

## ROCm Adaptations

These are pre-configured in all LumenRL DAPO scripts via `common.sh`:

| Adaptation | Value | Reason |
|------------|-------|--------|
| `gpu_memory_utilization` | 0.3 | ATOM sleep() ROCm adaptation; use low reservation |
| `free_cache_engine` | True | Explicitly frees KV cache between rollout and training |
| `enforce_eager` | True | CUDA graph capture disabled on ROCm |
| `TORCHDYNAMO_DISABLE` | 1 | Disables TorchDynamo on ROCm |

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| ATOM FP8 rollout fails on ROCm | Blocks FP8 experiments | Smoke test first (`smoke_test_fp8.sh`) |
| Qwen3 MoE OOM on 8 GPUs | Blocks experiments 2/3 | Reduce batch sizes, increase offloading |
| Blockwise CK GEMM untested on MI350X | GEMM may fail | Falls back to Triton; smoke test first |
| MXFP8 requires gfx950 (CDNA4) | MI300X would fail | MI350X is gfx950+; `is_cdna4()` gate in kernel |
| MXFP8 block_size clamp in code | Defeats 64/128 sweep | Must patch `quantize_input` (see Code Changes) |
| Trainer config missing blockwise env vars | Cannot select scaling | Wire env vars (see Code Changes) |
| FP8 attention Triton kernel NaN on varlen | Experiment 4 fails | Use `LUMEN_ATTN_KERNEL_BACKEND=auto`; fall back to csrc |
| BNB AdamW8bit + FSDP2 DTensor crash | Experiment 5A fails | Smoke test first; fall back to 5B (TorchAO) |

---

## Output Structure

```
Lumen-RL/
├── examples/DAPO/
│   ├── FP8_TRAINING_ALIGNMENT_PLAN.md     # This document
│   ├── common.sh                           # Shared config and launch function
│   ├── smoke_test.sh                       # BF16 smoke test
│   ├── smoke_test_fp8.sh                   # FP8 smoke test
│   ├── run_1a_bf16_baseline.sh             # Experiment 1A
│   ├── run_1b_fp8_rollout_tis.sh           # Experiment 1B
│   ├── run_1c_fp8_rollout_no_tis.sh        # Experiment 1C (to create)
│   ├── run_1d_fp8_e2e_blockwise.sh         # Experiment 1D
│   ├── run_1e_fp8_e2e_mxfp8_b32.sh        # Experiment 1E (to create)
│   ├── run_1e64_fp8_e2e_mxfp8_b64.sh      # Experiment 1E-64 (to create)
│   ├── run_1e128_fp8_e2e_mxfp8_b128.sh    # Experiment 1E-128 (to create)
│   ├── run_1f_fp8_e2e_per_tensor.sh        # Experiment 1F
│   ├── run_4a_fp8_e2e_attn_dpa.sh          # Experiment 4A (to create)
│   ├── run_4b_fp8_e2e_attn_dpa_mxfp8.sh   # Experiment 4B (to create)
│   ├── run_4c_fp8_e2e_attn_dpa_dynamic.sh  # Experiment 4C (to create)
│   ├── run_4d_fp8_e2e_attn_mha.sh          # Experiment 4D (to create)
│   ├── run_5a_fp8_e2e_optim_bnb8bit.sh    # Experiment 5A (to create)
│   ├── run_5b_fp8_e2e_optim_torchao.sh    # Experiment 5B (to create)
│   └── patches/                            # ROCm patches
│
└── output/DAPO/
    ├── <EXP_NAME>.log                      # Training logs
    ├── metrics/                            # Parsed metrics
    └── FP8_TRAINING_ALIGNMENT_RESULTS.md   # Final comparison (after runs)
```

---

## Quick Reference: Running Experiment 1A

```bash
# 1. Enter the container
docker exec -it lumenrl bash

# 2. Navigate to LumenRL DAPO directory
cd /workspace/Lumen-RL/examples/DAPO

# 3. Verify prerequisites
ls /dev/shm/model/qwen3-8b-base/config.json
ls /dev/shm/data/dapo-math-17k.parquet

# 4. Run smoke test first
bash smoke_test.sh

# 5. Run 1A baseline (275 steps, ~23 hours at ~5 min/step)
bash run_1a_bf16_baseline.sh

# 6. Monitor output
tail -f ../../output/DAPO/1A-qwen3-8b-bf16-baseline.log
```

**Key environment overrides** for quick testing:

```bash
# Run fewer steps
TOTAL_STEPS=10 bash run_1a_bf16_baseline.sh

# Use a different model path
MODEL_PATH=/path/to/model bash run_1a_bf16_baseline.sh

# Change GPU memory reservation
GPU_MEM_UTIL=0.5 bash run_1a_bf16_baseline.sh
```
