# KimiV2.5 Draft Model Distillation (8x MI300X, 4+4 GPU Split)

Train speculative decoding draft models (Eagle3 / DFlash) using teacher hidden states on **8x AMD MI300X** GPUs with a **4+4 GPU split strategy**, following the TorchSpec approach used by KimiV2.5.

- **GPUs 0-3**: ATOM inference engine, TP=4, MXFP4 quantization — frozen Kimi-K2.5 teacher
- **GPUs 4-7**: torchrun FSDP2 — Eagle3/DFlash draft model training (BF16)

## Overview

Speculative decoding accelerates LLM inference by using a small draft model to predict multiple future tokens, which the target model then verifies in parallel. This example trains such draft models via **off-policy distillation** from the target model's hidden states.

### Two Draft Architectures

| Architecture | Mechanism | Loss | Best For |
|---|---|---|---|
| **Eagle3** | Iterative refinement: fuse embedding + teacher hidden, predict step-by-step | Forward KL with position decay (0.8^i) | General-purpose spec decoding |
| **DFlash** | Multi-layer hidden fusion + block-causal masking, single-pass prediction | Cross-entropy with exp(-i/gamma) decay | High-throughput batch decoding |

### Data Flow (4+4 GPU Split, MORI-IO RDMA)

```
Dataset Prompts
  --> ATOM Teacher Forward (GPUs 0-3, frozen, TP=4, MXFP4)
       --> hidden states [B, T, D] written to pre-registered GPU buffer
       --> token embeddings [B, T, D]  (Eagle3 only)
       --> MORI-IO P2P RDMA: GPU 0 --> GPU 4 (GPU Direct, no CPU copy)
  --> Draft Model Forward (GPUs 4-7, with grad, FSDP2)
       --> logits via shared lm_head
       --> loss computation
  --> Backward --> Update Draft Model
```

Key difference from OPD: **no student rollout**. The teacher processes dataset sequences directly, and the draft model learns to predict from teacher representations.

## Hardware Requirements

| Resource | Requirement |
|---|---|
| GPUs | 8x AMD MI300X (192 GiB each) |
| GPU Split | GPUs 0-3: ATOM teacher (TP=4, MXFP4) / GPUs 4-7: FSDP2 training |
| CPU RAM | 256 GiB+ |
| MORI-IO | P2P RDMA for GPU Direct hidden state transfer (GPU 0→4) |
| Shared Memory | `/dev/shm` for input_ids + lm_head weight (small data) |
| ROCm | 6.4+ |

## Models

| Role | Model | Params | GPUs | Notes |
|------|-------|--------|------|-------|
| **Teacher** | [moonshotai/Kimi-K2.5](https://huggingface.co/moonshotai/Kimi-K2.5) | 1T MoE (32B activated) | 0-3 | hidden_dim=7168, 64 heads, 61 layers, vocab=160K |
| **Eagle3 Draft** | [lightseekorg/kimi-k2.5-eagle3](https://huggingface.co/lightseekorg/kimi-k2.5-eagle3) | 3B | 4-7 | Reference architecture for Eagle3 draft |

The teacher is frozen (forward-only, no gradients or optimizer states) and runs on the ATOM inference engine with MXFP4 quantization. With MXFP4, 1T params ≈ 500 GB sharded across 4 GPUs = ~125 GB/GPU, fitting within MI300X's 192 GiB. The draft model (3B) is negligible on the training GPUs.

## Quick Start

### Step 0: Environment Setup

```bash
# 1. Verify ROCm 6.4+ and 8x MI300X
rocm-smi --showid
# Should list 8 GPUs (GPU[0] through GPU[7])

# 2. Install Lumen-RL
pip install -e .

# 3. Install ATOM inference engine (required for full training)
cd third_party/ATOM && pip install -e . && cd ../..

# 4. Install MORI communication library (GPU Direct RDMA for hidden state transfer)
cd third_party/mori && pip install -r requirements-build.txt && git submodule update --init --recursive && pip install -e . && cd ../..
```

### Step 1: Smoke Test (4 GPUs only, ~5 min)

Verifies the full training pipeline using Qwen3-8B as a stand-in teacher.
Only uses **GPUs 4-7** with the HF backend — no ATOM, no quantization, no GPU split.

```bash
# Download stand-in model (~16 GB)
huggingface-cli download Qwen/Qwen3-8B --local-dir /dev/shm/model/qwen3-8b-base

# Run smoke test: 2 training steps, Eagle3 draft, HF teacher
bash examples/KimiV2.5_Draft_Distill_MI300/run_eagle3_bf16.sh --smoke-test
```

Expected output:
```
>>> SMOKE TEST: 2-step Eagle3 distillation (4+4 GPU split)
...
step 1/2  loss=X.XXX  accuracy=X.XXX  grad_norm=X.XXX
step 2/2  loss=X.XXX  accuracy=X.XXX  grad_norm=X.XXX
>>> Eagle3 distillation completed successfully.
```

### Step 2: Full Training (8x MI300X, 4+4 GPU split)

Uses the actual Kimi-K2.5 teacher (1T MoE) with the 4+4 GPU split:
- **GPUs 0-3**: ATOM subprocess loads Kimi-K2.5 with TP=4 + MXFP4 quantization (~125 GB/GPU)
- **GPUs 4-7**: torchrun FSDP2 trains the Eagle3/DFlash draft model (BF16)

```bash
# 1. Download Kimi-K2.5 teacher model (~2 TB, takes a while)
huggingface-cli download moonshotai/Kimi-K2.5 --local-dir /dev/shm/model/kimi-k2.5

# 2a. Eagle3 draft training (2000 steps)
#     Shell script sets CUDA_VISIBLE_DEVICES=4,5,6,7 for torchrun;
#     ATOM subprocess internally manages GPUs 0-3.
bash examples/KimiV2.5_Draft_Distill_MI300/run_eagle3_bf16.sh

# 2b. Or DFlash draft training (2000 steps)
bash examples/KimiV2.5_Draft_Distill_MI300/run_dflash_bf16.sh
```

Each training step:
1. Training rank 0 sends `input_ids` → ATOM subprocess (GPUs 0-3) via `/dev/shm`
2. ATOM does prefill forward → `hidden_states [B, T, D]` + `token_embeds [B, T, D]`
3. Results written to pre-registered GPU buffers on GPU 0
4. MORI-IO RDMA `session.read()`: GPU 0 → GPU 4 (GPU Direct, no CPU copy)
5. Training rank 0 broadcasts to ranks 1-3 (GPUs 4-7) via NCCL
6. Draft model forward → loss → backward → optimizer step

Logs are written to `output/KimiV2.5_Draft_Distill_MI300/kimiv2.5-eagle3-bf16/`.

### Custom Model Path

```bash
MODEL_PATH=/path/to/teacher/model bash examples/KimiV2.5_Draft_Distill_MI300/run_eagle3_bf16.sh
```

### Direct Python Launch (8x MI300X, 4+4 split)

```bash
# Training runs on GPUs 4-7; ATOM subprocess manages GPUs 0-3 internally
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 \
    -m lumenrl.trainer.main \
    --config examples/KimiV2.5_Draft_Distill_MI300/configs/eagle3_bf16.yaml \
    policy.model_name=/path/to/kimi-k2.5 \
    algorithm.teacher.model_name=/path/to/kimi-k2.5
```

### Monitoring

```bash
# Watch training log in real time
tail -f output/KimiV2.5_Draft_Distill_MI300/kimiv2.5-eagle3-bf16/kimiv2.5-eagle3-bf16.log

# Check GPU utilization (GPUs 0-3 should show ATOM inference, 4-7 should show training)
watch -n 2 rocm-smi
```

## Configuration

### Eagle3 Config (`configs/eagle3_bf16.yaml`)

```yaml
algorithm:
  name: spec_distill
  spec_distill:
    draft_type: eagle3        # Iterative refinement architecture
    loss_type: forward_kl     # D_KL(teacher || student)
    position_decay: 0.8       # Loss weight per step: 0.8^i
    num_target_layers: 1      # Number of teacher hidden state layers
  teacher:
    model_name: /dev/shm/model/kimi-k2.5
    inference_backend: atom        # ATOM engine on GPUs 0-3
    tensor_parallel_size: 4
    gpu_ids: [0, 1, 2, 3]
    mori_io_host: "127.0.0.1"     # MORI-IO OOB address
    mori_io_qp_per_transfer: 2    # RDMA queue pairs
  draft:
    from_scratch: true             # Random init (vs. pretrained draft)
    num_layers: 1                  # Transformer blocks in Eagle3
```

### DFlash Config (`configs/dflash_bf16.yaml`)

```yaml
algorithm:
  name: spec_distill
  spec_distill:
    draft_type: dflash             # Block-causal masking architecture
    loss_type: ce_decay            # Cross-entropy + exponential decay
    loss_decay_gamma: 7.0          # Decay: exp(-position / 7.0)
    num_target_layers: 1           # Teacher layers to fuse
  teacher:
    model_name: /dev/shm/model/kimi-k2.5
    inference_backend: atom        # ATOM engine on GPUs 0-3
    tensor_parallel_size: 4
    gpu_ids: [0, 1, 2, 3]
    mori_io_host: "127.0.0.1"     # MORI-IO OOB address
    mori_io_qp_per_transfer: 2    # RDMA queue pairs
  draft:
    from_scratch: true
    num_layers: 2                  # Transformer blocks in DFlash
```

## Key Metrics

| Metric | Description | Expected |
|---|---|---|
| `loss` | Draft model training loss | Should decrease steadily |
| `accuracy` | Token prediction accuracy | Should increase |
| `step_0_loss` / `step_0_acc` | Per-step metrics (Eagle3) | Step 0 should be best |
| `grad_norm` | Gradient norm | Should stabilize < 1.0 |

## Architecture Details

### Eagle3

```
For each speculative step i = 0..length-1:
  1. Fuse: concat(token_embed, teacher_hidden) -> Linear -> LayerNorm
  2. Pass through Transformer block (self-attn + SwiGLU FFN)
  3. Project to vocabulary logits via shared teacher lm_head
  4. Feed predicted token embedding back for next step
```

- Shares teacher's embedding layer and lm_head (tied weights)
- Loss: forward KL divergence, weighted by `position_decay^i` per step
- Reference: EAGLE-2 (Li et al., 2024)

### DFlash

```
  1. Receive teacher hidden states [B, T, D]
  2. Optional: fuse multi-layer hidden states via learned projection
  3. LayerNorm -> Transformer blocks with block-causal attention
  4. Project to vocabulary logits via shared teacher lm_head
```

- Block-causal masking: tokens within a block can attend to each other + all preceding blocks
- Loss: cross-entropy with exponential decay `exp(-position / gamma)`
- Reference: TorchSpec DFlash (Moonshot AI)

## References

- [TorchSpec](https://github.com/lightseekorg/TorchSpec) -- Speculative decoding draft model training framework
- [EAGLE-2](https://arxiv.org/abs/2406.16858) -- Context-aware draft model architecture
- [KimiV2.5 Technical Report](https://arxiv.org/abs/2505.01759) -- Kimi K2.5 with Eagle3 spec decoding
- [MORI](https://github.com/ROCm/mori) -- Modular RDMA Interface for GPU Direct communication
