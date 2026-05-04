# DeepSeek-V4 On-Policy Distillation (OPD) — 8x MI300X

Reproduce DeepSeek-V4's On-Policy Distillation training on **8x AMD MI300X** GPUs, where a student model learns from a teacher by minimising KL divergence on student-generated sequences.

## Overview

In DeepSeek-V4's training pipeline, **OPD replaces the mixed RL stage**. Instead of balancing rewards across multiple domains with complex RL, OPD directly distills a domain expert's knowledge into the student model.

### How It Works

```
1. Student Rollout (no grad)
   Student generates sequences using its current policy.

2. Teacher Forward (no grad)
   Teacher model produces logits on the student's sequences.
   With lazy_logits: teacher returns hidden states [B,T,D],
   TeacherLMHead reconstructs logits [B,T,V] at training time.

3. Student Training (with grad)
   Student does a second forward pass on the same sequences.
   Loss = KL(student_logits || teacher_logits)  (reverse KL)
   Backprop updates student model.
```

### Key Properties

| Property | OPD | Traditional RL (GRPO/DAPO) |
|---|---|---|
| **Data source** | Student rollout (on-policy) | Student rollout (on-policy) |
| **Signal** | Teacher logits | Reward function |
| **Loss** | KL divergence | Policy gradient |
| **Reward model** | Not needed | Required |
| **Use case** | Multi-expert fusion | Alignment / reasoning |

## Model

| Role | Model | Params | Notes |
|------|-------|--------|-------|
| **Student / Teacher** | [deepseek-ai/DeepSeek-V4-Pro](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro) | 1.6T MoE (49B activated) | Hybrid CSA+HCA attention, 1M context, FP4+FP8 weights |

> **Note:** DeepSeek-V4-Pro in BF16 is ~3.2TB. OPD loads both student and teacher.
> With `lazy_logits: true` (same-model OPD), only the lm_head is cached separately.
> For single-node 8x MI300X, use FP8/FP4 quantized weights.
> For full BF16, use 4+ nodes (32+ MI300X GPUs).

## Hardware Requirements

| Resource | Requirement |
|---|---|
| GPUs | 8x AMD MI300X (192 GiB each) |
| CPU RAM | 256 GiB+ |
| ROCm | 6.4+ |

## Quick Start

### Smoke Test (8x MI300X, ~5 min)

Uses Qwen3-8B as a stand-in for pipeline verification:

```bash
# Download stand-in model
huggingface-cli download Qwen/Qwen3-8B --local-dir /dev/shm/model/qwen3-8b-base

bash examples/DeepSeekV4_OPD_MI300/run_opd_bf16.sh --smoke-test
```

### Full Training (8x MI300X)

```bash
# Download DeepSeek-V4-Pro
huggingface-cli download deepseek-ai/DeepSeek-V4-Pro --local-dir /dev/shm/model/deepseek-v4-pro

# Same model as student and teacher
bash examples/DeepSeekV4_OPD_MI300/run_opd_bf16.sh

# Different teacher (distill expert → student)
TEACHER_PATH=/path/to/math-expert \
    bash examples/DeepSeekV4_OPD_MI300/run_opd_bf16.sh
```

### Direct Launch (8x MI300X)

```bash
torchrun --nproc_per_node=8 -m lumenrl.trainer.main \
    --config examples/DeepSeekV4_OPD_MI300/configs/opd_bf16.yaml \
    policy.model_name=/path/to/deepseek-v4-pro \
    algorithm.teacher.model_name=/path/to/deepseek-v4-pro
```

## Configurations

### Basic OPD (`configs/opd_bf16.yaml`)

Same architecture for student and teacher. The student learns to replicate the teacher's output distribution on its own generated sequences.

```yaml
algorithm:
  name: opd
  opd:
    kl_direction: reverse         # D_KL(student || teacher) — DeepSeek-V4
    temperature: 1.0
    opd_coeff: 1.0
    lazy_logits: true             # Memory-efficient: cache hidden states
```

### Multi-Teacher / Cross-Architecture (`configs/opd_multi_teacher.yaml`)

Distill from a larger teacher into a smaller student. When architectures differ, set `lazy_logits: false` (hidden dim mismatch prevents lazy reconstruction).

```yaml
algorithm:
  opd:
    lazy_logits: false            # Full logits mode for cross-architecture
    position_weighting: true      # Weight earlier tokens more
    position_decay: 0.8           # Decay: 0.8^position
```

## Key Metrics

| Metric | Description | Expected |
|---|---|---|
| `opd_kl` | KL divergence (student \|\| teacher) | Should decrease |
| `loss_total` | opd_coeff * opd_kl | Should decrease |
| `grad_norm` | Gradient norm | Should stabilize < 1.0 |
| `timing/gen_s` | Rollout time per step | Student generation |
| `timing/teacher_s` | Teacher forward time | Dominates for large teachers |
| `timing/train_s` | Student training time | Forward + backward + optim |

## Lazy Logits

For large vocabularies (e.g., 150K+ tokens), storing full teacher logits `[B, T, V]` can exceed GPU memory. **Lazy logits** solves this:

1. Teacher forward returns hidden states `[B, T, D]` instead of logits `[B, T, V]`
2. A frozen `TeacherLMHead` (just the lm_head weight) reconstructs logits during training
3. Memory savings: `D << V` (e.g., 4096 vs 150000)

Set `lazy_logits: true` when student and teacher share the same lm_head architecture.

## Combined RL + OPD

OPD can be combined with RL training (GRPO/DAPO) for joint optimization. Use a standard RL config with `opd.opd_coeff > 0`:

```yaml
algorithm:
  name: grpo
  grpo:
    num_generations: 4
    kl_coeff: 0.05
  opd:
    opd_coeff: 0.3                # > 0 enables joint RL + OPD
    kl_direction: reverse
    lazy_logits: true
```

This joint mode adds a teacher forward phase to the RL loop and computes `L = L_RL + opd_coeff * L_OPD`.

## References

- [DeepSeek-V4 Technical Report](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro) -- Section 5.2: RL and OPD Infrastructures
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) -- Pre-training and distillation pipeline
