# Welcome to LumenRL

**LumenRL** is a high-performance RL post-training framework for LLMs on AMD GPUs. It combines [Lumen](https://github.com/ZhangDanyang-AMD/Lumen) for quantized training with [ATOM](https://github.com/ROCm/ATOM) for optimized inference, so a single stack can move from rollout to policy update without leaving ROCm-native kernels and communication paths.

The project targets large-scale post-training workloads where throughput and memory dominate iteration time. FP8 rollout (W8A8 activations and weights, FP8 KV-cache) pairs with hybrid FP8 training recipes in Lumen, and optional rollout correction (TIS/MIS) keeps off-policy FP8 rollouts aligned with the BF16 training policy. For mixture-of-experts models, Rollout Routing Replay (R3) records router behavior during ATOM generation and replays it during training to close the train/inference router gap that otherwise destabilizes MoE RL.

LumenRL ships reference algorithms (GRPO, DAPO, PPO), Ray-based orchestration, and backend choices between FSDP2 and Megatron-Core so you can scale from a single MI300 node to multi-node clusters with SLURM launchers. The documentation below walks through installation, architecture, runnable examples, advanced numerics and MoE topics, and the Python APIs that bind configuration to workers and tensors.

```{toctree}
---
maxdepth: 2
caption: Getting Started
---
quickstart/install
quickstart/quick_start
```

```{toctree}
---
maxdepth: 2
caption: Architecture
---
architecture
```

```{toctree}
---
maxdepth: 2
caption: Examples
---
examples/grpo_training
examples/dapo_training
examples/ppo_training
examples/fp8_training
examples/moe_r3_training
```

```{toctree}
---
maxdepth: 2
caption: Advanced Features
---
advance/fp8_quantization
advance/moe_r3
advance/distributed
advance/algorithms
```

```{toctree}
---
maxdepth: 2
caption: API Reference
---
api/config
api/protocol
api/algorithms
api/quantization
api/moe
```
