# Qwen3-30B-A3B Agentic RL — AMD Kernel Agent Training

Train an AMD-specific HIP/Triton/FlyDSL kernel optimisation agent using GSPO + GEAK sandbox + TRLOO.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│  Qwen3-30B-A3B (MoE, 128 experts)                          │
│  ├── GSPO: sequence-level ratio, MoE-stable training        │
│  ├── TRLOO: turn-level LOO advantage for multi-turn agents  │
│  └── Multi-turn rollout: prompt → action → obs → ... → eval│
├─────────────────────────────────────────────────────────────┤
│  GEAK Sandbox                                               │
│  ├── KernelSandbox: isolated kernel workspace               │
│  ├── Three-stage evaluation: compile / correctness / perf   │
│  └── Profiling-based reward + Hacking detection             │
├─────────────────────────────────────────────────────────────┤
│  LumenRL Training                                           │
│  ├── Megatron: TP=4, EP=8 (MI300X/MI308X)                   │
│  └── ATOM: BF16 rollout, TP=2, FP8 KV cache                 │
└─────────────────────────────────────────────────────────────┘
```

## Prerequisites

1. **LumenRL** — installed with Megatron + ATOM configured
2. **GEAK** — cloned to `$DATA_ROOT/geak` with kernel task cases
3. **Qwen3-30B-A3B model** — downloaded to `$DATA_ROOT/models/Qwen3-30B-A3B`
4. **2-node MI300X/MI308X cluster** — 8 GPUs per node
5. **ROCm 6.x** + PyTorch 2.x + flash-attn (ROCm build)

## Quick Start

### 1. Set Environment Variables

```bash
export RL_ROOT=/path/to/your/rl/workspace
export DATA_ROOT=/mnt/raid0
```

### 2. Smoke Test (3-step validation)

```bash
MODE=smoke STEPS=3 bash examples/AgenticRL/Qwen3_30B_A3B/run_agentic_rl.sh
```

### 3. Full Training

```bash
MODE=longrun STEPS=200 bash examples/AgenticRL/Qwen3_30B_A3B/run_agentic_rl.sh
```

### 4. Custom Configuration

```bash
# Specify GEAK root and cases
GEAK_ROOT=/path/to/geak \
CASES_PATH=/path/to/cases.yaml \
MODE=longrun STEPS=100 \
bash examples/AgenticRL/Qwen3_30B_A3B/run_agentic_rl.sh

# Use a custom config file
CONFIG_OVERRIDE=path/to/your_config.yaml \
bash examples/AgenticRL/Qwen3_30B_A3B/run_agentic_rl.sh
```

## Configuration Reference

### Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `algorithm.name` | `gspo` | Sequence-level importance ratio, MoE-friendly |
| `algorithm.adv_estimator` | `trloo` | Turn-level Leave-One-Out advantage |
| `algorithm.gspo.clip_ratio` | `0.2` | Symmetric clip range for sequence-level ratio |
| `algorithm.gspo.num_generations` | `8` | Number of rollout episodes per prompt |
| `policy.learning_rate` | `5e-7` | Lower LR for agentic RL stability |
| `policy.max_total_sequence_length` | `8192` | Multi-turn agent requires long context |
| `policy.max_response_length` | `4096` | Max tokens per single turn |

### GEAK Sandbox Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `agentic_rl.max_turns` | `15` | Max turns per episode |
| `agentic_rl.gpu_ids` | `"0"` | GPU used for kernel evaluation |
| `agentic_rl.command_timeout` | `300` | Per-command timeout in seconds |
| `agentic_rl.baseline_repeats` | `3` | Number of baseline performance measurements |

### Reward Composition

```
R_total = R_correctness + R_performance + R_profiling + R_hacking_penalty

R_correctness:  compile pass +0.1, correctness pass +0.3, failure -1.0
R_performance:  log(speedup) + 0.5 * improvement_over_best
R_profiling:    0.3 * bandwidth + 0.3 * occupancy + 0.4 * instruction_efficiency
R_hacking:      lazy deletion / hardcoded output / test gaming -> penalty
```

### Hacking Detection

| Detector | Trigger | Penalty |
|----------|---------|---------|
| Lazy deletion | LOC drops >50% | -2.0 |
| Hardcoded output | Hardcoded tensor/zeros return values | -3.0 |
| Test gaming | Latency <10% of baseline with correctness pass | -2.5 |
| Copy-paste | Source identical to baseline | 0 (no penalty, but no optimisation reward) |
| Timeout | Any stage times out | -2.0 |

## Training Loop

```
for each prompt (kernel task):
    for i in 1..num_generations:
        1. Reset GEAK sandbox -> establish baseline performance
        2. Multi-turn rollout:
           - Model generates action (read/write/evaluate)
           - GEAK sandbox executes -> observation
           - Repeat for max_turns
        3. Per-turn reward assignment via TRLOO advantage estimation
        4. Hacking detection on final candidate
        5. Profiling-based reward from rocprof
    GSPO loss: sequence-level ratio -> gradient update
```

## Running Tests

```bash
cd experiments/multi-tune-agent
python -m pytest tests/test_gspo.py tests/test_trloo.py tests/test_geak_gym.py \
    tests/test_multi_turn_trajectory.py tests/test_multi_turn_rollout.py \
    tests/test_profiling_reward.py tests/test_hacking_detection.py \
    tests/test_turn_level_reward.py -v
```

## File Structure

```
examples/AgenticRL/Qwen3_30B_A3B/
├── README.md                              # This document
├── gspo_qwen3_30b_a3b_geak_smoke.yaml    # Training config
└── run_agentic_rl.sh                      # Launch script

experiments/multi-tune-agent/
├── geak_gym/                              # Gymnasium-compatible environment
│   ├── env.py                             # GEAKGymEnv
│   ├── spaces.py                          # Action/Observation definitions
│   └── reward_shaping.py                  # Configurable reward shaping
├── multi_turn_trajectory.py               # Multi-turn trajectory management
├── multi_turn_rollout.py                  # Multi-turn rollout orchestration
├── turn_level_reward.py                   # Per-turn reward computation
├── hacking_detection.py                   # Reward hacking detection
└── tests/                                 # Unit tests

lumenrl/algorithms/
├── gspo.py                                # GSPO Algorithm class
├── loss_functions.py                      # + gspo_loss()
└── advantage_estimators.py                # + gspo, trloo estimators

lumenrl/rewards/
└── profiling_reward.py                    # rocprof profiling reward
```
