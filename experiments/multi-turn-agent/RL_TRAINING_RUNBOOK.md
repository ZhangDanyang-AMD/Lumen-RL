# GEAK Kernel RL Training Runbook

## 0. 目标

在 SFT checkpoint 基础上，通过 Lumen-RL 的 GRPO 框架进一步提升 kernel coding policy。

**Base model**: `Zhangdanyang/Qwen3-Coder-30B-A3B-SFT-TH-4epoch`
**Hardware**: 8x AMD MI308X (gfx942, 192GB HBM3)
**Framework**: Lumen-RL (`lumenrl/algorithms/grpo.py` + `lumenrl/trainer/rl_trainer.py`)
**Reward**: GEAK sandbox (`lumenrl/rewards/profiling_reward.py`)

---

## 1. Lumen-RL GRPO 基建（已就绪）

### 1.1 核心组件

| 组件 | 路径 | 说明 |
|------|------|------|
| GRPO 算法 | `lumenrl/algorithms/grpo.py` | Asymmetric clip, KL penalty, rollout IS |
| Advantage | `lumenrl/algorithms/advantage_estimators.py` | 12+ estimators: grpo, trloo, rloo, dapo... |
| 训练循环 | `lumenrl/trainer/rl_trainer.py` | rollout → reward → advantage → update |
| Reward | `lumenrl/rewards/profiling_reward.py` | rocprof hardware counters + speedup |
| AgenticRL 配置 | `examples/AgenticRL/Qwen3_30B_A3B/gspo_qwen3_30b_a3b_geak_smoke.yaml` | GEAK sandbox 集成 |
| GRPO 配置 | `examples/GRPO/configs/grpo_qwen3_30b_a3b_vllm_ep8_smoke.yaml` | Qwen3-30B-A3B GRPO |
| Loss 函数 | `lumenrl/algorithms/loss_functions.py` | policy gradient, asymmetric clip, KL |
| Rollout 校正 | `lumenrl/algorithms/rollout_correction.py` | FP8 rollout IS/MIS |

### 1.2 Multi-Tune Agent 集成组件

| 组件 | 路径 | 说明 |
|------|------|------|
| Kernel Reward | `experiments/multi-tune-agent/rewards/kernel_reward.py` | GEAK sandbox batch reward（compile→correctness→performance） |
| Prompt Loader | `experiments/multi-tune-agent/rewards/kernel_prompt_loader.py` | SFT 数据 → GRPO rollout prompts |
| RL Trajectory | `experiments/multi-tune-agent/rewards/rl_trajectory.py` | Rollout 结果记录与分析 |
| GRPO Config | `experiments/multi-tune-agent/configs/rl/grpo_kernel_single_turn.yaml` | Single-turn GRPO 训练配置 |
| Fuzzy Patch | `experiments/multi-tune-agent/scripts/fuzzy_patch.py` | Context-insensitive patch apply |
| Benchmark | `experiments/multi-tune-agent/scripts/run_held_out_benchmark.py` | Multi-turn held-out evaluation |
| Harness 生成 | `experiments/multi-tune-agent/scripts/materialize_held_out_harnesses.py` | Held-out task harness 生成 |

### 1.3 Multi-Tune Agent Trajectory 系统

现有 `src/multi_tune_agent/trajectory.py` 提供 agent 运行的 trajectory 记录：
- `TrajectoryWriter`: append-only JSONL event log
- Per-turn: model text, tool args/results, token usage, wall time
- Per-run: `summary.json` with speedup, baseline, timing

RL 训练的 `rl_trajectory.py` 扩展了这个系统：
- 记录每个 rollout 的 reward、compile/correctness/speedup
- Per-step aggregation（mean reward, compile rate, correct rate）
- Per-operator 和 per-lane breakdown
- Training report（reward/compile/correct trajectory over steps）

### 1.4 已有 GEAK AgenticRL 配置

`examples/AgenticRL/Qwen3_30B_A3B/gspo_qwen3_30b_a3b_geak_smoke.yaml` 包含：
- GSPO + TRLOO + GEAK sandbox reward
- Multi-turn agent（max_turns=15）
- Profiling-based reward（bandwidth/occupancy/instruction efficiency）
- Hacking detection（lazy deletion/suspicion thresholds）
- MoE R3 router replay

---

## 2. Single-Turn GRPO 配置

基于现有 AgenticRL 配置，适配单节点 + single-turn：

**配置文件**: `experiments/multi-tune-agent/configs/rl/grpo_kernel_single_turn.yaml`

```yaml
cluster:
  num_nodes: 1
  gpus_per_node: 8
  ray_address: "auto"

controller:
  ray:
    enabled: true
    fuse_actor_ref: true
    actor:
      num_workers: 8
      dispatch_mode: dp_compute_proto
    rollout:
      num_workers: 4

policy:
  model_name: /home/danyzhan/Lumen/experiments/GEAK-agent-coder/outputs/qwen3-coder-full2000-4epoch-merged
  training_backend: fsdp2
  generation_backend: vllm
  max_total_sequence_length: 16384
  max_response_length: 8192
  train_global_batch_size: 32
  gen_batch_size: 8
  train_micro_batch_size: 1
  learning_rate: 5.0e-7
  training:
    optimizer_dtype: bf16
    fsdp_cfg:
      optimizer_offload: true
      reshard_after_forward: false
  generation:
    vllm_cfg:
      tensor_parallel_size: 1
      kv_cache_dtype: fp8
      enforce_eager: true

algorithm:
  name: grpo
  adv_estimator: grpo
  grpo:
    num_generations: 8
    kl_coeff: 0.0
    clip_ratio: 0.2
    clip_ratio_high: 0.28

reward:
  type: function
  function: experiments.multi_tune_agent.rewards.kernel_reward.kernel_reward_batch

agentic_rl:
  environment: geak
  geak_root: /home/danyzhan/GEAK
  cases_path: /home/danyzhan/geak_sft_dataset/phase1-production-wave-2000-v1/processed/train.jsonl
  max_turns: 1
  gpu_ids: "4,5,6,7"
  command_timeout: 120
  baseline_repeats: 3

num_training_steps: 500
seed: 42
```

---

## 3. Kernel Reward Function

**文件**: `experiments/multi-tune-agent/rewards/kernel_reward.py`

调用 GEAK sandbox 的 compile → correctness → performance pipeline：

```python
def kernel_reward_batch(batch_responses, batch_prompts, **kwargs):
    """Compute kernel optimization rewards using GEAK sandbox."""
    # For each response:
    # 1. Apply patch / write kernel.py to isolated workspace
    # 2. Run compile (GPU sandbox)
    # 3. Run correctness check
    # 4. Measure performance (3 repeats, median)
    # 5. Compute reward: -1 (compile fail), -0.5 (incorrect), log(speedup) (correct)
```

Reward 设计（来自 `geak_tool.py` 的已验证公式）：
```
reward = -1.0                              # compile 失败
       = -0.5                              # correctness 失败
       = 1.0 + clip(log(speedup), 0, log3) # 正确
       + 0.5 * max(0, speedup - prev_best) # 改进 bonus
```

---

## 4. 启动流程

### 4.1 环境准备

```bash
# 安装 Lumen-RL
cd /home/danyzhan/Lumen-RL
pip install -e .

# 安装 multi-tune-agent
cd experiments/multi-tune-agent
pip install -e .

# 确保 GEAK 可用
export GEAK_ROOT=/home/danyzhan/GEAK
export HELD_OUT_ROOT=/home/danyzhan/held-out-benchmark-aiter
export EVAL_GPU_IDS="4,5,6,7"
```

### 4.2 验证 Reward Pipeline

```bash
# 测试 kernel reward function 在单个 task 上的工作
cd /home/danyzhan/Lumen-RL/experiments/multi-tune-agent
python -c "
from rewards.kernel_reward import eval_single_kernel
from pathlib import Path

result = eval_single_kernel(
    response='No changes needed',
    task_id='heldv5-adversarial_boundary-rms_norm-triton-02',
    task_dir=Path('/home/danyzhan/held-out-benchmark-aiter/artifacts/kernel/heldv5-adversarial_boundary-rms_norm-triton-02/initial'),
    baseline_ms=0.05,
    family='rms_norm',
    gpu_id=4,
)
print(f'Reward: {result[\"reward\"]}, Stage: {result[\"stage\"]}')
"
```

### 4.3 运行 GRPO 训练

```bash
cd /home/danyzhan/Lumen-RL

# 启动 Ray cluster
ray start --head --num-gpus=8

# 运行 single-turn GRPO
python -m lumenrl.trainer.main \
    --config experiments/multi-tune-agent/configs/rl/grpo_kernel_single_turn.yaml
```

### 4.4 监控训练

```bash
# 查看 trajectory
tail -f /home/danyzhan/Lumen/experiments/GEAK-agent-coder/outputs/rl-grpo-kernel/rollouts.jsonl

# 查看 step summaries
tail -f /home/danyzhan/Lumen/experiments/GEAK-agent-coder/outputs/rl-grpo-kernel/step_summaries.jsonl
```

### 4.5 评估（使用 V8 held-out benchmark）

```bash
cd /home/danyzhan/Lumen-RL/experiments/multi-tune-agent

# Merge RL checkpoint
python /home/danyzhan/Lumen/experiments/GEAK-agent-coder/scripts/merge_to_hf.py

# Serve with vLLM and run benchmark
export PYTHONPATH="src:.:scripts:${PYTHONPATH}"
python scripts/run_held_out_benchmark.py rl-grpo both
```

---

## 5. Benchmark 指标

### 5.1 Result Quality

| Metric | SFT (V8) | RL Target |
|--------|----------|-----------|
| Patch Pass@1 | 5% | **15-25%** |
| Patch Pass@5 | 37% | **50-60%** |
| Gen Compile | 40% | **55-65%** |
| Gen Correct | 12% | **20-30%** |
| Speedup geomean | N/A | **>1.1x** |

### 5.2 Agent Loop Efficiency

| Metric | SFT | RL Target |
|--------|-----|-----------|
| First-pass rate | 5% | **15-25%** |
| Turns to pass | 2.9 | **<2.0** |
| Error recovery | 32% | **40-50%** |
| Cost-of-Pass | 75K tok | **<50K tok** |
| Compile error rate | 76% | **<50%** |

### 5.3 Per-Operator Targets

| Operator | SFT Pass@5 | RL Target |
|----------|-----------|-----------|
| rms_norm | ~50% | >70% |
| gemm | ~30% | >50% |
| fused_moe | ~40% | >60% |
| mha/mla/paged_attn | ~10% | >30% |

### 5.4 对比模型

Base, SFT-2e, SFT-4e, RL-GRPO, Claude Code Opus

---

## 6. 参考文献

- [Dr. Kernel (ICML 2026)](https://arxiv.org/abs/2602.05885)
- [Kevin (ICLR 2026)](https://arxiv.org/abs/2507.11948)
- [DRTriton (2026)](https://arxiv.org/abs/2603.21465)
- [AMDKernelVault (2026)](https://arxiv.org/abs/2609.12471)
- [LEAP (2026)](https://arxiv.org/abs/2608.01804)
- [Afterburner (2025)](https://arxiv.org/abs/2505.23387)


## 7. Multi-Turn GRPO Training (Primary)

Single-turn GRPO has sparse reward (~3% compile rate → 88% zero-gradient steps). Multi-turn RL provides denser signal through iterative optimization.

### 7.1 Script

```bash
python3 scripts/grpo_multiturn.py
```

### 7.2 Key Features

- **TRLOO (Turn-level Reinforce Leave-One-Out)**: Advantage estimation per turn, not per episode
- **Kevin-style credit assignment**: Discounted future rewards γ=0.4 for earlier turns
- **On-policy sync**: Merge LoRA weights → restart vLLM every training step to keep inference model current
- **Curriculum learning**: EpochIterator traverses easy→medium→hard tasks
- **Parallel rollouts**: ThreadPoolExecutor with 6 GPU workers for simultaneous eval
- **Context compression**: Old turns compressed to summary, keeping last 4 turns verbatim

### 7.3 On-Policy Sync (Critical)

Without on-policy sync, vLLM serves a frozen base model while LoRA updates → off-policy training.

Every training step:
1. Merge LoRA adapter into base model weights
2. Save merged checkpoint
3. Kill vLLM process
4. Restart vLLM with merged checkpoint
5. Wait for health check
6. Resume training with updated model

```python
# In grpo_multiturn.py
merge_lora_weights(base_model, lora_adapter)
save_merged(output_dir / f"step-{step}")
restart_vllm(merged_path)
```

### 7.4 Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| group_size | 4 | Rollouts per prompt |
| max_turns | 10 | Turns per rollout |
| lr | 5e-6 | Learning rate |
| kl_coeff | 0.02 | KL penalty coefficient |
| gamma | 0.4 | Kevin discount factor |
| eval_every | 5 | Online eval frequency (steps) |
| eval_tasks | 40 | Tasks for online eval |

### 7.5 W&B Monitoring

```bash
export WANDB_BASE_URL=https://forge.coreweave.com/api/wandb
export WANDB_PROJECT=coder-model-rl
```

Key metrics to watch:
- `reward/mean` — should trend upward
- `reward/compile_rate` — should reach >50%
- `reward/correct_rate` — should reach >30%
- `advantage/std` — should be non-zero (zero = no learning signal)
- `kl/mean` — should stay <0.1

### 7.6 Evaluation

Use the GEAK agent benchmark (not the old held-out benchmark):

```bash
python3 -u scripts/run_aiter_benchmark_geak.py \
    --model-label rl-step-N \
    --model-name rl-step-N \
    --max-turns 50
```

Dataset: `/home/danyzhan/held-out-benchmark-aiter/` (100 AITER tasks, autotuned baselines)

See `MODEL_KERNEL_AGENT_BENCHMARK.md` for full benchmark setup.

### 7.7 Launch Example

```bash
cd /home/danyzhan/Lumen-RL/experiments/multi-tune-agent
export PYTHONPATH="src:.:scripts:${PYTHONPATH}"
export EVAL_GPU_IDS="1,2,3,4,5,6,7"
export WANDB_BASE_URL=https://forge.coreweave.com/api/wandb
export WANDB_PROJECT=coder-model-rl

# Start vLLM on GPU 0 with tool calling support
ROCR_VISIBLE_DEVICES=0 python3 -m vllm.entrypoints.openai.api_server \
    --model <SFT_MERGED_PATH> \
    --served-model-name sft2 \
    --max-model-len 262144 \
    --enforce-eager --dtype bfloat16 \
    --trust-remote-code --gpu-memory-utilization 0.95 \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder &

# Start multi-turn GRPO training
python3 scripts/grpo_multiturn.py \
    --base-model <SFT_MERGED_PATH> \
    --output-dir outputs/rl-mt-sft2 \
    --max-steps 100
```
