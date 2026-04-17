# MoE R3 training example

Mixture-of-experts models amplify throughput but introduce a train/inference router mismatch: ATOM and Megatron may not route tokens identically, which shows up as exploding KL and collapsed expert usage. LumenRL’s Rollout Routing Replay (R3) records router logits during rollout and replays them during training forwards.

## Qwen3-MoE reference configuration

`configs/grpo_moe_fp8_r3.yaml` targets `Qwen/Qwen3-30B-A3B` with Megatron training parallelism, FP8 rollout/training, and R3 enabled:

```yaml
policy:
  model_name: Qwen/Qwen3-30B-A3B
  training_backend: megatron
  generation_backend: atom
  training:
    megatron_cfg:
      tensor_parallel_size: 4
      expert_parallel_size: 2
      num_experts: 128
      moe_grouped_gemm: true
  generation:
    atom_cfg:
      tensor_parallel_size: 4
      kv_cache_dtype: fp8

moe:
  r3:
    enabled: true
    record_router_logits: true
    replay_mode: distribution
```

`replay_mode` selects whether soft distributions or hard assignments are injected during replay; `distribution` is the default soft path.

## R3 enabled vs disabled

Compare against `configs/grpo_moe_r3.yaml` or a local copy with `moe.r3.enabled=false` but identical parallelism and learning rates:

```bash
# R3 on (recommended for MoE RL)
python examples/run_grpo.py --config configs/grpo_moe_fp8_r3.yaml

# R3 off (diagnostic only—may be unstable)
python examples/run_grpo.py --config configs/grpo_moe_fp8_r3.yaml moe.r3.enabled=false
```

When R3 is off, watch for sudden spikes in policy–reference KL and erratic reward variance even though throughput looks unchanged.

## Monitoring router stability

Use MoE diagnostics from `lumenrl.moe.moe_utils` during offline evaluation of captured `router_logits` tensors:

```python
from lumenrl.moe.moe_utils import compute_router_entropy, check_expert_utilization

summary = check_expert_utilization(router_logits, num_experts=128)
entropy = compute_router_entropy(router_logits)
print(summary["argmax_expert_mass_mean"], float(entropy))
```

Healthy training tends to maintain non-degenerate `mean_softmax_mass_per_expert` entries without a single expert hoarding all probability mass.

```{note}
R3 tensors travel inside `DataProto` under keys `router_dist_layer_{idx}`. Downstream code should preserve them across merge/split boundaries when scaling data parallel workers.
```

## Related documentation

- Conceptual guide: {doc}`/advance/moe_r3`
- Distributed setup for Megatron + EP: {doc}`/advance/distributed`
- Python APIs: {doc}`/api/moe`, {doc}`/api/protocol`
