# Quick start

This guide runs GRPO in three regimes—BF16 baseline, FP8 end-to-end, and MoE with R3—then shows how to scale out with SLURM.

## Run GRPO (BF16 baseline)

Use the dense BF16 reference config as a correctness baseline:

```bash
cd Lumen-RL
python examples/run_grpo.py --config configs/grpo_dense_bf16.yaml
```

The recipe trains `Qwen/Qwen3-0.6B` with `algorithm.name=grpo`, ATOM rollouts, and FSDP2 policy updates. Checkpointing writes under `checkpointing.checkpoint_dir` from the YAML (defaults to `results/grpo_dense_bf16`).

## Run GRPO (FP8 end-to-end)

Enable FP8 rollout, hybrid FP8 training, and TIS rollout correction:

```bash
python examples/run_grpo.py --config configs/grpo_dense_fp8.yaml
```

See {doc}`/examples/fp8_training` and {doc}`/advance/fp8_quantization` for how the YAML maps to quantizers and correction math.

## Run GRPO (MoE with R3)

For Qwen3-MoE style models, turn on Megatron training paths, expert parallel, FP8, and R3:

```bash
python examples/run_grpo.py --config configs/grpo_moe_fp8_r3.yaml
```

Router tensors are recorded during ATOM rollout and replayed during Lumen training when `moe.r3.enabled=true`. Details: {doc}`/examples/moe_r3_training` and {doc}`/advance/moe_r3`.

## YAML configuration sketch

Below is a trimmed skeleton showing the main sections you will edit across recipes:

```yaml
cluster:
  num_nodes: 1
  gpus_per_node: 8

policy:
  model_name: Qwen/Qwen3-0.6B
  training_backend: fsdp2
  generation_backend: atom
  max_total_sequence_length: 2048
  train_global_batch_size: 32
  train_micro_batch_size: 8

algorithm:
  name: grpo
  grpo:
    num_generations: 8
    clip_ratio: 0.2

quantization:
  rollout:
    precision: bf16
  training:
    fp8: null
  rollout_correction:
    enabled: false

moe:
  r3:
    enabled: false

num_training_steps: 200
seed: 42
```

Full recipes live under `configs/` in the repository; multi-node variants append `cluster.num_nodes` and Megatron parallelism blocks as needed.

## Multi-node SLURM launch

The README pattern parameterizes node count and forwards OmegaConf overrides after the script entrypoint:

```bash
NUM_NODES=2

COMMAND="python examples/run_grpo_moe.py \
    --config configs/grpo_moe_fp8_r3_multinode.yaml \
    cluster.num_nodes=$NUM_NODES \
    cluster.gpus_per_node=8 \
    policy.model_name=Qwen/Qwen3-30B-A3B \
    quantization.rollout.precision=fp8 \
    moe.r3.enabled=true \
    logger.wandb_enabled=true"

sbatch --nodes=$NUM_NODES --gres=gpu:8 scripts/ray.sub
```

Adapt `scripts/ray.sub` to your site’s partition, account, and module loads. Ray head/worker startup should match `cluster.ray_address` when using an external cluster; see {doc}`/advance/distributed`.

## Where to go next

- Hands-on walkthroughs: {doc}`/examples/grpo_training`, {doc}`/examples/fp8_training`, {doc}`/examples/moe_r3_training`
- Systems topics: {doc}`/advance/distributed`, {doc}`/advance/algorithms`, {doc}`/architecture`
- Python APIs: {doc}`/api/config`, {doc}`/api/protocol`, {doc}`/api/algorithms`
