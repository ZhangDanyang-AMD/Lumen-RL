# miles + LumenRL Plugin: Qwen3-8B Megatron + SGLang

Run Qwen3-8B RL training on **miles** with LumenRL's Megatron bridge, Lumen
FP8, and FP8 state management hooks, injected via miles's `--custom-*-path`
CLI arguments.

## What the plugin does

miles loads hook functions at runtime via `importlib.import_module`. LumenRL
provides six hooks in `lumenrl.plugin.miles.hooks`:

| CLI argument | Hook function | What it does |
|---|---|---|
| `--custom-megatron-init-path` | `custom_megatron_init` | Creates `FP8TrainingManager` |
| `--custom-model-provider-path` | `custom_model_provider` | Builds GPTModel with Lumen FP8 |
| `--rollout-function-path` | `generate_rollout` | ATOM rollout (optional, replaces SGLang) |
| `--custom-megatron-before-train-step-hook-path` | `custom_megatron_before_train_step_hook` | Resets FP8 scales before each micro-step |
| `--custom-megatron-before-log-prob-hook-path` | `custom_megatron_before_log_prob_hook` | Resets FP8 scales before log-prob pass |
| `--custom-megatron-post-save-hook-path` | `custom_megatron_post_save_hook` | Post-checkpoint callback (no-op, extend as needed) |

The three shared hooks (`custom_megatron_init`, `custom_model_provider`,
`generate_rollout`) are identical to the vime variants — they are re-exported
from `lumenrl.plugin.vime.hooks`.

## miles vs vime differences

| Aspect | vime | miles |
|---|---|---|
| Rollout engine | vLLM (subprocess) | SGLang (subprocess) |
| Orchestration | Sync (`ray.get()`) | Async (`asyncio.run()`) |
| Extra FP8 hooks | None | `before_train_step` + `before_log_prob` |
| Weight sync | NCCL / IPC / disk | NCCL / P2P RDMA / disk-delta |

The `before_train_step` and `before_log_prob` hooks call
`FP8TrainingManager.reset_fp8_state()` to clear stale FP8 scale statistics
after weight updates — this is critical for FP8 training correctness in the
colocated rollout+training setup.

## Prerequisites

- 8x AMD MI300X / MI325X / MI355X GPUs
- Megatron-LM (ROCm fork, `rocm_dev` branch)
- SGLang
- Model: `Qwen3-8B-Base` downloaded to `$DATA_ROOT/models/`
- Data: `dapo-math-17k.filtered.parquet` in `$DATA_ROOT/data_cached/`

## Docker build and run

```bash
# Build from Lumen-RL repo root
cd /path/to/Lumen-RL
docker build \
  -f examples/plugin/miles/Dockerfile \
  --build-arg MILES_REPO=https://github.com/your-org/miles.git \
  --build-arg MILES_BRANCH=main \
  -t lumenrl-miles-plugin .

# Run 3-step smoke test
docker run --rm -it \
  --device /dev/kfd --device /dev/dri \
  --group-add video --group-add render \
  --shm-size 64g \
  -v /data:/data \
  -e DATA_ROOT=/data \
  -e MILES_ROOT=/workspace/miles \
  lumenrl-miles-plugin

# Run with FP8 training
docker run --rm -it \
  --device /dev/kfd --device /dev/dri \
  --group-add video --group-add render \
  --shm-size 64g \
  -v /data:/data \
  -e DATA_ROOT=/data \
  -e MILES_ROOT=/workspace/miles \
  -e TRAIN_FP8=1 \
  -e STEPS=200 \
  lumenrl-miles-plugin
```

## Run without Docker

```bash
# Install packages
cd /path/to/Megatron-LM && pip install -e .
pip install sglang
cd /path/to/miles && pip install -e .
cd /path/to/Lumen-RL && pip install -e ".[engine,megatron]"

# Verify hooks
python3 -c "from lumenrl.plugin.miles.hooks import custom_megatron_init; print('OK')"

# Launch
DATA_ROOT=/data MILES_ROOT=/path/to/miles bash examples/plugin/miles/run.sh
```

## Hook sequencing

```
miles startup (asyncio.run)
  │
  ├─ Megatron distributed init
  │    └─ custom_megatron_init(args)
  │         → Creates FP8TrainingManager (when TRAIN_FP8=1)
  │
  ├─ Model construction
  │    └─ custom_model_provider(pre_process, post_process)
  │         → GPTModel + Lumen FP8
  │
  └─ Training loop (async)
       ├─ before_train_step_hook(args, rollout_id, step_id, model, ...)
       │    → FP8 scale reset
       ├─ Forward/backward
       ├─ before_log_prob_hook(args, model, prefix)
       │    → FP8 scale reset
       ├─ Weight update → miles native (NCCL/P2P/disk-delta)
       ├─ Rollout → SGLang (or ATOM via --rollout-function-path)
       └─ post_save_hook(args, rollout_id, ckpt_dir, hf_dir)
            → No-op (extend for custom export)
```

## Health check

A successful 3-step smoke should show:

```
LumenRL miles hooks: all 6 hooks importable
```

And training logs with valid loss values and no NaN.
