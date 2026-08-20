# vime + LumenRL Plugin: Qwen3-8B Megatron

Run Qwen3-8B RL training on **vime** with LumenRL's Megatron bridge and
optional Lumen FP8, injected via vime's `--custom-*-path` CLI hooks.

## What the plugin does

vime loads hook functions at runtime via `importlib.import_module`. LumenRL
provides three hooks in `lumenrl.plugin.vime.hooks`:

| CLI argument | Hook function | What it does |
|---|---|---|
| `--custom-megatron-init-path` | `custom_megatron_init` | Creates `FP8TrainingManager` and stores on `args` |
| `--custom-model-provider-path` | `custom_model_provider` | Builds Megatron GPTModel, applies Lumen FP8 |
| `--custom-generate-function-path` | `generate_rollout` | Replaces vLLM rollout with ATOM (optional) |

The hooks are called during vime's standard startup sequence — no source
modification to vime is required.

## Prerequisites

- 8x AMD MI300X / MI325X / MI355X GPUs
- Megatron-LM (ROCm fork, `rocm_dev` branch)
- Model: `Qwen3-8B-Base` downloaded to `$DATA_ROOT/models/`
- Data: `dapo-math-17k.filtered.parquet` in `$DATA_ROOT/data_cached/`

## Docker build and run

```bash
# Build from Lumen-RL repo root
cd /path/to/Lumen-RL
docker build \
  -f examples/plugin/vime/Dockerfile \
  --build-arg VIME_REPO=https://github.com/your-org/vime.git \
  --build-arg VIME_BRANCH=main \
  -t lumenrl-vime-plugin .

# Run 3-step smoke test
docker run --rm -it \
  --device /dev/kfd --device /dev/dri \
  --group-add video --group-add render \
  --shm-size 64g \
  -v /data:/data \
  -e DATA_ROOT=/data \
  -e VIME_ROOT=/workspace/vime \
  lumenrl-vime-plugin

# Run with FP8 training enabled
docker run --rm -it \
  --device /dev/kfd --device /dev/dri \
  --group-add video --group-add render \
  --shm-size 64g \
  -v /data:/data \
  -e DATA_ROOT=/data \
  -e VIME_ROOT=/workspace/vime \
  -e TRAIN_FP8=1 \
  -e STEPS=200 \
  lumenrl-vime-plugin
```

## Run without Docker

```bash
# Install packages
cd /path/to/Megatron-LM && pip install -e .
cd /path/to/vime && pip install -e .
cd /path/to/Lumen-RL && pip install -e ".[engine,megatron]"

# Verify hooks
python3 -c "from lumenrl.plugin.vime.hooks import custom_megatron_init; print('OK')"

# Launch
DATA_ROOT=/data VIME_ROOT=/path/to/vime bash examples/plugin/vime/run.sh
```

## Hook sequencing

```
vime startup
  │
  ├─ Megatron distributed init
  │    └─ custom_megatron_init(args)
  │         → Creates FP8TrainingManager (when TRAIN_FP8=1)
  │         → Stores on args._lumenrl_fp8_manager
  │
  ├─ Model construction
  │    └─ custom_model_provider(pre_process, post_process)
  │         → Builds GPTModel with LumenRL's transformer config
  │         → Calls fp8_manager.enable(model) if FP8 is active
  │
  └─ Training loop
       ├─ Weight update → vime native (NCCL broadcast)
       └─ Rollout → vime native vLLM (or ATOM via --custom-generate-function-path)
```

## Health check

A successful 3-step smoke should show:

```
LumenRL vime hooks: importable and signature-verified
```

And Megatron training logs with valid loss values and no NaN.
