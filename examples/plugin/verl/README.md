# verl + LumenRL Plugin: Qwen3-8B GRPO

Run Qwen3-8B GRPO math RL training on **verl** with LumenRL's FSDP2 training
engine and ATOM rollout, injected automatically via the `verl.plugins`
entry-point.

## What the plugin does

On `import verl`, verl auto-discovers LumenRL's entry-point and calls
`lumenrl.plugin.verl.register:register()`, which:

1. Inserts `("atom", "async")` into verl's `_ROLLOUT_REGISTRY` — the ATOM
   rollout adapter that manages `sleep`/`wake_up`/`generate` lifecycle.
2. Registers `lumenrl_fsdp2` (and `lumenrl_fsdp`) in verl's `EngineRegistry`
   — wraps LumenRL's `FSDP2Engine` with optional Lumen FP8 blockwise2d.
3. Registers `lumenrl_megatron` in verl's `EngineRegistry` — wraps LumenRL's
   `MegatronEngine`.

No monkey-patching. No manual registration code needed.

## Prerequisites

- 8x AMD MI300X / MI325X / MI355X GPUs
- Model: `Qwen3-8B-Base` downloaded to `$DATA_ROOT/models/`
- Data: `dapo-math-17k.filtered.parquet` in `$DATA_ROOT/data_cached/`

## Docker build and run

```bash
# Build from Lumen-RL repo root
cd /path/to/Lumen-RL
docker build -f examples/plugin/verl/Dockerfile -t lumenrl-verl-plugin .

# Run (mount data directory)
docker run --rm -it \
  --device /dev/kfd --device /dev/dri \
  --group-add video --group-add render \
  --shm-size 64g \
  -v /data:/data \
  -e DATA_ROOT=/data \
  lumenrl-verl-plugin
```

The container runs a 3-step smoke test by default. For a longer run:

```bash
docker run --rm -it \
  --device /dev/kfd --device /dev/dri \
  --group-add video --group-add render \
  --shm-size 64g \
  -v /data:/data \
  -e DATA_ROOT=/data \
  -e STEPS=200 \
  lumenrl-verl-plugin
```

## Run without Docker

```bash
# Install both packages
cd /path/to/verl && pip install -e .
cd /path/to/Lumen-RL && pip install -e ".[engine]"

# Verify plugin
python3 -c "import verl; from verl.workers.rollout.base import _ROLLOUT_REGISTRY; print(('atom','async') in _ROLLOUT_REGISTRY)"

# Launch
DATA_ROOT=/data bash examples/plugin/verl/run.sh
```

## Configuration

The config `qwen3_8b_grpo.yaml` sets:

| Parameter | Value | Notes |
|---|---|---|
| Training backend | `lumenrl_fsdp2` | LumenRL FSDP2 engine |
| Rollout backend | `atom` | ATOM with sleep/wake lifecycle |
| Batch size | 128 | 8 prompts × 16 generations |
| Learning rate | 1e-6 | Constant with 10-step warmup |
| Max response length | 2048 | Smoke setting |

## Health check

A successful 3-step smoke run should show:

```
LumenRL plugin: ATOM rollout registered
LumenRL plugin: lumenrl_fsdp2 engine registered
```

And training logs with decreasing `actor/loss`, non-zero `actor/entropy`, and
no NaN in any metric.
