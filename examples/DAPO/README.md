# DAPO

`run_dapo.sh` is the single supported shell entrypoint for DAPO experiments.
It launches the native LumenRL trainer and does not patch or overwrite an
installed verl package.

## Modes

| `MODE` | Rollout | Training |
| --- | --- | --- |
| `bf16` | vLLM BF16 | BF16 |
| `fp8` | vLLM per-block FP8 | BF16 or FP8 with `TRAIN_FP8=1` |
| `atomfp8` | ATOM per-block FP8 | BF16 or FP8 with `TRAIN_FP8=1` |

The default long-run configs are:

- `configs/dapo_qwen3_8b_ray_vllm_longrun.yaml`
- `configs/dapo_qwen3_8b_ray_vllm_fp8_longrun.yaml`
- `configs/dapo_qwen3_8b_ray_atom_fp8_longrun.yaml`

Megatron-Native and smoke configs can be selected with `CONFIG_OVERRIDE`.

## Required environment

```bash
export RL_ROOT=/absolute/path/to/lumen_rl
export DATA_ROOT=/absolute/path/to/data
```

Expected layout:

```text
$RL_ROOT/
├── Lumen-RL/
├── Lumen/
├── aiter/
└── ATOM/                 # only required for MODE=atomfp8

$DATA_ROOT/
├── models/Qwen3-8B-Base/
├── data_cached/qwen3-8b-maxprompt1024/
│   ├── dapo-math-17k.filtered.parquet
│   └── aime-2024.filtered.parquet
├── logs/
├── ckpts/
└── wandb/
```

## Smoke

Use a smoke config explicitly; changing only `STEPS` does not reduce the
long-run sequence and batch sizes.

```bash
S="$RL_ROOT/Lumen-RL/examples/DAPO/run_dapo.sh"

CONFIG_OVERRIDE=examples/DAPO/configs/dapo_qwen3_8b_ray_vllm_smoke.yaml \
STEPS=2 \
MODE=bf16 \
LOG="$DATA_ROOT/logs/dapo-smoke-bf16.log" \
bash "$S"
```

Megatron-Native DP2×TP2×PP2:

```bash
CONFIG_OVERRIDE=examples/DAPO/configs/dapo_qwen3_8b_ray_megatron_smoke.yaml \
STEPS=2 \
MODE=bf16 \
EXTRA_OVERRIDE="policy.training_backend=megatron_native \
policy.training.megatron_cfg.tensor_model_parallel_size=2 \
policy.training.megatron_cfg.pipeline_model_parallel_size=2 \
policy.training.megatron_cfg.context_parallel_size=1 \
checkpointing.resume=false \
checkpointing.checkpoint_dir=$DATA_ROOT/ckpts/lumenrl-dapo/native-smoke" \
LOG="$DATA_ROOT/logs/dapo-smoke-megatron-native.log" \
bash "$S"
```

## Long run

```bash
# FSDP2 + vLLM BF16
STEPS=1000 MODE=bf16 bash "$RL_ROOT/Lumen-RL/examples/DAPO/run_dapo.sh"

# FSDP2 + vLLM FP8 rollout
STEPS=1000 MODE=fp8 bash "$RL_ROOT/Lumen-RL/examples/DAPO/run_dapo.sh"

# FSDP2 + ATOM FP8 rollout
STEPS=1000 MODE=atomfp8 bash "$RL_ROOT/Lumen-RL/examples/DAPO/run_dapo.sh"
```

For a detached container run:

```bash
docker exec -d \
  -e RL_ROOT="$RL_ROOT" \
  -e DATA_ROOT="$DATA_ROOT" \
  <container> bash -lc \
  'STEPS=1000 MODE=bf16 bash "$RL_ROOT/Lumen-RL/examples/DAPO/run_dapo.sh"'
```

## Overrides

Common environment variables:

- `STEPS`
- `MODE`
- `TRAIN_FP8`
- `CONFIG_OVERRIDE`
- `MODEL_PATH`
- `TRAIN_FILE`
- `VAL_FILE`
- `RUN_ID`
- `LOG`
- `EXTRA_OVERRIDE`

`EXTRA_OVERRIDE` is passed as OmegaConf CLI overrides. Paths in it must already
be expanded absolute paths.

## Monitoring

```bash
grep -aE "callbacks: step=" "$DATA_ROOT/logs/<run>.log" | tail -5
grep -aiE "Traceback|OOM|OutOfMemory|NaN" "$DATA_ROOT/logs/<run>.log"
```

Healthy BF16 runs have finite entropy and grad norm, `ppo_kl` near zero, and no
OOM/NaN/traceback.

## Active support files

```text
examples/DAPO/
├── README.md
├── run_dapo.sh
├── atom_aiter_shim/sitecustomize.py
└── configs/
    ├── dapo_qwen3_8b_ray_vllm_longrun.yaml
    ├── dapo_qwen3_8b_ray_vllm_fp8_longrun.yaml
    ├── dapo_qwen3_8b_ray_atom_fp8_longrun.yaml
    ├── dapo_qwen3_8b_ray_atom_fp8_debug.yaml
    ├── dapo_qwen3_8b_ray_megatron_longrun.yaml
    └── *_smoke.yaml
```
