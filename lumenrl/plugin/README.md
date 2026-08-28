# LumenRL Plugin Adapters

Thin wiring code that injects LumenRL capabilities (ATOM rollout, Lumen FP8
training) into external RL frameworks. All business logic lives in the main
`lumenrl/` package — each file here is glue only.

## Directory layout

```
lumenrl/plugin/
├── verl/
│   └── register.py   # entry-point auto-loaded by verl on import
├── vime/
│   └── hooks.py      # CLI hook functions for --custom-*-path
└── miles/
    └── hooks.py      # CLI hook functions (superset of vime)
```

---

## verl

### How it works

verl auto-discovers packages that declare an entry-point in the `verl.plugins`
group. Installing LumenRL (`pip install -e .`) triggers this automatically:

```toml
# pyproject.toml (already configured)
[project.entry-points."verl.plugins"]
lumenrl = "lumenrl.plugin.verl.register:register"
```

On `import verl`, verl calls `register()` which injects:

| Target | verl key | What |
|---|---|---|
| `_ROLLOUT_REGISTRY` | `("atom", "async")` | ATOM rollout adapter |
| `EngineRegistry` | `backend="lumenrl_fsdp2"` | LumenRL FSDP2 engine (+ FP8) |
| `EngineRegistry` | `backend="lumenrl_megatron"` | LumenRL Megatron engine |

### Configuration

```yaml
# verl trainer config — FSDP2 variant
actor_rollout_ref:
  rollout:
    name: atom
trainer:
  engine:
    backend: lumenrl_fsdp2    # or lumenrl_fsdp (alias)
```

```yaml
# verl trainer config — Megatron variant
actor_rollout_ref:
  rollout:
    name: atom
trainer:
  engine:
    backend: lumenrl_megatron
```

### Weight sync

Not injected. ATOM manages its own weights via `sleep`/`wake_up`.
`ATOMRolloutAdapter.update_weights()` is a no-op.

### Disable

```bash
export VERL_USE_EXTERNAL_PLUGINS=none
```

---

## vime

Pass hook functions by their dotted import path via CLI arguments:

```bash
python train.py \
  --custom-megatron-init-path  lumenrl.plugin.vime.hooks.custom_megatron_init \
  --custom-model-provider-path lumenrl.plugin.vime.hooks.custom_model_provider \
  --rollout-function-path      lumenrl.plugin.vime.hooks.generate_rollout
```

### Hook reference

| CLI argument | Function | What it does |
|---|---|---|
| `--custom-megatron-init-path` | `custom_megatron_init` | Creates `FP8TrainingManager`, stores on `args` |
| `--custom-model-provider-path` | `custom_model_provider` | Builds GPTModel, applies FP8 if configured |
| `--rollout-function-path` | `generate_rollout` | Replaces vLLM rollout with ATOM |

### FP8 setup

Set `args.lumenrl_fp8_config` before training starts:

```python
args.lumenrl_fp8_config = {"training": {"fp8_recipe": "delayed"}}
```

### ATOM rollout prerequisite

Set `args._lumenrl_atom_manager` before the first rollout call:

```python
from lumenrl.engine.inference.atom_ray_server import ATOMReplicaManager

args._lumenrl_atom_manager = ATOMReplicaManager(
    actor_wg=worker_group,
    model_name="/path/to/model",
    engine_kwargs={"tensor_parallel_size": 8},
)
args._lumenrl_atom_manager.create()
```

---

## miles

miles is a fork of vime with SGLang rollout and additional hooks. The three
shared hooks are identical to vime and re-exported directly.

```bash
python train.py \
  --custom-megatron-init-path  lumenrl.plugin.miles.hooks.custom_megatron_init \
  --custom-model-provider-path lumenrl.plugin.miles.hooks.custom_model_provider \
  --rollout-function-path      lumenrl.plugin.miles.hooks.generate_rollout \
  --custom-megatron-before-train-step-hook-path \
      lumenrl.plugin.miles.hooks.custom_megatron_before_train_step_hook \
  --custom-megatron-before-log-prob-hook-path \
      lumenrl.plugin.miles.hooks.custom_megatron_before_log_prob_hook \
  --custom-megatron-post-save-hook-path \
      lumenrl.plugin.miles.hooks.custom_megatron_post_save_hook
```

### miles-only hooks

| CLI argument | Function | What it does |
|---|---|---|
| `--custom-megatron-before-train-step-hook-path` | `custom_megatron_before_train_step_hook` | Resets FP8 state before each micro-step |
| `--custom-megatron-before-log-prob-hook-path` | `custom_megatron_before_log_prob_hook` | Resets FP8 state before log-prob pass |
| `--custom-megatron-post-save-hook-path` | `custom_megatron_post_save_hook` | No-op; extend for custom export |

---

## Design principles

1. **No adapter logic** — plugin files only import and forward; all real
   computation stays in `lumenrl/`.

2. **Weight sync is host-native** — each framework uses its own mechanism
   (NCCL, P2P, disk). ATOM handles its own weight lifecycle.

3. **Frozen signatures** — hook function signatures are dictated by the host
   framework and must not change.

For the full design rationale, see [docs/plugin-integration.md](../docs/plugin-integration.md).
