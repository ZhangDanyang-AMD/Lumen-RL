# LumenRL Plugin Adapters

Thin wiring code that injects LumenRL capabilities (ATOM rollout, Lumen FP8
training) into external RL frameworks. All business logic lives in the main
`lumenrl/` package — each file here is glue only.

## Directory layout

```
lumenrl/plugin/
├── verl/
│   ├── register.py      # entry-point auto-loaded by verl on import
│   ├── atom_replica.py   # ATOMRolloutReplica for verl async server mode
│   └── lifecycle.py      # ATOM manager setup/teardown helpers
├── vime/
│   ├── hooks.py          # CLI hook functions for --custom-*-path
│   └── lifecycle.py      # ATOM manager setup/teardown helpers
├── miles/
│   ├── hooks.py          # CLI hook functions (independent, no vime imports)
│   └── lifecycle.py      # ATOM manager setup/teardown helpers
└── slime/
    ├── hooks.py          # CLI hook functions (independent, no vime imports)
    └── lifecycle.py      # ATOM manager setup/teardown helpers
```

**Independence**: Each Megatron-based plugin (vime, miles, slime) is fully
self-contained. There are no cross-plugin imports — all hook implementations
are inlined.

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

Registration uses `importlib.util.find_spec()` pre-checks instead of
`try/except ImportError`, so missing optional dependencies are handled
cleanly without hiding real import errors.

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

`ATOMRolloutAdapter.update_weights()` collects weight tensors from the
training engine, writes them to a temporary safetensors file, then calls
`reload_weights_from_path()` on each ATOM server to load the new weights.

For `ATOMRolloutReplica` (async server mode), the ATOM servers are managed
by the replica itself — `CheckpointEngineWorker` creates its own
`server_adapter` which handles weight sync internally.

### ATOM lifecycle

```python
from lumenrl.plugin.verl.lifecycle import setup_atom_manager, teardown_atom_manager

manager = setup_atom_manager(rollout_adapter, actor_worker_group, model_path, engine_kwargs)
try:
    # ... training loop ...
finally:
    teardown_atom_manager(rollout_adapter)
```

### Disable

```bash
export VERL_USE_EXTERNAL_PLUGINS=none
```

---

## vime

Pass hook functions by their dotted import path via CLI arguments:

```bash
python train.py \
  --custom-megatron-init-path              lumenrl.plugin.vime.hooks.custom_megatron_init \
  --custom-model-provider-path             lumenrl.plugin.vime.hooks.custom_model_provider \
  --rollout-function-path                  lumenrl.plugin.vime.hooks.generate_rollout \
  --custom-megatron-post-save-hook-path    lumenrl.plugin.vime.hooks.custom_megatron_post_save_hook \
  --custom-megatron-before-train-step-hook-path lumenrl.plugin.vime.hooks.custom_megatron_before_train_step_hook \
  --custom-megatron-before-log-prob-hook-path   lumenrl.plugin.vime.hooks.custom_megatron_before_log_prob_hook \
  --custom-update-weight-post-write-path   lumenrl.plugin.vime.hooks.custom_update_weight_post_write
```

### Hook reference

| CLI argument | Function | What it does |
|---|---|---|
| `--custom-megatron-init-path` | `custom_megatron_init` | Creates `FP8TrainingManager`, stores on `args` |
| `--custom-model-provider-path` | `custom_model_provider` | Builds GPTModel, applies FP8 if configured |
| `--rollout-function-path` | `generate_rollout` | Replaces vLLM rollout with ATOM |
| `--custom-megatron-post-save-hook-path` | `custom_megatron_post_save_hook` | No-op; extend for custom checkpoint export |
| `--custom-megatron-before-train-step-hook-path` | `custom_megatron_before_train_step_hook` | Resets FP8 state before each micro-step |
| `--custom-megatron-before-log-prob-hook-path` | `custom_megatron_before_log_prob_hook` | Resets FP8 state before log-prob pass |
| `--custom-update-weight-post-write-path` | `custom_update_weight_post_write` | Reloads ATOM weights from newly written HF checkpoint |

### FP8 setup

Set `args.lumenrl_fp8_config` before training starts:

```python
args.lumenrl_fp8_config = {"training": {"fp8_recipe": "delayed"}}
```

### ATOM lifecycle

```python
from lumenrl.plugin.vime.lifecycle import setup_atom_manager, teardown_atom_manager

setup_atom_manager(args, actor_worker_group)
try:
    # ... training loop ...
finally:
    teardown_atom_manager(args)
```

`setup_atom_manager` creates an `ATOMReplicaManager`, stores it on
`args._lumenrl_atom_manager`, and launches ATOM servers colocated with the
actor worker group.

### Weight sync

The `custom_update_weight_post_write` hook is called by the Megatron weight
updater after writing HF-format weights to disk. Only rank 0 triggers the
ATOM reload — servers broadcast via their TP groups.

---

## miles

miles uses independent hook implementations — no imports from vime or slime.

```bash
python train.py \
  --custom-megatron-init-path              lumenrl.plugin.miles.hooks.custom_megatron_init \
  --custom-model-provider-path             lumenrl.plugin.miles.hooks.custom_model_provider \
  --rollout-function-path                  lumenrl.plugin.miles.hooks.generate_rollout \
  --custom-megatron-post-save-hook-path    lumenrl.plugin.miles.hooks.custom_megatron_post_save_hook \
  --custom-megatron-before-train-step-hook-path lumenrl.plugin.miles.hooks.custom_megatron_before_train_step_hook \
  --custom-megatron-before-log-prob-hook-path   lumenrl.plugin.miles.hooks.custom_megatron_before_log_prob_hook \
  --custom-update-weight-post-write-path   lumenrl.plugin.miles.hooks.custom_update_weight_post_write
```

### Hook reference

| CLI argument | Function | What it does |
|---|---|---|
| `--custom-megatron-init-path` | `custom_megatron_init` | Creates `FP8TrainingManager`, stores on `args` |
| `--custom-model-provider-path` | `custom_model_provider` | Builds GPTModel, applies FP8 if configured |
| `--rollout-function-path` | `generate_rollout` | Replaces sglang rollout with ATOM; uses `miles.rollout.base_types` |
| `--custom-megatron-post-save-hook-path` | `custom_megatron_post_save_hook` | No-op; extend for custom checkpoint export |
| `--custom-megatron-before-train-step-hook-path` | `custom_megatron_before_train_step_hook` | Resets FP8 state before each micro-step |
| `--custom-megatron-before-log-prob-hook-path` | `custom_megatron_before_log_prob_hook` | Resets FP8 state before log-prob pass |
| `--custom-update-weight-post-write-path` | `custom_update_weight_post_write` | Reloads ATOM weights from newly written HF checkpoint |

### ATOM lifecycle

```python
from lumenrl.plugin.miles.lifecycle import setup_atom_manager, teardown_atom_manager

setup_atom_manager(args, actor_worker_group)
try:
    # ... training loop ...
finally:
    teardown_atom_manager(args)
```

---

## slime

slime uses independent hook implementations — no imports from vime or miles.

```bash
python train.py \
  --custom-megatron-init-path              lumenrl.plugin.slime.hooks.custom_megatron_init \
  --custom-model-provider-path             lumenrl.plugin.slime.hooks.custom_model_provider \
  --rollout-function-path                  lumenrl.plugin.slime.hooks.generate_rollout \
  --custom-megatron-post-save-hook-path    lumenrl.plugin.slime.hooks.custom_megatron_post_save_hook \
  --custom-megatron-before-train-step-hook-path lumenrl.plugin.slime.hooks.custom_megatron_before_train_step_hook \
  --custom-megatron-before-log-prob-hook-path   lumenrl.plugin.slime.hooks.custom_megatron_before_log_prob_hook \
  --custom-update-weight-post-write-path   lumenrl.plugin.slime.hooks.custom_update_weight_post_write
```

### Hook reference

| CLI argument | Function | What it does |
|---|---|---|
| `--custom-megatron-init-path` | `custom_megatron_init` | Creates `FP8TrainingManager`, stores on `args` |
| `--custom-model-provider-path` | `custom_model_provider` | Builds GPTModel, applies FP8 if configured |
| `--rollout-function-path` | `generate_rollout` | Replaces sglang rollout with ATOM; uses `slime.rollout.base_types` |
| `--custom-megatron-post-save-hook-path` | `custom_megatron_post_save_hook` | No-op; extend for custom checkpoint export |
| `--custom-megatron-before-train-step-hook-path` | `custom_megatron_before_train_step_hook` | Resets FP8 state before each micro-step |
| `--custom-megatron-before-log-prob-hook-path` | `custom_megatron_before_log_prob_hook` | Resets FP8 state before log-prob pass |
| `--custom-update-weight-post-write-path` | `custom_update_weight_post_write` | Reloads ATOM weights from newly written HF checkpoint |

### ATOM lifecycle

```python
from lumenrl.plugin.slime.lifecycle import setup_atom_manager, teardown_atom_manager

setup_atom_manager(args, actor_worker_group)
try:
    # ... training loop ...
finally:
    teardown_atom_manager(args)
```

---

## Design principles

1. **No adapter logic** — plugin files only import and forward; all real
   computation stays in `lumenrl/`.

2. **Plugin independence** — vime, miles, and slime plugins are fully
   self-contained with no cross-plugin imports. Each framework has its own
   `generate_rollout` returning framework-native types
   (`{framework}.rollout.base_types.RolloutFnTrainOutput`).

3. **Weight sync is host-native** — each Megatron framework writes HF weights
   to disk, then `custom_update_weight_post_write` triggers ATOM reload.
   verl uses tensor-level sync via `ATOMRolloutAdapter.update_weights()`.

4. **ATOM lifecycle helpers** — each plugin provides `lifecycle.py` with
   `setup_atom_manager` / `teardown_atom_manager` for consistent
   `ATOMReplicaManager` creation and teardown.

5. **Frozen signatures** — hook function signatures are dictated by the host
   framework and must not change.

6. **Clean conditional imports** — verl/register.py uses
   `importlib.util.find_spec()` instead of `try/except ImportError`.

For the full design rationale, see [docs/plugin-integration.md](../docs/plugin-integration.md).
