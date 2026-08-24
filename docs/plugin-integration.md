# LumenRL Plugin Integration Guide

This document explains the registration principles and usage of LumenRL's
plugin system for injecting ATOM rollout and Lumen FP8 training into external
RL frameworks.

---

## Architecture overview

```
┌──────────────────────────────────────────────────────────┐
│                    Host RL Framework                      │
│    (verl / vime / miles)                                  │
│                                                           │
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────┐  │
│  │  Training    │    │  Rollout     │    │  Weight     │  │
│  │  Engine      │    │  Engine      │    │  Sync       │  │
│  └──────┬──────┘    └──────┬───────┘    └─────────────┘  │
│         │                  │            (host-native,    │
│         │                  │             NOT injected)    │
└─────────┼──────────────────┼─────────────────────────────┘
          │                  │
          ▼                  ▼
┌─────────────────┐  ┌──────────────────┐
│ Lumen FP8       │  │ ATOM Rollout     │
│ (FP8Training-   │  │ (ATOMReplica-    │
│  Manager)       │  │  Manager)        │
│                 │  │                  │
│ lumenrl/        │  │ lumenrl/engine/  │
│ quantization/   │  │ inference/       │
└─────────────────┘  └──────────────────┘
```

The plugin layer (`lumenrl/plugin/`) sits between the host framework and
LumenRL's core modules. Each plugin file contains only import + forwarding
calls — no business logic.

---

## Registration principles

### 1. Thin wiring only

Plugin files must not implement algorithms, manage state, or duplicate logic
from core LumenRL modules. If a hook grows beyond forwarding a call, the logic
belongs in the core package.

### 2. Weight sync is host-framework-native

LumenRL does **not** inject weight synchronization. Each host framework (verl,
vime, miles) uses its own mechanism (NCCL broadcast, P2P RDMA, disk-delta,
CUDA IPC). ATOM manages its own weight lifecycle through
`sleep` / `wake_up` / `load_weights` on `ATOMRayServer`.

### 3. Function signatures are frozen by the host framework

The hook signatures are dictated by each framework's loader and must match
exactly. Extending them requires a change in the host framework.

### 4. Entry-points vs CLI paths

| Framework | Mechanism | When loaded |
|---|---|---|
| verl | Python packaging entry-points (`importlib.metadata`) | On `import verl` |
| vime | `importlib.import_module` triggered by CLI args | At training startup |
| miles | Same as vime | At training startup |

---

## verl integration

### Registration mechanism

verl discovers plugins via `importlib.metadata.entry_points(group="verl.plugins")`
in `verl/__init__.py`. LumenRL declares:

```toml
# pyproject.toml
[project.entry-points."verl.plugins"]
lumenrl = "lumenrl.plugin.verl.register:register"
```

The `register()` function (in `lumenrl/plugin/verl/register.py`) runs once on
`import verl` and mutates two registries:

**1. Rollout registry** — `_ROLLOUT_REGISTRY` is a plain `dict` in
`verl/workers/rollout/base.py`:

```python
_ROLLOUT_REGISTRY[("atom", "async")] = "lumenrl.plugin.verl.register.ATOMRolloutAdapter"
```

verl resolves the FQDN string lazily via `importlib.import_module` when the
trainer requests the `"atom"` rollout backend.

**2. Engine registry** — uses `@EngineRegistry.register()` decorator in
`verl/workers/engine/base.py`:

```python
# FSDP2 engine (with Lumen FP8 support)
EngineRegistry.register(
    model_type="language_model",
    backend=["lumenrl_fsdp", "lumenrl_fsdp2"],
    device=["cuda"],
)(LumenRLFSDP2Engine)

# Megatron engine
EngineRegistry.register(
    model_type="language_model",
    backend="lumenrl_megatron",
    device=["cuda"],
)(LumenRLMegatronEngine)
```

### Classes

| Class | Role |
|---|---|
| `ATOMRolloutAdapter` | Implements verl's `BaseRollout` ABC; forwards `resume()` → `wake_all()`, `release()` → `sleep_all()`, `update_weights()` → no-op |
| `LumenRLFSDP2Engine` | Wraps `lumenrl.engine.training.fsdp_engine.FSDP2EngineWithLMHead`; supports FP8 via `quant_config` |
| `LumenRLMegatronEngine` | Wraps `lumenrl.engine.training.megatron_engine.MegatronEngineWithLMHead`; delegates via `__getattr__` |

### Usage

```yaml
# verl hydra config — FSDP2 training + ATOM rollout
actor_rollout_ref:
  rollout:
    name: atom
    mode: async
trainer:
  engine:
    backend: lumenrl_fsdp2      # or lumenrl_fsdp
```

```yaml
# verl hydra config — Megatron training + ATOM rollout
actor_rollout_ref:
  rollout:
    name: atom
    mode: async
trainer:
  engine:
    backend: lumenrl_megatron
```

### Control

```bash
# Load only lumenrl (ignore other plugins):
export VERL_USE_EXTERNAL_PLUGINS=lumenrl

# Disable all external plugins:
export VERL_USE_EXTERNAL_PLUGINS=none
```

---

## vime integration

### Registration mechanism

vime resolves hook functions at runtime using
`vime.utils.misc.load_function(path)`, which calls
`importlib.import_module(module_path)` and `getattr(module, attr)`. Each hook
must be a top-level function in its module.

### Hook reference

| CLI argument | Dotted path | Signature |
|---|---|---|
| `--custom-megatron-init-path` | `lumenrl.plugin.vime.hooks.custom_megatron_init` | `(args: Namespace) -> None` |
| `--custom-model-provider-path` | `lumenrl.plugin.vime.hooks.custom_model_provider` | `(pre_process: bool, post_process: bool, vp_stage: int \| None = None) -> GPTModel` |
| `--rollout-function-path` | `lumenrl.plugin.vime.hooks.generate_rollout` | `(args, rollout_id: int, data_source, evaluation: bool = False) -> RolloutFnTrainOutput \| RolloutFnEvalOutput` |

### Hook sequencing

```
vime startup
  │
  ├─ Megatron distributed init
  │    └─ custom_megatron_init(args)           ← creates FP8TrainingManager
  │         stores on args._lumenrl_fp8_manager
  │
  ├─ Model construction
  │    └─ custom_model_provider(...)           ← builds GPTModel, applies FP8
  │         reads args._lumenrl_fp8_manager
  │
  └─ Training loop
       └─ generate_rollout(args, ...)          ← drives ATOM rollout
            reads args._lumenrl_atom_manager
```

### Prerequisites

Before training starts, the trainer script must set up the ATOM manager:

```python
from lumenrl.engine.inference.atom_ray_server import ATOMReplicaManager

args._lumenrl_atom_manager = ATOMReplicaManager(
    actor_wg=my_worker_group,
    model_name="/path/to/model",
    engine_kwargs={"tensor_parallel_size": 8},
)
args._lumenrl_atom_manager.create()
```

For FP8 training, set `args.lumenrl_fp8_config` before Megatron init:

```python
from lumenrl.core.config import QuantizationConfig, TrainingQuantConfig

args.lumenrl_fp8_config = QuantizationConfig(
    training=TrainingQuantConfig(fp8_recipe="blockwise2d"),
)
```

---

## miles integration

### Registration mechanism

Identical to vime — `miles.utils.misc.load_function()` with the same semantics.

### Hook reference

miles re-exports the three shared hooks from vime unchanged:

| CLI argument | Dotted path | Same as vime? |
|---|---|---|
| `--custom-megatron-init-path` | `lumenrl.plugin.miles.hooks.custom_megatron_init` | Yes |
| `--custom-model-provider-path` | `lumenrl.plugin.miles.hooks.custom_model_provider` | Yes |
| `--rollout-function-path` | `lumenrl.plugin.miles.hooks.generate_rollout` | Yes |

Three additional hooks are miles-only:

| CLI argument | Dotted path | Signature |
|---|---|---|
| `--custom-megatron-post-save-hook-path` | `...miles.hooks.custom_megatron_post_save_hook` | `(args, rollout_id: int, checkpoint_dir: str, hf_checkpoint_dir: str) -> None` |
| `--custom-megatron-before-train-step-hook-path` | `...miles.hooks.custom_megatron_before_train_step_hook` | `(args, rollout_id: int, step_id: int, model, optimizer, opt_param_scheduler) -> None` |
| `--custom-megatron-before-log-prob-hook-path` | `...miles.hooks.custom_megatron_before_log_prob_hook` | `(args, model, store_prefix: str) -> None` |

### What the miles-only hooks do

- **`before_train_step_hook`** — calls `FP8TrainingManager.reset_fp8_state()`
  on each model chunk before every micro-step. This clears stale FP8 scale
  statistics after weight updates.

- **`before_log_prob_hook`** — same FP8 reset, but before the forward-only
  log-prob evaluation pass.

- **`post_save_hook`** — no-op by default. Override for custom checkpoint
  export or upload.

---

## Injection scope summary

| Component | verl | vime | miles |
|---|---|---|---|
| **Lumen FSDP2 training** | `LumenRLFSDP2Engine` (`lumenrl_fsdp2`) | N/A (Megatron-only) | N/A (Megatron-only) |
| **Lumen Megatron training** | `LumenRLMegatronEngine` (`lumenrl_megatron`) | `custom_megatron_init` + `custom_model_provider` | Same as vime + `before_train_step` / `before_log_prob` hooks |
| **ATOM rollout** | `ATOMRolloutAdapter` in `_ROLLOUT_REGISTRY` | `generate_rollout` | `generate_rollout` |
| **Weight sync** | Not injected | Not injected | Not injected |
| **Megatron bridge** | Inside engine adapter | Inside `custom_model_provider` | Inside `custom_model_provider` |

---

## Adding a new framework

To support a fourth framework:

1. Create `lumenrl/plugin/<framework>/__init__.py` and a hooks module.

2. Identify the framework's hook discovery mechanism:
   - Entry-points? → Add to `pyproject.toml` and write a `register()` function.
   - CLI path? → Write functions matching the exact expected signatures.
   - Hardcoded import? → Create the expected package/module name.

3. Write one function per hook. Each body should be a single delegation call
   to an existing `lumenrl.*` module.

4. Add usage instructions to this document and `lumenrl/plugin/README.md`.

---

## Source code reference

| File | What |
|---|---|
| `lumenrl/plugin/verl/register.py` | verl entry-point, `ATOMRolloutAdapter`, `LumenRLMegatronEngine` |
| `lumenrl/plugin/vime/hooks.py` | `custom_megatron_init`, `custom_model_provider`, `generate_rollout` |
| `lumenrl/plugin/miles/hooks.py` | Re-exports + `post_save`, `before_train_step`, `before_log_prob` hooks |
| `lumenrl/quantization/fp8_training.py` | `FP8TrainingManager` (core FP8 logic) |
| `lumenrl/engine/inference/atom_ray_server.py` | `ATOMReplicaManager`, `ATOMRayServer` (core ATOM logic) |
| `lumenrl/engine/training/megatron_engine.py` | `MegatronEngine`, `MegatronEngineWithLMHead` |
