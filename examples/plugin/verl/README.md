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

## Quick start with pre-built Docker image

A pre-built image is available on Docker Hub with verl 0.9.0, LumenRL plugin,
and all dependencies (aiter, ATOM, Lumen, amdsmi, etc.) for MI300X:

```bash
docker pull zhangdanyangamd/lumen-rl:verlv0.9.0-lumenrl-plugin-example260820
```

### Step 1: Prepare data

Download and filter model/data on the host (one-time):

```bash
export DATA_ROOT=/path/to/data   # e.g. /home/user/models
mkdir -p $DATA_ROOT

# Download model
hf download Qwen/Qwen3-8B-Base --local-dir $DATA_ROOT/Qwen3-8B-Base

# Download datasets
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('BytedTsinghua-SIA/DAPO-Math-17k', repo_type='dataset',
                  local_dir='$DATA_ROOT/raw/DAPO-Math-17k')
snapshot_download('BytedTsinghua-SIA/AIME-2024', repo_type='dataset',
                  local_dir='$DATA_ROOT/raw/AIME-2024')
"

# Filter prompts to <=1024 tokens (see examples/docs/03-data.md for script)
```

### Step 2: Run with volume-mounted code

Mount the latest LumenRL source into the container so code changes take
effect without rebuilding:

```bash
git clone https://github.com/ZhangDanyang-AMD/Lumen-RL.git
cd Lumen-RL

docker run --rm -d \
  --network=host --ipc=host \
  --device /dev/kfd --device /dev/dri \
  --group-add video --group-add render \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  --shm-size 64g \
  --tmpfs /tmp:exec,size=10g \
  -v $DATA_ROOT:/data \
  -v $(pwd):/workspace/Lumen-RL \
  -e DATA_ROOT=/data \
  -e CONFIG=examples/plugin/verl/qwen3_8b_grpo_fp8.yaml \
  -e STEPS=3 \
  --name lumenrl-verl-fp8-test \
  --entrypoint bash \
  zhangdanyangamd/lumen-rl:verlv0.9.0-lumenrl-plugin-example260820 \
  -c 'cd /workspace/Lumen-RL && bash examples/plugin/verl/run.sh'
```

Key docker flags:
- `--network=host`: required for Ray inter-process communication
- `--tmpfs /tmp:exec,size=10g`: clean `/tmp` avoids stale Ray cluster state
- `-v $(pwd):/workspace/Lumen-RL`: mount code for live editing without rebuild
- `--shm-size 64g`: large shared memory for NCCL

Monitor logs:

```bash
docker logs -f lumenrl-verl-fp8-test
```

### Step 3 (optional): Build your own image

```bash
# Build from Lumen-RL repo root
cd /path/to/Lumen-RL
docker build -f examples/plugin/verl/Dockerfile.test \
  -t lumenrl-verl-fp8 .
```

## Run without Docker

```bash
# Install both packages
cd /path/to/verl && pip install -e .
cd /path/to/Lumen-RL && pip install -e ".[engine]"

# Install AMD GPU detection for Ray
pip install amdsmi cachetools

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
| Batch size | 128 | 8 prompts x 16 generations |
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

---

## Known issues and integration notes

### 1. ROCm torch gets replaced by CUDA torch during pip install

**Problem**: Installing `verl`, `tensordict`, `accelerate`, or any package that
depends on `torch>=X` will pull PyPI's CUDA-only `torch` wheel, silently
overwriting the ROCm torch already present in the base image.

**Workaround (Dockerfile.test)**: Backup the ROCm torch directory before
running `pip install`, then restore it afterward:

```dockerfile
# Backup
RUN cp -a /opt/venv/lib/python3.12/site-packages/torch /tmp/_torch_rocm_backup

# Install (pip may pull CUDA torch)
RUN pip install verl 'transformers>=5.5.3'

# Restore ROCm torch
RUN rm -rf /opt/venv/lib/python3.12/site-packages/torch && \
    mv /tmp/_torch_rocm_backup /opt/venv/lib/python3.12/site-packages/torch && \
    rm -rf /opt/venv/lib/python3.12/site-packages/nvidia* \
           /opt/venv/lib/python3.12/site-packages/cuda*
```

Alternatives considered but insufficient:
- `--no-deps`: breaks transitive deps of verl that are actually needed
- `PIP_CONSTRAINT`: pip fails to resolve since the ROCm torch version string
  (e.g. `2.10.0+rocm7.2.4.lw.git3d3aa833`) doesn't satisfy `torch>=2.6`

### 2. `__bases__` assignment fails on some Python builds

**Problem**: `register.py` originally used `LumenRLFSDP2Engine.__bases__ = (BaseEngine,)`
to make the adapter class a subclass of `BaseEngine` at runtime. This fails
with `TypeError: __bases__ assignment: 'BaseEngine' deallocator differs from
'object'` on certain CPython builds.

**Fix**: Create proper subclasses inside `_make_engine_classes()` using the
standard `class _LumenRLFSDP2Engine(VerlBaseEngine): ...` syntax. The function
is called lazily in `register()` to avoid importing verl at module-level.

### 3. verl plugin loader does not call the entry-point function

**Problem**: verl 0.9.0's `__init__.py` does `_ep.load()` which imports the
`register` function via `importlib.metadata`, but does **not** call it
(`_ep.load()()` would be needed). So the plugin entry-point is discovered but
`register()` never executes.

**Fix**: Add `register()` as a module-level call at the bottom of
`register.py`, so the import triggered by `_ep.load()` has the side effect of
running registration:

```python
# At the end of register.py
register()
```

### 4. verl's EngineRegistry vs LumenRL's EngineRegistry

**Problem**: verl has its own `EngineRegistry` at
`verl.workers.engine.base.EngineRegistry`. LumenRL also has one at
`lumenrl.engine.training.base_engine.EngineRegistry`. They are separate
objects. The plugin must register into **verl's** registry, not LumenRL's.

**Fix**: `register.py` imports `from verl.workers.engine.base import
EngineRegistry` and registers there. The `issubclass(cls, BaseEngine)` assert
inside the decorator requires the adapter to inherit from verl's `BaseEngine`,
not LumenRL's.

### 5. verl EngineRegistry does not allow re-registration

**Problem**: `EngineRegistry.register()` has
`assert key not in cls._engines[model_type][current_backend]`. You cannot
overwrite verl's built-in `fsdp2` backend — you must use a distinct name like
`lumenrl_fsdp2`.

**Fix**: Use `lumenrl_fsdp2` / `lumenrl_fsdp` / `lumenrl_megatron` as backend
names in the registry. Set `strategy: lumenrl_fsdp2` (not `fsdp2`) in the yaml
config.

### 6. Hydra config: `strategy` must map to a valid `_target_` class

**Problem**: verl's Hydra config system uses `_target_:
verl.workers.config.FSDPActorConfig` to instantiate the actor config. The
`strategy` field is validated inside `FSDPActorConfig.__post_init__()` and
passed to `EngineRegistry` as the backend name. If your yaml only sets
`strategy: lumenrl_fsdp2` without specifying `_target_`, verl falls back to its
default and may fail.

**Fix**: Use Hydra defaults to inherit verl's full config:

```yaml
defaults:
  - /actor@actor_rollout_ref.actor: dp_actor
  # ... other verl defaults ...
  - _self_

actor_rollout_ref:
  actor:
    strategy: lumenrl_fsdp2
    fsdp_config:
      strategy: lumenrl_fsdp2
```

The `dp_actor` default already includes `_target_: FSDPActorConfig` and all
required fields. The `_self_` directive at the end makes our overrides take
precedence.

### 7. Hydra config file must be in verl's config search path

**Problem**: `python3 -m verl.trainer.main_ppo` uses
`@hydra.main(config_path="config", ...)`, so Hydra only searches
`verl/trainer/config/`. Placing our yaml outside that directory causes
`defaults` references like `/actor@...: dp_actor` to fail resolution.

**Fix**: `run.sh` copies the plugin yaml into verl's config directory at
runtime:

```bash
VERL_CFG_DIR="$(python3 -c "import verl, pathlib; \
    print(pathlib.Path(verl.__file__).parent / 'trainer' / 'config')" 2>/dev/null)"
cp "$CONFIG_ABS" "$VERL_CFG_DIR/_lumenrl_plugin.yaml"

python3 -m verl.trainer.main_ppo --config-name _lumenrl_plugin ...
```

**Important**: Do not use `python3 -c "import verl; ..."` to capture the path
in a shell variable — `import verl` triggers `import apex`, which runs hipify
and dumps dozens of lines to **stdout**, polluting the variable. Use
`pip show verl | grep Location` instead.

### 8. Dockerfile COPY: `cp -a src/ dst/` nests when `dst/` already exists

**Problem**: `cp -a /workspace/Lumen-RL-new/examples /workspace/Lumen-RL/examples`
copies into the existing directory as `examples/examples/` rather than replacing.

**Fix**: Delete the target first:

```dockerfile
RUN rm -rf /workspace/Lumen-RL/examples && \
    cp -a /workspace/Lumen-RL-new/examples /workspace/Lumen-RL/examples
```

### 9. git safe.directory error during editable install

**Problem**: `pip install -e .` in a Docker build fails with
`fatal: detected dubious ownership in repository at '/workspace/Lumen-RL'`
because the COPY changes file ownership.

**Fix**: Run `git config --global --add safe.directory /workspace/Lumen-RL`
before the pip install.

### 10. FSDPEngineConfig hard-codes allowed strategy values

**Problem**: `FSDPEngineConfig.__post_init__` contains
`assert self.strategy in ["fsdp", "fsdp2"]`, and `FSDPActorConfig.__post_init__`
copies `actor.strategy` into `fsdp_config.strategy` via
`object.__setattr__(self.engine, "strategy", self.strategy)`. So any custom
strategy name like `lumenrl_fsdp2` triggers an assertion error.

**Fix**: Remove the hard-coded assert in verl's `engine.py`. In the Dockerfile:

```dockerfile
RUN VERL_ENGINE_CFG="$(pip show verl 2>/dev/null | awk '/^Location:/{print $2}')/verl/workers/config/engine.py" && \
    sed -i 's/assert self\.strategy in \["fsdp", "fsdp2"\].*/# assert removed for plugin extensibility/' "$VERL_ENGINE_CFG"
```

### 11. Ray cannot detect AMD GPUs (0 GPU)

**Problem**: Ray uses `pyamdsmi.smi_get_device_count()` to detect AMD GPUs.
If `amdsmi` is not installed, Ray reports 0 GPUs and verl fails with
`ValueError: Total available GPUs 0 is less than total desired GPUs 8`.

**Fix**: Install `amdsmi`:

```dockerfile
RUN pip install amdsmi
```

In run.sh, also clean stale Ray state and GPU visibility variables (per the
DAPO runbook):

```bash
rm -rf /tmp/ray
unset CUDA_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
unset RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES
```

### 12. verl and LumenRL have different config schemas

**Problem**: verl passes Hydra `DictConfig` objects (with `_target_`, `path`,
etc.) to `EngineRegistry.new()`. LumenRL's `FSDP2Engine.__init__` expects its
own `HFModelConfig`, `FSDPEngineConfig`, `OptimizerConfig` dataclasses with
different field names (e.g. `local_path` vs `path`, nested `LoRAConfig`).

**Fix**: The adapter's `_convert_verl_config()` method maps verl fields to
LumenRL dataclass instances, recursively handling nested dataclasses and
filtering out unknown fields. Key mappings:

- `model_config.path` → `model_config.local_path`
- Strip Hydra-internal fields (`_target_`, `_convert_`, etc.)
- Recursively instantiate nested dataclasses (e.g. `lora` → `LoRAConfig`)

### 13. BaseEngine abstract methods not delegated by `__getattr__`

**Problem**: `__getattr__` only triggers when the attribute is NOT found on the
class or its parents. verl's `BaseEngine` defines `get_data_parallel_rank()`,
`train_batch()`, etc. as concrete methods that `raise NotImplementedError`.
Since they exist on the parent class, `__getattr__` on the adapter never fires,
and the call hits the `raise NotImplementedError` in `BaseEngine`.

**Fix**: Dynamically generate forwarding methods for every `BaseEngine` method
that contains `raise NotImplementedError`, and attach them to the adapter class
at class-creation time:

```python
for name in _delegate_methods:
    def _make_forwarder(n):
        def _forwarder(self, *a, **kw):
            return getattr(self._inner, n)(*a, **kw)
        return _forwarder
    setattr(AdapterClass, name, _make_forwarder(name))
```

### 14. verl v0 vs v1 use different rollout registries

**Problem**: verl v1 trainer uses `_ROLLOUT_REGISTRY` (a dict in
`verl.workers.rollout.base`). verl v0 legacy trainer uses
`RolloutReplicaRegistry` (in `verl.workers.rollout.replica`). The plugin must
register ATOM in both.

**Fix**: `register()` registers in both:
- `_ROLLOUT_REGISTRY[("atom", "async")]` for v1
- `RolloutReplicaRegistry.register("atom", _load_atom)` for v0, where
  `_load_atom` returns `ATOMRolloutReplica` from `lumenrl.plugin.verl.atom_replica`

### 15. verl v1 requires `transfer_queue` (not on PyPI)

**Problem**: verl 0.9.0's v1 trainer (`use_v1: true`) unconditionally imports
`transfer_queue`, which is not published on PyPI.

**Workaround**: Set `trainer.use_v1: false` to use the legacy v0 trainer. The
v0 trainer is deprecated but functional and does not require `transfer_queue`.

### 16. transformers version mismatch

**Problem**: verl 0.9.0 requires `transformers>=5.5.3,<5.11`, but the base
ROCm image may ship an older version (e.g. 5.2.0). Missing model classes
(like `MistralForSequenceClassification`) cause `ModuleNotFoundError` at
`import verl` time.

**Fix**: Explicitly upgrade transformers in the Dockerfile:

```dockerfile
RUN pip install 'transformers>=5.5.3,<5.11,!=5.6.0'
```
