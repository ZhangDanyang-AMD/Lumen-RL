# verl + LumenRL Plugin: Qwen3-8B GRPO

Tested: verl 0.9.0 with LumenRL `lumenrl_fsdp2` training engine and vLLM BF16
rollout, 3-step GRPO training on Qwen3-8B-Base, 8x MI300X GPUs.

## What the plugin registers

On `import verl`, verl auto-discovers the LumenRL entry-point and runs
`lumenrl.plugin.verl.register:register()`, which:

1. **`lumenrl_fsdp2` training engine** in verl's `EngineRegistry` -- wraps
   LumenRL's `FSDP2Engine` as a verl `BaseEngine` subclass.
2. **`atom` rollout** in verl's `RolloutReplicaRegistry` -- ATOM rollout
   adapter (future ATOM support; not used in the tested config).
3. **vLLM rollout** works out of the box via verl's native vLLM support
   (`rollout.name: vllm` in the yaml).

No monkey-patching. No manual registration code.

## Prerequisites

- 8x AMD MI300X GPUs (MI325X / MI355X should also work)
- Host storage for model weights and dataset

## Quick start

### Option A: Docker with volume-mounted code (tested)

The tested base image is `vllm/vllm-openai-rocm:v0.23.0` with verl and
LumenRL installed on top. A pre-built image is on Docker Hub:

```
zhangdanyangamd/lumen-rl:verlv0.9.0-lumenrl-plugin-example260820
```

We recommend building from `Dockerfile.test` or volume-mounting your code
into the pre-built image so changes take effect without rebuilding.

**Step 1 -- Prepare data**

```bash
export DATA_ROOT=/path/to/data
mkdir -p $DATA_ROOT/data_cached/qwen3-8b-maxprompt1024

# Download model
huggingface-cli download Qwen/Qwen3-8B-Base --local-dir $DATA_ROOT/Qwen3-8B-Base

# Download and filter dataset to parquet
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('BytedTsinghua-SIA/DAPO-Math-17k', repo_type='dataset',
                  local_dir='$DATA_ROOT/raw/DAPO-Math-17k')
snapshot_download('BytedTsinghua-SIA/AIME-2024', repo_type='dataset',
                  local_dir='$DATA_ROOT/raw/AIME-2024')
"
# Filter prompts to <=1024 tokens and write parquet files into
# $DATA_ROOT/data_cached/qwen3-8b-maxprompt1024/
# (see examples/docs/03-data.md for the filtering script)
```

**Step 2 -- Run with volume-mounted code**

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
  --name lumenrl-verl-test \
  --entrypoint bash \
  zhangdanyangamd/lumen-rl:verlv0.9.0-lumenrl-plugin-example260820 \
  -c 'cd /workspace/Lumen-RL && bash examples/plugin/verl/run.sh'
```

Key docker flags:
- `--network=host`: required for Ray inter-process communication
- `--tmpfs /tmp:exec,size=10g`: clean `/tmp` avoids stale Ray cluster state
- `-v $(pwd):/workspace/Lumen-RL`: mount code for live editing without rebuild
- `--shm-size 64g`: large shared memory for NCCL/RCCL

Monitor logs:

```bash
docker logs -f lumenrl-verl-test
```

**Step 3 (optional) -- Build your own image**

```bash
docker build -f examples/plugin/verl/Dockerfile.test -t lumenrl-verl-plugin .
```

### Option B: Run without Docker

```bash
# Install both packages
cd /path/to/verl && pip install -e .
cd /path/to/Lumen-RL && pip install -e ".[engine]"

# Install AMD GPU detection for Ray
pip install amdsmi cachetools

# Patch verl's hard-coded strategy assertion (see known issues below)
VERL_ENGINE_CFG="$(pip show verl | awk '/^Location:/{print $2}')/verl/workers/config/engine.py"
sed -i 's/assert self\.strategy in \["fsdp", "fsdp2"\].*/# assert removed for plugin extensibility/' "$VERL_ENGINE_CFG"

# Verify plugin
python3 -c "import verl; from verl.workers.engine.base import EngineRegistry; print(EngineRegistry.get_engine_cls('language_model', 'lumenrl_fsdp2'))"

# Launch 3-step smoke test
DATA_ROOT=/data bash examples/plugin/verl/run.sh
```

## Configuration

`examples/plugin/verl/qwen3_8b_grpo_fp8.yaml` inherits verl defaults via
Hydra (`defaults: [_generated_ppo_trainer, _self_]`) and overrides only what
the LumenRL plugin needs:

| Parameter | Value | Notes |
|---|---|---|
| Training engine | `lumenrl_fsdp2` | `actor.strategy` and `ref.strategy` |
| Rollout | `vllm` (BF16) | verl-native vLLM, async mode |
| Batch size | 8 prompts | `train_batch_size: 8` |
| Learning rate | 1e-6 | Constant schedule, 10-step warmup |
| Max prompt / response | 512 / 512 | Smoke-test setting |
| Trainer | v0 legacy | `use_v1: false` (see known issues) |

## Health check

A successful 3-step smoke run prints:

```
Training Progress: 100% | 3/3
```

with non-NaN metrics (`actor/loss`, `actor/entropy`, `critic/reward`, etc.)
in the console log.

---

## Known issues

### 1. ROCm torch gets replaced by CUDA torch during pip install

Installing `verl` or any package that depends on `torch>=X` pulls PyPI's
CUDA-only torch, overwriting the ROCm torch in the base image.

**Workaround**: Backup the ROCm torch directory before pip install, restore
after. See `Dockerfile.test` for the exact steps.

### 2. verl FSDPEngineConfig hard-codes allowed strategy values

`FSDPEngineConfig.__post_init__` asserts `self.strategy in ["fsdp", "fsdp2"]`,
rejecting `lumenrl_fsdp2`.

**Fix**: Patch with sed:

```bash
sed -i 's/assert self\.strategy in \["fsdp", "fsdp2"\].*/# assert removed for plugin extensibility/' \
  "$(pip show verl | awk '/^Location:/{print $2}')/verl/workers/config/engine.py"
```

### 3. verl and LumenRL have different data formats

verl uses `NestedTensor` / padded tensors with its own batch format. LumenRL's
`FSDP2Engine` expects its own dataclass configs and tensor layout. The adapter
in the plugin handles the conversion (config mapping, tensor reshaping).

### 4. ATOM multi-replica shared memory broadcast deadlock

When running multiple ATOM rollout replicas, shared memory broadcast can
deadlock. Current workaround: use a single ATOM server. A proper fix is
planned. (Not relevant when using vLLM rollout as in the tested config.)

### 5. Ray cannot detect AMD GPUs without amdsmi

Ray uses `pyamdsmi.smi_get_device_count()` to detect AMD GPUs. Without
`amdsmi` installed, Ray reports 0 GPUs.

**Fix**: `pip install amdsmi`

### 6. verl v1 trainer requires transfer_queue

verl 0.9.0's v1 trainer (`use_v1: true`) imports `transfer_queue`, which is
not on PyPI. Use `trainer.use_v1: false` (the v0 legacy trainer) instead.
