# LumenRL Plugin Examples

Three examples demonstrating how to inject LumenRL's ATOM rollout and Lumen
FP8 training into external RL frameworks, all running **Qwen3-8B** on 8 AMD
GPUs.

| # | Framework | Training | Rollout | Plugin mechanism | Directory |
|---|---|---|---|---|---|
| 1 | verl | LumenRL FSDP2 | ATOM | `verl.plugins` entry-point (auto) | [verl/](verl/) |
| 2 | vime | Megatron + Lumen FP8 | vLLM (native) | `--custom-*-path` CLI hooks | [vime/](vime/) |
| 3 | miles | Megatron + Lumen FP8 | SGLang (native) | `--custom-*-path` CLI hooks | [miles/](miles/) |

## Quick start

Each example includes:
- `run.sh` — launch script with environment variables
- `Dockerfile` — self-contained Docker build
- `README.md` — detailed setup and usage instructions

### Fastest path (Docker)

```bash
# Pick one:
docker build -f examples/plugin/verl/Dockerfile  -t lumenrl-verl  .
docker build -f examples/plugin/vime/Dockerfile  -t lumenrl-vime  .
docker build -f examples/plugin/miles/Dockerfile -t lumenrl-miles .

# Run (adjust image name):
docker run --rm -it \
  --device /dev/kfd --device /dev/dri \
  --group-add video --group-add render \
  --shm-size 64g \
  -v /data:/data \
  -e DATA_ROOT=/data \
  <image-name>
```

### Without Docker

```bash
# 1. Install LumenRL
cd /path/to/Lumen-RL && pip install -e ".[engine]"

# 2. Install the target framework
cd /path/to/verl && pip install -e .   # or vime / miles

# 3. Run
DATA_ROOT=/data bash examples/plugin/verl/run.sh
```

## How injection works

See [docs/plugin-integration.md](../../docs/plugin-integration.md) for the
full design rationale and [lumenrl/plugin/README.md](../../lumenrl/plugin/README.md)
for the plugin code reference.

**Key principle**: weight sync is never injected. Each framework uses its own
native mechanism. ATOM manages its own weight lifecycle via `sleep`/`wake_up`.
