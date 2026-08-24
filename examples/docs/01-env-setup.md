> [Examples README](../README.md) > Environment Setup

# 1. Environment Setup

## 1.1 Path variables

Every command below uses these three variables. Moving to another machine means
changing only this block.

```bash
export RL_ROOT=/path/to/lumen_rl      # code root: Lumen-RL / Lumen / aiter / ATOM
export DATA_ROOT=/path/to/data        # models / data / logs / ckpt root
export CONTAINER=rl-vllm-fsdp
mkdir -p "$RL_ROOT" "$DATA_ROOT/logs"
```

Target layout:

```text
$RL_ROOT/
├── Lumen-RL/            # the RL framework (this repo)
├── Lumen/               # FSDP2 training backend (incl. FP8)
├── aiter/               # AMD kernels
└── ATOM/                # needed by examples 4 and 5

$DATA_ROOT/
├── models/Qwen3-8B-Base/
├── models/Qwen3-30B-A3B-Base/          # examples 6 and 7, ~57G
├── raw/                                # raw parquet
├── data_cached/qwen3-8b-maxprompt1024/ # filtered train/val
├── logs/
└── ckpts/
```

---

## 1.2 Clone the repos

```bash
cd "$RL_ROOT"
git clone -b dev/vllm-fsdp-dapo   https://github.com/ZhangDanyang-AMD/Lumen-RL.git
git clone -b amd-atom-rollout     https://github.com/ZhangDanyang-AMD/Lumen.git
git clone -b lumen/triton_kernels https://github.com/ZhangDanyang-AMD/aiter.git
git clone -b lumen-rl             https://github.com/xysheng-AMD/ATOM.git   # examples 4, 5

# aiter's JIT needs composable_kernel. Without it, examples 3/4/5 fail to find
# generate.py the moment they trigger module_rmsnorm.
cd "$RL_ROOT/aiter"
git submodule update --init --depth 1 3rdparty/composable_kernel
```

If GitHub is unreliable from your network, use a proxy mirror **for this command only**
— do not bake it into the repo remotes:

```bash
cd "$RL_ROOT"
GHP=https://gh-proxy.com/https://github.com
git -c http.version=HTTP/1.1 clone --depth 1 --single-branch -b dev/vllm-fsdp-dapo   "$GHP/ZhangDanyang-AMD/Lumen-RL.git"
git -c http.version=HTTP/1.1 clone --depth 1 --single-branch -b amd-atom-rollout     "$GHP/ZhangDanyang-AMD/Lumen.git"
git -c http.version=HTTP/1.1 clone --depth 1 --single-branch -b lumen/triton_kernels "$GHP/ZhangDanyang-AMD/aiter.git"
git -c http.version=HTTP/1.1 clone --depth 1 --single-branch -b lumen-rl             "$GHP/xysheng-AMD/ATOM.git"

cd "$RL_ROOT/aiter"
git -c http.version=HTTP/1.1 -c url."$GHP/".insteadOf=https://github.com/ \
  submodule update --init --depth 1 3rdparty/composable_kernel
```

Updating an existing checkout:

```bash
git -C "$RL_ROOT/Lumen-RL" pull --ff-only origin dev/vllm-fsdp-dapo
git -C "$RL_ROOT/Lumen"    pull --ff-only origin amd-atom-rollout
git -C "$RL_ROOT/aiter"    pull --ff-only origin lumen/triton_kernels
git -C "$RL_ROOT/aiter"    submodule update --init --depth 1 3rdparty/composable_kernel
```

---

## 1.3 Start the container

```bash
sudo docker pull vllm/vllm-openai-rocm:v0.23.0
# From a restricted network:
# sudo docker pull docker.m.daocloud.io/vllm/vllm-openai-rocm:v0.23.0
# sudo docker tag docker.m.daocloud.io/vllm/vllm-openai-rocm:v0.23.0 vllm/vllm-openai-rocm:v0.23.0

sudo docker rm -f "$CONTAINER" 2>/dev/null
sudo docker run -d --name "$CONTAINER" --entrypoint /bin/bash \
  --network=host --ipc=host \
  --device=/dev/kfd --device=/dev/dri --group-add=video \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --shm-size 64G \
  -v "$RL_ROOT":"$RL_ROOT" -v "$DATA_ROOT":"$DATA_ROOT" \
  -e RL_ROOT="$RL_ROOT" -e DATA_ROOT="$DATA_ROOT" -e HF_HOME="$DATA_ROOT/hf_home" \
  -e LUMEN_DIR="$RL_ROOT/Lumen" -e AITER_DIR="$RL_ROOT/aiter" \
  vllm/vllm-openai-rocm:v0.23.0 -lc 'sleep infinity'
```

> The `-e` variables are visible to every later `docker exec` session, so no script has
> to hardcode paths. Container `stop/start` keeps the installed dependencies; only
> `docker rm` loses them (after which §5 and §6 must be redone).

> **Confirm the container really has vLLM 0.23.0.** An existing
> `vllm/vllm-openai-rocm:latest` on the machine proves nothing — it may be an older vLLM.

> **The base image must be py3.12.** The prebuilt `.so` files in the local aiter use
> `_PyThreadState_UncheckedGet`, a private API that CPython 3.13 removed, so py3.13 /
> py3.14 images fail immediately with
> `ImportError: undefined symbol: _PyThreadState_UncheckedGet`.
