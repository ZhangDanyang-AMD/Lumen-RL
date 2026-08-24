> [Examples README](../README_cn.md) > 环境搭建

# 1. 环境搭建

## 1.1 路径变量

后续所有命令都用这三个变量，换机器只改这里。

```bash
export RL_ROOT=/path/to/lumen_rl      # 代码根：Lumen-RL / Lumen / aiter / ATOM
export DATA_ROOT=/path/to/data        # 模型 / 数据 / 日志 / ckpt 根
export CONTAINER=rl-vllm-fsdp
mkdir -p "$RL_ROOT" "$DATA_ROOT/logs"
```

目标布局：

```text
$RL_ROOT/
├── Lumen-RL/            # RL 主框架（本仓库）
├── Lumen/               # FSDP2 训练后端（含 FP8）
├── aiter/               # AMD kernel
└── ATOM/                # 例子 4、5 需要

$DATA_ROOT/
├── models/Qwen3-8B-Base/
├── models/Qwen3-30B-A3B-Base/          # 例子 6、7，约 57G
├── raw/                                # 原始 parquet
├── data_cached/qwen3-8b-maxprompt1024/ # 过滤后的 train/val
├── logs/
└── ckpts/
```

---

## 1.2 拉代码

```bash
cd "$RL_ROOT"
git clone -b dev/vllm-fsdp-dapo   https://github.com/ZhangDanyang-AMD/Lumen-RL.git
git clone -b amd-atom-rollout     https://github.com/ZhangDanyang-AMD/Lumen.git
git clone -b lumen/triton_kernels https://github.com/ZhangDanyang-AMD/aiter.git
git clone -b lumen-rl             https://github.com/xysheng-AMD/ATOM.git   # 例子 4、5

# aiter 的 JIT 依赖 composable_kernel，必须补齐，
# 否则例子 3/4/5 触发 module_rmsnorm 时找不到 generate.py
cd "$RL_ROOT/aiter"
git submodule update --init --depth 1 3rdparty/composable_kernel
```

国内网络 GitHub 直连不稳时，只对本次命令用代理镜像，**不要写死进仓库 remote**：

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

已有 checkout 更新到最新：

```bash
git -C "$RL_ROOT/Lumen-RL" pull --ff-only origin dev/vllm-fsdp-dapo
git -C "$RL_ROOT/Lumen"    pull --ff-only origin amd-atom-rollout
git -C "$RL_ROOT/aiter"    pull --ff-only origin lumen/triton_kernels
git -C "$RL_ROOT/aiter"    submodule update --init --depth 1 3rdparty/composable_kernel
```

---

## 1.3 起容器

```bash
sudo docker pull vllm/vllm-openai-rocm:v0.23.0
# 国内网络：
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

> `-e` 注入的变量对后续所有 `docker exec` 会话可见，脚本无需再硬编码路径。
> 容器 `stop/start` 不丢依赖，只有 `docker rm` 才丢（丢了要重跑 §5、§6）。

> **必须确认容器内是 vLLM 0.23.0**。本机已有的 `vllm/vllm-openai-rocm:latest`
> 不代表可用，可能是旧版。

> **底座必须是 py3.12。** 本地 aiter 的预编译 `.so` 用了 `_PyThreadState_UncheckedGet`，
> CPython 3.13 删掉了这个私有 API，py3.13 / py3.14 的镜像会直接
> `ImportError: undefined symbol: _PyThreadState_UncheckedGet`。
