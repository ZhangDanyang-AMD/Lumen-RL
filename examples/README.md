# LumenRL Examples Runbook

在一台**全新的 8 卡 AMD GPU 机器**上从零复现本目录下的 DAPO 数学 RL 训练例子。

所有例子共用同一个入口（`lumenrl.trainer.main`，Ray 控制器）和同一个启动脚本
（`examples/DAPO/run_dapo.sh`），差异全部由 config + 环境变量表达：
**单 Ray-driver 进程内 8 个训练 actor + 8 个同卡 colocated rollout replica（TP=1），
训练→rollout 权重经 ZMQ CUDA-IPC 同步**。

算法侧：clip-higher + dual-clip + token-mean 策略损失、GRPO 按 uid 组归一化、
动态采样 `filter_groups`、overlong 奖励缓冲、TIS rollout 修正。

**一句话复现**：设路径变量 → clone 仓库 → 起容器装依赖 →（FP8 才需要）打 patch →
下模型和数据 → smoke → `docker exec -d` 起长跑。

---

## 1. 已跑通的例子

| # | 例子 | 模型 | 训练后端 | Rollout / 精度 | GPU | Runtime | 启动开关 |
|---|---|---|---|---|---|---|---|
| 1 | 8B BF16 基线 | Qwen3-8B-Base | Lumen FSDP2，BF16 | vLLM / BF16 | 8× MI355X（gfx950）· 8× MI325X（gfx942） | `vllm/vllm-openai-rocm:v0.23.0` | `MODE=bf16` |
| 2 | 8B FP8 rollout | Qwen3-8B-Base | Lumen FSDP2，BF16 | vLLM / `fp8_per_block` | 同上 | 同上 | `MODE=fp8` |
| 3 | 8B FP8 E2E | Qwen3-8B-Base | Lumen FSDP2，**FP8 blockwise2d** | vLLM / `fp8_per_block` | 同上 | 同上 | `MODE=fp8 TRAIN_FP8=1` |
| 4 | 8B ATOM FP8 | Qwen3-8B-Base | Lumen FSDP2，**FP8 blockwise2d** | **ATOM** / `per_block_fp8` | 同上 | 同上 | `MODE=atomfp8 TRAIN_FP8=1` |
| 5 | 8B ATOM BF16 | Qwen3-8B-Base | Lumen FSDP2，BF16（纯 BF16，不打 Lumen norm patch） | **ATOM** / BF16 | 同上 | 同上 | `MODE=atombf16` |
| 6 | MoE FSDP2 | Qwen3-30B-A3B-Base | Lumen FSDP2，BF16 | vLLM / BF16 | 同上 | 同上 | `MODE=bf16` + MoE config |
| 7 | MoE Megatron EP=8 | Qwen3-30B-A3B-Base | **Megatron-Native**，TP=PP=CP=1 · EP=8 → DP=8 | vLLM / BF16 | 同上 | 同上 | `MODE=bf16` + Megatron config |

七个例子在 **8× MI355X** 和 **8× MI325X** 上都跑通过：smoke + 长跑，exit 0、无 Traceback、
无 OOM、无 `HSA_STATUS`，权重同步覆盖率断言全过，收尾后显存回到约 298 MB/卡的空闲基线。
两种卡用同一份 config，**不需要缩短序列长度或改任何显存参数**。

⚠️ **例子 5 是例子 4 的 BF16 对照组**：同一个 ATOM rollout 引擎、同样的
no-eager level=3 + sleep2，只把 rollout 的在线量化和训练侧的 FP8 一起关掉，
所以它和例子 1（vLLM BF16）是可比的。`MODE=atombf16` 会 unset 掉全部 `LUMEN_FP8*`，
**`TRAIN_FP8` 对它无效**。

⚠️ **同一批卡上不能同时跑两个训练后端**，也不能和别人共用节点——引擎按"占整卡比例"算
KV cache 预算。起之前先确认显存在空闲基线。

⚠️ **两个训练后端不能共用 checkpoint 目录**，格式不同。

---

## 2. 路径变量

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

## 3. 拉代码

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

## 4. 起容器

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

⚠️ **必须确认容器内是 vLLM 0.23.0**。本机已有的 `vllm/vllm-openai-rocm:latest`
不代表可用，可能是旧版。

⚠️ **底座必须是 py3.12。** 本地 aiter 的预编译 `.so` 用了 `_PyThreadState_UncheckedGet`，
CPython 3.13 删掉了这个私有 API，py3.13 / py3.14 的镜像会直接
`ImportError: undefined symbol: _PyThreadState_UncheckedGet`。

---

## 5. 装依赖

### 5.1 基础依赖（所有例子）

```bash
sudo docker exec "$CONTAINER" bash -lc '
set -e
# 容器 root 访问宿主挂载仓库时允许 git introspection（editable install / setuptools_scm）
git config --global --add safe.directory "$RL_ROOT/Lumen-RL" || true
git config --global --add safe.directory "$LUMEN_DIR" || true
git config --global --add safe.directory "$AITER_DIR" || true
git config --global --add safe.directory "$RL_ROOT/ATOM" || true

cd "$AITER_DIR" && AITER_USE_SYSTEM_TRITON=1 python3 setup.py develop || pip install -e .
pip install -e "$LUMEN_DIR" --no-deps || true
cd "$RL_ROOT/Lumen-RL" && pip install -e . --no-deps
pip install "ray[default]>=2.9" "accelerate>=0.28" datasets \
  "math_verify[antlr4_13_2]" "omegaconf>=2.3,<2.4" safetensors wandb
'
```

> 不加 `safe.directory` 会在 `pip install -e` 时报 `fatal: detected dubious ownership`。
> 例子 4、5 的 ATOM **无需单独 pip 安装**——`run_dapo.sh` 自动把 `$RL_ROOT/ATOM` 及
> `examples/DAPO/atom_aiter_shim` 加进 `PYTHONPATH`，`import atom` 即可用。

### 5.2 例子 6、7 追加：flydsl

```bash
sudo docker exec "$CONTAINER" bash -lc 'pip install "flydsl==0.1.8" && python3 -c "
import flydsl, transformers, vllm
print(\"flydsl\", flydsl.__version__, \"transformers\", transformers.__version__, \"vllm\", vllm.__version__)"'
```
> 期望 `flydsl 0.1.8 transformers 5.12.0 vllm 0.23.0`。

**为什么必做**：镜像自带的 flydsl 0.1.4.2 会让 `from aiter import flash_attn_varlen_func`
报版本不兼容，**训练前向直接挂**。

⚠️ **不要按 wheel 的 pin 回退。** `run_dapo.sh` 把本地 `$AITER_DIR` 放在 `PYTHONPATH` 最前，
运行时用的是仓库里的 aiter 源码（要求 flydsl ≥ `0.1.5.dev515`），而镜像自带的
`amd-aiter` wheel 反过来 pin `flydsl<0.1.5`，升级后它会报 `cannot import name 'fly_values'`。
两者互斥，走 `run_dapo.sh` 的 PYTHONPATH 才是对的。

⚠️ **transformers 必须是 5.x**，它把 Qwen3-MoE 的专家融合成 3D 张量，仓库的权重同步
（`lumenrl/engine/inference/vllm_moe_weight_sync.py`）按这个布局写。

### 5.3 例子 7 追加：megatron-core / Apex / TransformerEngine

已验证的三个组件和 revision：

- **megatron-core `0.18.2`**：`pip install --no-deps "megatron-core==0.18.2"`。
- **ROCm Apex `daed85255d51476425080e7e6203f0bee6d7e4cc`**：源码
  `setup.py install --cpp_ext --cuda_ext`，带 `PYTORCH_ROCM_ARCH=gfx950`。
- **ROCm TransformerEngine `6e541a10419a6e31bdc98b1516db04eb81a463b6`**
  （→ `2.15.0.dev0+6e541a1`）：源码 `pip install -v . --no-build-isolation`，约 9 分钟。

```bash
sudo docker exec "$CONTAINER" bash -lc 'pip install --no-deps "megatron-core==0.18.2"'
```

TE 编译要点：必须用 **ROCm fork**、必须递归拉全部 submodule（约 5.1 GiB，含 AOTriton /
CK JIT / Composable Kernel）、编译前先卸掉可能存在的 NVIDIA TE 包，并且要带
`TORCH_DONT_CHECK_COMPILER_ABI=1` —— ROCm 7.2.3 的 `hipcc -v` 在没有输入文件时返回 1，
CK-JIT 的编译器 ABI 探测会把它误判成"编译器不可用"。

⚠️ **绝对不要 `pip install transformer_engine`**，那会装成 NVIDIA 版，导入即 undefined symbol。

⚠️ 不要装 `megatron-bridge`。Qwen3 的 HF ↔ Megatron 转换由
`lumenrl/engine/training/qwen3_megatron_bridge.py` 负责。

### 5.4 验证

```bash
sudo docker exec "$CONTAINER" bash -lc '
python3 - <<PY
import torch, vllm, ray, lumenrl
from lumenrl.engine.inference.vllm_ray_server import VLLMRayServer
print("torch", torch.__version__, "vllm", vllm.__version__,
      "ray", ray.__version__, "GPUs", torch.cuda.device_count())
print("import OK")
PY
'
```
> 期望 `GPUs 8`、`vllm 0.23.0`、`import OK`。

---

## 6. vLLM AITER RMSNorm patch（例子 2、3、4 必需）

FP8 rollout 走 `VLLM_ROCM_USE_AITER=1`，vLLM 用 AITER RMSNorm，必须传
`use_model_sensitive_rmsnorm=1` 才能与训练侧 Lumen 的 model-sensitive RMSNorm 对齐，
否则 `rollout_corr/kl` 偏大。

⚠️ 该 patch 改的是**容器内的 vllm wheel**，`docker rm` 重建容器后会丢，**新容器必打一次**。
例子 1、5、6、7 是 BF16 rollout（`VLLM_ROCM_USE_AITER=0`），可跳过本节。

脚本幂等，只改 `kernels/aiter_ops.py` / `_aiter_ops.py` 两条 plain RMSNorm 路径，
不碰任何 quant-fusion 路径：

```bash
cat > "$RL_ROOT/patch_vllm_aiter_rmsnorm.py" <<'PYEOF'
#!/usr/bin/env python3
"""vLLM AITER RMSNorm Patch (model-sensitive / T5-like) for the rollout side.

Patches the *plain* (non-quant) AITER RMSNorm paths inside the container's
installed vLLM wheel so they pass ``use_model_sensitive_rmsnorm=1``. Quant-fusion
paths (rmsnorm2d_fwd_with_dynamicquant / *_fp8_group_quant / ...) are NOT touched.
Idempotent: safe to run repeatedly.
"""

import importlib.util
import os
import sys

RMS_OLD = """    if x.dim() > 2:
        x_original_shape = x.shape
        x = x.reshape(-1, x_original_shape[-1])
        x = rms_norm(x, weight, variance_epsilon)
        return x.reshape(x_original_shape)

    return rms_norm(x, weight, variance_epsilon)"""

RMS_NEW = """    if not getattr(_rms_norm_impl, "_lumen_logged", False):
        print("[vllm-aiter] rms_norm use_model_sensitive_rmsnorm=1", flush=True)
        _rms_norm_impl._lumen_logged = True

    if x.dim() > 2:
        x_original_shape = x.shape
        x = x.reshape(-1, x_original_shape[-1])
        x = rms_norm(x, weight, variance_epsilon, use_model_sensitive_rmsnorm=1)
        return x.reshape(x_original_shape)

    return rms_norm(x, weight, variance_epsilon, use_model_sensitive_rmsnorm=1)"""

ADD_OLD = """    rmsnorm2d_fwd_with_add(
        out,  # output
        x,  # input
        residual,  # residual input
        residual_out,  # residual output
        weight,
        variance_epsilon,
    )
    return out, residual_out"""

ADD_NEW = """    if not getattr(_rocm_aiter_rmsnorm2d_fwd_with_add_impl, "_lumen_logged", False):
        print(
            "[vllm-aiter] rmsnorm2d_fwd_with_add use_model_sensitive_rmsnorm=1",
            flush=True,
        )
        _rocm_aiter_rmsnorm2d_fwd_with_add_impl._lumen_logged = True
    rmsnorm2d_fwd_with_add(
        out,  # output
        x,  # input
        residual,  # residual input
        residual_out,  # residual output
        weight,
        variance_epsilon,
        use_model_sensitive_rmsnorm=1,
    )
    return out, residual_out"""

REPLACEMENTS = ((RMS_OLD, RMS_NEW), (ADD_OLD, ADD_NEW))


def _vllm_dir() -> str:
    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.submodule_search_locations:
        sys.exit("ERROR: vllm is not importable in this interpreter")
    return spec.submodule_search_locations[0]


def _patch_file(path: str) -> int:
    if not os.path.isfile(path):
        print(f"[skip] {path} (not found)")
        return 0
    with open(path, "r", encoding="utf-8") as fh:
        src = fh.read()
    changed = 0
    for old, new in REPLACEMENTS:
        if new in src:
            continue  # already patched
        if old in src:
            src = src.replace(old, new, 1)
            changed += 1
    if changed:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(src)
        print(f"[patched] {path} ({changed} site(s))")
    else:
        already = sum(new in src for _, new in REPLACEMENTS)
        print(f"[ok] {path} (already patched)" if already else f"[skip] {path} (no plain call sites)")
    return changed


def main() -> None:
    vdir = _vllm_dir()
    targets = [os.path.join(vdir, "kernels", "aiter_ops.py"), os.path.join(vdir, "_aiter_ops.py")]
    total = sum(_patch_file(p) for p in targets)
    print(f"[done] vLLM AITER RMSNorm patch: {total} replacement(s) applied")


if __name__ == "__main__":
    main()
PYEOF
sudo docker exec "$CONTAINER" bash -lc 'cd "$RL_ROOT" && python3 patch_vllm_aiter_rmsnorm.py'
```
> 期望 `[patched] .../kernels/aiter_ops.py (2 site(s))`，已打过时是 `[ok] ... (already patched)`。

验证 patch 生效：

```bash
sudo docker exec "$CONTAINER" bash -lc '
python3 - <<PY
import inspect
from vllm.kernels import aiter_ops as k
ms = all("use_model_sensitive_rmsnorm=1" in inspect.getsource(getattr(k, a))
         for a in ["_rms_norm_impl", "_rocm_aiter_rmsnorm2d_fwd_with_add_impl"])
print("RMSNorm model-sensitive patch:", ms, "(FP8 需为 True)")
PY
'
```

---

## 7. ATOM JIT 预编译（例子 4、5 必需）

例子 4、5 用本地 `ATOM` + 本地 `aiter` 源码，不同于 vLLM wheel 内置的 kernel。
ATOM 启动 FP8 rollout 时会按需 JIT 编译 aiter kernel，其中 `module_rmsnorm` 最慢。
**这是环境安装成本，不要算进训练性能**，先单独编好：

```bash
# 先确认 submodule 已拉取（§3），否则找不到 generate.py
test -f "$RL_ROOT/aiter/3rdparty/composable_kernel/example/ck_tile/10_rmsnorm2d/generate.py"

# 首次可能耗时十几到二十分钟；看到 PRECOMPILE_DONE 才算完成
sudo docker exec "$CONTAINER" bash -lc '
export PYTHONPATH="$AITER_DIR:${PYTHONPATH:-}"
python3 - <<PY
import torch
from aiter import rmsnorm2d_fwd
print("PRECOMPILE_START", flush=True)
x = torch.randn((1, 4096), device="cuda", dtype=torch.bfloat16)
w = torch.ones((4096,), device="cuda", dtype=torch.bfloat16)
y = rmsnorm2d_fwd(x, w, 1e-6, use_model_sensitive_rmsnorm=1)
torch.cuda.synchronize()
print("PRECOMPILE_DONE", y.shape, y.dtype, flush=True)
PY
'
```

被 `Ctrl-C` / 容器重启 / `pkill` 打断后可能留下 stale lock，表现为后续进程一直打印
`waiting for baton release at .../lock_module_rmsnorm`，但没有 `ninja` / `hipcc` / `clang-22`
编译进程。清理后重跑：

```bash
sudo docker exec "$CONTAINER" bash -lc '
rm -rf "$AITER_DIR/aiter/jit/build/lock_module_rmsnorm" \
       "$AITER_DIR/aiter/jit/build/module_rmsnorm"
'
```

> 例子 2、3 不需要这一步，它们走镜像内已安装的 vLLM/AITER 路径。

---

## 8. 模型与数据

### 8.1 下载

```bash
sudo docker exec "$CONTAINER" bash -lc '
python3 - <<PY
from huggingface_hub import snapshot_download
import os; D = os.environ["DATA_ROOT"]
snapshot_download("Qwen/Qwen3-8B-Base", local_dir=f"{D}/models/Qwen3-8B-Base",
                  allow_patterns=["*.json","*.txt","*.safetensors","*.model","tokenizer*"])
snapshot_download("BytedTsinghua-SIA/DAPO-Math-17k", repo_type="dataset",
                  local_dir=f"{D}/raw/DAPO-Math-17k")
snapshot_download("BytedTsinghua-SIA/AIME-2024", repo_type="dataset",
                  local_dir=f"{D}/raw/AIME-2024")
PY
'

# 例子 6、7 追加（约 57G）
sudo docker exec "$CONTAINER" bash -lc '
hf download Qwen/Qwen3-30B-A3B-Base \
  --local-dir "$DATA_ROOT/models/Qwen3-30B-A3B-Base" --max-workers 8'
```

国内网络改用 ModelScope，ID 同名（`Qwen/Qwen3-8B-Base`、`Qwen/Qwen3-30B-A3B-Base`、
`BytedTsinghua-SIA/DAPO-Math-17k`、`BytedTsinghua-SIA/AIME-2024`），落到同样的本地路径，
后续命令不用改：

```bash
sudo docker exec "$CONTAINER" bash -lc '
pip install modelscope
python3 - <<PY
from modelscope.hub.snapshot_download import snapshot_download
import os
D = os.environ["DATA_ROOT"]
snapshot_download("Qwen/Qwen3-8B-Base", local_dir=f"{D}/models/Qwen3-8B-Base",
    allow_patterns=["*.json","*.txt","*.safetensors","*.model","tokenizer*","*.py","*.tiktoken"],
    max_workers=8)
for rid, sub in (("BytedTsinghua-SIA/DAPO-Math-17k", "DAPO-Math-17k"),
                 ("BytedTsinghua-SIA/AIME-2024", "AIME-2024")):
    snapshot_download(repo_id=rid, repo_type="dataset", local_dir=f"{D}/raw/{sub}",
        allow_patterns=["*.parquet","*.json","*.jsonl","*.md","*.txt"], max_workers=4)
PY
'
```

⚠️ **MoE 必须用 Base 版**。instruct/thinking 版的 Qwen3-30B-A3B 在 `max_response_length` 内
**永远不闭合 `</think>`**（实测给到 3072 token 仍不闭合、也不出 `\boxed`），于是每条样本都被
截断、reward 恒为 −1、`filter_groups` 连续 10 轮 kept 0，直接抛
`RuntimeError: filter_groups collected no valid groups`。Base 版能正常输出 `Answer:`。

### 8.2 过滤 prompt ≤1024

不预过滤的话启动会进入耗时的 overlong-prompt 扫描。

```bash
cat > "$RL_ROOT/filter_prompts.py" <<'PYEOF'
import os, glob
import datasets
from transformers import AutoTokenizer

DATA = os.environ["DATA_ROOT"]
MODEL_PATH = f"{DATA}/models/Qwen3-8B-Base"
MAX_PROMPT_LENGTH = 1024
PROMPT_KEY = "prompt"
OUT_DIR = f"{DATA}/data_cached/qwen3-8b-maxprompt1024"

def first_parquet(*dir_globs):
    for g in dir_globs:
        hits = sorted(glob.glob(g, recursive=True))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"no parquet under {dir_globs}")

JOBS = [
    (first_parquet(f"{DATA}/raw/DAPO-Math-17k/**/*.parquet"),
     os.path.join(OUT_DIR, "dapo-math-17k.filtered.parquet")),
    (first_parquet(f"{DATA}/raw/AIME-2024/**/*.parquet"),
     os.path.join(OUT_DIR, "aime-2024.filtered.parquet")),
]

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

def doc2len(doc) -> int:
    return len(tokenizer.apply_chat_template(doc[PROMPT_KEY], add_generation_prompt=True, tokenize=True))

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    nproc = max(1, min(64, (os.cpu_count() or 8) // 4))
    for src, dst in JOBS:
        ds = datasets.Dataset.from_parquet(src)
        before = len(ds)
        ds = ds.filter(lambda d: doc2len(d) <= MAX_PROMPT_LENGTH, num_proc=nproc,
                       desc=f"Filtering prompts > {MAX_PROMPT_LENGTH} tokens")
        ds.to_parquet(dst)
        print(f"[{src}] -> {dst}: {before} -> {len(ds)} (removed {before-len(ds)})")

if __name__ == "__main__":
    main()
PYEOF
sudo docker exec "$CONTAINER" bash -lc 'cd "$RL_ROOT" && python3 filter_prompts.py'
```

产出即 `run_dapo.sh` 的默认 `TRAIN_FILE` / `VAL_FILE`：

```text
$DATA_ROOT/data_cached/qwen3-8b-maxprompt1024/dapo-math-17k.filtered.parquet   # train
$DATA_ROOT/data_cached/qwen3-8b-maxprompt1024/aime-2024.filtered.parquet       # val
```

> **数据只需过滤一次，六个例子共用。** Qwen3-8B-Base 与 Qwen3-30B-A3B-Base 的
> `tokenizer.json` / `vocab.json` / `merges.txt` 三个文件 md5 完全相同（vocab 151936），
> 所以按 8B tokenizer 过滤出的结果对 MoE 同样成立。

---

## 9. Config 与规模

全部在 `examples/DAPO/configs/`：

```text
# 1  8B BF16
dapo_qwen3_8b_ray_vllm_smoke.yaml                     resp=512
dapo_qwen3_8b_ray_vllm_longrun.yaml

# 2、3  8B vLLM FP8（共用 config，训练精度由 TRAIN_FP8 控制）
dapo_qwen3_8b_ray_vllm_fp8_smoke.yaml                 resp=512
dapo_qwen3_8b_ray_vllm_fp8_4k_smoke.yaml              resp=4096
dapo_qwen3_8b_ray_vllm_fp8_longrun.yaml

# 4  8B ATOM FP8
dapo_qwen3_8b_ray_atom_fp8_4k_smoke.yaml              resp=4096
dapo_qwen3_8b_ray_atom_fp8_longrun.yaml

# 5  8B ATOM BF16（规模与例子 4 逐字段相同，只关掉量化）
dapo_qwen3_8b_ray_atom_bf16_4k_smoke.yaml             resp=4096
dapo_qwen3_8b_ray_atom_bf16_longrun.yaml

# 6  MoE FSDP2
dapo_qwen3moe_a3b_ray_vllm_verlref_4k_smoke.yaml
dapo_qwen3moe_a3b_ray_vllm_verlref_longrun.yaml

# 7  MoE Megatron EP=8
dapo_qwen3moe_a3b_ray_megatron_verlref_4k_smoke.yaml
dapo_qwen3moe_a3b_ray_megatron_verlref_longrun.yaml
dapo_qwen3moe_a3b_ray_megatron_verlref_4k_longrun.yaml   # 压缩版，几小时出结论
```

**8B 长跑规模**（例子 1–5 相同）：1000 步、`train_global_batch_size=512`（32 prompt × 16）、
`gen_batch_size=96`、`max_response_length=20480`、`max_total_sequence_length=21504`、
lr 1e-6 / warmup 10 / wd 0.1 / clip_grad 1.0、clip 0.2/0.28/10 + token-mean、
`overlong_buffer` 512/1.0、`filter_groups` acc / max 10 轮、`rollout_is=token` 阈值 2.0、
`val_steps=10` / `save_steps=50` / seed 10086。
BF16 与 FP8 的 config **只差 `vllm_cfg.quantization` 一行**。

**MoE 长跑规模**（例子 6、7 相同）：prompt=2048、resp=20480、
**128 prompt × 16 = 2048 序列**、`gen_batch_size=384`、**lr warmup = 0**、1000 步。
两个 config 除了 `policy.training_backend` 和 `megatron_cfg` **逐字段相同**，
所以两条线的任何差异都只能来自训练后端。

⚠️ **注意单位**：`train_global_batch_size` 是**序列数**（2048），`gen_batch_size` 是
**prompt 数**（384）。框架用 `train_prompts = train_global_batch_size // num_generations`
反推 prompt 数。

**例子 7 的 `megatron_cfg`（长跑）**：

```yaml
use_distributed_optimizer: true
tensor_model_parallel_size: 1
pipeline_model_parallel_size: 1
context_parallel_size: 1
expert_model_parallel_size: 8       # 128 专家分到 8 卡，每卡 16 个
sequence_parallel: false
moe_grouped_gemm: true
moe_permute_fusion: true
moe_aux_loss_coeff: 0.0
moe_router_dtype: fp32              # 与 LUMENRL_FP32_MOE_ROUTER=1 配对
recompute_granularity: full         # resp=20480 必需
recompute_method: uniform
recompute_num_layers: 1
log_probs_chunk_size: 1024
enable_dynamic_batch: true
max_tokens_per_gpu: 8192            # 不是 22528，见 §12
```

**拓扑为什么选 EP=8**：`DP = 8 / (TP × PP × CP) = 8`，和 FSDP2 的 DP8 一致，每个 rank 仍然
看到 2048/8 = 256 条序列。任何缩小 DP 的改动都会让 distributed optimizer 的 state 每卡翻倍
（DP 8→4 多约 8.5 GB），把激活上省下来的又吃回去 —— **CP=2 实测当场 OOM**，比 CP=1 更早死。

---

## 10. 启动

`run_dapo.sh` 的开关全部走环境变量，**不需要改脚本内容**：

- `MODE`（默认 `bf16`）：`bf16` / `fp8` / `atomfp8` / `atombf16`，选 config + rollout 引擎与精度。
- `TRAIN_FP8`（默认 `0`）：`1` = 训练侧 Lumen FP8 blockwise2d，自动带 `FP8_PARAM_MANAGER=0`。
- `STEPS`（默认 `1000`）：覆盖 `num_training_steps`。
- `CONFIG_OVERRIDE`（默认按 `MODE` 推导）：直接指定 config，**跑 smoke 必须用它**。
- `EXTRA_OVERRIDE`（默认空）：追加任意 Hydra 覆盖，空格分隔。
- `MODEL_PATH` / `TRAIN_FILE` / `VAL_FILE`：换模型或数据，默认走 `$DATA_ROOT` 标准布局。
- `LOG`：日志路径，默认 `$DATA_ROOT/logs/$RUN_ID.log`，同时写进 `/tmp/run_dapo_log.txt`。
- `LUMENRL_FP32_MOE_ROUTER`（默认 `1`）：**例子 6、7 必须显式给**，见下。
- `PYTORCH_CUDA_ALLOC_CONF`：**启动时置空**，ROCm/HIP allocator 不支持 `expandable_segments`。

⚠️ **脚本是唯一来源，被误改就还原它**：`git -C "$RL_ROOT/Lumen-RL" checkout -- examples/DAPO/run_dapo.sh`。

所有命令统一带这段前缀。`export VAR=` 是"设为空值"，脚本据此把它 `unset`：

```bash
S=$RL_ROOT/Lumen-RL/examples/DAPO/run_dapo.sh
ENVX="export RL_ROOT='$RL_ROOT' DATA_ROOT='$DATA_ROOT' PYTORCH_CUDA_ALLOC_CONF=;"
```

⚠️ `run_dapo.sh` 开头有 `: "${RL_ROOT:?}"`，容器内 `RL_ROOT` 为空会直接退出。
`ENVX` 就是为防这个坑——detached exec 不要依赖 §4 的 `-e` 注入。

### 例子 1–5：Qwen3-8B-Base

先跑 smoke（前台等结果）。⚠️ **smoke 必须用 `CONFIG_OVERRIDE` 指到 `*_smoke.yaml`**，
只设 `STEPS=1` 会继续用长跑 config（resp=20480、batch=512），那不是 smoke：

```bash
# 例子 1
sudo docker exec "$CONTAINER" bash -lc "$ENVX \
  CONFIG_OVERRIDE=examples/DAPO/configs/dapo_qwen3_8b_ray_vllm_smoke.yaml \
  STEPS=1 MODE=bf16 LOG=$DATA_ROOT/logs/smoke-bf16.log bash '$S'; \
  tail -80 \"\$(cat /tmp/run_dapo_log.txt)\""

# 例子 2（TRAIN_FP8=0，只验 rollout fp8_per_block）
sudo docker exec "$CONTAINER" bash -lc "$ENVX \
  CONFIG_OVERRIDE=examples/DAPO/configs/dapo_qwen3_8b_ray_vllm_fp8_smoke.yaml \
  STEPS=1 MODE=fp8 LOG=$DATA_ROOT/logs/smoke-fp8.log bash '$S'; \
  tail -80 \"\$(cat /tmp/run_dapo_log.txt)\""

# 例子 4（4k 配置；先做完 §7 的预编译）
sudo docker exec "$CONTAINER" bash -lc "$ENVX \
  CONFIG_OVERRIDE=examples/DAPO/configs/dapo_qwen3_8b_ray_atom_fp8_4k_smoke.yaml \
  STEPS=1 MODE=atomfp8 LOG=$DATA_ROOT/logs/smoke-atomfp8.log bash '$S'; \
  tail -80 \"\$(cat /tmp/run_dapo_log.txt)\""

# 例子 5（同样先做完 §7 的预编译；不需要 §6 的 patch）
sudo docker exec "$CONTAINER" bash -lc "$ENVX \
  CONFIG_OVERRIDE=examples/DAPO/configs/dapo_qwen3_8b_ray_atom_bf16_4k_smoke.yaml \
  STEPS=1 MODE=atombf16 LOG=$DATA_ROOT/logs/smoke-atombf16.log bash '$S'; \
  tail -80 \"\$(cat /tmp/run_dapo_log.txt)\""
```

再起长跑（detached，防中断）：

```bash
sudo docker exec -d "$CONTAINER" bash -lc "$ENVX STEPS=1000 MODE=bf16                 bash '$S'"  # 1
sudo docker exec -d "$CONTAINER" bash -lc "$ENVX STEPS=1000 MODE=fp8                  bash '$S'"  # 2
sudo docker exec -d "$CONTAINER" bash -lc "$ENVX STEPS=1000 MODE=fp8      TRAIN_FP8=1 bash '$S'"  # 3
sudo docker exec -d "$CONTAINER" bash -lc "$ENVX STEPS=1000 MODE=atomfp8  TRAIN_FP8=1 bash '$S'"  # 4
sudo docker exec -d "$CONTAINER" bash -lc "$ENVX STEPS=1000 MODE=atombf16             bash '$S'"  # 5
```

> 例子 5 不要传 `TRAIN_FP8=1`：`MODE=atombf16` 会把 `LUMEN_FP8` / `FP8_PARAM_MANAGER` /
> `LUMEN_FP8_SCALING` 等一并 unset，训练侧无条件是 BF16。它也**不导入 Lumen/AITER 的 norm
> patch**（HF Qwen3 的 RMSNorm 本身就是 model-sensitive），这样才和例子 1 严格可比。

> 建议先 `STEPS=30` 起一版确认显存/指标健康，再上 1000 步。
> W&B 可选：把 `WANDB_API_KEY=xxxx` 放进 `$RL_ROOT/wandb.key`，脚本自动读。
> 换 ckpt 落盘频率用 `EXTRA_OVERRIDE='checkpointing.save_steps=10 checkpointing.save_total_limit=2'`；
> 8B FSDP2 单个 checkpoint 约 90 GB，先 `df -h`。

确认已经在跑：

```bash
sudo docker exec "$CONTAINER" bash -lc 'L=$(cat /tmp/run_dapo_log.txt); sleep 200
  grep -aE "setup .ray-controller. complete|filter_groups round|View run" "$L" | tail -3
  grep -aiE "Traceback|OutOfMemory|CUDA error" "$L" | tail'
```

### 例子 6：MoE + FSDP2

```bash
ENVX_MOE="export RL_ROOT='$RL_ROOT' DATA_ROOT='$DATA_ROOT' SCRATCH_ROOT='$DATA_ROOT' \
LUMENRL_FP32_MOE_ROUTER=0 PYTORCH_CUDA_ALLOC_CONF=;"

# smoke（4k，3 步，约 10 分钟，其中 5 分钟是 8 个 actor 各自加载 57GB 模型）
sudo docker exec "$CONTAINER" bash -lc "$ENVX_MOE \
  CONFIG_OVERRIDE=examples/DAPO/configs/dapo_qwen3moe_a3b_ray_vllm_verlref_4k_smoke.yaml \
  MODEL_PATH=\$DATA_ROOT/models/Qwen3-30B-A3B-Base STEPS=3 MODE=bf16 \
  LOG=\$DATA_ROOT/logs/moe-4k-smoke.log bash '$S'; \
  tail -40 \"\$(cat /tmp/run_dapo_log.txt)\""

# 长跑（先看磁盘：至少 400G 可用）
df -h "$DATA_ROOT"
sudo docker exec -d "$CONTAINER" bash -lc "$ENVX_MOE \
  CONFIG_OVERRIDE=examples/DAPO/configs/dapo_qwen3moe_a3b_ray_vllm_verlref_longrun.yaml \
  MODEL_PATH=\$DATA_ROOT/models/Qwen3-30B-A3B-Base STEPS=1000 MODE=bf16 \
  LOG=\$DATA_ROOT/logs/longrun-moe.log bash '$S'"
```

⚠️ **`MODEL_PATH` 必须显式给** —— `run_dapo.sh` 的默认值是 8B。

⚠️ **`LUMENRL_FP32_MOE_ROUTER=0` 必须给。** 框架默认是 fp32，而这条线要求 router 走 BF16：
FSDP2 和 vLLM 跑的是**同一个 PyTorch router op、同一种布局**，BF16 舍入会让两边落到同一组
top-8 专家，两侧对齐比单侧提精度更重要。日志里应看到
`[lumenrl] MoE router patched on 48 gates (fp32=False)`，`True` 说明忘了传。

⚠️ **`SCRATCH_ROOT` 必须导出**：config 用 `${oc.env:SCRATCH_ROOT}` 解析
`model_name` / `checkpoint_dir`，解析不到 omegaconf 直接退出。**即使关掉落盘也要给。**

**新机器第一次跑 MoE，先做一次权重同步的端到端确认。** transformers 5.x 的融合专家张量
（约 57 GB、**93% 的参数**）一旦匹配不上 vLLM 的 `expert_params_mapping`，会走 vLLM 的静默
`continue` 分支：不报错、不加载，rollout 引擎的专家永远停在磁盘加载值。覆盖率断言
（`LUMENRL_WEIGHT_SYNC_CHECK=error`）默认开着，再加一次逐位比对更稳：

```bash
sudo docker exec "$CONTAINER" bash -lc "$ENVX_MOE LUMENRL_WEIGHT_SYNC_VERIFY=1 \
  CONFIG_OVERRIDE=examples/DAPO/configs/dapo_qwen3moe_a3b_ray_vllm_verlref_4k_smoke.yaml \
  MODEL_PATH=\$DATA_ROOT/models/Qwen3-30B-A3B-Base STEPS=3 MODE=bf16 \
  LOG=\$DATA_ROOT/logs/moe-verify.log bash '$S'"
```

> 判据是**没有异常**：exit 0 就说明 96 个融合张量 × 8 replica × 3 次同步全部逐位一致。
> 失败会抛 `weight sync verify failed for ... shard w1/w3/w2` 或
> `weight sync (colocate-ipc) left N/M rollout parameters untouched: ...`。

顺带跑一遍纯 CPU 单测，确认代码完整：

```bash
sudo docker exec "$CONTAINER" bash -lc 'cd "$RL_ROOT/Lumen-RL" &&
  python3 -m lumenrl.tests.test_moe_weight_sync &&      # 11 项，融合专家同步
  python3 -m lumenrl.tests.test_rollout_routing &&      #  9 项
  python3 -m lumenrl.tests.test_dataproto_ragged &&     # 10 项
  python3 -m lumenrl.tests.test_mismatch_metrics'       #  4 项
```

### 例子 7：MoE + Megatron EP=8

```bash
# smoke：config 的 moe_router_dtype 是 null，所以这里是 =0
sudo docker exec "$CONTAINER" bash -lc "$ENVX_MOE \
  CONFIG_OVERRIDE=examples/DAPO/configs/dapo_qwen3moe_a3b_ray_megatron_verlref_4k_smoke.yaml \
  MODEL_PATH=\$DATA_ROOT/models/Qwen3-30B-A3B-Base STEPS=3 MODE=bf16 \
  LOG=\$DATA_ROOT/logs/moe-4k-smoke-megatron.log bash '$S'; \
  tail -40 \"\$(cat /tmp/run_dapo_log.txt)\""

# 长跑：config 的 moe_router_dtype 是 fp32，所以这里必须翻成 =1
ENVX_MEGA="export RL_ROOT='$RL_ROOT' DATA_ROOT='$DATA_ROOT' SCRATCH_ROOT='$DATA_ROOT' \
LUMENRL_FP32_MOE_ROUTER=1 PYTORCH_CUDA_ALLOC_CONF=;"

df -h "$DATA_ROOT"     # Megatron dist-checkpoint 约 400GB
sudo docker exec -d "$CONTAINER" bash -lc "$ENVX_MEGA \
  CONFIG_OVERRIDE=examples/DAPO/configs/dapo_qwen3moe_a3b_ray_megatron_verlref_longrun.yaml \
  MODEL_PATH=\$DATA_ROOT/models/Qwen3-30B-A3B-Base STEPS=1000 MODE=bf16 \
  LOG=\$DATA_ROOT/logs/longrun-moe-megatron.log bash '$S'"
```

⚠️ **`LUMENRL_FP32_MOE_ROUTER` 只作用于 vLLM worker**，训练侧 Megatron 读的是
`megatron_cfg.moe_router_dtype`。**两处必须一起翻。**

**为什么这条线的 router 是 fp32，而例子 6 是 BF16**：Megatron 走的是它自己的 `TopKRouter`
喂 grouped-GEMM，和 vLLM 不是同一个实现，BF16 下两者会在"近乎平票"的 token 上选出不同专家，
而翻一个专家会让那个 token 的 log-prob 变很多。实测 `moe_router_dtype: null` 时
`rollout_corr/kl` 到 step 77 一直平在 6.5e-4，然后每步约 +16% 爬到 step 110 的 2.4e-2；
换成 fp32 之后到 step 185 都还是 7e-4。

长跑 config 的 `save_steps: 5` 很激进（9.3 min/步 → 约 46 分钟一次 400GB 落盘），
按容错需求调大：`EXTRA_OVERRIDE='checkpointing.save_steps=20'`。

启动后先确认三件事再放手：`MoE+EP spec ... EP=8 ... router_dtype=fp32`、
无 `Traceback` / `HSA_STATUS`、首步在约 14 分钟内出 `callbacks: step=1`。

### 关掉 checkpoint 落盘（磁盘不够时）

```bash
EXTRA_OVERRIDE='checkpointing.save_steps=1000000 checkpointing.resume=false'
```

⚠️ **不要写 `checkpointing.checkpoint_dir=`**。Hydra 会把空值解析成 `None`，
omegaconf 立刻报 `Incompatible value 'None' for field of type 'str'` 并退出。
用一个跑不到的大 `save_steps` 才是干净做法。崩了只能从头跑，先想清楚。

---

## 11. 健康判据

**Smoke 通过的硬判据**：exit 0、无 `Traceback`、无 `HSA_STATUS`、日志里有
`RLTrainer.setup (ray-controller) complete: ... actor_workers=8`。

各例子的 `rollout_corr/kl` / 显存 / step 时间（`resp=20480`）/ checkpoint 大小：

- **例子 1**：kl ≈0.001，`mem/actor_allocated_gb` 11.6 GB，4–5 min/步，ckpt ~90 GB。
  `grad_norm` ~0.85，`ppo_kl` ≈0。
- **例子 2、3**：kl **≈0.003–0.004**（FP8 gap，正常；逼近 TIS 阈值 2.0 才警惕），
  显存与 step 时间同例子 1，ckpt ~90 GB。
- **例子 4**：kl ≈0.004（比 vLLM FP8 略高），显存同上；no-eager level=3 主要加速 rollout，
  但 sleep/wake + 权重同步会增加固定开销。ckpt ~90 GB。
- **例子 5**：kl 应落在例子 1 的量级（≈0.001）而不是例子 4 的 ≈0.004 —— 量化关掉了，
  剩下的只是 ATOM 与训练侧的实现差异。**这条就是判断 ATOM 对齐是否正确的判据**：
  如果 ATOM BF16 的 kl 也在 0.004 量级，说明差异不来自 FP8，去查 §13 的 ATOM RMSNorm 对齐。
  显存、step 时间、ckpt 与例子 4 同量级。
- **例子 6**：kl ~1.5e-3，`mem/actor_max_reserved_gb` 75–115 GB，~11 min/步，ckpt **~342 GB**。
  第 1 步的 `lr` 就是 `9.99998e-07`（满值），说明 warmup 确实是 0；看到 `2e-07` 说明用错了 config。
- **例子 7**：kl ~1.5e-3（健康区间 6e-4 ~ 1.8e-3），allocated 72 GB（4k）/ 130 GB（20k）、
  `max_reserved` 128–140 GB，~9.3 min/步（首步约 14 min，含 vLLM 加载），ckpt **~400 GB**。
  日志应有 `MoE+EP spec: num_experts=128 topk=8 moe_ffn=768 | tp=1 pp=1 cp=1 EP=8 etp=1
  -> local_experts/rank=16 | grouped_gemm=True router_dtype=fp32 pre_softmax=False`。

通用：`timing/weight_sync_s` 1.1–1.7 s 且**不随步数增长**；`mem/actor_allocated_gb` 恒定
（`max_reserved` 随每步 batch 波动是正常的，**存活内存在动才是泄漏**）。

**最重要的一条：`rollout_corr/kl` 不随步数单调爬升。** 它降下去是正常的（策略收敛变确定，
log 空间分歧自然缩小）；爬上去按概率排有三种原因：MoE router 精度两侧不匹配（§10）、
权重同步漏参数（用 `LUMENRL_WEIGHT_SYNC_VERIFY=1` 复查）、或者出现了新的对齐类 bug。

**长度崩塌看 `seq/max_len`。** 它在预算上限附近波动是健康的（说明每步都有序列打满），
单调往下走就是崩了。

### 实测参考曲线

**例子 6**（101 步 / 21.6 小时）：`reward/accuracy` 0.136 → 0.494（step 50）→ **0.581**。
AIME-2024 在线验证（每 10 步，greedy）`val-core/acc/mean@1` 从 step 10 的 0.041 涨到
step 90 的 **0.361**，`val/response_length_mean` 从 2407 涨到 10389 ——模型学会想更久，
这就是这条线跑通的证据。

**例子 7**（`verlref_4k_longrun` 压缩配方，91 步）：`reward/accuracy` 0.168 → 0.42、
`seq/mean_response_len` 773 → 925、`rollout_corr/kl` 0.00136 → 0.00060（在降，正确）、
AIME `mean@1` 0.086 → 0.199。

⚠️ **已知的熵坍缩不是 bug**：例子 6、7 的 config 就是 `entropy_coeff=0`，所以 entropy
单调下降（101 步 0.844 → 0.094）是配置的必然结果。只有在"entropy 掉到 0.05 以下
**且** 长度开始缩"同时出现时才需要警惕。要治得先加 `entropy_coeff`。

---

## 12. 监控 / 停止 / 续跑

```bash
# 监控
sudo docker exec "$CONTAINER" bash -lc 'L=$(cat /tmp/run_dapo_log.txt)
  grep -aE "callbacks: step=" "$L" | tail -5
  grep -aiE "Traceback|OutOfMemory|CUDA error|HSA_STATUS" "$L" | tail'

# 停止（连 Ray actor 一起清）
sudo docker exec "$CONTAINER" bash -lc '
  ray stop --force 2>/dev/null
  pkill -9 -f "[l]umenrl.trainer.main"; pkill -9 -f "[V]LLMRayServer"; pkill -9 -f "[E]ngineCore"
  sleep 10; rocm-smi --showmeminfo vram | grep -i used | head -1'   # 应回到 ~298MB/卡
```

**续跑**：config 里 `resume: true`，重跑同一条长跑命令即从最近 checkpoint 恢复。
新机器目录为空时就是从 step 0 开始。

---

## 13. 排障

**FP8 训练发散**（entropy ≈0.04 / `grad_norm` 1e4+ / `rollout_corr/kl` 1e4+）：
基本只有两个原因——`FP8_PARAM_MANAGER` 没设成 0（它与 native FSDP2 的 fp32-master 冲突），
或 §6 的 vLLM RMSNorm patch 没打（新容器要重打）。

**显存回退（OOM）**：
- 降 `policy.max_response_length=8192` + `max_total_sequence_length=9216` + `max_token_len_per_gpu=9216`；
- 或降 `train_global_batch_size` / `gen_batch_size`；
- Ray 路径 **不要**开 `fsdp_cfg.param_offload` / `optimizer_offload`，会报
  `parameters should be materialized on CPU`。

**例子 7 的 OOM 有个反直觉的点**：`max_tokens_per_gpu: 22528` 时会在 step 14 死在 actor
backward，但**崩溃不是 allocated 峰值的问题**（改前后都是约 130 GB），而是**碎片**——
ROCm 没有 `expandable_segments`，每步约 7 个打满 22.5k 的 bin 反复申请释放巨块，reserved
比 allocated 多出 42 GB。压到 8192 之后碎片间隙塌到 4–11 GB，峰值 reserved 从 177 GB 降到
134 GB。**所以那个 8192 不能随手调回去。**

**`weight sync (colocate-ipc) left N/M rollout parameters untouched`**：同步漏了参数。
异常会列出前 8 个名字；若是 `...experts.w13_weight` / `w2_weight`，说明融合 MoE 路由没生效，
要么代码没拉到最新，要么 vLLM/transformers 升级后布局假设失效了。

**ATOM rollout 退化**（`MODE=atomfp8` / `atombf16` 时 `filter_groups: kept 0/96` +
`Rollout reward: accuracy=0.0000` + 日志大量 `finished with reason max`、无 `eos`）：
rollout 生成崩坏。优先检查 ATOM `atom/model_ops/layernorm.py` 的 plain RMSNorm 有没有传
`use_model_sensitive_rmsnorm=1`；未对齐会先表现为 `rollout_corr/kl` 偏大（~0.007 而非 ~0.004）。
**用例子 5 定位更快**：它把量化关掉了，kl 若仍偏大，问题一定在 ATOM 与训练侧的对齐上，
而不在 FP8。

**`TORCHDYNAMO_DISABLE` 不要手工设。** 脚本全局保持 `=1`（训练 actor 关 dynamo）；
例子 4、5 的 no-eager level=3 rollout 所需的 dynamo 由 `ATOMReplicaManager` 在创建 ATOM Ray
actor 时通过 `runtime_env` 注入 `TORCHDYNAMO_DISABLE=0`，**只作用于 rollout 进程**。
顶层 `export TORCHDYNAMO_DISABLE=0` 会让训练 actor 一并继承，对训练侧纯属副作用。

**例子 4、5 的另外两个前提**（`MODE=atomfp8` / `atombf16` 都会自动设，手工覆盖时别丢）：
`ATOM_ISOLATE_TORCH_COMPILE_CACHE=1`（否则 8 个单卡 replica 并发写同一个 torch compile
cache，触发 Inductor `write_atomic -> rename` 的 `FileNotFoundError`）、
`enable_sleep_mode=true` 且 `sleep_level=2`（rollout 后释放 KV cache / weights / CUDA graph，
否则训练 backward 容易 `HSA_STATUS_ERROR_OUT_OF_RESOURCES`）。

**跑完/中断后显存不释放**（`rocm-smi` 每卡仍 ~90 GB，但 `ps` 里已无 trainer）：
`run_dapo.sh` 只在**启动前**清理进程，收尾不清，所以 ATOM EngineCore 的 `spawn_main`
子进程（及其 inductor compile worker）会变成孤儿继续占显存。手动清理时注意用
`spawn[_]main` 这类写法，否则 `pkill -f` 会匹配到自己的命令行而自杀：

```bash
sudo docker exec "$CONTAINER" bash -lc '
  pkill -9 -f "compile_[w]orker"   || true
  pkill -9 -f "spawn[_]main"       || true
  pkill -9 -f "resource[_]tracker" || true
  sleep 8; rocm-smi --showmeminfo vram | grep -i used | head -3'
```

**换 `DATA_ROOT` 要无条件覆盖。** §4 的 `docker run -e DATA_ROOT=...` 把值烤进了容器环境，
自建 wrapper 里写 `export DATA_ROOT="${DATA_ROOT:-/new/disk}"` **不会生效**（变量已存在），
ckpt 会静默写回旧盘。直接 `export DATA_ROOT=/new/disk`。

**vLLM worker 里的 `logger.info` 不进 driver 日志。** 所以例子 7 看不到那行
`weight sync coverage`。**不能**据此认为断言没跑——判断断言是否触发，看它有没有抛异常。
