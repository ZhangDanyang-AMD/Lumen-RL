# 02 · 在 MI350X / gfx950 上从零建 LumenRL 环境

> **稳定操作手册第二篇**，索引见 [`README.md`](README.md)。
> 前置：[`01-cluster-access.md`](01-cluster-access.md)（能在节点上执行命令）。
>
> 这份记的是**已经收敛的配方**，不是探索过程。为什么是这些选择、排除过哪些候选，
> 见 §6；出问题查 [`05-operational-pitfalls.md`](05-operational-pitfalls.md)。

---

## 0. 三十秒版本

```bash
# 1. 每次换作业先改这四个值（head 排第一，它同时是 torchrun 的 node_rank 顺序）
vi ~/4node/env.sh            # JOBID / HEAD_NODE / HEAD_IP / NODES

# 2. 每个节点起一次容器（⚠️ 两个变量必须传）
RL24_IMAGE=rocm/primus:v26.4 RL24_CONTAINER=anp-primus \
  bash <repo>/scripts/primus/rl24_container.sh

# 3. 容器内，跑任何东西之前
export RL_ROOT=... DATA_ROOT=...
source <repo>/scripts/primus/ray_env_primus.sh

# 4. 自检
python3 <repo>/scripts/primus/verify_vllm_primus.py    # 期望 VERDICT: vllm usable on primus
python3 $RL_ROOT/megatron_verify.py                    # 期望 VERDICT: ALL OK
python3 $RL_ROOT/probe_ray_gpu_auto.py                 # 期望 8/8 distinct physical GPUs
```

如果共享 python 树（§2）已经在 NFS 上，**换节点不需要重装任何东西**，第 2–4 步就够了。

---

## 1. 底座镜像：`rocm/primus:v26.4`

| 项 | 值 |
|---|---|
| OS / glibc / GLIBCXX | Ubuntu 24.04 / 2.39 / 3.4.33 |
| Python | **3.12.3** |
| torch | 2.12.0+rocm7.14 |
| RCCL | **2.28.9** |
| 自带 vLLM / ray / Megatron | **都没有** |

**它是唯一 RDMA 实测跑通的底座**，而约束比看上去窄：集群的 ANP 插件在 RCCL **2.26.6 和
2.27.7 上都 SIGSEGV**，只有 **2.28.9** 能跑。

⚠️ **这条有干净对照**：`rocm/primus:v26.3` 也是 24.04，插件在里面原样 dlopen 通过
（不需要任何 shim），只变 RCCL 这一个变量 → **16 rank 全 SIGSEGV**，而且崩得很靠后
（插件加载、八张 ionic 枚举、`NET/IB : Using [0]ionic_0 ... [7]ionic_7` 这些行和能跑的
2.28.9 完全一样）。所以 **OS 只决定插件能不能加载，真正卡的是 RCCL**。

> **换镜像时只看一件事**：`strings librccl.so | grep "RCCL version"`。
> ⚠️ **不要看 tag 名**——`nightly_cdna4_..._rocm7.14.0a*` 里的 7.14 是 torch 的构建 ROCm，
> 镜像装的是 7.2.3。现成工具：`~/4node/check_image_for_rdma.sh <image>`。

⚠️ 只有三个数据点，准确说法是「2.27.7 及更早不行，2.28.9 行」，2.28.x 整体如何未知。

⚠️ **py3.12 是硬约束**：仓库里 `aiter` 的预编译 `.so` 用了 `_PyThreadState_UncheckedGet`
（CPython 3.13 已删）。primus 是 3.12.3 ✓。

### 1.1 宿主机前提（逐项确认，缺哪个后面都会以很难懂的方式失败）

```bash
. /etc/os-release; echo $PRETTY_NAME              # 期望 Ubuntu 24.04.x
ls /opt/rocm-*/lib/librccl-net.so                 # ANP 插件，RDMA 必需
ls /usr/lib/x86_64-linux-gnu/libionic.so.*        # ionic provider
ls /sys/class/infiniband/                         # 期望 ionic_0..7 + mlx5_0
cat /etc/profile.d/99-rccl-anp.sh                 # 集群自己的 ANP 配方
ls -d /opt/openmpi
```

⚠️ **`/opt/rocm-7.0.1/lib/librccl-net.so` 只在宿主机上**，任何镜像里都没有，必须 bind-mount。

---

## 2. 共享 python 树（贵的那部分，做一次就好）

primus 不带 vLLM / ray / Megatron，全部装进**一棵 NFS 树**，靠 `PYTHONPATH` 挂进容器。
**换节点不用重装**，这是整套方案里最省事的一个决定。

```
$PRIMUS_SITE = /home/<user>/vllm_primus/site        约 2.7 G
├── vLLM                0.26.0+rocm714    源码编译，5 个扩展
├── ray                 2.57.0
├── megatron-core       0.18.2
├── apex                1.14.0a0          28 个编译扩展，gfx950
├── transformer_engine  2.15.0.dev0+6e541a10
├── datasets            4.0.0
├── flydsl              0.1.8
├── sitecustomize.py    ← 不是包，是 amdsmi 那个坑的修法
└── bin/ray             ← pip --target 没生成 console script，手补的
```

### 2.1 装 vLLM（源码编译，3 分 56 秒）

```bash
bash <repo>/scripts/primus/build_vllm_primus.sh      # -> $OUT/wheels/
bash <repo>/scripts/primus/install_vllm_primus.sh    # -> $OUT/site/
python3 <repo>/scripts/primus/verify_vllm_primus.py  # VERDICT: vllm usable on primus
```

**为什么是编译不是拷贝**：要求「稳定版 + 支持 DSv4」，而机器上稳定版 0.26.0 只有
**torch 2.11 / 22.04 / RCCL 2.27.7** 的构建，拷过来会把 2.27.7 带进去弄坏 RDMA；
唯一 torch 2.12 匹配的 donor 又是 nightly。三个条件锁死。
⚠️ **「ROCm 上编译 vLLM 要一小时以上」这句是错的**——实测 **3 分 56 秒**
（gfx950 单架构、`MAX_JOBS=96`、节点 236 核），带缓存重编 13 秒。**别再因为「编译太贵」去赌跨版本拷贝。**

判据是 5 个扩展全过 + **算子真的注册上了**：

```
OK  vllm._C   _C_stable_libtorch   _moe_C_stable_libtorch   _rocm_C   cumem_allocator
OK  torch.ops._moe_C.topk_softmax    torch.ops._C.rms_norm
OK  DeepseekV4ForCausalLM   backends ['amd','common','nvidia','xpu']
OK  librccl  RCCL version : 2.28.9
```

⚠️ **是 5 个扩展不是 4 个**，旧清单两处都错：`_moe_C` 已改名 `_moe_C_stable_libtorch`；
而 `vllm._C` **自己一个算子都不注册**——导入 `_C` 后 `hasattr(torch.ops._C,"rms_norm")` 是
`False`，导入 `vllm._C_stable_libtorch` 后才 `True`。照旧清单判会假报失败。
⚠️ **`import vllm` 成功不算过**——纯 python 部分能导入而扩展在符号上失败是常见形态。

### 2.2 装 ray

```bash
pip install --target $PRIMUS_SITE --upgrade \
  "ray[default]>=2.50" "protobuf==6.33.6" "grpcio==1.78.0"
```

⚠️ **后两个 pin 不能省**：不钉的话 pip 会把 `protobuf` 顶到 7.35.1（大版本跳）并动到树里
vLLM 那份 `grpcio`。

⚠️ `pip --target` **不会为 ray 生成 console script**，`ray start` 不在 PATH 上。
补一个 `$PRIMUS_SITE/bin/ray`（入口 `ray.scripts.scripts:main`），`ray_env_primus.sh`
已经把 `$PRIMUS_SITE/bin` 加进 PATH。

装完**立刻**验 GPU 隔离（换镜像必测的一条）：

```bash
python3 $RL_ROOT/probe_ray_gpu_auto.py     # 期望 8/8 distinct physical GPUs
```

某些镜像上 Ray 不写 `CUDA_/HIP_/ROCR_VISIBLE_DEVICES`，8 个 actor 全塌到 GPU 0，
症状是 `Free memory on device cuda:0` 和 `Multiple ranks detected using the same GPU`
（对应 ROCm#5780）。✅ **primus 上实测 8/8 一次过**，`BaseWorker._ensure_gpu_isolation` 在这里是 no-op。

### 2.3 装 Megatron（三样，全在 torch 2.12 上原样编过）

```bash
pip install --target $PRIMUS_SITE --no-deps "megatron-core==0.18.2"
bash <repo>/scripts/primus/build_megatron_primus.sh     # apex + TE 两个 wheel
bash <repo>/scripts/primus/install_megatron_primus.sh
```

| 组件 | 版本 / revision | 耗时 |
|---|---|---|
| megatron-core | `0.18.2` | 秒级 |
| ROCm Apex | `daed85255d51476425080e7e6203f0bee6d7e4cc` | 约 2 分钟 |
| **ROCm** TransformerEngine | `6e541a10419a6e31bdc98b1516db04eb81a463b6` → `2.15.0.dev0+6e541a1` | 约 11 分钟 |

⚠️ **绝对不要 `pip install transformer_engine`** —— 会装成 NVIDIA 版，导入即 undefined symbol。
必须用 **ROCm fork**、递归拉全部 submodule（约 2.8 G），并带 `TORCH_DONT_CHECK_COMPILER_ABI=1`
（ROCm 的 `hipcc -v` 在没有输入文件时返回 1，CK-JIT 的编译器探测会误判成「编译器不可用」）。

⚠️ **不要装 megatron-bridge**，Qwen3/DSv4 的 HF↔Megatron 转换由 LumenRL 自己的
`qwen3_megatron_bridge.py` / `dsv4_megatron_bridge.py` 负责。

⚠️⚠️ **`--cpp_ext --cuda_ext` 在这个 Apex revision 上是空操作。** `setup.py` 只是把它们从
`sys.argv` 里删掉，真正的开关是 **`APEX_BUILD_CPP_OPS=1 APEX_BUILD_CUDA_OPS=1`，默认都是 0**。
只传旗标会得到一个**一个 `.so` 都没有的 21 M wheel**，装进去的是 `compatibility/` 里的 JIT 桩，
于是每个 worker 第一次 import 时自己 hipcc 编一遍（单进程 38 秒）。带上环境变量后是
**107 M / 28 个扩展**。⚠️ **`megatron_verify.py` 两种情况都是 ALL OK，查不出这个**——
判据是 `ls $PRIMUS_SITE/apex/*.so | wc -l` 应为 **28**。

⚠️ **源码放节点本地盘**（`/mnt/m2m_nobackup/<user>/megatron_build`），两棵树带全部 submodule
是 apex 633 M + TE 2.2 G。

⚠️ **「torch 2.12 的 API 漂移是最大风险」这句已被证伪**——Apex 和 TE 都一次编过，没有任何漂移。
这一步真正的风险在 Apex 的构建开关（上面）和 `libz3`（§4 第 4 条）。

判据：

```
[ok] megatron-core: 0.18.2
[ok] apex FusedLayerNorm/FusedAdam: (4, 16, 128)
[ok] TE Linear/RMSNorm: 2.15.0.dev0+6e541a10 (4, 16, 256)
[ok] TE DotProductAttention: (32, 1, 128) finite=True
[ok] EngineRegistry: {'fsdp2': ..., 'megatron_native': ...}
VERDICT: ALL OK
```

### 2.4 其它两个包

```bash
pip install --target $PRIMUS_SITE --no-deps --upgrade "datasets==4.0.0"   # 3.6.0 读不了 List 特征类型
pip install --target $PRIMUS_SITE --no-deps --upgrade "flydsl==0.1.8"     # 0.1.6 缺 extract_to_ir_values
```

⚠️ **不要装 `datasets>=5`** —— 会把 numpy 2.5.2 / pandas 3.0.5 / pyarrow 25 / huggingface_hub 1.28
全换掉。4.0.0 用现有依赖就能跑。

### 2.5 ⚠️ 这棵树遮蔽了 primus 的几个包

`PYTHONPATH` **永远盖过** `site-packages`，所以要知道自己换掉了什么：

| 包 | primus 原有 | 挂上之后 | 说明 |
|---|---|---|---|
| `transformers` | 4.55.0 | **5.14.1** | vLLM 0.26 硬要求 `>=5.5.3`。✅ 实测没有绊倒 LumenRL |
| `numpy` | 2.4.6 | 2.3.5（**降级**） | 被 `mistral_common` 的 `numpy<2.4` 压的，不是解析器抖动 |
| `tokenizers` | — | 0.22.2 | 跟 transformers 5 一起来的 |

装新包时的规矩：**`--target $PRIMUS_SITE --no-deps`**，只装真正缺的那些。
⚠️ 不要直接 `pip install --target` 整个 wheel 的依赖闭包——`--target` 不看环境里已有什么，
会把 200 来个包全装一遍，等于把 primus 自己的每个包都遮蔽掉。

---

## 3. 容器

```bash
RL24_IMAGE=rocm/primus:v26.4 RL24_CONTAINER=anp-primus \
  bash <repo>/scripts/primus/rl24_container.sh
```

⚠️ **这两个变量必须传**。脚本的默认镜像还是老的 `rocm/vllm:...py3.14..._vllm_0.23.0`
（py3.14，早就排除了：site-packages 里 339 个 `cpython-314` 扩展），默认容器名是 `rl-vllm-24`。
这是从 24.04 早期实验演化来的遗留默认值。

脚本会把 RDMA 三件套从宿主 bind-mount 进去：

- 宿主的 `/opt/openmpi` → `:ro`。⚠️ **24.04 上必须挂，22.04 上绝对不能挂**（宿主那份自己就要 GLIBC_2.38）
- `libionic.so.1`
- ANP 插件 `/opt/rocm-7.0.1/lib/librccl-net.so` → 挂**四个名字**（`librccl-net.so` /
  `librccl-net-anp.so` / `libnccl-net.so` / `libnccl-net-anp.so`），因为不同版本的 RCCL 找的名字不一样

外加 `--network=host --ipc=host --privileged`、`--device=/dev/kfd --device=/dev/dri`、
`--ulimit memlock=-1`、`--shm-size 64G`。

⚠️ **`docker build` 在 `spur exec` 下**要 `env HOME=/tmp DOCKER_CONFIG=/tmp/dockercfg`
（`$HOME` 是不可写的 `/opt/spur`）。

---

## 4. 运行环境：`ray_env_primus.sh`

**跑任何东西之前 source 它。** 它不是一堆随手攒的变量，每一项旁边都写了为什么。
八条 primus 专属的坑里有六条的修法在这个文件里：

| 它做的事 | 对应的坑 |
|---|---|
| 共享树 + amdsmi + libz3 上 `PYTHONPATH` / `LD_LIBRARY_PATH` / `PATH` | 缺 amdsmi、`libz3.so.4.15` |
| `unset NVTE_FLASH_ATTN NVTE_FUSED_ATTN NVTE_UNFUSED_ATTN` | 镜像烤死的 `NVTE_FLASH_ATTN=0` |
| **不设** `HSA_DISABLE_FRAGMENT_ALLOCATOR` | 它打死节点内 reduce-scatter |
| 把 `GPU_ARCHS=native` 显式改写成 `gfx950` | aiter JIT 编出没有 gfx950 代码的内核 |
| `RAY_OVERRIDE_RESOURCES='{"GPU":8}'` | Ray 卡数检测不稳（保险） |
| 完整 ANP 配方，`unset NCCL_IB_DISABLE` | RDMA 常开（22.04 那三份 env 是写死关掉的） |

DSv4 另有一层：`ray_env_dsv4_primus.sh`（AITER=1、patched Megatron、vendored 插件、
`vime` 包、`NCCL_ALGO=Ring`）。

⚠️ **不要用 22.04 时代的 `ray_env.sh` / `ray_env_dsv4*.sh`**。它们仍是那些镜像的已验证配方，
但在 primus 上有两行是**致命**的：`NCCL_IB_DISABLE=1`（白扔 26 倍带宽）和
`HSA_DISABLE_FRAGMENT_ALLOCATOR=1`（直接打死 reduce-scatter）。

⚠️ **`run_dapo.sh` 会自己 export 一份 `HSA_DISABLE_FRAGMENT_ALLOCATOR`**，所以那边改成了
`${VAR-default}` 写法，调用时要**显式传空值**才真的去掉：

```bash
export PYTORCH_CUDA_ALLOC_CONF= HSA_DISABLE_FRAGMENT_ALLOCATOR=
```

八条坑的完整版（症状 / 根因 / 诊断手法）见 [`../../scripts/primus/README.md`](06-primus-pitfalls.md)。

---

## 5. RDMA 自检

```bash
# 两个节点同时跑，node_rank 跟 env.sh 的 NODES 顺序；head 用 spur exec，另一台用 nx.sh
bash <repo>/scripts/primus/run_a2a_anp.sh 0 --seqs 6144 --iters 6
bash <repo>/scripts/primus/run_a2a_anp.sh 1 --seqs 6144 --iters 6
```

**判据**（四条一起看）：

- `ANP plugin loaded successfully` × 每节点 8
- `NET/IB : Using [0]ionic_0:1/RoCE ... [7]ionic_7:1/RoCE`（**8 张，不含 mlx5_0**）
- 302 MB/rank **每轮约 0.010 s、单 rank 峰值约 40 GB/s**
- **`ens3` 字节计数器全程约 0**

对照：TCP 同尺寸约 0.26 s/轮，**RDMA 快 26 倍**。

⚠️ **不要等 ionic 计数器涨**——八个 `enP*` 也全是 0，RoCE 不过内核 netdev。
正面证据是「`ens3` 为零 + 单 rank 40 GB/s」这一对。

⚠️ **两个旋钮少一个就跑不起来**（不是变慢，脚本里已是默认值）：
`NCCL_IB_HCA=ionic`（排除 mlx5_0，它的 gid[1] 是 link-local，而且它就是 ens3 那张网卡）、
`NCCL_CROSS_NIC=0`（rail-optimized fabric，rail 之间不路由）。

---

## 6. 已经排除的候选，不要重试

| 候选 | 为什么不行 |
|---|---|
| `rocm/primus:v26.3`（24.04 + RCCL 2.27.7） | **16 rank 全 SIGSEGV**。这是那个干净对照 |
| `rocm/pytorch:rocm7.0.2_ubuntu24.04_py3.12...` | 训练是好的，但 RDMA 上 8 rank 全 SIGSEGV |
| `rocm/vllm-dev:nightly-aiter-...` 当 vLLM donor | py/torch 都对，但 **0.26.1rc1.dev，不是稳定版** |
| `yangyuhanintel/rocm-verl-dsv4:0807` / `amdsiloai/vllm-private:qwen3vl-...0805` | 有 0.26.0 稳定版，但 22.04 / torch 2.11 / **RCCL 2.27.7** |
| `rocm/vllm:...py3.14..._vllm_0.23.0` | **py3.14**，339 个 `cpython-314` 扩展 |
| miles 那条线的镜像 | 22.04 / py3.10 / RCCL 2.27.7 / 不带 vLLM |
| 把 2.28.9 单独 preload 进 22.04 镜像 | 和按 2.27.7 编的 torch 冲突，崩得更早 |
| 在 22.04 上拆 glibc 那道墙 | ✅ 能拆（`anp_glibc_shim.sh` + `lower_glibc_verneed.py`，插件加载实测通过），**但它不解决 RCCL**。等出现装了 ROCm 7.14 的 22.04 镜像再说 |

---

## 7. 换节点 / 换作业的检查清单

1. 改 `~/4node/env.sh` 四个值（`JOBID` / `HEAD_NODE` / `HEAD_IP` / `NODES`，**head 排第一**）。
   ⚠️ head 用 `spur exec $JOBID hostname` **实测**，不要假设编号最小。
2. `rocm-smi` 看卡是不是空的——**别人可能占着**，别默认失败是自己的问题。
3. 盘点 `/mnt/m2m_nobackup`：**作业结束不清盘**，命中哪台就省哪台的重建。
4. 每个节点起一次容器。
5. 跑 §0 那三个自检。
6. NFS 树在的话到此为止；不在就回 §2。

⚠️ **NFS 会满**。实测见过 `/home` 到 97%（只剩 319 G），而全 43 层产物是 1.9 TB/节点
（那个走节点本地盘，但 checkpoint 和切片会写 NFS）。**开工前 `df -h /home`。**
