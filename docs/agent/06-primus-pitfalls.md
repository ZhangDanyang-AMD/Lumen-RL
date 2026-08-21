# 06 · primus 底座的八条坑

> **稳定操作手册第六篇**，索引见 [`README.md`](README.md)。
>
> 这八条是 2026-08-20 那一轮把 LumenRL 搬到 `rocm/primus:v26.4` 时撞出来的。
> **它们的共同点是失败现场离病因很远**——报错行往往和真正的原因隔着几十帧，
> 有时甚至在另一个进程里。所以这份按「**它实际表现成什么样**」组织，而不是按根因。
>
> 修法基本都在 [`../../scripts/primus/ray_env_primus.sh`](../../scripts/primus/ray_env_primus.sh)
> 里，每一项旁边都写了原因。这份是它的展开解释。
>
> 环境搭建的操作步骤见 [`02-environment-setup.md`](02-environment-setup.md)；
> 与底座无关的运行期坑见 [`05-operational-pitfalls.md`](05-operational-pitfalls.md)。

---

## 症状索引（先按看到的现象查）

| 你看到的 | 去看 |
|---|---|
| `AcceleratorError: CUDA error: invalid argument`，栈停在 `clip_grads.py` 一个一元素的 `torch.zeros` | §1 |
| actor 直接 SIGSEGV，Ray 只说 `ActorDiedError ... SYSTEM_ERROR` | §1 |
| Megatron 分布式优化器**挂死**在第一个优化器步 | §1 |
| `RuntimeError: Device string must not be empty` | §2 |
| 8 个 actor 永远挂在 `No available node types can fulfill resource request {'GPU': 1.0}` | §2 |
| torchrun 子进程 `ZeroDivisionError: integer modulo by zero` | §2 |
| `torch.cuda.device_count()` 返回 0 而 `is_available()` 是 True | §2（也可能是 [`05`](05-operational-pitfalls.md) §1 的孤儿进程） |
| `AssertionError: NVTE_FLASH_ATTN set to 0, but expected 1` | §3 |
| `OSError: libz3.so.4.15: cannot open shared object file` | §4 |
| `ValueError: Feature type 'List' not found` | §5 |
| `ImportError: cannot import name 'extract_to_ir_values'` | §6 |
| `CUDA error: invalid device function`，来自 vLLM 采样器 | §7 |
| `ModuleNotFoundError: No module named 'vime'`，栈深在 `moe_layer.route` | §8 |

---

## 1. ⚠️⚠️ `HSA_DISABLE_FRAGMENT_ALLOCATOR=1` 打死节点内 reduce-scatter

**最贵的一条**，让 Qwen3-8B smoke 连挂五轮。它是从 22.04 配方继承下来的一行。

**症状**：rollout 和前反向全都正常，第一个优化器步死在

```
File megatron/core/optimizer/clip_grads.py, line 109, in get_grad_norm_fp32
  dummy_overflow_buf = torch.zeros(1, dtype=torch.int, device='cuda')
torch.AcceleratorError: CUDA error: invalid argument
```

一个**一元素的 `torch.zeros`** 报错——典型的「异步错误在下一次 API 调用时才落地」。
有时换成 actor 直接 SIGSEGV，Ray 只报 `ActorDiedError ... SYSTEM_ERROR`。

**真现场要去 Ray 的 worker 日志里捞**（`/tmp/ray/session_*/logs/worker-*.err`），
那里能看到栈停在 `param_and_grad_buffer.start_grad_sync` →
`_coalescing_manager.__exit__` → `ProcessGroupNCCL::reduce_scatter_tensor_coalesced`。

**根因**：在 ROCm 7.14 / RCCL 2.28.9 / torch 2.12 上，这个旋钮让**节点内 reduce-scatter** 坏掉：

- **合并形式**（Megatron 分布式优化器用的那个）**挂死**
- **普通形式**算出正确结果，但**把 HIP 上下文留在错误态**，于是下一次分配就炸

**最小复现**：[`../../scripts/primus/probes/probe_rs_coalesced.py`](../../scripts/primus/probes/probe_rs_coalesced.py)，
8 rank torchrun，不需要 Megatron / Ray / vLLM：

```bash
STAGES=plain     torchrun --nproc_per_node=8 probe_rs_coalesced.py   # 数值对，之后上下文已坏
STAGES=coalesced torchrun --nproc_per_node=8 probe_rs_coalesced.py   # 挂死
```

**二分结论**（都在这个探针上做的，一次一个变量）：`HSA_NO_SCRATCH_RECLAIM=1` 无辜、
`HIP_FORCE_DEV_KERNARG=1` 无辜、`CUDA_DEVICE_MAX_CONNECTIONS=1` 无辜、
**`NCCL_CUMEM_ENABLE=0` 无辜**（这条值得单独记，因为它看起来最像嫌疑人）；
ANP/RDMA 那一整组也无辜（`NCCL_IB_DISABLE=1` 照样坏）。
**只有 `HSA_DISABLE_FRAGMENT_ALLOCATOR=1` 单独就能复现。**

⚠️ **all-to-all 完全不受影响**，所以 RDMA 探针再怎么跑也发现不了它。

**修法**：`ray_env_primus.sh` 里不设它。⚠️ **但 `run_dapo.sh` 自己也 export 一份**，
所以那边改成了和 `PYTORCH_CUDA_ALLOC_CONF` 一样的 `${VAR-default}` 写法，调用时要**显式传空值**：

```bash
export HSA_DISABLE_FRAGMENT_ALLOCATOR=
```

⚠️ **只改 `ray_env_primus.sh` 不够**——为此白跑了一轮。

---

## 2. ⚠️⚠️ 缺 `amdsmi`，而补上它会让 torch 数不到卡

**这一条同时解释三种看起来毫不相干的故障，先读完再动手。**

### 故障 A（不补 amdsmi）

vLLM rollout actor 起不来：`RuntimeError: Device string must not be empty`
（来自 `vllm/config/device.py`）。

根因是 `vllm.platforms` 靠 `import amdsmi` + `amdsmi_get_processor_handles()` 判定 ROCm，
而 primus **没把 amdsmi 的 python 绑定放进任何 `sys.path`**（它只在 ROCm SDK wheel 的
`_rocm_sdk_devel/share/amd_smi/` 里）。于是探测抛 `ModuleNotFoundError`，vLLM **静默**
退回 `UnspecifiedPlatform`，`device_type` 变成空串，几十帧之后才以一个完全看不出病因的形式炸掉。

### 故障 B（把 `share/amd_smi` 加进 PYTHONPATH）

`torch.cuda.device_count()` **返回 0**。

torch 在 ROCm 上**优先用 amdsmi 数卡**（nvml 那套的对应物），而且它给自己的 `import amdsmi`
套了一个 ctypes hook，把 `libamd_smi.so` 重定向到加载器先找到的那一份——这是 torch 为绕开
[ROCm/amdsmi#72](https://github.com/ROCm/amdsmi/issues/72) 的 ODR 冲突做的。
在 primus 上那一份是 `_rocm_sdk_devel/lib/libamd_smi.so.26`，与 `share/amd_smi` 里的绑定不配套，
`amdsmi_get_processor_handles()` 返回空列表，**而且不报任何警告**。

```
import amdsmi; import torch   -> device_count 8, RocmPlatform
import torch;  import amdsmi  -> device_count 0, UnspecifiedPlatform
```

**下游症状分布很广，且没有一个提到 amdsmi**：

- **Ray 的 raylet 起来时 `--static_resource_list` 里没有 `GPU`**，`ray status` 也没有 GPU 行，
  8 个 `num_gpus=1` 的 actor 永远挂在
  `No available node types can fulfill resource request {'GPU': 1.0}`
  ⚠️ **driver 仍然会打印 `Ray cluster initialized: 1 nodes x 8 GPUs`——那是读配置文件的，
  什么都不证明。** 判据是 `ray status` 里有没有 `0.0/8.0 GPU`。
- **torchrun 的子进程 `rank % torch.cuda.device_count()` 除零**
- ⚠️ 它是**间歇性**的（取决于该进程里谁先 import），所以很容易被误判成并发/时序问题。
  实际上单进程也稳定复现。

### 修法

在共享树里放一个 `sitecustomize.py`，抢在 torch 之前 `import amdsmi`。
`sitecustomize` 是唯一足够早的钩子——**纯 PYTHONPATH 目录不会被处理 `.pth` 文件**。
已经放好：[`../../scripts/primus/sitecustomize.py`](../../scripts/primus/sitecustomize.py)，
拷进 `$PRIMUS_SITE` 即可。

⚠️ **试过但没用的三条**：`ROCM_PATH` 指向 SDK（torch 的 hook 会先试
`$ROCM_PATH/lib/libamd_smi.so`，那正是坏的那一份）、把绑定自带的库目录放进 `LD_LIBRARY_PATH`
（反而把 vLLM 也弄坏了）、给 `ray start` 传 `--num-gpus=8`（**会被 Ray 覆盖掉，不生效**）。
`RAY_OVERRIDE_RESOURCES='{"GPU":8}'` 能救 Ray 那一半，`ray_env_primus.sh` 里留着当保险，
但它治不了 `torch.cuda.device_count()`。

---

## 3. `NVTE_FLASH_ATTN=0` 是镜像烤进去的，Megatron 会直接断言失败

primus 的 Dockerfile ENV 里有 `NVTE_FLASH_ATTN=0` 和 `NVTE_FUSED_ATTN=1`。
Megatron 建 `TransformerConfig` 时用默认的 `attention_backend=auto`（LumenRL 从不覆盖它），
而 `auto` 会断言三个 `NVTE_*_ATTN` 必须**未设置或已经是 1**：

```
AssertionError: NVTE_FLASH_ATTN set to 0, but expected 1 for attention backend
type auto. unset NVTE_FLASH_ATTN, NVTE_FUSED_ATTN and NVTE_UNFUSED_ATTN.
```

**每个 actor 都会在 `GPTModel.__init__` 里死掉。** 修法就是照报错说的做：

```bash
unset NVTE_FLASH_ATTN NVTE_FUSED_ATTN NVTE_UNFUSED_ATTN
```

⚠️ **别以为「镜像设成 0 一定有道理，应该改成 fused」**——22.04 那些镜像**一个都没设**这些变量，
所以已验证的 FSDP2/Megatron 数字本来就是在 `auto` 下产生的，unset 才是回到那条路径。
（`flash_attn 2.8.3` 在镜像里确实装着，所以 `auto` 有得选。）

---

## 4. `import megatron.training` 死在 `libz3.so.4.15`

调用链：`megatron.training` → `megatron.core.ssm.mamba_mixer` → 镜像自带的 `mamba_ssm` →
`tilelang` → `tvm`，而 `libtvm` 会 dlopen `libz3.so.4.15`。

primus 里有这个库，但**只在 `z3_solver` 的 egg 目录下**，不在加载器路径上。
⚠️ **整条 traceback 里直到最后一帧才出现 z3**，而且 `megatron.core` 本身 import 得好好的，
所以很容易误判成 megatron-core 装坏了。

```bash
LD_LIBRARY_PATH=$(ls -d /opt/venv/lib/python3.12/site-packages/z3_solver-*/z3/lib):$LD_LIBRARY_PATH
```

`ray_env_primus.sh` 里已经用 glob 解析好了（不写死版本号）。

⚠️ 另外这棵树里的 `tilelang` 是 0.1.10，而镜像里的 `mamba_ssm 2.3.1` 钉的是 0.1.8——
pip 会为此警告，但**加上 libz3 之后并不影响**，别去动版本。

---

## 5. 过滤好的 parquet 需要 `datasets>=4`

镜像里是 3.6.0，读不了 `List` 特征类型：

```
ValueError: Feature type 'List' not found. Available feature types: ['Value', ...]
```

⚠️ **不要装 `datasets>=5`** —— 它会连带把 numpy 2.5.2 / pandas 3.0.5 / pyarrow 25 /
huggingface_hub 1.28 全换掉。**4.0.0 用现有依赖就能跑**，`--no-deps` 装。

---

## 6. `flydsl` 要 0.1.8，镜像自带 0.1.6

```
ImportError: cannot import name 'extract_to_ir_values' from 'flydsl.compiler.protocol'
```

来自 `aiter/ops/flydsl/kernels/` 的 GEMM 内核。旧文档里那句「MoE 路线要 0.1.8」说的就是这个。

```bash
pip install --target $PRIMUS_SITE --no-deps "flydsl==0.1.8"
```

---

## 7. ⚠️⚠️ `GPU_ARCHS=native` 在 primus 上等于「没有架构」

镜像预设了 `GPU_ARCHS=native`，而 aiter 的 `csrc/cpp_itfs/utils.py` 解析 `native` 的办法是
shell 出去调 `/opt/rocm/llvm/bin/amdgpu-arch` —— **primus 根本没有 `/opt/rocm`**
（torch 和 ROCm SDK 都在 `/opt/venv`）。于是拿到空串，编译命令变成裸的 `--offload-arch=`，
hipcc 用自己的默认值。

**实测编出来的 `lib.so` 里是 gfx906 和 gfx1250，没有 gfx950。**

**失败发生在启动时而不是编译时**：aiter 打印 `finish build ... cost 19.1s`，二十秒后
vLLM 的采样器（`top_k_top_p_sampling_from_probs`）报 `CUDA error: invalid device function`。

**修法**：`ray_env_primus.sh` 里把 `native` 显式改写成 `gfx950`。

⚠️ **不能用 `${GPU_ARCHS:-gfx950}`** —— 镜像已经设过了，`:-` 永远不生效。为此白跑一轮。

⚠️ 改完要清掉坏的缓存，`docker restart` **不会**清它：

```bash
rm -rf /root/.aiter/build/<op>_*
```

**诊断手法**：`strings <build>/lib.so | grep -oE "gfx[0-9]+" | sort -u`

---

## 8. `vime` 包被 DSv4 router 延迟导入

DSv4 的 router 在 `routing()` 里做延迟导入：

```python
from vime.utils.routing_replay import register_routing_replay
```

所以**直到第一个 MoE 层真的前向才炸**，栈深在 `moe_layer.route` 里，
看起来像 Megatron fork 的问题：

```
ModuleNotFoundError: No module named 'vime'
```

22.04 那条线是 vime 镜像的 site-packages 提供的。修法：`ray_env_dsv4_primus.sh` 里
把 `VIME_SRC`（默认 `/home/<user>/working/vime-rl/vime`）加进 `PYTHONPATH`。

---

## 9. 一条流程纪律

**`docker restart <容器>` 在每次 run 之前。**

失败的 run 会留下几百个孤儿 ray/vLLM 进程（实测 **822 进程 / 660 孤儿**）。
显存回到约 298 MB/卡，所以 `rocm-smi` 看着**完全正常**，但新进程随后会拿到
`torch.cuda.device_count() == 0`。

⚠️ **这一轮有三次 A/B 对照是在被污染的容器上做的，结论全是错的**，直接把 §1 的真因
排除掉了一轮。

确认方式是 `python3 -c "import torch;print(torch.cuda.device_count())"`，**不是 `rocm-smi`**。
详见 [`05-operational-pitfalls.md`](05-operational-pitfalls.md) §1。

---

## 10. 顺带：两个 RDMA 旋钮是 load-bearing 的

不属于「坑」但同样是「少一个就跑不起来」，放在这里方便一起查：

- **`NCCL_IB_HCA=ionic`** —— 排除 `mlx5_0`（它的 `gid[1]` 是 link-local，而且它就是 ens3 那张网卡）
- **`NCCL_CROSS_NIC=0`** —— 这是 rail-optimized fabric，rail 之间**不路由**

两个都是 `ray_env_primus.sh` 的默认值。缺任何一个**不是变慢，是跑不起来**。
