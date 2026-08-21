# DeepSeek-V4 on LumenRL —— 持续开发文档

> **这份文档是给接手的 agent 用的。** 目标是：**新 agent 在新环境里只读这一份（以及它指向的
> `docs/agent/`），就能直接接手开发，不需要任何仓库外的文档。**
>
> ## 开发分支
>
> ```
> dev/dsv4-dapo        （从 main 7545e2b 切出）
> ```
>
> **这条线的所有改动都提交在这个分支上。** 开工先确认：
>
> ```bash
> git -C <repo> rev-parse --abbrev-ref HEAD     # 期望 dev/dsv4-dapo
> ```
>
> ⚠️ **不要在 `main` 上改。** 2026-08-21 已经出过一次事故：一个致命修复
> （`run_dapo.sh` 的 `HSA_DISABLE_FRAGMENT_ALLOCATOR`，见 §4.3 第 1 条）
> 在合并到 main 时被冲掉了，直到重新跑 smoke 才发现。
>
> ## 文档结构
>
> | 文档 | 性质 |
> |---|---|
> | **本文档** | **活的**。当前状态、已解决、未解决、下一步。有进展就更新 |
> | [`agent/`](agent/) | **稳定操作手册**（5 篇）：连集群、建环境、备产物、判据、运行期坑。只在配方本身变化时才动 |
> | [`../scripts/primus/README.md`](../scripts/primus/README.md) | primus 底座八条坑的完整版 + 脚本用法 |
>
> **新 agent 的最短路径**：
> [`agent/01-cluster-access.md`](agent/01-cluster-access.md) →
> [`agent/02-environment-setup.md`](agent/02-environment-setup.md) §0 →
> 本文档 §0（一页现状）→ 从 §9 挑一件事做。
>
> **维护约定**（很重要，请遵守）：
> 1. **有新进展就更新这里**，不要另开文件。
> 2. 保持「**事实 / 已写的代码 / 欠的验证**」三分。用 ✅ 表示实测过、❌ 表示没做过、
>    ⚠️ 表示有坑或有前提。**不要把计划写成 ✅。**
> 3. **被推翻的假设要写进 §6**，那一节是这份文档里最省时间的部分——它让后人不必重走死路。
> 4. 数字要带条件（几个节点、多少层、什么并行度、什么序列长度）。一个没有条件的数字
>    等于没有数字，§6 里有好几条就是这么栽的。
> 5. **操作步骤写进 [`agent/`](agent/)，不要写在这里。** 这份只放当前状态和结论指针——
>    两边都写会立刻不同步。
> 6. 等 §5 清空、§9 的单机与多机两栏都做完，这份文档 + `agent/` 就可以直接压成一份
>    能跑通的 runbook。
>
> **站点相关**：文中的 `/home/xysheng`、`/mnt/m2m_nobackup`、`crsuse2-m2m-*` 是当前集群的路径和
> 节点名。换集群时这些要改，但结论不变。

---

## 目录

- [0. 一页现状](#0-一页现状)
- [1. 目标与路线选择](#1-目标与路线选择)
- [2. 环境与底座](#2-环境与底座)
- [3. 已完成并验证](#3-已完成并验证)
- [4. 已解决的 bug](#4-已解决的-bug)
- [5. 未解决的问题](#5-未解决的问题)
- [6. 不要重做](#6-不要重做)
- [7. 当前实测值汇总](#7-当前实测值汇总)
- [8. 两条权重同步通路](#8-两条权重同步通路)
- [9. 下一步](#9-下一步)
- [10. 文档索引](#10-文档索引)

---

## 0. 一页现状

**最后更新：2026-08-21**

| 环节 | 状态 |
|---|---|
| 底座选型（`rocm/primus:v26.4`） | ✅ 结论已定，唯一 RDMA 实测跑通的底座，§2.1 |
| RDMA（ANP 插件 + ionic RoCE） | ✅ 比 TCP 快 **26 倍**，三次复现，峰值 **39.36 GB/s**，§3.5 |
| primus 上装 vLLM / ray / Megatron | ✅ 全部装完并验过，两节点 `megatron_verify.py` ALL OK，§2.2 |
| Qwen3-8B（vLLM+Megatron+BF16）smoke | ✅ `exit 0`、`rollout_corr/kl=1.044e-03`，§3.5 |
| DSv4 数值正确性（前向 / 反向 / 确定性） | ✅ 已封闭，argmax 96.09%、梯度余弦最差 0.99675，§3.1 |
| DSv4 权重转换链路（四份产物） | ✅ 全模型 `34223` 张量 `0/0/0`，§3.2 |
| 三个显存修复（流式 gather / TP gather / offload fp32） | ✅ 全部在**全 43 层**上验完，§3.3 |
| DSv4 4 层切片端到端 DAPO | ✅ 单节点、2 节点都跑通，`rollout_corr/kl` 2.6–3.3e-3，§3.4 |
| DSv4 4 层切片三探针在 primus 上 | ✅ probe_67/68/70 全过，§3.5 |
| 全 43 层训练侧（4 节点 EP=32） | ✅ 一步 DAPO 跑通，困惑度 **2.88**，§3.4 |
| **全 43 层端到端（带 rollout）** | ❌ **一次都没成功过**，历史三次全死在 `_sync_weights_ipc`，§5.2 |
| **43 层的 `rollout_corr/kl`** | ❌ **从来没有实测值**，只能靠端到端拿到，§5.1 |
| seq 6144 的 EP all-to-all 卡死 | ❌ 未解决，消息尺寸线索已关闭，§5.3 |
| FP8 rollout | ❌ 无通路（vLLM 缺跨精度同步会话），§5.4 |
| DSA indexer 的辅助 loss | ❌ 一次都没接，§5.5 |
| 训推分离（disaggregated RDMA 权重同步） | ⚠️ 机制在仓库里且 DSv4 分支已接通，但 **DSv4 上没跑过**，§8.2 |
| 补丁 Megatron / vendored 插件归档 | ❌ 仍靠 `PYTHONPATH` 注入，没进任何仓库，§5.7 |

**一句话**：数值正确性和显存这两块已经封闭，底座也搬到了带 RDMA 的 primus 上；
**卡住整件事的是「全 43 层端到端从没跑通」，而它的直接后果是 43 层的 `rollout_corr/kl`
至今是个未知数**——那是判断这个模型能不能做 RL 的核心指标。

---

## 1. 目标与路线选择

### 1.1 模型

**DeepSeek-V4-Flash-Base**，284.3 B 参数、43 层，hidden 4096，256 routed experts + 1 shared，top-6。
**专家占 277 B，即 97%**——这个比例决定了后面所有显存账。

⚠️ **RL 必须用 Base 版**。instruct/thinking 版在 `max_response_length` 内永远不闭合，
reward 恒 −1，`filter_groups` 连续 10 轮 kept 0 直接抛错。

四样上游 Megatron 表达不了的结构，是所有工程量的来源：

| 结构 | 麻烦在哪 |
|---|---|
| **mHC 超连接** | 残差是四维 `[s,b,hc,d]`，stock Megatron 的契约是 `[s,b,h]`。反向只存在于 tile_kernels，必须自己重写 |
| **DSA 稀疏注意力 indexer** | ROCm 上只有 AITER 实现，关掉硬 `raise`；只输出整数索引，LM loss 到不了它 |
| **hash 路由 + 门控路由混用** | hash 层没有 `mlp.gate.weight`、`expert_bias` 是 `None`；门控层 `scoring_func` 是 `sqrtsoftplus` |
| **检查点自带命名** | 三套名字（原生 / DeepSeek-HF / vLLM 内部）互不兼容，见 §3.2 |

### 1.2 三条路线，为什么选了第三条

| 路线 | 结局 |
|---|---|
| **miles**（SGLang + Megatron 硬 fork） | ✅ 唯一真正跑出过全 43 层训练 step（4 节点 TP8/PP4/EP8，`grad_norm=0.106`）。**但绑死 SGLang**，且 `_get_parallel_config` 对 32 卡以外直接 `NotImplementedError` |
| **LumenRL + FSDP2** | 权重线收口了（transformers 5.13.1 自带 native→HF 映射，1537 张量 0/0/0），但**没有 EP**，跨节点全靠 all-gather；停在「分发 275 G」那一步 |
| **LumenRL + Megatron-native** | **当前主线**，2026-08-09 起所有工作都在这条上 |

主线的关键取舍：**不接管 miles 的 fork，只 cherry-pick DSv4 增量**（PR #28，19 文件 +511/−146，
70 个 hunk 里 67 个直接落地），再把四个 tilelang kernel 换成纯 PyTorch。

理由是爆炸半径：fork 的 `moe_utils.py:702` 在 `topk_routing_with_score_function` 里——
**那是每个 MoE 模型每次前向都走的路径**，整包接管会让 Qwen3-30B-A3B 也依赖 `miles.utils.replay_base`。
cherry-pick 之后整棵树里的硬 `miles.*` 依赖只剩两处，都是函数内 import。

**意外收益**：纯 PyTorch 路径里没有 `atomic_add`，所以 `--deterministic-mode` 能拿到**位级可复现**；
miles 的 tilelang 反向本质不确定。这把「不放 kernel」从纯成本项变成了有收益的选择。

---

## 2. 环境与底座

**操作步骤全在 [`agent/02-environment-setup.md`](agent/02-environment-setup.md)**，
这里只留结论和「为什么」。

| 结论 | 一句话理由 |
|---|---|
| 底座 `rocm/primus:v26.4` | **唯一** RDMA 实测跑通的。约束是 **RCCL 必须 2.28.9**——2.26.6 和 2.27.7 都 SIGSEGV，`primus:v26.3`（同为 24.04，只差 RCCL）是那个干净对照。**选镜像只看 `strings librccl.so`，不看 tag 名** |
| vLLM / ray / Megatron 装在一棵 NFS 树 | primus 三样都不带。树在 `$PRIMUS_SITE`（约 2.7 G），**换节点不用重装**——这是整套方案里最省事的决定 |
| vLLM 是**源码编译**不是拷贝 | 「稳定版 + 支持 DSv4 + 不带坏 RCCL」三个条件锁死。⚠️ 而且**编译只要 3 分 56 秒**，「ROCm 上编译要一小时」是错的 |
| Apex / TE 用固定 revision 源码编译 | ⚠️ **torch 2.12 没有任何 API 漂移**（旧文档说这是最大风险，已证伪） |
| 跑任何东西前 `source ray_env_primus.sh` | 八条 primus 专属坑里有六条的修法在这个文件里，§4.3 |

**当前树的内容**（版本变了就改这张表）：

```
$PRIMUS_SITE = /home/<user>/vllm_primus/site                约 2.7 G
  vLLM 0.26.0+rocm714 · ray 2.57.0 · megatron-core 0.18.2
  apex 1.14.0a0（28 个扩展）· transformer_engine 2.15.0.dev0+6e541a10
  datasets 4.0.0 · flydsl 0.1.8 · sitecustomize.py · bin/ray
```

⚠️ **`/mnt/m2m_nobackup` 是节点本地 NVMe，换节点就没**。DSv4 产物全在那里：
4 层切片约 460 G，全 43 层约 **1.9 TB/节点**。作业结束**不清盘**，拿到新分配先逐台盘点。
产物的重建见 [`agent/03-dsv4-artifacts.md`](agent/03-dsv4-artifacts.md)。

⚠️ **不在版本控制里的**（§5.7 记着这件事的风险）：`~/dsv4/mhc_probe/`
（探针、`megatron_dsv4` 补丁树、`vendored/miles_plugins`）、`~/4node/env.sh`（每个作业都变）。

---

## 3. 已完成并验证

### 3.1 数值正确性（已封闭）

⚠️ **判据是标定出来的，不是拍脑袋**：vLLM 和 transformers 这两个都被公认正确的实现，
彼此 argmax 只有 **91.4%** 一致、mean `|Δlogprob|` 0.217。所以门槛定在「不差于这条基线」，
**不是 99%**。原话解释了为什么这样有效：「张量放错位会让 top-1 一致率崩塌，而不是动第三位小数。」

| 项 | 实测（单卡 / 4 层切片 / seq 256，除非另注） |
|---|---|
| 前向 argmax 一致率 vs transformers | **96.09%**（手工建模 / 引擎建模 / 引擎确定性模式，三行完全相同） |
| mean / max `\|Δ logprob\|` | 0.049 / 1.20；`engine_compute_log_probs` 5.279e-2 |
| KL(ours‖hf) | 1.347e-3 ~ 1.485e-3（摆幅小于非确定路径自身噪声） |
| 反向梯度余弦 vs transformers autograd | 31 个参数**最差 0.99675** |
| `--deterministic-mode` | bitwise=True、argmax 翻转 **0%**（不开是 1.56%） |
| config 对拍 | `276 fields of MLATransformerConfig`，`UNEXPECTED differences (0)` |
| key-diff（引擎 vs 手工，256 专家） | **2147 参数 / 4243 sharded key 逐字节相同** |
| 全模型（43 层 / 4 节点）真实英文散文困惑度 | **2.88**（均匀分布是 129280） |

⚠️ **反向不能用有限差分**：每层有两处离散选择（MoE top-k、DSA top-k），loss 只是分段光滑。
实测 analytic 6.1e-4 对 numeric 6.8e-2，**差 110 倍**，纯粹是跨越了不连续点。

### 3.2 权重转换链路（四份产物）

| # | 产物 | 全量尺寸 | 命名 | 谁读 |
|---|---|---|---|---|
| 1 | `DeepSeek-V4-Flash-Base` | 275 G | 检查点原生 | 下载来的原始 FP8 |
| 2 | `-bf16` | 544 G | **DeepSeek-HF** | 中间产物 + transformers 参照，不分发 |
| 3 | `_torch_dist` | 530 G | **mcore** | Megatron 训练侧 `dc.load` |
| 4 | `-bf16-native` | 530 G | 检查点原生（bf16 数值） | **vLLM rollout** |

⚠️ **反直觉的一点：原生→HF 的改名发生在第 1 阶段不是第 2 阶段**——`fp8_cast_bf16.py` 反量化**同时**改名。

**为什么必须四份**：
- ①→② 原生是 FP8 块量化（`weight_block_size [128,128]`、`scale_fmt=ue8m0`），训练侧要 bf16。
- ②→③ 绕过 LumenRL HF 加载路径的三个问题：每 rank 读全量、峰值双倍、**完全没有 FP8 反量化**
  （裸 `load_file` 拿到 `float8_e4m3fn` 直接 `.to(bfloat16)`，无视 `_scale_inv`，**得到静默错误的权重**）。
  换来 `dc.load` 的**自动重分片**——转换时的并行度不约束训练时的并行度。
- ④ vLLM 这条线的唯一出路，见 §5.4。②→④ 是**纯重命名不动数值**。

**核对判据**：`python3 check_native_index.py` 期望
`tensors=34223 native=34217 hf_style=0 mtp=0 -> OK`（4 层切片是 3176）。

⚠️ **MTP 要丢掉**：`mtp.0.e_proj.weight` 在「bf16」检查点里**仍然是 float8_e4m3**，
`fp8_cast_bf16` 不认识 `mtp.0.*` 这套名字。放进已摘掉 `quantization_config` 的目录
等于给 bf16 层喂 FP8 张量。丢掉没有代价（vLLM 里 MTP 是独立模型，只在推测解码时构建）。

### 3.3 三个显存修复（全部在全 43 层上验完）

| 修复 | 症状 | 根因 | 实测结果 |
|---|---|---|---|
| **流式 gather** | 全 32 rank `Available Free mem : 0 MB` | `_full_megatron_named_params_moe` 把 EP all_gather 的每一份都存进 `stage` 字典，**上游先物化了，下游 `BucketedWeightSender` 的流式就没有意义** | 切片上同步期峰值 65.8→113.8 变成 65.8→**66.9 GiB**，每 rank 省 47.5 GB；全 43 层 **529.7 GiB 流过、峰值只涨 0.6 GiB** |
| **offload 的 fp32** | fp32 mHC 参数 **9/27** 更新，**不报错** | `_get_sub_optimizer_param_groups` 只给非 fp32 参数建主副本，于是「fp32 + GPU 侧」两个循环都不在；precision-aware 又把梯度放在 `.decoupled_grad`。**梯度好端端挂着，参数永不更新** | 切片 **27/27**、全模型 **87/261 → 261/261**，loss/grad_norm 与基线逐位相同 |
| **TP 的 gather** | vLLM `load_merged_column_weight` **IndexError**：要 2048 收到 1024 | MoE 路径只认 `ShardedTensor`，而 SwiGLU 融合的 fc1 是 **`ShardedTensorFactory`**，落进 `else: gathered[0]` 只发了 rank 0 的一半。**稠密路径一直是对的，MoE 路径漏了** | TP=2 **8 个形状不符 → 0**；全模型 34223 张量 **0/0/0** |

⚠️ **offload 那条是 Megatron 上游的 bug，不是 DSv4 特有的**——任何「fp32 参数 + offload +
precision-aware」组合都中招。**miles 用同样的 offload 跑 DSv4，很可能也在静默中招，未验证。**

⚠️ 流式化引入了一个新的**死锁面**：`get_per_tensor_param` 现在是生成器，每个 yield 后面
都跟着集合通信，**调用方必须抽干**；中途 break 会让别的 rank 卡在永远等不到的 all_gather 上。

### 3.4 规模进展

| 规模 | 状态 | 关键实测 |
|---|---|---|
| 单卡 / 4 层 | ✅ | 见 §3.1 |
| 单节点 8 卡 / 4 层 / EP=8 端到端 | ✅ 3 步 exit 0 | `rollout_corr/kl` 2.62/2.97/3.33e-3，`step_s` 约 172，峰值 114.4 GB |
| **2 节点 16 卡 / 4 层 / EP=8** | ✅ **比单节点更好** | kl 2.74/3.18/3.32e-3（落在单节点区间内），`step_s` 114→97（**每步 1.7×**），峰值 **反而降到 95.2 GB** |
| 全 43 层 / 4 节点 / EP=32 训练侧 | ✅ 一步 DAPO | 不开 offload 峰值 **189.6 GiB / 288**；offload 0.75 是 allocated 117.6 / **reserved 144.7** |
| **全 43 层端到端（带 rollout）** | ❌ | 见 §5.2 |

⚠️ **「2 节点比 1 节点显存更宽松」这个反直觉结论已被实测证实**，机制是
**`EP × EDP = world_size`**，所以每卡专家优化器 = 专家参数总量 × 12 字节 ÷ **world_size**，
**与 EP 怎么选无关**。推论：**「加大 EP 就能装下全 43 层」是错的**。

### 3.5 底座迁移到 primus（2026-08-20/21，job 32407 → 38218）

| 项 | 实测 |
|---|---|
| RDMA 基线 | 302 MB/rank 逐轮 **0.008–0.013 s**、峰值 **39.36 GB/s**、`ens3` 与八个 `enP*` 全零、16 rank 全加载 ANP 插件。对照 TCP 同尺寸 0.26 s/轮 |
| vLLM 树 | 两节点 `VERDICT: vllm usable on primus` |
| Ray GPU 隔离 | **8/8 distinct physical GPUs**，一次过 |
| Megatron 栈 | 两节点 `megatron_verify.py` **ALL OK**。⚠️ **torch 2.12 上 Apex 和 TE 都没有 API 漂移**（旧文档说这是最大风险，已证伪），编译 2 分钟 / 11 分钟 |
| Qwen3-8B smoke | `exit 0`、`step=1`、`actor_workers=8`、**`rollout_corr/kl=1.044e-03`**、`ppo_kl=1.42e-04`、`grad_norm=1.012`、`mem/actor_allocated_gb=57.4` |
| DSv4 4 层切片三探针 | `probe_67` ENGINE OK / 峰值 **57.0–58.3 GiB 卡**；`probe_68` **0/0/0 + PASS**；`probe_70` **`ulp@max 3.58, n_diff 1.486e-04 -> ROUNDING`** |
| 全 43 层产物重建 | 两节点各 275+544+530+530 G，`check_native_index` 两台都 OK。stage 1 下载 275 G 只要 **8 分 06 秒** |

⚠️ Qwen3-8B 的 `mem/actor_allocated_gb=57.4` 比 FSDP2 基线的 11.6 高 **5 倍**，
不是旧注释说的「MoE 上高约 68%」。这是 8B **dense** + 分布式优化器 + TE spec 的组合，
**别拿它推 DSv4 的预算**。

---

## 4. 已解决的 bug

只记有普遍教训的。共同点：**失败现场离病因很远**。

### 4.1 静默错误类（最贵，因为不报错）

| bug | 症状 | 教训 |
|---|---|---|
| `moe_router_dtype` 在 DSv4 路径被静默丢弃 | A/B **逐位相同**，误判参数无用 | **不要在没确认参数真的生效前相信 A/B**，让探针**回显模型实际构建时的值** |
| MoE 权重同步名字不匹配 | vLLM `load_weights` 跳过不认识的名字**不 raise**，93% 的参数没到 rollout | 这个 bug **存活了 54 个训练步**，只因为 `load_weights()` 的返回值被丢掉。现在每次同步必须核对每个参数 |
| `probe_64` 用手写的「vLLM 内部名」当对照 | 报 **101/101 全覆盖而它是错的** | **验证映射必须拿检查点自己的 `index.json` 当 oracle** |
| shared expert 单复数 | `mlp.shared_expert.*` vs `mlp.shared_experts.*`，bridge 的分支是 `if ... in hf` 保护的 | **不报错、静默跳过**，shared expert 完全不加载 |

### 4.2 `PP=1` 把整类 bug 变成恒等式

三个 PP 专属 bug 全是这个形状——`_pp_layer_offset_from_ssd` 在异质层上恒返回 0
（DSv4 命中异质分支）、`expert_bias` 没做跨 stage 广播、dtype 当字符串查一张只有三种浮点的表。

> **教训：不要假设「PP=1 下这段代码是对的」等于「它处理了 PP」。**
> 换拓扑先跑 `syncshapes`，不要拿训练侧探针通过就撤 `_dsv4_check_topology` 的守卫——
> 这个坑真的踩过：PP 在 probe_69 上逐位通过后撤了守卫，端到端**立刻死在权重同步**。

### 4.3 primus 底座的八条（2026-08-20 这一轮的主要产出）

**每一条都以离病因很远的方式失败。**

1. ⚠️⚠️ **`HSA_DISABLE_FRAGMENT_ALLOCATOR=1` 打死节点内 reduce-scatter。**
   22.04 配方继承来的一行。合并 reduce（Megatron 分布式优化器用的）**挂死**，普通 reduce
   **算出正确结果但把 HIP 上下文留在错误态**，于是报错落在 `clip_grads.py:109` 一个
   **一元素的 `torch.zeros`** 上：`AcceleratorError: CUDA error: invalid argument`。
   有时换成 actor 直接 SIGSEGV，Ray 只说 `ActorDiedError ... SYSTEM_ERROR`，真现场在
   `/tmp/ray/session_*/logs/worker-*.err`。
   二分证明 `NCCL_CUMEM_ENABLE`、`HSA_NO_SCRATCH_RECLAIM`、`HIP_FORCE_DEV_KERNARG`、
   `CUDA_DEVICE_MAX_CONNECTIONS` 和整组 ANP/RDMA **全部无辜**，只有它单独能复现。
   ⚠️ **all-to-all 不受影响**，所以 RDMA 探针永远抓不到它。
   最小复现 `scripts/primus/probes/probe_rs_coalesced.py`。
   ⚠️ **`run_dapo.sh` 自己也 export 一份**，所以改成了 `${VAR-default}`，调用时要显式传空值。

2. ⚠️⚠️ **缺 `amdsmi`，补上又会让 torch 数不到卡。**
   不补：vLLM 静默变 `UnspecifiedPlatform`，rollout actor 报
   `RuntimeError: Device string must not be empty`。
   补进 PYTHONPATH：torch 在 ROCm 上**优先用 amdsmi 数卡**，而它的 ODR hook 会把绑定接到
   不配套的 `libamd_smi.so`，**`device_count()` 变 0，且不报警**：

   ```
   import amdsmi; import torch   -> device_count 8, RocmPlatform
   import torch;  import amdsmi  -> device_count 0, UnspecifiedPlatform
   ```

   下游症状没一个提 amdsmi：Ray 的 raylet 起来时没有 GPU 资源、8 个 actor 永远排队；
   torchrun 子进程 `rank % 0` 除零。
   ⚠️ **driver 打印的 `Ray cluster initialized: 1 nodes x 8 GPUs` 是读配置的，什么都不证明。
   判据是 `ray status` 里有没有 `0.0/8.0 GPU`。**
   修法：`sitecustomize.py` 抢在 torch 之前 import amdsmi（`.pth` 对纯 PYTHONPATH 目录不生效）。
   试过没用的：`ROCM_PATH` 指向 SDK、把绑定的 lib 目录放 `LD_LIBRARY_PATH`（反而弄坏 vLLM）、
   `ray start --num-gpus=8`（**会被 Ray 覆盖**）。

3. **`NVTE_FLASH_ATTN=0` 是镜像烤进 Dockerfile ENV 的**，Megatron 默认 `attention_backend=auto`
   断言三个 `NVTE_*_ATTN` 必须未设置或为 1 → 每个 actor 在 `GPTModel.__init__` 就死。
   ⚠️ 别改成 `fused`——22.04 镜像一个都没设，`auto` 才是已验证数字的产生条件。

4. **`libz3.so.4.15` 挡住 `import megatron.training`**（经 mamba_ssm → tilelang → tvm）。
   整条 traceback 到最后一帧才出现 z3，而 `megatron.core` 导入正常，很像 megatron-core 装坏了。

5. **过滤好的 parquet 需要 `datasets>=4`**（`List` 特征类型），镜像里是 3.6.0。
   ⚠️ **不要装 `datasets>=5`**——会把 numpy 2.5.2 / pandas 3.0.5 / pyarrow 25 / hf_hub 1.28 全换掉。

6. **`flydsl` 要 0.1.8**，镜像 0.1.6 缺 `extract_to_ir_values`（aiter 的 flydsl GEMM 内核用）。

7. ⚠️⚠️ **`GPU_ARCHS=native` 在 primus 上等于「没有架构」。**
   镜像预设 `native`，而 aiter 解析它的办法是调 `/opt/rocm/llvm/bin/amdgpu-arch`——
   **primus 根本没有 `/opt/rocm`**（torch 和 ROCm SDK 在 `/opt/venv`）。拿到空串 →
   `--offload-arch=` 空 → hipcc 用自己的默认值。编出的 `lib.so` 里是 **gfx906 和 gfx1250，没有 gfx950**。
   **失败在启动时不在编译时**：aiter 打印 `finish build ... cost 19.1s`，二十秒后 vLLM 采样器报
   `CUDA error: invalid device function`。
   ⚠️ **`${GPU_ARCHS:-gfx950}` 不生效**（镜像已经设过了），要显式改写 `native`；
   改完还要 `rm -rf /root/.aiter/build/<op>_*`，`docker restart` 不清它。
   诊断：`strings <build>/lib.so | grep -oE "gfx[0-9]+" | sort -u`。

8. **`vime` 包被 DSv4 router 延迟导入**（`routing()` 里 `from vime.utils.routing_replay import ...`），
   所以直到**第一个 MoE 层真的前向**才炸，栈深在 `moe_layer.route` 里，看着像 Megatron fork 的问题。

### 4.4 硬件 / 内核类

- **gfx950 在 `torch.einsum("bsgd,grd->bsgr")` 的 batch stride 上越界**：超过约 2040 token 就
  memory fault + **69 GB core dump**。⚠️ **换长度只是换运气**——2040/2043/2044/2048/4096 都崩，
  1792 和 2052 恰好没事。「我在长度 X 上试过没事」不是结论。
  ⚠️ `.contiguous()` **能修前向但反向出 NaN**，要改成每组一次普通 GEMM。
- **indexer 的 tile 要 224 KB 而 gfx950 每 block 只有 160 KB LDS**：45 个测试全挂，
  改 `block_N=128, num_stages=2` 后全过。**不是算法问题是显存预算。**
- **rollout TP=1 在 gfx950 上必然 memory fault**，五种 MoE backend / AITER 组合全部复现过。
  上游验证矩阵里 TP=1 从来没被覆盖。

### 4.5 运维类

| 坑 | 正解 |
|---|---|
| `srun` + 远端 `setsid nohup &` | **被 job step 静默回收**，日志 0 字节。唯一稳的写法：**本地 `setsid nohup` 脱离 + 远端保持前台** |
| `pgrep` 跨会话判存活 | **恒返回 0**，`spur exec` 与 `srun` 命名空间互不可见。**它会诱使你起第二个实例**——真的因此让两个 `hf download` 同写一个目录。可靠信号：**远端日志在增长** > 容器内 `docker exec ps` > 产物变大 |
| `nx.sh` 从 `spur exec` 里调 | `Permission denied` + 退出码 127（`$HOME` 是不可写的 `/opt/spur`）。**只能从登录节点跑** |
| `run_probe_full43.sh` 的退出码 | **恒为 0**（最后一行是 `echo`，没开 `set -e`）。曾把一次全 rank SIGABRT 报成成功。看 `MILESTONE` / `FORWARD:` / `STEP:` |
| `rocm-smi` 在进程退出后量显存 | **恒读 0**。要运行期每 5 秒采样、事后取每卡最大 |
| **孤儿进程让机器「看不到 GPU」** | 失败的 run 留下几百个 ray/vLLM 进程（实测 822 进程 / 660 孤儿）。**显存回到 298 MB/卡所以 `rocm-smi` 看着完全正常**，但新进程 `device_count()` 变 0。⚠️ **这一轮有三次 A/B 因此得出完全错误的结论**，把 §4.3 第 1 条的真因排除掉了一轮 |
| 读 MEMDIAG | 8 个 rank 往同一个 fd print 会连成一行，先 `sed "s/MEMDIAG/\nMEMDIAG/g"` |
| 探针的 cwd | GPU memory fault 会往 cwd 写 **50–80 GB** core dump，先 `cd /mnt/m2m_nobackup/.../probe_scratch` |

---

## 5. 未解决的问题

按重要性排。**每条都注明卡在哪、已知线索、为什么还没解决。**

### 5.1 ❌ 43 层的 `rollout_corr/kl` 从来没有实测值 —— 最关键的空白

它是训练与 rollout 一致性的核心指标，**只能靠端到端拿到**（探针碰不到）。
切片上是 **2.6–3.3e-3**；43 层上预期显著更大，因为一个 bf16 ulp 会被 43 层放大
（§6 有量化数据）。参照：miles 的 DSv4 线是 **0.07**，Qwen3-30B-A3B 长跑约 1.7e-3。

**卡在 §5.2。**

### 5.2 ❌ 全 43 层端到端一次都没成功过

历史三次全死在 `_sync_weights_ipc`。三个显存修复修完之后，**只验了发送侧**——
`probe_68` 走的正是 `update_weights_ipc_send` 调的那个 `get_per_tensor_param()`。
**剩下的两段探针碰不到**：`BucketedWeightSender` 的 ZMQ / CUDA-IPC 传输，和 vLLM 接收端的 in-place load。

⚠️ **如果还死在 `_sync_weights_ipc`，不要再往显存和张量名上查**——那两处已经分别被
§3.3 和 §3.2 排掉了。

最近一次尝试是作业直接 `NODE_FAIL`。**缺机器 + 运气**，不是没定位到。

⚠️ **这两段在单节点上就能验**（colocated 的训练 actor 和 vLLM replica 是同机进程间传输），
见 §9。另有一条替代通路见 §8.2。

### 5.3 ❌ seq 6144 的 EP all-to-all 卡死

```
WorkNCCL(SeqNum=4, OpType=ALLTOALL_BASE, NumelIn=164364288, NumelOut=150994944,
         Timeout(ms)=1800000) ran for 1800009 milliseconds before timing out
```

`NumelOut = 6144 × 6 × 4096` → bf16 **302 MB/rank**。起跑约 90 秒进入，30 分钟后全 rank SIGABRT。
同一 PG 上**前 3 个集合通信正常完成**；seq 1280（63 MB/rank）在同一套节点上跑完。
**63 MB 过、302 MB 一个字节不走 → 是卡死不是慢。**

**消息尺寸这条线索已经关闭**：2 节点上同尺寸 0.8 秒、**1.8 GB/rank（6 倍）也过**。
剩下三个候选，**最像的是「那 4 台里有一对在大流量下坏掉」**——它正好解释这个形状，
而且如果坏的不在测试的那一对里，2 节点怎么压都是干净的。

⚠️ **所有 4 节点数字都是 TCP（`NCCL_IB_DISABLE=1`）下测的，而 primus 上 RDMA 常开。
先在 RDMA 上看它还在不在，再动手二分。**

**拿到 4 节点的第一件事就做这个**（15 分钟出结果，排在端到端前面）：改 `env.sh` 指向一对节点，
两边各跑一次 `run_a2a_ens3.sh 0|1 --seqs 6144`，把 6 对全过一遍。

### 5.4 ❌ FP8 rollout 没有通路

两端现在都 bf16，全模型 rollout 要 **71 GB/卡**而 FP8 只要 **36**。
**卡点不是显存是精度通路**：vLLM 缺 SGLang 那个跨精度同步会话
（`begin_weight_update` / `end_weight_update`），只对它自己 online 量化的模型
（`fp8_per_block`）有等价的 layerwise reload。三条死路见 §6。

### 5.5 ❌ DSA indexer 的辅助 loss 一次都没接

indexer 只输出整数 top-k 索引，**LM loss 到不了它，不接就永远不训练**——
而它决定注意力看哪些 token。Megatron 有 `dsa_indexer_loss_coeff` 说明它靠独立辅助 loss 训练。
对 RL 尤其要紧：等于把稀疏模式钉死在预训练状态。

### 5.6 ⚠️ 拓扑

| 维度 | 状态 |
|---|---|
| EP / DP | ✅ 一直在用（EP=8 切片、EP=32 全模型） |
| TP | ✅ 能用，但**只在 TP=2 上验过，TP=8 没试**。⚠️ MoE+TP 会强制打开 `sequence_parallel`，那条路第一次跑 |
| PP | ✅ 4 层切片上端到端打通、守卫已撤（loss 逐位相同、syncshapes 3176/3176、kl 1.8–3.0e-3）。❌ **但全 43 层用不了**：`num_layers % pp == 0` 是硬断言而 **43 是质数** |
| CP | ❌ **实测是错的**：CP=2 下 **509/511 个位置不同，从位置 0 就开始**。已排除 chunk 对齐和 padding；嫌疑在 DSv4 注意力内部的 CP 处理。按要求搁置，守卫保留 |

### 5.7 ❌ 代码归属未定（有意暂缓，但风险在涨）

补丁过的 Megatron（`~/dsv4/mhc_probe/megatron_dsv4`）和 vendored 的 `miles_plugins`（95 K，约 2396 行）
**至今靠 `PYTHONPATH` + `MEGATRON_OVERRIDE` 注入，没进任何仓库**。四个纯 PyTorch 替代实现同样。

这是 2026-08-06 有意暂缓的决定（等方案定型再一起归档，且 vendor 第三方代码需确认许可），
**但 2026-08-21 已经出过一次事故**：`run_dapo.sh` 里 §4.3 第 1 条的修复在合并到 main 时被冲掉了。
`scripts/primus/` 那批脚本已经入库，这两棵树还没有。

### 5.8 其它

- **残留数值差异**约 0.05 mean `|Δlogprob|` 未逐层定位。最便宜的下一步是直接对拍两条
  FP8 反量化路径（miles 的 `fp8_cast_bf16` vs transformers 的 `FineGrainedFP8Config`），
  **它们从没被互相对拍过**。
- **路由冻结的 A/B 没做**：miles 冻（`--moe-router-freeze-gate`），我们没冻，没有对比数据。
- **miles 的 `train_rollout_kl = 0.07` 悬案**：比 Qwen3 基线高两个数量级。
  一个具体嫌疑是 NVIDIA 版有个 TE 精度覆盖 YAML 把 indexer 的 `linear_weights_proj`
  在 FP8 下钉在 BF16，**AMD 分支把这一项整个删掉了**。判别实验是 `--fp8-training false`
  跑 BF16 对照——若 kl 掉回 1e-3 量级就说明偏差来自 FP8 路径而非权重同步。

---

## 6. 不要重做

**这一节是这份文档里最省时间的部分。** 每条都是实测排除过的。

### 6.1 判据类（最容易被误用）

| 度量 | 为什么不能用 |
|---|---|
| **`max \|Δ log-prob\|` 当距离用** | **它是混沌量**。`moe_router_dtype=fp32` 让隐状态分歧减半（512→300）却让它涨了近三倍（0.354→0.955）。不同轮次的 9.108e-01 / 3.539e-01 / 9.546e-01 **彼此不可比**。稳定的量是「不同位置的比例」（两组都约 80%） |
| **「32 rank 逐位相同」当正确性门槛** | 43 层 bf16 MoE **做不到**，≤1152 token 时能过是侥幸。留着它会让探针在每个真实长度上都是红的——「**那比没有判据更糟，因为它训练读者忽略判据**」 |
| `max\|d\| / max\|value\|` | 分歧元素通常远小于张量峰值，**无论什么原因都读作「不到一个 ulp」** |
| 逐元素相对差取全体最大 | **被接近零的元素主导**（+1e-8 vs −1e-8 得到恰好 2.0），实测层 19 之后每层都报 2.000 |
| loss 当拓扑判据 | CP=2 错到 509/511 个位置，**loss 仍然逐位相同**（被 advantage 主导） |
| 整模型有限差分验梯度 | analytic 6.1e-4 对 numeric 6.8e-2，差 110 倍 |
| argmax 一致率定 99% | 两个公认正确的实现彼此才 91.4% |
| `max_memory_allocated` 做显存规划 | 要看 **reserved**——那才是 rocm-smi 和 colocated vLLM 争的东西，init 时就高 26 GiB |
| 短序列上量显存然后外推 | 旧文档「余 98 GB」是在 **~130 token** 上量的，1280 就撑满 |
| 常驻显存判断 offload 生效 | 优化器的 fp32 master 和 Adam 动量是**第一次 `step()` 才落地**的 |

### 6.2 已经查清、不要再查的现象

- **「层 18 有什么特别」** —— 已退休。seq 6144 下第一个发散层是 **3**。它只是「该长度下第一次
  有元素跨过舍入边界」的位置。也**不要**去找「1152 和 1280 之间的特殊常数」——那个门槛是运气不是机制。
- **32 rank 前向不一致不是 bug** —— 是**一个 bf16 ulp 被 43 层放大**：层 0–17 逐位相同，
  层 18 首次出现 3.125e-02（正好一个 ulp），之后每层约 1.35×，层 42 到 512。
  ⚠️ 放大比想象剧烈：`n_diff` 在**一层之内**从 6e-05 跳到 0.235。
- **seq 6144 卡死的消息尺寸线索已关闭**（§5.3）。`NCCL_ALGO=Ring` 和不均衡切分**都不是触发条件，
  10 轮全过，别再做这两个 A/B**。
- **「修 RDMA 会顺带修掉卡死」** —— 不会，别把两件事混起来。一个是性能问题一个是正确性问题。

### 6.3 试过不行的路

**FP8 rollout 的三条死路**（要点是「谁拥有量化」）：
指向发布的 FP8 目录 → bf16 同步死在 `'Parameter' object has no attribute 'load_merged_column_weight'`；
指向 `fp8_cast_bf16` 输出（HF 名）→ vLLM 连加载都失败 `KeyError: 'layers.0.input_layernorm.weight'`；
bf16-native + `quantization: ""` → 能跑但 71 GB/卡放不下。
**活路只有 bf16-native + `quantization: fp8_per_block`。**

**其它**：

- ❌ 把 fp32 分片钉在 GPU 侧躲开 offload —— **从 9/27 变成 0/27**（能更新的那 9 个恰恰在 CPU 侧）
- ❌ `torch.cat(gathered, dim=0)` 拼 SwiGLU 的 fc1 —— 修好尺寸但**静默搞错顺序**，比原 bug 更危险
- ❌ 用 `param.tensor_model_parallel` 判断要不要跨 TP 拼 —— DSv4 的 `wq_a`/`wkv` 等带着这个标记
  却**是真复制的**。只有 shard 元数据能判
- ❌ 给 PP 的 dtype 元数据补表项 —— 是把失败面往后挪一格，正解是直接传 `torch.dtype`（pickle 是同一 singleton）
- ❌ 装 tilelang / tile_kernels —— 四个 kernel 全有对拍过的纯 PyTorch 替代，装进去会重撞版本墙
  **而且失去位级可复现**
- ❌ 用 mask 而不是 gather 实现稀疏 MQA —— mask 表达不了重复索引，代价 O(S×S_kv) 而非 O(S×topk)
- ❌ 「加大 EP 就能装下全 43 层」 —— 专家优化器只随 `world_size` 缩，与 EP 无关
- ❌ 用「每节点 16 进程 × 2 节点」凑 32 rank —— NCCL 不允许一 rank 多卡（`Duplicate GPU detected`）
- ❌ 在 22.04 上追 RDMA —— glibc 那道墙可以拆（`lower_glibc_verneed.py` 实测通过）
  **但它不解决 RCCL**，见 §2.1
- ❌ 用 vLLM 的 `prompt_logprobs` 当金标准 —— 实测在这个模型上有位置给出完全不合理的分布，
  **唯一一处实质分歧，错的是 vLLM**
- ❌ 自己写 native→HF 转换器 —— transformers 5.13.1 自带，全量 43 层验过 1537 张量 0/0/0
- ❌ 照抄 SGLang 的 `remap_weight_name_to_dpsk_hf_format` 当 HF 映射 —— 它的目标命名和
  transformers 不一致。⚠️ 但**取其逆**回到原生名是对的（正因为只做前缀改写所以可逆）

---

## 7. 当前实测值汇总

**判据本身**（阈值怎么定的、为什么这么定、以及哪些判据是错的）在
[`agent/04-probes-and-criteria.md`](agent/04-probes-and-criteria.md)。
这里只放**这条线目前测到的数字**——它们会随进展变，所以放在活文档里。

### 7.1 正确性

| 指标 | 实测 | 条件 |
|---|---|---|
| 前向 argmax 一致率 vs transformers | **96.09%**（门槛是 91.4%） | 4 层切片 / seq 256 |
| 反向梯度余弦 | 最差 **0.99675**（31 个参数） | 同上 |
| `rollout_corr/kl` | 1 节点 2.62/2.97/3.33e-3；2 节点 2.74/3.18/3.32e-3 | 4 层切片 EP=8 端到端 |
| ↳ 同上 | ❌ **43 层从来没有实测值**（§5.1） | — |
| 权重同步形状 | `0/0/0` + PASS | 切片 3176、全模型 34223 |
| 流式 gather | 切片 65.8→**66.9 GiB**；全模型 **529.7 GiB 流过、峰值 +0.6 GiB** | EP=8 / EP=32 |
| fp32 mHC 参数更新 | 27/27（切片）、**261/261**（全 43 层） | offload 0.75 |
| `probe_70` `ulp@max` / `n_diff` | 1280: 2.68 / 6.22e-05；6144: 1.88 / 6.59e-05；切片在 primus 上 **3.58 / 1.486e-04** | 全部判为 ROUNDING |
| 全模型困惑度（真实英文散文） | **2.88** | 43 层 / 4 节点 EP=32 |
| Qwen3-8B smoke（primus 回归） | `kl=1.044e-03`、`ppo_kl=1.42e-04`、`grad_norm=1.012` | 1 节点 8 卡 |

### 7.2 显存（MI350X，288 GB/卡）

| 项 | 数字 |
|---|---|
| 全模型 gather（修前） | **529 GiB/rank** —— 永远过不去 |
| 全 43 层一步峰值 | **189.6 GiB**（offload 0）/ allocated 117.6 + **reserved 144.7**（offload 0.75） |
| 全模型 rollout | FP8 **36 GB/卡** vs bf16 **71 GB/卡**（TP=8） |
| colocated 合计 | 约 **202 GiB / 288**，放得下 |
| 4 层切片一步峰值 | TP=1 **70.3** → TP=2 63.8 → PP=2 61.2 GiB |
| 切片累计收益 | `mem/actor_max_allocated_gb` **114.35 →（流式 gather）66.88 →（+offload）41.17 GB** |
| EP=8 下优化器 | `284B × 12 / 8 = 426 GB/rank` —— **单节点装不下的就是这个** |
| EP=8 下权重同步 | 只要权重 **83 GB**（实测 init 78.2 GiB）——所以 `build_optimizer=False` 的探针单节点跑得了 |
| Qwen3-8B（primus） | `mem/actor_allocated_gb` **57.4**。⚠️ 比 FSDP2 基线 11.6 高 5 倍，**别拿它推 DSv4 预算** |
| 主机内存 | 每节点 2.7 TB，**不是约束** |

### 7.3 DSv4 特有的必设开关

通用的（`NCCL_IB_HCA` / `HSA_DISABLE_FRAGMENT_ALLOCATOR` / `PYTORCH_CUDA_ALLOC_CONF` 等）
已经在 `ray_env_primus.sh` 里，见 [`agent/02-environment-setup.md`](agent/02-environment-setup.md) §4。
下面这几个是 **DSv4 模型本身**要求的：

| 开关 | 为什么 |
|---|---|
| `VLLM_ROCM_USE_AITER=1` | 稀疏注意力 indexer 在 ROCm 上**只有 AITER 实现，硬 raise 没有回退**。⚠️ 与 Qwen3 那条线冲突（那边要 0），所以 DSv4 单独一份 env |
| `moe_backend: triton` | ⚠️ **必需项不是优化项**。`triton_unfused` 是 FP4 专用；`aiter` 和 `auto`（会自动挑中 AITER）都死在 API 漂移 |
| `kv_cache_dtype=fp8_e4m3` | 不设会 `AssertionError: DeepseekV4 FlashMLA fp8 layout only supports fp8 kv-cache` |
| `--deterministic-mode` + **`NCCL_ALGO=Ring`** | 不开有 1.56% argmax 逐次翻转；不设 ALGO 会在参数校验就 assert |
| `--no-gradient-accumulation-fusion` | 没有 DDP wrapper 时 `main_grad` 是 None |
| `--disable-bias-linear` | mHC 的 `layer_post` 断言子层 bias 为 None |
| **seq 是最大 `compress_ratio` 的整数倍，且 ≥ 2×** | `compressor.forward_raw` 有断言。L4 上 ratio 最大 128，**最小有意义长度是 256**——seq 128 测不到 HCA 层 |
| `quantization: fp8_per_block` + `-bf16-native` | rollout 的唯一活路，§5.4 |

---

## 8. 两条权重同步通路

这是当前最值得关注的架构选择点。

### 8.1 colocated：ZMQ + CUDA IPC（当前 DSv4 走的路）

```
rl_trainer._sync_weights_ipc          (rl_trainer.py:729，总入口)
  → backend == "auto" 且同机且 replicas == actor_workers
  → BucketedWeightSender               (bucketed_weight_transfer.py:115)
    ZMQ REQ/REP 控制面 + reduce_tensor() 传 HIP IPC handle，共享一个固定 buffer
  → vLLMColocateWorkerExtension.update_weights_from_ipc
    → FusedMoEWeightRouter + assert_weight_sync_coverage
```

**状态**：切片上跑通（1 节点 / 2 节点），**43 层上发送侧已验、传输和接收端从没跑过**（§5.2）。

### 8.2 disaggregated：独立进程组 + RCCL broadcast（仓库里有，DSv4 没跑过）

```
rl_trainer._sync_weights_rdma          (rl_trainer.py:787)
  → send_weight_stream                 (rdma_weight_transfer.py:79)
    9-rank 独立 torch.distributed 组：rank 0 = Megatron actor rank0，rank 1-8 = vLLM TP worker
    每桶三次 dist.broadcast：4-word int64 header / JSON metadata / payload
  → receive_weight_stream              (rdma_weight_transfer.py:141)
    零拷贝切片 .view() 直接喂 model.load_weights()
```

⚠️ **它没有自己写 verbs，也没用 mori-io / NIXL / UCX**。`backend="nccl"` 在 ROCm 上映射到 RCCL，
RDMA 完全交给 RCCL 自己的 IB/RoCE transport。判定是否真走了 GPU Direct 只能抓日志：
**`Using network IB` + `NET/IB/0/GDRDMA` 两行都要有**。

唯一 tricky 的是 `lumenrl/utils/independent_process_group.py`（75 行）：sender 和 receiver
属于完全不同的进程世界，`torch.distributed.new_group` 只能从 default world 挑 rank，
所以走 `rendezvous()` + `PrefixStore` + `_new_process_group_helper`，最后手动补
`_world.pg_group_ranks[pg]`。注释说这是照 MILES 的 multi-main-process-group 模式抄的。

**已有实测**（`examples/docs/07-disaggregated-rdma.md` 引用的 doc 05，**Qwen3-30B-A3B / 2 节点 MI308X**）：
每步广播 **61.1 GB / 58 桶、2.51–3.90 s、有效 134–215 Gb/s**，`rollout_corr/kl` 8.4e-4。
⚠️ 200 步 longrun 在文档里只是「Target」，**没有跑完的结果**；端到端 step time 和显存实测都没有。

**两条路的关系**：分叉点在 `rl_trainer.py:745` 按 `weight_sync.backend` 分派
（`auto` / `shared_folder` / `rdma`）。**共用的只有首尾**：训练侧 `get_per_tensor_param()`
和 rollout 侧 `model.load_weights()`。中间的打包、握手、缓冲区管理各写各的。

⚠️ **RDMA 路径不走 `FusedMoEWeightRouter`**，直接 `load_weights()`。对 DSv4 不是问题
（`megatron_to_dsv4_native` 已经产出 checkpoint 名），但接别的 MoE 时要注意。

### 8.3 对 DSv4 的含义

**好消息**：`megatron_native_engine.get_per_tensor_param()` 里 **DSv4 分支（`megatron_to_dsv4_native`）
已经接好了，两条传输路共用它**。所以切到 RDMA 主要是配置层的事。

**要算的账**：61.1 GB ≈ 30B × 2 字节，是**全量广播没有增量**。DSv4 是 284 B → 约 568 GB/步。
按 134–215 Gb/s 线性外推是 **21–35 秒/步**。我们的 ionic fabric 实测 39.36 GB/s（约 315 Gb/s）
可能更快，但 `bucket_size_mb` 和 `timeout_s` 都要重算。

**两道闸门值得立刻抄到 colocated 路径上**（不需要等训推分离）：
`version = global_step + 1` 的版本校验；`verify_full_load` 用 `model.named_parameters()`
做差集。后者正是那个**静默存活 54 步**的 MoE bug 的解药。

⚠️ **一个未验证的推测**：RDMA 路径的代码**不检查两端是否同节点**，同节点时 RCCL 会走 XGMI/P2P。
所以理论上单机也能跑这条路，用来绕开 §5.2 那两段没验过的 CUDA-IPC 传输。
**这是从代码推的，文档里没有实测记录，不要当结论用**——但验证成本很低。

---

## 9. 下一步

### 9.1 单机就能做（当前只有一台机器时按这个走）

| # | 事项 | 为什么值得做 | 预估 |
|---|---|---|---|
| 1 | **DSv4 4 层切片端到端 1 节点 shortsmoke** | 唯一能在单机推进 §5.2 的动作——`_sync_weights_ipc` 剩下的两段是**同机进程间**传输，不跨节点 | 30 分钟 |
| 2 | **试 `weight_sync.backend: rdma` 在单机上** | §8.3 那个推测。成功就多一条不依赖 CUDA-IPC 的通路，失败也知道这机制对 colocated 不适用 | 1 小时 |
| 3 | **把两道闸门加到 colocated 路径** | version 校验 + `verify_full_load`，见 §8.3 | 小 |
| 4 | **全 43 层单机探针**（`probe_67` 真验「三题答对」+ `probe_68` TP=1/TP=2 回归到 primus） | 把 43 层的加载层问题在拿到多机前排掉 | 1.5 小时（含 1.9 TB 重建） |
| 5 | **归档补丁树和 vendored 插件**（§5.7） | 已经出过一次「合并到 main 冲掉修复」的事故 | 中 |
| 6 | **对拍两条 FP8 反量化路径**（§5.8） | 最便宜的一步，可能解释残留的 0.05 mean\|Δlogprob\| | 小 |
| 7 | **带完整 DSv4 环境跑一次 RDMA 探针** | 单机上不搬字节，但**插件加载和八张 ionic 的枚举在初始化阶段就能观察到**，提前排掉「`NCCL_ALGO=Ring` 和 DSv4 那棵 PYTHONPATH 会不会影响插件」这两个未知交互 | 5 分钟 |

### 9.2 需要多机

| # | 事项 | 备注 |
|---|---|---|
| 1 | **逐对压那 6 对节点**（§5.3） | **拿到 4 节点的第一件事，排在端到端前面**，15 分钟出结果。⚠️ 先在 RDMA 上看卡死还在不在 |
| 2 | **4 节点端到端 shortsmoke**（seq 1280） | 拿 43 层的 `rollout_corr/kl`（§5.1） |
| 3 | **DSv4 负载跨节点走 RDMA 的首次验证** | 昨天验的 RDMA 是独立探针，不带 DSv4 环境 |
| 4 | 全 43 层在 RDMA 下重验 | §3.4 那些数字都是 TCP 下测的 |

---

## 10. 文档索引

**这条线不需要引用任何仓库外的文档。** 下面就是全部。

### 10.1 稳定操作手册 —— [`agent/`](agent/)

配方定下来之后基本不动的那部分。**新 agent 从这里开始。**

| 文档 | 什么时候读 |
|---|---|
| [`agent/01-cluster-access.md`](agent/01-cluster-access.md) | 连集群、进节点、起长任务、判存活 |
| [`agent/02-environment-setup.md`](agent/02-environment-setup.md) | 建环境（镜像 / 容器 / NFS 树 / RDMA 自检），以及已排除的镜像候选 |
| [`agent/03-dsv4-artifacts.md`](agent/03-dsv4-artifacts.md) | 四份权重、三套命名、切片和全 43 层的重建 |
| [`agent/04-probes-and-criteria.md`](agent/04-probes-and-criteria.md) | 探针目录、判据阈值、**以及哪些判据是错的** |
| [`agent/05-operational-pitfalls.md`](agent/05-operational-pitfalls.md) | 运行期的坑。**出问题先翻这份** |

### 10.2 仓库里的其它相关文件

| 路径 | 内容 |
|---|---|
| [`../scripts/primus/README.md`](../scripts/primus/README.md) | primus 底座**八条坑的完整版**（症状 / 根因 / 诊断手法）+ 脚本用法 |
| [`../scripts/primus/`](../scripts/primus/) | 脚本本体。每个环境变量旁边都写了为什么 |
| [`../examples/DAPO/configs/dapo_dsv4_flash_*.yaml`](../examples/DAPO/configs/) | DSv4 的四个配置：4 层 1node smoke/shortsmoke、全 43 层 4node/4node_shortsmoke。头部注释带实测值和风险标注 |
| [`../examples/docs/07-disaggregated-rdma.md`](../examples/docs/07-disaggregated-rdma.md) | 训推分离部署全流程（另一条线：Qwen3-30B-A3B / 2 节点 MI308X），§8.2 的来源。有 `_cn` 版 |
| [`../examples/docs/05-multinode-rdma.md`](../examples/docs/05-multinode-rdma.md) | §8.2 的实测数字在它的 §5.5 |

⚠️ [`../examples/DeepSeekV4_OPD_MI300/`](../examples/DeepSeekV4_OPD_MI300/) **不要当参考**：
它是 OPD（在线策略蒸馏，不是 RL）、单节点、`generation_backend: hf`，README 里**没有任何实测数据**，
而且配置自相矛盾（注释说 BF16 需要 4+ 节点，交付的却是 `num_nodes: 1`）。疑似未跑过。

### 10.3 历史档案（可选，不是必需）

`~/working/amd-rl-runbook/` 下有 13 份 runbook / handoff（约 8400 行），是这份文档的原始素材。
**本文档 + `agent/` 已经覆盖了它们的 DSv4 部分，正常开发不需要去翻。**

只有两种情况值得回去查：想看某条结论的**完整证据链**（比如某个 bug 的逐步排查过程），
或者要做**非 DSv4** 的工作（Qwen3 / MoE / ATOM 那几条线）。

⚠️ 那批文件在 `.gitignore` 里（`*-handoff.md`），不在版本控制内，**换机器就没了**——
这也是为什么要把结论搬进仓库。两边冲突时**以实测日期新的为准**。

---

## 变更记录

| 日期 | 内容 |
|---|---|
| 2026-08-21 | 建档。汇总 13 份 runbook/handoff 的 DSv4 部分 + 2026-08-20/21 的 primus 迁移 + 训推分离方案调研 |
| 2026-08-21 | 新增 [`agent/`](agent/) 五篇稳定操作手册，把机器操作和环境构建从仓库外搬进来。**本文档不再引用任何仓库外文档**。§2 / §7 / §10 相应精简，避免与 `agent/` 重复。记录开发分支 `dev/dsv4-dapo` |
