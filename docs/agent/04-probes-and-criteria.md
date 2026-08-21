# 04 · 探针目录与判据

> **稳定操作手册第四篇**，索引见 [`README.md`](README.md)。
>
> 探针比端到端便宜一到两个数量级，而且能覆盖端到端覆盖不到的段。这份记的是
> **每个探针验什么、判据是什么、以及哪些判据是错的**。
>
> ⚠️ 最后一条最重要：这条线上**被推翻的判据比被推翻的实现还多**，§4 单独列。

---

## 1. 探针目录

### 1.1 环境层（换镜像 / 换节点必跑）

| 探针 | 验什么 | 判据 |
|---|---|---|
| `scripts/primus/verify_vllm_primus.py` | 5 个编译扩展 + **算子真的注册** + DSv4 在 registry + RCCL 仍 2.28.9 | `VERDICT: vllm usable on primus` |
| `$RL_ROOT/megatron_verify.py` | TE / Apex / megatron-core / 双 engine 注册 | `VERDICT: ALL OK` |
| `$RL_ROOT/probe_ray_gpu_auto.py` | Ray 的 GPU 隔离 | **`8/8 distinct physical GPUs`** |
| `scripts/primus/run_a2a_anp.sh` | RDMA 通不通 | 见 [`02-environment-setup.md`](02-environment-setup.md) §5 |
| `scripts/primus/probes/probe_devcount.py` | torchrun 下每个 rank 看到几张卡 | 每个 rank 都是 8 |
| `scripts/primus/probes/probe_rs_coalesced.py` | 合并 reduce-scatter 活不活 | 8 rank 全部打出 `coalesced reduce_scatter x4: OK` |
| `scripts/primus/probes/probe_ipc_primus.py` | 跨进程 CUDA IPC handle 能不能开 | `VERDICT: IPC works` |
| `~/4node/check_image_for_rdma.sh <image>` | 换镜像先跑：OS/glibc/py/**RCCL 版本**/ANP dlopen | RCCL 必须是 2.28.9 |

### 1.2 DSv4 模型层

| 探针 | 验什么 | 判据 | 要什么产物 |
|---|---|---|---|
| `probe_63`（`run_probes.sh`） | 训练侧前向/反向 vs transformers | mean `\|Δlogprob\|` 在带内、8 rank 分歧 **0.000e+00**、fp32 mHC **27/27** | 切片 |
| `probe_65` | offload 下 fp32 参数有没有更新 | **27/27**（切片）/ **261/261**（全模型），**MIN all-reduce 取 worst rank** | 切片或全模型 |
| **`probe_67`** | vLLM 能不能读 `-bf16-native` | 加载成功 + exit 0 + **记运行期峰值显存**。⚠️ 全 43 层上才看「三题答对」 | `-bf16-native` |
| **`probe_68`**（`run_probe_68_full43.sh`） | 权重同步**发送侧**的名字和形状 | **`0 missing / 0 extra / 0 SHAPE MISMATCH`** + `WEIGHT SYNC SHAPES: PASS` | `-bf16` + `_torch_dist` + `-bf16-native` |
| `probe_69` | 拓扑正确性（PP / CP） | `max \|Δ log-prob\|` vs **同 EP 同 REPEAT** 的基线逐位为 0 | 切片 |
| **`probe_70`** | 前向逐层一致性 | 第一个发散层 `ulp@max` 约 1–2、`n_diff` 约 6e-05 → **ROUNDING** | `-bf16` + `_torch_dist` |
| `probe_66` | 全模型一步 DAPO（无 rollout） | `loss` / `grad_norm` / `261/261` / 困惑度 **2.88** | 全 43 层 |

**在 primus 上跑 DSv4 探针要传两个开关**（脚本已改成可覆盖，默认仍是 22.04 那套）：

```bash
docker exec \
  -e RAY_ENV=<repo>/scripts/primus/ray_env_dsv4_primus.sh \
  -e NAME=dsv4-L4 \
  anp-primus bash ~/dsv4/mhc_probe/run_probe_68_full43.sh
```

- `RAY_ENV` **必须换成 primus 版**，否则会带上 `HSA_DISABLE_FRAGMENT_ALLOCATOR=1`
- `NAME` 切片和全模型共用同一套后缀，所以 `NAME=dsv4-L4` 就能选到那三份切片产物；
  全 43 层用默认值
- `probe_67` 用的是 `DSV4_PATH=` 而不是 `NAME=`

### 1.3 探针覆盖不到的那一段

⚠️ **`probe_68` 只验发送侧。** 它走的正是 `update_weights_ipc_send` 调的
`get_per_tensor_param()`，而 `build_optimizer: False` 让它不需要那 426 GB/rank 的优化器——
这正是它能在单节点上跑全模型的原因。

**探针碰不到的**：`BucketedWeightSender` 的 ZMQ / CUDA-IPC 传输，和 vLLM 接收端的 in-place load。
那两段**只能靠端到端**。

⚠️ 所以两种说法都是错的：
- ❌「流式 gather 要等 4 节点才能验」——**单节点就够**，别再把它列进多节点待办
- ❌「`_sync_weights_ipc` 已经验过了」——**验的只是发送侧**

---

## 2. 正确性判据

| 判据 | 阈值 | 已知实测值 |
|---|---|---|
| 前向 argmax 一致率 vs transformers | **「不差于 91.4%」**——那是 vLLM 和 transformers 这两个都被公认正确的实现彼此的一致率。⚠️ **不是 99%** | 96.09%（切片 seq 256） |
| mean `\|Δ logprob\|` | 同基线 0.217 | 0.049–0.053 |
| 反向梯度余弦 | **余弦是判别统计量**（两边 FP8 反量化路径不同，模长会漂 0.957–1.046，但**算错了方向会塌**） | 31 个参数最差 **0.99675** |
| `dc.load` | `missing=0 / unexpected=0`（忽略 `_extra_state`） | 达成 |
| 权重同步形状 | `0 missing / 0 extra / 0 SHAPE MISMATCH` + `PASS` | 切片 3176、全模型 34223 |
| `bf16-native` 索引 | `tensors=34223 native=34217 hf_style=0 mtp=0 -> OK` | 达成 |
| **`rollout_corr/kl`** | **权重同步内容正确性的判据**（形状对了不代表内容对——gate/up 顺序错了 kl 会差几个数量级）。切片基线 **2.6–3.3e-3** | 见 [`../dsv4_agent.md`](dsv4_agent.md) §3.4。⚠️ **43 层从来没有实测值** |
| `is_weight_mean` | ≈ 0.9999 | 达成 |
| fp32 mHC 参数更新 | **全数**（offload 静默 bug 的探测器） | 27/27、261/261 |
| 全模型唯一的正确性判据 | 真实英文散文困惑度（均匀分布是 129280） | **2.88** |
| 确定性 | `--deterministic-mode` 下 bitwise identical、argmax 翻转 0% | 达成（不开是 1.56%） |
| 流式 gather 生效 | `weight_sync gather` 的 peak **只比进入时高几个 GiB** | 切片 65.8→66.9 GiB；全模型 529.7 GiB 流过、峰值 **+0.6 GiB** |

### 2.1 `probe_70` 的 `ulp@max` + `n_diff`（前向一致性的正确判据）

在**第一个发散的层**上，看最大分歧是它**自身元素数值**的几个 bf16 ulp：
`|d| / (max(|a|,|b|) * 2^-8)`，在 `argmax|d|` 处求值。再配 `n_diff`（多少元素不同）做旁证。

```
ulp@max ~ 1，n_diff 极小              -> 舍入。专家 all-to-all 的求和顺序逐 rank 不同，
                                         网络再放大它。已知、已量化，不是 bug。
ulp@max >> 1，或 n_diff 是 hidden      -> 算错了而不是舍入错了：选错专家、重分片错位、
宽度的整数倍                             丢 token。这才是探针要抓的东西。
```

**`n_diff` 必须一起看**：选错一个专家会让那个 token **整行 4096 个元素同时变**，
所以「有多少元素不同」和「差多大」是两个**独立**信号。

阈值怎么定的：两个差异很大的长度上测出来几乎一样，说明它测的是一个**稳定的物理量**而非噪声。

| seq | 第一个发散层 | ulp@max | n_diff |
|---|---|---|---|
| 1280 | 18 | 2.68 | 6.22e-05 |
| 6144 | **3** | 1.88 | 6.59e-05 |
| 切片（primus 上重跑） | 0 of 4 | 3.58 | 1.486e-04 |

默认 `--ulp-tol 10`，余量充足。

### 2.2 环境未漂的判据

4 层切片跑 `run_probes.sh all`（约 3 分钟/节点），这一串数字就是「环境没变」：

```
276 字段 / 27.39 B / KL 1.485e-3 / argmax 96.09% / 41-41 pattern
loss −0.04613 / grad_norm 16.749379 / 70.3 GiB
```

---

## 3. 显存判据（MI350X，288 GB/卡）

| 项 | 数字 |
|---|---|
| 全模型 gather（修前） | **529 GiB/rank** —— 永远过不去 |
| 全 43 层一步峰值 | **189.6 GiB**（offload 0）/ allocated 117.6 + **reserved 144.7**（offload 0.75） |
| 全模型 rollout | FP8 **36 GB/卡** vs bf16 **71 GB/卡**（TP=8） |
| 4 层切片一步峰值 | TP=1 **70.3** → TP=2 63.8 → PP=2 61.2 GiB |
| EP=8 下优化器 | `284B × 12 / 8 = 426 GB/rank` —— **单节点装不下的就是这个** |
| EP=8 下权重同步 | 只要权重 **83 GB**（实测 init 78.2 GiB） |

⚠️ **量显存的三条纪律**：

1. **要看 `reserved` 不是 `allocated`**——前者才是 rocm-smi 和 colocated 的 vLLM 争的东西。
   init 时 reserved 就比 allocated 高 **26 GiB**。
2. **不要在短序列上量然后外推**——旧文档那个「余 98 GB」是在 **约 130 token** 上量的，1280 就撑满。
3. **不要拿常驻显存判断 offload 有没有生效**——优化器的 fp32 master 和 Adam 动量是
   **第一次 `step()` 才落地**的，init 后两者只差 10 GiB。
4. `rocm-smi` **要在运行期每 5 秒采样、事后取每卡最大**。放在 python 之后恒读 0。
   ⚠️ 也别指望 vLLM 自己的 memory-profiling 日志——这套环境把 vLLM 跑在 WARNING 上。

---

## 4. ⚠️ 这些判据是错的，不要用

**这一节比上面两节加起来更省时间。**

| 度量 | 为什么不能用 |
|---|---|
| **`max \|Δ log-prob\|` 当距离用** | **它是混沌量**。`moe_router_dtype=fp32` 让隐状态分歧减半（512→300）却让它涨了近三倍（0.354→0.955）——它取决于恰好哪个 token 的预测被翻转。不同轮次的 9.108e-01 / 3.539e-01 / 9.546e-01 **彼此不可比**。稳定的量是「不同位置的比例」（两组都约 80%） |
| **「32 rank 逐位相同」当正确性门槛** | 43 层 bf16 MoE **做不到**，≤1152 token 时能过是侥幸。「留着它只会让探针在每个真实长度上都是红的——**那比没有判据更糟，因为它训练读者忽略判据**」 |
| `max\|d\| / max\|value\|` | 分歧元素通常远小于张量峰值，**无论什么原因都读作「不到一个 ulp」**（层 18 是 0.02），分不开两种情况 |
| 逐元素相对差取全体最大 | **被接近零的元素主导**（两个 rank 一个 +1e-8 一个 −1e-8 就得到恰好 2.0），实测层 19 之后每层都报 2.000，毫无分辨力 |
| **loss 当拓扑判据** | CP=2 错到 **509/511 个位置**，loss 仍然逐位相同（被 advantage 主导） |
| 整模型有限差分验梯度 | 每层有两处离散选择（MoE top-k、DSA top-k），loss 只是**分段光滑**。实测 analytic 6.1e-4 对 numeric 6.8e-2，**差 110 倍** |
| argmax 一致率定 99% | 两个公认正确的实现彼此才 91.4% |
| `max_memory_allocated` 做显存规划 | 见 §3 |
| vLLM 的 `prompt_logprobs` 当金标准 | 实测在这个模型上有位置给出完全不合理的分布。**唯一一处实质分歧，错的是 vLLM** |
| bf16 输出直接对拍 | 1 ULP = 3.9e-3，没有意义。要用 fp64 真值当参照 |

### 4.1 探针本身的判据陷阱

| 陷阱 | 正解 |
|---|---|
| `run_probe_full43.sh` 的**退出码** | ❌ **恒为 0**（最后一行是 `echo`，没开 `set -e`，torchrun 无论成败都返回 0）。曾把一次全 rank SIGABRT 报成成功。**看 `MILESTONE` / `FORWARD:` / `STEP:`** |
| `run_probe_68_full43.sh` 的退出码 | ✅ 这个是真的（传播 torchrun 的） |
| 只看 rank 0 判断参数有没有更新 | ❌ probe_65 原先只在 rank 0 打印，别的 rank 出问题只表现为一个非零退出码。改成 **MIN all-reduce 报 worst rank** |
| PP 下 rank 0 打印的 `loss` 是 `nan` | ⚠️ **不是缺陷**——`_pp_update_policy` 对非最后 stage 故意不返回 `loss` 键 |
| 拿 `probe_63` 通过就认为拓扑可用 | ❌ **它不同步权重**。TP=2 在 probe_63 上全 PASS，端到端却死在权重同步。**换拓扑先跑 `syncshapes`** |
| 拿训练侧探针通过就撤 `_dsv4_check_topology` 的守卫 | ❌ 真踩过：PP 在 probe_69 上逐位通过后撤了守卫，端到端**立刻死在权重同步**，修完三个 bug 才重新撤 |
| 在没确认参数真的生效前相信 A/B | ❌ `moe_router_dtype` 曾在 DSv4 路径上被**静默丢弃**，害得做了一次无效实验。**让探针回显模型实际构建时的值** |
| 比较两个拓扑时改动一个以上的并行度 | ❌ `world=8` 固定，`PP=2` 必然把 `EP` 从 8 压到 4。曾拿 EP=8 基线比 PP=2 EP=4，得到看似 PP 有问题的 `grad_norm |d| = 2.348e-03`，补跑 PP=1 EP=4 基线后**全部来自 EP 变化** |

### 4.2 两条已经查清、不要再查的

- **「层 18 有什么特别」** —— 已退休。seq 6144 下第一个发散层是 **3**。它只是「该长度下第一次
  有元素跨过舍入边界」的位置。也不要去找「1152 和 1280 之间的特殊常数」——**那个门槛是运气不是机制**。
- **32 rank 前向不一致不是 bug** —— 是**一个 bf16 ulp 被 43 层放大**：层 0–17 逐位相同，
  层 18 首次出现 3.125e-02（正好一个 ulp），之后每层约 1.35×，层 42 到 512。
  ⚠️ 放大比想象剧烈：`n_diff` 在**一层之内**从 6e-05 跳到 0.235。

### 4.3 切片特有的「假故障」

| 现象 | 真相 |
|---|---|
| `probe_67` 输出**乱码** | ✅ **正确的**。切片把 layer-3 的残差直接喂给 `model.norm`/`lm_head`。「三题答对」是**全 43 层**的判据 |
| `reward/accuracy = 0` | ✅ 预期。配置里因此关掉了 `filter_groups`。⚠️ **上全模型时要打开** |
| 某一步 `grad_norm=0` / `loss=0` | ✅ 可能是对的——那一步所有采样都顶到长度上限、拿到相同的 overlong 惩罚，组内 reward 全等 ⇒ GRPO 的 advantage 恒 0 |
| 切片没有困惑度判据 | ✅ 是这样。全模型的「困惑度 2.88」正是切片给不了的那个 |
