# 03 · DeepSeek-V4 的产物：四份权重、三套命名

> **稳定操作手册第三篇**，索引见 [`README.md`](README.md)。
> 前置：[`01-cluster-access.md`](01-cluster-access.md)、[`02-environment-setup.md`](02-environment-setup.md)。
>
> 这一段是整条线里**最固定**的部分——命名规则和转换链路定下来之后再没变过。
> 大部分时间成本也在这里（全 43 层约 1.9 TB / 节点），所以值得先读清楚再动手。

---

## 0. 为什么需要四份

一句话：**训练侧和 rollout 侧要的是同一批数值的两种不同命名，而原始检查点是 FP8 的第三种命名。**

```
①  DeepSeek-V4-Flash-Base    275 G   原生 FP8，检查点自己的命名     ← hf download
        │  fp8_cast_bf16.py（反量化 + 改名，一步做两件事）
        ▼
②  -bf16                     544 G   bf16，DeepSeek-HF 命名         ← 中间产物，不分发
        ├─ convert_hf_to_torch_dist ──▶ ③ _torch_dist    530 G   mcore 命名  ← 训练侧 dc.load
        └─ make_native_bf16.py     ──▶ ④ -bf16-native    530 G   原生命名    ← vLLM rollout
```

⚠️ **反直觉的一点：原生→HF 的改名发生在第 ① 步不是第 ② 步。**
`fp8_cast_bf16.py` 反量化**同时**改名（改名函数就是 SGLang 的
`remap_weight_name_to_dpsk_hf_format`，靠嗅探 `embed.weight` 判断是不是原生格式）。

**每份的存在理由**：

| 产物 | 不能省的原因 |
|---|---|
| ② bf16 | 原生是 FP8 块量化（`weight_block_size [128,128]`、`scale_fmt=ue8m0`），训练侧要 bf16 |
| ③ torch_dist | 绕过 LumenRL 的 HF 加载路径**三个问题**：每 rank 读全量、峰值双倍、**完全没有 FP8 反量化**（裸 `load_file` 拿到 `float8_e4m3fn` 直接 `.to(bfloat16)`，无视 `_scale_inv`，**得到静默错误的权重**）。换来 `dc.load` 的**自动重分片**——转换时的并行度不约束训练时的并行度 |
| ④ bf16-native | vLLM 这条线的**唯一出路**，见 §3。②→④ 是**纯重命名不动数值** |

---

## 1. 三套命名（很容易挑错）

| 命名 | 长相 | 谁用 |
|---|---|---|
| **检查点原生** | `layers.0.attn.wq_a.weight`、`layers.0.attn_norm.weight`、`embed.weight` | **vLLM 的 loader**、SGLang、transformers 5.13.1（自带映射） |
| **DeepSeek-HF** | `model.layers.0.self_attn.wq_a.weight`、`...input_layernorm.weight` | `fp8_cast_bf16` 的输出、transformers 参照 |
| **vLLM 内部** | `model.layers.0.attn.fused_wqa_wkv.weight` | vLLM 自己的模块名，**不是**同步目标 |

**这三套名字造成过的实际故障**：

1. FSDP2 那条路直接死在这：`Worker failed with error ''layers.0.self_attn.sinks''`
2. 送 DeepSeek-HF 名给 `model.load_weights` → `KeyError: 'layers.0.input_layernorm.weight'`；
   rollout 指向 bf16 目录时是同一个 KeyError，连加载都过不去
3. ⚠️ **`probe_64` 第一版拿手写的「vLLM 内部名」当对照，报 101/101 全覆盖——而它是错的**，
   因为对照物本身就是脑补的。**验证映射必须拿检查点自己的 `index.json` 当 oracle。**
4. `mlp.shared_expert.*`（单数，Qwen2-MoE 风格）vs DeepSeek 的 `mlp.shared_experts.*`（复数）：
   bridge 的分支是 `if ... in hf` 保护的，**不报错、静默跳过**——shared expert 完全不加载，
   输出全错但无提示

⚠️ **不要照抄 SGLang 的 `remap_weight_name_to_dpsk_hf_format` 当 HF 映射**：它的目标命名和
transformers 并不一致（保留 `wq_a`/`wkv` 这类论文记号，只做前缀改写）。
**但取其逆回到原生名是对的**——正因为只做前缀改写所以可逆，`make_native_bf16.py` 用的就是这个性质。

---

## 2. 重建：4 层切片

**先用切片，不要一上来就全 43 层。** 切片产物小、迭代快，而且三个静默 bug 当初就是在它上面定位的。

```bash
# ⚠️ /mnt/m2m_nobackup 是节点本地盘，每个节点都要跑一遍
CONTAINER=anp-primus SITE=$PRIMUS_SITE STAGE=1 STOP=2 \
  bash ~/dsv4/mhc_probe/rebuild_l4_artifacts.sh      # 下载 + 切片（不占 GPU）

CONTAINER=anp-primus SITE=$PRIMUS_SITE MILES_IMAGE=<见下> STAGE=3 STOP=5 \
  bash ~/dsv4/mhc_probe/rebuild_l4_artifacts.sh      # bf16 / torch_dist / 参照

# 第六步：改名出 vLLM 要的那份
python3 ~/dsv4/mhc_probe/make_native_bf16.py \
  --src   $M/dsv4-L4-bf16 \
  --dst   $M/dsv4-L4-bf16-native \
  --reference $M/dsv4-native-L4
```

| 阶段 | 产物 | 实测耗时 / 大小 |
|---|---|---|
| 1 下载原生 FP8 | `DeepSeek-V4-Flash-Base` | **8 分 06 秒** / 275 G |
| 2 切 4 层 | `dsv4-native-L4` | 约 1 分钟 / 27 G |
| 3 FP8→bf16 | `dsv4-L4-bf16` | 约 42 秒 / 52 G |
| 4 →torch_dist | `dsv4-L4_torch_dist` | 约 100 秒 / 52 G |
| 5 transformers 参照 logits/grads | `dsv4_L4_ref_*_256.pt` | 约 2 分钟 |
| 6 →bf16-native | `dsv4-L4-bf16-native` | 约 1 分钟 / 52 G |

**判据**：`make_native_bf16.py` 打出 `3176/3176 names match`。

⚠️ 层 0–3 覆盖了每一种结构分支（sliding / sliding / CSA+indexer / HCA，以及 hash_moe 和 gated moe），
这是这个切片存在的理由。

### 2.1 `STAGE=` / `STOP=` 和两个环境变量

- **`STOP=`**：stage 3–4 是唯一需要 miles 镜像的，新节点上可能没有，而 275 G 的下载可以立刻开始。
  两件事要能分开。
- **`SITE=`**：把共享树挂进 `in_lumenrl` 的 `docker exec`。⚠️ **stage 5 少了它必挂**——
  它 `import transformers.models.deepseek_v4`，而那个模块**只在 transformers 5.x 里有**，
  primus 镜像自带的是 4.55.0。stage 1–4 全过、只有 stage 5 报 `ModuleNotFoundError`，
  很容易误判成转换器的问题。

### 2.2 ⚠️ miles 镜像认 tag 不认名字

stage 3/4 要在 miles 镜像里跑（只有它有 fork 的转换器）。**但不是所有 miles 镜像都完整**：

| 镜像 | stage 3 (`tools/fp8_cast_bf16.py`) | stage 4 (`scripts/models/deepseek-v4-flash.sh`) |
|---|---|---|
| `rocm/sgl-dev:miles-rocm720-mi35x-20260730` | ✅ | ✅ |
| 无 tag 的 `rlsys/miles` | ✅ | ❌ **缺这个文件** |

`MILES_IMAGE=` 可以传 image ID，但**先确认这两个文件都在**再动手。

---

## 3. 重建：全 43 层

```bash
CONTAINER=anp-primus SITE=$PRIMUS_SITE MILES_IMAGE=<完整的那个> STAGE=1 STOP=4 \
  bash ~/dsv4/mhc_probe/rebuild_full_l43.sh
```

| 阶段 | 产物 | 大小 |
|---|---|---|
| 1 | `DeepSeek-V4-Flash-Base` | 275 G |
| 2 | `-bf16` | 544 G |
| 3 | `_torch_dist` | 530 G |
| 4 | `-bf16-native` | 530 G |

**合计约 1.9 TB / 节点**，两节点并行实测约 **1 小时**（stage 1 已有的话更快）。
`STAGE=n` 从中间续；`STOP=3` 只产探针要的两份（省 530 G）。

**核对判据**（每个节点都要跑，`/mnt/m2m_nobackup` 是节点本地盘）：

```bash
python3 ~/dsv4/mhc_probe/check_native_index.py
# 期望： tensors=34223 native=34217 hf_style=0 mtp=0 -> OK
```

这一条同时挡住两个静默失败模式：HF 名字混进来（`hf_style` 非 0），以及 MTP 块还是 FP8。

### 3.1 ⚠️ MTP 必须丢掉

`make_native_bf16.py` 在全模型最后一个分片会崩：`AssertionError: unexpected tensor 'mtp.0.e_proj.scale'`。

根因：**`mtp.0.e_proj.weight` 在「bf16」检查点里仍然是 `float8_e4m3`**，旁边挂着 `[32, 32]` 的块 scale——
`fp8_cast_bf16` 不认识 `mtp.0.*` 这套名字，把整块原样留下了。放进已摘掉 `quantization_config`
的目录，等于给 bf16 层喂 FP8 张量。

修法是丢弃 MTP 并摘掉 `num_nextn_predict_layers` 键。**丢掉没有代价**：vLLM 里 MTP 是独立模型
`DeepSeekV4MTP`，只在推测解码时构建；训练侧 torch_dist 是 43 层、本来就没有 MTP。

### 3.2 ⚠️ `config.json` 里的 `quantization_config` 必须摘掉

否则 vLLM 仍按 FP8 建层。这是 `make_native_bf16.py` 顺带做的。

---

## 4. 启动长任务的写法

产物重建是典型的「几十分钟到一小时」任务，**用 [`01-cluster-access.md`](01-cluster-access.md) §11 那个写法**：
本地 `setsid nohup` 脱离 + 远端保持前台。

⚠️ 三个节点的重建曾经**全部静默失败、日志 0 字节**，原因就是用了
`srun` + 远端 `setsid nohup &`（被 job step 回收）。而且那还是脚本自己用法注释里推荐的写法。

⚠️ 判存活**不要用 `pgrep`**——它恒返回 0，会诱使你起第二个实例。真的因此让两个
`hf download` 同时往一个目录里写（一个已跑 342 秒、一个 170 秒，目录里 16 个 `.incomplete`）。
补救是安全的：`hf download` 会校验续传，**全停 + 单一下载器续跑**即可。

⚠️ 日志重定向的目标目录**要先建**——三个节点第一次额外失败就是因为
`/mnt/m2m_nobackup/<user>/` 不存在，**重定向先于脚本失败**。

---

## 5. 磁盘账

| 位置 | 性质 | 放什么 |
|---|---|---|
| `/home/<user>` | **NFS，跨节点共享** | 代码、共享 python 树、日志、模型（小的）、过滤好的数据 |
| `/mnt/m2m_nobackup` | **节点本地 NVMe，28 T/台，换节点就没** | DSv4 的所有产物、docker 镜像层、core dump、probe_scratch |

⚠️ **作业结束不清盘**，所以拿到新分配**先逐台盘点**，命中哪台就省哪台的重建。

⚠️ **探针的 cwd 要放本地盘**：GPU memory fault 会往进程 cwd 写 **50–80 GB** 的 core dump。
先 `cd /mnt/m2m_nobackup/<user>/probe_scratch`。

⚠️ **`/home` 会满**。实测见过 97%（只剩 319 G）。开工前 `df -h /home`。

---

## 6. 模型与数据（NFS 上，换节点不用重下）

```
$DATA_ROOT/models/{Qwen3-8B-Base, Qwen3-30B-A3B-Base}
$DATA_ROOT/data_cached/qwen3-8b-maxprompt1024/{dapo-math-17k,aime-2024}.filtered.parquet
```

⚠️ **RL 必须用 Base 版**。instruct/thinking 版在 `max_response_length` 内永远不闭合，
reward 恒 −1，`filter_groups` 连续 10 轮 kept 0 直接
`RuntimeError: filter_groups collected no valid groups`。

⚠️ 那份过滤好的 parquet **需要 `datasets>=4`** 才读得了（用了 `List` 特征类型），
镜像自带的 3.6.0 会报 `ValueError: Feature type 'List' not found`。见
[`02-environment-setup.md`](02-environment-setup.md) §2.4。

⚠️ DSv4 的 vocab 是 **129280**，`qwen3-8b-maxprompt1024/` 那套过滤阈值对它不再准确——
真要在 DSv4 上跑正式训练需要重新过滤。
