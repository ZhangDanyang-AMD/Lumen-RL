# SimWAM NAVSIM FlowGRPO 在 4× MI350X 上的整网优化

> 性能、精度与复现报告。English version: [README_PERF_EN.md](README_PERF_EN.md)。开关的完整说明见 [README.md](README.md)。

本文汇总 `examples/Embodied_SimWAM_NAVSIM` 的 FlowGRPO 训练在 4 张 AMD Instinct MI350X（gfx950）上的整网优化结果：
和比特一致基线比快多少、每项优化的来源和实测收益、精度是否保持，以及如何复现和切回比特一致版本。
SimWAM、DeepSpeed、Lumen 都没有修改，所有改动都是本目录 `plugin.py` 里带环境变量开关的运行时 patch。

---

## 1. 执行摘要

### 1.1 结论

> **20000 步整网训练从 87.8 分钟降到 43.6 分钟，平均每个周期 1054.1 → 522.9 ms（2.02×，−50%）；
> 全集 navtest PDMS 91.11 对比特一致基线的 91.13，逐 token 配对差 −0.02（95% CI −0.15 到 +0.11，p = 0.75），
> 平均分没有差别；rollout 和训练的 log-prob 在全部 5000 个周期 × 4 个 rank 中逐位相同。**
> 已知差异：gate 翻转偏向变差，主要在 TTC（见 3.3）。

| 层级 | 结论 | 证据 |
|---|---|---|
| 整网速度 | **快 2.02×** | 20000 步平均周期 1054.1 → 522.9 ms；总时长 87.8 → 43.6 min（同卡、同 seed、同配置） |
| GPU 阶段 | **快 2.78×** | 各阶段中位数之和 816.1 → 293.2 ms |
| 精度 | **平均分无差别，gate 翻转偏 TTC** | 全集 PDMS 91.11 对 91.13（配对 −0.02，p = 0.75）；217 个 token 翻转 gate，79 好 / 124 坏（p = 0.002），TTC 46 / 63 |
| 训推一致 | **逐位相同** | \|Δlogp\| = 0（基线均值 1.16e-3、最大 0.032、4.6% 超出 PPO clip） |
| 显存 | **下降** | 每卡已分配峰值 35.62 → 34.77 GiB，整卡占用 53.36 → 42.90 GiB |

| 项目 | 比特一致基线 BX20 | 优化版 FIN20 | 变化 |
|---|---:|---:|---:|
| 代码 | #51 `f3f3ca5`，`LUMEN_ENABLE=hf_attn_patch` | `hz/grpo-fast` `daf60b17`，全部默认值 | |
| 平均周期（全部 5000 个周期，含 eval、可视化） | 1054.1 ms | **522.9 ms** | **−50.4%（2.02×）** |
| 第 1 个周期到最后一个周期（20000 步） | 87.8 min | **43.6 min** | **−50.3%** |
| 周期中位数（全部周期，rank 0） | 814.1 ms | 398.9 ms | −51.0%（2.04×） |
| 非 eval 周期 10 / 90 分位（rank 0） | 804.1 / 912.8 ms | 293.4 / 664.9 ms | |
| GPU 阶段中位数之和 | 816.1 ms | 293.2 ms | −64.1%（2.78×） |
| 全集 navtest PDMS（12146 token） | 91.13 | 91.11 | −0.02（p = 0.75） |
| \|Δlogp\|（rollout 对训练） | 均值 1.16e-3，最大 0.032 | 0 | |

![cycle time](https://raw.githubusercontent.com/huizzhan/Lumen-RL/dac70f5d6fa9cc037709479c7212460dd87b3500/assets/grpo-fin20-vs-bx20-cycle-time.png)

### 1.2 为什么 GPU 阶段快 2.78×，整网只快 2.02×

每个周期 = 1 次 rollout（VAE、video prefill、10 步去噪、old / ref log-prob、PDM 打分）+ 4 次 optimizer step。
GPU 阶段的中位数从 816.1 降到 293.2 ms，但周期里还有 GPU 之外的时间：

- **PDM 打分的长尾**：奖励由 CPU 上的 PDM 闭环仿真算出（每个 rank 8 个 worker 进程），个别 token 的 metric cache 加载或仿真要几百毫秒，
  这部分和 GPU 优化无关。FIN20 非 eval 周期的 90 分位是 664.9 ms，中位数 387.9 ms，差距主要就是这段等待。
- **每 100 步一次的内置 eval 和可视化**：两边都有，计入平均周期。
- 所以本文以 20000 步的**平均周期和总时长**为主口径（包含了上面两项），中位数和 GPU 阶段只作补充。

![phases](https://raw.githubusercontent.com/huizzhan/Lumen-RL/dac70f5d6fa9cc037709479c7212460dd87b3500/assets/grpo-fin20-vs-bx20-phases.png)

| 阶段（中位数，ms，rank 0，第 16 步起、去掉 eval / 可视化周期） | BX20 | FIN20 | 主要来源 |
|---|---:|---:|---|
| VAE encode | 21.5 | 20.5 | FlyDSL 卷积（单次 encode −22%，整网看不出） |
| condition（video prefill） | 73.2 | 31.7 | prefill CUDA graph、去 mask、CK RMSNorm |
| 去噪（10 步） | 173.8 | 61.6 | XATTN_KV、LORA_CAT、融合 kernel、KV slot |
| rollout log-prob（policy / ref） | 53.7 / 41.4 | 12.9 / 9.4 | no-grad 合批、合批 GEMM、KV slot |
| 训练前向 ×4 | 189.8 | 63.6 | 合批 GEMM、融合 kernel、训练 graph 的 KV slot |
| 训练反向 ×4 | 248.8 | 78.8 | Triton RMSNorm 反向、aiter attention 反向、梯度直写 flat 缓冲 |
| optimizer ×4 | 13.9 | 14.7 | |

### 1.3 和 DeepSpeed 基线的对比（补充）

原始示例用 DeepSpeed ZeRO-1 训练。#51 第三轮在同一组 4 卡上测过比特一致的 DeepSpeed 版本（`f27ab82`，2026-11-18，GPU 4–7，
`LUMEN_ENABLE=hf_attn_patch`，每次 60 步，rank 0 取第 16 步起 11 个周期的中位数，两次）：1328 / 1305 ms，平均 1316.5 ms。

| | DeepSpeed ZeRO-1（`f27ab82`） | FIN20（`daf60b17`） | 比值 |
|---|---:|---:|---:|
| 周期中位数 | 1316.5 ms | 398.9 ms（全部周期）/ 387.9 ms（非 eval 周期） | 3.30× / 3.39× |

口径限制：DeepSpeed 的数字来自 60 步的短运行，FIN20 来自 20000 步；两者不在同一组运行里交替测量，DeepSpeed 没有 20000 步的平均周期和总时长。
所以这一行只作量级参考，主口径仍是 1.1 节和 BX20 的同口径对比。原始示例 `d0ada15`（DeepSpeed，`hf_attn_patch,lumen_linear`）
同期只测到一次 6076 ms（第二次被外部进程打断）。

### 1.4 报告边界

- 速度和精度都只有一个 seed（42）、一次 20000 步运行；不同 seed 的 4000 步子集 PDMS 波动约 ±0.15（3.2 节）。
- FIN20 跑的是 `daf60b17`。`62acd4c5` 在它之上又加了两个默认打开的开关（`SIMWAM_GRPO_EXPAND_VIEW`、`SIMWAM_GRPO_KV_SLOT_STEP_SHARED`），
  两者都和 `daf60b17` **逐位一致**（2 卡交替 3 轮，9 个 run 的 loss 逐位相同；全默认值 24 步 graph 逐次校验 0 次不一致），
  2 卡上 GPU 阶段每周期再快 1.8–6.0 ms、每卡显存已分配峰值再少 1.45 GB；没有单独跑 20000 步。
- 计时来自一个只在测量时加载的 hook（在阶段边界做 `torch.cuda.synchronize()`，约 1% 开销），不在本仓库里；基线和优化版都带着它。

---

## 2. 测试设计

### 2.1 固定环境

| 项 | 配置 |
|---|---|
| GPU | 4× AMD Instinct MI350X（gfx950），cv350 GPU 4–7，`NPROC=4`，每卡一个进程 |
| 容器 | `zhangdanyangamd/lumen-rl:embodied-MI350X`（torch `2.12.0+rocm7.14.1`，镜像自带 aiter） |
| 依赖 | SimWAM `1433ee7`（未修改）；Lumen `293629f`（`dev/world-model-rl`）；aiter checkout `huizzhan/aiter:hz/grpo-opt` `c9be3a5`（`AITER_SRC`，含 conv3d 和本例的 GEMM 调优表） |
| 任务 | `navsim_grpo_action_pdm_384x672_flowgrpo_lora`，scene filter `configs/navtrain_pdm_score_below0p9_complete.yaml`（2784 个 navtrain 场景），warm start `SimWAM.pt` |
| 关键超参 | `max_steps=20000`，每 rank `batch_size=1`，`group_size=8`，`num_inner_epochs=4`，lr 5e-5（100 步 warmup），LoRA r=16 / alpha=32，seed 42，每 100 步内置 eval 和可视化 |

### 2.2 对比的两次运行

| run | 代码 | `LUMEN_ENABLE` | 说明 |
|---|---|---|---|
| BX20 | #51 `f3f3ca5` | `hf_attn_patch` | 比特一致基线：4 卡 8 步对原始示例 `d0ada15` 逐位相同，前 4000 步和另一次运行 C51b 逐位相同 |
| FIN20 | `hz/grpo-fast` `daf60b17` | 脚本默认 `hf_attn_patch,aiter_conv,flydsl_conv` | 全部开关默认值，不含新的 GEMM 调优表 |

两次运行在同一组 GPU（4–7）、同一容器、同一 seed 和配置下各跑 20000 步，期间没有外部进程占卡（每秒采样）。

### 2.3 测量口径

- **平均周期**：全部 5000 个周期的周期间隔平均值，包含 eval、可视化和 PDM 等待（rank 0 按“第一个周期开始到最后一个周期结束 ÷ 5000”
  重算得到 523.3 / 1054.5 ms，差别不到 0.1%）；**总时长**是第一个周期开始到最后一个周期结束（不含容器启动、模型加载）。
- **中位数、分位数和阶段**：rank 0 每个周期的计时；阶段取第 16 步起、去掉 eval / 可视化周期的中位数。
- **精度**：第 20000 步 LoRA 按 `SimWAMGRPO.save_checkpoint` 的方式合并进 `SimWAM.pt`，在全集 navtest（12146 token）上用 NAVSIM v1 PDMS 评分；
  逐 token 配对比较（bootstrap 95% CI、配对 t 检验、gate 翻转的符号检验）。
- **训推一致**：每个周期第一次 inner update 时（权重还是产生这个 rollout 的那份），训练前向的 policy log-prob 和 rollout 的 `old_logp` 逐个比较
  （每 rank 每周期 24 个）。
- **单项优化的收益**是在各自加入时测的（2 卡或 4 卡、40 或 60 步、同卡交替），彼此不能直接相加，见第 4 节。

---

## 3. 实测结果

### 3.1 速度

见 1.1 节的表和两张图。补充：

- FIN20 的 4 个 rank 周期中位数相差不到 1 ms；BX20 同样。
- 两次运行都没有让卡、没有中断。

### 3.2 精度：全集 navtest PDMS（第 20000 步）

| 指标 | FIN20（优化版） | BX20（比特一致基线） | FIN20 − BX20 | `SimWAM.pt`（RL 之前，bf16 benchmark） |
|---|---:|---:|---:|---:|
| **PDMS** | **91.11** | **91.13** | **−0.02**（95% CI −0.15 到 +0.11，p = 0.75） | 90.27 |
| NC（无责碰撞） | 98.36 | 98.36 | 0.00 | 98.65 |
| DAC（可行驶区域） | 98.65 | 98.64 | +0.01 | 98.04 |
| EP（前进进度） | 85.80 | 85.75 | +0.05 | 84.02 |
| TTC（碰撞时间） | 95.28 | 95.42 | −0.14 | 95.78 |
| C（舒适度） | 99.99 | 99.98 | +0.01 | 99.99 |
| DDC（行驶方向，v1 权重 0） | 98.25 | 98.17 | +0.08 | 98.51 |

- 平均分没有差别；两边相对 RL 之前的 `SimWAM.pt` 都提高约 0.85。
- 4000 步阶段验证（FIN4，`1869ff4a`，navtest 前一半 6074 token）：91.15，对比特一致基线 C51b 的 91.07 配对 +0.07（CI −0.08 到 +0.23）。
  同一配方不同代码版本的 4000 步单 seed 结果在 91.01–91.26 之间，这是单 seed 训练本身的波动范围。

### 3.3 已知差异：gate 翻转偏向变差，主要在 TTC

FIN20 和 BX20 之间有 217 个 token 至少有一个 gate 的结果不同，其中 FIN20 更好 79 个、更差 124 个（精确符号检验 p = 0.002），
主要来自 TTC（FIN20 更好 / 更差 46 / 63）。平均 PDMS 没有差别（−0.02），因为这些 token 的损失被其余 token 上的 EP 等小幅提高抵消。

可以确定的：这是 FIN20 这一条训练轨迹相对 BX20 的真实差异，不是评分噪声。还不能确定的：它是某个改变数值的开关的系统性影响，
还是单 seed RL 轨迹的差异。之前的 4000 步消融里出现过同类现象：`bc8cd91e` 的运行 TTC 偏低，关掉 LORA_CAT 或 FUSED_MODULATE 中任何一个都会消失，
更像是那一条轨迹的特征。要区分，需要多个 seed 的重复运行，本文没有做。

### 3.4 训推一致性

| run | \|Δlogp\| 均值 | \|Δlogp\| 最大 | 超出 PPO clip（0.005） |
|---|---:|---:|---:|
| BX20 | 1.16e-3 | 0.0324 | 4.60% |
| FIN20 | **0** | **0** | **0%** |

基线的差异来自原始示例本身：rollout 的 attention 走 Lumen Triton（`hf_attn_patch`），训练前向回落到 torch SDPA。
优化版让 rollout 和训练走同一条 attention 和 GEMM 路径（去 mask、no-grad 合批、合批 GEMM 等），重要性比率在第一次更新前正好等于 1。

### 3.5 显存（每卡峰值）

| 指标 | BX20 | FIN20 | `62acd4c5` 再变化（2 卡实测） |
|---|---:|---:|---:|
| `max_memory_allocated`（rank 0 / 其余） | 35.62 / 35.45 GiB | 34.77 / 34.46 GiB | −1.45 GB |
| `max_memory_reserved` | 44.39 / 44.34 GiB | 39.61 / 39.54 GiB | −2.2 GB |
| 整卡占用（`mem_get_info`，最大） | 53.36 GiB | 42.90 GiB | 未同口径测量 |

峰值都出现在内置 eval。MI350X 每卡 288 GB，两边都用不到 20%。

---

## 4. 每项优化

按加入的先后列出。“逐位一致”指和加入前的代码逐位相同；“训推一致”指 rollout 和训练的 log-prob 逐位相同（|Δlogp| = 0）。
收益是加入时的实测，测量条件写在各段开头；各项不能直接相加。

### 4.1 #51 已有的比特一致部分（`d0ada15` → `f3f3ca5`）

4 卡、60 步、rank 0 中位数：原始示例 6076 ms（一次）→ `f3f3ca5` 812.5 ms（两次平均）。这些改动都**逐位一致**，也是 BX20 的代码。

| 开关（默认） | 作用 | 实测 |
|---|---|---|
| `SIMWAM_GRPO_REF_CACHE`（1） | reference log-prob 每个 rollout 只算一次 | 第一轮合计（BX，4 卡）：6076 → 1389.5 ms |
| `SIMWAM_GRPO_CUDA_GRAPH`（1） | no-grad `action_velocity`（去噪、old / ref log-prob）用 CUDA graph replay | 同上 |
| `SIMWAM_GRPO_BATCHED_LOGP`（1） | 3 个随机步的带梯度 log-prob 合成一次调用，GEMM 仍按步拆分 | 同上 |
| `SIMWAM_GRPO_LEAN_ATTN`（1） | Lumen attention 的 no-grad 调用可被 graph 捕获、一次 launch | 同上 |
| `SIMWAM_GRPO_PDM_WORKERS`（8） | PDM 奖励在进程池里并行打分 | 同上 |
| `SIMWAM_GRPO_OVERLAP_REWARD`、`SIMWAM_GRPO_FUSED_LORA` | PDM 打分和 rollout log-prob 重叠；LoRA 输出一次 `torch.add` | 第二轮合计（BX2，4 卡）：−9.0% |
| `SIMWAM_GRPO_MANUAL_DP`（1）、`SIMWAM_GRPO_TRAIN_GRAPH`（1） | 不用 DeepSpeed 的数据并行（fp32 master AdamW，和 ZeRO-1 逐位一致）+ 训练前向反向 CUDA graph | 4 卡：DeepSpeed 1316.5 → 812.5 ms（−38.3%） |

### 4.2 attention 和 GEMM 路径（`c040e02`，4 卡、60 步、逐项叠加，两次交替）

| 开关（默认） | 作用 | 实测 | 数值 |
|---|---|---|---|
| `SIMWAM_GRPO_DROP_MASKS`（1） | mask 全为 True 时不传给 SDPA，rollout 和训练都走 torch flash（底层 aiter CK） | 810 → 748 ms（−7.7%） | 不逐位，训推一致 |
| `SIMWAM_GRPO_BATCHED_GEMM`（1） | 合批 log-prob 里每个 GEMM 对所有步算一次 | 748 → 625.5 ms（−16.4%） | 不逐位，单开时训推不一致 |
| `SIMWAM_GRPO_BATCHED_LOGP_NOGRAD`（1） | rollout 的 old / ref log-prob 也合批，和训练走同一组 kernel | 625.5 → 606 ms（−3.1%） | 不逐位，和上一项一起训推一致 |
| `SIMWAM_GRPO_AITER_RMSNORM`（1） | RMSNorm 换 aiter CK `rms_norm` | 606 → 563.5 ms（−7.0%） | 不逐位，训推一致 |

### 4.3 kernel 融合和 aiter 替换（2 卡，40 步，同卡交替，按 GPU 阶段或周期中位数）

| 开关（默认） | 作用 | 实测 | 数值 |
|---|---|---|---|
| `SIMWAM_GRPO_TRITON_RMSNORM_BWD`（1） | aiter RMSNorm 的反向用一个 Triton kernel | 周期 −12.2% | 不逐位，训推一致 |
| `SIMWAM_GRPO_TRITON_ROPE`（1） | RoPE 前向 / 反向用 fp32 Triton kernel 代替 float64 复数运算 | −4.3% | 不逐位，训推一致 |
| `SIMWAM_GRPO_PREFILL_GRAPH`（1） | video prefill 用 CUDA graph | −4.2% | 不逐位，训推一致 |
| `SIMWAM_GRPO_AITER_ATTN_BWD`（1） | 带梯度 attention 的反向换 aiter（确定性模式），前向不变 | −8.5% | 不逐位，训推一致 |
| `LUMEN_ENABLE` 里的 `flydsl_conv` / `aiter_conv` | VAE 卷积走 aiter FlyDSL；DiT patch embedding 走 aiter Triton conv3d | 单次 VAE encode −22%，整网持平 | 不逐位，训推一致 |
| `SIMWAM_GRPO_AITER_GEMM`（1） | 调优表命中的 `nn.Linear` 前向走 aiter tuned GEMM | GPU 阶段 −2.2%（去噪 −5.5%） | 不逐位，训推一致 |
| `SIMWAM_GRPO_XATTN_KV`（1） | text context 的 cross-attention K/V 每个 condition 只算一次 | GPU 阶段 −8.9%（去噪 −16%），+0.76 GB/卡 | 不逐位，训推一致 |
| `SIMWAM_GRPO_FUSED_LN_MODULATE`（1） | LayerNorm + modulate 合成一个 Triton kernel | GPU 阶段 −6.5% | 不逐位，训推一致 |
| `SIMWAM_GRPO_LORA_CAT`（1） | LoRA 的 A 拼在 base 权重下面，一个 GEMM 得到 base 输出和 `x @ A.T`，再一个 Triton kernel 加上 `mid @ B.T` | GPU 阶段 −5.9%（去噪 −13.5%） | 不逐位，训推一致 |
| `SIMWAM_GRPO_AITER_LAYERNORM`（1） | 带仿射的 LayerNorm 换 aiter Triton `layer_norm` | kernel 快约 25%，周期无可测变化 | 不逐位，训推一致 |

### 4.4 去掉重复计算和拷贝（逐位一致）

| 开关（默认） | 作用 | 实测 |
|---|---|---|
| `SIMWAM_GRPO_SHARED_KV`（1） | 合批调用里重复 W 份的 condition 只存一套共享缓冲，graph 原地读 | 2 卡：训练峰值显存 −8.8 GB，eval −6.6 GB |
| `SIMWAM_GRPO_EVAL_EMPTY_CACHE`（1） | 每次 eval 后 `empty_cache` | eval 后整卡占用 −6.4 GB |
| `SIMWAM_GRPO_KV_SLOT`（1） | no-grad graph 里 video K/V 和 action K/V 的拼接只写 action 行 | 去噪 −5.8% |
| `SIMWAM_GRPO_KV_SLOT_TRAIN`（1） | 训练 graph 里同样只写 action 行 | 训练前向 −7.4%，显存 −2.1 GB |
| `SIMWAM_GRPO_XATTN_GROUPED`（nograd） | no-grad 合批调用按步分组做 cross-attention，不复制 context K/V | rollout log-prob −2.5 ms |
| `SIMWAM_GRPO_EXACT_GATE`（1） | `x + gate * residual` 合成一个舍入方式和 torch 相同的 Triton kernel | 去噪 −4% |
| `SIMWAM_GRPO_LORA_NOGRAD_SKIP`（1） | no-grad 调用里 LoRA 不保存反向用的中间结果 | GPU 阶段 −3.6% |
| `SIMWAM_GRPO_TRAIN_GRAD_FLAT`（1） | 训练 graph 的反向把梯度直接写进 optimizer 的 flat 缓冲 | 反向 −2.8～−3.8 ms，optimizer 约 −3 ms |
| `SIMWAM_GRPO_KV_SLOT_SHARED_NOGRAD`（1） | rollout 的合批 log-prob 也只写 action 行（复用训练的 slot 缓冲） | rollout log-prob −4.0～−5.2 ms |
| `SIMWAM_GRPO_EXPAND_VIEW`（1，`62acd4c5`） | batch 为 1 的 condition，video K/V 用 expand view 代替复制 8 份 | 显存 −716 MB |
| `SIMWAM_GRPO_KV_SLOT_STEP_SHARED`（1，`62acd4c5`） | 每步去噪的 graph 原地读共享缓冲的前 8 行，不再每次换 condition 都拷贝 | 拷进 graph 的量每 40 步 7.25 GB → 163 MB；GPU 阶段 −1.8～−6.0 ms；显存 −1.45 GB |

### 4.5 主机侧

| 开关（默认） | 作用 | 实测 | 数值 |
|---|---|---|---|
| `SIMWAM_GRPO_PDM_PREFETCH`（1） | 下一批的 PDM metric cache 在当前批训练时预先加载 | 2 卡周期 −4.6%；4 卡 1000 步对 `rollout` 模式平均周期 425 对 509 ms | 逐位一致 |
| `SIMWAM_GRPO_VIS_BACKGROUND`（1） | rank 0 的可视化放后台线程 | 4 卡每周期平均约 −26 ms（其余 rank 不再等 rank 0 画图） | 逐位一致，图逐字节相同 |

### 4.6 试过但没有保留

| 尝试 | 结果 |
|---|---|
| 带梯度 attention 换 aiter Triton FMHA（`SIMWAM_GRPO_AITER_FMHA_GRAD`） | 快 4.2%，但训推又不一致，默认关 |
| `SIMWAM_GRPO_FUSED_MODULATE`（addcmul） | 4 卡 4000 步里反向慢 2–3 ms、周期没变快，默认关 |
| LoRA 第二个 GEMM 用 `addmm`、NUMA 绑核 | 分别慢 3.1% 和 2.3%，默认关 |
| `SIMWAM_GRPO_PREFILL_NOCOPY` | 一半 prefill 只能走 eager，只省 0.8 GB，默认关 |
| LORA_CAT 新 GEMM 形状的 aiter 调优表 | 单 kernel 快 8–23%，整周期约 −1%（噪声内），这次不放 |

---

## 5. Runbook

### 5.1 环境

镜像、Lumen、aiter、SimWAM、权重和 NAVSIM 数据的准备见 [README.md](README.md) 的 “What you need” 和 “Environment”。本文的结果需要：

- `AITER_SRC` 指向 `huizzhan/aiter` 的 `hz/grpo-opt`（`c9be3a5`）checkout：conv3d 和本例的 GEMM 调优表都从这里加载。没设时会打 warning，conv3d 和 AITER_GEMM 不生效。
- `flydsl_conv` 需要 Lumen PR #32 的 `examples/wan2.2/flydsl_conv.py`；镜像自带的 aiter 先在容器里执行一次 `$LUMEN_DIR/examples/perf-veomni/overlay_aiter.py --apply`。

```bash
bash examples/Embodied_SimWAM_NAVSIM/launch_container.sh
docker exec -it simwam-rl bash examples/Embodied_SimWAM_NAVSIM/setup_env.sh
```

### 5.2 优化版（FIN20 的配置，全部默认值）

在容器里、Lumen-RL 根目录下：

```bash
NPROC=4 CUDA_VISIBLE_DEVICES=0,1,2,3 \
LUMEN_DIR=<Lumen> SIMWAM_ROOT=<SimWAM> AITER_SRC=<aiter hz/grpo-opt checkout> \
  bash examples/Embodied_SimWAM_NAVSIM/run_embodied_grpo.sh \
  data.train.scene_filter=<Lumen-RL>/examples/Embodied_SimWAM_NAVSIM/configs/navtrain_pdm_score_below0p9_complete.yaml \
  max_steps=20000
```

`LUMEN_ENABLE` 不设，用脚本默认的 `hf_attn_patch,aiter_conv,flydsl_conv`。日志里应看到 `aiter_conv bound 63 Conv3d and 10 Conv2d`，没有 conv3d 回落的 warning。

### 5.3 切回比特一致版本

**最可靠的做法**：用 #51 合并前的 head `f3f3ca5`（BX20 的代码），设 `LUMEN_ENABLE=hf_attn_patch`，其余同 5.2。
这个组合在 4 卡上 8 步对原始示例 `d0ada15` 逐位相同。

**在当前代码上**：设 `LUMEN_ENABLE=hf_attn_patch`，并关掉所有改变数值的开关（README.md 的 “Faster, not bit-identical” 一节）：

```bash
export LUMEN_ENABLE=hf_attn_patch
export SIMWAM_GRPO_DROP_MASKS=0 SIMWAM_GRPO_BATCHED_GEMM=0 SIMWAM_GRPO_BATCHED_LOGP_NOGRAD=0 \
       SIMWAM_GRPO_AITER_RMSNORM=0 SIMWAM_GRPO_TRITON_RMSNORM_BWD=0 SIMWAM_GRPO_AITER_ATTN_BWD=0 \
       SIMWAM_GRPO_TRITON_ROPE=0 SIMWAM_GRPO_PREFILL_GRAPH=0 SIMWAM_GRPO_XATTN_KV=0 \
       SIMWAM_GRPO_FUSED_LN_MODULATE=0 SIMWAM_GRPO_AITER_GEMM=0 SIMWAM_GRPO_AITER_LAYERNORM=0 \
       SIMWAM_GRPO_LORA_CAT=0 SIMWAM_GRPO_FUSED_MODULATE=0
```

逐位一致的开关（4.4、4.5 节，包括 `62acd4c5` 的 `SIMWAM_GRPO_EXPAND_VIEW` 和 `SIMWAM_GRPO_KV_SLOT_STEP_SHARED`）可以保持打开。
注意：这个组合在当前 head 上没有重新做过对 `d0ada15` 的逐位复验；需要严格的比特一致基线时用 `f3f3ca5`。
只想回到 `daf60b17`（FIN20）的行为：`SIMWAM_GRPO_EXPAND_VIEW=0 SIMWAM_GRPO_KV_SLOT_STEP_SHARED=0`。

### 5.4 验收

- 训练完成目标步数、退出码 0；日志里 conv3d 绑定 63 个、没有回落 warning。
- `SIMWAM_GRPO_CUDA_GRAPH_VERIFY=1 SIMWAM_GRPO_TRAIN_GRAPH_VERIFY=1` 跑一次短运行（如 24 步、`eval_every=20`）：每次 graph replay 和 eager 逐位相同。
- 性能比较用同卡交替的多次运行；会改变训练轨迹的开关按 GPU 阶段比较（40 步的周期中位数会被 PDM 打分的长尾主导）。

---

## 6. 故障排查

| 现象 | 原因 | 处理 |
|---|---|---|
| `aiter conv3d failed to load ...; Conv3d stays on torch` | `AITER_SRC` 没设，或指向的 checkout 没有 conv3d | 指向 `huizzhan/aiter:hz/grpo-opt` |
| `SIMWAM_GRPO_AITER_GEMM: AITER_CONFIG_GEMM_BF16 is unset` | `AITER_SRC` 里没有本例的调优表 | 同上，或显式设 `AITER_CONFIG_GEMM_BF16` |
| VAE 仍走 torch，日志有 FlyDSL warning | 镜像的 aiter 没有 FlyDSL conv kernel | 容器里先执行 `overlay_aiter.py --apply` |
| 周期时间偶尔几百毫秒的尖峰 | PDM 打分的长尾（个别 token 的 metric cache 加载或仿真） | 正常现象；比较时用平均周期或 GPU 阶段 |
| `SIMWAM_GRPO_BATCHED_LOGP_VERIFY=1` 崩溃 | 已知问题，和默认开关无关 | 不要在默认配置下打开它 |

---

## 7. 文件索引

| 文件 | 作用 |
|---|---|
| `plugin.py` | 全部运行时 patch 和开关 |
| `fused_kernels.py` | 自写的 Triton kernel（RMSNorm 反向、RoPE、LayerNorm + modulate、gate、LoRA） |
| `train_grpo.py` | 安装 patch 并启动 SimWAM 的 GRPO 训练 |
| `run_embodied_grpo.sh` | 训练入口，设置 `LUMEN_ENABLE`、`AITER_SRC` 和 GEMM 调优表 |
| `README.md` | 每个开关的详细说明和单项实测 |
