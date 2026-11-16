# Qwen3-8B dense BF16 训推一致性

记录 Qwen3-8B-Base、BF16、Lumen FSDP2 加 8 路 ATOM TP1 在 gfx950（8 卡，测量机为 MI355X）上的训推 logprob 对齐。FP8 不在本文范围。比较对象是博客 ROCm Figure 5 的 q 对 s^D：同一份权重、更新前，训练 logprob 与在线 decode logprob 的平均绝对差。栈与博客的 Megatron TP4/CP2 加 vLLM、MI300X 不同，数值不和那张图的曲线并列。

## 当前协议与结果

本节是当前代码的协议和结果。后面「旧协议」各节保留为历史记录，数字不要和本节混用。

### 开关

```bash
MODE=atombf16 LUMENRL_CONSISTENT=1 bash examples/DAPO/run_dapo.sh
```

`LUMENRL_CONSISTENT=1` 只接受 `atombf16` 和 `atomfp8`，其他 `MODE` 直接退出。它导出六个 `LUMEN_ATOM_OPS`（`rmsnorm,residual_rmsnorm,qkv,rope,varlen_attn,gate_up`）、`LUMENRL_ALIGN_DECODE=1`、`LUMENRL_ALIGN_CONTIG_KV=1`、`LUMENRL_ALIGN_BATCHED_DECODE=1`、`LUMENRL_PREFILL_DECODE_ATTN=1`、`LUMENRL_ROWWISE_TRAIN_ATTN=1`、`LUMENRL_ALIGN_SYNC_OUTPUT=0`、`LUMENRL_PREFILL_RESCORE=0`，并从 tokenizer 读出 `LUMENRL_TRAIN_VOCAB_MASK`（Qwen3-8B-Base 为 151669）。`ALIGN_DECODE=1` 时走 eager、`compilation_config.level=0`。

开关不设置 GEMM 表。下面的测量使用 `configs/qwen3_8b_bf16_tuned_gemm.opus.csv`，由调用方通过 `AITER_CONFIG_GEMM_BF16` 指定。

### 200 step 结果

两组都是 8 个 prompt × 16 条回复，8 卡，seed 1234，`kl_coeff=0`，`gpu_memory_utilization` 与旧协议相同。baseline 与 consistent 之间的差别如下：

| 项 | baseline | consistent |
| --- | --- | --- |
| `mismatch/abs_diff` 为 0 的步数 | 0 / 200 | 200 / 200 |
| `mismatch/abs_diff` 均值 | 0.0086 | 0 |
| `frac_abs_gt_0.1` 均值 | 0.0304 | 0 |
| `mismatch/abs_max` 均值 | 0.907 | 0 |
| rollout 吞吐，tok/GPU/s，1–200 步均值 | 253.8 | 227.7（约 90%） |
| rollout 吞吐，1–100 步 | 181.5 | 182.3 |
| rollout 吞吐，101–200 步 | 326.1 | 273.1 |
| 每步总时间，s | 105.7 | 108.3 |
| 生成，s | 95.1 | 99.1 |
| old_log_prob，s | 2.93 | 2.52 |
| actor update，s | 6.54 | 5.55 |
| 平均回复长度，tok | 1532 | 1399 |
| 生成时间 / 最长回复 token，ms | 14.33 | 15.86 |

rollout 吞吐定义为每步「每卡序列数 × 平均回复长度 ÷ 生成时间」，再对步取平均。多序列每卡 16 条序列。

后 100 步 consistent 的回复更短（均值 1902 对 2078），但每个最长回复 token 的生成时间更长（16.12 对 14.56 ms）。这个差异的来源还没有 profile。

baseline 的 mismatch 非零，原因是 `LUMEN_ATOM_OPS` 为空（HF RMSNorm、HF RoPE、HF attention）、vocab mask 为空、`ATOM_FORCE_ATTN_TRITON=0`。两组都用同一张 opus GEMM 表，`LUMENRL_TRAIN_ATOM_GEMM=1` 让 baseline 的 actor 线性层也走 `tgemm.mm`。consistent 另外设置 `LUMENRL_STABLE_GEMM=1`（`fsdp_backend.py` 在 `LUMEN_ATOM_OPS` 非空时 `setdefault`）和 `ATOM_FORCE_ATTN_TRITON=1`。

### 版本

- Lumen：`b816848`，即 `origin/main`（`7b5931d`）之上的 7 个提交，分支 `dev/ATOM_rollout`。
- Lumen-RL：`a7f4886` 加上「Gate consistent decode on LUMENRL_CONSISTENT and split keys past 1024」提交。200 step 的 pin 记录 `align_decode.py` sha256 前缀 `da41e5c70a592960`、`qlen1_attention.py` sha256 前缀 `4d0ad552b45cb4ba`，与当前 HEAD 的文件一致。
- ATOM：`6e9b3e1` 加上 sampler seed 补丁（未上游）。
- aiter：`4eb7dcac`。
- GEMM 表：`configs/qwen3_8b_bf16_tuned_gemm.opus.csv`，sha256 前缀 `ea31df209345cc00`。
- 镜像：`zhangdanyangamd/lumen-rl:dapo-gfx950-rocm7.2.3-260917`。
- 日志：`logs/ex5-fig100-seed-lumennew-base-opus-p8g16-200step-run.log` 和 `logs/ex5-fig100-seed-lumennew-consistent-opus-p8g16-200step-run.log`，位于训练机的 `lumen-atom-rollout/logs/`，未纳入 git。两组都以退出码 0 结束。

### GEMM 覆盖

只用 aiter 自带表加 `qwen3_8b_bf16_tuned_gemm.csv` 时，`qkv`、`o_proj`、`down_proj` 没有命中，`gate_up` 与 `lm_head` 只在 M≤16 命中。100 步多序列实测 13824 个唯一 `(M,N,K)` 未覆盖，覆盖 32 个。这就是要显式指定 opus 表的原因。

### 反向

packed 训练的反向仍是对同一 Q、K、V 的整段 causal varlen（`_RowwiseAttention.backward`）。它在短序列上与前向一致，query 读超过 2048 个 key 后与前向不同。mismatch 为 0 不代表更新使用的是产生 logprob 的那个前向的梯度。

## 比较定义

训练是 Lumen 的一次 teacher-forcing。在线打分是 ATOM `generate` 返回的 decode logprob。位置 i 的 hidden 给 token i+1 打分，丢掉序列最后一个 token。logprob 在 GPU 上用 float32 `log_softmax`。日志字段 `mismatch/abs_diff` 是回复 mask 上 `|rollout − 训练|` 的均值。`frac_abs_gt_0.1` 是绝对差超过 0.1 的回复 token 比例。

博客分解是 `q/μ = (q/s^P) × (s^P/s^D) × (s^D/μ)`。q 是 packed teacher-forcing，s^P 是整段 prefill，s^D 是逐步 decode。`LUMENRL_PREFILL_RESCORE=1` 会在生成结束后用整段 prefill 替换进 loss 的 logprob，这时日志里的差是 q 对 s^P。本文的在线数字都在 `LUMENRL_PREFILL_RESCORE=0` 下取得，对应 s^D。

Lumen 的训练 forward 只有 teacher-forcing，没有单独的 prefill 或 decode 接口。ATOM 的 prefill 指 rollout 对 prompt 的一次前向，decode 指随后逐 token 的前向。

Callback 打印的 step 是训练循环下标加 1。trace 文件名用 0-based 下标，所以显示 step 30 对应 `step-00029.pt`。温度是 1，`SamplingParams` 没有 seed 字段，trainer 的 seed 不进入 ATOM 采样。两次运行的 token 可以不同，每次只和当次保存的 logprob 比较。

## 旧协议结果（历史记录）

以下数字属于旧协议：私有 GEMM 表 `private.csv`、`SYNC_OUTPUT=1`、没有 split-KV。保留作历史记录，与上一节的当前协议不可直接比较。

在「同形状」条件下，Lumen teacher-forcing 与 ATOM 整段 prefill 的选中 logprob 逐位相同，包括 4096 token 和 8 卡 FSDP2。

在「在线 decode」条件下，关闭 rescore、打开下文的对齐开关之后，差分成两档：

- 多数 step 的 `mismatch/abs_diff` 在 1e-10 到 1e-9。200 step 里有 192 步低于 1e-4，这 192 步均值 1.43e-9，中位数 1.51e-10，其中 47 步为 0。
- 少数 prompt 长度上，整段 causal varlen 与 q_len=1 varlen 的第一个回复 token 差 0.01 这一量级。200 step 因此出现 8 个离群步，总均值被抬到 4.77e-4。把 prefill 改成与训练相同的 q_len=1 之后，同样 seed 的前 30 step 最大值为 1.72e-7，step 30 为 2.61e-10。step 57 及以后的离群下标尚未在这组 prefill 条件下重跑。

## 旧协议所用版本

下表是旧协议 200 step 与 30 step 数字所用的栈。当前协议的版本见「当前协议与结果」一节。

| 项 | 提交或路径 |
| --- | --- |
| Lumen-RL | `dev/qwen3-8b-bf16-align`，`dev_wul` |
| Lumen | `dev/ATOM_rollout` `0c2810c`。stable GEMM 与 fused residual 在 `lumen/config.py` |
| ATOM | ROCm `main` `6e9b3e1c`。测量中未改源码 |
| aiter | ROCm `main` `4eb7dcac`。测量中未改源码 |
| 模型 | `Qwen3-8B-Base`，BF16 |
| 镜像 | `zhangdanyangamd/lumen-rl:dapo-gfx950-rocm7.2.3-260917` |

本机镜像内的 aiter 导入 flydsl 时失败。测量进程把主机 aiter `4eb7dcac` 放在 `PYTHONPATH` 前面，并设置 `LUMENRL_ATOM_AITER_SRC`。`examples/DAPO/atom_aiter_shim/sitecustomize.py` 从这份源码补镜像 aiter 缺少的模块。

## 代码与生效条件

六个算子 tag 为 `rmsnorm,residual_rmsnorm,qkv,rope,varlen_attn,gate_up`。`LUMEN_ATOM_OPS` 为空时这些替换不发生。`run_dapo.sh` 中只有 `LUMENRL_CONSISTENT=1` 才把它们设成默认值，并同时设置 `LUMENRL_ALIGN_DECODE`、`LUMENRL_ALIGN_SYNC_OUTPUT`（默认 0）、`LUMENRL_ALIGN_CONTIG_KV`、`LUMENRL_ALIGN_BATCHED_DECODE`、`LUMENRL_PREFILL_DECODE_ATTN`、`LUMENRL_ROWWISE_TRAIN_ATTN`。`LUMENRL_PREFILL_RESCORE` 默认是 0。能读到 tokenizer 时，脚本把 `LUMENRL_TRAIN_VOCAB_MASK` 设成 tokenizer 词表长度，和 ATOM 截掉的 lm_head padding 用同一个分母。

| 条件 | 代码 | 测量到的作用 |
| --- | --- | --- |
| `LUMENRL_CONSISTENT=1` | `examples/DAPO/run_dapo.sh` | 打开上面这一组。只允许 `atombf16` 和 `atomfp8`。 |
| `LUMENRL_STABLE_GEMM=1` | Lumen `LumenConfig._stable_tgemm_mm`；`lumenrl/aiter_patch/sitecustomize.py` | M 补到 splitK=1 后再调 `tgemm.mm`。splitK>1 的 FlyDSL bf16 原子加在同一输入上两次结果不同；补齐后同一输入两次差为 0 |
| 有 `LUMEN_ATOM_OPS` | `fsdp_backend.py` | 空 quant 配置也会把 rollout 设为 ATOM，并 `setdefault` stable GEMM |
| `AITER_CONFIG_GEMM_BF16` | 调用方在进程启动前设置；`atom_ray_server.py` 和 `vllm_ray_server.py` 转发给 worker | aiter 在第一次查表前读取。未设置时用 aiter 自带的 GEMM 配置 |
| fused residual | Lumen `lumen/config.py` | residual 与 hidden 一起作为下一层输入。原先放在 `backbone._atom_residual` 时，activation checkpoint 重算单层走了无 residual 分支 |
| `LUMENRL_ROWWISE_TRAIN_ATTN=1` | `lumenrl/engine/training/packing.py` | 按 256 行分块的因果 attention（`chunked_causal_attention`），超过 1024 个 key 的行按 1024 切分后在 fp32 中合并，logprob 用 GPU float32 `log_softmax`。变量本身默认关闭，`LUMENRL_CONSISTENT=1` 时打开。反向仍是整段 causal varlen，见「当前协议与结果」中的「反向」。 |
| `ATOM_FORCE_ATTN_TRITON=1` 且 prefill | `lumenrl/aiter_patch/sitecustomize.py` | prefill RoPE 保持 aiter。该开关否则会让 prefill 使用 Triton RoPE，Q/K 与训练不一致 |
| `LUMENRL_ALIGN_DECODE=1` | `lumenrl/aiter_patch/align_decode.py` | decode 使用 aiter RoPE，KV 写入 pool，超过 1024 key 的行按 1024 切分（`decode_split_attention`）。prefill 由 `LUMENRL_PREFILL_DECODE_ATTN=1` 走同一条 q_len=1 归约。地址逐次变化，因此 eager、`compilation_config.level=0`。level 3 FULL graph 重放固定地址后中止 |
| `LUMENRL_ALIGN_CONTIG_KV=1` | 同上 | `nseq==1` 时 decode 读连续 K/V。容量 8192 |
| `LUMENRL_ALIGN_SYNC_OUTPUT=0`（当前协议） | 同上 | deferred D2H 读的 one-token gather 由 `record_stream` 保持到拷贝完成，下一步 forward 不会复用那块存储。`=1` 是旧协议的默认，返回前 clone 并同步当前 stream，只作诊断 |
| `LUMENRL_PREFILL_RESCORE=1` | `atom_ray_server.py`，`prefill_rescore.py` | 生成结束后用整段 prefill 替换 loss 中的 logprob。在线 s^D 的测量保持为 0。`run_dapo.sh` 的 BF16 ATOM 模式默认是 0；打开时每个 replica 写自己的目录 |
| `LUMENRL_ALIGN_SYNC_WEIGHTS=1` | `actor_worker.py`，`align_decode` 的 fence | 最后一桶 IPC 之后 `cuda.synchronize`，并打印 q_proj 指纹。默认关闭 |

`atom_ray_server.py` 把上表中的环境变量转给 ATOM worker，并把多桶权重更新放在 `weight_sync_session()` 内。`LUMENRL_IGNORE_EOS=1` 时采样忽略 EOS，用于把回复打满指定长度。`LUMENRL_TRAIN_DECODE_TRACE_DIR` 设置时，rank 0 在更新前保存 `input_ids`、mask、`old_log_probs`、`rollout_log_probs` 和 `prompt_lengths`。

## 在线测量的公共条件（旧协议）

seed 与 `reward.shuffle_seed` 都是 1234。`gen_batch_size=1`，`num_generations=8`，`train_global_batch_size=8`。回复上限 7168，总长上限 8192，`max_token_len_per_gpu=4096`，引擎 `max_model_len=8192`，`max_num_batched_tokens=16384`。`kl_coeff=0`。`dynamic_sampling=false`，`filter_groups.enable=false`。`gpu_memory_utilization=0.38`，sleep level 2，eager，compilation level 0。

`kl_coeff=0.001` 会再加载一份 reference model，8 张卡的显存不够第二组 worker。filter groups 打开时，同一组 8 条准确率没有方差，整步会被丢弃。这两项在本文的在线 run 里保持关闭。

在线 run 同时打开：六个 `LUMEN_ATOM_OPS`，`LUMENRL_ALIGN_DECODE=1`，`LUMENRL_ALIGN_SYNC_OUTPUT=1`，`LUMENRL_ALIGN_CONTIG_KV=1` 且容量 8192，`LUMENRL_ROWWISE_TRAIN_ATTN=1`，`LUMENRL_STABLE_GEMM=1`，`ATOM_FORCE_ATTN_TRITON=1`，`ATOM_H2D_BACKEND=direct`，`ATOM_USE_TORCH_RMSNORM=0`。`LUMENRL_PREFILL_RESCORE=0`。GEMM 表是当时的 `configs/qwen3_8b_bf16_tuned_gemm.private.csv`，该文件已经从仓库删除。这组 run 没有打开 `LUMENRL_ALIGN_BATCHED_DECODE`；200 step 的 prefill 仍是整段 varlen。当前 `run_dapo.sh` 不设置 `AITER_CONFIG_GEMM_BF16`。调用方要指定表时设置这个变量，仓库里保留的是 `configs/qwen3_8b_bf16_tuned_gemm.opus.csv`。未设置时 aiter 用自带配置。

## 旧协议测量（历史记录）

下面的数字都属于「旧协议所用版本」的栈，GEMM 用的是已删除的 `private.csv`。其中的「200 step」是 2026-10-20 的旧协议 run，与「当前协议与结果」中的 200 step 不是同一组。日志在训练机的 `lumen-atom-rollout/logs/`，未纳入 git。

**同形状 teacher-forcing 对整段 prefill。** 六个 tag，stable GEMM，prefill RoPE 在 aiter。16 与 24 token 的选中 logprob 差为 0。4096 token 在单卡和 8 卡 FSDP2 上，4095 个选中 logprob 的 max abs 为 0，第 0 层到第 35 层 hidden 为 0。

**2048 key。** 同一条 eager 对齐 decode 上，位置 0–2047 的 logits 逐位相同。位置 2048 起，整段 `flash_attn_varlen` 与 q_len=1 的归约不同：4095 个 logprob max abs 2.868e-1，mean abs 2.574e-2。单条 query 把 `max_seqlen_q` 设成 4096 不改变该差。`LUMENRL_ROWWISE_TRAIN_ATTN=1` 且 prefill 也用 q_len=1 时，2176 token 的 2175 个选中 logits 和 GPU float32 logprob 为 0。24 token 上两次 backward 的 loss 位模式相同，399/399 参数梯度逐位。

**GEMM 确定性。** o_proj 与 QKV 在部分 M 上查到 splitK>1，同一输入两次 bf16 结果不同。当时的 `configs/qwen3_8b_bf16_tuned_gemm.private.csv` 把这些形状改为 splitK=1 后，同一输入两次差为 0。Qwen3-8B 在这张已删除的表上探测过的 M 集合 `{1,2,4,8,16,32,64,128,256,512,1024,2176,2310}` 都是 splitK=1，且同一形状内 kernel 名相同。这个结果属于那张表。当前默认不设置 `AITER_CONFIG_GEMM_BF16`，未设置时走 aiter 自带配置；要沿用一张固定表，调用方自己导出该变量，仓库里保留的是 `configs/qwen3_8b_bf16_tuned_gemm.opus.csv`。训练前向曾按 M=1 逐段调用 `tgemm.mm`：8 卡、rescore 关闭、回复打满 2176、17408 个回复 token，`mismatch/abs_diff` 3.50e-9，`frac_abs_gt_0.1` 为 0。该逐段路径已从代码移除。日志 `logs/ex5-gemmchunk.log`。

**在线短回复。** rescore 关闭，逐行训练，`LUMENRL_ALIGN_DECODE=1` 且返回前 clone。回复上限 512：8 条序列 prompt 134、总长 646，4096 个回复 token，`mismatch/abs_diff` 为 0，trace 非零个数为 0。日志 `logs/ex5-logprob-snap.log`。

**在线长回复，prefill 仍是整段 varlen。** 回复打满 2176：17408 个 token，均值 8.26e-10，位置 0–511 为 0，trace 非零 54/17408，最大绝对差 9.54e-7，非零处 rollout 更高。模型自行停止的一步：序列长度 238–4670，10060 个回复 token，均值 2.43e-10，trace 非零 8。回复打满 7168、每卡 pack 上限 4096：8 条总长都是 7302，57344 个回复 token，均值 9.42e-10，`frac_abs_gt_0.1` 为 0，最大绝对差 3.81e-6。三步退出码都是 0。日志 `logs/ex5-snap2176.log`、`logs/ex5-snapmix.log`、`logs/ex5-snap7168.log`。

连续 K/V 之前，同一 2176 协议的 `mismatch/abs_diff` 是 3.84e-9（trace 非零 155）。加上连续 K/V 后降到 3.43e-10（trace 非零 13）。再打开整段 q_len=1 rescore 后为 0，17408/17408 逐位。日志 `logs/ex5-syncout2176.log`、`logs/ex5-contigkv2176.log`、`logs/ex5-rescore-rowwise2176.log`。这组 rescore 为 0 的数是 s^P，不是 200 step 的 s^D。

**200 step，prefill 仍是整段 varlen。** 公共条件，`LUMENRL_PREFILL_RESCORE=0`，未设置 `LUMENRL_IGNORE_EOS`，未打开逐 token prefill。2026-10-20 14:49:51Z 至 22:23:41Z，27230 秒，退出码 0。200 步均值 4.771e-4，中位数 1.506e-10，最小 0，最大 0.015071。192 步低于 1e-4，均值 1.425e-9，最大 3.729e-8，47 步为 0。区间 `[1e-7, 1e-3)` 内没有步。8 个离群步：

| step | abs_diff | frac_abs_gt_0.1 |
| --- | --- | --- |
| 57 | 0.015071 | 0.055954 |
| 83 | 0.014591 | 0.055089 |
| 30 | 0.013787 | 0.048584 |
| 199 | 0.013469 | 0.049678 |
| 126 | 0.011520 | 0.041075 |
| 120 | 0.010706 | 0.039154 |
| 186 | 0.008982 | 0.030303 |
| 160 | 0.007300 | 0.024986 |

日志 `logs/ex5-blog200-online.log`。更早一次没有连续 K/V、也没有同步返回的 hostcopy run 停在 step 66，其中 step 30 为 0.02176，step 57 为 0.014527。离群 step 的序号在两次数据顺序相同的 run 里重复，序列长度与尖峰没有对应关系。其中 4 步的前一步 `grad_norm` 为 0，另外 4 步前一步 `grad_norm` 大于 0。离群步的下一步回到约 1e-10。

**step 30 的 trace。** `LUMENRL_ALIGN_SYNC_WEIGHTS=1` 的 60 step run 在显示 step 49 停止。trace `logs/blog60-fence-trace/step-00029.pt`：prompt 长度 354，8 条序列，6252 个回复 token，均值 0.025631，最大 0.873559，598 个 token 绝对差超过 0.1。8 行从第一个回复 token 起就有差。位置错开一格时均值约 1.65，与 0.0256 不是同一类。前一条 trace `step-00028.pt` 的 prompt 长度为 117，前 32 个回复 token 为 0。显示 step 31 的 prompt 长度为 144，均值为 2.817e-10。8 张卡上同一 token 的 logprob 相同。训练侧 8 个 rank 的 q_proj crc 相同；ATOM 侧按 `layers.0.self_attn.q_proj.weight` 过滤时打印 missing。fence 打开后的 step 30 大于未开 fence 的 0.013787。

基座权重、单卡、强制喂入这条 trace 的前 8 个回复 token。整段 prefill 对 q_len=1 prefill：prompt 117 的 8 个 logprob 逐位相同。prompt 354 的三条序列，第一个回复 token 绝对差为 0.0584、0.0666、0.1834。整段加载 `fmha_fwd_hd128_bf16_causal_group`，逐 token 加载 `fmha_fwd_hd128_bf16_group`。基座权重下的绝对值和 step 30 训练权重下的 0.123105 不同。

**prefill 改为 q_len=1 之后的 30 step。** 条件与 200 step 相同，并设置 `LUMENRL_PREFILL_DECODE_ATTN=1`。`LUMENRL_ALIGN_DECODE=1` 现在也会走这条 prefill。2026-10-25 13:53–14:53 UTC，退出码 0。30 步 `mismatch/abs_diff` 最小 0，最大 1.7156e-7（step 16），均值 6.706e-9。step 29 为 9.64387e-11，step 30 为 2.61195e-10。30 步的 `frac_abs_gt_0.1` 都是 0。日志 `logs/ex5-blog30-q1fix.log`。1e-9 这一档仍在这 30 步里。step 57、83、120、126、160、186、199 不在这次长度内。
