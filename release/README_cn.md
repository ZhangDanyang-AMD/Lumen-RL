# LumenRL —— AMD Instinct 上的 DAPO 数学强化学习

LumenRL 强化学习栈在 AMD GPU 上的可复现发布版本：**训练与推理共用同一组 8 张卡**，
训练后端可选 FSDP2 或 Megatron，rollout 引擎可选 vLLM 或 ATOM，精度可选 BF16 或 FP8。

> English version: [README.md](README.md)

本文所有内容都在 [`versions.env`](versions.env) 固定的版本上端到端跑过，没有一条是"预期如此"。
§6 参考值表里的每个数字都标注了它是用哪条命令、哪条 config、几步测出来的。

**最短路径**（细节见 §4）：

```bash
export DATA_ROOT=/path/to/data
docker pull zhangdanyangamd/lumen-rl:dapo-gfx950-rocm7.2.3-260902
bash release/run_example.sh 1 --check
```

`release/run_example.sh` 是宿主机侧的启动器：它负责清场、起容器、拼好全部环境变量、
跑完之后把指标和内置参考值逐项比对打出 `PASS` / `FAIL`。**七个例子都只差一个数字。**

---

## 1. 这个版本包含什么

| | |
|---|---|
| 任务 | DAPO 数学 RL（GRPO 风格，按 uid 分组归一化） |
| 模型 | Qwen3-8B-Base（dense）、Qwen3-30B-A3B-Base（MoE，128 专家） |
| 训练 | Lumen FSDP2（BF16 / FP8 blockwise2d）、Megatron-Native（EP=8） |
| Rollout | vLLM 0.23.0（BF16 / `fp8_per_block`）、ATOM（BF16 / `per_block_fp8`） |
| 拓扑 | 单个 Ray driver 进程内 8 个训练 actor + 8 个同卡 colocated rollout replica（TP=1） |
| 权重同步 | ZMQ CUDA-IPC，同卡直传，带覆盖率断言 |
| 硬件 | **仅 gfx950** —— MI350X / MI355X，8 卡 |

算法侧：clip-higher + dual-clip + token-mean 策略损失、动态采样（`filter_groups`）、
overlong 奖励缓冲、TIS rollout 修正。

---

## 2. 七个已验证的例子

七个例子的训练和推理**都跑在同一组 8 张卡上**。实测环境：8x MI355X（gfx950）、ROCm 7.2、
镜像 `dapo-gfx950-rocm7.2.3-260902`，版本按 [`versions.env`](versions.env)。

### 2.1 概览

| # | 例子 | 训练 | Rollout | 跑它 |
|---|------|------|---------|------|
| 1 | 8B BF16 基线 | FSDP2 BF16 | vLLM BF16 | `bash release/run_example.sh 1 --check` |
| 2 | 8B FP8 rollout | FSDP2 BF16 | vLLM `fp8_per_block` | `bash release/run_example.sh 2 --check` |
| 3 | 8B FP8 端到端 | FSDP2 **FP8 blockwise2d** | vLLM `fp8_per_block` | `bash release/run_example.sh 3 --check` |
| 4 | 8B ATOM FP8 | FSDP2 **FP8 blockwise2d** | **ATOM** `per_block_fp8` | `bash release/run_example.sh 4 --check` |
| 5 | 8B ATOM BF16 | FSDP2 BF16 | **ATOM** BF16 | `bash release/run_example.sh 5 --check` |
| 6 | MoE FSDP2 | FSDP2 BF16 | vLLM BF16 | `bash release/run_example.sh 6 --check` |
| 7 | MoE Megatron EP=8 | **Megatron** TP=PP=CP=1，EP=8，DP=8 | vLLM BF16 | `bash release/run_example.sh 7 --check` |

### 2.2 每个例子的完整参数

启动器内部就是这张表。手工跑（附录 A）时**这一行的每一列都要给全**。

| # | `MODE` | `TRAIN_FP8` | `CONFIG_OVERRIDE`（相对 `$RL_ROOT/Lumen-RL`，都在 `examples/DAPO/configs/` 下） | `STEPS` | `max_response_length` | 模型 | 额外 env |
|---|---|---|---|---|---|---|---|
| 1 | `bf16` | `0` | `dapo_qwen3_8b_ray_vllm_smoke.yaml` | 3 | 512 | Qwen3-8B-Base | — |
| 2 | `fp8` | `0` | `dapo_qwen3_8b_ray_vllm_fp8_smoke.yaml` | 3 | 512 | Qwen3-8B-Base | — |
| 3 | `fp8` | `1` | `dapo_qwen3_8b_ray_vllm_fp8_smoke.yaml` | 3 | 512 | Qwen3-8B-Base | — |
| 4 | `atomfp8` | `1` | `dapo_qwen3_8b_ray_atom_fp8_4k_smoke.yaml` | 3 | 4096 | Qwen3-8B-Base | — |
| 5 | `atombf16` | `0` | `dapo_qwen3_8b_ray_atom_bf16_4k_smoke.yaml` | 1 | 4096 | Qwen3-8B-Base | — |
| 6 | `bf16` | `0` | `dapo_qwen3moe_a3b_ray_vllm_verlref_4k_smoke.yaml` | 3 | 4096 | Qwen3-30B-A3B-Base | `LUMENRL_FP32_MOE_ROUTER=0` |
| 7 | `bf16` | `0` | `dapo_qwen3moe_a3b_ray_megatron_verlref_4k_smoke.yaml` | 3 | 4096 | Qwen3-30B-A3B-Base | `LUMENRL_FP32_MOE_ROUTER=0` |

这 6 条 config（例子 2 和 3 共用一条）都是 `logger.wandb_enabled: false`，
**不需要 wandb 账号**（见 §4.6）。
`STEPS` 列是命令行 `num_training_steps` 覆盖值，不是 yaml 里的默认值。

> ⚠️ **`MODE` 和 `CONFIG_OVERRIDE` 必须成对给，不能只改一个。**
> `MODE` 决定一组环境变量和一批**追加的 Hydra override**；`CONFIG_OVERRIDE` 只替换 yaml。
> 两者不匹配时最典型的表现就是例子 4：`MODE=atomfp8` 无条件追加
> `compilation_config.level=3`，配上一条 vLLM 的 smoke yaml 就得到
> `RuntimeError: aot_compile is not supported by the current configuration`。
> 详见 §4.5 的 `CONFIG_OVERRIDE` 语义说明。

### 2.3 几个例子之间的关系

- **例子 2 / 3 共用同一条 yaml**，只差 `TRAIN_FP8`：`0` 是「只有 rollout 量化」，
  `1` 追加 `LUMEN_FP8=1 LUMEN_FP8_SCALING=blockwise2d`，训练前向也走 FP8。
- **例子 5 是例子 4 的 BF16 对照组**：同一个 ATOM 引擎、同样的 no-eager level=3 + sleep2，
  只把 rollout 的在线量化和训练侧 FP8 一起关掉。
- **例子 7 是例子 6 的 Megatron 孪生**。`configs/` 里有两条 Megatron MoE smoke 可选，
  这里选 `dapo_qwen3moe_a3b_ray_megatron_verlref_4k_smoke.yaml` 而不是
  `dapo_qwen3moe_a3b_ray_megatron_smoke.yaml`，原因有三条：
  1. 它和例子 6 的 yaml **除了 `training_backend` 和 `megatron_cfg` 之外逐字段相同**
     （同样的 4096 response、同样的 global batch 128、同样的 16 generations、同样的数据），
     所以两者的指标可以直接相减，差值就是训练后端本身的差异；
     `megatron_smoke` 是 512 response，和例子 6 没法比。
  2. 拓扑 TP=1 / PP=1 / CP=1 / EP=8 使 DP=8，和 FSDP2 的 DP8 每卡看到的序列数一致。
  3. 它的 `checkpoint_dir` 是空串，不落盘（见下）。

**关于「两个训练后端不能共用 checkpoint 目录」**：这条限制仍然成立，
但**本文档给出的七条命令都不会踩到它**，不需要你手工隔离目录。实际路径是：

| 例子 | smoke config 的 `checkpoint_dir` | longrun config 的 `checkpoint_dir` |
|---|---|---|
| 1 | `$DATA_ROOT/ckpts/lumenrl-dapo/ray-vllm-smoke` | `$DATA_ROOT/ckpts/lumenrl-dapo/longrun-ray-vllm-8b` |
| 2 / 3 | `$DATA_ROOT/ckpts/lumenrl-dapo/ray-vllm-fp8-smoke` | `$DATA_ROOT/ckpts/lumenrl-dapo/longrun-ray-vllm-fp8-8b` |
| 4 | `""`（不落盘） | `$DATA_ROOT/ckpts/lumenrl-dapo/longrun-ray-atom-fp8-8b` |
| 5 | `""`（不落盘） | `$DATA_ROOT/ckpts/lumenrl-dapo/longrun-ray-atom-bf16-8b` |
| 6 | `""`（不落盘） | `$SCRATCH_ROOT/ckpts/lumenrl-dapo/verlref-moe-a3b-bf16` |
| 7 | `""`（不落盘） | `$SCRATCH_ROOT/ckpts/lumenrl-dapo/verlref-moe-a3b-megatron-ep8-4k` |

例子 6 和 7 的 smoke 都不落盘，longrun 的两个目录也互不相同，所以**例子 6 之后可以直接跑
例子 7**（实测连续跑过，见 §6 的记录）。只有在你自己把两个后端指到同一个
`checkpoint_dir` 时才会出问题：两种格式互不兼容，而且引擎按"占整卡比例"算 KV cache 预算。

---

## 3. 环境要求

- 8x gfx950（MI350X 或 MI355X），所有卡处于空闲
- 宿主机 ROCm 7.2，`/dev/kfd` 与 `/dev/dri` 可访问
- Docker（`docker` 命令要能用；需要 sudo 的话见 §4.4 的 `DOCKER` 变量）
- 磁盘：见下
- 模型与数据：约 **74 GB**，清单见 §4.2

### 3.1 磁盘（实测于 035 节点）

| 项目 | 实测值 | 怎么量的 |
|---|---|---|
| 镜像下载量（content store，压缩态） | **11.8 GB** | `docker image inspect <tag> --format '{{.Size}}'` = 11821982364 |
| 镜像解包后占盘 | **47.3 GB** | `docker system df -v` 的 `SIZE` 列 |
| 其中与 `vllm/vllm-openai-rocm:v0.23.0` 共享 | 46.69 GB | 同上 `SHARED SIZE` 列 |
| 本镜像独有 | 603.4 MB | 同上 `UNIQUE SIZE` 列 |

**建议预留**：

- 只跑 §2 七个 smoke：**镜像 60 GB**（47.3 GB 解包 + 11.8 GB 压缩层留在 content store）
  **+ 模型与数据 74 GB** ≈ 134 GB。七个例子一共用到 6 条 smoke config（例子 2 和 3 共用一条），
  其中 4 条 `checkpoint_dir: ""`，另外 2 条虽然给了目录但 `save_steps: 1000` 而只跑 3 步，
  所以 smoke **一份 checkpoint 都不写**。
- 长跑（`--longrun`）另需 checkpoint 空间。8B FSDP2 一份 checkpoint（fp32 权重 + optimizer）
  量级是模型的若干倍，30B-A3B 一份约 342 GB（见
  `lumenrl/trainer/callbacks.py` 里 `_prune_old_checkpoints` 的注释）。
  `save_total_limit` 决定同时保留几份，写入时峰值是一份。

> 旧版本这里写"镜像约需 120 GB"，是错的；上面的数字是在 035 上量出来的。
> **冷拉取耗时本节点未实测**（镜像已在本地）。可按下载 11.8 GB 估算：
> 千兆网约 2 分钟起，看你的仓库带宽。已有镜像时 `docker pull` 是秒级——
> 那是「已预热」的表现，不要当成冷机的预期。

### 3.2 只能跑 gfx950

> **这个镜像只能跑 gfx950。** TransformerEngine 和 Apex 是用
> `NVTE_ROCM_ARCH=gfx950` / `PYTORCH_ROCM_ARCH=gfx950` 编译的，烘入的 16 个 aiter
> kernel 也是在 gfx950 上构建的。JIT 产物的文件名里不带架构标识，所以在 gfx942
> （MI300X / MI308X / MI325X）上会被直接加载然后在运行期出错，而不会重新编译。
> 要支持 gfx942 需要单独构建一个镜像：`PYTORCH_ROCM_ARCH=gfx942 bash release/build_image.sh`。

---

## 4. 快速开始

### 4.1 获取镜像

```bash
docker pull zhangdanyangamd/lumen-rl:dapo-gfx950-rocm7.2.3-260902
```

也可以自己构建，下面就是全部步骤，没有额外的手工操作：

```bash
git clone -b <branch> <lumen-rl-repo> && cd Lumen-RL
bash release/build_image.sh                  # 45-60 分钟，大头是 TransformerEngine
TAG=lumenrl:release-$(date +%Y%m%d) bash release/precompile_kernels.sh
```

`precompile_kernels.sh` 需要 GPU：aiter 的 kernel 是首次使用时才编译的，而 `docker build`
没有设备可用。预先烘进镜像能省下实打实的时间——例子 4 的 smoke 在完全预热的镜像上是
**447 秒**，冷镜像上是 **1256 秒**，差距几乎全在
`module_gemm_a8w8_blockscale_bpreshuffle_cktile` 这一类大 kernel 上。合成 warmup 只能覆盖
16 个 kernel 里的 5 个，想全部覆盖见该脚本头部的说明。

### 4.2 准备数据（跑之前请照这张表自查）

```bash
export DATA_ROOT=/path/to/data
```

`$DATA_ROOT` 下**必须存在**这些东西，启动器的预检查的就是这张清单：

| 路径（相对 `$DATA_ROOT`） | 实测体积 | 谁需要 |
|---|---|---|
| `models/Qwen3-8B-Base/` | 16 GB | 例子 1–5，以及所有例子的 tokenizer |
| `models/Qwen3-30B-A3B-Base/` | 57 GB | 例子 6、7 |
| `data_cached/qwen3-8b-maxprompt1024/dapo-math-17k.filtered.parquet` | 1.02 GB（1069626101 B） | 全部（train） |
| `data_cached/qwen3-8b-maxprompt1024/aime-2024.filtered.parquet` | 892 KB（913104 B） | 全部（val） |
| `logs/` | — | 启动器自动创建 |

自查一条命令：

```bash
ls -la "$DATA_ROOT/models/Qwen3-8B-Base" \
       "$DATA_ROOT/data_cached/qwen3-8b-maxprompt1024"
du -sh "$DATA_ROOT/models"/*
```

#### 从零下载（两步，都在容器里跑）

先按 §4.4 把容器起起来，然后：

```bash
# 1) 模型与原始数据集
docker exec -e DATA_ROOT="$DATA_ROOT" lumenrl-release bash -lc '
python3 - <<PY
from huggingface_hub import snapshot_download
import os; D = os.environ["DATA_ROOT"]
snapshot_download("Qwen/Qwen3-8B-Base", local_dir=f"{D}/models/Qwen3-8B-Base",
                  allow_patterns=["*.json","*.txt","*.safetensors","*.model","tokenizer*"])
snapshot_download("BytedTsinghua-SIA/DAPO-Math-17k", repo_type="dataset",
                  local_dir=f"{D}/raw/DAPO-Math-17k")
snapshot_download("BytedTsinghua-SIA/AIME-2024", repo_type="dataset",
                  local_dir=f"{D}/raw/AIME-2024")
PY'

# 例子 6、7 追加（约 57 GB）
docker exec -e DATA_ROOT="$DATA_ROOT" lumenrl-release bash -lc '
hf download Qwen/Qwen3-30B-A3B-Base \
  --local-dir "$DATA_ROOT/models/Qwen3-30B-A3B-Base" --max-workers 8'
```

```bash
# 2) 过滤掉 prompt > 1024 token 的样本，产出上表的两个 parquet。
#    不过滤的话启动时会进入一段很长的 overlong-prompt 扫描。
docker exec -e DATA_ROOT="$DATA_ROOT" lumenrl-release bash -lc '
python3 - <<PY
import os, glob, datasets
from transformers import AutoTokenizer
D = os.environ["DATA_ROOT"]; MAXLEN = 1024
OUT = f"{D}/data_cached/qwen3-8b-maxprompt1024"
tok = AutoTokenizer.from_pretrained(f"{D}/models/Qwen3-8B-Base")
def first(g): return sorted(glob.glob(g, recursive=True))[0]
jobs = [(first(f"{D}/raw/DAPO-Math-17k/**/*.parquet"), f"{OUT}/dapo-math-17k.filtered.parquet"),
        (first(f"{D}/raw/AIME-2024/**/*.parquet"),     f"{OUT}/aime-2024.filtered.parquet")]
os.makedirs(OUT, exist_ok=True)
nproc = max(1, min(64, (os.cpu_count() or 8) // 4))
for src, dst in jobs:
    ds = datasets.Dataset.from_parquet(src); n0 = len(ds)
    ds = ds.filter(lambda d: len(tok.apply_chat_template(d["prompt"], add_generation_prompt=True,
                                                        tokenize=True)) <= MAXLEN, num_proc=nproc)
    ds.to_parquet(dst); print(src, "->", dst, n0, "->", len(ds))
PY'
```

> **数据只需过滤一次，七个例子共用。** Qwen3-8B-Base 与 Qwen3-30B-A3B-Base 的
> `tokenizer.json` / `vocab.json` / `merges.txt` 三个文件 md5 完全相同（vocab 151936），
> 所以按 8B tokenizer 过滤出的结果对 MoE 同样成立。
>
> **MoE 必须用 Base 版。** instruct/thinking 版的 Qwen3-30B-A3B 在 `max_response_length`
> 内永远不闭合 `</think>`，于是每条样本都被截断、reward 恒为 -1、`filter_groups`
> 连续 10 轮 kept 0，直接抛 `RuntimeError: filter_groups collected no valid groups`。
>
> 国内网络可换 ModelScope，repo ID 与本地路径都不变，见
> [`../examples/docs/03-data_cn.md`](../examples/docs/03-data_cn.md)。

### 4.3 跑第一个例子

```bash
export DATA_ROOT=/path/to/data
docker pull zhangdanyangamd/lumen-rl:dapo-gfx950-rocm7.2.3-260902
bash release/run_example.sh 1 --check
```

就这三条。第三条会打印

```
== example 1 — 8B BF16 baseline
   config examples/DAPO/configs/dapo_qwen3_8b_ray_vllm_smoke.yaml   steps 3   model Qwen3-8B-Base
   container lumenrl-release created
   GPU idle check: 0/8 cards busy, peak 0.3 GB in use
   log: /path/to/data/logs/example-1-20260903-050508.log
   ...
== example 1 finished: exit=0, wall clock 190s (3m10s)

===== CHECK: example 1 (8B BF16 baseline) =====
  Traceback      0
  OutOfMemory    0
  CUDA error     0
  HSA_STATUS     0
  step-1 metrics:
  rollout_corr/k3_kl     0.00110719   -0.3%_vs_0.00111_(tol_30%)         PASS
  entropy                0.605645     -1.8%_vs_0.617_(tol_15%)           PASS
  rollout_corr/kl        0.000748721  |x|in[9.3e-05,0.0093]              PASS
  rollout_corr/ppl_ratio 1.00106      (informational, not checked)       INFO
RESULT: PASS
```

换例子只要换那个数字：`bash release/run_example.sh 4 --check`。
**不要去改 `MODE` / `TRAIN_FP8`**——那两个变量和 config 必须成套，启动器已经替你配好了
（§2.2 的表就是它的内部表；`--dry-run` 可以把它展开成完整命令）。

### 4.4 这三条命令背后发生了什么

顺序是**清场 → 起容器 → 跑 → 判健康**。启动器全都做了，这里给出手工等价物，
因为出问题时你需要单独执行其中某一步。

#### 第 1 步 · 清场（**不要跳过**）

有过一次实测：前人留下的容器让 GPU 4–7 各占着约 90.9 GB，而 `docker ps` 在文档里
根本没出现过，于是排查方向全跑偏了。

```bash
docker ps -a                                                    # 别人的容器还在不在
docker exec lumenrl-release bash -lc 'rocm-smi --showmeminfo vram | grep -i used'
```

八张卡都应该在**空闲基线约 298 MB**（MI355X 实测 297766912–297832448 B）。
高于此值说明有同租户，或上一次运行留下了孤儿进程。启动器把这一步做成了硬门槛：
任何一张卡超过 2 GB 就拒绝启动，并打印三种可能原因和对应命令；
`--force` 表示"我知道，照跑"。

#### 第 2 步 · 起容器

```bash
docker run -d --name lumenrl-release \
  --network=host --ipc=host \
  --device=/dev/kfd --device=/dev/dri --group-add=video \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --shm-size 64G \
  -v "$DATA_ROOT":"$DATA_ROOT" -e DATA_ROOT="$DATA_ROOT" \
  zhangdanyangamd/lumen-rl:dapo-gfx950-rocm7.2.3-260902 sleep infinity
```

容器名不要用 `lumenrl`，太容易和别人撞；启动器默认 `lumenrl-release`，
可以用 `CONTAINER=... ` 换掉。容器**已存在时启动器会 `docker restart`**，
因为上一次运行结束后每张卡仍可能占着约 85 GB（§7 第 1 条），不重启下一次的 KV cache
预算会被压低。

容器启动时会打印固定的四个 SHA。验证软件栈：

```bash
docker exec lumenrl-release bash -lc 'python3 -c "
import aiter, lumen, lumenrl, vllm, flydsl, transformers
print(\"vllm\", vllm.__version__, \"flydsl\", flydsl.__version__, \"transformers\", transformers.__version__)
print(\"aiter\", aiter.__file__)"'
```

期望输出 `vllm 0.23.0 flydsl 0.3.2 transformers 5.12.0`，且 `aiter` 的路径在
`/opt/lumenrl/aiter/` 下。

> **`release/` 目录不在镜像里。** Dockerfile 只 `COPY` 了三个文件，四棵源码树是镜像自己按
> 固定 SHA clone 的，所以容器内 `/opt/lumenrl/Lumen-RL/release/` 不存在。
> `run_example.sh` 是**宿主机侧**脚本，从你手上的 `release/` 目录运行即可，不用进容器。

#### 第 3 步 · 跑

启动器用 `docker exec -d` 起任务，然后在宿主机侧轮询日志文件。日志路径是固定可预测的：

```
$DATA_ROOT/logs/example-<N>-<时间戳>.log          # 训练日志（run_dapo.sh 写的）
$DATA_ROOT/logs/example-<N>-<时间戳>.launcher.log  # 包装层 stdout + 退出码
```

`--log PATH` 可以自己指定。手工等价命令见**附录 A**（七段各自完整），
或者直接让启动器生成：

```bash
bash release/run_example.sh 4 --dry-run     # 只打印它将要执行的命令，不跑
```

#### 第 4 步 · 判健康

```bash
bash release/run_example.sh 4 --check-only --log $DATA_ROOT/logs/example-4-xxx.log
```

`--check` / `--check-only` 做四件事：抠出 step 1 的
`rollout_corr/k3_kl` / `entropy` / `rollout_corr/kl` / `rollout_corr/ppl_ratio`，
与内置参考值（§6 的表）逐项比对，统计
`Traceback` / `OutOfMemory` / `CUDA error` / `HSA_STATUS` 的出现次数，
然后打 `PASS` / `FAIL`。人工判读见 §6。

#### 启动器的全部选项

```bash
bash release/run_example.sh --help
```

| 选项 / 变量 | 作用 |
|---|---|
| `--check` | 跑完自动比对指标并打 PASS/FAIL |
| `--check-only --log PATH` | 不跑，只校验一份已有日志 |
| `--steps N` | 覆盖训练步数 |
| `--longrun` | 换成该例子的 longrun config（见 §4.6） |
| `--detach` | `docker exec -d` 起完就返回；受限集群上推荐（附录 B） |
| `--dry-run` | 只打印将要执行的 `docker run` / `docker exec` |
| `--force` | 节点不干净时自动清理而不是报错退出 |
| `--no-restart` | 不重启容器（想复用编译缓存时用） |
| `--keep-cache` | 例子 4 ↔ 5 切换时不清编译缓存（默认会清，见 §7） |
| `--verbose` | 前台把整份日志刷出来，而不是只刷关键行 |
| `DATA_ROOT` | **必填**。宿主机数据目录 |
| `IMAGE` / `CONTAINER` | 换镜像 tag / 容器名 |
| `DOCKER` | 例如 `DOCKER="sudo docker"`，当前用户不在 docker 组时用 |
| `EXTRA_OVERRIDE` | 追加任意 Hydra override，空格分隔，原样拼到命令行末尾 |
| `WANDB_API_KEY` | 只有 `--longrun` 需要 |
| `STALL_LIMIT` | 日志多少秒没长就判定卡死并退出，默认 2400 |

### 4.5 `CONFIG_OVERRIDE` 的两条语义（手工跑时必读）

`run_dapo.sh` 第 156 行是 `CONFIG="${CONFIG_OVERRIDE:-$CONFIG}"`，所以：

1. **`CONFIG_OVERRIDE` 覆盖 `MODE` 选出的默认 yaml**，
   但 **`MODE` 追加的那批 Hydra override 照样会加上去**。
   `MODE=atomfp8` / `atombf16` 会无条件追加
   `policy.generation.vllm_cfg.enforce_eager=false`、
   `policy.generation.atom_cfg.engine_kwargs.enforce_eager=false`、
   `policy.generation.atom_cfg.engine_kwargs.compilation_config.level=3`、
   `enable_sleep_mode=true`、`sleep_level=2`。
   把它们叠到一条 **vLLM** 的 smoke yaml 上，就是例子 4 那个
   `RuntimeError: aot_compile is not supported by the current configuration`
   ——一边要求 aot_compile，一边 torch.compile 其实没真开。
2. **路径是相对 `$RL_ROOT/Lumen-RL` 的**（脚本里 `cd "$RL_ROOT/Lumen-RL"`），
   所以写 `examples/DAPO/configs/xxx.yaml`。写成绝对路径会找不到。

另外两个容易被忽略的默认值：

- **`MODE` 不给 config 时选的是 longrun，不是 smoke。** `MODE=atomfp8` 默认
  `dapo_qwen3_8b_ray_atom_fp8_longrun.yaml`（`wandb_enabled: true`、
  `max_response_length: 20480`）。这是"照 §2 的表只改 `MODE`"会失败的第二个原因。
- **`MODEL_PATH` 默认是 8B。** 例子 6、7 不显式给就静默跑错模型。

`PYTORCH_CUDA_ALLOC_CONF=` 后面那个空值不是笔误：脚本用 `${VAR-default}` 取值，
只有显式传一个空串才能关掉 `expandable_segments`。

### 4.6 wandb

| | smoke config（§2.2 那 6 条） | longrun config（`--longrun`） |
|---|---|---|
| `logger.wandb_enabled` | `false` | `true` |
| 需要账号 | **不需要** | 需要 `WANDB_API_KEY` |
| `max_response_length` | 512 / 4096 | 20480（例子 7 是 4096） |

所以 §2 的七个例子**不需要 wandb 账号**。只有 `--longrun` 才会碰到它：

```bash
# 有 key
WANDB_API_KEY=xxxx bash release/run_example.sh 1 --longrun --detach

# 没有 key：关掉即可。启动器在检测到没有 key 时会自动加上这一条并打印提示。
EXTRA_OVERRIDE=logger.wandb_enabled=false bash release/run_example.sh 1 --longrun --detach
```

> **Hydra 键名是 `logger.wandb_enabled`，不是顶层 `wandb_enabled`。**
> 猜错会得到 `ConfigKeyError: Key 'wandb_enabled' not in 'LumenRLConfig'`。
>
> ⚠️ **缺 key 的失败点在 `RLTrainer.setup ... complete` 之后。**
> 前面几分钟 8 张卡满负荷、日志一路正常，看着像跑起来了，然后才抛
> `wandb.errors.errors.UsageError: No API key configured`。
> 别把这几分钟的正常输出当成"跑通了"。
>
> `run_dapo.sh` 还会自动读 `$RL_ROOT/wandb.key` 或 `$RL_ROOT/../wandb.key`
> （格式 `KEY=xxxx`），有这个文件时不用传环境变量。

---

## 5. 基于镜像做开发

镜像提供的是环境，你的代码不必待在里面。把四棵源码树里的任意一棵 bind-mount 覆盖上去
即刻生效——四个都是 editable 安装。

```bash
docker run -d --name lumenrl-dev ... \
  -v "$PWD/Lumen-RL":/opt/lumenrl/Lumen-RL \
  zhangdanyangamd/lumen-rl:dapo-gfx950-rocm7.2.3-260902 sleep infinity
```

然后用 `CONTAINER=lumenrl-dev bash release/run_example.sh <N>` 跑例子。

**如果覆盖的是 `aiter`，记得同时把 `AITER_JIT_DIR` 指到一个新目录。** 编译好的 kernel 与
产生它的 aiter revision 绑定，复用旧的会在 import 阶段失败，而报错里既不提 aiter 也不提
分支：

```
AttributeError: module 'aiter.jit.module_aiter_core' has no attribute 'MlaVersion'
```

```bash
-e AITER_JIT_DIR=/tmp/aiter-jit-mybranch
```

---

## 6. 怎么判断跑得健康

按顺序看这四条。每一条都有人在上面耗掉过一小时。

**1. 开跑前每张卡都在空闲基线**（MI355X 实测约 298 MB）。见 §4.4 第 1 步。

**2. 源码安装压过镜像自带的 wheel。** `import aiter` 必须解析到 `/opt/lumenrl/aiter/`，
而不是 site-packages。

**3. 运行中：** 日志里有 `RLTrainer.setup ... complete`、`filter_groups round N` 和逐步指标；
且没有 `Traceback`、`OutOfMemory`、`CUDA error`、`HSA_STATUS`。
`--check` 就是数这四个词的出现次数。

**4. 数值对齐。** 见下面的参考值表和判据。

### 6.1 参考值表

**测量条件**（表里每个数字都是在这套条件下测的，换任何一条都不适用）：

- 硬件 8x MI355X（gfx950），镜像 `zhangdanyangamd/lumen-rl:dapo-gfx950-rocm7.2.3-260902`
- 命令就是 `bash release/run_example.sh <N>`，即 §2.2 那张表的整行参数
- `seed=10086`（`run_dapo.sh` 硬编码），`STEPS` 与 `max_response_length` 见表内列
- 取的是**第 1 步**（`step=1`，1-based）的指标
- 测量日期 2026-09-03，节点 `crsuse2-m2m-v2-035`

| # | config（`examples/DAPO/configs/`） | steps | resp | 端到端墙钟 | `rollout_corr/k3_kl` | `entropy` | `rollout_corr/kl`（有符号） | 实测次数 |
|---|---|---|---|---|---|---|---|---|
| 1 | `dapo_qwen3_8b_ray_vllm_smoke.yaml` | 3 | 512 | 156–190 s | **0.00109** ±30% | **0.603** ±25% | 0.00097 | 5 |
| 2 | `dapo_qwen3_8b_ray_vllm_fp8_smoke.yaml` | 3 | 512 | 166 s | **0.00469** ±30% | **0.789** ±25% | 0.00468 | 1 |
| 3 | `dapo_qwen3_8b_ray_vllm_fp8_smoke.yaml`（`TRAIN_FP8=1`） | 3 | 512 | 176 s | **0.00410** ±30% | **0.812** ±25% | 0.00412 | 1 |
| 4 | `dapo_qwen3_8b_ray_atom_fp8_4k_smoke.yaml` | 3 | 4096 | 557–602 s | **0.00286** ±50% | **0.597** ±50% | 0.00268 | 3 |
| 5 | `dapo_qwen3_8b_ray_atom_bf16_4k_smoke.yaml` | 1 | 4096 | 472 s | **0.000988** ±50% | **0.641** ±50% | 0.000821 | 2 |
| 6 | `dapo_qwen3moe_a3b_ray_vllm_verlref_4k_smoke.yaml` | 3 | 4096 | 552–557 s | **0.00154** ±50% | **0.864** ±60% | 0.00178 | 2 |
| 7 | `dapo_qwen3moe_a3b_ray_megatron_verlref_4k_smoke.yaml` | 3 | 4096 | 526–532 s | **0.00157** ±50% | **0.655** ±60% | 0.00166 | 2 |

粗体两列带容差的就是 `--check` 判 PASS/FAIL 的两项，参考值是「实测次数」列那么多遍的**均值**，
容差的来历见 §6.2。`rollout_corr/kl` 一列只做数量级判据，不给百分比容差。
「端到端墙钟」含容器重启和预检，即 `run_example.sh` 自己报的耗时。
一共 16 次运行，**退出码全部为 0**，`Traceback` / `OutOfMemory` / `CUDA error` / `HSA_STATUS`
计数**全部为 0**，用上表容差 `--check` **16/16 全部 PASS**。
每一次运行的完整原始记录在 [`VALIDATION-20260903.md`](VALIDATION-20260903.md)。

### 6.2 怎么用这张表（重要）

**`rollout_corr/kl` 不能当复现判据。** 它是 `(rollout_logp - train_logp)` 的
**有符号** token 均值，对称的分歧在里面会互相抵消，剩下的基本是噪声。
用同一条命令、同一个 `seed=10086` 把**例子 1 跑五遍**，实测：

| | 第 1 遍 | 第 2 遍 | 第 3 遍 | 第 4 遍 | 第 5 遍 | 相对均值最大偏差 |
|---|---|---|---|---|---|---|
| `rollout_corr/kl` | 0.00122282 | 0.00046895 | 0.00110177 | 0.000748721 | 0.00133158 | **±52%**，最大与最小差 2.8 倍 |
| `rollout_corr/ppl_ratio` | 1.00109 | 0.999031 | 1.00072 | 1.00106 | 1.00193 | **偏离 1 的方向翻过符号** |
| `rollout_corr/k3_kl` | 0.00110339 | 0.00117701 | 0.00105992 | 0.00110719 | 0.000994479 | ±9% |
| `entropy` | 0.635074 | 0.623073 | 0.591768 | 0.605645 | 0.560073 | ±7% |

抖动来源在 rollout 侧：vLLM / ATOM 连续批处理每次分批不同，`seed` 固定不了这一层。
**response 越长抖得越厉害**，因为动态采样（`filter_groups`）会因此选出明显不同的一批 prompt。
七个例子各自实测到的**相对均值最大偏差**：

| 例子 | resp | 实测次数 | `k3_kl` | `entropy` |
|---|---|---|---|---|
| 1 | 512 | 5 | ±9% | ±7% |
| 4 | 4096 | 3 | ±17% | ±16% |
| 5 | 4096 | 2 | ±4% | ±4% |
| 6 | 4096 | 2 | ±8% | **±19%** |
| 7 | 4096 | 2 | ±2% | ±16% |

（例子 2、3 只测了一遍，容差沿用同为 512 的例子 1。）

所以判据是这样定的：

- **`rollout_corr/k3_kl` 是主判据。** 它是同一个 train/rollout 差异的 k3 估计量，
  非负、不抵消，七个例子里实测最大偏差只有 ±17%。
  容差 **512 组（例子 1/2/3）±30%，4096 组（例子 4–7）±50%**。
- **`entropy` 在 4k 上只是粗筛。** 容差 **512 组 ±25%，ATOM 4k（例子 4/5）±50%，
  MoE 4k（例子 6/7）±60%**。
  ⚠️ **例子 6 的 `entropy` 是全表最不可复现的一个数**：两遍实测 1.03019 和 0.698045，
  而同两遍的 `k3_kl` 只差 8%、`response_length/mean` 只差 0.7%（785.0 / 779.5）。
  entropy 是在 `filter_groups` 筛完后那 128 条序列上取的均值，MoE 上逐 prompt 差异很大，
  所以它抖得比什么都厉害。**判 MoE 复现看 `k3_kl`，不要看 entropy。**
- 容差都按「实测最大偏差的约 3 倍」取，留出样本只有 2–5 次导致的低估。
- **`rollout_corr/kl` 只做数量级判据**：`--check` 检查 |实测| 落在
  参考值的 **1/10 到 10 倍**之间。**高出一个数量级才算坏**，
  最常见的原因是某一侧没开 model-sensitive RMSNorm。
- **`rollout_corr/ppl_ratio` 只作参考**，不参与 PASS/FAIL。

两个 BF16 rollout（例子 1 的 0.00109 和例子 5 的 0.000988）的 mismatch 落在 1e-3 附近，
三个 FP8 的（例子 2 的 0.00469、3 的 0.00410、4 的 0.00286）是它的 3–4 倍——
这个差距是量化的代价，属于正常。两个 MoE（0.00154 / 0.00157）介于两者之间，
且 FSDP2 与 Megatron 只差 2%，说明换训练后端没有引入额外的 train/rollout 漂移。

> **对不上表时的排查顺序：先怀疑跑的不是同一条 yaml，再怀疑数值 bug。**
> `grep -m1 'CONFIG=' $DATA_ROOT/logs/example-<N>-*.launcher.log`
> 会打印这一次实际用的 config 文件名、`MODE`、`TRAIN_FP8`、`STEPS`。
> 配置错位和数值回归在指标上长得一模一样，但前者常见得多。

---

## 7. 已知问题

**1. 跑完之后显存不会自动释放。** smoke 干净结束后每张卡仍可能占着约 85 GB：Ray worker 已经
退出，但显存留在那里，而且从容器内部看不到对应进程。两次运行之间重启容器，否则下一次
运行的 KV cache 预算会被压低。

```bash
docker restart lumenrl-release   # 显存回到空闲基线
```

启动器每次启动前都会做这件事，不需要你记得。

**2. 切换 ATOM 精度必须清编译缓存。** torch 的 inductor 缓存不按运行隔离，
例子 4 之后直接跑例子 5（或反过来）会死在 AOTAutograd：

```bash
docker exec lumenrl-release bash -lc \
  'rm -rf /tmp/aiter_configs /tmp/atom_torch_compile_cache /tmp/torchinductor_root'
```

启动器记住上一次 ATOM 例子的精度，只在精度真的变了的时候清（所以同一个例子跑两遍
仍然能复用缓存）。`--keep-cache` 关掉这个行为。

**3. `pgrep` 判不了长任务的存活。** `docker exec` 起的进程和你的 shell 不共享进程树，
`pgrep` 跨会话恒返回 0，会诱使你再起一个实例。**看日志文件在不在增长**：

```bash
watch -n 30 'ls -l $DATA_ROOT/logs/example-4-xxx.log'
```

**4. `docker restart` 会杀掉 `--detach` 起的任务。** 这两条建议是冲突的（§7 第 1 条要重启，
`--detach` 要求别动容器）。启动器的处理是：重启前先检查上一次的日志还在不在增长，
在增长就拒绝启动并告诉你怎么办；`--force` 表示"就是要杀掉它"。

**5. 日志里刷 `waiting for baton release` 不是卡死。** 8 个训练 actor 在等其中一个 JIT 编译
kernel，靠 baton 锁串行。如果跳过了 `precompile_kernels.sh`，那约 20 分钟就花在这里。
启动器的 `STALL_LIMIT`（默认 2400 秒无新日志才放弃）就是为这一段留的余量。

**6. `RUN_ID` 里硬编码了 `-ray-vllm-8b-`。** 不显式给 `LOG` 的话，MoE / ATOM 例子的日志
文件名也会带这一段，很容易找错文件。启动器总是显式传 `LOG`，所以只有手工跑时才会遇到；
附录 A 的每段命令都带了 `LOG`。

**7. `HSA_DISABLE_FRAGMENT_ALLOCATOR` 在本镜像上保持默认值 1 是安全的。**
`run_dapo.sh` 默认导出 `HSA_DISABLE_FRAGMENT_ALLOCATOR=1`，而脚本自己的注释记录了：
在 **ROCm 7.14 / RCCL 2.28.9 / torch 2.12（`rocm/primus:v26.4`）** 上这个开关会打死
节点内 reduce-scatter，也就是 Megatron 分布式优化器归约梯度的方式，症状是一个**无关的
一元素分配**报 `CUDA error: invalid argument`（不要去查 `clip_grads`）。
**这个组合不是本发布镜像。** 例子 7（Megatron，`use_distributed_optimizer: true`）
在本镜像上按默认值 1 实测跑完 3 步，`CUDA error` / `HSA_STATUS` 计数为 0，
所以七个例子都不需要显式传空值。如果你把源码树挂到别的 ROCm/torch 组合上跑到了这个
故障，关掉它的写法是显式传空串：`-e HSA_DISABLE_FRAGMENT_ALLOCATOR=`。

---

## 8. 版本固定

复现一个结果意味着四个仓库要一起复现。它们**不能各自独立升级**，原因见下面关于 aiter 的说明。

| 组件 | 仓库 | 分支 | Commit |
|---|---|---|---|
| Lumen-RL | `ZhangDanyang-AMD/Lumen-RL` | `dev/dsv4-dapo` | `6957ee9c1c79` |
| Lumen | `ZhangDanyang-AMD/Lumen` | `amd-atom-rollout` | `e6379cbd9057` |
| aiter | `ZhangDanyang-AMD/aiter` | `lumen/moe` | `4ebe6d69c7f4` |
| ATOM | `xysheng-AMD/ATOM` | `lumen-rl` | `7173f5b8f758` |
| composable_kernel | aiter submodule | — | `af9e1d1f1ae3` |

底座镜像 `vllm/vllm-openai-rocm:v0.23.0`，另加 `flydsl 0.3.2`、`megatron-core 0.18.2`、
ROCm Apex `daed8525`、ROCm TransformerEngine `6e541a10`。

### 为什么 aiter 钉在 fork 上

`aiter/lumen/moe` 分叉自 `ROCm/aiter` main 的 `f2f8ed9b2`。那是
[ROCm/aiter#5149](https://github.com/ROCm/aiter/pull/5149) 把
[#4978](https://github.com/ROCm/aiter/pull/4978) 前一天刚合入的那批 Triton kernel
整体 revert 掉之前的最后一个提交——被 revert 的包括 `cross_entropy`（含 chunked 变体）、
`moe_aux_loss`、`moe_gemm_mxfp8`、`moe_gemm_per_token`、`fast_transpose`、`quant_mxfp8`、
`gemm_mxfp8`、`requant_fp8_row_to_col`，以及 gfx942 的 tune 表。Lumen 里大部分是直接
import 的，所以当前的 `ROCm/aiter` main 用不了。

同时这个分支带着较新的 `aiter.ops.shuffle` API（`interleave_gate_up_rows`、
`moe_shuffle_weight`），而 ATOM 从 2026-07-11 起就在模块顶层 import 它们。往回钉则会反过来
弄坏 ATOM。这个分支是唯一同时满足两边的点。

有两处 Lumen 专有的修改从未上游，这个分支上也没有：
`fused_rms_fp8_per_tensor_static_quant` 的 `output_amax` / `output_rsigma` 两个参数，
它们只被 Megatron 的 `LumenLayerNormLinear` FP8 融合路径使用。上面七个例子都走不到那条路径。

### 如果你要升级 aiter

`flydsl` 必须跟着一起升。底座镜像自带 0.1.4.2，而 `aiter/lumen/moe` 要求 `>= 0.2.4`
且钉的是 0.3.2。弄错的表现是：

```
ImportError: Unsupported `flydsl` version: expected >=`0.2.4`, got `0.1.8`.
```

而且它是从 ATOM 的 `model_ops/moe.py` 在 import 阶段抛出来的，报错里完全不提 aiter。

---

## 9. 延伸阅读

完整的运维手册——装依赖、准备数据、启动、排障，以及双节点训推分离部署——在
[`../examples/`](../examples/README_cn.md)。

---

## 附录 A：不用 launcher 的等价手工命令

这七段是 `run_example.sh <N> --dry-run` 的输出，一段一个例子，**互不共用变量**：
每段都自带全部 `MODE` / `TRAIN_FP8` / `CONFIG_OVERRIDE` / `STEPS` / `MODEL_PATH` / `LOG`。
用 `docker exec -e` 传环境变量是有意的：这样就不需要在 `bash -lc "..."` 里嵌套引号，
而嵌套引号是 `CONFIG_OVERRIDE` 被吃掉、脚本路径被拆碎的主要来源
（在 `spur exec bash -lc '…'` 里手写要转义三层）。

先照 §4.4 第 2 步把容器起好（下面统一叫 `lumenrl-release`），并且

```bash
export DATA_ROOT=/path/to/data
```

日志不会走 stdout，`run_dapo.sh` 直接写到 `$LOG`；跟踪用
`tail -f "$LOG"`，抠指标用
`grep -o 'step=[0-9]* .*rollout_corr/kl=[^ ]*' "$LOG"`。
**每段之间都要 `docker restart lumenrl-release`**（§7 第 1 条），
**例子 4 和例子 5 之间还要清编译缓存**（§7 第 2 条）。

### 例子 1 · 8B BF16 基线

```bash
docker exec \
  -e RL_ROOT=/opt/lumenrl \
  -e DATA_ROOT=/path/to/data \
  -e SCRATCH_ROOT=/path/to/data \
  -e PYTORCH_CUDA_ALLOC_CONF= \
  -e MODE=bf16 \
  -e TRAIN_FP8=0 \
  -e STEPS=3 \
  -e CONFIG_OVERRIDE=examples/DAPO/configs/dapo_qwen3_8b_ray_vllm_smoke.yaml \
  -e MODEL_PATH=/path/to/data/models/Qwen3-8B-Base \
  -e LOG=/path/to/data/logs/example-1.log \
  lumenrl-release bash -lc 'bash /opt/lumenrl/Lumen-RL/examples/DAPO/run_dapo.sh'
```

### 例子 2 · 8B FP8 rollout

```bash
docker exec \
  -e RL_ROOT=/opt/lumenrl \
  -e DATA_ROOT=/path/to/data \
  -e SCRATCH_ROOT=/path/to/data \
  -e PYTORCH_CUDA_ALLOC_CONF= \
  -e MODE=fp8 \
  -e TRAIN_FP8=0 \
  -e STEPS=3 \
  -e CONFIG_OVERRIDE=examples/DAPO/configs/dapo_qwen3_8b_ray_vllm_fp8_smoke.yaml \
  -e MODEL_PATH=/path/to/data/models/Qwen3-8B-Base \
  -e LOG=/path/to/data/logs/example-2.log \
  lumenrl-release bash -lc 'bash /opt/lumenrl/Lumen-RL/examples/DAPO/run_dapo.sh'
```

### 例子 3 · 8B FP8 端到端

```bash
docker exec \
  -e RL_ROOT=/opt/lumenrl \
  -e DATA_ROOT=/path/to/data \
  -e SCRATCH_ROOT=/path/to/data \
  -e PYTORCH_CUDA_ALLOC_CONF= \
  -e MODE=fp8 \
  -e TRAIN_FP8=1 \
  -e STEPS=3 \
  -e CONFIG_OVERRIDE=examples/DAPO/configs/dapo_qwen3_8b_ray_vllm_fp8_smoke.yaml \
  -e MODEL_PATH=/path/to/data/models/Qwen3-8B-Base \
  -e LOG=/path/to/data/logs/example-3.log \
  lumenrl-release bash -lc 'bash /opt/lumenrl/Lumen-RL/examples/DAPO/run_dapo.sh'
```

### 例子 4 · 8B ATOM FP8

> 这条 config 是 **ATOM** 的 4k smoke。拿例子 1 的 vLLM yaml 改 `MODE=atomfp8`
> 会得到 `RuntimeError: aot_compile is not supported by the current configuration`。

```bash
docker exec \
  -e RL_ROOT=/opt/lumenrl \
  -e DATA_ROOT=/path/to/data \
  -e SCRATCH_ROOT=/path/to/data \
  -e PYTORCH_CUDA_ALLOC_CONF= \
  -e MODE=atomfp8 \
  -e TRAIN_FP8=1 \
  -e STEPS=3 \
  -e CONFIG_OVERRIDE=examples/DAPO/configs/dapo_qwen3_8b_ray_atom_fp8_4k_smoke.yaml \
  -e MODEL_PATH=/path/to/data/models/Qwen3-8B-Base \
  -e LOG=/path/to/data/logs/example-4.log \
  lumenrl-release bash -lc 'bash /opt/lumenrl/Lumen-RL/examples/DAPO/run_dapo.sh'
```

### 例子 5 · 8B ATOM BF16

> 跑这一段之前，如果上一次跑的是例子 4，先清编译缓存（§7 第 2 条）：
> `docker exec lumenrl-release bash -lc 'rm -rf /tmp/aiter_configs /tmp/atom_torch_compile_cache /tmp/torchinductor_root'`

```bash
docker exec \
  -e RL_ROOT=/opt/lumenrl \
  -e DATA_ROOT=/path/to/data \
  -e SCRATCH_ROOT=/path/to/data \
  -e PYTORCH_CUDA_ALLOC_CONF= \
  -e MODE=atombf16 \
  -e TRAIN_FP8=0 \
  -e STEPS=1 \
  -e CONFIG_OVERRIDE=examples/DAPO/configs/dapo_qwen3_8b_ray_atom_bf16_4k_smoke.yaml \
  -e MODEL_PATH=/path/to/data/models/Qwen3-8B-Base \
  -e LOG=/path/to/data/logs/example-5.log \
  lumenrl-release bash -lc 'bash /opt/lumenrl/Lumen-RL/examples/DAPO/run_dapo.sh'
```

### 例子 6 · MoE FSDP2

> `MODEL_PATH` 必须显式给（默认是 8B）；`LUMENRL_FP32_MOE_ROUTER=0` 必须给，
> 这样 FSDP2 和 vLLM 才会舍入到同一组 top-8 专家，日志里应看到
> `MoE router patched on 48 gates (fp32=False)`。
> `SCRATCH_ROOT` 只被 MoE 的 **longrun** config 用来解析 `checkpoint_dir`
> （`${oc.env:SCRATCH_ROOT}`），smoke 用不到，但一起导出更省心。

```bash
docker exec \
  -e RL_ROOT=/opt/lumenrl \
  -e DATA_ROOT=/path/to/data \
  -e SCRATCH_ROOT=/path/to/data \
  -e PYTORCH_CUDA_ALLOC_CONF= \
  -e MODE=bf16 \
  -e TRAIN_FP8=0 \
  -e STEPS=3 \
  -e CONFIG_OVERRIDE=examples/DAPO/configs/dapo_qwen3moe_a3b_ray_vllm_verlref_4k_smoke.yaml \
  -e MODEL_PATH=/path/to/data/models/Qwen3-30B-A3B-Base \
  -e LOG=/path/to/data/logs/example-6.log \
  -e LUMENRL_FP32_MOE_ROUTER=0 \
  lumenrl-release bash -lc 'bash /opt/lumenrl/Lumen-RL/examples/DAPO/run_dapo.sh'
```

### 例子 7 · MoE Megatron EP=8

> `MODEL_PATH` 必须显式给（默认是 8B）；`LUMENRL_FP32_MOE_ROUTER=0` 必须给，
> 这样 FSDP2 和 vLLM 才会舍入到同一组 top-8 专家，日志里应看到
> `MoE router patched on 48 gates (fp32=False)`。
> `SCRATCH_ROOT` 只被 MoE 的 **longrun** config 用来解析 `checkpoint_dir`
> （`${oc.env:SCRATCH_ROOT}`），smoke 用不到，但一起导出更省心。

```bash
docker exec \
  -e RL_ROOT=/opt/lumenrl \
  -e DATA_ROOT=/path/to/data \
  -e SCRATCH_ROOT=/path/to/data \
  -e PYTORCH_CUDA_ALLOC_CONF= \
  -e MODE=bf16 \
  -e TRAIN_FP8=0 \
  -e STEPS=3 \
  -e CONFIG_OVERRIDE=examples/DAPO/configs/dapo_qwen3moe_a3b_ray_megatron_verlref_4k_smoke.yaml \
  -e MODEL_PATH=/path/to/data/models/Qwen3-30B-A3B-Base \
  -e LOG=/path/to/data/logs/example-7.log \
  -e LUMENRL_FP32_MOE_ROUTER=0 \
  lumenrl-release bash -lc 'bash /opt/lumenrl/Lumen-RL/examples/DAPO/run_dapo.sh'
```


## 附录 B：在受限 / 有调度器的集群上运行

上面的命令假设你能直接在 GPU 宿主机上敲命令。本发布版实际验证所用的集群
（Crusoe m2m + spur 调度器）不是这样，这里把差异列出来，因为它们大概率也适用于你的环境。

**1. 唯一入口是调度器的 `exec`，不能 ssh 计算节点**（`sshd` 侧
`AllowUsers ubuntu root` 会拒绝普通用户）：

```bash
export SPUR_CONTROLLER_ADDR=http://<controller-host>:6817
spur exec <JobID> bash -lc '<你的命令>'
```

`release/run_example.sh` 在这种环境下照常工作——它只需要一个能跑 `docker` 的 shell：

```bash
spur exec <JobID> bash -lc '
  mkdir -p /tmp/.docker; export DOCKER_CONFIG=/tmp/.docker
  cd /path/to/Lumen-RL
  DATA_ROOT=/path/to/data bash release/run_example.sh 1 --check'
```

**2. `HOME` 不可写。** 容器外的 `HOME=/root/spur` 只读，每条命令都会打印一句
`bash: /root/spur/.bash_profile: Permission denied`——无害，但有两个后果：
用 docker 前必须先

```bash
mkdir -p /tmp/.docker; export DOCKER_CONFIG=/tmp/.docker
```

否则 Docker 写不了 `$HOME/.docker/config.json` 会刷权限警告；
并且脚本里**一律写绝对路径**，不要依赖 `~`。

**3. `exec` 的 stdout 是攒到最后一起吐的**，不是流式的。所以前台模式在这种环境下
看不到实时进度（跑完才一次性出来）。长任务用 `--detach`：

```bash
spur exec <JobID> bash -lc '
  mkdir -p /tmp/.docker; export DOCKER_CONFIG=/tmp/.docker
  cd /path/to/Lumen-RL
  DATA_ROOT=/path/to/data bash release/run_example.sh 1 --longrun --detach'
```

然后**轮询日志文件大小**判存活（`pgrep` 不可用，§7 第 3 条）：

```bash
spur exec <JobID> bash -lc 'ls -l /path/to/data/logs/example-1-*.log'
```

**4. 别把长任务放在 `exec` 前台。** 客户端一断进程就被打死，还会留下占着显存的孤儿容器
（正是 §4.4 第 1 步要清的东西）。要么 `--detach`，要么在节点上
`setsid nohup ... &` 把启动器自己也放到后台。

**5. `docker restart` 与 `--detach` 的冲突**见 §7 第 4 条：启动器会先探测上一次的日志
是否还在增长，再决定是拒绝还是重启。
