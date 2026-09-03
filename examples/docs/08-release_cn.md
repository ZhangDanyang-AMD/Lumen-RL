> [Examples README](../README_cn.md) > 用发布镜像跑

# 8. 用发布镜像跑七个例子

> English version: [08-release.md](08-release.md)

这一章用**已发布的容器镜像**跑 §8.2 的七个例子：软件栈已固定、aiter kernel 已预编译，
每个例子一条命令，不需要安装任何依赖，也不需要打开 config 或启动脚本。
第 [1](01-env-setup_cn.md)–[4](04-launching_cn.md) 章是另一条路——从源码搭一套环境，
要改代码、换模型或跑双节点（例子 8，见 [第 7 章](07-disaggregated-rdma_cn.md)）时用那一条。

> ⚠️ **本镜像仅支持 AMD gfx950 架构**（Instinct MI350X / MI355X），需要 8 张卡。
> 详见 §8.3.1。

```bash
export DATA_ROOT=/path/to/data
docker pull zhangdanyangamd/lumen-rl:dapo-gfx950-rocm7.2.3-260902
bash release/run_example.sh 1 --check
```

三条命令跑完第一个例子并自动判定结果是否正确。换例子只改最后那个数字。

---

## 8.1 这个镜像包含什么

| | |
|---|---|
| 任务 | DAPO 数学 RL（GRPO 风格，按 uid 分组归一化） |
| 模型 | Qwen3-8B-Base（dense）、Qwen3-30B-A3B-Base（MoE，128 专家） |
| 训练后端 | Lumen FSDP2（BF16 / FP8 blockwise2d）、Megatron-Native（EP=8） |
| Rollout 引擎 | vLLM 0.23.0（BF16 / `fp8_per_block`）、ATOM（BF16 / `per_block_fp8`） |
| 拓扑 | 单个 Ray driver 进程内 8 个训练 actor + 8 个同卡 colocated rollout replica（TP=1） |
| 权重同步 | ZMQ CUDA-IPC，同卡直传，带覆盖率断言 |
| 硬件 | **仅 AMD gfx950（MI350X / MI355X），8 卡** |

算法侧：clip-higher + dual-clip + token-mean 策略损失、动态采样（`filter_groups`）、
overlong 奖励缓冲、TIS rollout 修正。

镜像里的 aiter kernel 已**全部预编译完成**（16 个对象），首次运行不会再花时间编译：

```bash
docker run --rm --entrypoint /bin/bash \
  zhangdanyangamd/lumen-rl:dapo-gfx950-rocm7.2.3-260902 \
  -lc 'ls /opt/lumenrl/aiter-jit/*.so | wc -l'     # 16
```

### 8.1.1 版本固定

复现一个结果需要四个仓库一起复现，它们**不能各自独立升级**。

| 组件 | 仓库 | 分支 | Commit |
|---|---|---|---|
| Lumen-RL | `ZhangDanyang-AMD/Lumen-RL` | `dev/dsv4-dapo` | `6957ee9c1c79` |
| Lumen | `ZhangDanyang-AMD/Lumen` | `amd-atom-rollout` | `e6379cbd9057` |
| aiter | `ZhangDanyang-AMD/aiter` | `lumen/moe` | `4ebe6d69c7f4` |
| ATOM | `xysheng-AMD/ATOM` | `lumen-rl` | `7173f5b8f758` |
| composable_kernel | aiter submodule | — | `af9e1d1f1ae3` |

底座镜像 `vllm/vllm-openai-rocm:v0.23.0`，另加 `flydsl 0.3.2`、`megatron-core 0.18.2`、
ROCm Apex `daed8525`、ROCm TransformerEngine `6e541a10`。
完整清单见 [`release/versions.env`](../../release/versions.env)。

容器启动时会打印这四个 SHA。验证软件栈：

```bash
docker exec lumenrl-release bash -lc 'python3 -c "
import aiter, lumen, lumenrl, vllm, flydsl, transformers
print(vllm.__version__, flydsl.__version__, transformers.__version__)
print(aiter.__file__)"'
```

期望 `0.23.0 0.3.2 5.12.0`，且 `aiter` 解析到 `/opt/lumenrl/aiter/` 下。

---

## 8.2 七个例子

七个例子的训练与推理**都跑在同一组 8 张卡上**。

### 8.2.1 概览

| # | 例子 | 训练 | Rollout | 命令 |
|---|------|------|---------|------|
| 1 | 8B BF16 基线 | FSDP2 BF16 | vLLM BF16 | `bash release/run_example.sh 1 --check` |
| 2 | 8B FP8 rollout | FSDP2 BF16 | vLLM `fp8_per_block` | `bash release/run_example.sh 2 --check` |
| 3 | 8B FP8 端到端 | FSDP2 **FP8 blockwise2d** | vLLM `fp8_per_block` | `bash release/run_example.sh 3 --check` |
| 4 | 8B ATOM FP8 | FSDP2 **FP8 blockwise2d** | **ATOM** `per_block_fp8` | `bash release/run_example.sh 4 --check` |
| 5 | 8B ATOM BF16 | FSDP2 BF16 | **ATOM** BF16 | `bash release/run_example.sh 5 --check` |
| 6 | MoE FSDP2 | FSDP2 BF16 | vLLM BF16 | `bash release/run_example.sh 6 --check` |
| 7 | MoE Megatron EP=8 | **Megatron** TP=PP=CP=1，EP=8，DP=8 | vLLM BF16 | `bash release/run_example.sh 7 --check` |

- 例子 2 / 3 共用一条 config，只差 `TRAIN_FP8`：`0` 只量化 rollout，`1` 训练前向也走 FP8。
- 例子 5 是例子 4 的 BF16 对照组：同一个 ATOM 引擎，只关掉 rollout 在线量化与训练侧 FP8。
- 例子 7 是例子 6 的 Megatron 孪生：两条 config 除 `training_backend` 与 `megatron_cfg`
  外逐字段相同，拓扑 EP=8 使 DP=8 与 FSDP2 一致，所以两者指标可直接相减，
  差值就是训练后端本身的差异。

### 8.2.2 每个例子的完整参数

启动器内部就是这张表。用手工命令（§8.4.5）时这一行的每一列都要给全。

| # | `MODE` | `TRAIN_FP8` | `CONFIG_OVERRIDE`（都在 `examples/DAPO/configs/` 下） | `STEPS` | `max_response_length` | 模型 | 额外 env |
|---|---|---|---|---|---|---|---|
| 1 | `bf16` | `0` | `dapo_qwen3_8b_ray_vllm_smoke.yaml` | 3 | 512 | Qwen3-8B-Base | — |
| 2 | `fp8` | `0` | `dapo_qwen3_8b_ray_vllm_fp8_smoke.yaml` | 3 | 512 | Qwen3-8B-Base | — |
| 3 | `fp8` | `1` | `dapo_qwen3_8b_ray_vllm_fp8_smoke.yaml` | 3 | 512 | Qwen3-8B-Base | — |
| 4 | `atomfp8` | `1` | `dapo_qwen3_8b_ray_atom_fp8_4k_smoke.yaml` | 3 | 4096 | Qwen3-8B-Base | — |
| 5 | `atombf16` | `0` | `dapo_qwen3_8b_ray_atom_bf16_4k_smoke.yaml` | 1 | 4096 | Qwen3-8B-Base | — |
| 6 | `bf16` | `0` | `dapo_qwen3moe_a3b_ray_vllm_verlref_4k_smoke.yaml` | 3 | 4096 | Qwen3-30B-A3B-Base | `LUMENRL_FP32_MOE_ROUTER=0` |
| 7 | `bf16` | `0` | `dapo_qwen3moe_a3b_ray_megatron_verlref_4k_smoke.yaml` | 3 | 4096 | Qwen3-30B-A3B-Base | `LUMENRL_FP32_MOE_ROUTER=0` |

> ⚠️ **`MODE` 与 `CONFIG_OVERRIDE` 必须成对给出。** `MODE` 除了选择环境变量，还会**追加一批
> Hydra override**，`CONFIG_OVERRIDE` 只替换 config 文件而不会取消这些追加项。
> 两者不匹配的典型后果：`MODE=atomfp8` 会无条件追加 `compilation_config.level=3`，
> 配一条 vLLM 的 config 就报
> `RuntimeError: aot_compile is not supported by the current configuration`。
> 使用启动器时这一层已经配好，无需关心。

这七条 config 都是 `logger.wandb_enabled: false`，**不需要 wandb 账号**（见 §8.4.6）。
`STEPS` 是命令行 `num_training_steps` 的覆盖值。七条 smoke config 都不写 checkpoint，
所以例子之间可以任意顺序连续运行。

---

## 8.3 环境要求

### 8.3.1 硬件与驱动

- **8 张 AMD gfx950**（Instinct MI350X 或 MI355X），全部处于空闲
- 宿主机 ROCm 7.2，`/dev/kfd` 与 `/dev/dri` 可访问
- Docker（若当前用户不在 docker 组，见 §8.4.4 的 `DOCKER` 变量）

> ⚠️ **本镜像仅能在 gfx950 上运行。** TransformerEngine 与 Apex 是以
> `NVTE_ROCM_ARCH=gfx950` / `PYTORCH_ROCM_ARCH=gfx950` 编译的，预编译进镜像的 16 个
> aiter kernel 也是在 gfx950 上构建的。这些 JIT 产物的文件名不含架构标识，因此在
> gfx942（MI300X / MI308X / MI325X）上会被直接加载并在运行期出错，而不会重新编译。
> 需要 gfx942 请单独构建：`PYTORCH_ROCM_ARCH=gfx942 bash release/build_image.sh`。

### 8.3.2 磁盘

| 项目 | 实测值 |
|---|---|
| 镜像下载量（压缩层） | **11.8 GB** |
| 镜像解包后占盘 | **47.3 GB** |
| 其中与 `vllm/vllm-openai-rocm:v0.23.0` 共享 | 46.69 GB |
| 本镜像独有 | 603.4 MB |

**建议预留**：镜像 60 GB（解包 47.3 GB + 压缩层 11.8 GB 留在 content store）
+ 模型与数据 74 GB ≈ **134 GB**。七个例子都是 smoke，不写 checkpoint。
若要长跑（`--longrun`）另需 checkpoint 空间——30B-A3B 的一份 FSDP2 checkpoint
（fp32 权重 + optimizer）约 342 GB，`save_total_limit` 决定同时保留几份。

> 用 `docker system df -v` 核对上表时**按 IMAGE ID 匹配，不要按 REPOSITORY/TAG 匹配**：
> 同一镜像本地可能挂着别的 tag。取 ID：`docker image inspect <tag> --format '{{.Id}}'`。

### 8.3.3 模型与数据

`$DATA_ROOT` 下必须存在下列内容，启动器的预检查的就是这张清单：

| 路径（相对 `$DATA_ROOT`） | 体积 | 谁需要 |
|---|---|---|
| `models/Qwen3-8B-Base/` | 16 GB | 例子 1–5，以及所有例子的 tokenizer |
| `models/Qwen3-30B-A3B-Base/` | 57 GB | 例子 6、7 |
| `data_cached/qwen3-8b-maxprompt1024/dapo-math-17k.filtered.parquet` | 1.02 GB | 全部（train） |
| `data_cached/qwen3-8b-maxprompt1024/aime-2024.filtered.parquet` | 892 KB | 全部（val） |
| `logs/` | — | 启动器自动创建 |

从零准备见 §8.4.3。

---

## 8.4 使用

### 8.4.1 开跑前确认卡是空闲的

```bash
docker ps -a                                    # 有没有别人的容器还在占卡
rocm-smi --showmeminfo vram | grep -i used      # 宿主机上直接可用
```

八张卡都应处于**空闲基线约 298 MB**（MI355X 实测 297766912–297832448 B）。
高于此值说明有同租户，或上一次运行留下了孤儿进程。
启动器把这一步做成了硬门槛：任何一张卡超过 2 GB 就拒绝启动并给出处置建议，
`--force` 可跳过。

### 8.4.2 获取镜像

```bash
docker pull zhangdanyangamd/lumen-rl:dapo-gfx950-rocm7.2.3-260902
```

也可以自行构建，下面就是全部步骤：

```bash
git clone -b <branch> <lumen-rl-repo> && cd Lumen-RL
bash release/build_image.sh                  # 45–60 分钟，大头是 TransformerEngine
TAG=lumenrl:release-$(date +%Y%m%d) bash release/precompile_kernels.sh
```

`precompile_kernels.sh` 需要 GPU：aiter kernel 只在首次使用时编译，而 `docker build`
没有设备可用，所以要用带卡的容器把它们编译好再提交进镜像。该脚本的合成 warmup 覆盖
16 个 kernel 中的 5 个，覆盖全部 16 个的做法见脚本头部说明。

### 8.4.3 准备数据

```bash
export DATA_ROOT=/path/to/data
```

按 §8.3.3 的清单自查。从零下载分两步，都在容器里执行（先按 §8.4.4 起好容器）：

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
# 2) 过滤掉 prompt > 1024 token 的样本，产出 §8.3.3 的两个 parquet
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

> 数据只需过滤一次，七个例子共用：两个模型的 `tokenizer.json` / `vocab.json` /
> `merges.txt` md5 相同（vocab 151936），按 8B tokenizer 过滤的结果对 MoE 同样成立。
>
> **MoE 必须用 Base 版。** instruct / thinking 版的 Qwen3-30B-A3B 在
> `max_response_length` 内不会闭合 `</think>`，导致每条样本被截断、reward 恒为 -1、
> `filter_groups` 连续 10 轮为空，最终抛
> `RuntimeError: filter_groups collected no valid groups`。
>
> 国内网络可换 ModelScope，repo ID 与本地路径不变，见
> [`03-data_cn.md`](03-data_cn.md)。

### 8.4.4 启动器

`release/run_example.sh` 是宿主机侧脚本，负责检查卡是否空闲、管理容器、拼好全部环境变量、
落地可预测的日志路径，并在跑完后把指标与内置参考值逐项比对。

```bash
bash release/run_example.sh <1..7> [选项]
bash release/run_example.sh --help
```

容器由启动器创建并复用，默认名 `lumenrl-release`；已存在时会先 `docker restart`
（上一次运行结束后每张卡可能仍占着约 90.9 GB，见 §8.5.2）。等价的手工建容器命令：

```bash
docker run -d --name lumenrl-release \
  --network=host --ipc=host \
  --device=/dev/kfd --device=/dev/dri --group-add=video \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --shm-size 64G \
  -v "$DATA_ROOT":"$DATA_ROOT" -e DATA_ROOT="$DATA_ROOT" \
  zhangdanyangamd/lumen-rl:dapo-gfx950-rocm7.2.3-260902 sleep infinity
```

日志路径固定：

```
$DATA_ROOT/logs/example-<N>-<时间戳>.log           # 训练日志
$DATA_ROOT/logs/example-<N>-<时间戳>.launcher.log  # 包装层输出与退出码
```

| 选项 / 变量 | 作用 |
|---|---|
| `--check` | 跑完自动比对指标并给出 PASS / FAIL |
| `--check-only --log PATH` | 不运行，只校验一份已有日志 |
| `--steps N` | 覆盖训练步数 |
| `--longrun` | 换成该例子的 longrun config（见 §8.4.6） |
| `--detach` | 起完即返回，适合长任务；并打印判存活的方法 |
| `--dry-run` | 只打印将要执行的命令，不运行 |
| `--force` | 卡不空闲或有残留容器时自动清理而非报错退出 |
| `--no-restart` | 不重启容器（复用编译缓存时用） |
| `--keep-cache` | 例子 4 ↔ 5 切换时不清编译缓存（默认会清，见 §8.5.2） |
| `--verbose` | 前台输出完整日志而非关键行 |
| `DATA_ROOT` | **必填**，宿主机数据目录 |
| `IMAGE` / `CONTAINER` | 更换镜像 tag / 容器名 |
| `DOCKER` | 例如 `DOCKER="sudo docker"` |
| `EXTRA_OVERRIDE` | 追加任意 Hydra override，空格分隔 |
| `WANDB_API_KEY` | 仅 `--longrun` 需要 |
| `STALL_LIMIT` | 日志静默多少秒判定卡死，默认 2400 |

用自己的代码覆盖镜像（四棵源码树都是 editable 安装）：

```bash
docker run -d --name lumenrl-dev ... \
  -v "$PWD/Lumen-RL":/opt/lumenrl/Lumen-RL \
  zhangdanyangamd/lumen-rl:dapo-gfx950-rocm7.2.3-260902 sleep infinity
```

然后 `CONTAINER=lumenrl-dev bash release/run_example.sh <N>`。

### 8.4.5 不用启动器的手工命令

下面是例子 1 的完整命令。换其他例子时，按 §8.2.2 的表替换 `MODE`、`TRAIN_FP8`、
`CONFIG_OVERRIDE`、`STEPS`、`MODEL_PATH`，例子 6、7 再追加 `-e LUMENRL_FP32_MOE_ROUTER=0`。
`bash release/run_example.sh <N> --dry-run` 会直接生成对应例子的这段命令。

```bash
export DATA_ROOT=/path/to/data

docker exec \
  -e RL_ROOT=/opt/lumenrl \
  -e DATA_ROOT=$DATA_ROOT \
  -e SCRATCH_ROOT=$DATA_ROOT \
  -e PYTORCH_CUDA_ALLOC_CONF= \
  -e MODE=bf16 \
  -e TRAIN_FP8=0 \
  -e STEPS=3 \
  -e CONFIG_OVERRIDE=examples/DAPO/configs/dapo_qwen3_8b_ray_vllm_smoke.yaml \
  -e MODEL_PATH=$DATA_ROOT/models/Qwen3-8B-Base \
  -e LOG=$DATA_ROOT/logs/example-1.log \
  lumenrl-release bash -lc 'bash /opt/lumenrl/Lumen-RL/examples/DAPO/run_dapo.sh'
```

日志不走 stdout，`run_dapo.sh` 直接写入 `$LOG`；跟踪用 `tail -f "$LOG"`，
抠指标用 `grep -o 'step=[0-9]* .*rollout_corr/kl=[^ ]*' "$LOG"`。
每段之间需 `docker restart lumenrl-release`，例子 4 与 5 之间还需清编译缓存（§8.5.2）。

四个容易漏掉的点：

- `CONFIG_OVERRIDE` 的路径**相对 `$RL_ROOT/Lumen-RL`**，写绝对路径找不到。
- 不给 `CONFIG_OVERRIDE` 时，`MODE` 选中的是 **longrun** config
  （`wandb_enabled: true`、`max_response_length: 20480`），不是 smoke。
- `MODEL_PATH` 默认是 8B，例子 6、7 不显式给会静默跑错模型。
- `PYTORCH_CUDA_ALLOC_CONF=` 后面的空值不是笔误：只有显式传空串才能关掉
  `expandable_segments`。

### 8.4.6 wandb

| | smoke config（§8.2.2 那七条） | longrun config（`--longrun`） |
|---|---|---|
| `logger.wandb_enabled` | `false` | `true` |
| 需要账号 | **不需要** | 需要 `WANDB_API_KEY` |
| `max_response_length` | 512 / 4096 | 20480（例子 7 为 4096） |

所以 §8.2 的七个例子不需要 wandb 账号。仅 `--longrun` 会用到：

```bash
WANDB_API_KEY=xxxx bash release/run_example.sh 1 --longrun --detach

# 没有账号时关掉即可；启动器检测到无 key 会自动追加这一条
EXTRA_OVERRIDE=logger.wandb_enabled=false bash release/run_example.sh 1 --longrun --detach
```

> Hydra 键名是 `logger.wandb_enabled`，不是顶层 `wandb_enabled`；写错会得到
> `ConfigKeyError: Key 'wandb_enabled' not in 'LumenRLConfig'`。
> 缺 key 的失败发生在 `RLTrainer.setup ... complete` **之后**，前几分钟看起来一切正常。

---

## 8.5 判断结果是否正确

`--check` 自动完成本节的判定：抠出第 1 步的四个指标与内置参考值比对，
统计 `Traceback` / `OutOfMemory` / `CUDA error` / `HSA_STATUS` 的出现次数，
输出 PASS / FAIL。人工判读见下。

```bash
bash release/run_example.sh 1 --check
bash release/run_example.sh 1 --check-only --log $DATA_ROOT/logs/example-1-xxx.log
```

### 8.5.1 参考值表

**测量条件**：8x MI355X（gfx950），镜像 `dapo-gfx950-rocm7.2.3-260902`，
命令即 `bash release/run_example.sh <N>`（等价于 §8.2.2 的整行参数），
**`seed=10086`**（`run_dapo.sh` 内固定），取**第 1 步**（`step=1`）的指标。

| # | config（`examples/DAPO/configs/`） | steps | resp | 端到端墙钟 | `rollout_corr/k3_kl` | `entropy` | `rollout_corr/kl`（有符号） | 实测次数 |
|---|---|---|---|---|---|---|---|---|
| 1 | `dapo_qwen3_8b_ray_vllm_smoke.yaml` | 3 | 512 | 156–190 s | **0.00109** ±30% | **0.609** ±25% | 0.00094 | 6 |
| 2 | `dapo_qwen3_8b_ray_vllm_fp8_smoke.yaml` | 3 | 512 | 166 s | **0.00469** ±30% | **0.789** ±25% | 0.00468 | 1 |
| 3 | `dapo_qwen3_8b_ray_vllm_fp8_smoke.yaml`（`TRAIN_FP8=1`） | 3 | 512 | 176 s | **0.00410** ±30% | **0.812** ±25% | 0.00412 | 1 |
| 4 | `dapo_qwen3_8b_ray_atom_fp8_4k_smoke.yaml` | 3 | 4096 | 557–602 s | **0.00286** ±50% | **0.597** ±50% | 0.00268 | 3 |
| 5 | `dapo_qwen3_8b_ray_atom_bf16_4k_smoke.yaml` | 1 | 4096 | 472 s | **0.000988** ±50% | **0.641** ±50% | 0.000821 | 2 |
| 6 | `dapo_qwen3moe_a3b_ray_vllm_verlref_4k_smoke.yaml` | 3 | 4096 | 552–557 s | **0.00154** ±50% | **0.864** ±60% | 0.00178 | 2 |
| 7 | `dapo_qwen3moe_a3b_ray_megatron_verlref_4k_smoke.yaml` | 3 | 4096 | 526–532 s | **0.00157** ±50% | **0.655** ±60% | 0.00166 | 2 |

粗体两列带容差的即 `--check` 判定 PASS / FAIL 的两项，参考值是「实测次数」列那么多遍的均值。
共 17 次运行，**退出码全部为 0**，四类错误计数**全部为 0**，`--check` **17/17 PASS**。
逐次原始记录见 [`VALIDATION-20260903.md`](../../release/VALIDATION-20260903.md)。

**端到端墙钟不设容差、不参与判定**：它受 kernel 缓存冷热影响，同一机器同一镜像上偏差可达
±15%，只作耗时量级参考。

### 8.5.2 判据

- **`rollout_corr/k3_kl` 是主判据**，容差 512 组（例子 1/2/3）±30%、4096 组（例子 4–7）±50%。
  它是 train / rollout 分布差异的 k3 估计量，非负且不会正负抵消，是全表最稳定的指标
  （七个例子实测最大偏差 ±17%）。
- **`entropy` 是第二判据**，容差 512 组 ±25%、ATOM 4096 组（例子 4/5）±50%、
  MoE 4096 组（例子 6/7）±60%。它是 `filter_groups` 筛选后那一批序列上的均值，
  样本少、方差大，**MoE 上尤其不稳（实测跨度 0.698–1.030）**，
  所以判 MoE 复现请以 `k3_kl` 为准。
- **`rollout_corr/kl` 只作数量级判据**：它是有符号均值，对称分歧会相互抵消，
  同一命令重复运行可相差 2.8 倍，因此只检查实测绝对值是否落在参考值的 1/10–10 倍之间。
  **高出一个数量级才算异常**，最常见原因是某一侧未启用 model-sensitive RMSNorm。
- **`rollout_corr/ppl_ratio` 仅供参考**，不参与判定。

两个 BF16 rollout（例子 1 的 0.00109、例子 5 的 0.000988）落在 1e-3 附近；
三个 FP8 的（0.00469 / 0.00410 / 0.00286）是其 3–4 倍，这是量化的代价，属正常。
两个 MoE（0.00154 / 0.00157）介于两者之间，FSDP2 与 Megatron 仅差 2%，
说明更换训练后端未引入额外的 train / rollout 漂移。

> 指标对不上时**先确认跑的是否为同一条 config**：
> `grep -m1 'CONFIG=' $DATA_ROOT/logs/example-<N>-*.launcher.log`
> 会打印本次实际使用的 config、`MODE`、`TRAIN_FP8`、`STEPS`。
> 配置错位与数值回归在指标上表现相同，但前者常见得多。

---

## 8.6 问题处理

**1. 运行结束后显存不自动释放。** smoke 正常结束后每张卡可能仍占着约 90.9 GB
（实测 89960382464–90905997312 B）：Ray worker 已退出，但显存未归还，且容器内看不到对应进程。
两次运行之间重启容器，否则下一次运行的 KV cache 预算会被压低。启动器每次启动前都会做这件事。

```bash
docker restart lumenrl-release
```

**2. 切换 ATOM 精度需清编译缓存。** torch inductor 缓存不按运行隔离，
例子 4 之后直接跑例子 5（或反之）会在 AOTAutograd 处失败。
启动器会记录上一次的 ATOM 精度，仅在精度改变时清理（`--keep-cache` 可关闭）。

```bash
docker exec lumenrl-release bash -lc \
  'rm -rf /tmp/aiter_configs /tmp/atom_torch_compile_cache /tmp/torchinductor_root'
```

**3. 判断长任务是否存活要看日志，不要用 `pgrep`。** `docker exec` 启动的进程与你的 shell
不共享进程树，`pgrep` 跨会话恒返回 0。**看日志文件是否仍在增长**：

```bash
watch -n 30 'ls -l $DATA_ROOT/logs/example-4-xxx.log'
```

**4. `docker restart` 会终止 `--detach` 起的任务。** 启动器在重启前会先检查上一次的日志
是否仍在增长，若在增长则拒绝启动并说明处置方式；`--force` 表示强制终止。

**5. 日志中出现 `waiting for baton release` 不是卡死。** 8 个训练 actor 在等其中一个完成
JIT 编译，靠锁串行。发布镜像已预编译全部 kernel，正常不会出现；若挂载了自己的
aiter 源码则可能出现。启动器的 `STALL_LIMIT`（默认 2400 秒无新日志才放弃）为此留了余量。

**6. `filter_groups round N` 不是每个例子都有。** 该日志行只在启用动态采样的 config 上出现。
**例子 2、3 不会打印它**——它们共用的 config 显式设置了
`dynamic_sampling: false` + `filter_groups.enable: false`（`max_response_length: 512`
下 base 模型很少做完一道题，开启动态采样会筛掉所有 group）。
这不是故障：两者都会跑完 3 步、指标齐全、`--check` 通过。

**7. 覆盖 `aiter` 源码时必须同时更换 `AITER_JIT_DIR`。** 已编译的 kernel 与产生它的 aiter
revision 绑定，复用旧目录会在 import 阶段失败，且报错既不提 aiter 也不提分支：

```
AttributeError: module 'aiter.jit.module_aiter_core' has no attribute 'MlaVersion'
```

加上 `-e AITER_JIT_DIR=/tmp/aiter-jit-<你的分支>` 即可。

**8. `flydsl` 必须与 `aiter` 同步升级。** 底座镜像自带 0.1.4.2，而 `aiter/lumen/moe`
要求 `>= 0.2.4`（本镜像固定 0.3.2）。版本不匹配的表现是下面这条 import 期错误，
它由 ATOM 的 `model_ops/moe.py` 抛出，信息中完全不提 aiter：

```
ImportError: Unsupported `flydsl` version: expected >=`0.2.4`, got `0.1.8`.
```

**9. FP8 训练发散**（entropy 极低 / `grad_norm` 与 `rollout_corr/kl` 都是 1e4 量级）：
见 [6. 排障](06-troubleshooting_cn.md)。

---

## 8.7 延伸阅读

| 需求 | 去处 |
|---|---|
| 要改源码 / 换模型 / 不用镜像 | [1. 环境搭建](01-env-setup_cn.md) → [2. 装依赖](02-dependencies_cn.md) → [4. 启动](04-launching_cn.md) |
| 重做数据（§8.4.3 是精简版） | [3. 模型与数据](03-data_cn.md) |
| 本章 §8.6 未覆盖的故障 | [6. 排障](06-troubleshooting_cn.md) |
| 双节点训推分离（例子 8，镜像不覆盖） | [5. 多节点 RDMA](05-multinode-rdma_cn.md)、[7. 训推分离双节点 RDMA](07-disaggregated-rdma_cn.md) |

镜像的构建方式与版本固定见仓库根目录的 [`release/`](../../release/)。
