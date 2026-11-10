# DeepSeek-V4-Flash DAPO 示例

在 AMD Instinct GPU 上用 LumenRL 对 DeepSeek-V4-Flash 做 DAPO 强化学习：Megatron-Native 训练，vLLM rollout，两者同卡部署。
所有依赖都固定到具体版本，所有 Python 步骤都在固定 digest 的容器镜像里运行。在一台空机器上，按本文顺序执行即可复现。

## 支持范围

| 配置 | 状态 |
|---|---|
| 4 层切片（`DSV4_LAYERS=4`），单机 1×8 | 已验证 |
| 4 层切片，两机 2×8 | 训练配方已验证；本示例的多机脚本尚未端到端验证 |
| 全量 43 层（`DSV4_LAYERS=43`） | 开发中，暂不可用（见文末） |

4 层切片用于验证整条链路：权重转换、训练、rollout、权重同步。它的输出本身没有意义，`reward/accuracy=0` 是正常现象。

## 环境要求

**硬件**

- 每节点 8 张 gfx950 GPU（MI350X / MI355X 级别，每卡 288 GiB）。单机时训练侧每卡显存峰值约 115 GiB，另有 vLLM 占用总显存的 25%。
- 其他 AMD GPU 未验证。脚本启动时会检查 GPU 数、架构和显存，不符合就报错退出；确认风险后可以设 `FORCE=1` 继续。
- 多机之间走 TCP，不需要 RDMA。

**软件和网络**

- 宿主机只需要 bash（≥ 4.3）、docker、git、curl、coreutils。
- 能访问 GitHub、Docker Hub、Hugging Face（也可以用 `HF_ENDPOINT` 指向镜像站）。
- 机器上不能有其他 Ray 实例。脚本发现已有 Ray 进程就报错退出，不会去停它们。

**磁盘**

| 位置 | 用途 | 4 层 | 43 层 |
|---|---|---|---|
| Docker 数据目录 | 两个镜像 | 约 145 GB（containerd 镜像存储约 200 GB） | 同左 |
| `WORK_DIR` | 模型、转换产物、数据、代码 | 约 230 GB | 每节点约 2 TB |

两者在同一块盘上时，`check_env.sh` 会按两者之和检查。

## 快速开始（单机，4 层）

所有命令都在 bash 里、`examples/DSv4_DAPO/` 目录下执行。每一步都可以重复执行，已经完成的部分会核对后跳过。

**0. 获取代码，配置环境**

```bash
git clone -b dev/dsv4-dapo-example https://github.com/ZhangDanyang-AMD/Lumen-RL.git
cd Lumen-RL/examples/DSv4_DAPO
cp env.example.sh env.sh        # 编辑 env.sh，把 WORK_DIR 设为一个绝对路径
source env.sh
```

`WORK_DIR` 是唯一的必填项。其他配置项见「配置」一节。

**1–6. 准备（约 15–20 分钟，主要是下载）**

| 步骤 | 命令 | 完成标志 |
|---|---|---|
| 1 环境检查 | `bash scripts/check_env.sh` | `environment OK` |
| 2 拉取镜像 | `bash scripts/pull_images.sh` | `training-container flags accepted; torch sees 8 GPU(s)` |
| 3 获取依赖代码 | `bash scripts/prepare_code.sh` | `code ready under …` |
| 4 构建 Megatron | `bash scripts/build_megatron.sh` | `megatron_dsv4 OK` |
| 5 准备权重 | `bash scripts/rebuild_artifacts.sh` | `3176/3176 names match` 以及 `hf_style=0 mtp=0 -> OK` |
| 6 生成数据 | `bash scripts/make_data.sh` | 训练集 1,791,700 行，验证集 960 行 |

第 5 步依次执行：按需下载模型分片（4 层只需约 28.5 GB），并按 Hugging Face 提供的校验和逐个校验；
截取前 N 层；FP8 转 BF16；转成 Megatron 检查点；把张量名改回 vLLM 需要的原生名称。
第 6 步默认生成与参考运行完全相同的数据；`DATA_FILTER=tokens` 会按 1024 token 过滤 prompt，结果与参考值不能直接比较。
转换过程中出现的 `cp: cannot stat …`、`[ERROR] cache_position … not documented` 是上游工具的正常输出，可以忽略。

耗时较长的步骤可以放到后台执行：

```bash
mkdir -p "$WORK_DIR/logs"
setsid nohup bash scripts/rebuild_artifacts.sh > "$WORK_DIR/logs/rebuild.log" 2>&1 < /dev/null &
tail -f "$WORK_DIR/logs/rebuild.log"
```

**7. 训练（3 步约 11 分钟）**

```bash
bash scripts/run.sh             # 默认 STEPS=3；也可以 STEPS=6 bash scripts/run.sh
bash scripts/status.sh          # 随时查看进度和判定结果
```

`run.sh` 依次检查环境、创建训练容器、启动 Ray、核对 GPU 数，然后在后台启动训练，立刻返回。
日志在 `$WORK_DIR/logs/<RUN_ID>.log`，训练结束后 `<RUN_ID>.log.exit` 里是退出码。

**8. 结束**

```bash
bash scripts/stop.sh --rm       # 停止 Ray 并删除本示例创建的容器
```

## 判定结果

`status.sh` 按以下四条判定：

1. 训练跑完了指定的步数，退出码为 0。
2. 日志中没有 `Traceback`。
3. rollout replica 数等于节点数。
4. 每一步的 `rollout_corr/kl` 都在 2.0e-3 到 4.0e-3 之间；并且每一步 `grad_norm > 0` 当且仅当这一步有回复短于最大长度（4096 token）。

最终结论有三种：

| 结论 | 含义 |
|---|---|
| `PASS` | 四条全部通过，并且至少有一步有训练信号 |
| `INCONCLUSIVE` | 四条都通过，但没有任何一步有训练信号。重跑一次，或者用 `STEPS=6` |
| `FAIL` | 至少有一条没通过 |

4 层切片上每条回复的 reward 都相同，只有当回复提前结束、超长惩罚不同时，才会产生梯度。这个事件是随机的，平均每步大约发生一次，
所以个别步骤 `grad_norm=0` 属于正常现象，3 步全部为 0 的概率约为 13%。

**参考值（单机 1×8）**：每步 `rollout_corr/kl` 约 2.5e-3 到 3.4e-3，训练侧显存峰值约 115 GiB/卡，每步约 170–190 秒。
两机时每步约 100 秒，显存峰值约 96 GiB/卡。

## 多机（2×8）

1. 每台机器都满足环境要求；仓库和 `WORK_DIR` 在每台机器上的绝对路径相同（不要求共享存储）。
2. 在 head 节点的 env.sh 中设置：

   ```bash
   export NODES="<head 主机> <worker 主机>"   # head 排第一；所有命令都在 head 上执行
   export REMOTE_EXEC="ssh {host}"           # 默认值，可以替换成调度器的远程执行命令
   ```

   `REMOTE_EXEC` 要能把标准输入转给远程节点上的 `bash -s`。环境变量会随命令一起传过去，worker 节点不需要自己的 env.sh。

3. 在每台节点上准备镜像、代码和权重，然后启动：

   ```bash
   for s in pull_images.sh prepare_code.sh build_megatron.sh rebuild_artifacts.sh; do
     bash scripts/on_all_nodes.sh $s
   done
   bash scripts/make_data.sh       # 只有 head 节点需要
   bash scripts/run.sh
   ```

worker 数和每节点的放置都由 `NODES` 自动推出，保证每个 TP=8 的 rollout replica 都在同一个节点内。

## 配置

`env.example.sh` 里每一项都有说明，常用的几项：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `WORK_DIR` | 必填 | 工作目录 |
| `DSV4_LAYERS` | `4` | 层数 |
| `NET_IF` / `HEAD_IP` | 自动探测 | 默认路由所在的网卡及其 IPv4 |
| `RAY_PORT` / `RAY_DASHBOARD_PORT` | `6379` / `8265` | Ray 端口 |
| `DOCKER` / `DOCKER_HOST` | `docker` | 例如 `DOCKER="sudo docker"` |
| `HF_ENDPOINT` / `HF_TOKEN` | 官方地址 / 空 | 模型和数据集都是公开的 |
| `MODEL_SRC_DIR` | 空 | 指向已经下载好的模型目录，跳过下载、只做校验 |
| `STEPS` | `3` | 训练步数 |
| `FORCE` | `0` | 在未验证的硬件上继续运行 |

## 固定版本

| 组件 | 版本 |
|---|---|
| Lumen-RL | 本分支 |
| aiter | `ff1006d03b53a693424c30e192c6e700e632bef8`（ZhangDanyang-AMD/aiter），composable_kernel 子模块 `83566edb` |
| Lumen | `e6379cbd9057b03c18213fbf65a4d891160545ca`（ZhangDanyang-AMD/Lumen） |
| ATOM | `7173f5b8f75849fec98f3943077deb44c00925d2`（xysheng-AMD/ATOM） |
| 训练 / rollout 镜像 | `vllm/vime-rocm@sha256:3de38e0566effb9d97e14cd2412fa97467db96401776afb71fd424b09d9dfda9` |
| 权重转换镜像 | `rocm/sgl-dev:miles-rocm720-mi35x-20260803@sha256:fd662fc0ef3dbe99fb49f53fa6b393a921f50248f68d370655ac6562dc7f1301` |
| 模型 | `deepseek-ai/DeepSeek-V4-Flash-Base` @ `8855555deef230a27a21a8d6f294b7b7497759b6` |
| 数据 | `BytedTsinghua-SIA/DAPO-Math-17k`、`BytedTsinghua-SIA/AIME-2024`，tokenizer 用 `Qwen/Qwen3-8B-Base`，均固定到具体 revision |

DeepSeek-V4 所需的 Megatron 改动以补丁形式提供（`megatron/megatron_dsv4.patch`），由 `build_megatron.sh` 打到训练镜像自带的 Megatron-Core 上，
并按整棵树的 sha256 校验。`plugins/` 下是 DeepSeek-V4 模型插件的无 kernel 版本，其中部分代码来自 radixark/miles（Apache-2.0），
出处和修改说明见 `plugins/vendored/NOTICE`。

## 关键设置

以下设置是验证通过的组合，请不要修改：

- `VLLM_ROCM_USE_AITER=1`，其他 AITER 开关为 0：DeepSeek-V4 的稀疏注意力 indexer 在 ROCm 上只有 AITER 实现。
- MoE 后端用 `triton`，rollout TP=8：其他组合在 gfx950 上会出错。
- `NCCL_ALGO=Ring`：训练开启了确定性模式，它要求这个设置。
- `enable_dynamic_batch: false`：DeepSeek-V4 的注意力不支持把多条序列打包进一个 micro-batch，打包后结果会出错而且不报错。
- `enable_sleep_mode: false`：训练和 rollout 的显存是叠加的，配置已经按这一点留好了余量。
- 4 层切片上 `filter_groups` 保持关闭，否则所有样本组都会被过滤掉。
- Ray 必须由 `run.sh` 启动：rollout 进程继承的是 Ray 节点的环境，而不是训练驱动进程的环境。

运行环境定义在 `runtime/ray_env.sh`，训练容器的 docker 参数在 `scripts/lib.sh` 的 `TRAIN_DOCKER_FLAGS` 中（不使用 `--privileged`）。

## 常见问题

| 现象 | 处理 |
|---|---|
| `hardware differs from the validated setup` | 硬件不在验证范围内；确认后设 `FORCE=1` |
| `Ray is already running on this host` | 机器上已经有别的 Ray，先释放机器 |
| `port … is already in use` | 修改 `RAY_PORT` / `RAY_DASHBOARD_PORT` |
| `docker refused the training container's flags` | docker 的权限策略拒绝了某个参数，调整 `TRAIN_DOCKER_FLAGS` 或联系管理员 |
| 磁盘空间不足 | 把 docker 的 data-root 或 `WORK_DIR` 换到更大的盘 |
| 下载中断，或 `checksum mismatch` | 重新执行即可，支持断点续传；镜像站的文件可疑时换回官方地址 |
| `KeyError: 'layers.0.input_layernorm.weight'` | rollout 读到了未改名的权重目录，重新执行第 5 步 |
| `FileNotFoundError: … aime-2024.filtered.parquet` | 重新执行第 6 步 |
| 训练长时间没有输出 | 首次运行要编译 aiter 算子，日志在增长就耐心等待；确实卡住时执行 `stop.sh --rm` 后重新运行 |
| 结论为 `INCONCLUSIVE` | 见「判定结果」，重跑一次或增加 `STEPS` |

## 清理

训练容器以 root 运行，它生成的部分文件（aiter 编译产物、`__pycache__`、训练日志）在宿主机上属于 root。请用脚本删除：

```bash
bash scripts/stop.sh --rm
bash scripts/clean.sh --pycache --aiter-jit
bash scripts/clean.sh "$WORK_DIR/logs" "$WORK_DIR/models"
```

## 全量 43 层（开发中）

全量训练需要的代码和配置还没有合入本分支，`DSV4_LAYERS=43` 时 `run.sh` 会直接报错。权重准备脚本已经支持 43 层，但还没有验证。已知的约束：

- 至少需要 4 机 32 卡（EP=32）；两机显存不够。
- 每节点约 2 TB 磁盘。
- 层数 43 是质数，无法使用流水线并行。
- rollout 需要使用在线 FP8 量化。
