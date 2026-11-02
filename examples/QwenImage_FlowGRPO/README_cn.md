# Qwen-Image FlowGRPO 全参数 RL（8× MI350X）：基础版与 Lumen 版

> English version: [README.md](README.md)

在单节点 8× AMD Instinct MI350X 上复现 verl-omni 官方配方
[`run_qwen_image_ocr_veomni.sh`](https://github.com/verl-project/verl-omni/blob/9ab544d81f93cbf49754aabacfa67f6fcf01b505/examples/flowgrpo_trainer/qwen_image/run_qwen_image_ocr_veomni.sh)：
Qwen-Image（20B DiT）全参数 FlowGRPO，VeOmni 训练引擎，vLLM-Omni rollout，Qwen3-VL-8B 作 OCR reward，300 步。

- **基础版**：官方配方，只改路径、卡数和 ROCm 注意力后端（§4）。
- **Lumen 版**：在基础版之上，给 actor 加 2 项 [Lumen](https://github.com/ZhangDanyang-AMD/Lumen) 补丁（§5），其余完全相同。

## 1. 与官方结果对比

官方参考：verl-omni 公开的 W&B run
[`andyzhou/VeRL-Omni-demo/8p8y9olb`](https://wandb.ai/andyzhou/VeRL-Omni-demo/runs/8p8y9olb)（算法与超参数相同，
4× H200，FSDP 引擎，在第 277 步结束），以及
[`docs/algo/performance.md`](https://github.com/verl-project/verl-omni/blob/9ab544d81f93cbf49754aabacfa67f6fcf01b505/docs/algo/performance.md)。
本示例：8× MI350X，VeOmni 引擎，两个版本使用同一 seed。

### 正确性

验证 reward（OCR 测试集 1018 条）：

| step | 官方 | 基础版 | Lumen 版 |
|---:|---:|---:|---:|
| 0 | | 0.402 | 0.403 |
| 30 | 0.689 | 0.644 | 0.618 |
| 60 | 0.851 | 0.815 | 0.822 |
| 90 | 0.932 | 0.901 | 0.919 |
| 120 | 0.964 | 0.928 | 0.937 |
| 180 | 0.975 | 0.934 | 0.964 |
| 240 | 0.950 | 0.939 | 0.968 |
| 270 | 0.924 | 0.960 | 0.975 |
| 300 | | 0.949 | **0.978** |

训练 reward（窗口内均值）：

| steps | 官方 | 基础版 | Lumen 版 |
|---|---:|---:|---:|
| 1–60 | 0.407 | 0.359 | 0.354 |
| 61–120 | 0.850 | 0.806 | 0.814 |
| 121–180 | 0.958 | 0.919 | 0.922 |
| 181–240 | 0.950 | 0.883 | 0.952 |
| 241–277 | 0.924 | 0.926 | 0.954 |

- 三者达到同一水平（验证 0.95–0.98）。官方前 120 步上升更快，之后回落；本示例两个版本一直上升到第 300 步，
  Lumen 版最终最高。
- 训练健康指标与官方在同一范围（30 步均值）：clip 比例 0.11–0.31（官方 0.09–0.24），grad norm 0.005–0.018
  （官方 0.006–0.019），三者都没有非有限梯度。
- 同一配置的不同运行之间本身就有明显波动，比较时请用同一台机器、同一 seed。

### 性能

| run | GPU | 引擎 | 每步耗时 | 吞吐（images/GPU/s） |
|---|---|---|---:|---:|
| 官方 W&B run | 4× H200 | FSDP | 247.7 s | 0.517 |
| 官方文档 | 64× H100 | VeOmni | 100 s | 0.079 |
| 基础版 | 8× MI350X | VeOmni | 129.1 s | 0.496 |
| **Lumen 版** | 8× MI350X | VeOmni | **122.4 s** | **0.523** |

取全部训练步的中位数，不含验证和存档步。每步都训练 32 个 prompt × 16 张图，所以不同卡数之间应比较 images/GPU/s。

| Lumen 版对基础版 | 基础版 | Lumen 版 |
|---|---:|---:|
| actor（`old_log_prob` + `update_actor`） | 31.5 s | 29.1 s（−8%） |
| 整步 | 129.1 s | 122.4 s（−5%） |
| 300 步全程（含启动、验证与存档） | 约 12.5 h | 约 12.0 h |

Lumen 只加速 actor，actor 约占每步的 24%。整步少用的 6.7 s 中，actor 占 2.4 s；其余来自权重同步和生成，
这两段不经过补丁，属于两次运行之间的环境差异。

## 2. 硬件与环境要求

- 8× MI350X（gfx950，288 GB），单卡显存峰值约 208 GB。
- 主机内存 ≥ 1.5 TB：8 个 rollout 副本休眠时整条 pipeline 留在主存（约 900 GB）。
- `$WORK` 所在磁盘 ≥ 1 TB：模型 71 GB；每个 checkpoint 228 GB，保留最近 2 个。
- Docker；宿主机 ROCm 驱动需兼容镜像内的 ROCm 7.2；能访问 GitHub、Hugging Face、Docker Hub、PyPI。

## 3. 复现步骤

```bash
export WORK=/path/to/work                       # 代码、模型、数据、运行结果都放在这里
git clone https://github.com/ZhangDanyang-AMD/Lumen-RL.git $WORK/src/Lumen-RL
EX=$WORK/src/Lumen-RL/examples/QwenImage_FlowGRPO

docker pull vllm/vllm-openai-rocm:v0.27.0
WORK=$WORK bash $EX/launch_container.sh         # 容器名 qwen-image-rl
docker exec qwen-image-rl bash $EX/setup_env.sh # 约 30 分钟；末尾应输出 19653 / 1018 行与 SETUP OK
```

训练（两个版本各用 8 卡，依次运行；日志在 `$WORK/runs/<RUN_NAME>/train.log`）：

```bash
docker exec -d qwen-image-rl bash -c "RUN_NAME=base bash $EX/run_flowgrpo.sh"           # 基础版
docker exec -d qwen-image-rl bash -c "RUN_NAME=lumen LUMEN=1 bash $EX/run_flowgrpo.sh"  # Lumen 版
docker exec qwen-image-rl python3 $EX/compare.py $WORK/runs/base $WORK/runs/lumen      # 与官方、彼此对比
```

- 快速对比：在命令末尾加 `trainer.total_training_steps=90`，第 30/60/90 步验证，每个版本约 3.5 h。
- W&B：给 `docker exec` 加 `-e WANDB_API_KEY=<key> -e WANDB_PROJECT=<project>` 即在线上传；不设则离线记录在运行目录里。
- 确认 Lumen 生效：`docker exec qwen-image-rl bash -c 'grep -h lumen-rl /tmp/ray/session_latest/logs/worker-*.out | head'`
  应能看到 `installed ... ['rmsnorm_fuse', 'local_adamw']` 和非零的 `fused` 计数。
- 停止运行：`docker exec qwen-image-rl bash -c 'ray stop --force; pkill -9 -f VLLM'`，再用 `rocm-smi --showpids` 确认 GPU 已空。

## 4. 与官方配方的差异

算法与超参数不改动。只有以下几项不同：

| 项 | 官方 | 本示例 | 原因 |
|---|---|---|---|
| GPU 数 | 脚本默认 4 | 8（`NUM_GPUS`） | 单节点 8 卡 |
| actor 参数 / 优化器 offload | 开 | 关 | 288 GB 显存放得下；开启后主存不够，且 offload 拷贝很慢 |
| 注意力后端 | 默认 | actor `native`（torch SDPA），rollout `TORCH_SDPA`、`enforce_eager=True` | ROCm 兼容 |
| 训练前验证 | 关 | 开 | 得到第 0 步的对比点 |
| seed | 默认 | 数据与 rollout 固定为 42 | 便于同 seed 对比 |

所有版本均已固定：镜像 `vllm/vllm-openai-rocm:v0.27.0`，verl、verl-omni、VeOmni、vllm-omni、flow_grpo、Lumen 的 commit，
以及两个模型的 revision，都写在 `setup_env.sh` 里。

## 5. Lumen 补丁（`LUMEN=1`）

补丁代码来自 Lumen 的 [`examples/perf-veomni/veomni_patches`](https://github.com/ZhangDanyang-AMD/Lumen/tree/7b5931d21a56869f3e13b55992d7a4e1f8484267/examples/perf-veomni/veomni_patches)；
`setup_env.sh` 按该 commit 把 Lumen 克隆到 `$WORK/src/Lumen`，也可以用 `LUMEN_DIR` 指向已有的、包含这些补丁的
Lumen。`lumen_rl_patch.py` 只在构建 VeOmni 训练引擎的 actor 进程里安装补丁；rollout 和 judge 在各自的进程里，不受影响。

| `LUMEN_PATCH` | 作用 | 数值 |
|---|---|---|
| `rmsnorm_fuse` | diffusers RMSNorm 前向经 `torch.compile` 融合 | 相对 FP32 为 55.6 dB，原实现为 52.6 dB |
| `local_adamw` | fused AdamW 直接在 FSDP2 本地分片上执行 | 与原实现逐位一致 |

`lumen_rl_patch.py` 另提供 `mask_all_true_to_none`，默认关闭。它去掉训练 micro-batch 中全为 True 的文本 mask，
让注意力走 flash kernel，actor 还能再快约 3%；但实测会让第 30–150 步的训练 reward 明显落后基础版（第 300 步
不受影响），原因尚未查明。需要时设 `LUMEN_PATCH=mask_all_true_to_none:rmsnorm_fuse:local_adamw`。

perf-veomni 的其余优化不用于 RL：actor 不运行 VAE，所以 VAE 相关优化用不上；关梯度检查点和关 reshard
会占用 rollout 所需的显存；`sdpa_efficient` / `attn_triton_fwd` 所用的 aiter 融合反向，在 RL 的真实激活上会产生非有限梯度。

## 6. 文件

| 文件 | 运行位置 | 作用 |
|---|---|---|
| `launch_container.sh` | 宿主机 | 启动容器，把 `$WORK` 挂载到相同路径 |
| `setup_env.sh` | 容器内 | 拉取固定版本的源码（含 Lumen），安装 RL 依赖（不动镜像自带的 torch / vLLM / triton），下载模型，生成数据 |
| `run_flowgrpo.sh` | 容器内 | 官方配方加 §4 的覆盖项；`LUMEN=1` 启用 §5 的补丁 |
| `lumen_rl_patch.py` | Ray worker 内 | 只在 actor 进程里安装 Lumen 补丁，并打印生效计数 |
| `compare.py` | 任意位置 | 多个 run 的验证 reward、训练 reward、分阶段耗时和吞吐，并与官方 run 对比 |
