# LumenRL Examples Runbook

在一台**全新的 8 卡 AMD GPU 机器**上从零复现本目录下的 DAPO 数学 RL 训练例子。

> English version: [README.md](README.md)

所有例子共用同一个入口（`lumenrl.trainer.main`，Ray 控制器）和同一个启动脚本
（`examples/DAPO/run_dapo.sh`），差异全部由 config + 环境变量表达：
**单 Ray-driver 进程内 8 个训练 actor + 8 个同卡 colocated rollout replica（TP=1），
训练->rollout 权重经 ZMQ CUDA-IPC 同步**。

算法侧：clip-higher + dual-clip + token-mean 策略损失、GRPO 按 uid 组归一化、
动态采样 `filter_groups`、overlong 奖励缓冲、TIS rollout 修正。

**一句话复现**：设路径变量 -> clone 仓库 -> 起容器装依赖 ->（FP8 才需要）打 patch ->
下模型和数据 -> smoke -> `docker exec -d` 起长跑。

---

## 已跑通的例子

| # | 例子 | 模型 | 训练后端 | Rollout / 精度 | GPU | Runtime | 启动开关 |
|---|---|---|---|---|---|---|---|
| 1 | 8B BF16 基线 | Qwen3-8B-Base | Lumen FSDP2，BF16 | vLLM / BF16 | 8x MI355X（gfx950）/ 8x MI325X（gfx942） | `vllm/vllm-openai-rocm:v0.23.0` | `MODE=bf16` |
| 2 | 8B FP8 rollout | Qwen3-8B-Base | Lumen FSDP2，BF16 | vLLM / `fp8_per_block` | 同上 | 同上 | `MODE=fp8` |
| 3 | 8B FP8 E2E | Qwen3-8B-Base | Lumen FSDP2，**FP8 blockwise2d** | vLLM / `fp8_per_block` | 同上 | 同上 | `MODE=fp8 TRAIN_FP8=1` |
| 4 | 8B ATOM FP8 | Qwen3-8B-Base | Lumen FSDP2，**FP8 blockwise2d** | **ATOM** / `per_block_fp8` | 同上 | 同上 | `MODE=atomfp8 TRAIN_FP8=1` |
| 5 | 8B ATOM BF16 | Qwen3-8B-Base | Lumen FSDP2，BF16（纯 BF16，不打 Lumen norm patch） | **ATOM** / BF16 | 同上 | 同上 | `MODE=atombf16` |
| 6 | MoE FSDP2 | Qwen3-30B-A3B-Base | Lumen FSDP2，BF16 | vLLM / BF16 | 同上 | 同上 | `MODE=bf16` + MoE config |
| 7 | MoE Megatron EP=8 | Qwen3-30B-A3B-Base | **Megatron-Native**，TP=PP=CP=1，EP=8，DP=8 | vLLM / BF16 | 同上 | 同上 | `MODE=bf16` + Megatron config |
| 8 | MoE 双节点 RDMA | Qwen3-30B-A3B | **Megatron-Native**，TP=4，EP=8 | vLLM TP=2 x 4 / BF16 | 2x 8x MI308X（gfx942） | `qwen3-30b-a3b:rollout` + `trainer` | [训推分离部署指南](docs/07-disaggregated-rdma_cn.md) |

例子 1-7 在 **8x MI355X** 和 **8x MI325X** 上都跑通过：smoke + 长跑，exit 0、无 Traceback、
无 OOM、无 `HSA_STATUS`，权重同步覆盖率断言全过，收尾后显存回到约 298 MB/卡的空闲基线。

例子 8 在 **2x 8x MI308X** 上运行训推分离部署：节点 1 Megatron 训练，节点 2 vLLM rollout，
通过 RCCL/RoCE GPU Direct RDMA 权重同步（9-rank 进程组）。完整部署流程见
[训推分离双节点 RDMA 部署指南](docs/07-disaggregated-rdma_cn.md)。

> **例子 5 是例子 4 的 BF16 对照组**：同一个 ATOM rollout 引擎、同样的
> no-eager level=3 + sleep2，只把 rollout 的在线量化和训练侧的 FP8 一起关掉。

> **同一批卡上不能同时跑两个训练后端**，也不能和别人共用节点——引擎按"占整卡比例"算
> KV cache 预算。起之前先确认显存在空闲基线。

> **两个训练后端不能共用 checkpoint 目录**，格式不同。

---

## 文档

| 步骤 | 文档 | 内容 |
|---|---|---|
| 1 | [环境搭建](docs/01-env-setup_cn.md) | 路径变量、拉代码、起容器 |
| 2 | [装依赖](docs/02-dependencies_cn.md) | pip 安装、vLLM patch、ATOM JIT 预编译、导入链验证 |
| 3 | [模型与数据](docs/03-data_cn.md) | HuggingFace / ModelScope 下载、prompt 过滤 |
| 4 | [启动](docs/04-launching_cn.md) | Config、smoke、长跑、健康判据、监控 |
| 5 | [多节点 RDMA](docs/05-multinode-rdma_cn.md) | RDMA 预检、启动/checkpoint 验证、基线（仅双节点） |
| 6 | [排障](docs/06-troubleshooting_cn.md) | 所有已知故障模式及修复 |
| 7 | [训推分离双节点 RDMA](docs/07-disaggregated-rdma_cn.md) | 完整部署：Megatron 训练 + vLLM rollout，RDMA 权重同步，从零搭建 |
