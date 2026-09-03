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

> ⚡ **不想从源码搭？** 例子 1–7 有一条用**已发布容器镜像**的快速路径：镜像里软件栈已固定、
> kernel 已烘好，每个例子一条命令，还能自动比对参考值判 PASS/FAIL。
> 见 [**8. 用发布镜像跑七个例子**](docs/08-release_cn.md)——
> 从 `docker pull` 到第一个例子出结论只要三条命令，本页 1–4 章可以整段跳过。
> 要改源码、换模型或跑双节点（例子 8）才需要下面这条从零搭的路。

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

## 开跑前自检

四条检查，每一条都有人在上面耗掉过一小时。装完
[依赖](docs/02-dependencies_cn.md) 之后、第一个 smoke 之前跑一遍。

```bash
# 1. 镜像必须是 v0.23.0。机器上有更新的 vllm/vllm-openai-rocm tag 不能替代，
#    而 :latest 往往反而是更旧的 vLLM。
sudo docker exec "$CONTAINER" bash -lc \
  'python3 -c "import vllm, transformers; print(vllm.__version__, transformers.__version__)"'
#    期望：0.23.0 5.12.0

# 2. 每张卡都在约 298 MB 的空闲基线。高于此值说明有同租户，或上一次中断留下了孤儿进程
#    ——先看排错文档，别直接开跑。
sudo docker exec "$CONTAINER" bash -lc 'rocm-smi --showmeminfo vram | grep -i used'

# 3. 源码安装必须压过镜像自带的 wheel（PYTHONPATH 是必需的）。
sudo docker exec "$CONTAINER" bash -lc \
  'export PYTHONPATH="$RL_ROOT/Lumen-RL:$AITER_DIR:$LUMEN_DIR";
   python3 -c "import aiter, lumenrl, lumen;
from aiter import flash_attn_varlen_func; print(aiter.__file__)"'
#    期望是 $RL_ROOT/aiter/ 下的路径

# 4. 仅例子 7：TE 的 layer spec 必须能构造，而不只是 import megatron.core 成功。
sudo docker exec "$CONTAINER" bash -lc \
  'python3 -c "from megatron.core.models.gpt.gpt_layer_specs import \
get_gpt_layer_with_transformer_engine_spec as s; print(type(s()).__name__)"'
#    期望 ModuleSpec
```

**一次会话里连跑多个例子时：** 两次精度不同的 ATOM 运行之间要清编译缓存，否则例子 4
之后跑例子 5 会死在 AOTAutograd（见[排错](docs/06-troubleshooting_cn.md)）——torch 的
inductor 缓存不按运行隔离。

```bash
sudo docker exec "$CONTAINER" bash -lc \
  'rm -rf /tmp/aiter_configs /tmp/atom_torch_compile_cache /tmp/torchinductor_root'
```

各例子容易漏掉的追加项：例子 2、3、4 在任何一次 `docker rm` 之后都要重新打
[RMSNorm patch](docs/02-dependencies_cn.md)；例子 4、5 需要
[ATOM JIT 预编译](docs/02-dependencies_cn.md)；例子 6、7 要显式指定 `MODEL_PATH`
指向 MoE 模型并 export `SCRATCH_ROOT`，其中例子 7 还需要从源码编译 TransformerEngine。

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
| 8 | [**用发布镜像跑七个例子**](docs/08-release_cn.md) | ⚡ **快速路径，可替代 1–4 章**：`docker pull` + 一条命令跑例子 1–7，自带清场、日志、指标比对（`--check`）、参考值与容差 |

第 1–7 章是**从源码搭一套**，第 8 章是**用已发布的镜像**。两条路跑的是同一批例子
（第 8 章的 1–7 就是上面表里的 1–7），指标可以互相对照；例子 8（双节点）只有第 7 章覆盖。
