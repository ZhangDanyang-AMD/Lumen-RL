> [Examples README](../README_cn.md) > 排障

# 6. 排障

**FP8 训练发散**（entropy ~0.04 / `grad_norm` 1e4+ / `rollout_corr/kl` 1e4+）：
基本只有两个原因——`FP8_PARAM_MANAGER` 没设成 0（它与 native FSDP2 的 fp32-master 冲突），
或 [vLLM RMSNorm patch](02-dependencies_cn.md#27-vllm-aiter-rmsnorm-patch例子-2-3-4-必需) 没打（新容器要重打）。

**显存回退（OOM）**：
- 降 `policy.max_response_length=8192` + `max_total_sequence_length=9216` + `max_token_len_per_gpu=9216`；
- 或降 `train_global_batch_size` / `gen_batch_size`；
- Ray 路径 **不要**开 `fsdp_cfg.param_offload` / `optimizer_offload`，会报
  `parameters should be materialized on CPU`。

**例子 7 的 OOM 有个反直觉的点**：`max_tokens_per_gpu: 22528` 时会在 step 14 死在 actor
backward，但**崩溃不是 allocated 峰值的问题**（改前后都是约 130 GB），而是**碎片**——
ROCm 没有 `expandable_segments`，每步约 7 个打满 22.5k 的 bin 反复申请释放巨块，reserved
比 allocated 多出 42 GB。压到 8192 之后碎片间隙塌到 4-11 GB，峰值 reserved 从 177 GB 降到
134 GB。**所以那个 8192 不能随手调回去。**

**`weight sync (colocate-ipc) left N/M rollout parameters untouched`**：同步漏了参数。
异常会列出前 8 个名字；若是 `...experts.w13_weight` / `w2_weight`，说明融合 MoE 路由没生效，
要么代码没拉到最新，要么 vLLM/transformers 升级后布局假设失效了。

**ATOM rollout 退化**（`MODE=atomfp8` / `atombf16` 时 `filter_groups: kept 0/96` +
`Rollout reward: accuracy=0.0000` + 日志大量 `finished with reason max`、无 `eos`）：
rollout 生成崩坏。优先检查 ATOM `atom/model_ops/layernorm.py` 的 plain RMSNorm 有没有传
`use_model_sensitive_rmsnorm=1`；未对齐会先表现为 `rollout_corr/kl` 偏大（~0.007 而非 ~0.004）。
**用例子 5 定位更快**：它把量化关掉了，kl 若仍偏大，问题一定在 ATOM 与训练侧的对齐上，
而不在 FP8。

**`TORCHDYNAMO_DISABLE` 不要手工设。** 脚本全局保持 `=1`（训练 actor 关 dynamo）；
例子 4、5 的 no-eager level=3 rollout 所需的 dynamo 由 `ATOMReplicaManager` 在创建 ATOM Ray
actor 时通过 `runtime_env` 注入 `TORCHDYNAMO_DISABLE=0`，**只作用于 rollout 进程**。
顶层 `export TORCHDYNAMO_DISABLE=0` 会让训练 actor 一并继承，对训练侧纯属副作用。

**例子 4、5 的另外两个前提**（`MODE=atomfp8` / `atombf16` 都会自动设，手工覆盖时别丢）：
`ATOM_ISOLATE_TORCH_COMPILE_CACHE=1`（否则 8 个单卡 replica 并发写同一个 torch compile
cache，触发 Inductor `write_atomic -> rename` 的 `FileNotFoundError`）、
`enable_sleep_mode=true` 且 `sleep_level=2`（rollout 后释放 KV cache / weights / CUDA graph，
否则训练 backward 容易 `HSA_STATUS_ERROR_OUT_OF_RESOURCES`）。

**跑完/中断后显存不释放**（`rocm-smi` 每卡仍 ~90 GB，但 `ps` 里已无 trainer）：
`run_dapo.sh` 只在**启动前**清理进程，收尾不清，所以 ATOM EngineCore 的 `spawn_main`
子进程（及其 inductor compile worker）会变成孤儿继续占显存。手动清理时注意用
`spawn[_]main` 这类写法，否则 `pkill -f` 会匹配到自己的命令行而自杀：

```bash
sudo docker exec "$CONTAINER" bash -lc '
  pkill -9 -f "compile_[w]orker"   || true
  pkill -9 -f "spawn[_]main"       || true
  pkill -9 -f "resource[_]tracker" || true
  sleep 8; rocm-smi --showmeminfo vram | grep -i used | head -3'
```

**换 `DATA_ROOT` 要无条件覆盖。** 起容器时的 `docker run -e DATA_ROOT=...` 把值烤进了容器环境，
自建 wrapper 里写 `export DATA_ROOT="${DATA_ROOT:-/new/disk}"` **不会生效**（变量已存在），
ckpt 会静默写回旧盘。直接 `export DATA_ROOT=/new/disk`。

**vLLM worker 里的 `logger.info` 不进 driver 日志。** 所以例子 7 看不到那行
`weight sync coverage`。**不能**据此认为断言没跑——判断断言是否触发，看它有没有抛异常。

多节点 RDMA 排障见 [多节点 RDMA](05-multinode-rdma_cn.md)。
