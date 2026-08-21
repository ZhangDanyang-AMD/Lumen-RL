# 05 · 运行期的坑

> **稳定操作手册第五篇**，索引见 [`README.md`](README.md)。
>
> 这里收的是**跑起来之后**才会遇到的问题。共同点：**失败现场离病因很远**——
> 报错行往往和真正的原因隔着几十帧，甚至在另一个进程里。
>
> 环境搭建期的坑在 [`02-environment-setup.md`](02-environment-setup.md)；
> primus 底座那八条的完整版在 [`../../scripts/primus/README.md`](06-primus-pitfalls.md)。

---

## 0. 三条铁律

1. **每次 run 之前 `docker restart <容器>`**，然后确认
   `python3 -c "import torch;print(torch.cuda.device_count())"` **是 8**。理由见 §1。
2. **判 GPU 干净不要只看 `rocm-smi`**，同上。
3. **不要在没确认参数真的生效前相信 A/B。** 见 §5。

---

## 1. ⚠️⚠️ 孤儿进程会让机器「看不到 GPU」，而显存看起来是空的

**这条排第一，因为它会污染你所有后续的实验结论。**

失败的 run 会在容器里留下几百个 ray/vLLM 进程（实测一次积到 **822 个 python 进程 / 660 个孤儿**）。
`ray stop` 只回收自己的 actor，vLLM 的 `EngineCore` 和 `multiproc_executor` worker 是普通子进程，
会活下来。

症状很坑：

- **显存回到 298 MB/卡**，所以 `rocm-smi` 看着**完全正常**
- 但新起的进程 `torch.cuda.device_count()` 变成 **0**
- 下一次 run 会**静默挂在 `socket.recv()` 上**（崩溃的 run 还占着权重同步的 IPC 端点，
  新 run 的发送侧等一个不存在的接收者）

⚠️ **这一轮真的因此得出过三次完全错误的 A/B 结论**，把一个真因排除掉了一整轮。

**处理**：

```bash
docker restart <容器>          # stop/start 不丢依赖，只有 docker rm 才丢
# 或者精细一点：
pkill -9 -f lumenrl.trainer.main; ray stop --force
pkill -9 '^VLLM::'             # ⚠️ 匹配进程名，不是 -f
pkill -9 -f LumenActorWorker; pkill -9 -f VLLMRayServer
```

⚠️ **`pkill -f 'VLLM::'` 会自杀**——执行它的那条命令行自己就含 `VLLM::`，
启动器每次在 1 秒内退出码 137、**无任何输出**，极难查。用 `pkill -9 '^VLLM::'`（匹配进程名）。

⚠️ 用 `pgrep` 判断「vLLM 在不在跑」是**假阳性**：容器 PID 1 是 `sleep infinity` 不回收 zombie，
一轮崩溃能攒几十个 `VLLM::` defunct 进程。

---

## 2. 长任务的启动与存活判断

见 [`01-cluster-access.md`](01-cluster-access.md) §11。三句话版本：

- **本地 `setsid nohup` 脱离 + 远端保持前台**，两种直觉写法都错
- 判存活看**远端日志在增长**，`pgrep` 跨会话**恒返回 0**
- 「停滞」要连续 4 次轮询（8 分钟）无输出才算

---

## 3. 显存与性能测量

| 陷阱 | 正解 |
|---|---|
| `rocm-smi` 排在 python 之后 | **恒读 0**（引擎一退出显存就还回去了）。要**运行期每 5 秒采样、事后取每卡最大** |
| 指望 vLLM 自己的 memory-profiling 日志 | 这套环境把 vLLM 跑在 **WARNING** 上，那些 INFO 行一行都不会出现 |
| 用 `allocated` 做规划 | 要看 **`reserved`**——那才是 rocm-smi 和 colocated vLLM 争的东西，init 时就高 26 GiB |
| 短序列上量然后外推 | 「余 98 GB」是在约 130 token 上量的，1280 就撑满 |
| 拿常驻显存判断 offload 生效 | 优化器的 fp32 master 和 Adam 动量**第一次 `step()` 才落地** |
| `awk` 里写 `2**30` | 不认，要写 `/1073741824` |

⚠️ **探针的 cwd 要放节点本地盘**：GPU memory fault 会往 cwd 写 **50–80 GB** core dump。
`cd /mnt/m2m_nobackup/<user>/probe_scratch`。

⚠️ **分辨「慢」和「挂」**：先看 `rocm-smi --showuse`（应 85%+），诊断挂起用 `py-spy dump --pid`。
一次 9 分钟的「卡住」实际是在写 core dump。

---

## 4. 读日志

- ⚠️ **读 MEMDIAG 要先断行**：8 个 rank 往同一个 fd 上 `print` 会连成一行。

  ```bash
  sed "s/MEMDIAG/\nMEMDIAG/g" "$LOG" | grep -aoE "MEMDIAG\[rank 0\].*GiB"
  ```

- ⚠️ **`grep -c` 判计数**：没匹配时它自己打印 `0` 且返回 1，`|| echo 0` 会拼成 **`00`**。
- 固定的健康检查三条：

  ```bash
  grep -a "lumenrl.trainer.callbacks: step=" "$LOG" | tail -1
  grep -a "RDMA weight sync committed" "$LOG" | tail -1     # 训推分离才有
  grep -aiE "Training failed|Traceback|OutOfMemory|NCCL.*timeout|SIGABRT|=nan" "$LOG" | tail
  ```

- ⚠️ **真正的崩溃现场常常不在 driver 日志里**。Ray actor 硬崩时 driver 只说
  `ActorDiedError ... SYSTEM_ERROR`，栈在
  `/tmp/ray/session_*/logs/worker-*.err`（找 `Fatal Python error` / `SIGSEGV`）。

---

## 5. A/B 实验的纪律

| 规矩 | 反例 |
|---|---|
| **确认参数真的生效了** | `moe_router_dtype` 曾在 DSv4 路径上被**静默丢弃**（引擎那一支根本没读它），于是做了一次结论为「逐位相同」的无效 A/B。**让探针回显模型实际构建时的值** |
| **一次只改一个变量** | `world=8` 固定，所以 `PP=2` 必然把 `EP` 从 8 压到 4。拿 EP=8 基线比 PP=2 EP=4，得到看似 PP 有问题的 `grad_norm \|d\| = 2.348e-03`；补跑 PP=1 EP=4 基线后发现**全部来自 EP 变化**，PP 的贡献是 0 |
| **容器要干净** | 见 §1 |
| **数字要带条件** | 一个没有条件（几节点、多少层、什么并行度、什么序列长度）的数字等于没有数字 |

---

## 6. 硬件相关（gfx950）

- **`torch.einsum("bsgd,grd->bsgr")` 越界**：超过约 2040 token 就 `Memory access fault` +
  **69 GB core dump**。⚠️ **换长度只是换运气**——2040/2043/2044/2048/4096 都崩，
  1792 和 2052 恰好没事。**「我在长度 X 上试过没事」不是结论。**
  修法是改成每组一次普通 GEMM。⚠️ `.contiguous()` **能修前向但反向出 NaN**。
- **LDS 只有 160 KB/block**（Hopper/Blackwell 是 228 KB）：DSA indexer 的默认 tile 要约 224 KB，
  直接 `hipModuleLaunchKernel ... invalid argument`。改 `block_N=128, num_stages=2` 后 45/45 全过。
  **不是算法问题是显存预算。**
- **rollout TP=1 必然 memory fault**：五种 MoE backend / AITER 组合全部复现过。
  上游验证矩阵里 TP=1 从来没被覆盖。

---

## 7. 容器内的杂项

- **容器内默认是 root**，`$HOME` 是 `/root` 而不是登录节点的家目录。
  ⚠️ 脚本里写 `~/xxx` 会解析成 `/root/xxx` → `No such file or directory`。**用绝对路径。**
- **`ps` 报 `Error, do this: mount -t proc proc /proc`**：先 `mount -t proc proc /proc`。
  ⚠️ 在 `nx.sh` / srun 的命名空间里 `/proc` 没挂，会得到「进程已死」的假阳性。**看日志 mtime。**
- **没有 `nvidia-smi`**，用 `rocm-smi`。
- **`ulimit -n`**：容器启动和训练脚本**都**要设（默认软限制 1024，8 个引擎 + 训练 actor
  会在第一次 rollout 结束时耗尽，raylet 崩成 `Too many open files`，看着像 Ray 的 bug）。
- **多节点的每个节点都要清进程**，不只是 head。

---

## 8. Ray

- ⚠️ **`Ray cluster initialized: N nodes x 8 GPUs` 是读配置文件的，什么都不证明。**
  判据是 `ray status` 里有没有 `0.0/8.0 GPU`。
- ⚠️ **`ray start --num-gpus=8` 会被 Ray 覆盖**。要强制用
  `RAY_OVERRIDE_RESOURCES='{"GPU":8}'`。
- ⚠️ **`process_on_nodes` 不是可选的**：不给它 Ray 会自己挑 rank→node，
  一个 TP=8 的 rollout replica 可能横跨两个节点，`VLLMReplicaManager.create` 会拒绝
  （`replica i spans 2 nodes`）。
  ⚠️ `OmegaConf.from_dotlist` 按 YAML 解析，**列表里不能有空格**（`[8,8]` 不是 `[8, 8]`）。
- ⚠️ **`lumenrl.workers.base_worker` 在 import 时就调了一次 `ray.init()`**，
  所以 driver 早已连上一个本地实例，后面带 `address` 的那次只会打印
  `Calling ray.init() again after it has already been called`，**地址被忽略**。
  单节点跑的时候不用先起 raylet；多节点要靠 `cluster.ray_address` + 先起 head。
