# `docs/agent/` —— DeepSeek-V4 开发的全部文档

**这个目录是自足的。接手 DSv4 这条线，读这里就够，不需要看仓库里其它任何 md。**

开发分支：**`dev/dsv4-dapo`**（详见 [`dsv4_agent.md`](dsv4_agent.md) 开头）。

---

## 一句话：现在卡在哪

**除了「拿到 4 节点」，已知的阻塞项都清掉了。**
数值正确性、显存、底座、权重同步通路、全 43 层的加载层——五块都已封闭并有实测
（[`dsv4_agent.md`](dsv4_agent.md) §0 的「已封闭」表）。
唯一卡住整件事的是**全 43 层端到端从没跑通**，而它的直接后果是 43 层的
`rollout_corr/kl` 至今未知——那是判断这个模型能不能做 RL 的核心指标。

⚠️ **如果你手上只有一台机器**：能做的都在 §9.1，现在只剩四项（归档、safetensors 补
version 校验、43 层 AITER=0 的对照、单机 RDMA）。**不要重开 §9.1 顶部那八项已关闭的**。

⚠️ **如果你拿到了 4 节点**：第一件事是 §9.2-1（逐对压那 6 对节点，15 分钟出结果），
**排在端到端前面**——§5.3 那个 seq 6144 卡死最像「有一对节点在大流量下坏掉」。

---

## 新 agent 的最短路径

1. [`01-cluster-access.md`](01-cluster-access.md) —— 先能连上机器
2. [`02-environment-setup.md`](02-environment-setup.md) **§0**（三十秒版本）—— 把环境跑起来
3. [`dsv4_agent.md`](dsv4_agent.md) **§0**（一页现状）—— 知道做到哪了
4. 从 [`dsv4_agent.md`](dsv4_agent.md) **§9** 挑一件事做

中间遇到问题：判据不确定查 [`04`](04-probes-and-criteria.md)，
报错查 [`06`](06-primus-pitfalls.md) 的症状索引，跑第二次就不对了查 [`05`](05-operational-pitfalls.md)。

---

## 文档清单

### 活文档（有进展就更新）

| 文档 | 内容 |
|---|---|
| [`dsv4_agent.md`](dsv4_agent.md) | **当前开发状态**：一页现状、已完成、已解决的 bug、未解决的问题、不要重做的、当前实测值、下一步 |

### 稳定操作手册（配方本身变了才动）

| # | 文档 | 什么时候读 |
|---|---|---|
| 01 | [`01-cluster-access.md`](01-cluster-access.md) | 连集群、进节点、起长任务、判进程存活 |
| 02 | [`02-environment-setup.md`](02-environment-setup.md) | 拿到新节点建环境；或想知道为什么是 `rocm/primus:v26.4` |
| 03 | [`03-dsv4-artifacts.md`](03-dsv4-artifacts.md) | 要跑 DSv4 但 `/mnt/m2m_nobackup` 是空的。四份权重、三套命名、重建流程 |
| 04 | [`04-probes-and-criteria.md`](04-probes-and-criteria.md) | 跑完了不知道算不算过；想挑便宜的探针替代端到端；**以及哪些判据是错的** |
| 05 | [`05-operational-pitfalls.md`](05-operational-pitfalls.md) | 运行期的坑。**「跑第二次就不对了」先翻这份** |
| 06 | [`06-primus-pitfalls.md`](06-primus-pitfalls.md) | primus 底座的八条坑，**按症状索引**。报错查不到出处时来这里 |

---

## 目录之外的东西

**下面这些都不是必读**，本目录已经覆盖了需要的结论。列在这里只是方便你要动手改的时候找得到。

| 位置 | 什么时候才需要 |
|---|---|
| [`../../scripts/primus/`](../../scripts/primus/) | 脚本和探针的**本体**。要改配方时看，每个环境变量旁边都写了原因 |
| [`../../examples/DAPO/configs/dapo_dsv4_flash_*.yaml`](../../examples/DAPO/configs/) | DSv4 的五个配置。要改超参或拓扑时看 |
| [`../../examples/DAPO/configs/dapo_qwen3moe_a3b_dsv4stack_1node_smoke.yaml`](../../examples/DAPO/configs/) | **诊断用**：Qwen3-MoE 跑在 DSv4 的栈上，分离「模型的问题」和「基础设施的问题」。见 [`dsv4_agent.md`](dsv4_agent.md) §3.9 |
| [`../../scripts/primus/run_dsv4_dapo_1node.sh`](../../scripts/primus/) | 单节点 DSv4 DAPO 的 primus 版启动器。⚠️ `~/4node/07_dsv4_megatron_1node.sh` 是 22.04 时代的，在 primus 上有两行是致命的，别用 |
| [`../../examples/docs/07-disaggregated-rdma.md`](../../examples/docs/07-disaggregated-rdma.md) | **只有真要搭训推分离时**才需要（那是另一条线的部署全流程）。机制本身 [`dsv4_agent.md`](dsv4_agent.md) §8 已经讲清楚了 |
| `~/working/amd-rl-runbook/` | 13 份历史 runbook，本目录的原始素材。**不在版本控制内，换机器就没了**。只有想看某条结论的完整证据链，或要做非 DSv4 的工作时才回去翻 |

---

## 写这些文档的约定

1. **`✅` 实测过 / `❌` 没做过 / `⚠️` 有坑或有前提。不要把计划写成 ✅。**
2. **数字要带条件**——几个节点、多少层、什么并行度、什么序列长度。
   一个没有条件的数字等于没有数字。
3. **被推翻的东西要留下来**，而不是删掉。这几份里最省时间的部分就是
   「试过、不行」和「这个判据是错的」。
4. 记**症状**不只记根因。下次遇到时你看到的是症状。
5. **操作步骤写进 01–06，当前状态写进 `dsv4_agent.md`。**
   两边都写会立刻不同步——改配方就地改手册，不要在活文档里复制一份。
