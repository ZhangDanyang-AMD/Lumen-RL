# `docs/agent/` —— DeepSeek-V4 开发的全部文档

**这个目录是自足的。接手 DSv4 这条线，读这里就够，不需要看仓库里其它任何 md。**

开发分支：**`dev/dsv4-dapo`**（详见 [`dsv4_agent.md`](dsv4_agent.md) 开头）。

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
| [`../../examples/DAPO/configs/dapo_dsv4_flash_*.yaml`](../../examples/DAPO/configs/) | DSv4 的四个配置。要改超参或拓扑时看 |
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
