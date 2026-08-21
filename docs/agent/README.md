# `docs/agent/` —— 给 Agent 的稳定操作手册

这个目录放**不怎么变的操作知识**：怎么连集群、怎么建环境、怎么准备产物、怎么判断跑对了、
以及运行期会踩什么坑。它们和「今天在做什么」无关，换任务换模型都适用。

**正在进行的开发**记在 [`../dsv4_agent.md`](../dsv4_agent.md)——那份是活的，会随进展更新；
这个目录里的五份是它的地基，只在配方本身变化时才动。

---

## 读的顺序

| # | 文档 | 什么时候读 |
|---|---|---|
| 01 | [`01-cluster-access.md`](01-cluster-access.md) | **第一份**。连不上集群、进不去节点、长任务起不来 |
| 02 | [`02-environment-setup.md`](02-environment-setup.md) | 拿到新节点要建环境；或者想知道为什么是 `rocm/primus:v26.4` |
| 03 | [`03-dsv4-artifacts.md`](03-dsv4-artifacts.md) | 要跑 DSv4，但 `/mnt/m2m_nobackup` 是空的 |
| 04 | [`04-probes-and-criteria.md`](04-probes-and-criteria.md) | 跑完了不知道算不算过；或者想挑一个便宜的探针替代端到端 |
| 05 | [`05-operational-pitfalls.md`](05-operational-pitfalls.md) | **出问题先翻这份**，尤其是「跑第二次就不对了」 |

**新 agent 的最短路径**：01 → 02 §0（三十秒版本）→ [`../dsv4_agent.md`](../dsv4_agent.md) §0（一页现状）→
从 `dsv4_agent.md` §9 挑一件事做。中间遇到问题回来查 04 和 05。

---

## 相关但不在这个目录里的

| 位置 | 内容 |
|---|---|
| [`../dsv4_agent.md`](../dsv4_agent.md) | **当前开发状态**：做到哪了、还剩什么、下一步。活文档 |
| [`../../scripts/primus/README.md`](../../scripts/primus/README.md) | primus 底座那**八条坑**的完整版（症状 / 根因 / 诊断手法）+ 脚本用法 |
| [`../../scripts/primus/`](../../scripts/primus/) | 脚本本体。每个环境变量旁边都写了为什么 |
| [`../../examples/docs/07-disaggregated-rdma.md`](../../examples/docs/07-disaggregated-rdma.md) | 训推分离部署（另一条线，Qwen3-30B-A3B / 2 节点 MI308X） |

---

## 写这些文档的约定

沿用 `dsv4_agent.md` 的规矩，因为它们会被一起读：

1. **`✅` 实测过 / `❌` 没做过 / `⚠️` 有坑或有前提。不要把计划写成 ✅。**
2. **数字要带条件**——几个节点、多少层、什么并行度、什么序列长度。
   一个没有条件的数字等于没有数字。
3. **被推翻的东西要留下来**，而不是删掉。这几份里最省时间的部分就是
   「试过、不行」和「这个判据是错的」。
4. 记**症状**不只记根因。下次遇到时你看到的是症状。

改配方时**就地改这里**，不要在 `dsv4_agent.md` 里复制一份——那份只放当前状态和结论指针。
