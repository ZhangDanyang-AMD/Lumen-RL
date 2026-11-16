# MICake Planner RSI

只训练 Planner。Manager 和 Critic 每一代都是冻结的 DSV4.1。冠军同时是训练数据的来源和 LoRA 基座：HOLD 的权重不会进入下一代。

目标是在与训练集不相交的 held-out 形状上，找到一个比 DSV4.1 显著更快的 GEMM kernel。任务是 MICake C144 的 schedule 搜索：Planner 提交 typed config，环境编译、测正确性并计时。每条 episode 最多 10 次提交。

## 一代里面发生什么

状态写在 `<root>/lineage.json`。进程中断后用同一条命令重跑，会从当前代未完成的阶段接着做。

1. **采集。** 冠军 Planner 在训练形状上跑端到端 episode（`collect_c144` 和 `collect_large`）。这些目录进入经验池。held-out 测评目录不会进入。
2. **造数据。** 只保留冠军自己写出的采集目录。一条样本是一次真实测到的 `new_best` 提交。本代新轨迹全部保留，更早的轨迹最多回放同样多的条数。脚本生成的单轴修正不是训练目标。
3. **训练。** 在冠军合并权重上做 LoRA SFT，再合并成挑战者。注意力 rank 32（alpha 64），专家 rank 16（alpha 32），8 卡数据并行。基座是 `Qwen/Qwen3-Coder-30B-A3B-Instruct`。
4. **测评。** 冠军、挑战者、DSV4.1 三个 Planner 交错跑同一批任务。`--test-portfolio heldout` 时形状与训练集不相交。
5. **门控。** 挑战者相对冠军有两条晋升路径，并且预算浪费不能比冠军高出 0.05 以上：
   - `faster_kernel`：平均 best speedup 的增益至少 0.005。
   - `faster_path`：增益不低于 -0.002，到达 1.04 倍的时间不超过冠军的 0.85，且到达率不低于冠军。
   晋升后挑战者成为冠军，上一代冠军的合并权重会删掉。HOLD 的合并权重在下一次合并前删掉。adapter 都留着。

对 DSV4.1 另有一条停止规则，不参与晋升。`--until-dsv4` 时，挑战者要在全体样本上、并且在每个 dtype 上都同时满足：平均 best 增益至少 0.005，配对置换检验 p < 0.05。否则循环继续。

采集或测评里，Planner 子任务 0 回合就 `ERROR` 的记录会被删掉再补跑（服务卡住，不是策略结果）。三次补跑后条数仍不够就退出，不会在残缺集合上做门控。记录按 trial key 去重。

## 机器占用

| 用途 | GPU | 端口 |
|---|---|---|
| MICake 编译和计时 | 0, 1, 6, 7 | |
| 冠军 Planner | 4 | 8108 |
| 挑战者 Planner | 3 | 8111 |
| DSV4.1 Manager / Critic | | `https://127.0.0.1:8443/v1` |

一次合并大约写出 57 GiB。空闲磁盘低于 100 GiB 时不会开始合并；运行中低于 30 GiB，或可用内存低于 16 GiB，训练会被停掉。`/tmp` 里超过一天的 MICake 临时目录会清掉。

换冠军时，新服务会先拆掉占着同一端口的旧 `atom-micake-*` 容器。就绪的条件是 `/v1/models` 返回的模型路径等于目标 checkpoint。

## 怎么跑

在 Lumen-RL 仓库根目录执行。`DSV4_API_KEY` 只放在环境变量里，不要写进命令行或日志。

```bash
export DSV4_API_KEY=...   # 从密钥文件读入，不要打印

python3 experiments/micake-subagent-rsi/scripts/rsi_planner.py \
  --root output/micake-subagent-rsi/rsi_mixed \
  --champion p3 \
  --champion-checkpoint /home/danyzhan/Lumen-RL/output/micake-subagent-rsi/rsi_planner/p3/checkpoint \
  --dtypes bf16,fp8 \
  --gen-prefix m \
  --test-portfolio heldout \
  --seed-experience output/micake-subagent-rsi/rsi_planner/p3/eval \
  --samples-per-shard 9 \
  --until-dsv4 \
  >> output/micake-subagent-rsi/rsi_mixed/driver.log 2>&1
```

`--samples-per-shard 9` 配合 4 张 MICake 卡，每个臂 36 条 episode，三个臂一共 108 条。默认值是 3，即每臂 12 条。采集默认每张卡每个 portfolio 8 条。

`--champion`、`--champion-checkpoint`、`--dtypes`、`--test-portfolio`、`--seed-experience` 只在 `<root>/lineage.json` 还不存在时生效。之后以 lineage 为准。

只跑固定代数时去掉 `--until-dsv4`，加上 `--generations N`。默认配置是 `configs/gen0_planner_critic.yaml`。

## 输出布局

```
<root>/
  lineage.json          冠军、经验池、每一代的决策和对比
  driver.log            重定向后的日志
  m21/
    collect_c144/       训练形状上的采集
    collect_large/
    data/planner.jsonl  本代训练样本
    train/planner/      adapter 和 sft.yaml
    checkpoint/         合并权重；HOLD 或被替换后删除
    test/               held-out 记录 e2e_s*.jsonl
```

看最近几代的结论：

```bash
python3 -c 'import json; s=json.load(open("output/micake-subagent-rsi/rsi_mixed/lineage.json"));
print("champion", s["champion"]["name"])
[print(g["name"], g.get("decision"), g.get("gate",{}).get("reasons")) for g in s["generations"][-5:]]'
```

日志里过滤资源心跳：`grep -v "resources watch" driver.log`。
