# 运维脚本

这几个脚本是 `../TRAIN_FROM_SCRATCH.md` 的可执行配套。**只提交了最有用的几个**，
不是我们跑七轮用过的全部工具。

**它们的价值在逻辑，不在路径。** 每一条判据都对应一次真实的机时浪费，写在各自的
文件头注释里。站点相关的部分（共享存储路径、节点名、镜像 tag）都提到了文件顶部、
可以用环境变量覆盖。

## 你必须自己提供的一个函数

四个 shell 脚本都依赖 `onnode`，契约是**"在第 N 台机器上跑一条命令"**：

```bash
onnode() { local n="$1"; shift; <你的方式> "$*"; }
```

我们用的是集群自带的 `spur exec`（定义在 `env.sh` 里，通过 `ONNODE_ENV` 指过去）。
换成 `ssh`、`srun`、`kubectl exec` 都可以。`monitor_training.sh` 还需要一个
`held`，契约是"列出当前持有的节点号"。

## 清单

| 脚本 | 干什么 | 里面编码的教训 |
|---|---|---|
| `launch_multinode.sh` | 并行起五个 rank | 顺序起会因节点发现超时失败；判断启动成功要看协调文件不要看 stdout；Mooncake 段大小只能从环境变量给 |
| `monitor_training.sh` | 看守 + 自动续跑 | **分副本**监控 `teacher_s`（平均值会掩盖"一个副本慢 35 倍"）；只读当前 run 的 step（跨 run 取最大值会让停滞判据永远成立）；同一故障签名重演就停下别重启 |
| `show_progress.sh` | 进度 / eval / loss 看板 | 每次刷新都重新解析 run 目录（自动重启会换目录）；三把 eval 尺子分列；非 tty 时不发 `clear` |
| `run_benchmark.sh` | benchmark 驱动 | 引擎会在跑到一半死掉且 `/health` 仍返回 200，所以必须超时退出 + 重启续跑 + 主动轮转限损 |
| `bench_report.py` | 和官方 draft 并排出表 | 分母只能用同机同镜像实测的官方 draft |
| `build_balanced_dataset.py` | 按类配额组装训练集 | 各类池子差两个数量级，"均衡"只能是"设上限、稀缺全取、充裕截断" |
| `lr_plan.py` | 推 cosine / WSD 的入口与收尾 lr | cosine 下入口和收尾绑死，续训时必须反解峰值 |

自检脚本在 `../selfcheck/`：

| 脚本 | 断言什么 |
|---|---|
| `test_collate_equivalence.py` | 新的按行取数 + collate 与旧的 pad-to-full-width 路径**逐元素**相同。纯张量，不需要 GPU |
| `verify_cache_format.py` | 预处理缓存存的是 int32 ndarray，且取出来的张量与旧 `list[int]` 路径逐元素相同 |

## 用法

```bash
# 训练
export ONNODE_ENV=/path/to/your/env.sh      # 提供 onnode()
export SHARED_ROOT=/your/shared/lumenrl
export DATASET_PATH=$SHARED_ROOT/datasets/<你的>/train.jsonl
export CKPT_DIR=$SHARED_ROOT/checkpoints/<你的>
export TOKEN_CACHE_DIR=$SHARED_ROOT/cache/<你的>
export RUN_TAG=myrun
bash launch_multinode.sh

# 看板（常驻，默认 20 分钟刷一次）
BASE_STEP=0 TARGET_STEP=<你的 num_training_steps> \
  WATCH=1 bash show_progress.sh

# 看守
RUN_GLOB='myrun-*' RUN_TAG=myrun bash monitor_training.sh

# benchmark
DRAFT=<导出目录> LABEL=mine SETS="gsm8k humaneval mtbench" bash run_benchmark.sh
BENCH_RESULTS=<结果目录> python3 bench_report.py mine
```

/!\ `show_progress.sh` 的 `B_EVAL` / `B_GEN5` / `B_GEN6` 是**起点基线**，来自开训前
那次 `VERIFY=1` 的 lr=0 读数。从零训没有基线，填 0 即可；续训必须填上一轮的记录，
否则看板上的"涨了多少"是错的。
