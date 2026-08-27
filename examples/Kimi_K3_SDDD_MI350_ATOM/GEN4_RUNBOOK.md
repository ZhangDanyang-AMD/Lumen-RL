# Kimi-K3 DSpark draft —— 从零到服务端数字的完整手册（gen-4）

这份文档的目标：**拿到一个干净的 MI355X 集群和这一份文档，就能从头做完一轮 draft 训练
并测出可信的服务端接受长度**。所有需要的命令、版本号、参数取值和它们的理由都在这里，
不引用任何其他文档。

读之前假设你懂分布式训练的一般概念（TP、FSDP、checkpoint、显存），但没见过这套代码、
没用过 ATOM、不了解投机解码。

**这一轮的结果**（细节见第 9、10 节）：draft 在与 gen-3 严格可比的固定 eval 切片上从
AL 1.6203 涨到 **2.0169**（+24.5%），服务端四集均值从 2.7647 涨到 **3.1682**（+14.6%），
达到官方参考 draft 在同一台机器上读数的 **85.4%**。

---

## 0. 这件事在做什么

训练一个 **DSpark draft 模型**（5 层，约 23.9 亿可训练参数），给 **Kimi-K3**
（93 层，hidden 7168，词表 163840，mxfp4 权重 1.56 TB）做投机解码加速。draft 训练好后由
ATOM 推理引擎加载，一次前向猜 7 个 token，target 批量校验，从而降低解码延迟。

衡量指标是 **接受长度 `tok/fwd`**：

```
tok/fwd = 1 + 已接受的 draft token / target 前向步数
```

那个 `1` 是 target 每步必出的 bonus token，所以即使接受率为 0，`tok/fwd` 也是 1.0；
猜 7 个 token 时上限是 8。这个量和官方 model card 上的 "acceptance length" 同尺度。

训练方式是 **离线蒸馏（off-policy prefill distillation）**：先用 K3 把数据集里所有回答
重新生成一遍存下来，训练时 teacher 只需要对这些已知 token 做 prefill 并交出中间层的
hidden state，不再需要在训练环里解码。draft 学的是"给定 target 在每个位置的 hidden
state，预测接下来 7 个 token"。

整个流程分四段，第 3–6 节各一段：

```
① 数据准备（CPU + 8 节点 GPU）  原始数据集 -> 去重去图文去污染 -> K3 重新生成 -> 逐行校验
② 训练（5 节点 GPU）            4 台 teacher 出 hidden state -> 1 台 8 卡训 draft
③ 导出（CPU）                   训练 checkpoint -> ATOM 能加载的 safetensors
④ benchmark（1 节点 GPU）       起 ATOM 服务 -> 跑题目集 -> 读 /debug/mtp_stats
```

---

## 1. 集群与访问方式

### 1.1 硬件

| 项 | 值 |
|---|---|
| GPU | 8 × AMD Instinct MI355X（gfx950），单卡 VRAM **288 GiB**（309,220,868,096 B） |
| CPU | 236 核 |
| 内存 | 2.8 TB |
| 本地盘 | `/mnt/m2m_nobackup`，28 TB NVMe，**每节点独立** |
| 共享盘 | `/shared_nfs`，300 TB（约 29 TB 可用），所有节点可见 |
| 家目录 | `/home`，10 TB NFS 共享，**但通常 98% 满，不要往里放大文件** |
| `/dev/shm` | 1.4 TB tmpfs |
| 网卡 | `ionic_0..6`（7 张 RoCE）+ `mlx5_0`。**注意没有 `ionic_7`** |

**GPU 数量不能少于 8。** `tensor_parallel_size: 8` 是硬约束——K3 权重切成 8 份才刚好
塞得下（每卡约 217.6 GiB），卡更少直接放不进去。

### 1.2 调度器：spur，不要 ssh 计算节点

计算节点被管理员加了 `AllowUsers ubuntu root`，普通用户 `ssh` 一律 `Permission denied`。
一律用 `spur exec <JobID> bash -lc "..."`。

```bash
# 登录节点上先确认链路
hostname -f
echo "$SPUR_CONTROLLER_ADDR"     # 这套 v2 集群是 http://crs-m2m-cpu-spur-v2-001.crusoe.amd.com:6817
squeue -u "$USER"
```

申请节点（**必须带 `--exclusive`**，否则别的作业能落到同一台机器的空闲 CPU 上）：

```bash
sbatch --parsable -J k3_035 -A <your-account> -p default -N1 \
       --exclusive -G 8 -w crsuse2-m2m-v2-035 -t 168:00:00 \
       -o ~/logs/slurm_035.out --wrap "sleep 604800"
```

`--exclusive` 和 `-G 8` 都要写：前者占住 CPU 和内存，后者占住 8 张卡。只写
`--exclusive` 的作业在 `sinfo` 里会显示 `mix` 而不是 `alloc`。**spur 不支持在运行中改
这个属性**（`scontrol update JobId=N OverSubscribe=EXCLUSIVE` 只会回 `unknown update
key`），所以申请时写错就只能取消重申请。

一个便利的封装（后文的 `onnode` / `onall` 都指它）：

```bash
jobid() { cat ~/jobs/$1.jobid; }
onnode() { local n="$1"; shift; spur exec "$(jobid "$n")" bash -lc "$*"; }
onall()  { for n in $NODES; do ( onnode "$n" "$@" 2>&1 | sed "s/^/[$n] /" ) & done; wait; }
```

### 1.3 三个会浪费你时间的环境细节

- **`spur exec` 的 shell 里 `$HOME` 是 `/opt/spur`，不可写。** `docker build` 会因为
  建不了配置目录而失败（`mkdir /opt/spur/.docker: permission denied`）。build 前设
  `export HOME=/tmp/xxx DOCKER_CONFIG=/tmp/xxx/.docker`。
- **`spur exec` 的容器有自己私有的 `/dev/shm`。** 你在 spur shell 里 `ls /dev/shm` 看到的
  和 docker 容器里 `-v /dev/shm:/dev/shm` 挂到的**不是同一个**（后者是宿主机的）。
  找 teacher worker 日志要 `docker exec <容器> ls /dev/shm/...`。
- **`spur exec` 里如果留下了后台进程，这条命令要约 100 秒才返回。** 并行起多个节点时
  必须把这些调用并行化，否则第一个和最后一个节点会差好几分钟（见 5.3）。

---

## 2. 镜像：这一节跳过去后面全是白费

ATOM 对版本极其敏感。**`rocm/atom-dev:latest` 是滚动 tag，已经漂过至少两次，每次都
以不同的方式弄坏这套流程。** 必须按 digest 钉死。

### 2.1 基础镜像

| 项 | 值 |
|---|---|
| 镜像 | `rocm/atom-dev@sha256:2f8bd4206ad15d014ae48115eae1ee9f1db83781848a8542de7177cfbd4ac914` |
| 也可用 tag | `rocm/atom-dev:nightly_202608141640` |
| ATOM | `0.1.6rc1.dev275+g83e71001e`（commit `83e71001e94602ee8e1d04581809bded49c3f08b`） |
| torch | `2.13.0+rocm7.14.0` |
| HIP | `7.14.60850` |
| python | 3.12 |
| ATOM 路径 | `/app/ATOM` |

拉取并逐项核对，四项全对才往下走：

```bash
IMAGE=rocm/atom-dev:nightly_202608141640
docker pull $IMAGE
docker images --digests | grep nightly_202608141640     # 期望上面那个 sha256

# atom.__version__ 在这个 build 上会抛 AttributeError，必须走 importlib.metadata
docker run --rm --entrypoint python $IMAGE -c "
import atom, torch, subprocess
from importlib.metadata import version
print('atom  :', version('atom'))
print('torch :', torch.__version__)
print('hip   :', torch.version.hip)
print('path  :', atom.__file__)
print('commit:', subprocess.run(['git','-C','/app/ATOM','rev-parse','HEAD'],
      capture_output=True, text=True).stdout.strip())
"
```

`path` 必须落在 `/app/ATOM` 下。如果打出来是 `third_party/ATOM` 里的路径，说明
`PYTHONPATH` 被污染了——镜像里 `third_party/ATOM` 是刻意不进 `PYTHONPATH` 的，
一旦进去就会遮蔽 `/app/ATOM`，你跑的就不是校验过的那份 ATOM。

### 2.2 训练镜像

训练镜像 = 上面这个基础镜像 + LumenRL 侧的东西。**从仓库根目录构建**：

```bash
cd /path/to/Lumen-RL
HOME=/tmp/k3build DOCKER_CONFIG=/tmp/k3build/.docker \
docker build -f examples/Kimi_K3_SDDD_MI350_ATOM/docker/Dockerfile \
             -t kimi_k3_dspark_atom:pinned .
```

Dockerfile 第一步就断言基础镜像的 atom/torch/HIP/commit 四项，对不上直接 build 失败。
这条断言是有来历的：**2026-08-24 用 `latest` 构建出来的镜像坐在
atom `0.1.6rc1.dev323` + torch `2.10.0+rocm7.2.4` 上——ATOM 比验证过的版本新 48 个提交，
而 torch/ROCm 反而更旧。在那个栈上 teacher 会在第二次 hidden state 抽取时死锁：
8 个 TP rank 全部卡在一个 2 元素的 NCCL BROADCAST 上，`last enqueued work: 7, last
completed work: 6`，GPU 显示 100% 但功耗只有 320 W（自旋，不是在算），600 秒后
`timed out waiting for response to cmd=extract_hidden`，零训练步。** 换回钉死的 digest
后一次通过。

### 2.3 让五台机器跑同一份镜像

**不要在每台机器上各 build 一次。** 各自 build 出来的 image ID 不同，pip 解析也可能不同。
在一台上 build，然后 save/load 分发：

```bash
onnode 035 'docker save kimi_k3_dspark_atom:pinned -o /shared_nfs/<you>/images/k3train.tar'
for n in 037 038 039 040; do
  onnode $n 'docker load -i /shared_nfs/<you>/images/k3train.tar' &
done; wait
onall 'docker images --format "{{.ID}}" kimi_k3_dspark_atom:pinned'   # 五个 ID 必须一样
```

### 2.4 一个不能漏的环境变量

**绝对不要设 `PYTORCH_CUDA_ALLOC_CONF`。** `expandable_segments:True` 会让 ROCm 的虚拟
保留不被 `empty_cache()` 归还，可用显存从 265 GiB 掉到 175 GiB，teacher 起不来。启动
脚本里要显式 unset，并且启动后从容器里读回确认它确实没被设置——它会从调用方的 shell
继承进来。

---

## 3. 数据准备

### 3.1 权重

来源必须是 `moonshotai/Kimi-K3` 官方仓库，并且钉住 revision：

| 项 | 值 |
|---|---|
| repo | `moonshotai/Kimi-K3` |
| revision | `a590ce090cb049c93a33dfe8c208ec652aa20503` |
| 大小 | 1.561 TB，96 个 safetensors |

**落每台机器的本地 NVMe（`/mnt/m2m_nobackup`），不要放 NFS。** NFS 冷读约 200 MiB/s，
完全冷启动约 2.2 小时；本地 NVMe 约 4 分钟。这个差值要乘上被抢卡重启的次数。

下完一定做**字节级校验**，别只看文件数——1.5 TB 的下载中断很常见，缺一个 shard 的表现
是加载到 90% 才报错，白等几十分钟。

```bash
onnode 035 'ls /mnt/m2m_nobackup/<you>/models/Kimi-K3/*.safetensors | wc -l'   # 必须 96
```

### 3.2 prompt 池：去重、去图文、去评测集污染

源数据（本轮）：

| repo | 用到的 split |
|---|---|
| `nvidia/Nemotron-Post-Training-Dataset-v2`（gated，要 HF token） | `chat` `code` `math` `stem` `multilingual_ja` `_de` `_it` `_es` `_fr` |
| `CohereLabs/aya_dataset` | `train` |

三件事必须在**采样之前**做完，顺序不能反：

**一、图文过滤。** K3 是多模态模型，但这条生成路径走纯文本 token，从来拿不到图，
所以图文样本的回答一定是幻觉——模型会自信地描述一张它没看见的图。只滤真正有问题的四类：

| 类别 | 判据 |
|---|---|
| content 不是字符串 | `type(content) is not str` |
| 图文 JSON 外壳 | 匹配 `^\s*\[\s*\{\s*"type"\s*:` |
| 模型图片 token | 含 `<\|image\|>` / `<\|media_begin\|>` / `<\|vision_start\|>` |
| `<image>` 占位符 | 正文里出现 `<image>` |

**不要把 `<img` / `image_url` / `data:image/` 放进黑名单。** 早期版本这么写，丢掉了 2.55%
的数据，逐条抽查发现绝大部分是误伤：HTML 代码题的 `<img` 标签、REST API 的 `image_url`
字段名、JSON 请求体示例里的 `data:image/png;base64,...`。这些是纯文本编程题，内容完全正常。
判据要基于"这一行是否真的需要一张图才能回答"，而不是"这一行里是否出现过跟图片有关的字符串"。

**二、去重，而且必须是"入库即去重"。** 按 prompt 的 NFKC 正规化 + 转小写 + 折叠连续空白
之后取哈希，维护一个**跨所有 split 的全局 `seen` 集合**，在采样选中的当时就判重。

先采样再去重会踩坑：`code` split 期望 50,000 只拿到 27,236 行，因为重复 prompt 被反复
选中、去重后大量落空。跨 split 去重也是必要的而非保险——五个 multilingual split 之间
有大量重叠。**处理顺序决定谁赢**：先处理的占住哈希。所以先英文四类，再 aya（人工撰写），
最后才是 Nemotron 那五个合成 multilingual，让质量高的那份留下来。

**三、评测集去污染。** 对要测的 benchmark 抓 eval prompt，建 13-gram 索引，命中即剔除。
n-gram 取 13 是折中：更短会误伤（常见代码片段撞车），更长会漏（改写过的题）。数量看着小
（本轮共剔除约 1.4 万条），但留着的后果是评测数字虚高且无法察觉。

**实测的漏斗**（这些数字本身就是有用的先验）：

| split | 源行数 | 唯一 prompt | 留存率 |
|---|---|---|---|
| chat | 627,720 | 439,845 | 70.1% |
| code | 175,000 | **31,261** | **17.9%** |
| math | 239,467 | 101,962 | 42.6% |
| stem | 355,000 | 354,758 | 99.9% |
| aya | 202,362 | 165,863 | 82.0% |
| multilingual ×5 | 4,944,227 | 4,453,283 | 90.1% |
| **合计** | **6,543,776** | **5,546,972** | 84.8% |

`code` 只留 17.9% 是这张表最重要的一行：175,000 行里只有 31,261 条唯一 prompt
（同一题平均 5.6 个回答）。**这是物理上限，不是选择**，它直接决定了代码类数据的天花板。

**不加配额全量走一遍要多久**：555 万行按实测吞吐（每节点约 2,100 tok/s，mean_gen 约
1,600）在 8 节点上约 **6.3 天**。所以配额本质上是算力约束而不是数据约束。本轮取 40 万，
按"英文优先吃满上限"分配：chat/math/stem 各 82,913，code 吃满 31,261，aya 42,000，
五个 multilingual 各 15,600。

### 3.3 用 K3 重新生成所有回答

**为什么要重新生成**：SDDD 的监督信号是 teacher 的 hidden state，draft 学的是 target
真实会说的话。数据集里原本的回答是别的模型写的，用它训出来的 draft 会去猜一个 target
永远不会产出的分布。

**渲染 prompt** 必须复用训练侧同一个 parser（`lumenrl/data/kimi_k3_parser.py` 的
`KimiK3Parser`），不要自己拼 chat template——训练时会用同一个 parser 重新渲染
prompt+response，两边必须逐 byte 一致。

`thinking_effort` 什么都不用传：K3 的 `tokenization_kimi.py` 里有
`kwargs.setdefault("thinking_effort", "max")`，不传就是 max，实测传与不传渲染结果逐 byte 相同。

`max_prompt_tokens` 用 **14304 = 16384 − 2048 − 32**，含义是保证每行至少还剩 2048 token
作答。**不要把训练侧的 `max_prompt_tokens` 抄过来**：训练期它是个筛子随时能改，生成期
它决定哪些行会被生成，丢掉的行要重跑 GPU 才能拿回来。

**生成引擎配置**（每个数都是实测的）：

```
max_model_len          16384
max_num_seqs           384      # 不是保守，是实测更快，见下
max_num_batched_tokens 32768    # 必须 >= max_model_len
gpu_memory_utilization 0.93
kv_cache_dtype         fp8      # ATOM 只接受 "bf16" | "fp8"
enforce_eager          False    # CUDA graph 只覆盖 decode，值 2.4x
enable_prefix_caching  False
enable_chunked_prefill False
tensor_parallel_size   8
temperature            0.0
speculative_config     不传     # 关掉 draft，实测快 2x
max_tokens             逐行算 = 16384 - len(prompt) - 32
```

三个必须解释的取值：

**`kv_cache_dtype: fp8` 是硬要求。** 16384 窗口下 MLA 的 paged KV 是主要开销：K3 的 MLA
entry 是 `kv_lora_rank + qk_rope_head_dim = 512 + 64 = 576` 元素/token/层 × 24 个
full-attn 层，bf16 是 27 KB/token，fp8 是 13.8 KB/token —— **fp8 直接把可用 window 翻倍**。
（注意这和训练时用 fp8 的理由不同，见 5.2。）

**`max_num_seqs: 384` 而不是 512。** K3 有 69 个 KDA（线性注意力）层，它的循环状态池按
`max_num_seqs` 线性吃显存（实测 56.17 MB/槽），和 paged KV 池抢同一份
`available_for_kv ≈ 44.9 GB`：512 槽位时状态池 26.78 GB、只剩 86,055 个 block，长 prompt
下需求溢出 24%，有效并发反而只有 279；384 槽位时状态池 20.09 GB、有 120,425 个 block，
实际并发就是 384。引擎启动时会打印 "Concurrent capacity vs context length" 表，看
`bound by slots` 还是 `bound by blocks` 就行，别自己估。

**关掉 draft。** 打开官方 DSpark draft 会让 KDA 状态池 ×8（`entries_per_req = 1 + num_spec`，
给每个投机 token 留一个回滚槽），并发从 512 压到 64，吞吐从约 3000 tok/s 掉到约 1500。
投机把前向次数省了 3.64 倍，但 MoE 解码吞吐对 batch size 非常敏感，并发掉 8 倍亏得更多。

**生成脚本的两个设计要点，都不是可选项：**

1. **滚动提交，不是 sweep-and-wait。** 一个 sweep 要等最慢的序列跑完，长 cap 下尾部会
   退化成几条序列独占 8 张卡。要始终维持 `max_num_seqs` 条在飞，完成一条补一条。
2. **自己调 `io_processor.preprocess()` 拿 Sequence 对象建行映射**，而不是用
   `engine.generate()`。后者把完成的序列按内部 id 排序再和输入顺序 zip，假设 id 的排序
   等于提交顺序——配错一行就是静默的训练数据污染。要在每条完成时校验
   `num_tokens_input` 和自己发出的 prompt 长度一致，不一致立刻停。

**多节点切分按 stride，不按连续区间**（第 i 行给 shard `i % N`）。源数据是按 split 分块
拼的，连续区间切会让每个 shard 的组成完全不同、ETA 全失真。

**要有 watchdog 和 supervisor。** ATOM/RCCL 在长时间多卡运行下会偶发死锁：某个 broadcast
卡住，torch watchdog 480 秒后报 "got stuck"，8 个 rank abort，而**父进程阻塞在
`engine.step()` 里永远不返回，`docker ps` 看着一切正常**。曾经因此白转 8 小时。两层修复：
脚本内独立线程的 watchdog（必须是独立线程 + `os._exit`，主线程已经卡死在 C 扩展里，
任何依赖它的退出路径都不会生效），外层 supervisor 负责拉起来跑到完为止。

### 3.4 转成训练数据集：重点是校验，不是转换

K3 在 `thinking=true` 下的 completion 形状是：

```
<think 正文><|close|>think<|sep|><|open|>response<|sep|><response 正文><|close|>response<|sep|><|close|>message<|sep|><|end_of_msg|>
```

按这三个标记切开，写回 `reasoning_content` + `content`。**这一步的重点是校验**：训练吃的
是 LumenRL 重新渲染出来的 input_ids，不是我们存的 completion_ids，中间隔着
「completion 文本 → 拆 reasoning/content → 塞回 conversation → 重渲染 → 重 tokenize」
这一整条路，任何一步不保真，draft 学到的就是 K3 从没产出过的 token，而且没人会报错。

**每一行都验，验不过就丢。** 抽样验不够——踩到过的两个 bug（`strip()` 吃空格、工具调用
被静默吞掉）在抽样里都只表现为个位数。切分**不做 strip**，K3 会在
`... Done. <|close|>think<|sep|>` 这样的位置留空格，strip 掉就少一个 token。只接受
「think → response → 结束」这一种规整形状，带工具调用的整行丢掉。

**实测丢弃率 8.8%**（399,842 → 364,767），构成：

| 原因 | 占比 | 含义 |
|---|---|---|
| `no_think_close` | 6.14% | 顶到生成上限，think 段没闭合 |
| `response_mismatch` | 2.21% | 重新分词后 token 序列与生成时不一致 |
| `no_response_close` | 0.32% | 顶到上限，response 段没闭合 |
| 其余两类 | 0.10% | 首个 channel 不是 response / 出现额外 channel |

`response_mismatch` 不是 bug，是编码/生成不对称：BPE 编码看整串按合并优先级贪心（唯一解），
而模型逐 token 往外吐、不受规范切分约束。例如文本 `FORMCHECK`，BPE 规范编码是
`['FORM','CHECK']`，K3 实际吐的是 `['FOR','MC','HECK']`——解码回文本 100% 相同，但
重新编码就回不到原路径。所以"存成文本"这一步本身就是有损的，不是校验太严。结论是丢弃：
这份数据集相对"随便重跑一遍"的唯一增量就是逐行验证过的 token 保真，破了这条没法向下游解释。

**丢弃率超过阈值就非零退出，不要直接调大阈值**，先看原因分布。

---

## 4. draft 的初始权重

从零训和接着已有 draft 训是两条路。本轮是**接着 gen-3 训**，所以要把已发布的
safetensors 导回训练器。

**这里有一个会静默毁掉整轮训练的坑。** 通用的 `draft.resume_from` 分支是给 Eagle3 写的，
它把 `midlayer.` 改成 `layers.0.`、`norm` 改成 `out_norm`。拿它加载 DSpark 的导出，
每个张量都变成 unexpected key，`strict=False` 保留随机初始化，**什么都不报错**——loss
从冷启动值开始，看起来只是"收敛得慢"。

正确的映射（ATOM 命名 → LumenRL 命名）：

| 导出里的名字 | 模型里的名字 |
|---|---|
| `context_proj.weight` | `fc.weight` |
| `context_norm.weight` | `hidden_norm.weight` |
| `final_norm.weight` | `norm.weight` |
| `layers.*` / `markov_head.*` / `confidence_head.*` | 同名 |
| `embed_tokens.weight` | **跳过**（teacher 的冻结 embedding，训练器从 teacher 拿） |

而且要**断言**：每个导出张量都必须有归宿，每个模型参数都必须被填上，形状必须一致，
任何一条不满足就报错退出。加载成功的标志是 **67 个张量 / 2,387,907,841 个参数**。

**上线前用 `lr=0` 验一次。** 跑两步、学习率设 0，eval 出来的 AL 必须和上一代记录的数字
一致。本轮实测 1.61526，而 gen-3 训练曲线在 step 3280 记录的也是 1.61526——逐位一致，
一次性验完了权重导入、teacher hidden state、eval 分片三条链路。这个检查花 12 分钟，
能挡住的错误值好几天。

---

## 5. 训练

### 5.1 拓扑

```
teacher0 TP=8 ─┐
teacher1 TP=8 ─┼─ Mooncake TCP ─> draft 节点 8 个 rank（FSDP2 replicate）
teacher2 TP=8 ─┤                   每个 rank 取 128 条里属于自己的 16 条
teacher3 TP=8 ─┘
```

四台常驻 ATOM TP=8 teacher，整批数据轮流分给它们（`batch_id % 4`），每台对完整序列做
prefill 后把每条序列的 hidden state 发布到 Mooncake；Ray 只传 token、mask 和 key，
hidden state 走 Mooncake 数据面。第五台跑 `torchrun --nproc-per-node=8` 训 draft。

**为什么不能单节点。** teacher 以 TP=8 起来后每张卡占约 217.6 GiB，draft 训练无法同时
驻留，单节点方案必须让两者轮流上下卡（一轮 22 分钟里三分之二花在传输和重载）。分离之后
teacher 常驻、draft 常驻，各自不用让位。

### 5.2 配置：动了会出事的值

配置在 `configs/train.yaml`。

**序列与批次**

| 值 | 设定 | 为什么 |
|---|---|---|
| `policy.max_total_sequence_length` | 8192 | 与生成窗口和 teacher 的 `max_model_len` 一致 |
| `train_global_batch_size` | 128 | TorchSpec 参考配方，和 lr/max_grad_norm 是一组，要改一起改 |
| `train_micro_batch_size` | 1 | |
| `spec_length` | 7 | 和服务时的 `--num-speculative-tokens 7` 对齐 |

**teacher（ATOM）**

| 值 | 设定 | 为什么 |
|---|---|---|
| `tensor_parallel_size` | 8 | 硬约束 |
| `max_model_len` | 8192 | 和训练窗口一致 |
| `max_num_seqs` | 32 | KDA 循环状态随它线性增长，256 会把显存吃穿 |
| `max_num_batched_tokens` | **8192** | **必须 >= `max_model_len`**。它同时决定 prefill 的激活峰值：32768 时约 14–18 GiB，8192 时约 7–9 GiB |
| `enable_prefix_caching` | false | 前缀命中不计入 scheduled_tokens，hidden state 会缺 |
| `enable_chunked_prefill` | false | ATOM 每个调度步用同一个 key 重写，不看 `is_final_chunk`，切开就只剩最后一块 |
| `gpu_memory_utilization` | 0.90 | |
| `kv_cache_dtype` / `index_cache_dtype` | **fp8** | 见下 |
| `enforce_eager` | true | |
| `generate_mode` | prefill | 离线蒸馏，不做自回归生成 |

**训练时 KV 用 fp8 的理由和生成时不同，而且是相对更早版本的一次反转。** 早期版本用
bf16，理由是 fp8 量化会给 label 引入噪音。但**服务栈用的就是 fp8——训练时的 teacher
必须和服务时的 teacher 是同一个函数**，否则 draft 学的是一个上线后不存在的分布。
正确性优先于噪音。

**draft 与损失**

| 值 | 设定 |
|---|---|
| `num_layers` / `num_heads` / `head_dim` / `ffn_dim` | 5 / 64 / 128 / 14336 |
| `q_lora_rank` / `kv_lora_rank` | 1536 / 512 |
| `qk_nope_head_dim` / `qk_rope_head_dim` / `v_head_dim` | 128 / 64 / 128 |
| `rope_theta` | 50000.0 |
| `rope_scaling` | yarn，factor 32.0，original_max_pos 32768，beta_fast 32.0，beta_slow 1.0，**mscale 1.0 / mscale_all_dim 1.0** |
| `aux_hidden_state_layer_ids` | `[2, 23, 47, 71, 89]`（0-based，取第 i 层的输出） |
| `capture_mode` | postnorm |
| `separate_last_hidden` | **false** |
| `anchor_num` | 512（峰值显存的主要来源，实测 57.8 GiB） |
| `ce_loss_alpha` / `l1_loss_alpha` / `confidence_loss_alpha` / `loss_decay_gamma` | 0.1 / 0.9 / 1.0 / 4.0 |

**`separate_last_hidden` 在 ATOM 上必须是 false，和 vLLM 变体相反**：vLLM 把 5 个 aux 层
拆成「前 4 个 → hidden_states，第 5 个 → last_hidden_states」且是 pre-norm，需要训练器
自己拼回去再做 RMSNorm；ATOM 的 `hidden_states` 已经装了全部 5 层（35840 宽），
`last_hidden_states` 是模型最终输出且已经过 final norm。设成 true 会让 `fc` 收到
`6×7168=43008` 而不是 `5×7168`，直接形状错误，而且 norm 被做两次。

**优化器**

| 值 | 设定 |
|---|---|
| `learning_rate` | 5.0e-5 峰值，cosine |
| `warmup_ratio` | 0.04 |
| `max_grad_norm` | 1.0 |
| `min_lr` | **3.93e-5，不是 0** |

**`min_lr` 不设 0 是这一轮最重要的配置决定。** 上一代用 min_lr 0 跑单个 epoch，余弦把 lr
带到 7.6e-10，eval AL 从 step 3300 起在 1.613–1.617 平台化——那是 schedule 造成的假平台，
不是数据用尽，而且让"接着再训一个 epoch"变得没有意义（必须重开一段 lr schedule）。
本轮取"三个 epoch 的余弦跑到第一个 epoch 末尾"的值：

```
decay_ratio = (1 − 0.04×3) / (3 − 0.04×3) = 0.3056
coeff       = 0.5 × (1 + cos(0.3056π)) = 0.7868
min_lr      = 0.7868 × 5e-5 = 3.93e-5
```

全程几乎不退火，所以下一个 epoch 可以直接接着跑。

**num_training_steps 必须实测，不能估。** 在 CPU 上跑一次和 trainer 完全相同的预处理：

```bash
python3 selfcheck/preprocess_dataset.py \
    --dataset <你的 train.jsonl> --tokenizer <K3 路径> \
    --max-length 8192 --max-prompt-tokens 0 \
    --thinking true --last-turn-loss-only true --min-loss-tokens 14 \
    --eval-samples 128 --batch-size 128 \
    --cache-dir <共享路径>/tokens --workers 96
```

**每个参数都进 cache key**（数据集路径 + 文件大小 + mtime、tokenizer 路径、max_length、
chat_template、last_turn_loss_only、min_loss_tokens、split、PARSER_VERSION、drop_overlong、
max_prompt_tokens、thinking），差一个就会生出第二份缓存、正式训练再把预处理的钱付一遍。
`num_workers` 不在 key 里。

本轮实测：364,767 行 → **345,476 行存活**（8192 窗口丢 5.3%），
`num_training_steps = floor((345476 − 128) / 128) = 2698`。

**`max_prompt_tokens: 0` + `drop_overlong: true` 时超长行是整行丢弃，不是截断。**
（`check_total = drop_overlong and max_prompt_tokens <= 0`，mpt=0 时恒为 True。）
如果想改成截断，把 mpt 设成正数。

**eval：必须用两把尺子。** eval 集默认取训练集尾部，**换数据集就等于换评测基准**，
两代之间不可比。曾经有人因此把"数据更多更好的一轮 AL 更低"当成退化去查。做法：

- `eval/*`：本轮训练集尾部 128 行（新分布）
- `eval_old/*`：把上一代那 128 行**原样带过来**（同一份旧数据集、同一 seed、同一预处理
  参数），用一个离线脚本materialise 成固定文件，并记录行内容的指纹

```yaml
eval:
  enabled: true
  interval: 10
  num_samples: 128          # 必须是 world_size × micro_batch_size 的倍数
  micro_batch_size: 2
  extra_slices:
    - path: <共享路径>/eval/gen3_old_128.pt
      prefix: eval_old
```

**checkpoint**：`save_steps: 100`、`save_total_limit: 30`（一次 33 GB，全留 27 份约
900 GB，放共享盘）。一次保存约 55 秒。**注意 `num_training_steps` 如果不是 `save_steps`
的倍数，训练结束时不会补存最后一份**——本轮 2698 步的最后一份是 step 2600，最后 98 步
没落盘（实际损失可忽略，两者 AL 差 0.004，但这是个应该修的缺陷）。

### 5.3 Mooncake 走 TCP，不要 RDMA

这是本轮花时间最多的一个发现。

**症状**：teacher actor 的 `extract_hidden` 600 秒超时，零训练步，`docker ps` 一切正常。

**根因**：默认配置要求一个 teacher 进程注册 **128 个各自独立的 2 GiB RDMA 段**（因为
Ionic 不接受超过 2 GiB 的单个 MR，所以只能靠"多开几个小段"凑容量）。但实测
**这套集群的 Ionic HCA 上，一个进程每张网卡只能建一个 RDMA client**：

```
--devices ionic_0 --stores 3
  store 1/3   1.66s   ok
  store 2     Failed to register memory 0x...: Invalid argument [22]
              -> Mooncake setup failed (error=-600)
```

七张网卡轮着开时前 7 个成功、第 8 个失败（`index % 7` 绕回 `ionic_0`）。所以段容量上限是
**7 × 2 GiB = 14 GiB**，而一个 batch 的 128 条序列要发约 **20 GiB**。更要命的是 manifest
是等 128 条**全部**写完才发布的，消费端拿不到 manifest 就腾不出空间——写的人等空间、
取的人等清单，死锁到超时。

**改成 TCP，问题从设计里消失**：TCP 段不需要注册给网卡，可以直接开 128 GB。

```bash
MOONCAKE_PROTOCOL=tcp
MOONCAKE_DEVICE_NAME=                       # 空
MOONCAKE_GLOBAL_SEGMENT_SIZE=128GB
MOONCAKE_LOCAL_BUFFER_SIZE=8GB
LUMENRL_TEACHER_MOONCAKE_SEGMENT_POOL_SIZE=1     # 段池坍缩成 1
LUMENRL_TEACHER_MOONCAKE_SEGMENT_SIZE=128GB
LUMENRL_DRAFT_MOONCAKE_SEGMENT_SIZE=4GB          # draft 只读，段可以小
```

**NCCL 仍然走 RDMA**（draft 节点内 8 卡的梯度 all-reduce），
`NCCL_IB_HCA=ionic_0,...,ionic_6`——注意是 0 到 6，**没有 ionic_7**，写错会有一路 fallback。

实测带宽（128 条 × 平均 1832 token = 每条 150 MiB，共 18.8 GiB）：

| | 时间 | 带宽 |
|---|---|---|
| teacher put（写自己的段） | 9.5 s | 2.0 GiB/s |
| draft 8 并发 get（跨节点） | 3.6 s | 5.27 GiB/s |

### 5.4 启动

**如果五个节点是五个独立的单节点作业**（而不是一个 `-N5` 分配），不能直接
`srun --nodes=5`——那会排队等自己占着的节点。启动脚本只读 `SLURM_PROCID` 和
`SLURM_NTASKS` 并通过共享盘上的文件协调，所以用五个 `spur exec` 手工设这两个变量即可。

**必须并行起，而且要放宽节点发现的超时。** `spur exec` 在留下后台进程时约 100 秒才返回，
顺序起五个会让第一个和最后一个节点的协调文件差 8 分钟，而默认的发现窗口只有 120 秒。

```bash
for rank in 0 1 2 3 4; do
  onnode "${NODES[$rank]}" "
      cd /path/to/Lumen-RL &&
      SLURM_PROCID=$rank SLURM_NTASKS=5 LUMENRL_RUN_ID=$RUN_ID \
      DATA_ROOT=... SHARED_ROOT=... MODEL_PATH=... DATASET_PATH=... \
      DOCKER_IMAGE=kimi_k3_dspark_atom:pinned \
      setsid nohup bash examples/Kimi_K3_SDDD_MI350_ATOM/run_multinode_rank.sh \
          > $LOG_DIR/rank-$rank.out 2>&1 < /dev/null &
      sleep 1" &
done
wait
```

**Ray 的端口要挪开。** 容器用 `--network host`，Ray 默认的 client server 端口 10001
很容易被同机的其他租户占着，`ray start --head` 会以
`Failed to bind to address 127.0.0.1:10001` 退出，而外层只看到 "Ray cluster has 0/5
active nodes"。用 `--ray-client-server-port 26380 --include-dashboard=false`，
主端口也挪到 26379。

**要有看守进程。** 三种失败都值得自动处理：训练进程死亡、进程活着但 step 计数器不动
（那次 RCCL 死锁就是 `docker ps` 正常、GPU 显示 100% 但功耗只有 320 W 的自旋）、
spur 作业消失。恢复动作都一样：干净停掉，用新的 run id 加 `checkpointing.resume=true`
重新拉起。**但看守必须自检恢复是否真的接上了**：比较重启后第一条 step 行和重启前的
checkpoint 编号，接不上要明确报出来——一个"成功恢复"但实际从头开始的重启，会让你在
十几个小时后才发现。

---

## 6. 导出

```bash
python3 output/export_dspark_hf.py \
    --ckpt <共享路径>/checkpoints/.../checkpoint_2600.pt \
    --base-model <K3 路径> \
    --output <共享路径>/drafts/gen4-step2600 \
    --train-config examples/Kimi_K3_SDDD_MI350_ATOM/configs/train.yaml
```

导出脚本做两件断言，**都不能跳过**：

1. **config.json 与训练 YAML 交叉核对**。特别是不能emit `rope_interleave`——ATOM 会把它
   映射成 `is_neox_style`，也就是训练器没在用的那种半分旋转。
2. **68 个张量名的 key 契约**。ATOM 对不认识的张量名不告警不报错，直接静默忽略。
   曾经导出过一份用 LumenRL 内部命名（`fc` / `hidden_norm` / `norm`）的权重，ATOM 把躯干
   加载了、**静默丢掉了整条输入通路**，结果是 first-token 准确率 59% 的 draft 在服务端
   接受率 **0.00%，3060 个前向步一次都没中**，而 server 日志从头到尾完全健康。

成功的标志：`ATOM key contract: OK (68 tensors)`、`Total size: 7.12 GB`。

---

## 7. Benchmark

### 7.1 起服务

```bash
docker run -d --name atom-dspark \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  --security-opt seccomp=unconfined --cap-add=SYS_PTRACE \
  --ipc=host --shm-size 128g --network host \
  -v <K3 路径>:/target:ro -v <draft 路径>:/draft:ro \
  rocm/atom-dev@sha256:2f8bd4206ad15d014ae48115eae1ee9f1db83781848a8542de7177cfbd4ac914 \
  python -m atom.entrypoints.openai_server \
    --model /target --served-model-name Kimi-K3 \
    --method dspark --draft-model /draft --num-speculative-tokens 7 \
    --kv_cache_dtype fp8 -tp 8 --trust-remote-code \
    --max-model-len 16384 --max-num-seqs 8 --max-num-batched-tokens 10240 \
    --gpu-memory-utilization 0.93 --block-size 128 \
    --no-enable_prefix_caching --server-port 8000
```

`max_num_seqs` 用 8 就够——客户端是串行的，而 KDA 循环状态按槽位分配，64 时状态池要
28.91 GB 而 KV 预算只有约 19 GB，引擎会拒绝启动。服务端约 271 秒起来（本地 NVMe）。
启动后日志里应该有 `Detected MLA DSpark drafter` 和
`DSparkProposer aux capture on target layers: (2, 23, 47, 71, 89)`。

### 7.2 测量协议

接受长度从引擎自己的计数器 `/debug/mtp_stats` 读，**不要从时间反推**。计数器是自服务启动
以来累计的，所以每个题目集的测量是"跑之前拍一张快照、跑完再拍一张、取差值"——这样多个
题目集可以共用一个服务（K3 加载一次要 4.5 分钟，每集重启一次纯属浪费）。

从累计的接受长度直方图 `{k: 次数}` 就能算出全部指标：

```
前向步数    = Σ 次数
已接受 token = Σ k × 次数
tok/fwd     = 1 + 已接受 / 前向步数
接受率      = 已接受 / (前向步数 × num_spec)
```

协议：`temperature=0`、并发 1、`num_speculative_tokens=7`、fp8 KV、Kimi-K3 chat template、
官方 prompt 数。

**客户端必须能扛住引擎卡死**：单请求超时（超时即以特定退出码上报"引擎卡死"）、每 25 条
把计数器增量折叠进累计分布并落盘、`--resume` 跳过已完成条目。有过一次 aiter 跨 rank
广播死锁，8 个 rank 自旋、`/health` 仍返回 200（那是 API 前端，和 engine core 是两个
进程），而客户端把 550 条已完成的工作全丢了。

### 7.3 题目集

官方 card 的 14 个 benchmark 里，本栈能测 13 个：

| benchmark | 来源 | prompts |
|---|---|---|
| GSM8K | `openai/gsm8k` main/test | 1319 |
| MATH-500 | `HuggingFaceH4/MATH-500` test | 500 |
| AIME 2026 | `MathArena/aime_2026` | 30 |
| HumanEval | `openai/openai_humaneval` test | 164 |
| MBPP | `google-research-datasets/mbpp` full/test 前 256 | 256 |
| MT-Bench | `HuggingFaceH4/mt_bench_prompts` | 80 |
| SWE-bench Pro | `ScaleAI/SWE-bench_Pro` test 前 128（只发 issue 正文） | 128 |
| SPEED-Bench × 5 | `nvidia/SPEED-Bench` config `qualitative`，按 `category` 取 coding / multilingual / rag / qa / writing | 各 80 |
| SPEED-Bench low-entropy | `nvidia/SPEED-Bench` config `throughput_16k`，`category == low_entropy` | 512 |

**AA-LCR 测不了，而且不能糊弄。** card 是在 71k–115k token 的多文档 prompt 上测的，
而数据集暴露的字段只有问题本身（平均 206 字符）。拿光秃秃的问题去跑，等于用同一个名字
测了另一个 benchmark；要挂上文档就需要 `max-model-len ≈ 131072`，而本栈是 16384。

`speed_throughput16k` 的输入平均约 10.4k token、最长约 22.5k，需要单独用
`--max-model-len 32768` 的服务端。其余 12 集在 16384 窗口下用 `max_tokens=12288`
（留出 4096 给最长的 prompt）。

### 7.4 最重要的一条：分母要用同机同引擎测出来

**不要拿你在 ATOM 上的数字去除以官方 card 在 vLLM 上的数字。** 把官方
`Inferact/Kimi-K3-DSpark` 下载下来，在**同一台机器、同一镜像、同一协议、同样的 prompt 数**
下测一遍，用它当分母。

本轮实测：同一个官方 draft 在本机相对它自己 card 的偏差是 MT-Bench −1.2%、AIME **+1.5%**、
GSM8K −13.1%、HumanEval −23.8%——**范围从 +1.5% 到 −23.8%，和数据集强相关，不存在可外推
的统一系数**。此前文档里"ATOM 读数系统性低约 10%"的说法来自单点、12 条 prompt、另一套
集群和镜像的测量，用它换算是错的。

顺带一个容易搞错的算术：官方 card 上那个 **3.85 是 14 个 benchmark 的总均值**，
不是任意四集的均值。四集（GSM8K/HumanEval/MT-Bench/AIME）的 card 均值是
`(5.64+5.34+3.14+2.72)/4 = 4.21`。

---

## 8. 耗时分析与优化

### 8.1 实测

2,698 步，墙钟 **12.82 小时**，纯 step 时间 10.34 小时（差值是 26 次 checkpoint、
270 次 eval、启动加载）。

| | mean | p50 | p90 | p99 | max |
|---|---|---|---|---|---|
| `step_s` | **13.80** | 13.77 | 15.09 | 16.57 | 57.18 |
| `teacher_s` | **8.99** | 9.02 | 10.08 | 11.12 | 27.89 |
| `train_s` | **4.81** | 4.73 | 5.16 | 7.15 | 29.29 |

**teacher 阶段占一步的 65.1%。** 它包含三件事，拆开是：

```
teacher_s = 8.99 s
├─ 纯网络取数            ~3.6 s   (40%)   ← 实测，5.27 GiB/s × 18.8 GiB
├─ 主机侧 pad / stack    ~4.4 s   (49%)   ← 推算
└─ 等 manifest + barrier ~1.0 s   (11%)
```

**等 manifest 接近于零**：一个 teacher 处理一个 batch 是 prefill（128×1832 ≈ 234k token，
约 9 s）+ put 9.5 s ≈ 18.5 s，四台并行即每 4.6 秒产出一个 batch，是消费速度的 3 倍，
队列始终满。

**真正的意外是那 4.4 秒不是通信，是主机内存拷贝。** 取数函数每取一条就 `F.pad` 到 batch
内最长行、再 `torch.stack`。实测 `seq/max_len` 平均 **7649**，而行长平均只有 1832 ——
**padding 占 76%**。每 rank 每步约 23 GB 的 memcpy（padded hidden_states 8.8 GB、stack
再复制 8.8 GB、last_hidden 1.8+1.8 GB、`token_embeds.clone()` 1.8 GB），8 个 rank 合计
约 184 GB/步。

而这些 padding **立刻被扔掉**：训练循环里 `actual_len = mb_mask.sum().max()`，
`micro_batch=1` 时就是这一行自己的长度，然后 `value[:, :actual_len].contiguous()`
又复制一次。等于"补到 7649 → stack → 切回 1832 → 再复制"，三次纯浪费。

### 8.2 优化，按收益/风险排序

| # | 优化 | 预计省 | 风险 |
|---|---|---|---|
| 1 | 取数不再 pad/stack，保持每行独立的 tensor，让训练循环直接用 | ~4 s/step | 低，改动局限在一个函数 |
| 2 | 后台线程预取下一步的 batch，与训练重叠 | 把剩下的约 4.6 s 藏进 4.8 s 的训练里 | 中，要保证 barrier 顺序 |
| 3 | 去掉 `token_embeds` 的 `.clone()`（先确认训练步是否真的用它） | ~0.3 s/step | 低 |
| 4 | 改回 RDMA | 网络那 3.6 s 或许能到 1 s | 高 |

1+2 做完，step 从 13.80 s 降到约 **5.5–6 s**，一个 epoch 从 10.3 小时降到约 **4.5 小时**，
2.3× 加速，而且都不动数值、不动传输协议。

**优化 4 的收益比直觉小得多**：网络只占 26%，就算 RDMA 把它清零，单靠它也只能把 step 从
13.8 降到 10.2。**"通信慢"在这里是个错觉——真正的开销是为通信做的数据整理。** 如果确实
要走 RDMA，先花 5 分钟验证"一个 client 能不能注册一个 64 GiB 的大段"（如果每张网卡的
限制是注册预算而不是 client 计数，这条就通了），否则就得把 manifest 改成逐条发布。

---

## 9. 本轮结果

### 训练侧

| | 起点（上一代权重） | 终点 | 变化 |
|---|---|---|---|
| `eval_old`（上一代那 128 行，可比 1.617） | 1.6203 | **2.0169** | **+24.5%** |
| `eval_old` step_0_acc | 0.7745 | **0.8210** | +6.0% |
| `eval_new`（新数据集尾部 128 行） | 1.0029 | **2.0008** | +99.5% |

曲线形态值得记住：换分布后 `eval_old` 先**下探**到 1.5834（step 110），然后回升并在
step ~190 越过起点，最后到 2.017。如果只看前 100 步会误判成"迁移失败"。斜率衰减规整：
每 100 步 +0.0274（step 200–600）→ +0.0243 → +0.0176 → +0.0108（step 1400–1840）。

### 服务端（ATOM，同机同引擎）

| benchmark | prompts | 本轮 | 上一代 | 变化 | 官方 draft（同机） | 本轮/官方 |
|---|---|---|---|---|---|---|
| GSM8K | 1319 | 3.9625 | 3.9795 | −0.4% | 4.9022 | 80.8% |
| HumanEval | 164 | 3.4520 | 3.1009 | +11.3% | 4.0701 | 84.8% |
| MT-Bench | 80 | 2.8123 | 2.1101 | **+33.3%** | 3.1022 | 90.7% |
| AIME 2026 | 30 | 2.4459 | 1.8682 | **+30.9%** | 2.7620 | 88.6% |
| **四集均值** | | **3.1682** | 2.7647 | **+14.6%** | **3.7091** | **85.4%** |

三条读得出来的结论：

- **MT-Bench 和 AIME 涨得最多**，而它们恰好是上一代最弱的两项。AIME 尤其反直觉——上一代
  的结论是"竞赛数学题的 token 分布离训练数据太远，长度和协议都救不了"，而九类池里
  74,540 行 math 的长思维链把它从 1.87 拉到 2.45。
- **GSM8K 一动不动**。上一代就已经是接受率 42%、k=7 全中占 13.9%，数学推导格式高度
  模板化，5 层 / 7 token 的结构在这类任务上看起来已经饱和。**要再涨得改结构，不是加数据。**
- **代码类是最明确的补法**。落到训练集的 code 只有 23,080 行（受限于那 17.9% 的去重
  留存率），而官方参考 draft 的训练数据里另有 `nvidia/OpenCodeInstruct`（150 万行单轮编程）。

---

## 10. 排错速查

| 症状 | 原因 | 处理 |
|---|---|---|
| `extract_hidden` 600 秒超时，零训练步 | 镜像版本漂了（atom dev323 + torch 2.10），或 Mooncake 走了 RDMA | 用钉死 digest 重 build；`MOONCAKE_PROTOCOL=tcp` |
| `Failed to register memory: Invalid argument [22]` | 同一张网卡上开第二个 RDMA client | 走 TCP，或把段池降到 ≤ 网卡数 |
| `Ray cluster has 0/5 active nodes` | Ray 的 10001 端口被同机其他租户占了 | `--ray-client-server-port 26380 --include-dashboard=false` |
| `node discovery timed out (1/5)` | 五个 rank 是顺序起的，协调文件差了几分钟 | 并行起，并把发现窗口放宽到 600 秒 |
| `'BF16Optimizer' object is not iterable` | torch 的 `get_optimizer_state_dict` 只接受 `torch.optim.Optimizer` | 非分片时直接用 `opt.state_dict()`；分片时必须报错而不是写半份 |
| `Incompatible value 'None' for field of type 'str'` | override 写成了 `mooncake.device_name=`（等号后为空）会被解析成 None | 值为空时不要下发这个 override |
| `mkdir /opt/spur/.docker: permission denied` | `spur exec` 的 HOME 不可写 | `HOME=/tmp/x DOCKER_CONFIG=/tmp/x/.docker` |
| 服务端接受率 ≈ 0% 但日志全正常 | 导出的张量名和 ATOM 读的对不上，输入通路被静默丢弃 | 重新导出（key 契约断言会拦下） |
| 训练指标漂亮但服务端很差 | RoPE 旋转约定 / YaRN `mscale²` 与服务端不一致 | 跑 RoPE 对齐自检，相对 L2 误差应在 1e-8 量级 |
| eval 指标在某一步突然变好 | 采样器回绕进了 eval 切片 | 确认训练可用长度排除了尾部的 eval 行 |
| GPU 显示 100% 但功耗只有约 320 W，无输出 | 集合通信自旋死锁 | 看 worker 日志里的 NCCL watchdog；重启续跑 |
| 一堆 `bgemm_internal_cublaslt error ... Will attempt to recover` | hipBLASLt 在某些形状上失败后回退到 cublas | 噪音，结果是对的，忽略 |
| `No available memory for the cache blocks` | KV 预算不够 | 确认没有别的进程占卡；降 `max_num_seqs` |
| 加载完只剩 175 GiB 可用显存 | `PYTORCH_CUDA_ALLOC_CONF` 漏进容器了 | `docker inspect` 确认它为空 |

---

## 11. 一句话流程

```
① 申请节点（--exclusive -G 8）
② 拉钉死 digest 的基础镜像，四项版本核对；build 训练镜像，一台 build 后 save/load 分发
③ 每台下 K3 权重到本地 NVMe，字节级校验
④ 建 prompt 池：图文过滤 -> 入库即去重 -> 13-gram 去污染 -> 按上限分配配额
⑤ 渲染 prompt（mpt=14304），多节点按 stride 切分，K3 生成（fp8 KV，384 并发，关 draft）
⑥ 逐行 token 回环校验转成训练集，丢弃率超阈值就停
⑦ CPU 预处理量出存活行数，定 num_training_steps，预热 token cache
⑧ 导入上一代 draft 权重（68 张量键契约 + lr=0 验 AL 逐位一致）
⑨ 五节点开训（Mooncake TCP，min_lr 不设 0，两把 eval 尺子，看守带 resume 自检）
⑩ 导出（key 契约断言）-> 起 ATOM 服务 -> 跑题目集 -> 同机再测一遍官方 draft 当分母
```
