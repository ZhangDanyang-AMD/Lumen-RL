# Kimi-K3 DSpark draft —— 从空集群到长上下文的完整手册

**目标：拿到一批干净的 MI355X 节点和这一份文档，就能从头训出一个 DSpark draft 并测出
可信的服务端接受长度。所有命令、版本号、参数取值和它们的理由都在这里，不引用任何其他
文档。**

读之前假设你懂分布式训练的一般概念（TP、FSDP、checkpoint、显存），但没见过这套代码、
没用过 ATOM、不了解投机解码。

这份手册是四轮训练（gen-3 到 gen-6）踩出来的合集。**凡是标了 `/!\` 的地方都是有来历的
——那类错误的共同特征是不报错、不崩溃，训练指标甚至很漂亮，跑完做 benchmark 才发现整轮
白费。** 这套流程里已经栽过至少五次：RoPE 旋转约定、导出张量名、eval 尺子换掉、
预处理缓存键、draft 权重导入路径。

**四代结果**（同机同镜像，服务端四集均值；官方参考 draft 在同一台机器上是 3.7091）：

| | gen-3 | gen-4 | gen-5 | **gen-6** |
|---|---|---|---|---|
| 训练样本（存活） | 345,476 | 345,476 | **5,011,091** | 1,264,631 |
| 步数 | 3,358 | 2,698 | **39,148** | **+9,878**（续训一个 epoch） |
| step 时间 | — | 13.80 s | **6.13 s** | 8.9 s（窗口大 4 倍） |
| 训练窗口 | 8192 | 8192 | 8192 | **32768** |
| 服务端四集均值 | 2.7647 | 3.1682 | 3.5087 | **3.8890** |
| 对官方（同机） | — | 85.4% | 94.6% | **104.9%** |

13 集完整对照见第 10 节。gen-5 均值对官方 **91.1%**，12 集落在 84.9–102.8%、三集反超，
**只有长上下文那一集是 41.7%** —— 这就是 gen-6 存在的原因。

**gen-6 结果**：13 集均值 **3.8218 / 3.6398 = 105.0%**，**13 集全部高于 gen-5，
12 集超过官方 draft**。长上下文那一集从 1.805 到 **4.521**（41.7% → 104.5%，接受率
11.5% → 50.3%），`swebench_pro` 从 84.9% 到 **100.6%**。做法是**换 32768 窗口 + 补长
文本长代码 + 从 gen-5 续训一个 epoch**，只花了 9,878 步（第 8 节）。

---

## 0. 这件事在做什么

训练一个 **DSpark draft 模型**（5 层，约 23.9 亿可训练参数），给 **Kimi-K3**
（93 层，hidden 7168，词表 163840，mxfp4 权重 1.56 TB）做投机解码加速。draft 训练好后由
ATOM 推理引擎加载，一次前向猜 7 个 token，target 批量校验，从而降低解码延迟。

衡量指标是**接受长度 `tok/fwd`**：

```
前向步数     = Σ 次数                       （接受长度直方图 {k: 次数}）
已接受 token = Σ k × 次数
tok/fwd     = 1 + 已接受 token / 前向步数
接受率      = 已接受 token / (前向步数 × num_spec)
```

那个 `1` 是 target 每步必出的 bonus token，所以即使接受率为 0，`tok/fwd` 也是 1.0；
猜 7 个 token 时上限是 8。这个量和官方 model card 上的 "acceptance length" 同尺度。

训练方式是**离线蒸馏（off-policy prefill distillation）**：先用 K3 把数据集里所有回答
重新生成一遍存下来，训练时 teacher 只需要对这些已知 token 做 prefill 并交出中间层的
hidden state，不再需要在训练环里解码。draft 学的是"给定 target 在每个位置的 hidden
state，预测接下来 7 个 token"。

整个流程分四段：

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

**GPU 数量不能少于 8。** `tensor_parallel_size: 8` 是硬约束 —— K3 权重切成 8 份才刚好
塞得下（每卡约 217.6 GiB），卡更少直接放不进去。

### 1.2 调度器：spur，不要 ssh 计算节点

计算节点被管理员加了 `AllowUsers ubuntu root`，普通用户 `ssh` 一律 `Permission denied`。
一律用 `spur exec <JobID> bash -lc "..."`。

```bash
hostname -f
echo "$SPUR_CONTROLLER_ADDR"     # v2 集群是 http://crs-m2m-cpu-spur-v2-001.crusoe.amd.com:6817
squeue -u "$USER"
```

申请节点（**必须带 `--exclusive`**，否则别的作业能落到同一台机器的空闲 CPU 上）：

```bash
sbatch --parsable -J k3_035 -A <your-account> -p default -N1 \
       --exclusive -G 8 -w crsuse2-m2m-v2-035 -t 168:00:00 \
       -o ~/logs/slurm_035.out --wrap "sleep 604800"
```

`--exclusive` 和 `-G 8` 都要写：前者占住 CPU 和内存，后者占住 8 张卡。只写 `--exclusive`
的作业在 `sinfo` 里显示 `mix` 而不是 `alloc`。**spur 不支持运行中改这个属性**
（`scontrol update JobId=N OverSubscribe=EXCLUSIVE` 只回 `unknown update key`），
申请时写错只能取消重申请。

后文用到的封装：

```bash
jobid() { cat ~/jobs/$1.jobid; }
onnode() { local n="$1"; shift; spur exec "$(jobid "$n")" bash -lc "$*"; }
onall()  { for n in $NODES; do ( onnode "$n" "$@" 2>&1 | sed "s/^/[$n] /" ) & done; wait; }
```

### 1.3 三个会浪费你时间的环境细节

- **`spur exec` 的 shell 里 `$HOME` 是 `/opt/spur`，不可写。** `docker build` 会因为
  建不了配置目录而失败（`mkdir /opt/spur/.docker: permission denied`）。build 前设
  `export HOME=/tmp/xxx DOCKER_CONFIG=/tmp/xxx/.docker`。
- **`spur exec` 的容器有自己私有的 `/dev/shm`。** 你在 spur shell 里看到的和 docker
  容器里 `-v /dev/shm:/dev/shm` 挂到的**不是同一个**（后者是宿主机的）。找 teacher
  worker 日志要 `docker exec <容器> ls /dev/shm/...`。
- **`spur exec` 里留下后台进程时，这条命令要约 100 秒才返回。** 并行起多个节点时必须
  把这些调用并行化，否则第一个和最后一个节点会差好几分钟。

上机先做两件事：

```bash
rocm-smi --showproductname                        # 8 张卡都在
rocm-smi --showmeminfo vram | grep -i used        # 每张卡 used 接近 0（空闲基线约 0.28 GB/卡）
sudo sh -c 'echo 0 > /proc/sys/kernel/numa_balancing'   # 关 NUMA auto-balancing
```

关 NUMA balancing 的原因：不关的话 aiter 每次导入都告警，AMD 官方文档明确说它在
MI300 系列上会引起错误。这是**主机级设置，容器里改不了**。

### 1.4 计算节点会以"看起来健康"的方式坏掉

一台节点变成 `State=IDLE` 但完全不接受作业：`scontrol show node` 上**没有 `Gres` 行**
（兄弟节点各有 8 条），于是 `-G 8` 永远 `PENDING Resources`；去掉 `-G` 之后变成
`JobLaunchFailure (dispatch confirmation failed: 0 of 1 nodes confirmed)`。这是两个
独立故障。**换节点不便宜**：K3 权重 1.56 TB 在每台机器的本地 NVMe 上，新节点没有。

---

## 2. 镜像：这一节跳过去后面全是白费

ATOM 对版本极其敏感。**`rocm/atom-dev:latest` 是滚动 tag，已经漂过至少两次，每次都以
不同的方式弄坏这套流程。** 必须按 digest 钉死。

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

`path` 必须落在 `/app/ATOM` 下。打出来是 `third_party/ATOM` 的话说明 `PYTHONPATH` 被
污染了 —— 镜像里 `third_party/ATOM` 是刻意不进 `PYTHONPATH` 的，一旦进去就会遮蔽
`/app/ATOM`，你跑的就不是校验过的那份 ATOM。

### 2.2 训练镜像

训练镜像 = 基础镜像 + LumenRL。**从仓库根目录构建**：

```bash
cd /path/to/Lumen-RL
HOME=/tmp/k3build DOCKER_CONFIG=/tmp/k3build/.docker \
docker build -f examples/Kimi_K3_SDDD_MI350_ATOM/docker/Dockerfile \
             -t kimi_k3_dspark_atom:pinned .
```

Dockerfile 第一步就断言基础镜像的 atom/torch/HIP/commit 四项，对不上直接 build 失败。
产出约 72.9 GB。

**四个反直觉的依赖决定，别顺手修好：**

| 决定 | 原因 |
|---|---|
| pip 装依赖一律带 **`--no-deps`** | 让 pip 自由解析 `accelerate`（它只要求 `torch>=2.0`）会"升级"到 torch 2.13 的 **CUDA 版本**，把 ROCm 的 torch 盖掉；反过来 pin torch，解析器会一头钻进 ray 的依赖图直到 resolution-too-deep。真正缺的传递依赖是手写死的。 |
| **`antlr4-python3-runtime==4.9.3`** | omegaconf 2.3 生成的 parser 在 4.13 上反序列化语法失败。 |
| 装 `opentelemetry`，**即使 `WANDB_MODE=disabled`** | wandb 0.28 起 `wandb.analytics` 硬依赖它，**在 import 阶段就炸**，跟 `WANDB_MODE` 无关。 |
| **镜像里没有 vLLM，这是设计** | lumenrl 里每一处 `import vllm` 都在 backend 分支后面，必须保持这样。vLLM 的 K3 decode 路径在 B=64 下每 15–20 个 batch 就会把 GPU 打挂（KDA 循环状态槽位在请求撞上 `max_tokens` 时的处理有问题），比一轮训练需要的连续 batch 数还短，**没有 fallback**。build 时断言 `find_spec("vllm") is None`。 |

还有一个 shim：ATOM 的 `configure_hidden_states()` 从 `torchspec` 包 import mooncake
管道，而基础镜像不提供这个包。Dockerfile 把 `docker/torchspec` 装成真包、把那两个
import 路径映射到 `lumenrl.transfer`，**ATOM 本身保持原样，不打补丁**。

/!\ 这条断言的来历：用 `latest` 构建出来的镜像坐在 atom `0.1.6rc1.dev323` + torch
`2.10.0+rocm7.2.4` 上 —— ATOM 比验证过的版本新 48 个提交，而 torch/ROCm 反而更旧。
在那个栈上 teacher 会在第二次 hidden state 抽取时死锁：8 个 TP rank 全部卡在一个 2 元素
的 NCCL BROADCAST 上，`last enqueued work: 7, last completed work: 6`，**GPU 显示 100%
但功耗只有 320 W（自旋，不是在算）**，600 秒后 `timed out waiting for response to
cmd=extract_hidden`，零训练步。

### 2.3 让所有机器跑同一份镜像

**不要在每台机器上各 build 一次**，各自 build 出来的 image ID 不同，pip 解析也可能不同。
在一台上 build，然后 save/load 分发：

```bash
onnode 035 'docker save kimi_k3_dspark_atom:pinned -o /shared_nfs/<you>/images/k3train.tar'
for n in 037 038 039 040; do
  onnode $n 'docker load -i /shared_nfs/<you>/images/k3train.tar' &
done; wait
onall 'docker images --format "{{.ID}}" kimi_k3_dspark_atom:pinned'   # 所有 ID 必须一样
```

### 2.4 一个不能漏的环境变量

**绝对不要设 `PYTORCH_CUDA_ALLOC_CONF`。** `expandable_segments:True` 会让 ROCm 的虚拟
保留不被 `empty_cache()` 归还，可用显存从 265 GiB 掉到 175 GiB，teacher 起不来。启动
脚本里要显式 unset，并且启动后从容器里读回确认它确实没被设置 —— 它会从调用方的 shell
继承进来。

```bash
docker inspect --format '{{range .Config.Env}}{{println .}}{{end}}' <容器> | grep ALLOC
# 必须为空
```

---

## 3. 六个必须一次做对的正确性决定

这六条的共同点：**做错了不会报错**。训练照跑、loss 照降、eval 指标照涨，只有服务端
数字会莫名其妙地差，而那时你已经花掉几十小时机时。

**这不是假设，是第一轮的实际结局**（MT-Bench 12 prompts，greedy，num_spec=7，同机同镜像）：

| draft | 接受率 | tok/fwd |
|---|---|---|
| 官方参考 draft | **26.13%** | 2.83 |
| 我们发布的配置 | **6.01%** | 1.42 |
| 我们 + 两个服务端补偿开关 | 14.82% | 2.04 |

训练侧 eval AL 当时是 **1.863**，看着很健康。两个 bug（3.1 和 3.2）单独都不报错。
那两个"补偿开关"值得说清楚：它们是**把 ATOM 也改错**、让服务端与训练器的错误对齐，
**不是修复**，而且这样的 checkpoint 和 TorchSpec / vLLM / SGLang 全都不兼容。

### 3.1 RoPE 必须是 interleaved 旋转（而且这个参数名的语义是反的）

/!\ **ATOM 的 `rope_interleave` 映射到 aiter 的 `is_neox_style`，语义和字面意思相反：**

```python
# atom/models/kimi_k3_dspark.py —— DeepSeek 风格的 MLA rope 默认就是 interleaved
is_neox_style=bool(getattr(config, "rope_interleave", False)),

# aiter/rotary_embedding.py
if is_neox_style:
    x1, x2 = torch.chunk(x, 2, dim=-1)      # half-split（NeoX/Llama）
else:
    x1 = x[..., ::2]; x2 = x[..., 1::2]     # interleaved pairs（GPT-J/DeepSeek）
```

所以 `rope_interleave: true` 得到的是 **half-split**，不设它才是 interleaved。

| 实现 | 实际旋转 | 怎么选中的 |
|---|---|---|
| TorchSpec `_apply_rope_interleaved` | interleaved | 硬编码 |
| 官方发布的 ATOM config（没有 `rope_interleave`） | interleaved | 默认 |
| ATOM `rope_interleave: true` | **half-split** | 显式（名字骗人） |
| LumenRL 修复前的 `_apply_rope_by_position` | **half-split** | 用了 `_rotate_half` |

**正确写法**（cos/sin cache builder 不动，在 apply 时做转换）：

```python
half = cos_pos.shape[-1] // 2
cos_pos = cos_pos[..., :half].repeat_interleave(2, dim=-1)
sin_pos = sin_pos[..., :half].repeat_interleave(2, dim=-1)
return (x * cos_pos) + (_rotate_half_interleaved(x) * sin_pos)
```

**导出时不得 emit `rope_interleave`**（`export_dspark_hf.py` 会断言这一条）。

### 3.2 softmax scale 必须带 YaRN 的 mscale²

YaRN 不只改位置插值，还改 attention 的 softmax 温度。漏掉 `mscale²` 这个因子，
logits 会**冷 1.8133 倍**。

```python
def yarn_get_mscale(scale=1, mscale=1):
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0

# 训练侧要这么算
softmax_scale = 1.0 / math.sqrt(self.qk_head_dim)
if rope_cfg.get("rope_type", "yarn") == "yarn":
    ms = _yarn_get_mscale(float(rope_cfg["factor"]),
                          float(rope_cfg["mscale_all_dim"]))
    softmax_scale *= ms * ms
```

发布配置下的具体数值（`factor=32`，`mscale_all_dim=1.0`，`qk_head_dim=192`）：

```
yarn_get_mscale(32, 1.0) = 0.1 × ln(32) + 1 = 1.34657
mscale²                                     = 1.8133
不带 mscale：1/√192                          = 0.07217
正确值：    1.8133/√192                      = 0.13087    ← 两边必须都是这个
```

/!\ **只对比 rotary 会漏掉这条。** cos/sin 的幅度两边都用
`yarn_get_mscale(f, mscale) / yarn_get_mscale(f, mscale_all_dim)`，`mscale ==
mscale_all_dim` 时恒为 1.0 —— 所以 rotary 可以逐 byte 相同而 attention scale 还是错的。
配置里 `rope_mscale: 1.0` / `rope_mscale_all_dim: 1.0` 和这条是一体的，不要单独改一边。

### 3.3 `aux_hidden_state_layer_ids: [2, 23, 47, 71, 89]`

**0-based，取的是第 i 层的输出**（参考实现索引 `hidden_states[layer_id + 1]`，
即 layer `layer_id` 的 OUTPUT，这正是 ATOM 的 aux-tap 约定）。93 层里取这五层，
draft 的 `fc` 输入宽度是 `5 × 7168 = 35840`。

改这个列表等于换监督信号，导出时 config.json 里的 `target_layer_ids` 必须和训练 YAML
一致，服务端启动日志里会打
`DSparkProposer aux capture on target layers: (2, 23, 47, 71, 89)`，对不上就是错的。

### 3.4 `separate_last_hidden` 在 ATOM 上必须是 `false`（和 vLLM 相反）

| | vLLM 变体 | **ATOM** |
|---|---|---|
| `hidden_states` | 前 4 个 aux 层，pre-norm | **全部 5 层**（35840 宽） |
| `last_hidden_states` | 第 5 个 aux 层，pre-norm | **模型最终输出，已过 final norm** |
| 训练器要做的事 | 自己拼回去再做 RMSNorm | 直接用 |

设成 `true` 会让 `fc` 收到 `6 × 7168 = 43008` 而不是 `5 × 7168`，直接形状错误，
而且 norm 被做两次。

### 3.5 off-policy prefill，不是 generate

teacher 只对数据集里**已知的** token 做 prefill 并交出 hidden state，不在训练环里
自回归解码。这样 teacher 常驻、吞吐可预测，而且 draft 学的分布严格等于 target 在这批
真实 token 上的分布。

配套要求：**数据集里的回答必须是 K3 自己生成的**（见 4.3）。用别的模型写的回答，
draft 会去猜一个 target 永远不会产出的分布。

### 3.6 训练时 teacher 的 KV 必须和服务时一样是 fp8

早期版本训练用 bf16，理由是 fp8 量化会给 label 引入噪音。**这个理由是错的**：服务栈
用的就是 fp8，训练时的 teacher 必须和服务时的 teacher 是同一个函数，否则 draft 学的是
一个上线后不存在的分布。**正确性优先于噪音。**

（注意生成阶段用 fp8 的理由不同，那是为了省显存，见 4.3。）

### 3.7 上机自检：开训前必须全绿

这几个脚本是唯一能在花掉机时之前拦住 3.1/3.2 那类错误的东西。**总共十几分钟。**

公共 docker 参数：

```bash
DOCKER_ARGS="--rm --device /dev/kfd --device /dev/dri --group-add video --group-add render \
  --ipc=host --shm-size=64g --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
  -v /path/to/Lumen-RL:/workspace -w /workspace"
```

**① RoPE + mscale² 对齐**（要 GPU，几十秒）—— 直接验 3.1 和 3.2：

```bash
docker run $DOCKER_ARGS kimi_k3_dspark_atom:pinned \
  python3 examples/Kimi_K3_SDDD_MI350_ATOM/selfcheck/verify_rope_alignment.py
```

通过标准：

```
softmax scale
  ATOM    : 0.1308608000 (1.8133x plain)
  LumenRL : 0.1308608000 (1.8133x plain)
rotation vs ATOM published (interleaved)      rel L2 = 3.127e-08   MATCH
rotation vs ATOM rope_interleave=true (half-split) rel L2 = 9.874e-01   differs

PASS: trainer attention matches ATOM under the published config
```

- 两边 softmax scale 必须**完全相同**（`abs(got − want) <= 1e-9`）
- interleaved 相对 L2 在 **1e-8** 量级
- /!\ **half-split 那一行也 MATCH 的话是 FAIL** —— 说明训练器还在用半分旋转

/!\ 这个脚本在 CPU 默认设备上会报 `Expected all tensors to be on the same device`：
`DeepseekScalingRotaryEmbedding._compute_inv_freq` 里一半硬编码了 `device="cuda"`。
用 `with torch.device("cuda"):` 包住 `get_rope` 调用。

**② flex_attention 与稠密 mask 路径等价**（要 GPU，几十秒）：

```bash
docker run $DOCKER_ARGS kimi_k3_dspark_atom:pinned \
  python3 examples/Kimi_K3_SDDD_MI350_ATOM/selfcheck/verify_flex_attention.py
```

通过标准：输出 finite、被 mask 掉的行两边都严格为 **0**、`rel L2` 在 3e-03 量级、
flex 占用显存更少、`PASS: flex_attention matches the dense mask path`。

/!\ **不要给编译加 `dynamic=False`。** 每步的 `KV_LEN` 都在变，固定 shape 会触发反复
重编译、dynamo cache 打满，退回 eager flex 后显存 33.7 GiB，**比稠密路径还差** ——
而且结果正确、不报错、只是更慢。日志里 `8165:5ms 8128:1166ms 8037:1483ms 7954:6ms`
这种形态是正常的（两次重编译后稳定在 5 ms）。

**③ ATOM API preflight**（不要 GPU，几秒）：

```bash
python3 examples/Kimi_K3_SDDD_MI350_ATOM/selfcheck/preflight.py
```

它断言的东西正好是这套集成最容易漂掉的接口：`configure_hidden_states(self,
aux_layer_ids, mooncake_config)` 的签名、`separate_last_hidden: false`、
`max_num_batched_tokens >= max_model_len`、chunked prefill 和 prefix caching 都关、
worker 的 `sys.path` 排除了 `third_party/ATOM`。通过标准是 `PREFLIGHT PASSED`、exit 0。

**④ 端到端 smoke test**（约 20 分钟，主要是 teacher 加载）：

/!\ **smoke 和正式的这四个值必须完全一样**，否则 smoke 就失去意义：
`max_num_batched_tokens` / `max_num_seqs` / `max_model_len` / `kv_cache_dtype`。

/!\ **smoke 的步数要足够跨过一次 teacher 重启**。只跑 5 步（= 1 round）测不到重启路径，
而"首次启动装得下、重启时装不下"是这套系统最常见的故障形态（见 11 节）。

---

## 4. 数据准备

### 4.1 K3 权重

| 项 | 值 |
|---|---|
| repo | `moonshotai/Kimi-K3` |
| revision | `a590ce090cb049c93a33dfe8c208ec652aa20503` |
| 大小 | 1.561 TB，96 个 safetensors |

**落每台机器的本地 NVMe（`/mnt/m2m_nobackup`），不要放 NFS。** NFS 冷读约 200 MiB/s，
完全冷启动约 2.2 小时；本地 NVMe 约 4 分钟。这个差值要乘上被抢卡重启的次数。

下完一定做**字节级校验**，别只看文件数 —— 1.5 TB 的下载中断很常见，缺一个 shard 的
表现是加载到 90% 才报错，白等几十分钟。

```bash
onnode 035 'ls /mnt/m2m_nobackup/<you>/models/Kimi-K3/*.safetensors | wc -l'   # 必须 96
```

### 4.2 prompt 池：去重、去图文、去评测集污染

源数据：

| repo | 用到的 split |
|---|---|
| `nvidia/Nemotron-Post-Training-Dataset-v2`（gated，要 HF token） | `chat` `code` `math` `stem` `multilingual_ja` `_de` `_it` `_es` `_fr` |
| `CohereLabs/aya_dataset` | `train` |
| `nvidia/OpenCodeInstruct` | 全量（gen-6 用） |
| `lightseekorg/kimi-mtp-dataset` | 全量 |

三件事必须在**采样之前**做完，顺序不能反：

**一、图文过滤。** K3 是多模态模型，但这条生成路径走纯文本 token，从来拿不到图，
所以图文样本的回答一定是幻觉 —— 模型会自信地描述一张它没看见的图。只滤真正有问题的
四类：

| 类别 | 判据 |
|---|---|
| content 不是字符串 | `type(content) is not str` |
| 图文 JSON 外壳 | 匹配 `^\s*\[\s*\{\s*"type"\s*:` |
| 模型图片 token | 含 `<\|image\|>` / `<\|media_begin\|>` / `<\|vision_start\|>` |
| `<image>` 占位符 | 正文里出现 `<image>` |

/!\ **不要把 `<img` / `image_url` / `data:image/` 放进黑名单。** 早期版本这么写，丢掉了
2.55% 的数据，逐条抽查发现绝大部分是误伤：HTML 代码题的 `<img` 标签、REST API 的
`image_url` 字段名、JSON 请求体示例里的 `data:image/png;base64,...`。判据要基于"这一行
是否真的需要一张图才能回答"，而不是"这一行里是否出现过跟图片有关的字符串"。

实测各源的图文占比：kimi-mtp **27.3%**（122,812 行 `vlm_json_shell` + 346 个 `<image>`，
全是 COCO 图片 URL；21.8% 的 thinking 里明说自己看不见，28.4% 的回答仍然断言"我看到了
这张图"）；nine-category 建池时已滤过，二次过滤只抓到 12 行漏网的。

**二、去重，而且必须是"入库即去重"。** 按 prompt 的 NFKC 正规化 + 转小写 + 折叠连续
空白之后取哈希（`blake2b`，digest_size=16），维护一个**跨所有 split 的全局 `seen`
集合**，在采样选中的当时就判重。

先采样再去重会踩坑：`code` split 期望 50,000 只拿到 27,236 行，因为重复 prompt 被反复
选中、去重后大量落空。跨 split 去重也是必要的 —— 五个 multilingual split 之间有大量
重叠。**处理顺序决定谁赢**：先处理的占住哈希。所以先英文四类，再 aya（人工撰写），
最后才是 Nemotron 那五个合成 multilingual。

**三、评测集去污染。** 对要测的 benchmark 抓 eval prompt，建 13-gram 索引，命中即剔除。
n-gram 取 13 是折中：更短会误伤（常见代码片段撞车），更长会漏（改写过的题）。数量看着
小（约 1.4 万条），但留着的后果是评测数字虚高且无法察觉。

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

`code` 只留 17.9% 是这张表最重要的一行：175,000 行里只有 31,261 条唯一 prompt（同一题
平均 5.6 个回答）。**这是物理上限，不是选择**，它直接决定了代码类数据的天花板 ——
gen-4 和 gen-5 的 code 绝对量几乎一样（23,083 vs 23,080）就是因为它。

/!\ **不设配额的代价**：gen-5 跑全量不设配额，而五个 multilingual split 本身就有 445 万
条唯一 prompt，结果 **74.8% 的梯度步花在多语种上**。绝对量上每一类都没变少，但评测尺子
被带偏了（见 5.8）。

### 4.3 用 K3 重新生成所有回答

**渲染 prompt 必须复用训练侧同一个 parser**（`lumenrl/data/kimi_k3_parser.py` 的
`KimiK3Parser`），不要自己拼 chat template —— 训练时会用同一个 parser 重新渲染
prompt+response，两边必须逐 byte 一致。

`thinking_effort` 什么都不用传：K3 的 `tokenization_kimi.py` 里有
`kwargs.setdefault("thinking_effort", "max")`，不传就是 max，实测传与不传渲染结果逐
byte 相同。

**生成引擎配置**（每个数都是实测的）：

```
max_model_len          16384          # 长上下文轮次用 32768
max_num_seqs           384            # 不是保守，是实测更快，见下
max_num_batched_tokens 32768          # 必须 >= max_model_len
gpu_memory_utilization 0.93
kv_cache_dtype         fp8            # ATOM 只接受 "bf16" | "fp8"
enforce_eager          False          # CUDA graph 只覆盖 decode，值 2.4x
enable_prefix_caching  False
enable_chunked_prefill False
tensor_parallel_size   8
temperature            0.0
speculative_config     不传           # 关掉 draft，实测快 2x，见下
max_tokens             逐行算 = max_model_len - len(prompt) - 32
max_prompt_tokens      14304 = 16384 - 2048 - 32   （32768 窗口下是 32224，见 8.2）
```

四个必须解释的取值：

**`kv_cache_dtype: fp8` 是硬要求。** 16384 窗口下 MLA 的 paged KV 是主要开销：K3 的 MLA
entry 是 `kv_lora_rank + qk_rope_head_dim = 512 + 64 = 576` 元素/token/层 × 24 个
full-attn 层，bf16 是 27 KB/token，fp8 是 13.8 KB/token —— **fp8 直接把可用 window 翻倍**。

**`max_num_seqs: 384` 而不是 512。** K3 有 69 个 KDA（线性注意力）层，它的循环状态池按
`max_num_seqs` 线性吃显存（实测 56.17 MB/槽），和 paged KV 池抢同一份
`available_for_kv ≈ 44.9 GB`：512 槽位时状态池 26.78 GB、只剩 86,055 个 block，长 prompt
下需求溢出 24%，有效并发反而只有 279；384 槽位时状态池 20.09 GB、有 120,425 个 block，
实际并发就是 384。引擎启动时会打印 "Concurrent capacity vs context length" 表，看
`bound by slots` 还是 `bound by blocks` 就行，别自己估。

**关掉 draft。** 打开官方 DSpark draft 会让 KDA 状态池 ×8（`entries_per_req = 1 +
num_spec`，给每个投机 token 留一个回滚槽），并发从 512 压到 64，吞吐从约 3000 tok/s 掉到
约 1500。投机把前向次数省了 3.64 倍，但 **MoE 解码吞吐对 batch size 非常敏感，并发掉
8 倍亏得更多**。32768 窗口下这个结论更硬，见 8.3。

**`max_prompt_tokens` 不要抄训练侧的值。** 训练期它是个筛子随时能改；生成期它决定哪些
行会被生成，丢掉的行要重跑 GPU 才能拿回来。

**生成脚本的两个设计要点，都不是可选项：**

1. **滚动提交，不是 sweep-and-wait。** 一个 sweep 要等最慢的序列跑完，长 cap 下尾部会
   退化成几条序列独占 8 张卡。要始终维持 `max_num_seqs` 条在飞，完成一条补一条。
2. **自己调 `io_processor.preprocess()` 拿 Sequence 对象建行映射**，而不是用
   `engine.generate()`。后者把完成的序列按内部 id 排序再和输入顺序 zip，假设 id 的排序
   等于提交顺序 —— 配错一行就是静默的训练数据污染。要在每条完成时校验
   `num_tokens_input` 和自己发出的 prompt 长度一致，不一致立刻停。

**多节点切分按 stride，不按连续区间**（第 i 行给 shard `i % N`）。源数据是按 split
分块拼的，连续区间切会让每个 shard 的组成完全不同、ETA 全失真。

**要有 watchdog 和 supervisor。** ATOM/RCCL 在长时间多卡运行下会偶发死锁：某个 broadcast
卡住，torch watchdog 480 秒后报 "got stuck"，8 个 rank abort，而**父进程阻塞在
`engine.step()` 里永远不返回，`docker ps` 看着一切正常**。曾经因此白转 8 小时。两层修复：
脚本内独立线程的 watchdog（必须是独立线程 + `os._exit`，主线程已经卡死在 C 扩展里，
任何依赖它的退出路径都不会生效），外层 supervisor 负责拉起来跑到完为止。第 9 节有一套
更完整的编排。

### 4.4 转成训练数据集：重点是校验，不是转换

K3 在 `thinking=true` 下的 completion 形状是：

```
<think 正文><|close|>think<|sep|><|open|>response<|sep|><response 正文><|close|>response<|sep|><|close|>message<|sep|><|end_of_msg|>
```

按这三个标记切开，写回 `reasoning_content` + `content`。**这一步的重点是校验**：训练吃
的是 LumenRL 重新渲染出来的 input_ids，不是我们存的 completion_ids，中间隔着
「completion 文本 → 拆 reasoning/content → 塞回 conversation → 重渲染 → 重 tokenize」
这一整条路，任何一步不保真，draft 学到的就是 K3 从没产出过的 token，而且没人会报错。

**每一行都验，验不过就丢。** 抽样验不够 —— 踩到过的两个 bug（`strip()` 吃空格、工具
调用被静默吞掉）在抽样里都只表现为个位数。切分**不做 strip**，K3 会在
`... Done. <|close|>think<|sep|>` 这样的位置留空格，strip 掉就少一个 token。只接受
「think → response → 结束」这一种规整形状，带工具调用的整行丢掉。

**实测丢弃率 8.8%**（399,842 → 364,767），构成：

| 原因 | 占比 | 含义 |
|---|---|---|
| `no_think_close` | 6.14% | 顶到生成上限，think 段没闭合 |
| `response_mismatch` | 2.21% | 重新分词后 token 序列与生成时不一致 |
| `no_response_close` | 0.32% | 顶到上限，response 段没闭合 |
| 其余两类 | 0.10% | 首个 channel 不是 response / 出现额外 channel |

`response_mismatch` 不是 bug，是编码/生成不对称：BPE 编码看整串按合并优先级贪心
（唯一解），而模型逐 token 往外吐、不受规范切分约束。例如文本 `FORMCHECK`，BPE 规范
编码是 `['FORM','CHECK']`，K3 实际吐的是 `['FOR','MC','HECK']` —— 解码回文本 100% 相同，
但重新编码就回不到原路径。所以"存成文本"这一步本身就是有损的，不是校验太严。

**丢弃率超过阈值就非零退出，不要直接调大阈值**，先看原因分布。

/!\ `no_think_close` + `no_response_close` 那 6.46% 是**顶到了生成上限**的行 ——
它们是长上下文数据的富矿，见第 8 节。

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
驻留，单节点方案必须让两者轮流上下卡（一轮 22 分钟里三分之二花在传输和重载）。分离
之后 teacher 常驻、draft 常驻，各自不用让位。

### 5.2 配置：动了会出事的值

**序列与批次**

| 值 | 设定 | 为什么 |
|---|---|---|
| `policy.max_total_sequence_length` | 8192（gen-6：32768） | 与生成窗口和 teacher 的 `max_model_len` 一致 |
| `train_global_batch_size` | 128 | TorchSpec 参考配方，和 lr/max_grad_norm 是一组，要改一起改 |
| `train_micro_batch_size` | 1 | |
| `spec_length` | 7 | 和服务时的 `--num-speculative-tokens 7` 对齐 |

**teacher（ATOM）**

| 值 | 设定 | 为什么 |
|---|---|---|
| `tensor_parallel_size` | 8 | 硬约束 |
| `max_model_len` | 8192（gen-6：32768） | 和训练窗口一致 |
| `max_num_seqs` | 32 | KDA 循环状态随它线性增长，256 会把显存吃穿 |
| `max_num_batched_tokens` | 8192（gen-6：32768） | **必须 >= `max_model_len`**。它同时决定 prefill 激活峰值：32768 时约 14–18 GiB，8192 时约 7–9 GiB |
| `enable_prefix_caching` | false | 前缀命中不计入 scheduled_tokens，hidden state 会缺 |
| `enable_chunked_prefill` | false | ATOM 每个调度步用同一个 key 重写，不看 `is_final_chunk`，切开就只剩最后一块 |
| `gpu_memory_utilization` | 0.90 | |
| `kv_cache_dtype` / `index_cache_dtype` | **fp8** | 见 3.6 |
| `enforce_eager` | true | |
| `generate_mode` | prefill | 离线蒸馏，不做自回归生成 |

**draft 与损失**

| 值 | 设定 |
|---|---|
| `num_layers` / `num_heads` / `head_dim` / `ffn_dim` | 5 / 64 / 128 / 14336 |
| `q_lora_rank` / `kv_lora_rank` | 1536 / 512 |
| `qk_nope_head_dim` / `qk_rope_head_dim` / `v_head_dim` | 128 / 64 / 128 |
| `rope_theta` | 50000.0 |
| `rope_scaling` | yarn，factor 32.0，original_max_pos **32768**，beta_fast 32.0，beta_slow 1.0，mscale 1.0 / mscale_all_dim 1.0 |
| `aux_hidden_state_layer_ids` | `[2, 23, 47, 71, 89]`（见 3.3） |
| `capture_mode` | postnorm |
| `separate_last_hidden` | **false**（见 3.4） |
| `anchor_num` | 512（峰值显存的主要来源，实测 57.8 GiB） |
| `ce_loss_alpha` / `l1_loss_alpha` / `confidence_loss_alpha` / `loss_decay_gamma` | 0.1 / 0.9 / 1.0 / 4.0 |
| `markov_rank` / `block_size` / `mask_token_id` | 256 / 7 / 163837 |

/!\ **`rope_scaling.original_max_pos` 是 32768，所以 32768 窗口本来就在原生（未插值）
范围内**，把训练窗口从 8192 改到 32768 **不需要动模型结构** —— 只是训练时从没访问过
8192 以上的位置而已。RoPE 本身无参数。

**优化器**

| 值 | 设定 |
|---|---|
| `learning_rate` | 5.0e-5 峰值 |
| `warmup_ratio` | 0.04 |
| `max_grad_norm` | 1.0 |
| `lr_decay_style` | cosine（gen-6 改 WSD，见 8.5） |
| `min_lr` | **3.93e-5，不是 0** |

/!\ **`min_lr` 不设 0 是这套配方里最容易被"顺手改回去"的一条。** gen-3 用 min_lr 0 跑
单个 epoch，余弦把 lr 带到 **7.6e-10**，eval AL 从 step 3300 起在 1.613–1.617 平台化
—— 那是 schedule 造成的假平台，不是数据用尽，而且让"接着再训一个 epoch"变得没有意义。
取"三个 epoch 的余弦跑到第一个 epoch 末尾"的值：

```
decay_ratio = (1 − 0.04×3) / (3 − 0.04×3) = 0.3056
coeff       = 0.5 × (1 + cos(0.3056π))    = 0.7868
min_lr      = 0.7868 × 5e-5               = 3.93e-5
```

### 5.3 num_training_steps 必须实测，不能估

在 CPU 上跑一次和 trainer 完全相同的预处理：

```bash
python3 selfcheck/preprocess_dataset.py \
    --dataset <你的 train.jsonl> --tokenizer <K3 路径> \
    --max-length 8192 --max-prompt-tokens 0 \
    --thinking true --last-turn-loss-only true --min-loss-tokens 14 \
    --eval-samples 128 --batch-size 128 \
    --cache-dir <共享路径>/tokens --workers 96
```

/!\ **每个参数都进 cache key**（数据集路径 + 文件大小 + mtime、tokenizer 路径、
max_length、chat_template、last_turn_loss_only、min_loss_tokens、split、
PARSER_VERSION、drop_overlong、max_prompt_tokens、thinking），差一个就会生出第二份
缓存、正式训练再把预处理的钱付一遍。`num_workers` 不在 key 里。

**换窗口就必须重建缓存** —— `drop_overlong` 的阈值变了。

实测：gen-5 的 5,358,149 行 → **5,011,091 行存活**（8192 窗口丢 347,054 行超长 + 4 行
含词表外 token），`num_training_steps = floor((5011091 − 128) / 128) = 39148`。
存活行长度 **mean 2009 / p50 1399 / p90 4560 / max 8191**。

**`max_prompt_tokens: 0` + `drop_overlong: true` 时超长行是整行丢弃，不是截断**
（`check_total = drop_overlong and max_prompt_tokens <= 0`，mpt=0 时恒为 True）。
想改成截断就把 mpt 设成正数。

### 5.4 500 万行会在启动时 OOM，除非 input_ids 存成 int32

**每个 draft rank 都把整份预处理数据集常驻内存**，而 `input_ids` 原本存成 `list[int]`：
Python int 每个 28 字节的对象加 list 里 8 字节的槽位 = **36 字节/token**。

| 存储 | 单 rank | × 8 rank | 节点内存 |
|---|---|---|---|
| `list[int]` | 346 GiB | **2766 GiB** | 2867 GiB |
| int32 ndarray | 74 GiB | **594 GiB** | 2867 GiB |

旧路径是节点内存的 96.5%，加上 trainer、Mooncake 段和 CUDA context 必然爆。gen-4 的
34.5 万行只要 23 GiB/rank，所以这个坑在 15 倍数据量之前不会出现。

**改法**（`lumenrl/data/dataset.py`）：`_tokenize_single` 返回
`input_ids.to(torch.int32).numpy()`。词表 163840，实测 token id 上界 163589，int32 无损。
消费端（trainer 4 处、`build_eval_slice.py` 1 处）统一改成
`torch.as_tensor(..., dtype=torch.long)`。

/!\ **不要保留 `isinstance(ids, list)` 的分支** —— ndarray 会绕过它、以 int32 直接送到
`F.embedding`。

顺带把 `_drop_out_of_vocab_samples` 向量化（原本逐 token 的 Python 比较，500 万行是
10¹⁰ 次）。实测改完 draft 节点占 990 / 2751 GB。

### 5.5 两项性能优化：step 从 13.80 s 到 6.13 s

**gen-4 的耗时拆解**（这是改动的出发点）：

```
step_s = 13.80 s
├─ teacher_s = 8.99 s
│   ├─ 纯网络取数            ~3.6 s   (40%)
│   ├─ 主机侧 pad / stack    ~4.4 s   (49%)   ← 优化一
│   └─ 等 manifest + barrier ~1.0 s   (11%)
└─ train_s  = 4.81 s                          ← 优化二把网络藏进这里
```

**关键认识：`teacher_s` 里最大的一块不是通信，是为通信做的数据整理。** 实测
`seq/max_len` 平均 7649，而行长平均只有 1832 —— **padding 占 76%**，每 rank 每步约
23 GB 的 memcpy，8 个 rank 合计约 184 GB/步。所以"通信慢"是个错觉，直接改 RDMA 只能动
那 26%。

**优化一：取数不再 pad 到全批宽度。** 原路径每取一行就 `F.pad` 到整个 global batch 的
最长行、`torch.stack` 成矩形张量；训练循环里立刻 `actual_len = mb_mask.sum().max()`、
`value[:, :actual_len].contiguous()` 把 padding 切回去 —— 等于"补到 7649 → stack →
切回 1832 → 再复制"，三次纯浪费。改成两层：

- `_load_disagg_rows(manifest, rows)` —— 按行的自然长度取回，不做任何 pad
- `_collate_disagg_rows(fetched, width=None)` —— 只在需要矩形张量的地方对齐

训练循环改成**在 micro-batch 内部才 collate**。`train_micro_batch_size: 1` 下每个
micro-batch 就是一行，于是走 `unsqueeze` 视图，**一次拷贝都没有**：

```python
if len(fetched) == 1 and int(fetched[0]["input_ids"].shape[0]) == width:
    return {k: fetched[0][k].unsqueeze(0) for k in (...)}
```

顺带去掉 `token_embeds` 这个键：两个引擎都把第一个 aux 层当"embed proxy"发布在这个名字
下，取数时 `.clone()` 要 1.75 GiB/rank/step —— 但训练步从不读它，而是用
`F.embedding(input_ids, embed_w)` 从 teacher 冻结的 embedding 表自己重算。

/!\ 掩码宽度要 clamp 并且要出声：掩码仍按 manifest 全宽送来，collate 出来的张量只有本批
宽度，取 `min(mask_len, width)`，并在 `mask_len > width` 时打 warning —— 掩码声称的真实
token 比 teacher 发布的多，意味着 hidden state 被截断了。

**优化二：后台线程预取。** 两条约束决定了实现只能是这个样子：

1. **所有集合通信必须留在主线程。** manifest 广播、段 barrier、指标 all-reduce 都是集合
   操作。两个线程对同一个 process group 发集合操作，故障形态是 watchdog 超时且没有栈。
2. **接收只落 CPU。** 从后台线程碰设备会在 ROCm/MI350 上触发 `VM_L2_PROTECTION_FAULT`。
   所有 `.to(device)` 留在主线程。

用 `threading.Thread(daemon=True)` 而不是 `ThreadPoolExecutor` —— 后者的线程非守护，
主线程异常退出时会被 atexit 卡在一个 Mooncake get 里，直到 `get_retry_max_wait_seconds`
（300 秒）才放手。

每步的顺序（**barrier 必须在第 1 步之后、第 4 步之前**）：

```
[循环外] receive manifest(0) -> 交给后台取数(0)
每步 N:
  1. fetcher.get()            等本步数据（稳态约 0.2 s）
  2. barrier                  所有 rank 都清掉了第 N 批
  3. submit(N + depth)        teacher 可以写下一批了
  4. receive manifest(N+1) -> 交给后台取数(N+1)
  5. train(N)                 后台取数在这段时间里跑完
```

**实测**（gen-5 全程 39,148 步逐步解析出来的，不是抽样）：

| | gen-4 | gen-5 mean | gen-5 p50 |
|---|---|---|---|
| `step_s` | 13.80 s | **6.13 s** | 5.96 s |
| `teacher_s` | 8.99 s | 1.79 s | 1.55 s |
| `train_s` | 4.81 s | 4.34 s | 4.31 s |
| `fetch_wait_s` | — | 0.39 s | 0.27 s |

全程稳定，没有漂移（step 1k–5k 窗口 mean 6.04，36k+ 窗口 6.19）。p90 7.44 / p99 8.87，
长尾是 checkpoint 和 eval。

**系统吞吐 = 128 × 2009 / 6.13 ≈ 41,950 token/s**，这是外推 gen-6 耗时要用的数。

**瓶颈在 teacher**：单台热态 `extract_hidden` 23.15 s，四台并行 = **5.79 s/step 的下限**，
实测 6.13 比这条线高 6% —— 余下的是 draft 侧没被完全藏住的部分。

/!\ 早期记录里有个 **5.72 s** 的"稳态"读数，比全程实测低 7%，按它外推会系统性低估
耗时。**报 ETA 用 mean 6.13，不要用某个顺手的窗口。**

/!\ **smoke test 的 step 读数会骗人。** gen-5 的 smoke 读到 3.8 s/step，全程实测 6.13 s
—— smoke 只有 5 步而启动时预提交了 `stream_prefetch_batches` 个批，**teacher 的产出
速率从没被测到**。要去 rank 日志里翻 `extract_hidden` 的热态耗时（冷启动那几条 61–63 s
含 aiter JIT 和 FlyDSL kernel 回退，不算数）。

### 5.6 Mooncake 走 TCP，不要 RDMA

**症状**：teacher actor 的 `extract_hidden` 600 秒超时，零训练步，`docker ps` 一切正常。

**根因**：默认配置要求一个 teacher 进程注册 128 个各自独立的 2 GiB RDMA 段（Ionic 不
接受超过 2 GiB 的单个 MR）。但实测**这套集群的 Ionic HCA 上，一个进程每张网卡只能建
一个 RDMA client**：

```
--devices ionic_0 --stores 3
  store 1/3   1.66s   ok
  store 2     Failed to register memory 0x...: Invalid argument [22]
              -> Mooncake setup failed (error=-600)
```

七张网卡轮着开时前 7 个成功、第 8 个失败（`index % 7` 绕回 `ionic_0`）。所以段容量上限
是 **7 × 2 GiB = 14 GiB**，而一个 batch 的 128 条序列要发约 20 GiB。更要命的是 manifest
是等 128 条**全部**写完才发布的 —— 写的人等空间、取的人等清单，死锁到超时。

**改成 TCP，问题从设计里消失**（TCP 段不需要注册给网卡）：

```bash
MOONCAKE_PROTOCOL=tcp
MOONCAKE_DEVICE_NAME=                       # 空
MOONCAKE_GLOBAL_SEGMENT_SIZE=128GB          # 32768 窗口要提到 256-384GB，见 8.6
MOONCAKE_LOCAL_BUFFER_SIZE=8GB
LUMENRL_TEACHER_MOONCAKE_SEGMENT_POOL_SIZE=1     # 段池坍缩成 1
LUMENRL_TEACHER_MOONCAKE_SEGMENT_SIZE=128GB
LUMENRL_DRAFT_MOONCAKE_SEGMENT_SIZE=4GB          # draft 只读，段可以小
```

/!\ 值为空时不要下发 `mooncake.device_name=` 这个 override —— 等号后为空会被解析成
`None`，报 `Incompatible value 'None' for field of type 'str'`。

**NCCL 仍然走 RDMA**（draft 节点内 8 卡的梯度 all-reduce），启动脚本里这四个一起设：

```bash
NCCL_IB_HCA=ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6   # 没有 ionic_7
NCCL_IB_GID_INDEX=1
NCCL_DMABUF_ENABLE=1
NCCL_SOCKET_IFNAME=spur0
```

`ionic_7` 写错会有一路 fallback（能跑但慢）。上面这些和 Mooncake 那组都是
`run_multinode_rank.sh` 里带默认值的 export，覆盖它们只要在调用时传同名环境变量。

/!\ **同一个脚本里的路径默认值写死了别人的用户名**（`DATA_ROOT` 默认
`/mnt/m2m_nobackup/danyzhan`、`SHARED_ROOT` 默认 `/shared_nfs/danyzhan/lumenrl`）。
新人不显式传 `DATA_ROOT` / `SHARED_ROOT` / `MODEL_PATH` / `DATASET_PATH`，会指向不存在
或不属于自己的目录。

实测带宽：teacher put 2.0 GiB/s、draft 8 并发跨节点 get 4.44–5.27 GiB/s。

### 5.7 启动

**如果五个节点是五个独立的单节点作业**（而不是一个 `-N5` 分配），不能直接
`srun --nodes=5` —— 那会排队等自己占着的节点。启动脚本只读 `SLURM_PROCID` 和
`SLURM_NTASKS` 并通过共享盘上的文件协调，所以用五个 `spur exec` 手工设这两个变量即可。

**必须并行起，而且要放宽节点发现的超时。** `spur exec` 在留下后台进程时约 100 秒才返回，
顺序起五个会让第一个和最后一个节点的协调文件差 8 分钟，而默认发现窗口只有 120 秒
（放宽到 600 秒）。

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

**Ray 的端口要挪开。** 容器用 `--network host`，Ray 默认的 client server 端口 10001 很
容易被同机其他租户占着，`ray start --head` 会以 `Failed to bind to address
127.0.0.1:10001` 退出，而外层只看到 "Ray cluster has 0/5 active nodes"。用
`--ray-client-server-port 26380 --include-dashboard=false`，主端口挪到 26379。

/!\ **判断启动成功要看 `$COORD_DIR` 里有没有 `node-0..4` 和 `ray-0..4`，不要看
`launch.sh` 的 stdout** —— `spur exec` 留下后台进程时返回很慢，实测 5 个 rank 全起来了
但 stdout 只打了 2 行就被 `timeout 900` 掐掉。

**要有看守进程。** 三种失败都值得自动处理：训练进程死亡、进程活着但 step 计数器不动
（RCCL 死锁的表现是 `docker ps` 正常、GPU 100% 但功耗只有 320 W 的自旋）、spur 作业消失。
恢复动作都一样：干净停掉，用新的 run id 加 `checkpointing.resume=true` 重新拉起。
**但看守必须自检恢复是否真的接上了** —— 比较重启后第一条 step 行和重启前的 checkpoint
编号，一个"成功恢复"但实际从头开始的重启，会让你在十几个小时后才发现。

**checkpoint 要自己清。** `save_steps: 100`、一次 33 GB。trainer 永远不会替我们删，
而共享盘是别人也在写的 —— 实测一天内 `/shared_nfs` 从 11 TB 涨到 28 TB 又掉到 2.9 TB，
别人的吃盘速度到过 1.2–1.6 TB/h。盘满的表现是 `torch.save` 写一半抛异常 → torchrun
拆掉 job → 看守从最后一个完好的 checkpoint 续跑（损失 ≤ 500 步）。用一个"保留末尾 8 个
+ 每 5000 步一个梯子"的清理脚本把占用压在恒定值。注意 NFS 的 `df` 有约 40 秒滞后。

/!\ `num_training_steps` 如果不是 `save_steps` 的倍数，训练结束时**不会补存最后一份**。

### 5.8 eval：必须用两把尺子，而且要和目标 benchmark 语言匹配

eval 集默认取训练集尾部，**换数据集就等于换评测基准**，两代之间不可比。

**gen-5 用这件事换了一个教训**：训练侧 eval AL 从 2.017 涨到 2.767（+37%），而服务端
MT-Bench 只涨 4%，兑现率从 89.9% 掉到 **69.5%**。原因是 eval 取训练集尾部，而那份语料
74.8% 是多语种 —— 它量的分布和四个英文 benchmark 都不重合。gen-5 还主动去掉了
`eval_old`（因为是从零训，"验权重导入"这个作用确实不存在了），代价是**训练全程没有
任何信号能提示服务端的真实位置**，直到跑完做 benchmark 才知道。

**规矩**：语料分布和目标 benchmark 不一致时，固定的、语言匹配的 eval 切片不是可选项。
它便宜（一个 eval 点约 8 秒），而它挡住的是"训练指标漂亮、服务端没动"这一整类误判 ——
gen-1 就是这么栽的（eval AL 1.863，服务端 1.42）。

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

`eval_old` 是把上一代那 128 行**原样带过来**（同一份旧数据集、同一 seed、同一预处理
参数），用离线脚本 materialise 成固定文件并记录指纹。

### 5.9 接着上一代训：draft 权重导入

从零训和接着已有 draft 训是两条路。**接着训有一个会静默毁掉整轮的坑。**

通用的 `draft.resume_from` 分支是给 Eagle3 写的，它把 `midlayer.` 改成 `layers.0.`、
`norm` 改成 `out_norm`。拿它加载 DSpark 的导出，每个张量都变成 unexpected key，
`strict=False` 保留随机初始化，**什么都不报错** —— loss 从冷启动值开始，看起来只是
"收敛得慢"。

正确的映射（ATOM 命名 → LumenRL 命名）：

| 导出里的名字 | 模型里的名字 |
|---|---|
| `context_proj.weight` | `fc.weight` |
| `context_norm.weight` | `hidden_norm.weight` |
| `final_norm.weight` | `norm.weight` |
| `layers.*` / `markov_head.*` / `confidence_head.*` | 同名 |
| `embed_tokens.weight` | **跳过**（teacher 的冻结 embedding，训练器从 teacher 拿） |

要**断言**：每个导出张量都必须有归宿，每个模型参数都必须被填上，形状必须一致。加载成功
的标志是 **67 个张量 / 2,387,907,841 个参数**。

**上线前用 `lr=0` 验一次。** 跑两步、学习率设 0，eval 出来的 AL 必须和上一代记录的数字
一致。实测 1.61526 与 gen-3 step 3280 记录的逐位相同 —— 一次性验完了权重导入、teacher
hidden state、eval 分片三条链路。这个检查花 12 分钟，能挡住的错误值好几天。

**从 checkpoint 续训**（和上面那条导入路径不同）：

```yaml
num_training_steps: 78296     # 改大即可，见下
checkpointing:
  resume: true                # 从 checkpoint_dir 里最新的那个续
```

三个必须知道的细节：

1. **`num_training_steps` 改大不会和恢复出来的 scheduler 冲突。** resume 只恢复
   `scheduler.last_epoch`，`total_steps` 是 setup 时从配置重建的，所以 lr 曲线按新总步数
   重算。step 39,000 / 78,296 处算出 **4.50e-5**，相对第一个 epoch 收尾的 3.93e-5 是
   +14.6%，之后再余弦回到 3.93e-5。**刻意不去平滑它** —— 带着完全退火的 lr 进第二个
   epoch 就是 `min_lr: 0` 那次的下场。
2. **`draft.from_scratch` 保持 `true` 不是矛盾。** 它管的是 `draft.resume_from` 那条
   导入路径（不用）。权重从 `checkpoint_dir` 经 `_resume_from_checkpoint` 进来，那一步
   在 draft 建好之后跑、会覆盖随机初始化。它用 `strict=False` 加载，**所以要去日志里
   确认那行的 `missing=` 是空的**。
3. **Adam 动量必须带过来。** 重置动量会造成 **40% → 8%** 的精度回退，要几千步才能恢复。

采样器走 `(step * bs) % _trainable_len`，`_trainable_len` 保证回绕落在第 0 行而不是
eval 尾部。

---

## 6. 导出

```bash
python3 output/export_dspark_hf.py \
    --ckpt <共享路径>/checkpoints/.../checkpoint_39000.pt \
    --base-model <K3 路径> \
    --output <共享路径>/drafts/gen5-step39000 \
    --train-config examples/Kimi_K3_SDDD_MI350_ATOM/configs/train.yaml
```

导出脚本做两件断言，**都不能跳过**：

1. **config.json 与训练 YAML 交叉核对。** 特别是不能 emit `rope_interleave` —— ATOM 会
   把它映射成 `is_neox_style`，也就是训练器没在用的那种半分旋转（见 3.1）。
2. **68 个张量名的 key 契约。** ATOM 对不认识的张量名**不告警不报错，直接静默忽略**。

/!\ 曾经导出过一份用 LumenRL 内部命名（`fc` / `hidden_norm` / `norm`）的权重，ATOM 把
躯干加载了、**静默丢掉了整条输入通路**，结果是 first-token 准确率 59% 的 draft 在服务端
接受率 **0.00%，3060 个前向步一次都没中**，而 server 日志从头到尾完全健康。

成功的标志：`ATOM key contract: OK (68 tensors)`、`Total size: 7.12 GB`
（3.562B 参数，bf16）。

导出的 `config.json` 里几个要核对的字段：

```json
{
  "architectures": ["K3DSparkModel"],   "model_type": "k3_dspark",
  "hidden_size": 7168,      "num_hidden_layers": 5,
  "target_num_hidden_layers": 93,
  "target_layer_ids": [2, 23, 47, 71, 89],
  "max_position_embeddings": 1048576,
  "rope_parameters": {
    "rope_type": "yarn", "factor": 32.0,
    "original_max_position_embeddings": 32768,
    "rope_theta": 50000.0, "beta_fast": 32, "beta_slow": 1,
    "mscale": 1.0, "mscale_all_dim": 1.0
  }
}
```

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

`max_num_seqs` 用 8 就够 —— 客户端是串行的，而 KDA 循环状态按槽位分配，64 时状态池要
28.91 GB 而 KV 预算只有约 19 GB，引擎会拒绝启动。服务端约 271 秒起来（本地 NVMe）。
启动后日志里应该有 `Detected MLA DSpark drafter` 和
`DSparkProposer aux capture on target layers: (2, 23, 47, 71, 89)`。

### 7.2 测量协议

接受长度从引擎自己的计数器 `/debug/mtp_stats` 读，**不要从时间反推**。计数器是自服务
启动以来累计的，所以每个题目集的测量是"跑之前拍一张快照、跑完再拍一张、取差值" ——
这样多个题目集可以共用一个服务（K3 加载一次要 4.5 分钟，每集重启一次纯属浪费）。

协议：`temperature=0`、**并发 1**、`num_speculative_tokens=7`、fp8 KV、Kimi-K3 chat
template、官方 prompt 数。

**客户端必须能扛住引擎卡死。** 有过一次 aiter 跨 rank 广播死锁：8 个 rank 全部 200% CPU
自旋、显存 248 GB/卡、aiter 每 60 秒打印 `No available shared memory broadcast block
found`，而 **`/health` 仍返回 200**（那是 API 前端，和 engine core 是两个进程），
客户端把 550 条已完成的工作全丢了。三轮约 3,800 个请求只死这一次，触发条件没找到
（"请求数阈值"的解释被后一轮 674 个请求无死锁推翻），**钉死镜像也规避不了**。

所以客户端要有三件东西，而且它们能成立是因为**接受长度分布在前向步上可加**
（temperature=0 贪心、prefix caching 关，所以跨服务生命周期累计等价于一次跑完）：

```python
EXIT_ENGINE_STUCK = 42      # 单请求超时 -> 退出码 42
EXIT_ROTATE       = 43      # 跑满 --rotate-after 条主动轮转 -> 43

--request-timeout 900       # 超时即上报卡死
--rotate-after 400          # 主动重启服务，不等它死
--resume                    # 跳过已完成条目
--flush-every 25            # 每 25 条把计数器增量折叠进累计分布并落盘
```

外层驱动看退出码决定是否重启续跑（42/43 都重启，非预期码也重启）。实测两轮的 GSM8K
各跨 **4 个服务生命周期**，之后 3,106 个请求没再死锁；每次轮转代价约 4 分钟（K3 重载）。

驱动脚本的用法：

```bash
DRAFT=/shared_nfs/.../drafts/gen5-step39000 \
LABEL=gen5-t0 \
SETS="mtbench gsm8k humaneval aime2026 math500 mbpp swebench_pro speed_coding ..." \
MAX_TOKENS=12288 MAX_MODEL_LEN=16384 \
REQUEST_TIMEOUT=900 ROTATE_AFTER=400 MAX_RESTARTS=12 \
bash run_bench.sh
```

它内部调 `benchmark_dspark.sh <draft_dir> <label> <num_prompts> <max_tokens>`，
`num_prompts=0` 表示全量。长上下文那一集要单独用 `MAX_MODEL_LEN=32768` 再跑一遍。

结果落在 `k3-bench/results/bench_<label>-<set>.json`，进行中的是同名
`.partial.json`。JSON 顶层字段：

```json
{
  "n": 80, "protocol": {"max_tokens": 12288, "temperature": 0.0, "num_speculative_tokens": 7, "concurrency": 1},
  "forward_steps": 45584, "accepted_draft_tokens": 82613,
  "tok_per_fwd": 2.8123, "acceptance_rate": 0.2589,
  "accepted_length_distribution": {"0": 13440, "1": 12331, "...": "..."},
  "completion_tokens": {"mean": 1603.5, "median": 1287, "max": 7544, "capped": 0},
  "per_prompt": [{"id": "...", "seconds": 64.1, "prompt_tokens": 109, "completion_tokens": 2703, "finish_reason": "stop"}],
  "elapsed_s": 1247.6, "tok_per_s": 102.8
}
```

`completion_tokens.capped` 是补全长度 ≥ `max_tokens − 16` 的条数，用来判断有没有被
上限截断。

### 7.2b 两个关于测量本身的结论

这两条都是受控实验得出的，能帮你不做无谓的重跑：

**`max_tokens` 几乎不影响接受长度。** 同一 checkpoint、同一全量 prompt，只把
`max_tokens` 从 2048 提到 15360：

| 集 | 2048 | 15360 | 变化 | 补全均值变化 | 前向步数变化 |
|---|---|---|---|---|---|
| MT-Bench | 2.1573 | 2.1101 | −2.2% | 1,201 → 1,616 | +38% |
| GSM8K | 3.9509 | 3.9795 | +0.7% | 256 → 253 | −1.8% |
| HumanEval | 3.1521 | 3.1009 | −1.6% | 1,037 → 1,100 | +7.8% |
| AIME 2026 | 1.8645 | 1.8682 | +0.2% | 1,697 → **6,684** | **+293%** |

AIME 的生成长度涨了 3.9 倍、前向步数涨了 4 倍，`tok/fwd` 只动 **+0.2%**。
**"接受率在推理轨迹深处更高、短上限严重低估"这个假设是错的** —— 接受长度在生成方向
很快饱和，几百 token 之后就是稳态。

**12 条 prompt 的抽样噪声是 ±5% 量级**，比协议差异大一倍。所以要么跑全量，要么别拿
小样本的数字做世代对比。

（`15360` 的来历：`16384 − 1024 = 15360`，最长 prompt 是 MT-Bench 的约 547 token，
留 1024 余量。）

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
| SPEED-Bench 长上下文 | `nvidia/SPEED-Bench` config `throughput_16k`，`category == low_entropy` | 512 |

**AA-LCR 测不了，而且不能糊弄。** card 是在 71k–115k token 的多文档 prompt 上测的，
而数据集暴露的字段只有问题本身（平均 206 字符）。拿光秃秃的问题去跑，等于用同一个名字
测了另一个 benchmark。

`speed_throughput16k` 的输入平均约 9.5k token、最长约 22.5k，**需要单独用
`--max-model-len 32768` 的服务端**。其余 12 集在 16384 窗口下用 `max_tokens=12288`
（留出 4096 给最长的 prompt）。

### 7.4 最重要的一条：分母要用同机同引擎测出来

**不要拿你在 ATOM 上的数字去除以官方 card 在 vLLM 上的数字。** 把官方
`Inferact/Kimi-K3-DSpark` 下载下来，在**同一台机器、同一镜像、同一协议、同样的 prompt
数**下测一遍，用它当分母。

实测：同一个官方 draft 在本机相对它自己 card 的偏差是 MT-Bench −1.2%、AIME **+1.5%**、
GSM8K −13.1%、HumanEval −23.8% —— **范围从 +1.5% 到 −23.8%，和数据集强相关，不存在
可外推的统一系数**。"ATOM 读数系统性低约 10%"这种说法来自单点、12 条 prompt 的测量，
用它换算是错的。

顺带一个容易搞错的算术：官方 card 上那个 **3.85 是 14 个 benchmark 的总均值**，不是
任意四集的均值。四集（GSM8K/HumanEval/MT-Bench/AIME）的 card 均值是
`(5.64+5.34+3.14+2.72)/4 = 4.21`。

---

## 8. 长上下文：把窗口拉到 32768

gen-5 在 13 集上做到官方的 91.1%，**只有 `speed_throughput16k` 是 41.7%**。这一节是
补这个窟窿的做法，**结果是那一集做到 104.5%、13 集均值 105.0%**（10.4）。

### 8.1 先确认因果，再动手

| | 超过 8191 token 的 prompt 占比 | prompt 均长 | gen5/官方 |
|---|---|---|---|
| speed_throughput16k | **58.9%** | 9,527 | **41.7%** |
| 其余 12 集 | **0.0%** | 全部 < 4,096 | 84.9–102.8% |

**唯一有长 prompt 的集，恰好是唯一垮掉的集。** 而 gen-5 的训练数据在 8192 窗口下
`drop_overlong` 掉了 34.7 万行，存活行 max 8,191 —— draft 从来没有在位置 8192 以上做过
一次预测。接受率 11.5% 对官方 47.5%，比 tok/fwd 的差距更刺眼。

### 8.2 判据：不是"哪些样本长"，是"哪些样本被窗口截断过"

最准的信号是**上一轮撞了生成上限**：`finish_reason == "max_tokens"` 意味着 K3 还想往下
写、被窗口拦住了。九类池里这类占 **7.11%（40.2 万行）**，它们上一轮的答案 mean 15,883
/ p50 16,109，**整整齐齐顶在 16,260 这个天花板上**。

/!\ `finish_reason` 的取值：无 draft 时 EOS 报 `stop_163586`、开 draft 报 `eos`、
撞上限报 `max_tokens`。判据必须是 `== "max_tokens"`，不是 `!= "stop"`。

用 32768 重生成之后（实测 9,284 条）：

```
mean 20,373  p50 21,723  p90 32,543  max 32,640
>8,191: 78.8%   >16,383: 61.8%   >32,000: 28.3%
```

**61.8% 超过 16,383** —— 这部分是旧窗口物理上写不出来的，是纯增量。另有 28.3% 顶到了
32,000，说明 32768 下仍有近三成被截断。

### 8.3 长 prompt 是便宜十倍的长上下文数据

一开始我按"32768 下要留够 2048 作答预算"把 prompt 超过 30,686 的行丢了。**这个判据是
错的**：那是生成效率的思路，套到训练数据上方向是反的。训练要的是**序列总长**，不在乎
长度来自 prompt 还是答案；而生成成本只跟答案长度有关：

| | 生成开销 | 得到的序列 | 每个长 token 的成本 |
|---|---|---|---|
| 撞上限行（500 prompt + 2 万答案） | 2 万 token | 2.05 万 | 1.0 |
| 长 prompt 行（3 万 prompt + 1.2 千答案） | 1.2 千 token | 3.1 万 | **约 0.05** |

上限应该取 **32,224 = 32768 − 512 − 32**（只保证 512 的最低作答预算）。

哪里有长 prompt：

| 池子 | prompt 均长 | ≥ 8,000 的占比 |
|---|---|---|
| nine-category（69 万抽样） | 310 | **0.02%** |
| capped（撞上限那批） | 469 | 0.06% |
| **kimi-mtp** | **5,459** | **34.88%** |

**kimi-mtp 是唯一的廉价来源。** 如果它的第一轮响应文件不在手边（拿不到 `finish_reason`），
可以反推：源数据集里有、已校验集里没有的行，加上 prompt ≥ 8,000 的行。

### 8.4 撞上限那批 91.7% 是多语种 —— 要重筛

| 分类 | capped 数 | 占比 | 池中总数 | 该类撞限率 |
|---|---|---|---|---|
| multilingual ×5 | 364,649 | **90.8%** | 4,453,283 | 7.1–8.6% |
| chat | 11,485 | 2.9% | 439,845 | 2.6% |
| math | 9,492 | 2.4% | 101,962 | 9.3% |
| code | 7,620 | 1.9% | 31,261 | **24.4%** |
| stem | 4,792 | 1.2% | 354,758 | 1.4% |
| aya | 3,505 | 0.9% | 165,863 | 2.1% |

**不是多语种更容易写长**（撞限率反而低于 code 的 24.4% 和 math 的 9.3%），纯粹是基数
压倒。抽样看题面有个缓和因素：这些多语种行**内容本身就是数学题和竞赛编程题**，只是换了
语言外壳。但既然目标是"增编程数学、减多语种"，还是重筛成只含 `code / math / stem /
chat` 的 32,552 条。

### 8.5 各批实测长度与训练耗时

| 批次 | 行数 | prompt | 答案 | 序列 |
|---|---|---|---|---|
| capped 英文 | 15,000 | 469 | 20,373 | 20,842 |
| capped 多语种（先跑的那批，保留） | 9,957 | 469 | 20,373 | 20,842 |
| mtp | 29,944（池子全量） | 5,459 | 3,814 | 9,273 |
| OCI | 300,000 | 306 | 4,598 | 4,904 |

/!\ mtp 那一行原计划只取 8,000（每分片 1,000），实际会跑满 29,944 —— 这正是 9.4 节
那条"target 只在两次 attempt **之间**检查、不在运行中截断"的实例。mtp 便宜（3.3 个
节点小时），跑满无所谓，但**如果某一批你真的要卡在指定行数，得手动停容器**。

**决定训练耗时的是数据均长，不是窗口大小。** teacher 的开销是 prefill 计算加上逐 token
84 KiB 的 hidden state 发布，两者都按 token 数走。gen-5 基线 128 × 2,009 = 25.7 万
token/步、**6.13 s**（全程实测 mean），即约 **41,950 token/s** 系统吞吐。

| gen-6 数据组成 | 总行数 | 均长 | s/步 | 2 万步 | epoch |
|---|---|---|---|---|---|
| 只用新数据 | 33.3 万 | 6,204 | 18.9 | 105–126 h | 7.7 |
| 新数据 + 100 万旧语料 | 133 万 | 3,057 | 9.3 | 52–62 h | 1.9 |
| **新数据 + 200 万旧语料** | **233 万** | **2,608** | **8.0** | **44–53 h** | **1.10** |
| 新数据 + 全部 536 万旧语料 | 569 万 | 2,254 | 6.9 | 38–46 h | 0.45 |

区间上界给 attention 的 L² 项留了 20% 余量（这一项没有实测数据）。推荐第三行：2 万步
约 1.1 epoch 不重复喂数据。"只用新数据"要跑 7.7 个 epoch，必然过拟合。

参照系：gen-5 的 39,148 步实际墙钟 **71.6 小时**（纯 step 66.7 小时，差值是 checkpoint、
eval 和启动加载）。

**这个配比下 2 万步共喂 6.68B token，落在位置 8192 以上的约 0.36B，占 5.3%。** 在长
上下文扩展的常见区间（5–10%）里，但是薄边际。提升不够时最直接的杠杆是**提高长样本在
采样中的权重**，不是加步数。

### 8.6 续训而不是重训，学习率用 WSD

用 5.9 那条实测缩放律（3,358 步 → 2.7647，39,148 步 → 3.5087，即 **+0.210 AL/翻倍**）：

| 方案 | 预期（四集均值） |
|---|---|
| 从头训 2 万步 | 3.305 |
| **续训** | **3.634** |
| gen-5 现在 | 3.509 |

**从头训 2 万步会比 gen-5 现在还低 0.203** —— 2 万步比已经走过的 39,148 步还少，等于
把积累扔掉重来。第二个理由更本质：**"先短后长"本来就是长上下文扩展的标准做法**。

结论被 gen-6 验证了：续训 9,878 步（不到预案里 2 万步的一半），13 集均值从
**3.3162 到 3.8218**，固定 eval 尺子的增益还跑赢了纯步数外推（10.3）。

**学习率不要照搬 gen-5 的 min_lr 3.93e-5。** 那个值是为"同一份语料的第二、三个 epoch"
留余量的。照搬到续训上，入口 4.23e-5、收尾 3.93e-5，**跨度 1 倍，整个续训等于恒定学习率，
末尾完全没有退火**。但也不能简单设 0：从 66% 处切进 cosine 时
`入口 lr = min_lr + (peak − min_lr) × 0.2761`，min_lr 设 0 会让入口掉到 1.38e-5，
比 gen-5 收尾还低 65%。

**cosine 把入口和收尾绑死了，WSD 没有这个问题**（`policy.wsd_decay_ratio` /
`wsd_decay_style` 已接好，不用改代码）。它退火开始前 `coeff = 1.0`，即 lr 恒等于
`learning_rate`：

gen-6 实际用的值（`configs/train_stage2.yaml`，**新建一个文件，不要改 `train.yaml`**）：

```yaml
policy:
  learning_rate: 4.2e-5     # WSD 下这就是恒定段的 lr
  min_lr: 1.0e-5            # /!\ 不是 0，见下
  lr_decay_style: WSD
  wsd_decay_ratio: 0.0504   # int(0.0504 x 49026) = 2470 步
num_training_steps: 49026   # 39148 + 9878（一个 epoch 的实测步数）
```

得到：39,148–46,556 恒定在 4.20e-5，之后余弦退到 1.00e-5。实测逐点吻合公式
（step 47,750 读数 2.6834e-05，代入 `min_lr + (peak−min_lr)·0.5·(1+cos(π·0.4834))`
也是 2.6834e-05），终点精确落在 `min_lr`。

/!\ `wsd_decay_steps = int(wsd_decay_ratio × total_steps)`，是对**总步数**取，不是对
续训的那些步。

/!\ **`min_lr` 不要设 0。** gen-3 单 epoch + `min_lr 0` 把 lr 退到了 0，下一代想接着
训就没有非零的入口 lr 可用，只能从零训。留 1.0e-5 是下一代续训的入场券。

WSD 同时满足了两个原本矛盾的目标：**恒定段内任何一步停下来接着训，调度都没有断点**
（这正是当初设 min_lr 3.93e-5 想要的"留余量"），末尾还白送一次退火。

### 8.7 换到 32768 要一起改的配置

| 项 | 8192 | 32768 |
|---|---|---|
| `policy.max_total_sequence_length` | 8192 | 32768 |
| teacher `max_model_len` / `max_num_batched_tokens` | 8192 | 32768 |
| teacher `max_num_seqs` | 32 | **24**（见下） |
| 生成侧 `--max-prompt-tokens` | 14304 | **32224** |
| `mooncake.global_segment_size` | 128GB | **256–384GB** |
| 预处理缓存 | 8192 口径 | **必须重建** |

/!\ **`max_num_batched_tokens` 必须 ≥ `max_model_len`**，因为 `enable_chunked_prefill`
是 `false`。写成 30720 这种"看起来省显存"的值，30,720–32,768 长度的序列会被静默截断，
hidden state 就错了而且不报错。

**`max_num_seqs` 降到 24** 是 gen-6 唯一真正卡住的问题的解药：32 的时候某些 batch
打包会让一个 teacher 副本挂死在 `extract_hidden` 里，600 秒超时后整个 run 被带走，
而且续训会确定性地重放同一个 batch、每次都死在同一步（10.5）。降到 24 之后 teacher
单步 3.57 s vs 3.33 s，**在噪声范围内，等于没有代价**。

**Mooncake 内存要重算。** 128GB 是按"每批约 24 GiB、8192 全满时 88 GiB、每副本两批"
定的。均长 6,204 时一批就是 `128 × 6204 × 84 KiB ≈ 50 GiB`，两批 99 GiB —— 勉强够但
没余量；32768 全满的极端批是 344 GiB，直接爆。主机内存 2.8 TB，提到 256–384 GB 即可。

`local_buffer_size: 8GB` 不用改：32768 单序列是 2.7 GiB（8192 时 672 MiB）。

**模型不用动**：`rope_scaling.original_max_pos` 本来就是 32768，checkpoint 里存的
（`model_state_dict` / `optimizer_state_dict` / `fp32_params` /
`scheduler_last_epoch` / `step`）全都与序列长度无关。

### 8.8 smoke test 要测的三件事

1. **step 时间**：验证 8.5 的线性外推。`timing/fetch_wait_s` 接近 0 说明预取重叠仍然
   生效；涨回 `teacher_s` 说明瓶颈换成了 teacher。
2. **Mooncake 显存/内存**：`global_segment_size` 够不够。
3. **位置 8192 以上真的被训到**：用长样本子集，确认 loss mask 和 hidden state 在那一段
   是对齐的 —— 5.5 节那条掩码宽度 clamp 的 warning 在长序列下更容易触发。

---

## 9. 大规模生成的运维

生成 30 万到 100 万条 K3 回答要几十到几百个节点小时，跨越好几个夜晚。这一节是那套编排
踩出来的东西。**每一条都对应一次真实的机时浪费。**

### 9.1 容器 "Up" 不等于还活着

编排判断"这台机器忙不忙"如果只看有没有容器，会漏掉一整类故障。实测一夜：一台的引擎
18:54 就停了（GPU 掉到 0%）、另一台 23:28 卡死（**GPU 还是 100% 在空转**，多半是 NCCL
集合通信死锁），**两个容器都一直显示 "Up"**，于是编排认为节点忙、整晚没管它们 ——
白丢 **15.9 个节点小时**。supervise 也发现不了，它在等 `docker wait`，而容器根本没退出。

唯一可靠的活性信号是**输出文件的 mtime**：

```
起步阶段不判：装权重约 7 分钟，第一行还要生成两万 token 才落盘，给足 60 分钟宽限
之后 30 分钟没产出就判定卡死，杀掉容器让 supervise 重启（已完成的行会被续跑跳过）
```

/!\ 杀掉容器后要**仍然向调度逻辑报"忙"**：supervise 的 `docker wait` 这时会返回并自动
重试，这一轮别再派新活，否则会和 supervise 重启的那个撞在同一台机器上抢卡。万一
supervise 真的已经死了，下一轮看到没有容器自然会当空闲重新派。

上线当天就抓到一次真故障：日志停在 05:39、文件停在 05:50、检测器 06:16 杀掉，
supervise 从 26,731 行续跑。

### 9.2 另外三个会静默烧机器的东西

**编排脚本的 `kill` 可能不生效。** bash 要等前台子进程结束才处理信号，而编排循环里有
耗时几分钟的远程调用。`kill` 之后 `sleep 3` 就重启，会得到**两个编排实例**，旧的那个
内存里还是旧的函数定义，会派出已经废弃的工作单元。要 `kill -9` 并确认。

**停容器和杀 supervise 要成对。** 只杀进程不清容器，节点会一直被判定为忙；只停容器
不杀 supervise，它会把容器再拉起来。

**显存释放有十几秒空窗。** 容器退出后"有没有容器"立刻返回否，但 ROCm 还没把显存还回去
（实测两次采样之间从 1948 GB 掉到 2 GB）。这期间"节点是否就绪"的检查会因显存那一条失败，
进而触发重新铺机，而铺机的显存预检看到 >210 GB 会把节点**拉黑 12 小时**。180 秒一轮撞上
的概率约 8%，代价是整晚失去一台机器。**失败后等 45 秒复检**就能堵住。

### 9.3 抢到节点不等于能用

`--exclusive` 拿到节点**并不驱逐别人已经在跑的 docker 容器**。实测一台：作业到手、
权重铺好，但另一位同事的 SGLang 容器占着 8 张卡各约 203 GB，K3 要 217.6 GB/卡，引擎以
`Received unexpected SHUTDOWN signal from DP rank 0 during initialization` 失败，
supervise 白重试 5 轮。

所以铺机前要有**预检**，失败就拉黑并立刻释放作业：

| 检查 | 阈值 | 空闲基线 |
|---|---|---|
| 每卡显存占用 | 任一卡 > 210 GB 就拒 | 约 0.28 GB/卡（合计 2.2 GB） |
| `/mnt/m2m_nobackup` 剩余 | < 2 TB 就拒 | 需要 1.56 TB 放 K3 权重 + 余量 |

（另一台的故障是本地盘被别人的 23 TB 缓存占满，docker pull 报
`no space left on device`，而铺机脚本没检查退出码、报了成功。）

### 9.4 会无限重启的"零产出"单元：要有放弃机制（根因未解）

**症状**：8 个 mtp 分片里 7 个跑满 3,743 行，**s0 连续 7 次一行都产不出来**。每次都是
engine 正常初始化（4.2 分钟）、warmup 通过、KV 预算正常（`available_for_kv=44.51GB`、
118,559 blocks）、请求正常提交，然后 2–20 分钟后 8 个 rank 一起 TCPStore/NCCL 报错、
**显存掉到 0 而容器仍显示 Up**。活性检测每小时杀一次、supervise 每小时重启一次，
**白烧 6.5 个节点小时**。

**排除掉的解释**（都查过，都不是）：

- **不是数据分布**：s0 的 prompt 长度和别的分片几乎一样（mean 5,534 / max 31,653，
  而跑成功的 s1 是 5,426 / **31,855**，比它还长）
- **不是节点坏**：同一台机器前一天跑满了 12 小时的 capped 分片，权重 96 个分片齐、
  本地盘 8% 占用
- **不是处理顺序**：`--shuffle-seed` 换成别的值，崩溃点从第 2 分钟推到第 20 分钟，
  **照样死**。（我一度以为换种子解决了 —— 那是在它还在 prefill 爬坡期就下了结论，
  **看到 GPU 100% 和调度推进不等于它能活过第一批**。）

**目前只知道**：崩前最后的日志是大批 prefill 走 aiter 未调优的 GEMM 路径
（`M:29448, N:6284, K:7168 ... not found tuned config`）。mtp 的 prompt 长
（p90 12,507），几条凑一起就逼近 `max_num_batched_tokens` 的 32,768，怀疑是某个
批形状触发的 aiter/RCCL 问题，但没有定位到具体行。

**所以要的不是修复，是止损。** 编排里加了三段：

```bash
ZK_RESEED=2      # 连续几次"零产出被杀"后换 shuffle 种子再试
ZK_GIVEUP=6      # 再不行就彻底跳过这个单元，别再占机器
zero_kills() { cat "$STATE/zerokill_$1" 2>/dev/null || echo 0; }
```

1. 活性检测杀容器时，**看它有没有产出过**：0 行就给这个单元的计数 +1，有行就清零
   （有产出的卡死是另一类，NCCL 死锁那种，重启就好）
2. 计数到 `ZK_RESEED` 时**连 supervise 一起杀**。只杀容器没用 —— supervise 会用
   一模一样的命令行原地重启，参数不变就还是同一个顺序。杀掉它单元才会变成未分配，
   下一轮才能带新种子重派。
3. 计数到 `ZK_GIVEUP` 就在派工时直接跳过。手工放弃某个单元：
   `echo 99 > $STATE/zerokill_<unit>`。

换种子这一步保留是因为它便宜且无害（种子只影响处理顺序，temperature=0 贪心，
不影响生成内容），但**别指望它一定管用**。

**下次先试降 `max_num_seqs`。** gen-6 训练侧撞到的挂死（10.5）症状是同一类 —— 长
prompt、大批 prefill、引擎进去不出来、日志里也是 aiter 调优表相关的噪音 —— 而那边
把 `max_num_seqs` 从 32 降到 24 就彻底过去了，代价在噪声范围内。**两者是不是同一个
bug 没有验证**，但换种子只是改顺序、并不改变"能拼出多大的批"，而降并发直接让那个
会挂的批形状拼不出来。下次遇到零产出单元，先降并发再考虑换种子。

/!\ 一般化的教训：**活性检测本身会把"崩一次"放大成"每小时崩一次"**。任何"检测到
异常就重启"的机制，都必须配一个"重试到第几次就放弃"的出口，否则它会忠实地把一台
机器锁死在同一个故障上一整夜。

### 9.5 工作单元的顺序就是优先级

队列一度把大批次全排在前面，机器一直被它占满，**小而高价值的批次一行都没跑成**。
顺序按"每节点小时产出的目标 token"排，不是按批次大小排。

另一个坑：**一个工作单元只能分给一台机器**。如果某批只切了 4 份而你有 7 台机器，
等其他批跑完就会有 3 台完全空着。切分份数要 ≥ 机器数。

/!\ supervise 的 target 只在两次 attempt **之间**检查，不会在运行中截断。想让某个分片
停在指定行数，得手动停容器。

### 9.6 开官方 draft 加速生成：不成立，而且原因可量化

直觉是"投机解码能让 K3 生成快几倍"。在**确认显存已释放到 2 GB 的干净节点**上按生产
并发实测，引擎自己给出了答案：

```
Per-request cache tensor (173.44GB for 3072 slots) exceeds available KV budget (41.23GB)
Even --gpu-memory-utilization 1.0 is insufficient (would need 1.39)
```

3072 slots = 384 序列 × (7 投机 token + 1)。单卡 288 GiB 被 K3 权重和 workspace 吃完
只剩约 41 GB 给 KV，投机验证要 173 GB，差 4.2 倍。要塞下必须把并发从 384 砍到约 **91**：

- 不开 draft：382 序列 × 1 token/步 = **382**
- 开 draft：91 × 4.33 tok/fwd ≈ **394**

产出持平，而开 draft 每步还要多做 7 次 draft 前向、验证 720 个 token 而不是 382 个。
**净亏。** 根子在于投机解码换的是**延迟**（benchmark 那种单流场景），而生成是**吞吐**
场景，382 条并发已经把机器喂饱，没有空闲算力可换。

/!\ 测这类东西**必须先停编排、确认显存归零**。第一次测的时候编排在腾节点的空隙里把另一个
单元派回了同一台机器，两个 K3 引擎抢卡导致 NCCL 挂死、读数也不可信。

未测的变体：`num_speculative_tokens` 调到 1–3 能按 slot 数线性降低 cache 需求。

---

## 10. 五代结果

### 10.1 训练曲线（gen-5，从零训 39,148 步）

| step | eval loss | AL | | step | eval loss | AL |
|---|---|---|---|---|---|---|
| 0 | 14.39 | 0 | | 11,650 | 1.96 | 2.182 |
| 500 | 6.56 | 0.262 | | 21,607 | 1.61 | **2.501** |
| 1,000 | 5.15 | 0.663 | | 27,450 | 1.51 | 2.604 |
| 3,000 | 3.24 | 1.368 | | 35,231 | — | 2.711 |
| **8,269** | 2.21 | **2.001** | | **39,100** | **1.371** | **2.767** |

**曲线在整个 run 上是对数线性的，斜率约 +0.36/翻倍**（1,500→3,000 是 +0.443，
8,250→11,650 是 +0.361，21,607→27,450 是 +0.299）。step 8,000–10,000 附近掉了一档
（0.45 → 0.36）之后就稳住了。

/!\ **外推时用"每翻倍步数"的口径，不要用"每 100 步"** —— 后者会让人误判成持续减速。
中途按"增益 ∝ step^−0.5 持续衰减"拟合，把 AL 2.0 推到 step 11,000–12,000，实际 8,269。

### 10.2 服务端 13 集（gen-5，ATOM 同机同镜像，temperature 0，并发 1）

| 题目集 | n | gen-5 | 官方同机 | gen5/官方 | 接受率 我们/官方 |
|---|---|---|---|---|---|
| **speed_throughput16k** | 512 | **1.805** | **4.327** | **41.7%** | **11.5% / 47.5%** |
| swebench_pro | 200 | 2.947 | 3.471 | 84.9% | 27.8% / 35.3% |
| MBPP | 500 | 3.489 | 3.747 | 93.1% | 35.6% / 39.2% |
| AIME 2026 | 30 | 2.572 | 2.762 | 93.1% | 22.5% / 25.2% |
| GSM8K | 1319 | 4.594 | 4.902 | 93.7% | 51.3% / 55.7% |
| MT-Bench | 80 | 2.924 | 3.102 | 94.3% | 27.5% / 30.0% |
| speed_writing | 80 | 2.696 | 2.820 | 95.6% | 24.2% / 26.0% |
| speed_rag | 80 | 3.421 | 3.556 | 96.2% | 34.6% / 36.5% |
| HumanEval | 164 | 3.944 | 4.070 | 96.9% | 42.1% / 43.9% |
| speed_coding | 80 | 4.499 | 4.575 | 98.3% | 50.0% / 51.1% |
| speed_qa | 80 | 3.112 | 3.070 | 101.4% | 30.2% / 29.6% |
| MATH-500 | 500 | 3.532 | 3.439 | 102.7% | 36.2% / 34.8% |
| speed_multilingual | 80 | 3.575 | 3.476 | 102.8% | 36.8% / 35.4% |
| **13 集均值** | | **3.3162** | **3.6398** | **91.1%** | |

### 10.3 gen-6：在 32768 窗口上续训一个 epoch

gen-5 唯一的短板是长上下文（`speed_throughput16k` 41.7%），gen-6 就只做这一件事：
**把窗口从 16384 换到 32768，补长文本和长代码，从 gen-5 的 checkpoint 续训一个
epoch**（step 39,148 → 49,026，9,878 步；实测 8.9 s/步，纯计算约 24.4 小时）。

数据配比的改动（1,264,631 行）：

| | gen-5 | gen-6 | 为什么 |
|---|---|---|---|
| 多语种 | 74.8% | **12.3%** | gen-5 那 74.8% 是纯浪费，见下 |
| 代码合计 | 0.43% | **20.0%** | `swebench_pro` 84.9% 是第二弱项 |
| OCI（长 prompt） | — | 18.0% | 长 prompt 是便宜十倍的长上下文数据（8.3） |

两把 eval 尺子（5.8）的读数：

| | 起点（gen-5 权重） | 终点（step 49,026） | 变化 |
|---|---|---|---|
| `eval`（gen-6 新分布） | 2.230 | **2.563** | +0.333 |
| `eval_gen5`（gen-5 那 128 行，固定尺子） | 2.767 | **2.972** | +0.205 |

固定尺子涨 +0.205，**跑赢了 gen-5 曲线的外推**：按 10.1 的 +0.36/翻倍，39,148 →
49,026 是 0.325 个翻倍，理论增益 +0.117。也就是说换窗口 + 换配比本身带来了额外增益，
不只是"多训了一些步"。

### 10.4 服务端 13 集（gen-6，与 10.2 同机同镜像同协议）

| 题目集 | n | gen-5 | **gen-6** | 官方同机 | gen6/官方 | vs gen-5 |
|---|---|---|---|---|---|---|
| speed_qa | 80 | 3.112 | **3.366** | 3.070 | 109.6% | +8.2% |
| AIME 2026 | 30 | 2.572 | **3.016** | 2.762 | 109.2% | +17.2% |
| speed_multilingual | 80 | 3.575 | **3.752** | 3.476 | 107.9% | +5.0% |
| HumanEval | 164 | 3.944 | **4.371** | 4.070 | 107.4% | +10.8% |
| MATH-500 | 500 | 3.532 | **3.681** | 3.439 | 107.0% | +4.2% |
| MT-Bench | 80 | 2.924 | **3.314** | 3.102 | 106.8% | +13.3% |
| MBPP | 256 | 3.489 | **4.004** | 3.747 | 106.8% | +14.7% |
| speed_writing | 80 | 2.696 | **2.989** | 2.820 | 106.0% | +10.9% |
| **speed_throughput16k** | 512 | **1.805** | **4.521** | 4.327 | **104.5%** | **+150.5%** |
| speed_rag | 80 | 3.421 | **3.669** | 3.556 | 103.2% | +7.2% |
| speed_coding | 80 | 4.499 | **4.654** | 4.575 | 101.7% | +3.5% |
| **swebench_pro** | 128 | 2.947 | **3.492** | 3.471 | **100.6%** | **+18.5%** |
| GSM8K | 1319 | 4.594 | 4.856 | 4.902 | 99.1% | +5.7% |
| **13 集均值** | | **3.3162 (91.1%)** | **3.8218** | **3.6398** | **105.0%** | **+15.2%** |

**13 集全部高于 gen-5，12 集超过官方 draft**，只有 GSM8K 差 0.9%。长上下文那集的
接受率从 **11.5% 到 50.29%**（官方 47.5%）。

### 10.5 gen-6 唯一真正卡住的问题：teacher 在 `extract_hidden` 里挂死

gen-6 有 **7 次连续重启都死在完全相同的一步（41,570）**，报的都是同一条：

```
TimeoutError: timed out waiting for response to cmd=extract_hidden after 600.0s
RuntimeError: remote ATOM teacher failed: ... AtomTeacherRayActor.process_prefill_batch()
```

一个 ATOM teacher 副本进了 `extract_hidden` 就不返回，trainer 的 600 秒超时到了就
把整个 run 带走。**七次都是同一个副本（043）**。

**为什么能精确复现同一步。** 采样器是 `(step * bs) % _trainable_len`，副本分配也是
step 的确定函数。从 `checkpoint_41500` 续训就是在重放一模一样的 batch 序列，于是
第 41,570 步那个 batch 每次都落到同一个副本上、每次都挂。**同一步反复失败时不要
只是重启 —— 重启一定还死在那里。** 七次 ≈ 5 小时全是白烧的。

**不是坏机器。** 同一条错误后来在 step 44,191 又出现了一次，换到了 039。

**有效的处理：改 teacher 的 batch 打包。** 把 `max_num_seqs` 从 **32 降到 24**，
那个会挂的（batch 组合，GEMM 形状）就再也拼不出来了。改完立刻过了 41,570，又跑了
7,456 步到收敛。代价是零 —— 实测 teacher 单步 3.57 s vs 3.33 s，在噪声范围内。

/!\ **一条被证伪的根因。** 当时注意到失败的 run 里有 16 条
`FlyDSL kernel ... is not recognized by the current catalog; falling back to next
candidate`，而手边那个"正常"的 run 是 0 条，就认定是 aiter 的调优表和内核目录不匹配。
**错了。** 事后把全部 11 个 run 逐个数了一遍：**跑完的那个 run 有 64 条**（是失败 run
的四倍），另有两个失败的 run 是 0 条。当时那个"0 条"的 run 只跑了几十步、还没攒出
回退而已。这条和 `bgemm_internal_cublaslt` 一样是噪音，统计错误时要排除。

教训是一样的：**"失败组有、成功组没有"这种证据，要在两组都跑够长之后再数。**

**想接着查机制的话，`checkpoint_41500` 特意留着没删。** 用 `max_num_seqs: 32` 从它
续训，第 41,570 步必挂 —— 这是目前唯一已知的稳定复现路径。

### 10.6 四条被推翻的结论

**"GSM8K 对这个架构已经饱和"** —— gen-4 的判断来自"加了 7.4 万行 math 却没动"。同样的
5 层 / 7 token 结构在 gen-5 涨了 15.9%（3.96 → 4.59），k=7 全中从 13.9% 到 **28.1%**。
真正的约束是**总训练量**，不是架构、也不是 math 数据本身。

**"code 是最薄的一片，所以 HumanEval 相应最弱"** —— 两轮的 code 绝对量几乎完全一样
（23,083 vs 23,080，都卡在那 17.9% 的去重留存率上），而 HumanEval 从最弱一项
（84.8%）变成**最强一项之一**（96.9%）。HumanEval 缺的不是 code 行数。

**"ATOM 读数系统性低约 10%"** —— 见 7.4，偏差范围是 +1.5% 到 −23.8%，和数据集强相关。

**"多语种要占大头才能保住多语种能力"** —— gen-5 给了多语种 **74.8%** 的配额，
gen-6 砍到 **12.3%**，`speed_multilingual` 不降反升（102.8% → **107.9%**）。腾出来的
配额给了代码（0.43% → 20%），`swebench_pro` 从 84.9% 到 100.6%。**那 74.8% 是纯浪费**。
一个 5 层的 draft 学的是"下一个 token 在局部上下文里怎么接"，这件事跨语言共享的程度
远高于直觉；把同一种能力重复喂 75% 的配额，边际收益早就归零了。

---

## 11. 排错速查

| 症状 | 原因 | 处理 |
|---|---|---|
| `extract_hidden` 600 秒超时，零训练步 | 镜像版本漂了（atom dev323 + torch 2.10），或 Mooncake 走了 RDMA | 用钉死 digest 重 build；`MOONCAKE_PROTOCOL=tcp` |
| `Failed to register memory: Invalid argument [22]` | 同一张网卡上开第二个 RDMA client | 走 TCP，或把段池降到 ≤ 网卡数 |
| `Ray cluster has 0/5 active nodes` | Ray 的 10001 端口被同机其他租户占了 | `--ray-client-server-port 26380 --include-dashboard=false` |
| `node discovery timed out (1/5)` | 五个 rank 是顺序起的，协调文件差了几分钟 | 并行起，发现窗口放宽到 600 秒 |
| `'BF16Optimizer' object is not iterable` | torch 的 `get_optimizer_state_dict` 只接受 `torch.optim.Optimizer` | 非分片时直接用 `opt.state_dict()` |
| `Incompatible value 'None' for field of type 'str'` | override 写成 `mooncake.device_name=`（等号后为空）被解析成 None | 值为空时不要下发这个 override |
| `mkdir /opt/spur/.docker: permission denied` | `spur exec` 的 HOME 不可写 | `HOME=/tmp/x DOCKER_CONFIG=/tmp/x/.docker` |
| 服务端接受率 ≈ 0% 但日志全正常 | 导出的张量名和 ATOM 读的对不上，输入通路被静默丢弃 | 重新导出（key 契约断言会拦下） |
| 训练指标漂亮但服务端很差 | RoPE 旋转约定 / YaRN `mscale²` 与服务端不一致 | 跑 RoPE 对齐自检，相对 L2 误差应在 1e-8 量级 |
| 训练侧 AL 涨但服务端不动 | eval 切片的分布和目标 benchmark 不重合 | 加语言匹配的固定 eval 切片（5.8） |
| eval 指标在某一步突然变好 | 采样器回绕进了 eval 切片 | 确认 `_trainable_len` 排除了尾部的 eval 行 |
| loss 从冷启动值开始、"收敛得慢" | draft 权重导入走了 Eagle3 的重命名分支，全部变成 unexpected key | 用 5.9 的映射表，并断言张量数 67 |
| GPU 显示 100% 但功耗只有约 320 W，无输出 | 集合通信自旋死锁 | 看 worker 日志里的 NCCL watchdog；重启续跑 |
| 生成容器 "Up" 但不产出 | 引擎死了或空转死锁，`docker wait` 不返回 | 按输出文件 mtime 判活性（9.1） |
| 某个分片反复重启、始终 0 行，同批别的分片正常 | 根因未定位（疑似大批 prefill 触发 aiter/RCCL） | 别追，换种子试一次，不行就跳过（9.4） |
| `SHUTDOWN signal from DP rank 0 during initialization` | 别人的容器占着卡，`--exclusive` 不驱逐它 | 铺机前查每卡显存（9.3） |
| 一堆 `bgemm_internal_cublaslt error ... Will attempt to recover` | hipBLASLt 在某些形状上失败后回退到 cublas | 噪音，结果是对的，忽略（统计错误数时要排除） |
| `FlyDSL kernel ... not recognized by the current catalog; falling back` | aiter 调优表里的条目当前内核目录认不出 | 同样是噪音。跑完的 run 比失败的 run 多四倍这种日志（10.5） |
| `timed out waiting for response to cmd=extract_hidden after 600.0s` | 某个 teacher 副本挂死在 prefill 里 | 降 teacher `max_num_seqs`（32→24）改变 batch 打包；**别原样重启**（10.5） |
| 续训每次都死在同一步 | 采样器确定性重放同一个 batch | 改的是打包方式而不是重试次数（10.5） |
| 续训第一个 matmul 就报 `mat2 is on cpu, different from other tensors on cuda:N` | `_resume_from_checkpoint` 故意把 draft 卸到 CPU，靠调用方搬回去，而只有 `_batch_alternating_train` 做了 | `_streaming_disaggregated_train` 开头补一次 `_load_draft_to_gpu()`；gen-5 从零训所以一直没暴露 |
| `No available memory for the cache blocks` | KV 预算不够 | 确认没有别的进程占卡；降 `max_num_seqs` |
| 加载完只剩 175 GiB 可用显存 | `PYTORCH_CUDA_ALLOC_CONF` 漏进容器了 | `docker inspect` 确认它为空 |
| `N 行 prompt 超过 max_prompt_tokens` 启动即退 | 生成侧默认 14304 是给 16384 窗口的 | 32768 窗口传 `--max-prompt-tokens 32224` |
| 训练启动时节点内存爆掉 | `input_ids` 存成 `list[int]`，36 字节/token | 改 int32 ndarray（5.4） |

---

## 12. 一句话流程

```
① 申请节点（--exclusive -G 8），铺机前查每卡显存和本地盘
② 拉钉死 digest 的基础镜像，四项版本核对；build 训练镜像，一台 build 后 save/load 分发
③ 每台下 K3 权重到本地 NVMe，字节级校验 96 个 shard
④ 建 prompt 池：图文过滤 -> 入库即去重 -> 13-gram 去污染 -> 按上限分配配额
⑤ 渲染 prompt（用训练侧同一个 parser），多节点按 stride 切分
⑥ K3 生成（fp8 KV，384 并发，关 draft，滚动提交，watchdog + supervisor）
⑦ 逐行 token 回环校验转成训练集，丢弃率超阈值就停
⑧ CPU 预处理量出存活行数，定 num_training_steps，预热 token cache
⑨ 五节点开训（Mooncake TCP，min_lr 不设 0，两把 eval 尺子，看守带 resume 自检）
⑩ 导出（key 契约断言 68 张量）-> 起 ATOM 服务 -> 跑题目集 -> 同机再测官方 draft 当分母
```

长上下文轮次在 ④ 和 ⑤ 之间插一步：**按 `finish_reason == max_tokens` 捞出被窗口截断
的行、按分类重筛、加上长 prompt 那批**，然后用 32768 窗口重生成（`--max-prompt-tokens
32224`），训练走续训 + WSD（第 8 节）。gen-6 这么做，9,878 步把 13 集均值从官方的
**91.1% 提到 105.0%**。
