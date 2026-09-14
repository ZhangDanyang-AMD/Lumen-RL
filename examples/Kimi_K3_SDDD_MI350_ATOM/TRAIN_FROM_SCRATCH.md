# Kimi-K3 DSpark Draft —— 从零一次训成（单数据集 × 3 epoch，32768 窗口）

这份文档是自包含的。读它不需要任何其他文档，也不引用任何其他文档。它来自七轮
真实训练（gen-1 到 gen-7）的踩坑记录，重新组织成**一条单轮从零训练的路径**——
不是我们实际走的"先短窗口训一轮、再换长窗口续训两轮"，而是把那三轮的结论前置，
一个数据集、一个窗口、一次训完。

**凡是标了 `/!\` 的地方都是有来历的。** 那类错误的共同特征是：**不报错、不崩溃，
训练指标甚至很漂亮，跑完做 benchmark 才发现整轮白费。** 这套流程栽过至少六次：
RoPE 旋转约定、softmax 缺 YaRN 因子、导出张量名、eval 尺子换掉、预处理缓存键、
draft 权重导入路径。每一次的代价是几天机时。

---

## 0. 先把目标定义清楚

### 0.1 要训的东西

一个 **DSpark draft 模型**：5 层稠密 MLA，块内非因果注意力，一次前向猜 **7** 个
token，外加一个低秩 Markov 头和一个 confidence 头。**68 个张量，3,562,312,961 个
参数（bf16，7.12 GB），其中可训练 2,387,907,841**（差额是冻结的 `embed_tokens`）。

它服务的 target 是 **Kimi-K3**：93 层 MoE，hidden 7168，词表 163840，mxfp4 权重
**1.56 TB**。draft 由 **ATOM** 推理引擎加载，`--method dspark`，target 批量校验。

### 0.2 衡量指标

```
前向步数     = Σ 次数                       （接受长度直方图 {k: 次数}）
已接受 token = Σ k × 次数
tok/fwd     = 1 + 已接受 token / 前向步数
接受率      = 已接受 token / (前向步数 × num_spec)     # num_spec = 7
```

那个 `1` 是 target 每步必出的 bonus token，所以接受率为 0 时 `tok/fwd` 也是 1.0，
猜 7 个时上限是 8。这个量和官方 model card 上的 "acceptance length" 同尺度。

### 0.3 /!\ "达到 model card 的效果"必须重新定义

官方 card 上的数字（GSM8K 5.64、HumanEval 5.34、MT-Bench 3.14、AIME 2.72，
14 集均值 3.85）是在 **vLLM + GB300** 上测的。**把你在 ATOM 上的读数去除这些值是
无效的。**

我们把官方发布的那个 draft（`Inferact/Kimi-K3-DSpark`）下载下来，在**同一台机器、
同一个镜像、同一套协议、同样的 prompt 数**上重测，得到的偏差是：

| 数据集 | 官方 card（vLLM） | 同机 ATOM 实测（同一个官方 draft） | 偏差 |
|---|---|---|---|
| AIME 2026 | 2.72 | 2.7620 | **+1.5%** |
| MT-Bench | 3.14 | 3.1022 | −1.2% |
| GSM8K | 5.64 | 4.9022 | **−13.1%** |
| HumanEval | 5.34 | 4.0701 | **−23.8%** |

**偏差从 +1.5% 到 −23.8%，和数据集强相关，不存在可外推的统一系数。** 早期有过
"ATOM 读数系统性低约 10%"的判断，来自 12 条 prompt 的抽样，后来被全量测量推翻。

**所以唯一有意义的判据是：把官方 draft 在你自己的机器上、用你自己的协议测一遍，
拿那个数当分母。** 本文后面所有"官方"都指这个同机读数。

### 0.4 这条路径的预期落点

我们实际走的三轮（短窗口 1 epoch → 长窗口续训 2 轮）最终的 13 集成绩：

| | 13 集均值 | 对同机官方 | 长上下文那一集 |
|---|---|---|---|
| 只训短窗口（8192），语料 74.8% 是多语种 | 3.3162 | 91.1% | **41.7%** |
| 换 32768 + 配平语料后 | **3.8218** | **105.0%** | **104.5%** |

**12 集超过官方，13 集全部提升。** 本文给的配方就是把第二行的条件（32768 窗口 +
配平语料）从第一步就用上，用 3 个 epoch 一次到位。

/!\ **诚实说明：我们没有真正跑过一次"从零单轮三 epoch"。** 下面的步数预算是从
累计轨迹推出来的（我们总共走了 49,026 步到 105%），配比和窗口是实测有效的，但
"3 个 epoch 一定够"这句话没有被单独验证过。第 8.4 节给了判断是否走偏的检查点。

---

## 1. 硬件与集群

### 1.1 拓扑：五个节点，不能少

| 角色 | 数量 | 干什么 |
|---|---|---|
| **teacher** | **4** | 每台一个 ATOM 实例，`tensor_parallel_size=8`，对数据集里已有的回答做 prefill，交出中间层 hidden state |
| **draft** | **1** | `torchrun --nproc-per-node=8`，FSDP2 训 draft |

teacher 要 4 个是因为单个 teacher 的 prefill 吞吐跟不上 draft 的训练速度。实测一步
里 teacher 占 2.7 s、draft 训练占 5.1 s，四个 teacher 轮转 + 预取才能把 draft 喂饱。

### 1.2 每台机器的硬性要求

| 项 | 要求 | 说明 |
|---|---|---|
| GPU | **8 张，一张都不能少** | `tensor_parallel_size: 8` 是硬约束 |
| 型号 | gfx950（MI355X / MI350X） | |
| 单卡 VRAM | **≥ 252 GiB** | K3 权重每卡约 **217.6 GiB**。MI355X 是 288 GiB，MI350X 是 251.98 GiB —— /!\ **不要假设两台机器规格一样**，换机器先重新核一遍 |
| 本地盘 | **≥ 2 TB 真实磁盘** | K3 权重 1.56 TB + checkpoint |
| `/dev/shm` | **≥ 1.2 TB** | tmpfs |
| 主机内存 | **≥ 1 TB**，建议 2.8 TB | Mooncake 段池 256 GB/副本 |
| 共享存储 | 数据集 + checkpoint + 日志 | checkpoint 每份 33 GB |

**开工前必须查的两件事：**

```bash
rocm-smi --showproductname                       # 必须列出 8 张
rocm-smi --showmeminfo vram | grep -i used       # 必须接近 0
```

/!\ **任何一张卡上有别人的进程，这套配方跑不起来**（teacher 要 217.6 GiB/卡）。
"抢到节点"不等于"能用"：调度器给你独占节点，**不会驱逐别人已经在跑的容器**，
症状是 `SHUTDOWN signal from DP rank 0 during initialization`。

**关掉 NUMA auto-balancing（主机级，容器里改不了）：**

```bash
sudo sh -c 'echo 0 > /proc/sys/kernel/numa_balancing'
```

不关的话 aiter 每次 import 都告警，AMD 文档说 MI300 系列会因此出错。

### 1.3 网络

hidden state 走 **Mooncake，协议必须是 TCP，不要 RDMA**。

/!\ 走 RDMA 的失败是 `Failed to register memory: Invalid argument [22]` 或
`Mooncake setup failed error=-600`。根因：同一张网卡上开第二个 RDMA client 就会
撞，而单批 hidden state 约 20 GiB，远超 7×2 GiB 的 RDMA 注册上限。TCP 实测带宽
5.27 GiB/s，够用。

### 1.4 /!\ 计算节点会以"看起来健康"的方式坏掉

这条在长跑里一定会遇到。三种形态：

1. **容器显示 "Up" 但引擎已死。** 实测一夜里两台机器：一台 18:54 引擎就停了
   （GPU 掉到 0%），另一台 23:28 卡死但 **GPU 还是 100% 在空转**（集合通信自旋
   死锁），两个容器都一直显示 "Up"。**判活性要看输出文件的 mtime，不要看
   `docker ps`。**
2. **`/health` 返回 200 但请求全挂。** API 前端和 engine core 是两个进程。
3. **一个 teacher 副本先慢下来，几小时后才彻底不响应。** 见 9.3，这是最贵的一种。

**另外两件会让"重启"失效的事：**

/!\ **worker 进程会活得比父进程久。** 杀掉训练容器之后 GPU 上还挂着孤儿进程，下一次
启动就会撞到"显存被占"。启动脚本要带 EXIT trap：

```bash
trap 'pkill -f AsyncLLMEngine; pkill -f EngineCore; pkill -f atom_teacher; pkill -f mooncake_master' EXIT
```

/!\ **调度器的 epilog 会直接删掉容器，那不是 crash，所以 `--restart=on-failure` 不
触发。** 节点被标记故障、作业被回收时，容器和 `/dev/shm` 一起消失。要在登录节点上
另起一个看守：容器不在 `running`/`restarting`/`created` 里就带 `resume` 重新起，
并且**先检查最新 checkpoint 是不是已经到了目标步数**——否则会把一个已经跑完的 run
重新拉起来。

---

## 2. 镜像：这一节跳过去，后面全是白费

### 2.1 基础镜像必须按 digest 钉死

```
rocm/atom-dev@sha256:2f8bd4206ad15d014ae48115eae1ee9f1db83781848a8542de7177cfbd4ac914
```

（这个 digest 当时对应的滚动 tag 是 `nightly_202608141640`。）

/!\ **`rocm/atom-dev:latest` 漂过至少两次。** 一次漂到 `atom dev323 + torch 2.10`，
症状是 **teacher 在第二次 hidden state 抽取时死锁**——第一次成功，第二次 600 秒
超时，零训练步。另一次漂掉了 `configure_hidden_states()` 的 `capture_mode` 参数，
build 期断言能拦住，但拦住之后要改代码。

**拉下来之后核四个版本，对不上就停：**

| 组件 | 必须是 |
|---|---|
| atom | `0.1.6rc1.dev275+g83e71001e` |
| torch | `2.13.0+rocm7.14.0` |
| HIP | `7.14.60850` |
| python | 3.12 |

```bash
docker run --rm <image> python3 -c "
import atom, torch
print('atom', atom.__version__)
print('torch', torch.__version__, 'hip', torch.version.hip)"
```

### 2.2 训练镜像

在基础镜像上装训练依赖。四个坑：

| 坑 | 后果 | 怎么办 |
|---|---|---|
| pip 不带 `--no-deps` | 自由解析 `accelerate` 会"升级"到 **torch 2.13 的 CUDA 版**，把 ROCm torch 覆盖掉；或者 pin torch 之后 `resolution-too-deep` | 所有训练依赖都用 `--no-deps` 装 |
| `antlr4-python3-runtime` 装到 4.13 | omegaconf 2.3 的 parser 反序列化失败 | 钉在 **4.9.3** |
| 漏装 `opentelemetry` | wandb 0.28+ 在 **`import wandb` 阶段就炸**，和 `WANDB_MODE=disabled` 无关 | 显式装 opentelemetry |
| 装了 vLLM | 设计上禁止。K3 在 vLLM 下 decode（B=64）**每 15–20 个 batch 挂一次 GPU**，比一轮工作的 MTBF 还短 | build 期断言 `importlib.util.find_spec("vllm") is None` |

**build 期还要断言 ATOM 的四个 API 仍然在**（漂了就在 build 挂，别漏到运行期——
teacher 要加载 20 分钟才会暴露）：

- `configure_hidden_states(self, aux_layer_ids, mooncake_config)` 恰好是这个签名
- `_store_hidden_states` 存在
- `run_model` 存在
- `calculate_eagle3_buffer_size` 仍接受 `num_aux_layers`

### 2.3 /!\ 启动容器前要清 aiter 的 JIT 锁

aiter 会在首次用到某个 kernel 时 JIT 编译，锁文件落在 `/tmp`。上一次运行留下的
陈旧锁会让**首次编译直接死锁**——没有报错，就是不往下走。

```bash
rm -rf /tmp/*aiter* /tmp/*hiprtc* 2>/dev/null || true
```

把这条放进启动脚本，每次起容器前无条件跑。

### 2.4 五台机器必须跑同一份镜像

一台 build 完之后 `docker save` / `docker load` 分发，**不要每台各自 build**——
基础镜像可能在两次 pull 之间漂了。

### 2.5 /!\ 一个不能漏的环境变量

**`PYTORCH_CUDA_ALLOC_CONF` 必须是空的。**

设成 `expandable_segments:True` 的后果：ROCm 上 segment 不被 `empty_cache()`
unmap，驱动把整个虚拟预留都算成 used。轮次切换时可用显存从 **265 GiB 掉到
175 GiB**，teacher 要 253 GiB，于是**每次轮次切换都失败**，伴随
`AttributeError: 'LumenRLModelRunner' object has no attribute 'model'`
（实测 6 小时内 20 次）。

/!\ **删掉 export 语句不够**——启动脚本会从调用方 shell 继承。启动后从容器里读回来
确认是空的：

```bash
docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' <container> | grep ALLOC
```

训练器里也加了 `_assert_allocator_returns_memory()`，检测到就拒绝启动。

---

## 3. 七个必须一次做对的正确性决定

这一节是全文密度最高的地方。**每一条的错误形态都是"训练侧一切正常、服务端接受率
垮掉"**，而且只有跑完 benchmark 才会发现。

### 3.1 RoPE 必须是 interleaved 旋转（而且这个参数名的语义是反的）

Kimi 的 MLA 用 **interleaved pairs**（GPT-J / DeepSeek 风格）：

```python
def _rotate_half_interleaved(x):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)
```

**不是** NeoX 的 half-split（`torch.chunk(x, 2, dim=-1)`）。

/!\ **ATOM config 里的 `rope_interleave` 语义和名字相反**：

```python
# atom/models/kimi_k3_dspark.py
is_neox_style=bool(getattr(config, "rope_interleave", False)),
```

| `rope_interleave` | 实际旋转 |
|---|---|
| 缺省 / false | **interleaved（正确）** |
| true | half-split（错误） |

**导出的 `config.json` 里不得出现 `rope_interleave`。** 写上就等于告诉 ATOM 用
half-split。

如果 cos/sin cache 是 NeoX 布局（大多数实现都是），运行时要转换：

```python
half = cos_pos.shape[-1] // 2
cos_pos = cos_pos[..., :half].repeat_interleave(2, dim=-1)
sin_pos = sin_pos[..., :half].repeat_interleave(2, dim=-1)
return (x * cos_pos) + (_rotate_half_interleaved(x) * sin_pos)
```

**错了的代价**：训练全程无报错、loss 正常下降，服务端接受率 **6.01%**，而官方
draft 在同机是 26.13%。

### 3.2 softmax scale 必须带 YaRN 的 mscale²

YaRN 不只改位置插值，它还改 attention 的 softmax 温度。漏掉这个因子，logits 会
**冷 1.8133 倍**。

```python
def yarn_get_mscale(scale=1.0, mscale=1.0):
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0

softmax_scale = 1.0 / math.sqrt(self.qk_head_dim)
if rope_cfg.get("rope_type", "yarn") == "yarn":
    ms = yarn_get_mscale(float(rope_cfg["factor"]),
                         float(rope_cfg["mscale_all_dim"]))
    softmax_scale *= ms * ms            # /!\ 平方
```

发布配置下的具体数值（`factor=32`、`mscale_all_dim=1.0`、
`qk_head_dim = qk_nope_head_dim + qk_rope_head_dim = 128 + 64 = 192`）：

```
yarn_get_mscale(32, 1.0) = 0.1 × ln(32) + 1 = 1.34657
mscale²                                     = 1.8133
不带 mscale：1 / √192                        = 0.07216103
正确值：    1.8133 / √192                    = 0.13086080    ← 两边必须都是这个
```

/!\ **cos/sin 的幅度不是 bug 来源**：`cache_amp = mscale(f, mscale) /
mscale(f, mscale_all_dim)`，当两者都是 1.0 时等于 1.0。所以只对比 rotary 的测试
会通过，漏的是 softmax 那一路。

### 3.3 aux hidden state 取哪五层

```
aux_hidden_state_layer_ids: [2, 23, 47, 71, 89]
```

| 项 | 值 |
|---|---|
| 索引基准 | **0-based**，取值范围 0..92 |
| 语义 | **第 i 层的输出**（post-layer），HF 的 `hidden_states[i+1]` |
| 是否有 ±1 偏移 | **ATOM / 本流程没有**（vLLM 路径的 "+1" 说明不适用） |
| capture_mode | **`postnorm`**（hook 抓 `hidden_states + residual`） |
| 拼接宽度 | **35840 = 5 × 7168** |

/!\ 不要抄别处的 `[7, 31, 47, 63, 87]`——那是另一个 stage1 示例。**这五层是 draft
的特征契约，teacher 必须抓完全相同的五层。**

每 token 的缓存开销：`(5 aux + 1 token_embeds + 1 last_hidden) × 7168 × 2 B = 98 KB`。

### 3.4 `separate_last_hidden` 在 ATOM 上必须是 `false`（和 vLLM 相反）

| 侧 | 值 | 语义 |
|---|---|---|
| **ATOM** | **`false`** | `hidden_states` 已含全部 5 层（35840）；`last_hidden_states` 是模型最终输出、**已经过 final norm** |
| vLLM 变体 | `true` | 前 4 层进 `hidden_states`，第 5 层单独进 `last_hidden_states`（pre-norm），trainer 要自己 rejoin 并补 RMSNorm |

设成 `true` 的症状：`context_proj`（内部叫 `fc`）的输入维变成
**43008 = 6 × 7168**（应该是 35840），而且 RMSNorm 被**应用两次**。

### 3.5 off-policy prefill，不是 generate

`teacher.generate_mode: prefill`。

做法是：**先用 K3 把数据集里所有回答重新生成一遍存下来**（第 6 节），训练时
teacher 只对这些已知 token 做 prefill、交出 hidden state，**不在训练环里解码**。
draft 学的是"给定 target 在每个位置的 hidden state，预测接下来 7 个 token"。

这不只是省时间。在训练环里 decode 的路径（gen-1 走过）在 K3 上每 15–20 个 batch
就挂一次 GPU。

三个配套的引擎约束，任何一条错了 hidden state 就是错的标签：

| 参数 | 必须 | 错了会怎样 |
|---|---|---|
| `enable_prefix_caching` | **false** | prefix cache 命中不计入 `scheduled_tokens`，**prompt 段的 hidden state 整段缺失** |
| `enable_chunked_prefill` | **false** | 同一个 key 被多个 chunk 覆写，**只有序列尾部的 hidden state 存活** |
| `max_num_batched_tokens` | **≥ `max_model_len`** | prefill 被分步，同 key 覆写 |

### 3.6 teacher 的 KV 必须和服务时一样是 fp8

```yaml
kv_cache_dtype: fp8
index_cache_dtype: fp8
```

服务时用 fp8 KV。prefill 时 k/v 是从 paged cache 读回来的，所以**bf16 训出来的标签
配不上 fp8 服务时的特征**。ATOM 的默认是 bf16，必须显式设。

/!\ **ATOM 对不认识的 kwargs 静默丢弃，所以不能只看 yaml。** 去 worker 日志里确认
出现了 `kv_cache_dtype': 'fp8'`。

（K3 的 93 层里只有 24 层用 paged KV，其余 69 层是 KDA recurrent state。）

### 3.7 draft 的注意力是非因果的

| 项 | 值 |
|---|---|
| SDPA | **`is_causal=False`**，用显式 boolean mask |
| 块内 | **双向** |
| 块间 | 互不可见 |
| context 可见性 | 每个 block 只看得到 `kv_idx < anchor_pos` 的 context |
| KV 布局 | `[Context (ctx_len) | Draft blocks (num_anchors × block_size)]` |
| block_size / spec_length | **7** |
| anchor_num | **512** |
| 无效 block（`keep=False`） | 输出必须是 **0.0**，不是 NaN |

`anchor_num: 512` 是峰值显存的主要来源（实测 57.8 GiB）。

**用 flex_attention + BlockMask，不要用 dense mask + SDPA**：实测 **914.6× 提速、
显存 1/8.8**。两个坑：

- /!\ flex 在 eager 下**更差**（33.7 GiB vs SDPA 24.6 GiB），必须 `torch.compile`
- /!\ `torch.compile(..., dynamic=False)` 会撤销这个优化：KV_LEN 每步都变，
  dynamo 缓存被撑爆、永久退回 eager，日志里是每步 1000+ ms 的重编译
  （`varying KV_LEN: 8165:5ms, 8128:1167ms, ...`）。**不要传 `dynamic`**，让它自动。

### 3.8 上机自检：开训前必须全绿

这三个检查加起来不到半小时，挡住的错误值好几天。

**(a) RoPE + softmax 对齐**

```
ATOM/LumenRL softmax scale: 0.1308608000 (1.8133x plain)
rotation vs ATOM published (interleaved) rel L2 = 3.127e-08   MATCH
rotation vs ATOM rope_interleave=true (half-split) rel L2 = 9.874e-01   differs
PASS
```

断言：softmax scale 差 ≤ 1e-9；对 interleaved 的相对 L2 < 2e-3（实测 **1e-8 量级**）；
**并且对 half-split 必须不匹配**（只测第一条会漏掉两边同时错的情况）。

/!\ 构造 ATOM 的 rope 要包在 `with torch.device("cuda")` 里，CPU default device
下建不起来。

**(b) flex attention 与 dense mask 等价**

```
rel L2: 3.081e-03        # bf16 累积噪声，容差 2e-2
speedup ~914.6x, memory ~8.8x smaller
dropped block rows are exactly 0.0
PASS
```

**(c) hidden state 契约**

teacher 交出来的 `hidden_states` 宽度必须是 **35840**，`last_hidden_states` 单独
一路且已 norm，每条请求是 ragged 的 `T_i`（读的时候右 pad 到 batch 的 T）。

---

## 4. Kimi-K3 权重

| 项 | 值 |
|---|---|
| repo | `moonshotai/Kimi-K3` |
| revision | **钉住**（我们用的是 `a590ce090cb049c93a33dfe8c208ec652aa20503`） |
| safetensors 分片数 | **96** |
| 总字节 | **1,560,860,324,864** |
| 落盘位置 | **每台机器的本地 NVMe**，不要放 tmpfs（会吃光内存） |

下载完必须核 `config.json`：

```
model_type            = kimi_k3
architectures         = [KimiK3ForConditionalGeneration]
num_hidden_layers     = 93          # aux 层 [2,23,47,71,89] 必须落在 0..92
hidden_size           = 7168
vocab_size            = 163840
total_size            = 1560860324864
```

还要有 `configuration_kimi_k3.py` / `encoding_k3.py`（`auto_map` 会引用）。

/!\ **权重不随节点分配迁移。** 如果本地盘是节点私有的，换节点就要重下 1.56 TB
（约 15 分钟）；被别人重启过的节点本地盘也会被清空。

---

## 5. 数据：怎么选，配多少

这一节是本文相对原始记录**改动最大**的地方。我们实际走的路是先用一份 74.8% 是
多语种的 500 万行语料训了 39,148 步，再换配比续训——**那 39,148 步里有一大半是
浪费的**。单轮训练不要重复这个错误。

### 5.1 /!\ 最重要的一条结论：配比比体量重要得多

两组实测对照：

| | 语料 | 步数 | 窗口 | 13 集对官方 |
|---|---|---|---|---|
| A | **5,011,091 行**，多语种 74.8% | 39,148 | 8192 | **91.1%** |
| B | A 训完之后，换 **1,264,631 行**配平语料续训 | +9,878 | 32768 | **105.0%** |

B 只用了 A 的 **1/4 语料、1/4 步数**，却把成绩从 91.1% 拉到 105.0%。

更直接的证据：**把多语种占比从 74.8% 砍到 12.3%，多语种那一集的成绩反而从官方的
102.8% 涨到 107.9%。** 一个 5 层的 draft 学的是"下一个 token 在局部上下文里怎么
接"，这件事跨语言共享的程度远高于直觉；把同一种能力重复喂 75% 的配额，边际收益
早就归零了。

腾出来的配额给了代码（0.43% → 20%），`swebench_pro` 从 84.9% 涨到 100.6%——
13 集里相对提升最大的一集。

### 5.2 推荐配比

目标 **约 210 万行**（理由见 5.4 的步数预算）。按下面的份额配：

| 类别 | 份额 | 210 万行对应 | 来源 |
|---|---|---|---|
| **代码** | **20%** | 420,000 | `nvidia/OpenCodeInstruct` 为主（18%）+ Nemotron 的 code split（2%，去重后物理上限只有 2.3 万） |
| chat | 23% | 483,000 | `nvidia/Nemotron-Post-Training-Dataset-v2` chat |
| STEM | 23% | 483,000 | 同上 stem |
| math | 9% | 189,000 | 同上 math（**要长思维链的**） |
| 多轮 / 推理痕迹 | 10% | 210,000 | `lightseekorg/kimi-mtp-dataset` 一类 |
| **多语种** | **12%** | 252,000 | Nemotron multilingual_{ja,de,it,es,fr} + `CohereLabs/aya_dataset` |
| 其他小类 | 3% | 63,000 | 创意写作、工具调用等 |

**代码 20% 和多语种 12% 这两个数是实测有效的，其余几类的边界没有单独验证过**，
按上表附近浮动都可以。

### 5.3 长度画像：必须有真正的长样本

光把窗口设成 32768 没用——如果语料里没有长行，draft 在 8192 以上的位置**从来没做过
一次预测**。这就是 A 组长上下文那一集只有 41.7% 的原因：8192 窗口下
`drop_overlong` 丢了 34.7 万行，**存活行的 max 是 8,191**。

B 组（105% 那一份）的实测长度画像，**照着这个配**：

| 指标 | 值 |
|---|---|
| mean | 2,643 |
| p50 | 1,342 |
| p90 | 6,328 |
| p99 | 18,923 |
| max | 32,430 |
| **> 4096** | **16.2%** |
| **> 8192** | **7.1%** |
| **> 16384** | **1.4%** |

长样本从两处来，**第二处便宜十倍**：

1. **被窗口截断过的行，用大窗口重新生成。** 判据不是"哪些样本长"，而是
   **哪些样本的 `finish_reason == max_tokens`**——那是模型想写更多但被截断的行。
2. **长 prompt 的数据集**（我们用 `nvidia/OpenCodeInstruct`，18% 的配额）。
   /!\ **长 prompt 是便宜十倍的长上下文数据**：prompt 段只需要 prefill 一次，而长
   回答段的每个 token 都要生成。同样拿到一个 8192 以上的位置，长 prompt 的行在
   生成阶段的成本大约是长回答行的 1/10。

### 5.4 步数预算与 lr

| | 值 |
|---|---|
| 语料 | 约 2,100,000 行 |
| 存活率（32768 口径） | 约 **99.99%**（实测 2,468,462 → 2,468,457，只因 5 行含超出 embedding 表的 token id 被丢） |
| 一个 epoch | `floor((存活行 − 128) / 128)` ≈ **16,400 步** |
| **3 个 epoch** | **≈ 49,200 步** |
| step 时间（32768，实测） | **mean 8.9 s，p50 8.7 s** |
| **总墙钟** | **≈ 122 小时 ≈ 5.1 天**（不含故障重启） |

49,200 步这个数是对着我们累计走到 105% 的 **49,026 步**取的。配比和窗口从第一步
就对，所以这个预算应该有余量，但没被单独验证过。

/!\ **`num_training_steps` 必须实测，不能估。** 用 CPU 预处理脚本在
**和训练完全相同的参数**下跑一遍，报出存活行数再算。这个数还负责把采样器挡在
eval 尾部之外，给大了就会**训到自己的 eval 切片上**——症状是 eval 指标某一步突然
变好，这个项目栽过一次。

### 5.5 去重：入库即去重，不要先采样后去重

哈希键 = NFKC 正规化 + 小写 + 折叠空白后的 prompt，**跨所有 split 共享一个
`seen` 集合**。

实测留存率（总 654 万行 → 唯一 prompt 555 万，84.8%）：

| split | 源行数 | 唯一 prompt | 留存 |
|---|---|---|---|
| chat | 627,720 | 439,845 | 70.1% |
| **code** | 175,000 | **31,261** | **17.9%** |
| math | 239,467 | 101,962 | 42.6% |
| stem | 355,000 | 354,758 | 99.9% |
| aya | 202,362 | 165,863 | 82.0% |
| multilingual ×5 | 4,944,227 | 4,453,283 | 90.1% |

/!\ **code 的 17.9% 是物理上限**：17.5 万行里只有 31,261 个唯一 prompt（平均一题
5.6 个回答）。这就是为什么代码配额要靠 `OpenCodeInstruct` 补，靠 Nemotron 的 code
split 凑不出 20%。

/!\ **顺序敏感**：先处理英文四类，再 aya，最后 multilingual——先处理的占住哈希。
反过来会让稀缺类别被多语种挤掉。曾经"先采样再去重"，code 期望 5 万只得到 27,236。

### 5.6 去评测集污染：13-gram

对所有要测的 benchmark 的 prompt 建 **13-gram** 索引，命中就剔除。实测剔掉约 1.4 万条。

n=13 是折中：更短会误伤（代码片段容易撞车），更长会漏（改写过的题目）。

### 5.7 /!\ 图文样本必须过滤，但只用这四条判据

多模态样本的回答是 K3 **在没有收到图片的情况下编出来的**，训进去 draft 会学着
模仿编造的视觉断言。某个常用数据集有 **27.3%** 是 COCO 图文行。

**只用这四条：**

| 判据 | 规则 |
|---|---|
| content 不是字符串 | `type(content) is not str` |
| 图文 JSON 外壳 | 正则 `^\s*\[\s*\{\s*"type"\s*:` |
| 模型的图片 token | `<\|image\|>` / `<\|media_begin\|>` / `<\|vision_start\|>` |
| 正文占位符 | 含 `<image>` |

/!\ **不要把 `<img`、`image_url`、`data:image/` 加进黑名单。** 试过，误伤 **2.55%**：
那些命中大多是纯文本编程题里的 HTML 标签、API 字段名、base64 示例。

---

## 6. 用 K3 重新生成所有回答

数据集里原本的回答不能直接用——draft 要学的是 **K3 的**分布。

### 6.1 生成参数

| 参数 | 值 | 理由 |
|---|---|---|
| `max_model_len` | **32768** | 和训练窗口一致 |
| `--max-prompt-tokens` | **32224** | `= 32768 − 512 − 32`，给回答和特殊 token 留位置 |
| `max_num_seqs` | **384** | /!\ 不是 512：512 时有效并发只有 279，384 时是 384 |
| `kv_cache_dtype` | **fp8** | 和训练/服务一致 |
| `temperature` | **0** | 贪心，可复现 |
| speculative_config | **不传** | 见下 |
| `thinking` | **true** | K3 是保留思维链的模型 |

/!\ **`--max-prompt-tokens` 的默认值会让作业启动即退。** 生成脚本的默认 14304 是
给 16384 窗口的；任何 prompt 超过它就 `SystemExit`，报
`N 行 prompt 超过 max_prompt_tokens`。32768 窗口必须显式传 **32224**。

/!\ **不要为了"加速生成"打开官方 draft。** 实测不成立，而且原因可量化：

```
Per-request cache tensor (173.44GB for 3072 slots) exceeds available KV budget (41.23GB)
```

3072 slots = 384 序列 × (7 投机 token + 1)。单卡被 K3 权重吃完只剩约 41 GB 给 KV，
投机验证要 173 GB。要塞下必须把并发从 384 砍到约 91：

- 不开 draft：382 序列 × 1 token/步 = **382**
- 开 draft：91 × 4.33 tok/fwd ≈ **394**

产出持平，而开 draft 每步还多做 7 次 draft 前向。**投机解码换的是延迟，生成是吞吐
场景，382 条并发已经把机器喂饱，没有空闲算力可换。**

### 6.2 运维（几十到几百个节点小时，跨好几个夜晚）

| 坑 | 症状 | 处理 |
|---|---|---|
| **引擎卡死但容器 "Up"** | `engine.step()` 阻塞不返回，`docker ps` 正常，8 小时白转 | 独立线程 watchdog 按**输出文件 mtime** 判活，超时 `os._exit`；外面再套 supervisor |
| **`kill` 对阻塞在长 exec 里的循环无效** | 进程还在跑旧的内存里的函数定义 | `kill -9` 并**回查确认**，否则会有两个编排同时跑 |
| **显存释放有竞态** | "不忙"判断通过，但铺机时显存还没释放 | 判定后隔 45 秒**再查一次** |
| **自己的容器被自己的预检当成别人的** | `显存已被占用 2196GB（别人的容器）` | 容器名前缀白名单要**包含自己所有阶段**的名字 |
| **有的分片反复重启、始终 0 产出** | 根因未定位（疑似大批 prefill 触发 aiter/RCCL） | 见 9.3，**先降 `max_num_seqs`**，再考虑换种子；连续失败 N 次就跳过这个单元，别占着机器 |

---

## 7. 转训练集：重点是校验，不是转换

### 7.1 逐 token 回环校验

把生成结果转成训练集时，**对每一行做 token 级回环校验**：重新渲染 + 重新 tokenize，
和生成时的 token 序列逐位对比。

实测丢弃率约 **8.8%**（399,842 → 364,767），分布：

| 原因 | 占比 | 是不是 bug |
|---|---|---|
| `no_think_close` | 6.14% | 不是，思维链没闭合 |
| `response_mismatch` | 2.21% | **不是 bug**：BPE 贪心分词和逐 token 生成路径本来就可能不一致（`FORMCHECK` → `['FORM','CHECK']` vs `['FOR','MC','HECK']`）。丢掉就好 |
| `no_response_close` | 0.32% | 不是 |
| 其余 | 0.10% | |

/!\ **丢弃率超过阈值就停下来查，不要继续。** 有一次转换丢了 **99.59%**
（`prompt_len_drift` 94.70%），根因是传错了 prompt 源文件——选择脚本保留的是原始
文件的行号，而我传的是一个抽取过的子集。**范围检查通过不等于对齐**：我当时查了
"最大索引 999,999 < 100 万行"就放行了，那是巧合。

### 7.2 训练集的两个口径

| 参数 | 值 | 说明 |
|---|---|---|
| `drop_overlong` | **true** | |
| `max_prompt_tokens` | **0** | |

/!\ 这两个合起来的语义是：**超长的行被整行丢弃，而不是被截断**。代码里是
`check_total = drop_overlong and max_prompt_tokens <= 0`。这是唯一端到端验证过的
路径，而且会跟着窗口变。**不要训截断的回答。**

### 7.3 /!\ 500 万行会在启动时把节点内存打爆

`input_ids` 如果存成 `list[int]`，每个 token 约 **36 字节**。改成 **int32 ndarray**
（4 字节/token）。

### 7.4 预处理缓存的键包含每一个参数

先用 CPU 把预处理跑完、预热缓存，否则 teacher 已经占住 GPU 之后 trainer 才开始
付这笔钱。

/!\ **预处理脚本的参数必须和训练 yaml 逐个对齐**，差一个就是第二份缓存——你预热的
那份 trainer 会直接丢掉。`num_workers` 是唯一不进缓存键的参数。

---

## 8. 训练配置

### 8.1 完整的关键值

```yaml
cluster:
  gpus_per_node: 8
  num_nodes: 1                 # launcher 会把五节点折叠成 Ray 集群

policy:
  max_total_sequence_length: 32768
  train_global_batch_size: 128
  train_micro_batch_size: 1
  learning_rate: 5.0e-5        # 峰值
  weight_decay: 0.0
  max_grad_norm: 1.0
  warmup_ratio: 0.04
  lr_decay_style: cosine
  min_lr: 2.0e-6               # /!\ 不是 0，见 8.2
  training_backend: fsdp2

algorithm:
  name: spec_distill
  spec_distill:
    draft_type: dspark
    loss_type: dspark
    sequential_mode: streaming_disaggregated
    teacher_replicas: 4
    stream_prefetch_batches: 8
    spec_length: 7
    anchor_num: 512
    capture_mode: postnorm
    separate_last_hidden: false
    aux_hidden_state_layer_ids: [2, 23, 47, 71, 89]
    num_target_layers: 5
    loss_decay_gamma: 4.0
    ce_loss_alpha: 0.1
    l1_loss_alpha: 0.9
    confidence_loss_alpha: 1.0
  teacher:
    generate_mode: prefill
    atom:
      max_model_len: 32768
      max_num_batched_tokens: 32768   # /!\ 必须 >= max_model_len
      max_num_seqs: 24                # /!\ 见 9.3
      enable_prefix_caching: false
      enable_chunked_prefill: false
      gpu_memory_utilization: 0.90
      kv_cache_dtype: fp8
      index_cache_dtype: fp8
      enforce_eager: true

mooncake:
  protocol: tcp                # /!\ 不要 RDMA
  global_segment_size: 256GB   # /!\ 见 8.3
  local_buffer_size: 8GB

dataset:
  drop_overlong: true
  max_prompt_tokens: 0
  thinking: true
  last_turn_loss_only: "true"
  min_loss_tokens: 14
  num_preprocess_workers: 120

eval:
  enabled: true
  interval: 50
  num_samples: 128
  micro_batch_size: 2

checkpointing:
  save_steps: 500
  save_total_limit: 90
  resume: false                # 从零训就是 false

num_training_steps: <实测>      # floor((存活行-128)/128) × 3
seed: 42
```

### 8.2 /!\ `min_lr` 绝不能是 0

**这是一次完整的训练事故。** 某一轮单 epoch 配了 `min_lr: 0`，余弦把 lr 带到
**7.6e-10**，eval AL 从 step 3,300 起在 **1.613–1.617** 平台化，斜率从
+0.197/100 步衰减到 **+0.001/100 步**。

当时的判断是"数据用尽"。**错了，那是 schedule 假平台。** 而且代价是双份的：
lr 已经归零，"接着再训一个 epoch"根本不成立，必须重开 schedule。

单轮三 epoch 的正确做法：**余弦跨完整 3 个 epoch，min_lr 取一个小但非零的值**。
`2.0e-6` 是 `5.0e-5` 的 1/25，既真正完成了退火，又给"以后想再补一轮"留了非零的
入场券。

/!\ 如果你只打算训 1 个 epoch、但**将来可能续训**，那 min_lr 要按"3 epoch 余弦跑到
第 1 epoch 末尾"的值取，而不是取小值：

```
decay_ratio = (1 − 0.04×3) / (3 − 0.04×3) = 0.3056
coeff       = 0.5 × (1 + cos(0.3056π)) = 0.7868
min_lr      = 0.7868 × 5e-5 = 3.93e-5
```

### 8.3 Mooncake 段大小要按批算，而且只能用环境变量给

`global_segment_size` 要装得下一个副本在飞的所有批。

```
一批 = 128 条 × 均长 token × 84 KiB
均长 2,643 时：128 × 2643 × 84 KiB ≈ 28 GiB
32768 全满的极端批：128 × 32768 × 84 KiB ≈ 344 GiB
```

8192 窗口下 128 GB 够用；**32768 窗口必须提到 256–384 GB**。主机内存 2.8 TB。

/!\ 段太小的症状是 **`code=-200`**：最老的 key 被 evict，consumer 等 90 秒后死。
**批大小一变就要重算段大小。**

/!\ **段大小只能从环境变量给。** launcher 会用
`mooncake.global_segment_size=${MOONCAKE_GLOBAL_SEGMENT_SIZE:-128GB}` 作为 runtime
override，**把 YAML 里的值覆盖掉**——只改 YAML 不生效。

```bash
export MOONCAKE_GLOBAL_SEGMENT_SIZE=256GB
export LUMENRL_TEACHER_MOONCAKE_SEGMENT_SIZE=256GB
export MOONCAKE_PROTOCOL=tcp
export MOONCAKE_DEVICE_NAME=          # /!\ 空值时不要下发这个 override，见 12 节
```

`local_buffer_size: 8GB` 不用改（32768 单序列是 2.7 GiB）。

Mooncake master 由 rank 0 起，端口在 **51000–52000** 和 **8100–9100** 两个区间里
自动挑——排查端口冲突时要连这两段一起看。它在 Phase 切换期间必须保持运行，即使
teacher engine 已经 tear down。

### 8.4 /!\ eval 必须用两把尺子

**这是用一整轮训练换来的教训。** 某一轮只留了"自己训练集尾部"这一把尺子，而那份
语料 74.8% 是多语种，于是**训练侧 eval AL 涨了 37%，而服务端 MT-Bench 只涨 4%**。
整轮训练没有任何信号能提示服务端的真实位置，直到跑完 benchmark 才知道。

```yaml
eval:
  num_samples: 128
  interval: 50
  extra_slices:
    - path: <固定切片>.pt
      prefix: eval_fixed
```

- `eval/*` —— 本轮训练集尾部 128 行，**只说明"学会了本轮这个分布"**
- `eval_fixed/*` —— 一份**固定不动**的 128 行，跨轮可比

/!\ **换了语料就等于换了基准，两轮之间的 eval AL 不可比。** 曾经有人因此把"数据
更多更好的一轮 AL 更低"当成退化去查——实际上更大更多样的数据集**本来就该读更低**。

固定切片怎么做：用**完全相同的预处理参数和 seed**，在**同一份数据集**上重跑一遍
预处理，取同样的尾部 128 行，写成文件。切片的指纹里要记 `dataset_rows_after_preprocess`，
和训练时的存活行数不一致就说明取的不是同一批行，尺子就白搭了。

**第三把尺子（可选但推荐）**：如果你中途调整过配比，再加一把盯**旧配比**的尺子。
它掉下去就是在拿旧能力换新能力。

### 8.5 判断是否走偏的检查点

/!\ **不要跨语料比 eval AL**（见 8.4）。下面这些数来自我们的短窗口从零训练轮，
语料是多语种重的，**你的配比更好但 eval 尾部更难，读数可能更低**——所以把它们当
"数量级参考"，不是"必须达到"。

| step | eval loss | eval AL | step_0_acc |
|---|---|---|---|
| 0 | 14.39 | 0 | — |
| 500 | 6.56 | 0.262 | — |
| 1,000 | 5.15 | 0.663 | — |
| 3,000 | 3.24 | 1.368 | — |
| **8,269** | 2.21 | **2.001** | — |
| 11,650 | 1.96 | 2.182 | — |
| 21,607 | 1.61 | **2.501** | — |
| 27,450 | 1.51 | 2.604 | — |
| **39,100** | **1.371** | **2.767** | 0.847 |

**曲线在整个 run 上是对数线性的，斜率约 +0.36 AL / 翻倍步数**（1,500→3,000 是
+0.443，8,250→11,650 是 +0.361，21,607→27,450 是 +0.299）。step 8,000–10,000
附近掉一档（0.45 → 0.36）之后就稳住。

/!\ **外推时用"每翻倍步数"的口径，不要用"每 100 步"**——后者会让人误判成持续
减速。曾经按"增益 ∝ step^−0.5 持续衰减"拟合，把 AL 2.0 推到 step 11,000–12,000，
实际是 8,269。

**真正可靠的检查点是"跑完第一个 epoch 就导出、benchmark 一次"。** 5 天的训练里
花 2 小时确认方向，比跑完才发现便宜得多。

### 8.6 /!\ `training_backend: fsdp2` 是骗人的，draft 实际不分片

配置里写的是 `fsdp2`，但训练器会自己判断能不能跳过分片：

```python
_SKIP_FSDP_MAX_GPU_GB = 80.0     # bf16 权重 + fp32 master + Adam m/v = 14 B/param

est_gpu_gb = num_params * 14.0 / (1024 ** 3)
if est_gpu_gb < 80.0 and not has_dropout:
    # 走 torch.distributed._composable.replicate，不是 FSDP2
```

draft 是 2,387,907,841 个可训练参数 × 14 B = **31.1 GiB < 80 GiB**，而且没有
Dropout，所以**永远走 `replicate`（DDP 式的梯度全归约），每个 rank 持有完整副本**。

开训后日志里应该看到的是这一行，而不是 FSDP 的分片信息：

```
Using composable replicate for gradient sync (2387907841 params)
```

**为什么要知道这件事**：DDP 的梯度桶约 **4.45 GiB**，它是 Phase 切换后 trainer 残留
显存（约 13.6 GiB）的最大一块，而那 13.6 GiB 会被 ATOM 重复计费进它自己的 KV
预算——就是排错表里 `InsufficientPoolBudget` 那一行的来源。把 draft 换大（超过
80 GiB）会自动切回 FSDP2，显存画像会完全不同。

### 8.7 吞吐参考

| | 8192 窗口 | **32768 窗口** |
|---|---|---|
| step_s mean | 6.13 | **8.9** |
| step_s p50 | 5.96 | **8.7** |
| teacher_s | — | **2.7** |
| train_s | — | **5.1** |
| 吞吐 | 41,950 tok/s | — |

checkpoint 每 500 步一次，每次约 **33 GB、55 秒**。`save_steps` 不要设 100：
39,000 步就是 390 次保存、6 小时纯写入、13 TB。

/!\ **`save_total_limit` 给大一点（我们用 90）。** 删 checkpoint 已经让这个项目
重算过一次——曲线上一段可疑的区间需要 step 2,400，而它已经被删了。

**已知的性能优化空间（未实现）**：32768 窗口下行均长 2,643、padding 到批内最长
（实测 `seq/max_len` 22,000–29,000），**padding 浪费约 90%**。按 ragged 取数、
不 pad/stack，估计能省 4 s/step；再加上后台预取与训练重叠，估计能到 5.5–6 s/step。
按长度分桶组批理论上也能大幅减少 padding，但**没有验证过**，而且会动采样器，要
小心和 eval 尾部的交互。

---

## 9. 启动与看守

### 9.1 五个 rank 必须并行起

```bash
for rank in 0 1 2 3 4; do
  <在第 rank 台机器上> \
    SLURM_PROCID=$rank SLURM_NTASKS=5 \
    setsid nohup bash run_multinode_rank.sh > rank-$rank.out 2>&1 &
done
wait
```

/!\ **顺序起会失败。** 在远程 exec 留下后台进程时，调用方约 100 秒才返回，顺序起
五个会让第一个和最后一个节点的协调文件差好几分钟，而默认的节点发现窗口只有
120 秒。**并行起，并且把发现窗口放宽到 600 秒**（`LUMENRL_NODE_DISCOVERY_TIMEOUT=600`）。

/!\ **判断启动成功要看协调目录里的文件，不要看 stdout。** 实测五个 rank 全起来了
但 stdout 只打了两行就被外层 timeout 掐掉。协调目录里应该出现
`node-0..4` + `ray-0..4` 共 10 个文件。

/!\ rank 0 之后还有一段**硬编码 240 秒**的"等 5 个活跃 Ray 节点"循环，没有开关
可调。

/!\ **rank 0 是 Ray head / driver，不一定是 draft 节点。** draft 由 Ray 放置。
看 draft 日志要找那个跑着 `torchrun --nproc-per-node=8` 的节点。

### 9.2 开训后立刻确认三件事

1. `Loaded model weights ... (67 keys, missing=[], unexpected=[])`
   —— /!\ **`missing=` 必须是空的。** 权重导入走 `strict=False`，漏了不报错。
2. `timing/fetch_wait_s` 接近 0 —— 说明预取和训练重叠生效了
3. worker 日志里有 `kv_cache_dtype': 'fp8'`

### 9.3 /!\ 最贵的故障：teacher 副本先慢几小时，再挂死

**这个故障在 gen-6 让七次重启全部失败、烧掉约 5 小时，在 gen-7 又烧掉约 4 小时。**

**终态**是这条：

```
TimeoutError: timed out waiting for response to cmd=extract_hidden after 600.0s
RuntimeError: remote ATOM teacher failed: ... AtomTeacherRayActor.process_prefill_batch()
```

一个 teacher 副本进了 `extract_hidden` 就不返回，trainer 的 600 秒超时到了就把整个
run 带走。

**但它不是突然挂的。** 实测的前兆形状（分副本 p50，近 1000 步）：

| | 退化前 | 退化后 |
|---|---|---|
| replica 0 | 2.90 s | 1.42 s |
| **replica 1** | 2.73 s | **47.85 s**（p50 49.46，max 79.77） |
| replica 2 | 3.00 s | 1.34 s |
| replica 3 | 3.09 s | 1.33 s |

**一个副本慢 35 倍，另外三个反而更快**（队列被拖空、没有竞争）。所以
`teacher_s` 的平均值只从 2.7 涨到 13.3，看着像"整体慢了一点"——**平均值会掩盖
这个故障，必须分副本看。** 它就这样又跑了七小时才彻底不响应。

**三条处理办法：**

1. **看守要分副本监控 `timing/teacher_s`。** 判据：某个副本的 p50 > 3 倍其余副本
   的中位数**且**绝对值超出 8 秒，连续 30 分钟就重启。这把"白烧七小时 + 丢掉整轮"
   压缩成 30 分钟内重启。
2. **`max_num_seqs` 设 24，不要 32。** 32 的时候某些 batch 打包会稳定触发这个挂死。
   降到 24 之后 gen-6 立刻过了那个卡点、又跑了 7,456 步到收敛。代价实测为零
   （teacher 单步 3.57 s vs 3.33 s，在噪声范围内）。
   /!\ **不要改 `max_num_batched_tokens` 去躲**——它必须 ≥ `max_model_len`，而
   `enable_chunked_prefill` 是 false，调小会让 30,720–32,768 长度的序列被静默截断。
3. **续训会确定性地重放同一个 batch。** 采样器是 `(step × bs) % _trainable_len`，
   副本分配也是 step 的确定函数。从同一个 checkpoint 续训就是在重放一模一样的
   batch 序列，于是**同一步反复失败时，原样重启一定还死在那里**——gen-6 在
   step 41,570 连死七次就是这么来的。**改的是 batch 打包方式，不是重试次数。**

**不是坏机器。** 同一条错误在 gen-6 出现在两台不同的机器上，gen-7 又换了第三台。

/!\ **一条被证伪的根因，写在这里免得别人再走一遍。** 当时注意到失败的 run 里有
16 条
`FlyDSL kernel ... is not recognized by the current catalog; falling back to next candidate`，
而手边那个"正常"的 run 是 0 条，就认定是 aiter 的调优表和内核目录不匹配。**错了。**
事后把全部 11 个 run 逐个数了一遍：**跑完的那个 run 有 64 条**（是失败 run 的四
倍），另有两个失败的 run 是 0 条。当时那个"0 条"的 run 只跑了几十步、还没攒出回退
而已。这条日志和 `bgemm_internal_cublaslt` 一样是噪音。

**教训："失败组有、成功组没有"这种证据，要在两组都跑够长之后再数。**

### 9.4 看守脚本要做的事

| 做什么 | 为什么 |
|---|---|
| **只读当前 run 的 step** | 跨 run 取最大值会让停滞判据永远成立。我的看守犯过这个错，连杀了三次健康的训练 |
| **五台机器一起数容器** | 只查一台会误判 |
| 停滞判据用**时间**（1800 秒），不是迭代次数 | eval 点会让单步看起来变慢 |
| 分副本监控 `teacher_s`（9.3） | |
| 启动期要有宽限（20 分钟） | 装 K3 权重约 5 分钟，此时还没有容器 |
| 同一个故障签名重演就**停下来别重启** | 见 9.3 第 3 条 |
| 重启必须带 `resume` | |

/!\ **看板脚本每次刷新都要重新解析 run 目录。** 自动重启会换 run 目录，如果循环体
里用的是启动时算好的路径，就会一直读那个已经不再增长的旧日志——症状是"step 和
s/步 卡在一个值上一百分钟不动"，看起来像训练变慢或卡死，实际训练在另一个目录里
跑得很正常。**这个坑我踩过两次。**

/!\ **`pgrep -f 'xxx.sh'` 会匹配到自己**（pgrep 自己的命令行就含这个串），
永远返回"在跑"。用 `ps -eo args | grep ... | grep -v grep`。

### 9.5 `resume` 之后 draft 会留在 CPU 上

/!\ 这是一个真实的代码 bug，从零训练不会遇到，但**任何一次故障重启都会**：

```
RuntimeError: mat2 is on cpu, different from other tensors on cuda:N
```

`_resume_from_checkpoint` 会故意把 draft 卸到 CPU，靠调用方搬回去，而只有
batch-alternating 那条路径做了。streaming 路径要在开头补一次：

```python
if self._draft_model is not None and any(
    p.device.type == "cpu" for p in self._draft_model.parameters()
):
    self._load_draft_to_gpu()
```

### 9.6 作业时长

/!\ 占节点的作业别用交互式会话——会话一结束节点就释放，checkpoint 变成不可达
（不是丢失，但拿不到）。用批处理作业占着，而且**检查 `sleep` 的时长和你申请的
`--time` 是否一致**。我有一次 `--wrap "sleep $SLEEP_SECS"` 用了默认的 43200 秒
而 `WALL` 传的是 48 小时，训练在 12 小时整齐地停了、作业状态是 `COMPLETED`，
一开始还以为是集群限制。

---

## 10. 导出

```bash
python3 output/export_dspark_hf.py \
    --ckpt <共享路径>/checkpoints/.../checkpoint_<N>.pt \
    --base-model <K3 路径> \
    --output <共享路径>/drafts/<名字> \
    --train-config <你的 train.yaml>
```

导出脚本做两件断言，**都不能跳过**：

1. **`config.json` 与训练 YAML 交叉核对**，特别是不能 emit `rope_interleave`
2. **68 个张量名的 key 契约**

**成功的标志：**

```
ATOM key contract: OK (68 tensors)
Total parameters: 3,562,312,961
Total size: 7.12 GB
config.json cross-checked against <你的 yaml>: OK
```

### 10.1 /!\ ATOM 对不认识的张量名不告警、不报错，直接静默忽略

**这是一次 3060 个前向步一次都没中的事故。** 导出过一份用训练框架内部命名的权重：

| ATOM 期望 | 曾经导出的 |
|---|---|
| `context_proj.weight` | `fc.weight` |
| `context_norm.weight` | `hidden_norm.weight` |
| `final_norm.weight` | `norm.weight` |

65 个张量对得上，ATOM 把躯干加载了、**静默丢掉了整条输入通路**。结果是一个
first-token 准确率 59% 的 draft 在服务端接受率 **0.00%**，而 server 日志从头到尾
完全健康（`Detected MLA DSpark drafter`、aux layers `(2,23,47,71,89)` 都正常）。

### 10.2 68 个张量的完整清单

**顶层 8 个：**

```
embed_tokens.weight
context_proj.weight        # 内部名 fc.weight，[7168, 35840]
context_norm.weight        # 内部名 hidden_norm.weight
final_norm.weight          # 内部名 norm.weight / out_norm.weight
markov_head.markov_w1.weight
markov_head.markov_w2.weight
confidence_head.proj.weight
confidence_head.proj.bias
```

**每层 12 个 × 5 层 = 60 个**（`layers.{0..4}.`）：

```
input_layernorm.weight
post_attention_layernorm.weight
self_attn.q_a_proj.weight
self_attn.q_a_layernorm.weight
self_attn.q_b_proj.weight
self_attn.kv_a_proj_with_mqa.weight
self_attn.kv_a_layernorm.weight
self_attn.kv_b_proj.weight
self_attn.o_proj.weight
mlp.gate_proj.weight
mlp.up_proj.weight
mlp.down_proj.weight
```

/!\ **不要导出 `lm_head.weight`**（带上会变成 9.47 GB 而不是 7.12 GB）。

### 10.3 导出的 config.json

```json
{
  "architectures": ["K3DSparkModel"],
  "model_type": "k3_dspark",
  "hidden_size": 7168,
  "intermediate_size": 14336,
  "num_hidden_layers": 5,
  "num_attention_heads": 64,
  "num_key_value_heads": 64,
  "q_lora_rank": 1536,
  "kv_lora_rank": 512,
  "qk_nope_head_dim": 128,
  "qk_rope_head_dim": 64,
  "v_head_dim": 128,
  "mla_use_nope": false,
  "mla_use_output_gate": false,
  "vocab_size": 163840,
  "rms_norm_eps": 1e-05,
  "max_position_embeddings": 1048576,
  "rope_theta": 50000.0,
  "num_target_layers": 5,
  "target_hidden_size": 7168,
  "target_num_hidden_layers": 93,
  "target_layer_ids": [2, 23, 47, 71, 89],
  "mask_token_id": 163837,
  "bos_token_id": 163584,
  "eos_token_id": 163586,
  "pad_token_id": 163839,
  "markov_rank": 256,
  "markov_head_type": "vanilla",
  "enable_confidence_head": true,
  "confidence_head_with_markov": true,
  "tie_word_embeddings": false,
  "draft_vocab_size": 163840,
  "_torchspec_version": "0.1.0",
  "torch_dtype": "bfloat16",
  "rope_parameters": {
    "rope_type": "yarn",
    "factor": 32.0,
    "original_max_position_embeddings": 32768,
    "rope_theta": 50000.0,
    "beta_fast": 32,
    "beta_slow": 1,
    "mscale": 1.0,
    "mscale_all_dim": 1.0
  }
}
```

/!\ **没有 `rope_interleave` 这一项**，这是故意的（3.1）。

/!\ `original_max_position_embeddings` 是 **32768**，`factor` 是 32，所以
**32768 本来就在原生范围内**，训练窗口拉到 32768 不需要动模型配置。

---

## 11. Benchmark

### 11.1 起服务

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

启动约 271 秒（本地 NVMe）。日志里必须有：

```
Detected MLA DSpark drafter
DSparkProposer aux capture on target layers: (2, 23, 47, 71, 89)
```

几个 flag 的理由：

- `--max-num-seqs 8`：客户端是串行的。设 64 时 K3 的 KDA recurrent state 池光自己
  就要 28.91 GB，而 KV 预算只有约 19 GB，服务直接拒绝启动
- `--no-enable_prefix_caching`：prefix 命中不计入 `scheduled_tokens`，统计会偏
- `--kv_cache_dtype fp8`：16384 窗口下 bf16 是 27 KB/token，fp8 是 13.8 KB/token

### 11.2 测量协议

| 参数 | 值 |
|---|---|
| temperature | **0** |
| top_p | 1.0 |
| 并发 | **1** |
| `num_speculative_tokens` | **7** |
| `max_tokens` | **12288**（16384 窗口，给最长 prompt 留 4096） |
| prompt 数 | **各集官方全量** |
| chat template | Kimi-K3 官方 |

**接受率从 `/debug/mtp_stats` 读，不要从墙钟时间反推。** 计数器是**自服务启动以来
累计**的，所以每集测量 = 跑前快照 + 跑后快照 + **取差值**。多集可以共用一个服务
（K3 加载一次约 4.5 分钟，每集重启是浪费）。

/!\ **`max_tokens` 不要设小。** 曾经用 256 测，被认为严重低估——后来实测从 2048
到 15360，AIME 的生成长度 ×3.9、前向步数 ×4，而 `tok/fwd` 只变了 **+0.2%**。
所以 12288 够了，但 256 那种冒烟值不能用于正式对比（12 条 prompt 的抽样噪声是 ±5%）。

### 11.3 题目集

| benchmark | 来源 | prompts |
|---|---|---|
| GSM8K | `openai/gsm8k` main/test | 1319 |
| MATH-500 | `HuggingFaceH4/MATH-500` test | 500 |
| AIME 2026 | `MathArena/aime_2026` | 30 |
| HumanEval | `openai/openai_humaneval` test | 164 |
| MBPP | `google-research-datasets/mbpp` full/test 前 256 | 256 |
| MT-Bench | `HuggingFaceH4/mt_bench_prompts` | 80 |
| SWE-bench Pro | `ScaleAI/SWE-bench_Pro` test 前 128 | 128 |
| SPEED-Bench coding / multilingual / rag / qa / writing | `nvidia/SPEED-Bench`，config `qualitative` | 各 80 |
| **SPEED-Bench low-entropy (16k)** | `nvidia/SPEED-Bench`，config `throughput_16k`，`category == low_entropy` | **512** |

/!\ **low-entropy 那一集必须单独用 32768 窗口的服务跑**（`--max-model-len 32768
--max-num-batched-tokens 32768`，`max_tokens` 用 8192）。它的输入均值约 10.4k、
最长约 22.5k，**58.9% 的 prompt 超过 8191**——这是 13 集里唯一有长 prompt 的一集，
也是唯一能验证长上下文能力的一集。

**AA-LCR 测不了**：官方 card 是在 71k–115k token 的多文档 prompt 上测的，而数据集
暴露的字段只有问题本身（均值 206 字符）。

### 11.4 客户端必须能续跑

/!\ **benchmark 跑到一半引擎会死。** 实测在 GSM8K 550/1319 处撞过一次 aiter 跨 rank
广播死锁：8 个 rank 全在自旋（`Rl` 状态、200% CPU、248 GB/卡、约 300 W），错误串是
每 60 秒一条的 `No available shared memory broadcast block found`（累计 200 条），
而 **`/health` 仍然返回 200**（那是 API 前端，和 engine core 是两个进程）。客户端
把 550 条已完成的工作全丢了。

用固定 digest 的镜像**不能规避**。曾经假设"约 630 请求是阈值"，后来一个生命周期
跑了 674 请求无恙，**推翻**——更像是特定输入形状偶发。

**所以客户端要有四件事：**

| 机制 | 值 |
|---|---|
| 单请求超时 → 主动退出 | **900 秒**，退出码 42 |
| 增量落盘 | 每 **25 条**把 `/debug/mtp_stats` 的增量折进 `<out>.partial.json` |
| `--resume` | 跳过已完成的条目 |
| `--rotate-after` | **400** 条主动交还，退出码 43 |

外面套一层：收到 42/43 就重启服务并带 `--resume` 续跑，上限 12 次。每次轮转代价
约 4 分钟（K3 重载）。

**跨服务生命周期续跑是有效测量**：接受长度分布在前向步上可加，客户端把 partial 里
的累计分布读回来当基线、服务端计数器只作增量起点，而且 prefix caching 关、采样贪心。

/!\ **主动轮转并不能证明是它避免了死锁**（请求计数不是触发条件），它的真实作用就是
把单次故障的损失上限压到 400 条。保证跑完的是 42 的重启续跑路径。

/!\ 重启前要把**已经产出最终 JSON 的题目集从列表里摘掉**——客户端在最终 flush 时
会删掉 `.partial.json`，对已完成的集子 `--resume` 无从下手，不摘就会从头重跑。

### 11.5 同机再测一遍官方 draft

把 `Inferact/Kimi-K3-DSpark` 下载下来，**同一台机器、同一个镜像、同一套协议、
同样的 prompt 数**跑一遍，拿它当分母。理由见 0.3。

---

## 12. 排错速查

| 症状 | 原因 | 处理 |
|---|---|---|
| `extract_hidden` 600 秒超时，**零训练步** | 镜像漂了（atom dev323 + torch 2.10），或 Mooncake 走了 RDMA | 用钉死 digest 重 build；`MOONCAKE_PROTOCOL=tcp` |
| `timed out waiting for response to cmd=extract_hidden after 600.0s`，**跑了几小时之后** | 某个 teacher 副本挂死（9.3） | `max_num_seqs` 32 → 24；**别原样重启** |
| 续训每次都死在同一步 | 采样器确定性重放同一个 batch | 改打包方式，不是改重试次数（9.3） |
| `Failed to register memory: Invalid argument [22]` / `Mooncake setup failed error=-600` | 同一张网卡上开第二个 RDMA client | 走 TCP |
| Mooncake `code=-200`，consumer 等 90 秒后死 | `global_segment_size` 装不下整批 | 按 8.3 重算；批大小变了就要重算 |
| `Ray cluster has 0/5 active nodes` | Ray 的 10001 端口被同机其他租户占了 | `--ray-client-server-port 26380 --include-dashboard=false` |
| `node discovery timed out (1/5)` | 五个 rank 是顺序起的 | 并行起，发现窗口放宽到 600 秒 |
| `Incompatible value 'None' for field of type 'str'` | override 写成 `mooncake.device_name=`（等号后为空）被解析成 None | 值为空时不要下发这个 override |
| `mkdir /opt/spur/.docker: permission denied` | 远程 exec 的 HOME 不可写 | `HOME=/tmp/x DOCKER_CONFIG=/tmp/x/.docker` |
| **服务端接受率 ≈ 0% 但日志全正常** | 导出的张量名和 ATOM 读的对不上，输入通路被静默丢弃（10.1） | 重新导出，看到 `ATOM key contract: OK (68 tensors)` |
| **训练指标漂亮但服务端很差** | RoPE 旋转约定 / YaRN `mscale²` 与服务端不一致（3.1 / 3.2） | 跑 RoPE 对齐自检，相对 L2 应在 1e-8 量级 |
| **训练侧 AL 涨但服务端不动** | eval 切片的分布和目标 benchmark 不重合（8.4） | 加固定 eval 切片 |
| eval 指标在某一步突然变好 | 采样器回绕进了 eval 切片 | 确认训练长度排除了尾部的 eval 行（5.4） |
| loss 从冷启动值开始、"收敛得慢"、无报错 | draft 权重导入走了 Eagle3 的重命名分支，全部变成 unexpected key | 用正确的映射表，并断言张量数 67 |
| `mat2 is on cpu, different from other tensors on cuda:N` | resume 把 draft 留在 CPU（9.5） | streaming 路径开头补 `_load_draft_to_gpu()` |
| eval AL 在某一步之后完全平掉 | `min_lr: 0`，lr 已经退到 7.6e-10（8.2） | min_lr 非零；已经发生的只能重开 schedule |
| GPU 显示 100% 但功耗只有约 320 W，无输出 | 集合通信自旋死锁 | 看 worker 日志里的 NCCL watchdog；重启续跑 |
| `No available shared memory broadcast block found`（benchmark 期间） | aiter 跨 rank SHM 广播死锁 | 客户端超时退 42 + 重启续跑（11.4） |
| `/health` 返回 200 但请求挂死 | API 前端 ≠ engine core | 不要用 health 判断引擎存活 |
| 生成容器 "Up" 但不产出 | 引擎死了或空转死锁，`docker wait` 不返回 | 按输出文件 mtime 判活性 |
| `SHUTDOWN signal from DP rank 0 during initialization` | 别人的容器占着卡，独占分配不驱逐它 | 铺机前查每卡显存（1.2） |
| `InsufficientPoolBudget: state pool needs ...B of -...B` | ATOM 用设备全局 `(total-free)` 算 non_torch，训练进程的残留被重复计费 | teacher 重启时要按残留量上调 `gpu_memory_utilization`；单一固定值无法同时满足首启和重启 |
| 一堆 `bgemm_internal_cublaslt error ... Will attempt to recover` | hipBLASLt 在某些形状上失败后回退 cublas | **噪音**，结果是对的，统计错误数时要排除 |
| `FlyDSL kernel ... not recognized by the current catalog; falling back` | aiter 调优表条目当前内核目录认不出 | **同样是噪音**（9.3 末） |
| `No available memory for the cache blocks` | KV 预算不够 | 确认没有别的进程占卡；降 `max_num_seqs` |
| 加载完只剩 175 GiB 可用显存（应约 265） | `PYTORCH_CUDA_ALLOC_CONF` 漏进容器了 | `docker inspect` 确认它为空（2.5） |
| `N 行 prompt 超过 max_prompt_tokens` 启动即退 | 生成侧默认 14304 是给 16384 窗口的 | 32768 窗口传 `--max-prompt-tokens 32224` |
| 训练启动时节点内存爆掉 | `input_ids` 存成 `list[int]`，36 字节/token | 改 int32 ndarray（7.3） |
| `'BF16Optimizer' object is not iterable` | torch 的 `get_optimizer_state_dict` 只接受 `torch.optim.Optimizer` | 非分片时直接用 `opt.state_dict()` |
| 训练在 12 小时整齐地停了，作业状态 `COMPLETED` | 占位作业的 `sleep` 时长用了默认值 | 核对 `sleep` 和 `--time`（9.6） |

---

## 13. 一句话流程

```
① 申请 5 个节点（独占，-G 8），铺机前查每卡显存和本地盘；关掉 numa_balancing
② 拉钉死 digest 的基础镜像，四项版本核对；build 训练镜像（--no-deps、antlr 4.9.3、
   opentelemetry、禁 vLLM、四个 ATOM API 断言）；一台 build 后 save/load 分发
③ 每台下 K3 权重到本地 NVMe，核 96 个分片和 config.json
④ 上机自检三项全绿：RoPE 相对 L2 ~1e-8 且与 half-split 不匹配、flex 与 dense 等价、
   hidden state 宽度 35840
⑤ 建 prompt 池：四条判据过滤图文 -> 入库即去重（先英文后多语种）-> 13-gram 去污染
   -> 按 5.2 的份额分配 210 万配额，其中代码 20%、多语种 12%
⑥ 用 K3 重新生成回答（32768 窗口，--max-prompt-tokens 32224，max_num_seqs 384,
   fp8 KV，temperature 0，关 draft），watchdog 按 mtime 判活 + supervisor
⑦ 逐 token 回环校验转成训练集，丢弃率约 8.8% 属正常，超阈值就停下来查对齐
⑧ CPU 预处理量出存活行数 -> num_training_steps = 一个 epoch × 3；物化固定 eval 切片
⑨ 五节点并行开训（32768、cosine 5e-5 -> min_lr 2e-6、Mooncake TCP 256GB、
   teacher max_num_seqs 24、两把 eval 尺子、看守分副本监控 teacher_s）
⑩ 第一个 epoch 结束就导出 + benchmark 一次确认方向；训完再导出
⑪ 导出（key 契约断言 68 张量、7.12 GB）-> 起 ATOM 服务 -> 跑 13 集全量
   -> **同机再测一遍官方 draft 当分母**
```

**预算**：数据生成几十到几百个节点小时；训练约 122 小时（5.1 天，49,200 步 ×
8.9 s）；benchmark 单机串行约 6.7 小时，五机并行约 2 小时。
