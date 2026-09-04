# Multi-Tune Agent

Multi-Tune is a framework-independent, multi-role code-agent harness for GPU
kernel optimization. The in-tree `geak_utils` package provides
orchestration-neutral GEAK task, sandbox, and evaluation adapters; MultiTune
owns stateful sessions, forks, rewards, resume behavior, and its tool schema.
It does **not** import or register verl or Uni-Agent.

The internal runtime borrows two useful design ideas from
[verl agent_loop](https://github.com/verl-project/verl/tree/main/verl/experimental/agent_loop)
and [Uni-Agent](https://github.com/verl-project/uni-agent):

- an explicit `AgentLoop` state machine with durable tool sessions;
- separate `Agent`, `Task`, `Tool`, `Sandbox`, `Environment`, reward, and
  trajectory abstractions.

All implementations live in this directory.

## Flow

For each kernel case:

1. Director creates a canonical GEAK session and freezes the baseline.
2. TechLead analyzes the task and emits independent optimization directions.
3. Engineers run concurrently against private forks of the current best source.
4. GEAK independently compiles, checks correctness, and benchmarks every result.
5. Verifier agents interpret evidence, but cannot override the deterministic gate.
6. Integrator receives candidate session IDs, reads candidate source through the
   GEAK tool, and attempts a measured merge in another private fork.
7. Only a candidate that clears `min_improvement` becomes the next canonical best.
8. Director performs a final independent validation.

GPU work remains serialized by GEAK's `gpu_lock.sh`; model calls and source
analysis can overlap.

## Package layout

- `core.py`: framework-neutral Agent/Task/Tool/Sandbox/Environment protocols.
- `runtime.py`: OpenAI-compatible model backend and async tool-agent loop.
- `geak_tool.py`: forkable GEAK sessions, source boundary, evaluation, reward.
- `agents.py`: GEAK role prompts and structured/code role agents.
- `flow.py`: hierarchical multi-round scheduler.
- `trajectory.py`: thread-safe JSONL event log and summary.
- `cli.py`: probe and run commands.
- `geak_utils/`: packaged adapters, task contracts, and trusted example
  templates. Upstream GEAK remains unmodified.

## 从新克隆到启动

当前部署由两个进程组成：

- `geak-phase1-vllm`：在 GPU 0 上提供 OpenAI-compatible 模型接口；
- MultiTune CLI：在宿主机编排 Agent，并让 GEAK 在 GPU 1 上执行 kernel
  compile/correctness/performance。

MultiTune 暂时没有供多人远程提交任务的 HTTP API。端口 `8000` 是原始模型
接口，不是 MultiTune 任务服务，不应直接暴露到公网。

### 1. 前置条件

- Linux、Docker 和支持 ROCm 的 AMD GPU；
- Docker 可以访问 `/dev/kfd` 和 `/dev/dri`；
- 推荐至少两张 GPU：GPU 0 运行模型，GPU 1 测试 kernel；
- Python 3.10 或更高版本；
- 当前用户有执行 Docker 的权限。

### 2. 克隆仓库

```bash
git clone https://github.com/ZhangDanyang-AMD/Lumen-RL.git
git clone https://github.com/AMD-AGI/GEAK.git /desired/path/GEAK
cd Lumen-RL/experiments/multi-tune-agent
```

`geak_utils` 已包含在 MultiTune 目录中，不需要单独克隆或安装。上游 GEAK
checkout 可以放在任意位置，并保持未修改；通过 `GEAK_HOME` 指向它。

### 3. 创建 Python 环境

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
GEAK_HOME=/desired/path/GEAK python3 -m pip install -e ".[test]"
```

这一条 editable install 会同时安装 `multi_tune_agent`、内嵌的
`geak_utils`、测试依赖，以及声明自
`geak @ git+https://github.com/AMD-AGI/GEAK.git` 的官方 GEAK。
安装后可使用 `multi-tune` 命令。

如果需要手动管理 GEAK 安装（例如离线镜像），可先安装基础依赖和本地
GEAK checkout，再用 `--no-deps` 安装 MultiTune：

```bash
python3 -m pip install PyYAML requests pytest
python3 -m pip install -e "$GEAK_HOME"
python3 -m pip install -e . --no-deps
```

也可以不安装 editable package，改用
`PYTHONPATH=src:. python3 -m multi_tune_agent.cli`。

### 4. 修改机器配置

编辑 `configs/mi300x.yaml`。其他用户至少需要替换其中硬编码的 home
目录：

```yaml
geak_root: /home/<user>/GEAK
cases_path: ../cases/examples_cases.yaml
trajectory_root: /home/<user>/multi_tune_runs
base_url: http://127.0.0.1:8000/v1
model: Qwen/Qwen3-Coder-Next
gpu_ids: "1"
```

主要配置项：

- `geak_root`：GEAK 仓库绝对路径；
- `cases_path`：case catalog，相对路径以配置文件目录为基准；
- `trajectory_root`：run 日志、summary 和隔离 workspace 的保存位置；
- `base_url`、`model`：OpenAI-compatible vLLM endpoint 和 served model；
- `gpu_ids`：kernel evaluation 使用的 GPU，不能与模型 GPU 冲突；
- `request_timeout`、`command_timeout`：模型请求和 GEAK 命令超时；
- `baseline_repeats`：冻结 baseline 时的性能重复次数；
- `max_rounds`、`engineers_per_round`：优化轮数和每轮并行 Engineer 数；
- `engineer_tool_rounds`、`integrator_tool_rounds`：代码 Agent 最大交互轮数；
- `candidate_floor`：进入候选集所需的最低 speedup；
- `min_improvement`：候选晋升为当前最佳版本所需的最小提升；
- `target_speedup`：达到后允许提前停止；
- `keep_sessions`：是否保留 Engineer/Integrator workspace。

`GEAK_HOME` 会覆盖 YAML 的 `geak_root`；更具体的
`LUMEN_CODE_GEAK_ROOT` 会覆盖两者。环境变量中的相对路径与 YAML 路径
一样，以配置文件所在目录为基准解析。示例：

```bash
GEAK_HOME=/desired/path/GEAK multi-tune --config configs/mi300x.yaml probe
```

### 5. 启动 vLLM 模型服务

首次启动前创建模型缓存：

```bash
mkdir -p "$HOME/.cache/huggingface"
```

首次创建容器：

```bash
docker run -d --name geak-phase1-vllm \
  --device /dev/kfd --device /dev/dri --group-add video \
  --ipc=host --network=host --security-opt seccomp=unconfined \
  -e HIP_VISIBLE_DEVICES=0 \
  -e VLLM_ROCM_USE_AITER=0 \
  -e HF_HOME=/root/.cache/huggingface \
  -v "$HOME:$HOME" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  vllm/vllm-openai-rocm:v0.15.0 \
  Qwen/Qwen3-Coder-Next-FP8 \
  --served-model-name Qwen/Qwen3-Coder-Next \
  --max-model-len 32000 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.90 \
  --kv-cache-dtype fp8 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --port 8000
```

容器已经创建但停止时，不要再次 `docker run --name`，直接启动：

```bash
docker start geak-phase1-vllm
```

查看模型加载日志：

```bash
docker logs -f geak-phase1-vllm
```

出现服务启动完成的信息后按 `Ctrl+C` 只会退出日志查看，不会停止容器。

### 6. 检查服务

```bash
cd ~/Lumen-RL/experiments/multi-tune-agent
source .venv/bin/activate
multi-tune --config configs/mi300x.yaml probe
```

输出中的 `configured_model_present` 必须为 `true`。如果使用
`PYTHONPATH` 方式：

```bash
PYTHONPATH=src:. python3 -m multi_tune_agent.cli \
  --config configs/mi300x.yaml probe
```

### 7. 启动交互 CLI

前台运行：

```bash
multi-tune --config configs/mi300x.yaml interactive
```

使用 tmux 后台运行：

```bash
cd ~/Lumen-RL/experiments/multi-tune-agent
tmux new-session -d -s multi-tune-cli \
  '.venv/bin/multi-tune --config configs/mi300x.yaml interactive'
tmux attach -t multi-tune-cli
```

也可以加载项目提供的 Shell functions，然后用一个命令启动或重新连接：

```bash
cd ~/Lumen-RL/experiments/multi-tune-agent
source env.sh
lumen-code
```

`source env.sh` 会把 `src` 和项目根目录无重复地加入
`PYTHONPATH`，并设置 `GEAK_HOME`、`LUMEN_CODE_GEAK_ROOT`、
`GEAK_CONTAINER_NAME`、配置路径和项目路径。`lumen-code`
会启动已存在但停止的 `geak-phase1-vllm` 容器、创建或复用
`multi-tune-cli` tmux session、开启 mouse mode 并 attach。辅助命令：

```bash
lumen-code-status
lumen-code-stop
kill-lumen-code
```

`lumen-code-stop` 和 `kill-lumen-code` 等价，只停止 CLI 的 tmux session，
不会停止 `geak-phase1-vllm` 模型容器。

默认网络变量也由 `env.sh` 统一提供：

```text
LUMEN_CODE_SERVER_HOST=10.194.134.84
LUMEN_CODE_VLLM_HOST=127.0.0.1
LUMEN_CODE_VLLM_PORT=8000
LUMEN_CODE_TUNNEL_PORT=18000
LUMEN_CODE_BASE_URL=http://127.0.0.1:8000/v1
LUMEN_CODE_REMOTE_BASE_URL=http://10.194.134.84:8000/v1
```

`LUMEN_CODE_BASE_URL` 会覆盖 YAML 中的 `base_url`。可以在 source 前预设
变量，例如通过 SSH tunnel 使用本地 `18000` 端口：

```bash
export LUMEN_CODE_VLLM_PORT=18000
source env.sh
```

如需在任意新终端自动提供这些命令，可将下面一行加入 `~/.bashrc`：

```bash
source ~/Lumen-RL/experiments/multi-tune-agent/env.sh
```

如果 `tmux attach` 返回 `no sessions`，说明 CLI 已退出，重新执行
`tmux new-session` 即可。若提示 `no server running`，也应先创建 session，
而不是先执行 `tmux source-file`。

开启鼠标滚动，在 `~/.tmux.conf` 中加入：

```text
set -g mouse on
```

新 tmux server 会自动读取配置。已有 server 可执行：

```bash
tmux set-option -g mouse on
```

无鼠标时按 `Ctrl+b`，松开后按 `[` 进入滚动模式；按 `q` 退出。

### 8. CLI 选项

`interactive` 支持：

1. 运行 `cases/examples_cases.yaml` 中的已有 case；
2. 从包含 GPU、`M`、`N`、`K` 的自然语言请求生成 GEMM task，并明确选择
   FP16、FP8 A8W8 或原生 MXFP4；
3. 使用现有的自定义 GEAK task 目录；
4. 失败或中断后通过 `[c] continue` 恢复最近 checkpoint。

自然语言 task 创建会先调用已配置的 vLLM 模型识别 operator、GPU、format、
M/N/K 和 language，再对结构化结果做数值证据与范围校验并要求用户确认。
模型不可用时会回退到本地解析；模型不得补造原请求中不存在的维度，也不能
在原文没有证据时补造 format 或 language（默认分别为 FP16 和 Triton）。

自动 GEMM 路径通过不可变 template registry 选择并复制
`examples/tasks/gemm`、`gemm_fp8` 或 `gemm_mxfp4` 的可信 seed，在
`examples/tasks/` 下生成
shape-specific harness，并把任务注册到 `cases/examples_cases.yaml`。
MI300/MI308 映射到 gfx942，MI350/MI355 映射到 gfx950；当前 FP8 仅支持
gfx942，MXFP4 仅支持 gfx950，且后者保持原生 AITER、无模拟 fallback。
其他 operator 仍需要现有可信 harness。自定义目录必须包含
`config.yaml`、`COMMANDMENT.md` 或 `unittest.py + meta.json`。

`[c] continue` 会把最近 candidate workspace 复制到新 run，保留原始冻结
baseline，并把最近失败结果传给 Agent。它不是恢复原模型 KV cache，而是从
持久化代码与测试证据继续。

### 9. 非交互运行

```bash
multi-tune --config configs/mi300x.yaml run \
  --case dense-gemm-fp16 \
  --request "Optimize this GEMM shape for MI308X without correctness regressions." \
  --stream
```

省略 `--case` 会顺序运行 catalog 中的全部 case。单个 case 内的 Engineer
会并行生成候选，但 GPU evaluation 仍由 GEAK lock 串行执行。

### 10. 常见状态与错误

- `read_file/write_file/state ok=True`：工具操作成功；
- `evaluate ok=False`：compile、correctness 或 performance 命令失败，具体
  traceback 在 `trajectory.jsonl`；
- GPU `Memory access fault`：候选 kernel 通常发生越界访问；
- `run_shell_command ok=False`：GEAK 有意禁止任意 shell；允许的 action
  只有 `list_files/read_file/read_candidate/write_file/evaluate/state`；
- HTTP 400 context error：模型输入与预留输出超过 `--max-model-len`；
- `[c]` 未显示：没有可恢复的未完成 run，或该 checkpoint 已被继续过。

永久 case 目标保存在 `cases/examples_cases.yaml`。生成的本地 task 位于
`examples/tasks/` 并被 Git 忽略。原始 task 不会被 Agent 直接修改。

## 自动生成缺失的 kernel 模板

交互菜单 `[2] generate kernel task` 支持任意 operator。用户描述 operator、
GPU、shape、format/dtype 和量化 scale 契约后，系统按以下顺序处理：

1. 精确匹配 bundled template 或已通过 GPU 验证的本地模板；命中后直接创建
   case，不调用模板生成模型；
2. 没有模板时，先让 vLLM 直接生成隔离的 `kernel.py` 和独立
   `task_runner.py`；
3. direct draft 静态校验失败时，才只读搜索 `$AITER_HOME` 中的 public
   wrapper、test、reference、benchmark 和架构配置，并让模型修复整个 draft；
4. 通过 GEAK GPU lock 顺序执行 compile、correctness、performance；
5. 三项均通过后提升为本地可信模板并注册 case；默认
   `bootstrap_auto_promote: true` 会自动继续正常优化，设为 `false` 时才要求
   用户逐次确认。

### 决策优先级

这套顺序是控制面必须遵守的优先级，而不是给 Agent 的建议：

1. **P0：契约与硬件 hard gate。** 先验证 operator、shape、dtype/format、
   scale granularity、block size 和目标架构。FP8/gfx942、MXFP4/gfx950 等
   不兼容组合在创建 task 前立即拒绝，不进入模型生成或 GPU 执行。
2. **P1：复用可信实现。** 优先使用仓库内不可变的 canonical template；
   其次按完整 `contract_hash` 复用已通过 GPU gate 的本地模板。命中后直接
   开始优化，不重新生成、不搜索 AITER，也不重复消耗验证资源。
3. **P2：LLM 直接生成并测试。** 没有可信模板时，默认假设模型可能已经知道
   kernel 的实现方式，让模型直接生成完整 draft。只要静态检查通过，就直接
   进入 GPU compile/correctness/performance；不会为了“找依据”先搜索代码库。
4. **P3：AITER evidence-assisted repair。** 只有 direct draft 无法通过静态
   安全检查时，才搜索 AITER。搜索是只读、按 operator/format/scale/arch
   hard gate 和置信度排序的恢复路径；只把 public wrapper、独立 reference、
   test、benchmark/config 等证据交给模型修复，不直接执行或复制未知实现。
5. **P4：信任与晋升。** 静态检查只证明 draft 没有明显绕过 harness；最终仍
   必须按 compile → correctness → performance 顺序通过独立 GPU gate。只有
   `bootstrap_auto_promote` 允许时才写入本地可信 registry 和 case catalog。
   任何失败 draft 都只保留诊断信息，不会成为可优化 baseline。

核心原则是：**已知且可信就复用，模型会写就直接写并实测，模型写不对才借助
AITER 修复；搜索不能替代独立 oracle 和 GPU 测量。**

已明确写在请求中的 format、scale granularity 不会被重复询问。缺失字段会
一次性提示补充。例如：

```text
MI308X FP8 GEMM M=16 N=4096 K=4096,
activation per-token scale, weight per-channel scale, FP16 output
```

本地文件位置：

- 未通过验证的 draft：`examples/tasks/.generated/<contract-hash>/`；
- 验证失败诊断：同目录下 `.failed-<contract-hash>*`；
- 已通过 GPU gate 的模板：`examples/tasks/generated/<contract-hash>/`；
- 本地模板索引：`examples/tasks/generated/templates.yaml`。

这些运行时生成的模板不会进入 wheel。`metadata.json` 保存完整 contract、
模型 provenance、可选 AITER artifact hash 和 GPU gate 证据。失败 draft
不会写入 `cases/examples_cases.yaml`。`bootstrap_enabled: false` 可以关闭
自动生成；`AITER_HOME`、`generated_template_root` 和
`bootstrap_min_aiter_score` 可在配置或环境中调整。

原生 OCP MXFP8/MXFP6/MXFP4 依赖 gfx950 的 CDNA4 block-scaled MFMA。
MI308/gfx942 请求会在模板生成前拒绝，并建议改用 MI350/MI355 或 gfx942
原生 FP8 A8W8 E4M3FNUZ；系统不会把普通 FP8 仿真实现注册成原生 MXFP。

GPU 三项通过只能证明模板在指定输入下满足其独立 oracle；静态校验还会阻止
runner/oracle 成为可写源码、恒真比较、跳过 correctness、过宽容差和架构门禁
缺失。无法建立独立可信 oracle 时，模板不会注册。

### 硬件验证状态

- gfx942 / GPU 1：FP8 A8W8 的 compile、4 个 correctness case 和 4 个
  performance case 已通过；
- gfx942 / GPU 1：模型生成的非 GEMM softmax 模板已通过完整 GPU gate；
- gfx942：MXFP4 在 task 创建前被拒绝，并明确提示仅支持 gfx950；
- gfx950：MXFP4 native compile/correctness/performance 仍需在 MI350/MI355
  硬件上验证，gfx942 结果不能替代该验证。

## 与 GEAK Claude Code CLI / Workflow 的取舍

这里比较的是两种**模型接入和控制面部署方式**：MultiTune 直接调用常驻的
OpenAI-compatible vLLM 服务；GEAK v4 通常由 Claude Code CLI 启动
deterministic JS Workflow。两者都使用 GEAK 的 workspace、GPU lock 和真实
测量能力。GEAK 本身也具备确定性编排、并行 Engineer、独立验证和恢复机制，
因此 MultiTune 的优势不应表述成“GEAK 不稳定或不能并行”，而主要体现在以下
方面。

### 大规模并行

- **模型侧路径更直接。** MultiTune 的多个角色由同一个 Python scheduler
  直接向常驻 vLLM endpoint 发请求，不把 Claude Code CLI 的生命周期、权限
  交互和本地工具加载作为每个优化 run 的依赖。服务端可以做 continuous
  batching，Agent 数量增加时更容易共享模型权重和 KV-cache 容量，并统一做
  并发限制与 backpressure。GEAK Workflow 的 sub-agent 不一定各自对应一个
  CLI 进程，因此优势是服务化路径和集中调度，而不是简单的“少启动 N 个进程”。
- **候选天然隔离。** 每个 Engineer 从当前最佳 session fork 独立 workspace，
  模型分析和源码编辑可以并行；GPU compile/correctness/performance 再按
  GEAK lock 串行，避免并发 benchmark 互相污染。模型并行度和 GPU 并行度可以
  分别配置。
- **更适合作为训练或调度后端。** 请求、tool call、logprob、reward 和
  latency 都是结构化数据，不必解析 CLI 文本或终端状态，便于后续接入队列、
  多租户调度、RL rollout 和批量实验。

当前实现还不是完整的分布式调度器：`run_all` 仍按 case 顺序运行，单个
`base_url` 是模型服务单点，同一 GPU 上的 evaluation 也有意串行。要扩展到
多机大规模运行，还需要全局任务队列、每 GPU lock、endpoint pool、限流和
失败重试。GEAK 已支持按 kernel 启动独立 Agent，以及在多 GPU bake-off 中
并行 lane；在这些现成功能上它目前更成熟。

### 稳定性与可恢复性

- **控制流不交给模型。** round、fan-out、candidate gate、promotion、stop 和
  final validation 都由 Python 状态机决定；模型只负责结构化判断和受限源码
  修改。错误的 Agent 结论不能覆盖 deterministic correctness/performance
  结果。
- **缩小可变运行面。** 常驻 HTTP 模型服务不依赖交互式 CLI 会话、权限确认、
  本地插件和终端状态。模型只看到一个受限 `geak` tool，不能执行任意 shell，
  也不能修改 runner、oracle、metadata 或 config。
- **证据可重放。** baseline、workspace lineage、每次命令结果、模型消息、
  reward 和 checkpoint 持久化到 trajectory；中断后从代码和冻结 baseline
  恢复，而不是依赖某个 CLI 进程或模型 KV-cache 仍然存活。
- **生成模板有独立信任边界。** 未验证 draft 与可信 registry 分离；静态
  validator、架构 gate 和 GPU gate 全部通过前，不会进入正常优化 catalog。

代价是 MultiTune 主 Python 进程目前也是单点，`requests`/vLLM endpoint
故障会影响同一批运行；GEAK 的独立 CLI/Workflow 进程在进程级故障隔离、
成熟的 hang guard、Web/Profiling 工具生态和端到端 serving workflow 上更强。
另外，受限 tool 提高了安全性，但 task 中受信任的 Docker command 仍具有执行
能力，不能等同于完整的系统级 sandbox。

### 可扩展性与可维护性

- **模型后端可替换。** `ModelBackend` 是小接口；任何兼容 chat completions
  和 tool-calling 的本地或远程模型都可替换，不绑定 Claude Code CLI 的版本、
  账号体系或权限模式。
- **operator 与编排解耦。** `TaskSpec`、contract metadata、template
  registry、静态 validator 和 GEAK adapter 不依赖具体角色。新增 operator
  主要是补充可信 harness/contract/knowledge mapping，不需要重写 Agent loop。
- **策略可实验。** Director/TechLead/Engineer/Verifier/Integrator、并发数、
  reward、candidate threshold 和停止条件都是显式模块或配置，适合做消融、
  A/B 和训练数据采集。
- **上游 GEAK 保持不修改。** MultiTune 把 GEAK 当作 materialization、
  locking 和 evaluation 基础设施，通过 `geak_utils` 适配；升级控制面时不需要
  fork 上游 GEAK。

GEAK 的优势则是覆盖面更广：现有 e2e serving 优化、profile/Amdahl routing、
多 backend/language bake-off、Web research 和 learned knowledge workflow
都比 MultiTune 完整。MultiTune 当前更适合“可控、可观测、可训练的
kernel-candidate 生成与验证服务”；GEAK CLI/Workflow 更适合“开箱即用、
工具丰富的单机或端到端自主优化”。长期方向不是替代 GEAK，而是保留其可信
执行层，并把 MultiTune 的服务化调度、结构化轨迹和训练接口叠加在上面。

## 添加新的 kernel 类型与模板 case

模板不是按 shape 创建的。同一种 operator、输入输出布局、数值语义、量化
scale contract 和目标架构下，新 case 只需要复用模板并提供新参数。只有新增
kernel 类型，或上述任一契约发生变化时，才需要添加新的可信模板。例如普通
FP8 A8W8 shape 复用 `gemm_fp8`；如果改成不同的 scale 粒度或输出类型，则应
建立另一个模板。模板被删除后不会自动重建，task factory 会明确报缺少可信
seed。

### 1. 模板目录

在 `examples/tasks/<template_name>/` 中至少提供：

```text
examples/tasks/<template_name>/
├── config.yaml
├── kernel.py
└── scripts/
    └── task_runner.py
```

- `kernel.py`：Agent 唯一允许修改的 kernel 实现或调优配置；
- `scripts/task_runner.py`：不可由 Agent 修改的 compile、correctness 和
  performance harness；
- `config.yaml`：声明可写源码、目标函数和三类验证命令；
- `metadata.json`：可选；量化格式、packing、scale、架构限制较复杂时建议
  添加，`gemm_mxfp4` 可作为示例。

最小 `config.yaml` 示例：

```yaml
source_file_path:
  - kernel.py
target_kernel_functions:
  - my_kernel
compile_command:
  - docker exec -e HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-1} -w "$PWD" ${GEAK_CONTAINER_NAME:-geak-phase1-vllm} python3 scripts/task_runner.py compile
correctness_command:
  - docker exec -e HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-1} -w "$PWD" ${GEAK_CONTAINER_NAME:-geak-phase1-vllm} python3 scripts/task_runner.py correctness
performance_command:
  - docker exec -e HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-1} -w "$PWD" ${GEAK_CONTAINER_NAME:-geak-phase1-vllm} python3 scripts/task_runner.py performance
task_type: triton2triton
prompt:
  instructions: Preserve the complete operator contract while optimizing latency.
```

不要把 runner、oracle、配置或 metadata 加入 `source_file_path`，否则 Agent
可以通过修改验证标准获得虚假收益。

### 2. Harness 要求

`task_runner.py` 必须接受三个独立 mode：

```bash
python3 scripts/task_runner.py compile
python3 scripts/task_runner.py correctness
python3 scripts/task_runner.py performance
```

Harness 应满足以下要求：

1. correctness 使用独立可信 oracle、固定随机种子和明确容差；
2. 量化算子以实际送入 kernel 的量化值及 scale 计算 reference，不以量化前
   原始张量代替；
3. performance 在预热后只测 kernel 热路径，不包含输入生成、量化或
   reference；
4. 每个 shape 输出 `Perf: <latency_ms> ms (<case_id>)`，并可写入
   `build/performance_report.json`；
5. compile、correctness 或 performance 失败时必须返回非零状态；
6. 架构不支持时应在导入 JIT 后端或分配大张量前拒绝，不能用软件模拟结果
   冒充原生 kernel 性能。

可以参考：

- `examples/tasks/gemm`：FP16 Triton 模板；
- `examples/tasks/gemm_fp8`：gfx942 FP8 A8W8 及逐 token/channel scale；
- `examples/tasks/gemm_mxfp4`：gfx950 原生 MXFP4 和架构拒绝路径；
- `examples/tasks/fused_attention`、`grouped_gemm`：非 dense GEMM 模板。

### 3. 接入 MultiTune

新增模板后按以下顺序接入：

1. 在 `cases/examples_cases.yaml` 添加稳定 case ID、case type、模板路径和
   优化目标；
2. 新 GEMM format 在 `geak_utils/templates.py` 注册模板目录、case type、
   支持架构、backend 以及输入/scale/输出 contract；
3. 新 operator case type 加入 `geak_tool.py` 的
   `_SUPPORTED_CASE_TYPES`，同时加入 `cli.py` 的 `_CASE_TYPES`；
4. 在 `agents.py` 的 `_CASE_KNOWLEDGE` 映射对应
   `GEAK/perf_knowledge/operators/<operator>/` 知识目录；
5. 在 `.gitignore` 放行可信模板，并在 `pyproject.toml` 与 `setup.py` 中
   加入 wheel data files；
6. 添加 parser、catalog、架构门禁、任务生成和 flow 测试，并在受支持 GPU
   上分别运行 compile、correctness、performance。

Catalog 条目示例：

```yaml
- id: my-kernel-case
  type: my_kernel_type
  kernel_path: ../examples/tasks/my_kernel_template
  direction: Optimize supported shapes without changing numerical semantics.
```

只新增已有模板支持的 shape 时，不需要重复以上步骤；通过 task factory
生成 shape-specific `task_spec.json` 和 catalog 条目即可。

## Agent, GEAK, and harness boundaries

- **Agent** is the model-driven decision layer in `agents.py` and `runtime.py`.
  It analyzes evidence, proposes directions, edits allowed source, and decides
  which GEAK action to call.
- **MultiTune GEAK tool** in `geak_tool.py` owns stateful sessions, forks,
  baseline caching, rewards, resume integration, and the single OpenAI tool
  schema.
- **geak_utils** owns task loading, adapters, trusted templates, source write
  boundaries, deterministic evaluation, and calls into an unmodified upstream
  GEAK checkout for workspace materialization and GPU locking.
- **Harness** is supplied by each task directory or a deterministic
  operator-specific task factory, never improvised by the optimizing Agent. Its
  compile/correctness/performance commands define objective ground truth and
  emit parseable latency measurements. The agent cannot edit harness files.
- **MultiTuneFlow** is the deterministic control plane. It schedules roles and
  rounds, applies correctness/candidate/promotion gates, and persists evidence.

## GEAK tool contract

One OpenAI function named `geak` exposes these actions:

- `list_files`
- `read_file`
- `read_candidate`
- `write_file`
- `evaluate` (`compile`, `correctness`, `performance`, or `full`)
- `state`

Every tool call is bound to one session. `write_file` delegates to
`geak_utils.KernelSandbox`, so tests, scripts, metadata, configuration, and oracle files are
not writable. Forks inherit source from their parent but retain the original
frozen baseline.

## Reward

Incorrect or uncompilable candidates receive `-1`. Correct candidates receive:

```text
1 + clipped(log(speedup), -1, log(3)) + 0.5 * max(0, speedup - previous_best)
```

Correctness and measured promotion remain deterministic; Verifier and Director
outputs are advisory records.

## Trajectories

Each run writes:

```text
<trajectory_root>/runs/<case>_<timestamp>/
  trajectory.jsonl
  summary.json
```

Events include role, phase, round, model text, tool arguments/results, token
logprobs, usage, reward components, workspace lineage, compile/correctness/
performance evidence, and wall-clock timing. Private workspaces live under
`<trajectory_root>/sessions/`.

`summary.json` 中的 `final_kernel_performance` 明确记录 baseline/final
geomean 延迟、逐 case 延迟和 speedup。只有通过 compile、correctness 与
performance measurement 后，最终 kernel 速度才会标记为有效。

The HTTP backend records completion logprobs but intentionally does not invent
token IDs for tool observations. A later trainer-specific ETL can tokenize the
stored messages and mask tool-observation tokens from policy loss.

