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

