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

## From a fresh clone to startup

The current deployment consists of two processes:

- `geak-phase1-vllm`: provides an OpenAI-compatible model endpoint on GPU 0;
- MultiTune CLI: orchestrates Agents on the host and has GEAK run kernel
  compile/correctness/performance on GPU 1.

MultiTune currently has no HTTP API for multiple users to submit tasks remotely.
Port `8000` is the raw model endpoint, not a MultiTune task service, and should
not be exposed directly to the public Internet.

### 1. Prerequisites

- Linux, Docker, and a ROCm-capable AMD GPU;
- Docker access to `/dev/kfd` and `/dev/dri`;
- at least two GPUs are recommended: GPU 0 runs the model and GPU 1 tests kernels;
- Python 3.10 or later;
- permission for the current user to run Docker.

### 2. Clone the repositories

```bash
git clone https://github.com/ZhangDanyang-AMD/Lumen-RL.git
git clone https://github.com/AMD-AGI/GEAK.git /desired/path/GEAK
cd Lumen-RL/experiments/multi-tune-agent
```

`geak_utils` is included in the MultiTune directory and does not need to be
cloned or installed separately. The upstream GEAK checkout can be placed
anywhere and kept unmodified; point `GEAK_HOME` to it.

### 3. Create the Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
GEAK_HOME=/desired/path/GEAK python3 -m pip install -e ".[test]"
```

This editable install installs `multi_tune_agent`, the bundled `geak_utils`,
test dependencies, and the official GEAK declared by
`geak @ git+https://github.com/AMD-AGI/GEAK.git`. The `multi-tune` command is
available after installation.

To manage the GEAK installation manually (for example, with an offline image),
first install the base dependencies and local GEAK checkout, then install
MultiTune with `--no-deps`:

```bash
python3 -m pip install PyYAML requests pytest
python3 -m pip install -e "$GEAK_HOME"
python3 -m pip install -e . --no-deps
```

Alternatively, skip installing the editable package and use
`PYTHONPATH=src:. python3 -m multi_tune_agent.cli`.

### 4. Configure the machine

Edit `configs/mi300x.yaml`. Other users must at least replace the hard-coded
home directories:

```yaml
geak_root: /home/<user>/GEAK
cases_path: ../cases/examples_cases.yaml
trajectory_root: /home/<user>/multi_tune_runs
base_url: http://127.0.0.1:8000/v1
model: Qwen/Qwen3-Coder-Next
gpu_ids: "1"
```

Key configuration options:

- `geak_root`: absolute path to the GEAK repository;
- `cases_path`: case catalog; relative paths are resolved from the configuration
  file's directory;
- `trajectory_root`: location for run logs, summaries, and isolated workspaces;
- `base_url`, `model`: OpenAI-compatible vLLM endpoint and served model;
- `gpu_ids`: GPU used for kernel evaluation; it must not conflict with the model GPU;
- `request_timeout`, `command_timeout`: timeouts for model requests and GEAK commands;
- `baseline_repeats`: number of performance repetitions when freezing the baseline;
- `max_rounds`, `engineers_per_round`: optimization rounds and concurrent Engineers per round;
- `engineer_tool_rounds`, `integrator_tool_rounds`: maximum interaction rounds for code Agents;
- `candidate_floor`: minimum speedup required to enter the candidate set;
- `min_improvement`: minimum improvement required to promote a candidate to the current best;
- `target_speedup`: permits early stopping once reached;
- `keep_sessions`: whether to retain Engineer/Integrator workspaces.

`GEAK_HOME` overrides the YAML `geak_root`; the more specific
`LUMEN_CODE_GEAK_ROOT` overrides both. Relative paths in environment variables,
like YAML paths, are resolved from the directory containing the configuration
file. Example:

```bash
GEAK_HOME=/desired/path/GEAK multi-tune --config configs/mi300x.yaml probe
```

### 5. Start the vLLM model service

Create the model cache before the first startup:

```bash
mkdir -p "$HOME/.cache/huggingface"
```

Create the container for the first time:

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

If the container already exists but is stopped, do not run `docker run --name`
again; start it directly:

```bash
docker start geak-phase1-vllm
```

View the model loading logs:

```bash
docker logs -f geak-phase1-vllm
```

After the service reports that startup is complete, pressing `Ctrl+C` only exits
the log view; it does not stop the container.

### 6. Check the service

```bash
cd ~/Lumen-RL/experiments/multi-tune-agent
source .venv/bin/activate
multi-tune --config configs/mi300x.yaml probe
```

`configured_model_present` in the output must be `true`. When using the
`PYTHONPATH` approach:

```bash
PYTHONPATH=src:. python3 -m multi_tune_agent.cli \
  --config configs/mi300x.yaml probe
```

### 7. Start the interactive CLI

Run in the foreground:

```bash
multi-tune --config configs/mi300x.yaml interactive
```

Run in the background with tmux:

```bash
cd ~/Lumen-RL/experiments/multi-tune-agent
tmux new-session -d -s multi-tune-cli \
  '.venv/bin/multi-tune --config configs/mi300x.yaml interactive'
tmux attach -t multi-tune-cli
```

You can also load the Shell functions provided by the project, then start or
reconnect with one command:

```bash
cd ~/Lumen-RL/experiments/multi-tune-agent
source env.sh
lumen-code
```

`source env.sh` adds `src` and the project root to `PYTHONPATH` without
duplicates, and sets `GEAK_HOME`, `LUMEN_CODE_GEAK_ROOT`,
`GEAK_CONTAINER_NAME`, the configuration path, and the project path.
`lumen-code` starts the existing but stopped `geak-phase1-vllm` container,
creates or reuses the `multi-tune-cli` tmux session, enables mouse mode, and
attaches to it. Helper commands:

```bash
lumen-code-status
lumen-code-stop
kill-lumen-code
```

`lumen-code-stop` and `kill-lumen-code` are equivalent. They stop only the
CLI's tmux session, not the `geak-phase1-vllm` model container.

`env.sh` also provides the default network variables:

```text
LUMEN_CODE_SERVER_HOST=10.194.134.84
LUMEN_CODE_VLLM_HOST=127.0.0.1
LUMEN_CODE_VLLM_PORT=8000
LUMEN_CODE_TUNNEL_PORT=18000
LUMEN_CODE_BASE_URL=http://127.0.0.1:8000/v1
LUMEN_CODE_REMOTE_BASE_URL=http://10.194.134.84:8000/v1
```

`LUMEN_CODE_BASE_URL` overrides `base_url` in the YAML. Variables can be set
before sourcing the file; for example, to use local port `18000` through an SSH
tunnel:

```bash
export LUMEN_CODE_VLLM_PORT=18000
source env.sh
```

To make these commands available automatically in every new terminal, add the
following line to `~/.bashrc`:

```bash
source ~/Lumen-RL/experiments/multi-tune-agent/env.sh
```

If `tmux attach` returns `no sessions`, the CLI has exited; run
`tmux new-session` again. If it reports `no server running`, create a session
first instead of running `tmux source-file`.

To enable mouse scrolling, add this to `~/.tmux.conf`:

```text
set -g mouse on
```

A new tmux server reads the configuration automatically. For an existing
server, run:

```bash
tmux set-option -g mouse on
```

Without a mouse, press `Ctrl+b`, release it, and then press `[` to enter scroll
mode; press `q` to exit.

### 8. CLI options

`interactive` supports:

1. running an existing case from `cases/examples_cases.yaml`;
2. generating a GEMM task from a natural-language request containing the GPU,
   `M`, `N`, and `K`, with an explicit choice of FP16, FP8 A8W8, or native MXFP4;
3. using an existing custom GEAK task directory;
4. resuming the latest checkpoint with `[c] continue` after a failure or interruption.

Natural-language task creation first calls the configured vLLM model to identify
the operator, GPU, format, M/N/K, and language. It then validates numeric
evidence and ranges in the structured result and asks the user to confirm.
When the model is unavailable, it falls back to local parsing. The model may not
invent dimensions absent from the original request, nor may it invent a format
or language without evidence in the original text (the defaults are FP16 and
Triton, respectively).

The automatic GEMM path uses an immutable template registry to select and copy
the trusted seed from `examples/tasks/gemm`, `gemm_fp8`, or `gemm_mxfp4`. It
generates a shape-specific harness under `examples/tasks/` and registers the
task in `cases/examples_cases.yaml`. MI300/MI308 map to gfx942, while
MI350/MI355 map to gfx950. FP8 currently supports only gfx942, and MXFP4
supports only gfx950; the latter remains native AITER with no emulation
fallback. Other operators still require an existing trusted harness. A custom
directory must contain `config.yaml`, `COMMANDMENT.md`, or
`unittest.py + meta.json`.

`[c] continue` copies the latest candidate workspace into a new run, preserves
the original frozen baseline, and passes the latest failure result to the Agent.
It does not restore the original model KV cache; it continues from persisted
code and test evidence.

### 9. Non-interactive execution

```bash
multi-tune --config configs/mi300x.yaml run \
  --case dense-gemm-fp16 \
  --request "Optimize this GEMM shape for MI308X without correctness regressions." \
  --stream
```

Omitting `--case` runs every case in the catalog sequentially. Engineers within
a single case generate candidates concurrently, but GPU evaluation remains
serialized by the GEAK lock.

### 10. Common statuses and errors

- `read_file/write_file/state ok=True`: the tool operation succeeded;
- `evaluate ok=False`: a compile, correctness, or performance command failed;
  the specific traceback is in `trajectory.jsonl`;
- GPU `Memory access fault`: the candidate kernel usually performed an
  out-of-bounds access;
- `run_shell_command ok=False`: GEAK intentionally prohibits arbitrary shell
  commands; the only allowed actions are
  `list_files/read_file/read_candidate/write_file/evaluate/state`;
- HTTP 400 context error: the model input plus reserved output exceeds
  `--max-model-len`;
- `[c]` is not shown: there is no resumable incomplete run, or that checkpoint
  has already been continued.

Permanent case targets are stored in `cases/examples_cases.yaml`. Generated
local tasks are placed under `examples/tasks/` and ignored by Git. The Agent
never modifies the original task directly.

## Automatically generate missing kernel templates

The interactive menu option `[2] generate kernel task` supports any operator.
After the user describes the operator, GPU, shape, format/dtype, and
quantization scale contract, the system proceeds in this order:

1. Exactly match a bundled template or a local template that has passed GPU
   validation. On a match, create the case directly without calling the template
   generation model.
2. If no template exists, first have vLLM directly generate an isolated
   `kernel.py` and an independent `task_runner.py`.
3. Only if static validation of the direct draft fails, search `$AITER_HOME`
   read-only for public wrappers, tests, references, benchmarks, and architecture
   configurations, then have the model repair the entire draft.
4. Run compile, correctness, and performance sequentially through the GEAK GPU lock.
5. After all three pass, promote the result to a trusted local template and
   register the case. The default `bootstrap_auto_promote: true` automatically
   continues normal optimization; setting it to `false` requires confirmation
   at each step.

### Decision priorities

This order defines priorities that the control plane must enforce; it is not
advice for the Agent:

1. **P0: Contract and hardware hard gate.** First validate the operator, shape,
   dtype/format, scale granularity, block size, and target architecture.
   Incompatible combinations such as native MXFP on gfx942 are rejected
   immediately before task creation and never enter model generation or GPU execution.
2. **P1: Reuse trusted implementations.** Prefer immutable canonical templates
   in the repository, followed by local templates that passed the GPU gate,
   matched by the complete `contract_hash`. On a match, begin optimization
   directly without regenerating, searching AITER, or consuming validation
   resources again.
3. **P2: Direct LLM generation and testing.** When no trusted template exists,
   assume by default that the model may already know how to implement the kernel,
   and have it generate a complete draft directly. If static checks pass, proceed
   directly to GPU compile/correctness/performance; do not search the codebase
   first merely to find supporting evidence.
4. **P3: AITER evidence-assisted repair.** Search AITER only when the direct
   draft fails static safety checks. The search is a read-only recovery path
   ranked by operator/format/scale/architecture hard gates and confidence. Give
   the model only evidence such as public wrappers, independent references,
   tests, and benchmark/config files for repair; do not directly execute or copy
   unknown implementations.
5. **P4: Trust and promotion.** Static checks only establish that the draft does
   not obviously bypass the harness. It must still pass an independent GPU gate
   in compile → correctness → performance order. Write it to the local trusted
   registry and case catalog only when `bootstrap_auto_promote` permits it. Any
   failed draft retains only diagnostic information and never becomes an
   optimizable baseline.

The core principle is: **reuse what is known and trusted; if the model can write
it, have it write and test it directly; use AITER-assisted repair only when the
model's implementation is incorrect. Search cannot replace an independent
oracle and GPU measurement.**

The system does not ask again for a format or scale granularity explicitly
stated in the request. It prompts once for all missing fields. For example:

```text
MI308X FP8 GEMM M=16 N=4096 K=4096,
activation per-token scale, weight per-channel scale, FP16 output
```

Local file locations:

- unvalidated draft: `examples/tasks/.generated/<contract-hash>/`;
- validation failure diagnostics: `.failed-<contract-hash>*` in the same directory;
- template that passed the GPU gate: `examples/tasks/generated/<contract-hash>/`;
- local template index: `examples/tasks/generated/templates.yaml`.

These runtime-generated templates are not included in the wheel. `metadata.json`
stores the complete contract, model provenance, optional AITER artifact hash,
and GPU gate evidence. Failed drafts are not written to
`cases/examples_cases.yaml`. Set `bootstrap_enabled: false` to disable automatic
generation. `AITER_HOME`, `generated_template_root`, and
`bootstrap_min_aiter_score` can be adjusted in the configuration or environment.

Native OCP MXFP8/MXFP6/MXFP4 depends on gfx950's CDNA4 block-scaled MFMA.
MI308/gfx942 requests are rejected before template generation, with a
recommendation to use MI350/MI355 or native gfx942 FP8 A8W8 E4M3FNUZ. The
system never registers an ordinary emulated FP8 implementation as native MXFP.

Passing all three GPU checks only proves that the template satisfies its
independent oracle for the specified inputs. Static validation also prevents the
runner/oracle from becoming writable source, always-true comparisons, skipped
correctness checks, overly broad tolerances, and missing architecture gates.
The template is not registered when an independent trusted oracle cannot be
established.

### Hardware validation status

- gfx942 / GPU 1: FP8 A8W8 compile, four correctness cases, and four
  performance cases passed;
- gfx942 / GPU 1: a model-generated non-GEMM softmax template passed the full GPU gate;
- gfx942: MXFP4 is rejected before task creation with an explicit message that
  only gfx950 is supported;
- gfx950: native MXFP4 compile/correctness/performance still requires validation
  on MI350/MI355 hardware; gfx942 results cannot substitute for this validation.

## Tradeoffs with the GEAK Claude Code CLI / Workflow

This compares two **model integration and control-plane deployment approaches**:
MultiTune calls a persistent OpenAI-compatible vLLM service directly, while
GEAK v4 typically runs a deterministic JS Workflow launched by the Claude Code
CLI. Both use GEAK's workspace, GPU lock, and real measurement capabilities.
GEAK itself also provides deterministic orchestration, concurrent Engineers,
independent validation, and recovery. MultiTune's advantages therefore should
not be characterized as "GEAK is unstable or cannot run concurrently"; they
primarily lie in the following areas.

### Large-scale concurrency

- **A more direct model-side path.** MultiTune's roles use one Python scheduler
  to send requests directly to a persistent vLLM endpoint, without making the
  Claude Code CLI lifecycle, permission interactions, and local tool loading
  dependencies of every optimization run. The server can perform continuous
  batching. As the number of Agents grows, they can more easily share model
  weights and KV-cache capacity, with centralized concurrency limits and
  backpressure. GEAK Workflow sub-agents do not necessarily map one-to-one to
  CLI processes, so the advantage is the service-oriented path and centralized
  scheduling, not simply "starting N fewer processes."
- **Candidates are isolated by design.** Each Engineer forks an independent
  workspace from the current best session. Model analysis and source editing can
  run concurrently, while GPU compile/correctness/performance remains serialized
  by the GEAK lock to prevent concurrent benchmarks from contaminating one
  another. Model and GPU concurrency can be configured separately.
- **Better suited as a training or scheduling backend.** Requests, tool calls,
  logprobs, rewards, and latency are structured data. There is no need to parse
  CLI text or terminal state, which simplifies later integration with queues,
  multi-tenant scheduling, RL rollouts, and batch experiments.

The current implementation is not yet a complete distributed scheduler:
`run_all` still runs cases sequentially, a single `base_url` is a model-service
single point of failure, and evaluation on the same GPU is intentionally
serialized. Scaling to large multi-machine runs also requires a global task
queue, per-GPU locks, an endpoint pool, rate limiting, and failure retries. GEAK
already supports launching independent Agents per kernel and concurrent lanes
in multi-GPU bake-offs; it is currently more mature in these existing
capabilities.

### Stability and recoverability

- **The model does not control execution flow.** Rounds, fan-out, candidate
  gates, promotion, stopping, and final validation are determined by the Python
  state machine. The model is responsible only for structured judgments and
  restricted source modifications. Incorrect Agent conclusions cannot override
  deterministic correctness/performance results.
- **A smaller variable runtime surface.** The persistent HTTP model service does
  not depend on interactive CLI sessions, permission confirmations, local
  plugins, or terminal state. The model sees only one restricted `geak` tool; it
  cannot run arbitrary shell commands or modify the runner, oracle, metadata, or
  configuration.
- **Replayable evidence.** The baseline, workspace lineage, every command result,
  model messages, rewards, and checkpoints are persisted to the trajectory.
  After an interruption, recovery starts from the code and frozen baseline
  instead of depending on a surviving CLI process or model KV cache.
- **An independent trust boundary for generated templates.** Unvalidated drafts
  are separated from the trusted registry. They do not enter the normal
  optimization catalog until the static validator, architecture gate, and GPU
  gate all pass.

The tradeoff is that MultiTune's main Python process is currently also a single
point of failure, and `requests`/vLLM endpoint failures affect the same batch of
runs. GEAK's independent CLI/Workflow processes provide stronger process-level
failure isolation, mature hang guards, a Web/Profiling tool ecosystem, and
end-to-end serving workflows. In addition, the restricted tool improves safety,
but trusted Docker commands in a task still have execution capability and are
not equivalent to a complete system-level sandbox.

### Extensibility and maintainability

- **Replaceable model backend.** `ModelBackend` is a small interface. Any local
  or remote model compatible with chat completions and tool-calling can replace
  it, without binding to a Claude Code CLI version, account system, or
  permission model.
- **Operators are decoupled from orchestration.** `TaskSpec`, contract metadata,
  the template registry, static validator, and GEAK adapter do not depend on
  specific roles. Adding an operator primarily means supplying trusted
  harness/contract/knowledge mappings, without rewriting the Agent loop.
- **Experimentable strategies.** Director/TechLead/Engineer/Verifier/Integrator
  roles, concurrency, rewards, candidate thresholds, and stopping conditions
  are explicit modules or configuration, making them suitable for ablations,
  A/B testing, and training-data collection.
- **Upstream GEAK remains unmodified.** MultiTune uses GEAK as materialization,
  locking, and evaluation infrastructure through the `geak_utils` adapter.
  Upgrading the control plane does not require forking upstream GEAK.

GEAK's advantage is broader coverage: its existing end-to-end serving
optimization, profile/Amdahl routing, multi-backend/language bake-offs, Web
research, and learned-knowledge workflows are more complete than MultiTune's.
MultiTune is currently better suited to a "controllable, observable, trainable
kernel-candidate generation and validation service," while the GEAK
CLI/Workflow is better suited to "turnkey, tool-rich single-machine or
end-to-end autonomous optimization." The long-term direction is not to replace
GEAK, but to retain its trusted execution layer and add MultiTune's
service-oriented scheduling, structured trajectories, and training interfaces
on top.

## Add new kernel types and template cases

Templates are not created per shape. For the same operator, input/output layout,
numerical semantics, quantization scale contract, and target architecture, a new
case only needs to reuse the template and provide new parameters. A new trusted
template is required only for a new kernel type or when any of those contracts
changes. For example, ordinary FP8 A8W8 shapes reuse `gemm_fp8`; a different
scale granularity or output type requires another template. Deleted templates
are not rebuilt automatically; the task factory explicitly reports a missing
trusted seed.

### 1. Template directory

Provide at least the following under `examples/tasks/<template_name>/`:

```text
examples/tasks/<template_name>/
├── config.yaml
├── kernel.py
└── scripts/
    └── task_runner.py
```

- `kernel.py`: the only kernel implementation or tuning configuration the Agent
  is allowed to modify;
- `scripts/task_runner.py`: the compile, correctness, and performance harness,
  which the Agent cannot modify;
- `config.yaml`: declares writable source, target functions, and the three types
  of validation commands;
- `metadata.json`: optional; recommended for complex quantization formats,
  packing, scales, or architecture restrictions. See `gemm_mxfp4` for an example.

Minimal `config.yaml` example:

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

Do not add the runner, oracle, configuration, or metadata to `source_file_path`;
otherwise, the Agent could obtain false gains by modifying the validation
criteria.

### 2. Harness requirements

`task_runner.py` must accept three independent modes:

```bash
python3 scripts/task_runner.py compile
python3 scripts/task_runner.py correctness
python3 scripts/task_runner.py performance
```

The harness must meet these requirements:

1. Correctness uses an independent trusted oracle, fixed random seeds, and
   explicit tolerances.
2. Quantized operators compute the reference from the quantized values and
   scales actually passed to the kernel, not from the original pre-quantization
   tensors.
3. After warmup, performance measures only the kernel hot path and excludes
   input generation, quantization, and reference computation.
4. Each shape emits `Perf: <latency_ms> ms (<case_id>)` and may write
   `build/performance_report.json`.
5. A compile, correctness, or performance failure must return a nonzero status.
6. An unsupported architecture must be rejected before importing the JIT backend
   or allocating large tensors. Software-emulated results must not be presented
   as native kernel performance.

References:

- `examples/tasks/gemm`: FP16 Triton template;
- `examples/tasks/gemm_fp8`: gfx942 FP8 A8W8 with per-token/channel scales;
- `examples/tasks/gemm_mxfp4`: native gfx950 MXFP4 and architecture rejection path;
- `examples/tasks/fused_attention`, `grouped_gemm`: non-dense-GEMM templates.

### 3. Integrate with MultiTune

After adding a template, integrate it in this order:

1. Add a stable case ID, case type, template path, and optimization target to
   `cases/examples_cases.yaml`.
2. For a new GEMM format, register the template directory, case type, supported
   architectures, backend, and input/scale/output contract in
   `geak_utils/templates.py`.
3. Add a new operator case type to `_SUPPORTED_CASE_TYPES` in `geak_tool.py`
   and to `_CASE_TYPES` in `cli.py`.
4. Map the corresponding `GEAK/perf_knowledge/operators/<operator>/` knowledge
   directory in `_CASE_KNOWLEDGE` in `agents.py`.
5. Allowlist the trusted template in `.gitignore`, and add its wheel data files
   to `pyproject.toml` and `setup.py`.
6. Add parser, catalog, architecture-gate, task-generation, and flow tests, then
   run compile, correctness, and performance separately on a supported GPU.

Example catalog entry:

```yaml
- id: my-kernel-case
  type: my_kernel_type
  kernel_path: ../examples/tasks/my_kernel_template
  direction: Optimize supported shapes without changing numerical semantics.
```

When adding only a shape supported by an existing template, these steps do not
need to be repeated. Use the task factory to generate a shape-specific
`task_spec.json` and catalog entry.

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

`final_kernel_performance` in `summary.json` explicitly records baseline/final
geomean latency, per-case latency, and speedup. The final kernel speed is marked
valid only after compile, correctness, and performance measurement pass.

The HTTP backend records completion logprobs but intentionally does not invent
token IDs for tool observations. A later trainer-specific ETL can tokenize the
stored messages and mask tool-observation tokens from policy loss.

