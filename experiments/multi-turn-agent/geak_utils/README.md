# GEAK Utils

`geak-utils` provides orchestration-neutral adapters for loading kernel tasks,
materializing isolated workspaces, constraining source edits, and running
deterministic correctness and performance evaluation. It does not include an
agent loop, model client, CLI, prompt builder, or trajectory logger.

An upstream GEAK checkout is required at runtime. Its root is passed explicitly
to `KernelSandbox`; the adapter uses:

- `kernel_workflow/scripts/materialize_workspace.sh`
- `kernel_workflow/scripts/gpu_lock.sh`

## Install

```bash
cd /path/to/Lumen-RL/experiments/multi-tune-agent
python -m pip install -e .
```

The package is installed as part of MultiTune. For development, install
`-e ".[test]"`. Run the integrated tests from the project root:

```bash
PYTHONPATH=src:. pytest
```

## Public API

```python
from geak_utils import (
    AiterQuery,
    CommandResult,
    EvaluationResult,
    KernelSandbox,
    SandboxError,
    TaskSpec,
    discover_aiter,
    get_gemm_template,
    validate_generated_template,
    validate_template_target,
    load_tasks,
)

tasks = load_tasks("cases/examples_cases.yaml")
sandbox = KernelSandbox(
    upstream_root="/path/to/GEAK",
    run_root="/path/to/runs",
    gpu_ids="1",
)
workspace = sandbox.prepare(tasks[0], "/path/to/runs/task-0")
baseline = sandbox.establish_baseline(repeats=3)
result = sandbox.evaluate()
```

`get_gemm_template()` exposes immutable FP16, FP8 A8W8, and MXFP4 descriptors,
including template directory, MultiTune case type, backend, architecture set,
and operand/scale/output contracts. `validate_template_target()` maps
MI300/MI308 to gfx942 and MI350/MI355 to gfx950 and rejects unsupported or
unknown targets before task materialization.

`discover_aiter()` builds a read-only filesystem index of an AITER checkout and
scores public wrappers together with matching correctness tests, independent
references, benchmarks, configs, scale contracts, and architecture gates. It
never imports or executes AITER. `validate_generated_template()` statically
checks an untrusted generated template before GPU execution, including its file
boundary, metadata/provenance, runner modes, architecture gate, oracle
independence, tolerances, and performance output contract.

Locally generated templates are recorded through
`VerifiedTemplateRecord`, `register_verified_template()`, and
`find_verified_template()`. The registry accepts only templates whose
`metadata.json` contains a matching contract hash and `trust.trusted: true`.
The model-driven direct generation and GPU promotion workflow remains in
`multi_tune_agent.template_bootstrap`; `geak_utils` itself does not call an
LLM.

`TaskSpec.task_type` is normalized but intentionally unrestricted. Consumers
decide which task families they support. Compatibility properties
`case_id`/`case_type` and constructor aliases are retained for existing
orchestration integrations.

## Task contract

Each task directory must contain one of:

- `config.yaml`, `config.yml`, or `config.json` with
  `correctness_command` and `performance_command` and an optional
  `compile_command`;
- `unittest.py` together with `meta.json`; or
- `COMMANDMENT.md` exposing `CORRECTNESS` and `FULL_BENCHMARK` commands and an
  optional `COMPILE` command.

Commands run from the materialized workspace through GEAK's GPU lock.
Performance output may use `Perf: <milliseconds> ms (<case-id>)` or
`GEAK_RESULT_LATENCY_MS=<milliseconds> case=<case-id>`. The optional
`build/performance_report.json` format is also supported. Baselines use the
per-case median over repeated runs; candidate speedup is reported over matching
cases as both geometric and arithmetic means.

Only discovered kernel source files are writable. Test files, task runners,
scripts, build outputs, caches, and paths outside the workspace are blocked.
The original task directory is never edited.

## Included examples and harness setup

The parent project's `cases/examples_cases.yaml` catalog points to FP16 dense
GEMM, gfx942 FP8 A8W8, gfx950 native MXFP4, fused attention, and grouped GEMM
fixtures under `../examples/tasks` relative to this directory. Their commands
prefer `GEAK_CONTAINER_NAME` and otherwise
use the existing compatible `geak-phase1-vllm` service. They default to GPU 1
inside the container and honor `HIP_VISIBLE_DEVICES`.

One ROCm-capable setup matching the checked-in commands is:

```bash
docker run -d --name geak-phase1-vllm \
  --device /dev/kfd --device /dev/dri --group-add video \
  --ipc=host --network=host --security-opt seccomp=unconfined \
  -e HIP_VISIBLE_DEVICES=0 -e VLLM_ROCM_USE_AITER=0 \
  -v /home/danyzhan:/home/danyzhan \
  -v /home/danyzhan/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai-rocm:v0.15.0 \
  Qwen/Qwen3-Coder-Next-FP8 \
  --served-model-name Qwen/Qwen3-Coder-Next \
  --max-model-len 32000 --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.90 --kv-cache-dtype fp8 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --port 8000
```

Mount the upstream checkout, task directories, and run directories at the same
absolute paths inside the container because the example commands use the
workspace's host path as their container working directory. Ensure the service
GPU and benchmark GPU do not overlap.
