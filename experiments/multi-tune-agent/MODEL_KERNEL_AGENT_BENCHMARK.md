# Model Kernel Agent Benchmark

Reproducible benchmark for evaluating kernel optimization models on AMD MI300X (gfx942) using the GEAK tool-calling agent flow.

## Overview

This benchmark measures a model ability to optimize Triton GPU kernels through multi-turn tool-calling interactions. It uses the same GEAK agent infrastructure that generates SFT training data, ensuring training/eval consistency.

**Key design choices:**
- **GEAK ToolAgentLoop** — model uses `read_file`, `write_file`, `evaluate` tools (matches real deployment)
- **Autotuned baselines** — both original and optimized kernels are benchmarked with `@triton.autotune` over `num_warps`/`num_stages`, ensuring speedups reflect genuine algorithmic improvements
- **AITER SOTA kernels** — baselines are verbatim AITER library kernels, representing production-grade optimization
- **100 held-out tasks** — 8 operator types, 100 unique production shapes, zero overlap with training data

## Prerequisites

```bash
# Inside the geak-sft-32k container
cd /home/danyzhan/Lumen-RL/experiments/multi-tune-agent
export PYTHONPATH="src:.:scripts:/home/danyzhan/Lumen-RL/experiments/sandbox:${PYTHONPATH}"
export EVAL_GPU_IDS="2,3,4,5,6,7"  # GPUs for kernel eval (GPU 0 = vLLM inference)
```

### Dataset
- Location: `/home/danyzhan/held-out-benchmark-aiter/`
- HuggingFace: `Zhangdanyang/agent-phase1-held-out-aiter`
- 100 tasks, 8 operators: gemm(24), rms_norm(18), mha(14), paged_attention(12), rope_kv_cache(12), sampling(12), mla(6), fused_moe(2)

### Models
| Label | Path | Description |
|-------|------|-------------|
| base  | `Qwen/Qwen3-Coder-30B-A3B-Instruct` | Pre-trained base model |
| sft2  | `outputs/qwen3-coder-full2000-2epoch-merged` | SFT 2-epoch |
| sft4  | `outputs/qwen3-coder-full2000-4epoch-merged` | SFT 4-epoch |

Model paths are under `/home/danyzhan/Lumen/experiments/GEAK-agent-coder/outputs/`.

## Step 1: Start vLLM with Tool Calling

```bash
ROCR_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 \
python3 -m vllm.entrypoints.openai.api_server \
    --model <MODEL_PATH> \
    --served-model-name <LABEL> \
    --tensor-parallel-size 1 \
    --max-model-len 262144 \
    --enforce-eager --dtype bfloat16 \
    --trust-remote-code \
    --gpu-memory-utilization 0.95 \
    --port 8000 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder
```

**Critical**: `--tool-call-parser qwen3_coder` is required for Qwen3-Coder models. Using `hermes` or other parsers will silently break tool calling (model responds with text instead of function calls).

Wait for health check:
```bash
curl -s http://localhost:8000/health  # returns 200 when ready
```

## Step 2: Verify Baselines (once)

Ensures all 100 kernels compile, pass correctness, and have stable autotuned performance numbers.

```bash
python3 -u scripts/run_aiter_benchmark.py --verify-only
```

Output: `held-out-benchmark-aiter/receipts/aiter_baseline_results.json`

All 100 kernels have `@triton.autotune` for fair comparison:
- **gemm**: Full config search (BLOCK_SIZE_M/N/K + num_warps)
- **mha**: AITER native autotune
- **Others**: num_warps/num_stages sweep via `tuned_bench.py`

## Step 3: Run GEAK Agent Benchmark

```bash
python3 -u scripts/run_aiter_benchmark_geak.py \
    --model-label sft2 \
    --model-name sft2 \
    --max-turns 50
```

This runs the full GEAK tool-calling agent loop for each task:
1. Model receives task description with available tools
2. Model calls `read_file("kernel.py")` to read the kernel
3. Model analyzes and calls `write_file("kernel.py", optimized_code)` 
4. Model calls `evaluate` to test compile → correctness → performance
5. If not satisfied, model iterates (up to `--max-turns`)

### Output
- Results: `held-out-benchmark-aiter/receipts/geak-benchmark-<label>/results.json`
- Summary: `held-out-benchmark-aiter/receipts/geak-benchmark-<label>/summary.json`
- Sessions: `held-out-benchmark-aiter/receipts/geak-flow-<label>/sessions/` (when `keep_sessions=True`)
  - Each session has `workspace/kernel.py` (optimized) and `workspace/metadata.json`
  - Original kernel: `held-out-benchmark-aiter/artifacts/kernel/<task_id>/initial/kernel.py`

### Metrics
| Metric | Description |
|--------|-------------|
| compile_rate | % of tasks where model produced a compiling kernel |
| correct_rate | % of tasks passing correctness check |
| fast12_rate | % of tasks achieving ≥1.2x speedup over autotuned baseline |
| speedup_geomean | Geometric mean of speedups (correct tasks only) |
| avg_turns | Average tool-calling turns used |
| avg_tools | Average tool calls per task |

## Step 4: Multi-Model Comparison

Run for all three models sequentially (switch vLLM between runs):

```bash
# SFT-2e
# (vLLM already running with sft2)
python3 -u scripts/run_aiter_benchmark_geak.py --model-label sft2 --model-name sft2 --max-turns 50

# Switch to Base
pkill -f vllm; sleep 15
# Start vLLM with Qwen/Qwen3-Coder-30B-A3B-Instruct --served-model-name base
python3 -u scripts/run_aiter_benchmark_geak.py --model-label base --model-name base --max-turns 50

# Switch to SFT-4e
pkill -f vllm; sleep 15
# Start vLLM with qwen3-coder-full2000-4epoch-merged --served-model-name sft4
python3 -u scripts/run_aiter_benchmark_geak.py --model-label sft4 --model-name sft4 --max-turns 50
```

## Step 5: Extract Kernel Pairs

After benchmark completes, extract original/optimized kernel pairs:

```python
import json, shutil
from pathlib import Path

HELD_OUT = Path("/home/danyzhan/held-out-benchmark-aiter")
model = "sft2"
results = json.load(open(HELD_OUT / f"receipts/geak-benchmark-{model}/results.json"))
sessions = HELD_OUT / f"receipts/geak-flow-{model}/sessions"

out = HELD_OUT / f"receipts/kernel-pairs-{model}"
out.mkdir(exist_ok=True)

for sess_dir in sessions.iterdir():
    meta = json.loads((sess_dir / "workspace/metadata.json").read_text())
    tid = meta["task_id"]
    pair_dir = out / tid
    pair_dir.mkdir(exist_ok=True)
    shutil.copy2(HELD_OUT / f"artifacts/kernel/{tid}/initial/kernel.py", pair_dir / "original.py")
    opt = sess_dir / "workspace/kernel.py"
    if opt.exists():
        shutil.copy2(opt, pair_dir / "optimized.py")
```

## Architecture Notes

### Why GEAK ToolAgentLoop (not simple chat)?
- SFT model retains Qwen3-Coder native tool-calling capability
- SFT kernel optimization knowledge applies in tool-calling context
- Matches real deployment: model reads → writes → evaluates → iterates
- Previous simple chat benchmark had patch-application failures (LLM-generated diffs have inaccurate line numbers)

### Why autotuned baselines?
- Without autotune, model can achieve "speedup" by just changing BLOCK_SIZE or num_warps
- With autotune, both baseline and candidate run at their best config
- Speedup reflects genuine algorithmic improvements (memory access patterns, tiling strategies, instruction-level optimization)

### Autotune implementation
- `tuned_bench.py` in each task scripts/ dir — uses `triton.testing.do_bench` with 50 warmup + 100 reps
- gemm: `@triton.autotune` over BLOCK_SIZE_M/N/K + num_warps (12 configs)
- Other operators: `@triton.autotune` over num_warps/num_stages (6 configs)
- mha: AITER native autotune preserved

### Context compression
For the simple chat benchmark (`run_aiter_benchmark_fast.py`), context compression is applied after `KEEP_RECENT=4` turns — old turns are summarized to prevent context overflow on large kernels (mha=2863 lines/99KB).

## Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| `turns=0 tools=0 FAIL` | Wrong tool-call-parser | Use `--tool-call-parser qwen3_coder` |
| `ModuleNotFoundError: tuned_bench` | Missing in workspace | Copy `tuned_bench.py` to each task scripts/ dir |
| `baseline benchmark failed` | task_runner error | Run `python3 scripts/task_runner.py compile` in workspace to debug |
| MHA timeout | 99KB kernel prompt too large | Increase `--timeout` or use context compression |
| Container crash | GPU OOM from zombie processes | `docker restart geak-sft-32k` |
| All 0x speedup | Patch apply failure | Check eval pipeline fallbacks (fuzzy→GNU patch→robust→reconstruct) |

## File Layout

```
multi-tune-agent/
├── scripts/
│   ├── run_aiter_benchmark_geak.py    # GEAK tool-calling benchmark (primary)
│   ├── run_aiter_benchmark_fast.py    # Simple chat benchmark (fast, less accurate)
│   ├── run_aiter_benchmark.py         # Baseline verification
│   ├── run_aiter_multi_benchmark.py   # Multi-model comparison
│   ├── grpo_multiturn.py              # Multi-turn RL training
│   ├── grpo_agentic.py                # Agentic RL training
│   ├── grpo_train.py                  # Basic RL training
│   ├── fuzzy_patch.py                 # Diff patch application
│   ├── eval_checkpoint.py             # Checkpoint evaluation
│   ├── build_aiter_held_out.py        # Dataset building
│   ├── build_rl_dataset.py            # RL dataset building
│   ├── upload_hf.py                   # HuggingFace upload
│   └── wandb_sync.py                  # W&B sync
├── src/multi_tune_agent/
│   ├── runtime.py                     # ToolAgentLoop, OpenAIModelBackend
│   ├── geak_tool.py                   # GEAKStatefulTool, GEAKToolEnvironment
│   ├── config.py                      # MultiTuneConfig
│   └── ...
├── geak_utils/
│   ├── sandbox.py                     # Kernel workspace + benchmark sandbox
│   └── ...
└── held-out-benchmark-aiter/          # (at /home/danyzhan/)
    ├── artifacts/kernel/*/initial/
    │   ├── kernel.py                  # AITER baseline (with autotune)
    │   ├── kernel_clean.py            # Without autotune (for reference)
    │   ├── config.yaml                # Compile/correctness/perf commands
    │   ├── metadata.json              # Task metadata + AITER provenance
    │   └── scripts/
    │       ├── task_runner.py         # Compile/correctness/perf harness
    │       └── tuned_bench.py         # Autotune benchmark utility
    ├── receipts/
    │   ├── aiter_baseline_results.json
    │   ├── geak-benchmark-<model>/    # Results per model
    │   └── geak-flow-<model>/sessions/  # Agent sessions (keep_sessions=True)
    └── tasks/kernel.jsonl             # Task manifest
```



## Lessons Learned (Pitfalls)

### 1. Tool Call Parser Must Be `qwen3_coder`

vLLM supports multiple tool-call parsers. Using the wrong parser causes the model to output text instead of function calls — the agent loop terminates after 1 turn with tools=0.

```bash
# WRONG - model will not use tools
--tool-call-parser hermes

# CORRECT for Qwen3-Coder models
--tool-call-parser qwen3_coder
```

### 2. Model Replaces Triton with PyTorch

Without explicit constraints, the SFT model sometimes replaces the Triton kernel with a PyTorch reference implementation (e.g., torch.argsort instead of Triton bitonic sort). This passes correctness but is not a valid kernel optimization.

**Fix**: System prompt includes: "You MUST keep the kernel as a Triton kernel (@triton.jit). Do NOT replace it with PyTorch ops."

### 3. Model Deletes @triton.autotune

The model deletes @triton.autotune and hardcodes a specific config. If that config happens to be better than what autotune searched, it looks like "speedup" but is actually just config tuning.

**Fix**: System prompt includes: "You MUST preserve @triton.autotune if present." The baseline is already autotuned, so config-only changes should not produce speedup.

### 4. Autotune Baseline is Critical for Fair Comparison

Without autotuning both baseline and candidate:
- Model adds @triton.autotune to a kernel that had fixed config -> "2.4x speedup" that is pure config tuning
- Model changes BLOCK_SIZE -> appears as algorithmic improvement but is just config

**Fix**: All 100 baseline kernels have @triton.autotune (gemm: full config search, others: num_warps/num_stages sweep).

### 5. SFT Training Format != GEAK Tool-Calling Format

SFT training data is single-turn input->patch format (geak_kernel_sft_v1), not multi-turn tool-calling. However, this does NOT break tool-calling because Qwen3-Coder has native tool-calling ability from pre-training. SFT adds domain knowledge (kernel optimization) that transfers to the tool-calling context.

### 6. LLM-Generated Diffs Have Inaccurate Line Numbers

Models produce unified diffs with wrong hunk headers, causing fuzzy_patch, GNU patch --fuzz=3, and _reconstruct_from_diff to all fail.

**Fix**: Use the GEAK tool-calling flow where the model writes complete files via write_file tool, avoiding diff application entirely.

### 7. MHA Kernel Baseline Timeout

MHA kernel (2863 lines, 99KB) needs >120s for baseline establishment (compile + autotune warmup). Default command_timeout=120 causes turns=0 tools=0 FAIL.

**Fix**: Set command_timeout=300 in the benchmark config.

### 8. Zombie GPU Processes After Container Restart

When vLLM crashes or is killed, GPU memory may not be freed (zombie processes with parent PID 1). New vLLM instances cannot allocate GPU memory.

**Fix**: docker restart container to clean up all processes and free GPU memory.

### 9. Agent Terminates Early Without Speedup

When the model generates text without tool calls, the ToolAgentLoop terminates immediately. Tasks may stop at turn 17 or 26 without reaching 50 turns.

**Fix**: Modified runtime.py — if no tool calls AND no speedup achieved yet, inject a user feedback message to continue. If speedup already achieved, allow early termination.

### 10. tuned_bench.py Import Path in GEAK Sandbox

GEAK sandbox copies task files to a new workspace directory. tuned_bench.py must be in each task scripts/ directory (not just the root), because the workspace has a different absolute path.

**Fix**: Copy tuned_bench.py into every task scripts/ directory, use local path import.
