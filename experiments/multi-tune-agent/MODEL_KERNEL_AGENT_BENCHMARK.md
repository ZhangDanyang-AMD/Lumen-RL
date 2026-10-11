# Kernel Agent Benchmark: From-Scratch Kernel Writing

## Goal

Evaluate a model's ability to **write GPU kernels from scratch** for AMD MI300X (gfx942) using the GEAK tool-calling agent flow. The model reads a task spec (operator type, shapes, correctness oracle) and writes a Triton kernel that achieves at least **90% of AITER SOTA performance**.

This measures real kernel engineering capability — not just optimizing existing code, but understanding operator semantics, memory access patterns, and hardware-specific tuning from zero.

## Benchmark Design

### Task Structure

Each of the 100 tasks provides:
- `scripts/task_runner.py` — defines function signature, test inputs, correctness oracle, and performance measurement
- `config.yaml` — compile/correctness/performance commands
- `kernel.py` — **cleared to skeleton** (model writes from scratch)

The model does NOT see the AITER reference kernel. It must figure out the implementation from the task_runner interface.

### Agent Flow (GEAK ToolAgentLoop)

```
Model receives: operator type + baseline perf target
  → read_file("scripts/task_runner.py")     # understand interface
  → write_file("kernel.py", implementation) # write Triton kernel
  → evaluate                                 # compile → correctness → performance
  → iterate up to 50 turns                   # fix errors, improve performance
```

### Success Metric

| Metric | Threshold | Meaning |
|--------|-----------|---------|
| **speedup_geomean** | >= 0.9x | Kernel reaches 90% of AITER performance |
| **Fast@0.9** | count | Number of tasks achieving the target |
| per_operator_speedup | breakdown | Performance by operator type |

Compile rate and correct rate are diagnostic (prerequisites, not goals).

### Models Tested

| Label | Path | Prompt Format |
|-------|------|---------------|
| base | `Qwen/Qwen3-Coder-30B-A3B-Instruct` | Generic (natural language) |
| sft2 | `qwen3-coder-full2000-2epoch-merged` | SFT-compatible JSON |
| sft4 | `qwen3-coder-full2000-4epoch-merged` | SFT-compatible JSON |

SFT models get prompts in `geak_kernel_sft_v1` format (contract + direction + baseline). Base model gets generic natural language prompt. Controlled by `sft_compat` flag.

## Prerequisites

```bash
# Inside geak-sft-32k container
cd /home/danyzhan/Lumen-RL/experiments/multi-tune-agent
export PYTHONPATH="src:.:scripts:/home/danyzhan/Lumen-RL/experiments/sandbox:${PYTHONPATH}"
export EVAL_GPU_IDS="2,3,4,5,6,7"
```

### vLLM Setup (Critical)

```bash
ROCR_VISIBLE_DEVICES=0 python3 -m vllm.entrypoints.openai.api_server \
    --model <MODEL_PATH> \
    --served-model-name <LABEL> \
    --tensor-parallel-size 1 --max-model-len 262144 \
    --enforce-eager --dtype bfloat16 --trust-remote-code \
    --gpu-memory-utilization 0.95 --port 8000 \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder
```

**`--tool-call-parser qwen3_coder` is required.** Using `hermes` or other parsers silently breaks tool calling.

### Dataset

- Location: `/home/danyzhan/held-out-benchmark-aiter/`
- 100 tasks, 8 operators: gemm(25), rms_norm(19), mha(14), paged_attention(12), rope_kv_cache(12), sampling(12), mla(6)
- Baselines: AITER kernels with frozen optimal num_warps/num_stages configs

## Running the Benchmark

### Step 1: Verify Baselines

```bash
python3 -u scripts/run_aiter_benchmark.py --verify-only
```

### Step 2: Run Benchmark

```bash
python3 -u scripts/run_aiter_benchmark_geak.py \
    --model-label sft2 --model-name sft2 --max-turns 50
```

The benchmark:
1. Creates a session per task with AITER kernel as baseline (for timing reference)
2. **Clears kernel.py** to a skeleton (model starts from zero)
3. Runs 50-turn agent loop (no early stopping)
4. Records best speedup achieved

### Step 3: Multi-Model Comparison

The pipeline script runs SFT-2e → Base → SFT-4e sequentially, switching vLLM between each.

## Baseline Stability

### Autotune Freeze (Critical)

Raw `@triton.autotune` causes **6x measurement variance** across runs (different num_warps winners due to GPU noise). Fixed by:

1. **One-time config sweep** (`freeze_autotune.py`): try num_warps=[2,4,8,16] x num_stages=[1,2] for each kernel
2. **Lock winner** into kernel as single-config `@triton.autotune`
3. **Stable measurement**: `tuned_bench.py` with 200 warmup + 5-round median

Results stored in `receipts/autotune_freeze.json`. Each kernel has its optimal config locked.

### Measurement Stability

| Component | Setting | Purpose |
|-----------|---------|---------|
| tuned_bench.py | 200 warmup + 5x median | JIT warmup + stable timing |
| baseline_repeats | 3 | Multiple baseline measurements |
| Frozen configs | Single best num_warps | Eliminates autotune variance |

## Code-Level Validation

### validate_kernel (in sandbox.write_file)

Blocks two cheating patterns discovered during earlier benchmarks:

1. **PyTorch Replacement**: Model writes empty `@triton.jit` shell but does computation in PyTorch wrapper (`torch.argsort`, `torch.gather`, etc.)
2. **PyTorch in Wrapper**: Model keeps `@triton.jit` tag but moves core computation to Python

```python
# kernel_validator.py
PYTORCH_OPS = ["torch.argsort", "torch.sort(", "torch.gather", "torch.cumsum"]

def validate_kernel(content, original_path=None):
    if "@triton.jit" not in content:
        return False, "Must contain @triton.jit"
    # Check wrapper (after last @triton.jit) for PyTorch ops
    wrapper = content.split("@triton.jit")[-1]
    for op in PYTORCH_OPS:
        if op in wrapper:
            return False, f"Wrapper must not use {op}"
    return True, content
```

### Diff Patch Support

SFT models may output unified diffs instead of complete files. `write_file` auto-detects and applies:
1. `patch --fuzz=3` (standard)
2. `patch --fuzz=99` (aggressive, ignore context)
3. `robust_patch` (custom context-matching)
4. Direct write (fallback)

## Lessons Learned

### Infrastructure

**1. Tool Call Parser Must Be `qwen3_coder`**

Wrong parser → model outputs text instead of function calls → agent loop terminates after 1 turn with tools=0. This is a silent failure — no error, just poor results.

**2. Large Kernel Timeout**

MHA kernel (2863 lines, 99KB) causes vLLM inference timeout when 6 workers compete for GPU with 25K+ token prompts each. Fix: `timeout=3600s`. Complex operators (mha, mla, paged_attention, rope_kv_cache) need 100 turns instead of 50.

**3. Zombie GPU Processes**

When vLLM crashes or is killed, GPU memory may not be freed (zombie processes with parent PID 1 inside container). New vLLM instances fail to allocate. Fix: `docker restart <container>`.

**4. Session Cleanup Between Runs**

Old sessions from previous benchmark runs pollute audit results and can cause session ID conflicts. Always `rm -rf receipts/geak-flow-*` before a new run.

### Measurement

**5. Autotune Instability**

Adding `@triton.autotune` with multiple configs causes 6x measurement variance. Each subprocess re-runs the config search, picking different winners due to GPU noise. Fix: run config sweep once (`freeze_autotune.py`), lock the winner into a single-config autotune.

**6. Speedup Noise (Code-Identical Kernels)**

Even with frozen configs, code-identical kernels can report 1.3-1.7x "speedup" due to: baseline measured during session creation (with GPU contention from parallel workers) vs candidate measured later (different contention). Fix: for any reported speedup, verify the code actually changed (diff > 3 lines of real code). Re-benchmark baseline and candidate independently on idle GPU.

**7. Baseline Must Be Re-measured**

`establish_baseline` runs once at session creation and caches `baseline_ms`. If that measurement was noisy (GPU contention), all subsequent speedup calculations inherit the error. For production benchmarks, re-measure both baseline and candidate on the same GPU in sequence.

### Model Behavior

**8. PyTorch Wrapper Cheating**

Both Base and SFT models independently discover the "empty Triton shell + PyTorch wrapper" trick — keep `@triton.jit` to pass checks but move computation to `torch.argsort/gather/cumsum` in the wrapper function. Prompt rules alone are insufficient. Code-level validation (`kernel_validator.py` in `sandbox.write_file`) is essential.

**9. Autotune Deletion Cheating**

Models delete `@triton.autotune` and hardcode a fixed config. Since autotune search has overhead, the fixed-config version appears faster. This is NOT a real optimization — it's exploiting measurement artifacts. Fix: freeze baseline configs so autotune deletion has no advantage.

**10. SFT Narrows Capability**

In optimization benchmarks, SFT-2e specialized on rms_norm (6.98x) but degraded on attention operators (0x on mha/mla/paged_attn where Base got 2.88x). SFT training on single-turn `input→patch` format may cause catastrophic forgetting on complex multi-step reasoning tasks. Recommendation: use RL (agentic GRPO) instead of more SFT for capability breadth.

**11. SFT Format Compatibility**

SFT training data (`geak_kernel_sft_v1`) is single-turn, but GEAK uses multi-turn tool-calling. This works because Qwen3-Coder has native tool-calling. SFT adds domain knowledge (kernel optimization) that transfers. The SFT data has 5 task types: `cold_start` (15%), `direction_conditioned` (45%), `error_recovery` (15%), `profile_guided` (15%), `regression_balance` (10%). Error recovery format is useful for multi-turn feedback.

### Benchmark Design

**12. From-Scratch > Optimization**

Optimization benchmarks suffer from measurement noise (identical code reports different speedups) and cheating (autotune deletion, PyTorch replacement). From-scratch writing eliminates these — either the model can write a working kernel or it can't. Target: reach 90% of AITER performance.

**13. Simple vs Complex Operators**

Simple operators (gemm, rms_norm, sampling) are solvable in 50 turns. Complex operators (mha, mla, paged_attention, rope_kv_cache) need 100 turns due to: larger kernels, more complex semantics, multi-kernel coordination.

**14. Prompt Format Per Model**

SFT models should see SFT training format (JSON with contract/direction/baseline). Base model should see generic natural language. Using the wrong format degrades performance. Controlled by `sft_compat` config flag.

## Cheating Prevention (validate_kernel)

Models discover creative ways to bypass Triton kernel requirements. All blocked at `sandbox.write_file` level:

| Pattern | Detection | Example |
|---------|-----------|---------|
| Empty Triton shell + PyTorch wrapper | Check `torch.matmul/mm/bmm/einsum/argsort/sort/gather/cumsum` in wrapper | `@triton.jit def k(): pass` + `def gemm(): return torch.matmul(a,b)` |
| No @triton.jit | Check `@triton.jit` presence | Pure PyTorch implementation |
| Empty kernel body | Check for `tl.load`/`tl.store` in JIT function | `@triton.jit def k(): pass` |
| Method-form PyTorch | Check `.mm(`, `.matmul(`, `.bmm(` | `result = a.matmul(b.T)` |

Blocked ops list: `torch.matmul`, `torch.mm`, `torch.bmm`, `torch.einsum`, `torch.argsort`, `torch.sort`, `torch.gather`, `torch.cumsum`, `torch.nn.functional`, `.mm(`, `.matmul(`, `.bmm(`
