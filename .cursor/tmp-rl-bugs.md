# Temporary Training Bug Notes

This file lives at `.cursor/tmp-rl-bugs.md` relative to the `Lumen-RL` repo root. Read the whole file at the start of every new LumenRL training debug session.

Use it to keep track of possible bugs found during testing. Do not treat any entry here as proof. Re-check against the current reference diff and current repro before acting.

Treat any fresh return to the same debugging problem as a new debug session:

- a new chat or agent session
- a new day or work block
- returning after unrelated work
- starting a new round of debug after prior tests finished

Write back only meaningful tests or experiments that change confidence in a hypothesis, such as a new repro, written diff, backend toggle, layerwise compare, kernel test, or targeted integration check. Do not log every identical rerun. Do log negative results that rule a suspicion out.

**Bug origin guide** (see `lumenrl-debug` skill):
- Lumen bugs → fix in `third_party/Lumen/`, push to `dev/RL` branch
- ATOM bugs → fix in `third_party/ATOM/`, push to `Lumen/RL` branch
- AITER bugs → fix in `third_party/aiter/`, push to `lumen/triton_kernels` branch
- MORI bugs → fix in `third_party/mori/`, push to `sdma-new` branch
- LumenRL bugs → fix in `lumenrl/`

## Open

### [2026-04-18 checkpoint-callback-deadlock] — RESOLVED
- Symptom: Training hangs immediately after step 0 completes. Rank 0 at 172-198% CPU (running),
  ranks 1-7 sleeping on `futex_wait_queue_me`, vLLM idle on `pipe_read`. Only GPU[3] at 100%.
  Log file has zero output after step 0 callback line. Confirmed NOT log buffering — process
  was genuinely stuck for 1.5+ hours. Occurs on every restart.
- Origin: LumenRL (`lumenrl/trainer/rl_trainer.py` + `lumenrl/trainer/callbacks.py`)
- Root cause: `on_step_end` callbacks were called only from rank 0 (`if self._rank == 0`),
  but `CheckpointCallback.on_step_end()` calls `get_model_state_dict()` and
  `get_optimizer_state_dict()` with `full_state_dict=True` — these are FSDP2 collective
  operations requiring ALL ranks to participate. Rank 0 entered all-gather, ranks 1-7 waited
  at a different barrier → deadlock. Triggered at step 0 because `0 % save_interval(25) == 0`.
- Fix: Changed `rl_trainer.py` to call `on_step_end` on ALL ranks (removed the `if self._rank == 0`
  guard). Added `if trainer._rank != 0: return` guards to `LoggingCallback`, `WandbCallback`,
  and `EvalCallback` to prevent duplicate logging. `CheckpointCallback` already had correct
  structure (FSDP2 ops on all ranks, save on rank 0 only, barrier at end).
- Status: resolved

### [2026-04-18 sync-trainer-rocm-page-fault]
- Symptom: ATOM/vLLM subprocess crashes with `Memory access fault by GPU node-2 ... Write access
  to a read-only page`. Non-deterministic: crashes between step 3 and step 21 depending on run.
- Origin: ROCm driver (`hipMemSetAccess` / VMM bug on MI350X)
- Root cause: `hipMemSetAccess` is broken on ROCm 7.12 / MI350X (rocm-systems#2516). Any memory
  management operation that touches the VMM layer can trigger "write to read-only page" faults.
  This includes vLLM model loading, `LLM.sleep()`/`wake_up()`, subprocess creation/destruction,
  and even normal vLLM generation after enough cumulative GPU memory churn.
- Investigation (3 approaches tried):
  1. **Persistent vLLM (no sleep/wake)**: Keeps vLLM alive across steps. Survived 10-21 steps
     before page fault during normal generation. Memory pressure from colocation (vLLM + FSDP2
     on rank 0's GPU: ~116 GiB vs ~68 GiB on other ranks) accelerates the fault.
  2. **In-process sleep/wake** (`LLM.sleep(level=1)` / `wake_up()`): Works for ~3 cycles, then
     `wake_up()` triggers page fault via `hipMemSetAccess` internally. Required clearing
     `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` from the vLLM subprocess environment
     (incompatible with `CuMemAllocator`).
  3. **Dedicated GPU** (7 GPUs training + GPU 7 for vLLM): Page fault still occurs ON the
     isolated vLLM GPU, proving the bug is not caused by memory pressure from colocation.
     The bug is fundamental to `hipMemSetAccess` on this hardware/driver.
- After any page fault, GPU state is corrupted and subsequent runs crash immediately.
  Container restart (`docker restart`) clears the corruption.
- Current mitigation: Auto-restart loop in `run_1a_bf16_sync.sh` with `save_steps=2` for
  frequent checkpointing. Each crash loses at most 1 step. Training makes steady progress
  across restarts.
- Code kept: `sleep_inprocess()`/`wake_inprocess()` in `AtomEngine` (useful when ROCm driver
  is fixed), `gpu_id` config (useful for dedicated-GPU setups), auto-restart in launch script.
- Status: MITIGATED — auto-restart + frequent checkpoints. Awaiting ROCm driver fix from AMD.

### [2026-04-18 async-trainer-oom-step11]
- Symptom: OOM at step 11 during `old_log_probs` forward pass. `lm_head` linear tried to allocate
  60.70 GiB, only 30 GiB free on all 8 GPUs. PyTorch had 136.84 GiB alloc + 79.93 GiB reserved.
- Origin: LumenRL (`lumenrl/trainer/async_trainer.py:627`)
- Root cause: `old_log_probs` were computed on the full merged batch (4 prompts × 16 gens = 64 seqs)
  in a single forward pass. The `lm_head` output tensor `[64, seq_len, 151936]` × 2B ≈ 60+ GiB.
  The training step already used mini-batches, but the old_log_probs forward did not.
- Fix: Chunked the `old_log_probs` forward pass using `mini_bs` (=1) as chunk size, processing
  one sequence at a time through the model, then concatenating results. Same computation, bounded memory.
- Evidence: v1 run OOM'd at step 11 on all 8 GPUs. v2 run with fix passed step 13 (ongoing).
- Status: CONFIRMED FIXED — v2 passed step 11 (the previous OOM point) and is now at step 13 with stable memory usage.

### [2026-04-17 async-trainer-nan-grads]
- Symptom: 226-232 of 399 parameters consistently have NaN gradients during backward pass.
  Forward pass and loss are valid. Model weights remain clean. Happens on every step.
- Origin: LumenRL (`lumenrl/trainer/async_trainer.py`) + FSDP2 backward
- Config: AsyncRLTrainer staged pipeline, 8×MI350X, require_batches=4, BF16.
- Observations:
  1. Forward pass works: log-probs are valid (nan=0), ratios are reasonable (0.1-7.0).
  2. Loss is valid: 0.025, 0.079, -0.043 etc.
  3. After backward, ~226/399 params have NaN grads. Always the same set of params.
  4. Zeroing NaN grads before clip_grad_norm and optimizer.step preserves model health.
  5. grad_norm after zeroing is ~10-22 (reasonable for BF16 FSDP2).
  6. Model weights stay clean (nan_params=0/399) across all steps.
- Workaround: `torch.where(grad.isnan(), zeros, grad)` before optimizer step. Training proceeds.
- Possible root cause:
  a) FSDP2 gradient all-reduce producing NaN for specific sharded params on ROCm.
  b) `_fused_token_log_probs` backward through `float().logsumexp()` chain.
  c) Padding tokens in merged DataProto creating extreme logit values during backward.
- Next check: Identify which specific params have NaN grads (embedding? attention? MLP?).
  Compare with sync RLTrainer to see if same issue exists there.
- Status: open (workaround applied)

### [2026-04-17 atom-nan-loss]
- Symptom: loss_pg=nan, loss_total=nan on ALL steps (0-15+). Model not learning.
  Accuracy fluctuates 2-20% (base model stochastic) but no upward trend.
- Origin: LumenRL (`lumenrl/trainer/rl_trainer.py` — training step)
- Config: ATOM/vLLM gen, 8 prompts × 16 gens, seq=8K, BF16. DAPO with kl_coeff=0.
- Observations:
  1. `timing/ref_s=~26μs` — reference log-prob compute is effectively skipped/instant.
     This means `old_logprobs` may be zeros or NaN → `exp(logp - old_logp)` = Inf/NaN → loss=nan.
  2. vLLM generation works (responses are real, non-empty, ~960 tok avg).
  3. Rewards are non-trivial (acc ~11% avg, reward ~-0.77 avg). Advantage should be non-zero.
  4. `_sync_weights_to_atom()` is disabled (FSDP2 lazy storage workaround) — but NaN loss is
     independent of weight sync (loss was NaN from step 0 before any weight update).
  5. FSDP2 forward/backward completes (train_s ~56s). No OOM.
- Possible bugs:
  a) `_compute_ref_logprobs()` returns zeros or skips computation entirely.
  b) `old_logprobs` from rollout are not correctly computed for the generated tokens.
  c) Token-level log-probs from `_fused_token_log_probs()` produce NaN for padding tokens,
     and masking doesn't exclude them before the ratio computation.
- Next check: Add debug prints in training step to inspect old_logprobs, logprobs, advantages,
  and ratio tensors for NaN/Inf. Check `_compute_ref_logprobs()` code path.
- Status: open

### [2026-04-17 atom-weight-sync-disabled]
- Symptom: ATOM always generates with initial (untrained) model weights.
- Origin: LumenRL (`lumenrl/trainer/rl_trainer.py` — `_sync_weights_to_atom()`)
- Root cause: `safetensors.torch.save_file()` crashes with `RuntimeError: Attempted to access
  the data pointer on an invalid python storage` when given FSDP2 state_dict with lazy storage.
- Impact: Model doesn't use trained weights for generation. Acceptable for initial pipeline
  validation, but must be fixed for actual training.
- Next check: Implement FSDP2 `get_model_state_dict()` with `StateDictOptions(full_state_dict=True)`
  to consolidate shards before serializing.
- Status: open

### [2026-04-10 fp8-training-alignment-repro]
- Goal: Demonstrate FP8 training aligns with BF16 using LumenRL (Lumen + ATOM) on 8× MI350X.
- Models: Qwen3-8B-Base (dense), Qwen3-30B-A3B (MoE)
- Data: dapo-math-17k (train), aime-2024 (val) at `/dev/shm/data/`
- Experiment structure:
  - Exp 1 (8B dense): 1A BF16, 1B FP8 rollout+TIS, 1C FP8 no TIS, 1D FP8 E2E blockwise,
    1E MXFP8 sweep, 1F per-tensor delayed, 4A-4D FP8 attention sweep, 5A-5B optimizer sweep
  - Exp 2 (30B MoE Base): 2A-2E same FP8 variants
  - Exp 3 (30B MoE Instruct): 3A-3E same FP8 variants
- Scripts: `examples/DAPO/` (run_1a, run_1b, run_1d, run_1f + common.sh)
- Status: open — 1A baseline validated at 275 steps in prior Lumen/VERL experiments
  (val_acc ~28-29% at step 245, matching reference). Need to reproduce with LumenRL native trainer.

## Ruled Out

### [2026-04-18 sync-trainer-nan-logprobs] — RESOLVED
- Was: ALL `old_log_probs` NaN (151360/151360 elements) from step 0 in sync RLTrainer.
  Caused 399/399 params NaN grads, grad_norm=0, loss=0 — model not learning.
- Root cause: `_load_hf_model` in `fsdp_backend.py` loaded model with `from_config` on meta
  device for non-rank-0 processes, then materialized with `torch.empty()` (uninitialized memory).
  FSDP2's `fully_shard` does NOT broadcast weights from rank 0 — it assumes each rank already
  has correct parameters. Result: non-rank-0 shards contained garbage → all-gathered parameters
  mixed real weights with garbage → NaN logits everywhere.
- Fix: All ranks now load full model via `from_pretrained` (model on `/dev/shm` is fast).
  Eliminates meta-device materialization entirely.
- Additional fix: Switched `_fused_token_log_probs` from `float().logsumexp()` to per-row
  `F.log_softmax` (matching VERL's `logprobs_from_logits_v2` bf16 path). More memory efficient,
  avoids float32 promotion of full vocab dimension.
- Evidence: Step 0 changed from `nan=151360` to `nan=0`, grad_norm from 0 to 0.297607.
- Status: resolved

Move disproved suspicions here instead of deleting them.

### [2026-04-17 atom-nan-loss] — RESOLVED
- Moved from Open. Was: loss_pg=nan, loss_total=nan on ALL steps.
- Root causes found and fixed in previous session:
  1. `asymmetric_clip_loss` misinterpreted epsilon values as raw ratio bounds → extreme clamping.
  2. `_optimizer.step()` called per mini-batch instead of per global step → stale old_log_probs.
  3. `NaN * 0 = NaN` propagation through masked padding tokens.
  4. `_fused_token_log_probs` using `F.cross_entropy` produced NaN on bf16 logits.
- Fixes applied:
  1. Fixed `asymmetric_clip_loss` to use `1-clip_low` and `1+clip_high`.
  2. Implemented proper gradient accumulation (zero_grad before loop, step after loop).
  3. Added `torch.where(mask, pg, zeros)` to prevent NaN propagation.
  4. Reverted `_fused_token_log_probs` to row-chunked `float().logsumexp()`.
  5. Added optimizer CPU offload between rollout and training phases for OOM fix.
- Status: resolved

### [2026-04-17 atom-colocate-oom] — RESOLVED
- Was: vLLM OOM on startup after training (37 GiB free vs 50 GiB requested).
- Root cause: FSDP2 optimizer states (~215 GiB across 8 GPUs) not freed before vLLM spawn.
- Fix: Added `_offload_optimizer_to_cpu()` before ATOM rollout and `_reload_optimizer_to_gpu()`
  after. This frees ~30+ GiB per GPU for vLLM startup.
- Status: resolved

## Resolved — Lumen (third_party/Lumen)

These bugs were found and fixed in the Lumen submodule. They apply to LumenRL because
LumenRL uses Lumen for FP8 quantized training via `third_party/Lumen`.

### [2026-04-09 fp8pm-fsdp2-memory-regression]
- Symptom: FSDP2 with FP8ParamManager uses MORE GPU memory than BF16 when offloading.
- Origin: Lumen (`lumen/quantize/fp8_params.py`)
- Root cause: `_FP8LinearFunc.forward` calls `ctx.save_for_backward(fp8_weight, scale)` which pins
  the allgathered parameter tensor. FSDP2 `param_offload` can't reclaim allgathered memory.
- Fix: `ctx.save_for_backward(fp8_weight.clone(), scale)` — clone creates independent FP8 copy.
- Evidence (Qwen 0.5B, 4 GPU, FSDP2):
  | Config | Offload | Peak VRAM | vs BF16 |
  |---|---|---|---|
  | BF16 | Yes | 48.06 GB | baseline |
  | FP8PM (before fix) | Yes | 69.18 GB | +44% regression |
  | FP8PM (after fix) | Yes | 45.50 GB | -5% savings |
  | BF16 | No | 73.49 GB | baseline |
  | FP8PM | No | 54.87 GB | -25% savings |
- Status: resolved in `third_party/Lumen`

### [2026-04-08 fp8pm-dequant-memory-leak]
- Symptom: FP8PM + LoRA (210.4 GB) used more memory than BF16 + LoRA (142.9 GB) on 70B model.
- Origin: Lumen (`lumen/quantize/fp8_params.py`)
- Root cause: `F.linear(input, dequant_weight)` caused autograd to save full BF16 dequant copy per layer.
  All 225 layers' copies (~140 GB for 70B) accumulated until backward.
- Fix: Custom `_FP8LinearFunc(torch.autograd.Function)` that saves only FP8 weight + scalar scale
  in `save_for_backward`. Reconstructs BF16 on-the-fly during backward.
- Evidence: 70B FP8PM+LoRA: 210.43 GB → 73.48 GB (-65%). 8B: 27.22 GB → 13.31 GB.
- Status: resolved in `third_party/Lumen`

### [2026-04-08 fp8pm-lora-compat-fix]
- Symptom: FP8ParamManager + LoRA crashed with `NotImplementedError: "addmm_cuda" not implemented for Float8_e4m3fn`.
- Origin: Lumen (`lumen/config.py`)
- Root cause: PEFT casts LoRA adapter weights to match base layer dtype → FP8 cast on LoRA weights.
- Fix: Post-PEFT fixup in `LumenConfig._apply_lora()` re-casts FP8-typed LoRA params back to BF16.
- Status: resolved in `third_party/Lumen`

### [2026-04-08 fp8-architectural-fixes]
- Symptom: Three FP8 features crashed or had no effect.
- Origin: Lumen (various)
- Fixes:
  1. FP8 Weight Cache: wired `store_weights_fp8()` into LumenConfig. +3.3% throughput.
  2. FP8 Activation Store: extended `_apply_pre_quant` to `nn.Linear`. No measurable effect.
  3. AITER kernel crash: added `weight.contiguous()` + `TORCH_CHECK`. Crash resolved, +1.4% throughput.
- Status: resolved in `third_party/Lumen`

### [2026-04-09 megatron-fp8pm-on-the-fly]
- Symptom: Lumen FP8ParamManager can't target Megatron-Core `ColumnParallelLinear`/`RowParallelLinear`.
- Origin: Lumen (`lumen/quantize/fp8_params.py`)
- Fix: Extended `FP8ParamManager` with `_get_quantizable_types()` + `_FP8MegatronLinearFunc`
  (on-the-fly quantization, keeps params BF16 for distributed optimizer compatibility).
- Evidence: Megatron+SGLang FP8PM: 50.06 GB (-29% vs BF16 70.52 GB). Throughput regression noted.
- Also: Custom `MegatronLoraAdapter` in `lumen/models/lora_adapter.py` for TP-aware LoRA.
- Status: resolved in `third_party/Lumen`

## Resolved — ROCm Platform Knowledge

Hard-won ROCm lessons from prior experiments. These inform LumenRL's ATOM integration and
config defaults on MI300X/MI350X.

### [2026-04-10 rocm-gpu-memory-management]
- Problem: On ROCm, inference engine `sleep()`/`wake_up()` behavior differs from CUDA.
  CUDA uses CuMemAllocator (VMM-based, fine-grained memory pool management).
  ROCm requires explicit weight offload to CPU + KV cache deallocation.
- Key findings:
  1. `hipMemSetAccess` silently fails on ROCm 7.12/MI350X → CuMemAllocator broken on ROCm
  2. ROCm sleep/wake is all-or-nothing: sleep frees everything, wake restores everything
  3. Between sleep and wake, no dynamic memory management — PyTorch default allocator only
  4. `expandable_segments=True` is safe on ROCm (no CuMemAllocator pool conflict)
- Memory budget rules for 8× MI350X (252 GB usable per GPU):
  | gpu_memory_util | KV cache | Remaining for training | Stability |
  |---|---|---|---|
  | 0.3 | ~75 GiB | ~177 GiB | Survived 1021K seqlen (best) |
  | 0.6 | ~151 GiB | ~101 GiB | OOM at 593K seqlen without dyn sampling |
  | 0.6 + dyn sampling | ~151 GiB | ~101 GiB | Survived 1507K, OOM'd at step 61 |
  | 0.85 + dyn sampling | ~209 GiB | ~43 GiB | Ran 34+ steps, not crash-tested to limit |
  | 0.9 | ~227 GiB | ~25 GiB | OOM at step 79 (training activations) |
- Implication for LumenRL + ATOM: ATOM must implement equivalent sleep/wake that offloads
  model weights to CPU and deallocates KV cache between rollout and training phases.
  `AtomRolloutWorker` should manage this lifecycle.
- Status: resolved (knowledge captured)

### [2026-04-10 rocm-training-oom-patterns]
- Three distinct OOM failure modes on ROCm during RL training:
  1. **Rollout OOM** (generation phase): KV cache too small for long concurrent sequences.
     Fix: reduce `gpu_memory_utilization` or `max_num_seqs`. Trades throughput for stability.
  2. **Training OOM** (compute_log_prob/actor_step): FSDP activations exceed remaining GPU memory
     after inference engine overhead. Fix: ensure sleep() properly frees all inference memory.
     Dynamic sampling (`filter_groups`) reduces effective batch load by filtering uninformative prompts.
  3. **Host OOM** (checkpointing): FSDP offload + `/dev/shm` checkpoints fill host RAM.
     Fix: use disk-backed checkpoints, limit `max_ckpt_to_keep`.
- Dynamic batching (`use_dynamic_bsz=True`, `ppo_max_token_len_per_gpu=21504`) is essential
  for handling variable sequence lengths. Without it, OOM at step 55.
- `max_num_seqs` reduction (256 → 64) is the most effective single fix for rollout OOM.
  Cost: -40% throughput (810s vs 488s per step).
- Status: resolved (knowledge captured)

### [2026-04-13 rocm-sleep-wake-implementation]
- Working ROCm sleep/wake implementation for inference engines on MI350X:
  - `_rocm_sleep(level)`: offload model weights to CPU pinned memory, free KV cache tensors,
    `gc.collect()` + `torch.cuda.empty_cache()`. Level 2 additionally saves named_buffers.
  - `_rocm_wake_up()`: restore weights from CPU copies, full KV cache reinitialization
    (not just tensors — must reinit attention backend binding + input batch).
  - Must be called explicitly by the trainer between rollout→training phase transitions.
    CUDA's CuMemAllocator handles this transparently; ROCm cannot.
- ATOM integration note: `AtomRolloutWorker` should call `atom_engine.sleep()` after generation
  and `atom_engine.wake_up()` before next generation. The ATOM engine must implement these
  methods following the same pattern (weight offload + KV dealloc).
- Status: resolved (implementation reference available)

### [2026-04-15 rocm-cumem-allocator-investigation]
- Investigated using native CuMemAllocator on ROCm MI350X (vLLM 0.16 era).
- Three approaches all FAILED:
  1. C++ per-chunk `hipMemSetAccess` patch → "Memory access fault: Write access to read-only page"
  2. Python monkey-patch sleep/wake → same memory fault at allocation time
  3. Full bypass + manual offload → works, but child process architecture blocks cross-process memory free
- Root cause: `hipMemSetAccess` silently fails on ROCm 7.12/MI350X (rocm-systems#2516).
  All VMM-based memory allocation is broken. Must use standard PyTorch allocator.
- Implication for ATOM: ATOM MUST NOT use CuMemAllocator or VMM-based allocation on ROCm.
  Use standard `torch.cuda.memory` allocation with explicit offload/restore for sleep/wake.
- Status: resolved (approach abandoned; standard allocator is the only viable path on ROCm)

## Resolved — FP8 Memory Benchmarks

Reference numbers from Lumen FP8 experiments. Useful for validating LumenRL FP8 integration.

### [2026-04-08 fp8-memory-savings-reference]
- Llama-3.1-8B, single MI300X, 3 steps, process-isolated:
  | Config | Peak Alloc (MB) | vs BF16 |
  |---|---|---|
  | BF16 baseline (AdamW) | 76,861 | baseline |
  | FP8ParamManager | 28,909 | -62.4% peak |
  | FP8PM + 8-bit Adam | 27,923 | -63.7% peak |
  | FP8 Attention (dpa) | 76,860 | -0.0% peak |
- Key: FP8PM saves memory from BOTH weights (bf16→fp8: -7 GB) AND optimizer states
  (AdamW only tracks non-FP8 params: 30,633→2,005 MB). FP8 Attention alone saves nothing
  with gradient checkpointing.
- FSDP2 results (Qwen 0.5B, 4 GPU):
  | Config | Peak Mem/GPU | Elapsed | vs BF16 |
  |---|---|---|---|
  | BF16 full | 34.57 GB | 122.7s | baseline |
  | FP8 Linear only | 38.85 GB | 1279.2s | +12% mem, 10.4× slower |
  | FP8 Linear + FP8 Attn (dpa) | 30.92 GB | 1586.5s | -11% mem, 12.9× slower |
  | FP8 Linear + FP8 Attn + Act Store | 30.89 GB | 1581.3s | -11% mem, 12.9× slower |
- Status: resolved (reference numbers)

## Entry Template

```markdown
### [YYYY-MM-DD session-name]
- Symptom:
- Origin: Lumen | ATOM | AITER | MORI | LumenRL
- Possible bug:
- Evidence so far:
- Next check:
- Status: open | ruled out | resolved
```
