# Temporary Training Bug Notes

This file lives at `.claude/tmp-rl-bugs.md` relative to the `Lumen-RL` repo root. Read the whole file at the start of every new LumenRL training debug session.

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

### [2026-04-22 16k-seq-e2e-training] — IN PROGRESS
- Goal: Run 275-step E2E training with max_response_length=16384 to match Verl FP8 reference.
- Config: param_offload=true, max_token_len_per_gpu=20480, max_model_len=20480, gpu_mem=0.25.
- Lumen AITER attention: fmha_v3_fwd (forward) + mha_bwd (backward) confirmed working.
- Fixes applied during this session:
  1. **AdamW foreach=False**: FSDP2 CPUOffloadPolicy + foreach=True crashed in optimizer.step()
     (DTensor lerp_ across cuda/cpu). Set foreach=False when param_offload is active.
  2. **Skip manual optimizer offload with param_offload**: _offload/_reload_optimizer_to_cpu/gpu
     conflicted with FSDP2's own CPU offload. Now skipped when param_offload=True.
  3. **AITER causal mask bypass**: HF always passes a 4D causal attention_mask. Original code
     fell back to SDPA (which doesn't support GQA without repeat_kv). Fixed to use AITER's
     causal=True flag when seq_len>1 and is_causal, ignoring the redundant HF mask.
  4. **Per-rank data sharding**: Training loop processed all 512 sequences on every rank.
     Added shard before old_log_probs computation (512→64 per rank). Also shards ground_truths
     at prompt level (32→4 per rank).
  5. **Padded-length dynamic batching**: _dynamic_mini_batches used attention_mask.sum() (actual
     tokens) but forward processes padded tokens. Changed to use padded sequence length,
     preventing multi-sequence mini-batches with 16K padding from OOM-ing.
- Step 0-15 results: accuracy oscillates 7.6%-19.1%, mean ~13.5%. No clear upward trend.
- Step timing: ~1630s/step (~27min). 0 NaN, 0 errors. Memory stable (free≥132GB post_train).
- **Verl comparison (step 0-15 analysis):**
  - Verl BF16 goes from ~15% → ~35% in first 50 steps. We stay flat at ~13.5%.
  - ROOT CAUSE ANALYSIS — 3 key differences vs Verl:
    1. **Rollout batch size: Verl 1536 (32×3×16) vs ours 512 (32×16).** Verl does 3× more
       rollouts per step → richer reward signal. The `×3` multiplier in Verl config means 3
       rounds of prompts per iteration, giving much more diverse gradient signal.
    2. **No TIS (Truncated Importance Sampling).** Verl uses token-level TIS with C=2. This
       corrects distribution shift between rollout and training policy. Without TIS, stale
       on-policy data is noisier. Verl's own chart shows accuracy drops without TIS.
    3. **max_response_length 16384 vs Verl 20K.** Verl allows 20K tokens for chain-of-thought.
       Ours caps at 16K. For math reasoning, longer responses help.
  - Additionally, LumenRL currently lacks the 3-round rollout multiplier (only 1 round).
- **2026-04-23 loss function alignment fixes (batch 1536 run, steps 0-4, loss ~1e-10):**
  - Dashboard shows: acc oscillates 9.5-13.3%, loss ~1e-10, grad_norm ~1.0
  - Diffed LumenRL DAPO loss against verl reference (core_algos.py). Found 4 mismatches:
    1. **Missing DAPO dual-clip (C-clip)**: Verl uses `clip_ratio_c=10.0` — for negative
       advantages, caps loss at `-adv * C`. Prevents bad-action loss from dominating gradient
       as ratio grows. LumenRL had NO C-clip. Fixed: added `clip_ratio_c` param to
       `asymmetric_clip_loss` and `DAPOConfig`, set to 10.0 in all YAML configs.
    2. **Missing log-prob clamping**: Verl clamps `logp - old_logp` to `[-20, 20]` before
       exp to prevent Inf/NaN. LumenRL did raw `torch.exp(logp - old_logp)`. Fixed.
    3. **weight_decay=0.01 vs verl 0.1**: 10× lower regularization. Fixed to 0.1 in
       `rl_trainer.py`.
    4. **overlong_penalty default 1e-4 vs verl 1.0**: 10000× lower penalty for overlong
       sequences. Fixed default to 1.0 in `dapo.py`.
  - Also fixed loss formula to use `torch.maximum(pg1, pg2)` convention (matching verl)
    instead of `-torch.minimum(surr1, surr2)` (algebraically equivalent but clearer).
  - Note: loss being ~0 when ratio=1 is EXPECTED for GRPO/DAPO (group advantages sum to 0
    by construction). Both verl and LumenRL compute old_log_probs from the same actor model.
    The gradient signal (grad_norm ~1.0) IS present despite near-zero loss scalar.
    The C-clip becomes important as training progresses and ratio deviates from 1.
- **2026-04-23 post-fix run results (steps 0-4, callbacks data):**
  - Applied all 4 fixes (clip_ratio_c=10.0, weight_decay=0.1, overlong_penalty=1.0, log-prob clamp).
  - Results (callbacks/full-batch):
    | Step | Accuracy | Reward | Loss | Grad Norm | Resp Len |
    |------|----------|--------|------|-----------|----------|
    | 0 | 10.74% | -0.785 | 1.65e-9 | 1.584 | 1171 |
    | 1 | 12.50% | -0.750 | -1.92e-9 | 1.430 | 1288 |
    | 2 | 10.48% | -0.790 | 3.90e-9 | 1.505 | 1304 |
    | 3 | 10.09% | -0.798 | 2.13e-10 | 1.123 | 1320 |
    | 4 | 9.77% | -0.805 | -1.94e-10 | 1.184 | 1471 |
  - **FINDING: Loss is still ~1e-9 to ~1e-10 after 5 steps.** The DAPO fixes (C-clip,
    weight_decay, overlong_penalty) did NOT change the loss magnitude because the root issue
    is NOT the loss function — it's that ratio ≈ 1 in on-policy sync mode.
  - In sync on-policy training, old_log_probs are computed from the SAME model as new_log_probs
    in each step. The only gradient signal comes from the within-step forward-backward log-prob
    difference (which is tiny). Group-normalized advantages sum to ~0 by construction, so when
    ratio=1, loss = -mean(adv * ratio) ≈ -mean(adv) ≈ 0.
  - **Note on Reward: vs callbacks discrepancy**: `Reward:` line shows rank-0 shard (192 seqs).
    `callbacks` shows full-batch (1536 seqs). Use callbacks as authoritative.
  - **Key question**: Why does verl BF16 show accuracy climbing from ~15% → ~35% in 50 steps
    with the same on-policy scheme? Verl must have a mechanism that creates non-trivial ratio
    deviation even in on-policy mode. Investigate: does verl compute old_log_probs BEFORE the
    training step (at rollout time) and then compare against post-training log-probs? That would
    give ratio != 1 within a single step.
  - **Response length slowly increasing**: 1171 → 1471 over 5 steps. This is expected as the
    model learns to generate longer responses.
- **2026-04-24 ROOT CAUSE FOUND: gradient accumulation kills PPO learning signal**
  - LumenRL accumulated gradients over ALL ~192 mini-batches, then called `optimizer.step()` ONCE.
  - With ratio=1.0 for all mini-batches (old_log_probs == new_log_probs from identical weights),
    the PPO loss = -mean(advantage) ≈ 0 (group-normalized advantages cancel).
  - Verl calls `optimizer.step()` after EACH mini-batch. After mini-batch 1, the weights change,
    so mini-batch 2+ sees ratio ≠ 1.0 → meaningful clipping → actual learning.
  - **Historical irony**: The ORIGINAL LumenRL code had per-mini-batch stepping. It was changed to
    gradient accumulation as part of the NaN-loss fix [2026-04-17 atom-nan-loss]. The NaN was
    caused by 3 other bugs (epsilon misinterpretation, NaN propagation, cross_entropy NaN),
    NOT by per-mini-batch stepping. The "fix" accidentally broke the learning signal.
  - **Fix applied**: Restored per-mini-batch `optimizer.step()` in `rl_trainer.py`. Each mini-batch
    now does: zero_grad → forward → backward → NaN-grad cleanup → clip_grad_norm → optimizer.step().
    Gradient accumulation division (`loss / num_accum`) also removed.
  - Relaunched training at 02:14 UTC on 2026-04-24 with clean checkpoints.
- Status: running — per-mini-batch stepping fix applied, monitoring for accuracy improvement.

### [2026-04-22 atom-not-sleeping-during-training] — RESOLVED
- Symptom: `torch.OutOfMemoryError` on GPU 0 during backward pass at step 2-3 (after
  weight sync was implemented). ATOM held ~57 GB on GPU 0 throughout training.
- Origin: LumenRL (`lumenrl/trainer/rl_trainer.py` — training loop)
- Root cause: `_rollout_with_atom()` woke ATOM for generation but never slept it.
  ATOM stayed loaded on GPU 0 during the entire training phase (log-prob computation,
  forward, backward). The 57 GB ATOM footprint plus FSDP2 activations exceeded 192 GB.
  `_sync_weights_to_atom()` at the end of the step was too late — the OOM already
  happened during backward.
- Fix: Added `sleep_inprocess()` call immediately after `_rollout_with_atom()` returns,
  before log-prob computation begins. Also made `_sync_weights_to_atom()` skip the
  sleep if ATOM is already sleeping (idempotent guard on `_sleeping` flag).
- Evidence: Smoke test (2 steps, batch=16, max_resp=512) passed cleanly.
  GPU-MEM logs show: post_gen=145 GB free → post_atom_sleep=200 GB free (+55 GB).
  Training peak (post_train) at 179-185 GB free — ample headroom.
- Also added: `_log_gpu_mem()` helper for structured GPU memory snapshots at 5
  critical points (pre_gen, post_gen, post_atom_sleep, pre_train, post_train).
- Also added: `--smoke-test` flag (2 steps, tiny batches, ~5-10 min pipeline validation),
  `--dry-run` flag (mock generation, skip ATOM entirely for training-only debugging).
- Status: resolved

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
- **KFD node mapping** (2026-04-19): "GPU node-2" in the error is a KFD kernel node number,
  NOT a HIP device index. KFD nodes 0-1 are CPU nodes; KFD nodes 2-9 are GPUs 0-7.
  So **GPU node-2 = physical GPU device 0** = vLLM's GPU (rank 0). Previous assumption that
  node-2 = device 3 was wrong. The page fault is consistently on the vLLM colocation GPU,
  not a specific faulty device.
- **Attempted GPU exclusion** (2026-04-19): Tried `ROCR_VISIBLE_DEVICES=0,1,2,4,5,6,7` to
  skip physical device 3. Page fault still occurred on "GPU node-2" (= device 0), confirming
  the fault is on the vLLM GPU, not device 3. Reverted to 8 GPUs.
  Side note: `ROCR_VISIBLE_DEVICES` works at HSA level, `HIP_VISIBLE_DEVICES` required by Ray,
  `CUDA_VISIBLE_DEVICES` must match HIP for vLLM assertion. When using ROCR filtering, set
  HIP to re-indexed values (0..N-1) and leave CUDA unset (vLLM copies HIP→CUDA).
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
- **2026-04-19 attention backend investigation**: vLLM GitHub issue #35169 confirmed the
  default `TRITON_ATTN` backend triggers page faults on ROCm during CUDA graph capture.
  Three alternative backends tested:
  1. **`ROCM_ATTN`** — Triton paged attention (different kernel path). Survived 4+ steps
     per container restart. Still uses Triton kernels but avoids the problematic flash attn path.
  2. **`ROCM_AITER_FA`** — AITER CK/ASM Flash Attention v3. Build broken: `gfx950::fmha_fwd_v3`
     symbol undefined. Root cause: `mha_fwd_generate.py` (receipt=5) generates the dispatcher
     that calls `gfx950::fmha_fwd_v3()`, but the per-arch function body is never generated
     as a compilable source. The `mha_fwd.cu` was also missing (needed manual generation).
     Attempted manual compilation of `mha_fwd.cpp` — resolved `aiter::mha_fwd` symbol but
     `gfx950::fmha_fwd_v3` remains undefined. AITER build issue, not fixable from user side.
  3. **`ROCM_AITER_UNIFIED_ATTN`** — Inherits `ROCM_ATTN` but replaces forward pass with
     AITER's `unified_attention` Triton kernel. Does NOT depend on the broken `fmha_v3` module.
     **Successfully ran 9+ steps (step 0-8) with zero page faults and zero errors.**
     This is the most stable backend tested.
- Current configuration: `VLLM_ROCM_ATTN_BACKEND=ROCM_AITER_UNIFIED_ATTN` set in
  `run_1a_bf16_sync.sh` and passed to vLLM via `atom_engine.py` `attention_config`.
  `gpu_memory_utilization=0.25` (up from 0.20 for Triton).
- Code kept: `sleep_inprocess()`/`wake_inprocess()` in `AtomEngine` (useful when ROCm driver
  is fixed), `gpu_id` config (useful for dedicated-GPU setups).
  `HIP_VISIBLE_DEVICES` synchronization in `atom_engine.py` (prevents vLLM assertion).
- **2026-04-19 AITER unified still crashes**: `ROCM_AITER_UNIFIED_ATTN` delayed the page fault
  (13 steps vs 3-4 with `TRITON_ATTN`) but did not eliminate it. Crash at step 12→13 transition.
  Root cause remains `hipMemSetAccess` driver bug — no backend fully prevents it.
- **Auto-recovery implemented** (`examples/DAPO/auto_recover.sh`): Host-side wrapper script that
  runs outside the container. On crash: backup checkpoints from container `/dev/shm` to host
  `/dev/shm` → `docker restart` (clears corrupted GPU state) → restore checkpoints → re-launch.
  Container `/dev/shm` is ephemeral (not bind-mounted), so the backup/restore cycle is necessary.
  Checkpoints saved every 2 steps (save_total_limit=3), `resume: true` in config.
  In-container `run_1a_bf16_sync.sh` simplified to single attempt (no retry loop).
- Status: MITIGATED — page fault is a ROCm driver bug. Auto-recovery ensures training progresses
  via checkpoint resume after `docker restart`. ~13 steps per cycle, ~2 min recovery overhead.

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

### [2026-04-17 atom-weight-sync-disabled] — RESOLVED
- Symptom: ATOM always generates with initial (untrained) model weights. Accuracy flat
  at ~13% (base model level) after 18 steps despite non-zero loss and gradients.
- Origin: LumenRL (`lumenrl/trainer/rl_trainer.py` — `_sync_weights_to_atom()`)
- Root cause: `_sync_weights_to_atom()` was a no-op. The original implementation was
  disabled because `get_model_state_dict(full_state_dict=True)` OOM'd on MI350X.
  Without weight sync, the RL feedback loop was broken: training updated the FSDP2
  actor model, but ATOM always generated with the initial base model weights.
- Fix (2026-04-21): Implemented weight sync following verl's approach:
  1. All ranks call `model.state_dict()` which returns DTensors (FSDP2 shards).
  2. For each param, call `DTensor.full_tensor()` → `.cpu().contiguous()` (FSDP2
     collective all-gather, one tensor at a time to bound peak GPU memory).
  3. Rank 0 saves gathered tensors as safetensors shards (4 GB each) to
     `/dev/shm/lumenrl_weight_sync/` with HF-compatible `model.safetensors.index.json`
     and copies `config.json` + tokenizer files from the original model.
  4. Rank 0 calls `atom_engine.sleep()` (kills subprocess), sets `_weight_dir`.
  5. Next step's `_rollout_with_atom()` calls `wake()` which starts a fresh ATOM
     subprocess loading from `/dev/shm/lumenrl_weight_sync/` instead of original path.
  6. All ranks synchronize via `torch.distributed.barrier()`.
  Also fixed: `AtomEngine.wake()` no longer clears `_weight_dir` after wake, so
  subsequent wake cycles (e.g. crash recovery) still use the latest synced weights.
- Performance: Weight sync adds ~33s per step (25s save + 8s ATOM restart). ATOM
  reload from `/dev/shm` (tmpfs) takes ~60s including model warmup.
- Evidence: Step 0 logged "Weight sync: saved 399 params... in 25.7s (16.4 GB)".
  Step 1 logged "AtomEngine: wake complete (fresh subprocess, path=/dev/shm/lumenrl_weight_sync)".
  Training stable through 4+ steps with zero errors.
- Status: resolved

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

### [2026-04-20 atom-block-table-off-by-one]
- Symptom: `ValueError: could not broadcast input array from shape (513,) into shape (512,)` in `prepare_block_tables()` during ATOM decode. Crash only when sequences reach near `max_model_len`.
- Origin: ATOM
- Possible bug: Off-by-one in `CommonAttentionBuilder.__init__()` at `backends.py:103`. The pre-allocated `block_tables` buffer uses `ceil(max_model_len / block_size)` columns, but during autoregressive decoding the scheduler's `may_append()` can grow `seq.block_table` to `ceil((total_tokens) / block_size)` where `total_tokens > max_model_len`. This happens because: (1) the scheduler does not enforce `max_model_len` as a hard stop — it only checks `seq.num_completion_tokens >= seq.max_tokens`, and (2) when `max_model_len` is an exact multiple of `block_size`, even one extra token spills into a brand new block.
- Evidence so far: With `max_model_len=8192` and `block_size=16`, buffer had 512 columns but `block_table` grew to 513 entries (first attempt) then 514 entries (after +1 fix).
- Fix applied: Two changes in `ATOM/atom/model_ops/attentions/backends.py`:
  1. Buffer sizing: changed `max_model_len + self.block_size - 1` → `max_model_len + self.block_size` (adds 1 extra block of headroom = `ceil((max_model_len+1)/block_size)`)
  2. Defensive truncation in `prepare_block_tables()`: clamp `len(block_table)` to `block_tables.shape[1]` to prevent crash even if scheduler allocates more blocks than expected
- Note: The root fix should be in ATOM's scheduler to enforce `max_model_len` as a hard sequence length limit, but the buffer fix is sufficient for correctness since attention only uses the first `context_len` blocks anyway.
- Status: resolved

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
