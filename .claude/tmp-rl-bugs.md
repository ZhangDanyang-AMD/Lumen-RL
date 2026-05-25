# Temporary Training Bug Notes

This file lives at `.claude/tmp-rl-bugs.md` relative to the `Lumen-RL` repo root. Read the whole file at the start of every new debug session.

Use it to keep track of possible bugs found during testing. Do not treat any entry here as proof. Re-check against the current reference diff and current repro before acting.

Treat any fresh return to the same debugging problem as a new debug session:

- a new chat or agent session
- a new day or work block
- returning after unrelated work
- starting a new round of debug after prior tests finished

Write back only meaningful tests or experiments that change confidence in a hypothesis, such as a new repro, written diff, backend toggle, layerwise compare, kernel test, or targeted integration check. Do not log every identical rerun. Do log negative results that rule a suspicion out.

## Open

### [2025-05-21 kimi-k25-sddd-bs-scaling]
- Symptom: HIP memory fragmentation causes OOM after a few steps on MI350 GPUs when using batch sizes >8 for Eagle3 SDDD training
- Possible bug: HIP allocator does not support `expandable_segments`, causing permanent fragmentation from repeated large tensor alloc/free cycles during draft model forward/backward
- Evidence so far:
  - **Root cause identified**: HIP caching allocator fragments GPU address space across micro-batch iterations. `expandable_segments:True` is NOT supported on MI350/HIP.
  - Four optimizations applied (in `spec_distill_trainer.py` and `eagle3.py`):
    1. **Removed lm_head from Eagle3** — lm_head (1.17B params, 163840×7168) was 58% of model params but is frozen (uses teacher weights). Removed from model entirely; forward uses `teacher_lm_head_weight` directly. Saves ~21GB/GPU (FP32 master + Adam states). Trainable params: 2B → 829M.
    2. **Conditional FSDP2/replicate skip** — Added `_can_skip_distributed_wrapping()` auto-detect: skips DDP when model <80GB footprint, no dropout, identical data on all ranks (always true in spec_distill). Eliminates NCCL all-reduce hooks during backward that caused extra fragmentation.
    3. **`torch.cuda.empty_cache()` between micro-batches** — Forces HIP allocator to return reserved memory to system after each micro-batch forward/backward, preventing cross-mb fragmentation accumulation.
    4. **aiter `flash_attn_varlen_func` in Eagle3 attention** — Replaced `F.scaled_dot_product_attention` + 4D mask (O(N²) math backend) with aiter's Flash Attention varlen API. Uses `cu_seqlens` for variable-length batching and `causal=True` for causal mask. Eliminates O(T²) memory for long sequences (T=20000 → 48GB saved). Handles both causal masking and padding natively in O(N) memory.
  - **Test results (bs=64, mb=2, no FSDP2, no lm_head, before flash attn fix)**:
    - Step 0 completed: loss=38.7, grad_norm=27.8, 210s/step
    - Memory pattern per mb: base 16.9G → draft_forward 145-174G → backward 16.9G → empty_cache → reserved back to 40G
    - Step 1 started without OOM
  - **Attention OOM root cause**: `F.scaled_dot_product_attention` with explicit `attn_mask` forces O(N²) math backend. At T=20000, single attention head needs 20000²×2 bytes = 0.75GB; with 64 heads = 48GB. Exceeds GPU memory on step 2 when a sample has max-length sequence.
  - **Flash Attention fix**: aiter's CK-based `flash_attn_varlen_func` supports: causal mask via `causal=True`, padding via `cu_seqlens`, GQA natively, BF16 on gfx950 (MI350). No bias or 4D mask needed — padding handled by packing valid tokens contiguously.
  - Previous failed tests (with FSDP2 + lm_head):
    - bs=64 mb=1: step 0 OK, step 1 OOM
    - bs=48 mb=1: step 0 OK, step 1 OOM
    - bs=24 mb=1: steps 0-1 OK, step 2 OOM
    - bs=16 mb=1: steps 0-3 OK, step 4 OOM
  - Previous failed tests (no FSDP2, no lm_head, but mb=1 without empty_cache):
    - bs=64 mb=1: step 0-1 OK, step 2 OOM (reserved grew 98G→123G→187G→260G)
  - Previous failed tests (no FSDP2, no lm_head, empty_cache, trim, but SDPA math backend):
    - bs=64 mb=1: step 0-1 OK, step 2 OOM on T≈20000 sample (195.7G forward allocation from O(N²) attention)
- Next check:
  1. Test bs=64 mb=1 with flash_attn_varlen_func — should handle T=20000 without OOM
  2. If yes, double to bs=128 mb=1 and test
  3. Binary search for max stable batch size
  4. Compare loss/accuracy with bs=8 baseline to verify numerical equivalence
- Status: open

### [2025-05-21 kimi-k25-sddd-lumenconfig-compat]
- Symptom: `TypeError: LumenConfig.__init__() got an unexpected keyword argument 'lumen_linear'` on all 4 ranks
- Possible bug: Container's Lumen package doesn't support `lumen_linear` parameter in LumenConfig
- Evidence so far: Fixed by using `inspect.signature` to check available params before passing. Only passes kwargs that LumenConfig actually accepts.
- Fix: `spec_distill_trainer.py` lines 283-292, introspects `LumenConfig.__init__` params
- Status: resolved

## Ruled Out

### [2025-05-21 kimi-k25-sddd-attention-mask]
- Symptom: Investigated whether LumenRL's Eagle3 attention mask was incorrect
- Evidence: Compared with TorchSpec's implementation. Both create causal + padding 4D mask `[B, 1, T, T]` and pass to `F.scaled_dot_product_attention(attn_mask=...)`. Logic is identical.
- Status: ruled out

### [2025-05-21 kimi-k25-sddd-fsdp2-unshard]
- Symptom: Initially suspected FSDP2 unshard/reshard during backward as the fragmentation source
- Evidence: Default strategy is `replicate` (DDP-like), not `full_shard`. No param sharding occurs. Fragmentation comes from gradient all-reduce NCCL buffers + draft forward/backward activation allocations, not from param unshard/reshard.
- Status: ruled out (fragmentation source was misidentified; actual sources are DDP hooks + activation alloc/free cycles)

## Resolved

### [2025-05-21 kimi-k25-mooncake-shutdown]
- Symptom: vLLM teacher process not cleaned up properly on training exit
- Fix: Synced upstream fix from `dev/OPD` branch — `vllm_teacher_engine.py` now kills entire vLLM process group via `os.killpg()`, and `cleanup()` shuts down teacher engine BEFORE mooncake master
- Status: resolved

## Entry Template

```markdown
### [YYYY-MM-DD session-name]
- Symptom:
- Possible bug:
- Evidence so far:
- Next check:
- Status: open | ruled out | resolved
```
