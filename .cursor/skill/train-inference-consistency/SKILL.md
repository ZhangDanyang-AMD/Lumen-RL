---
name: train-inference-consistency
description: >-
  Make Lumen-RL rollout and training logprobs bitwise identical on AMD MI350X,
  and recover rollout speed without reopening that gap. Use when the user asks
  for train-inference consistency, on-policy alignment, bitwise logprobs,
  mismatch or KL between ATOM rollout and Lumen training, LUMENRL_CONSISTENT,
  or Qwen3-8B dense BF16 alignment. Training is Lumen, rollout is ATOM, the
  framework is Lumen-RL.
---

# Train-inference consistency

Make the ATOM rollout logprob and the Lumen training forward agree on the same token, then bring rollout speed back toward a frozen baseline. The measured path is Qwen3-8B-Base, dense, BF16, Lumen FSDP2, ATOM TP1, 8× AMD Instinct MI350X (gfx950).

Compare `log p(y[t] | prompt, y[:t])` in GPU float32 `log_softmax`, on the response mask, dropping the last token of the sequence. The log field is `mismatch/abs_diff` (mean absolute gap) and `frac_abs_gt_0.1`. Also record the first differing position and `mismatch/chi2_token`. A small mean can hide a break at one position or a few arbitrary tokens.

`LUMENRL_PREFILL_RESCORE=0` compares training against online decode. Setting it to 1 replaces the loss logprob with a second full prefill and hides a real decode gap.

The current code is the commit "Gate consistent decode on LUMENRL_CONSISTENT and split keys past 1024" on `opt/consistent-batched-decode`. `examples/DAPO/qwen3-8b-dense-bf16-align.md` has the current protocol in its first section. The later sections are the older record (`SYNC_OUTPUT=1`, the deleted `private.csv`, no split-KV). Do not mix numbers across the two.

## Turn it on

`MODE` selects the engine and the numeric format. `LUMENRL_CONSISTENT=1` is the alignment switch, and only ATOM modes accept it:

```bash
MODE=atombf16 LUMENRL_CONSISTENT=1 bash examples/DAPO/run_dapo.sh
```

`MODE=atombf16` alone is the unquantized ATOM BF16 control. `MODE=bf16` or `MODE=fp8` with the switch exits.

The switch exports the six `LUMEN_ATOM_OPS` (`rmsnorm,residual_rmsnorm,qkv,rope,varlen_attn,gate_up`), `LUMENRL_ALIGN_DECODE`, `LUMENRL_ALIGN_CONTIG_KV`, `LUMENRL_ALIGN_BATCHED_DECODE`, `LUMENRL_PREFILL_DECODE_ATTN`, `LUMENRL_ROWWISE_TRAIN_ATTN`, `LUMENRL_PREFILL_RESCORE=0`, `LUMENRL_ALIGN_SYNC_OUTPUT=0`, and `LUMENRL_TRAIN_VOCAB_MASK` from the tokenizer length. `ALIGN_DECODE=1` forces eager and `compilation_config.level=0`, because the KV gather allocates every step and a level-3 graph replays fixed addresses.

The shell switch does not export `LUMENRL_STABLE_GEMM` or `AITER_CONFIG_GEMM_BF16`. Once `LUMEN_ATOM_OPS` is non-empty, `lumenrl/engine/training/fsdp_backend.py` does `setdefault("LUMENRL_STABLE_GEMM", "1")`. A caller that already exported `0` keeps `0`. A baseline with empty `LUMEN_ATOM_OPS` does not get that default. Pin the GEMM table yourself when the measurement needs one. Quantization stays on `MODE`. fp8 and mxfp4 reuse this switch and still need their own quant config plus a new mismatch run. The 256-row bitwise result below is bf16, head dim 128.

`LUMENRL_TRAIN_ATOM_GEMM=1` is a different path: empty `LUMEN_ATOM_OPS`, actor linears on `tgemm.mm`, norm / RoPE / attention left on HF. That is how the opus baseline aligns GEMM only. The consistent path aligns those ops through `LUMEN_ATOM_OPS`, not through this flag.

Lumen's Qwen3 FSDP options (`CACHE_FROZEN_WEIGHT`, `AITER_ATTN`, `LUMEN_NORM`, `FUSE_ROPE`, `GRAD_CKPT`) stay off. The consistent actor log shows `norm=False`, `fused_mlp=False`, `fused_rope=False`. RoPE on this path is ATOM `aiter.rotary_embedding` via `_patch_qwen3_rope`.

## Work order

1. Freeze the baseline: image, commits, GEMM table, seed, and the launch script. Give every run a new log name. A log that already exists is a reason to stop.
2. Localize the first divergence on one fixed sequence before changing a kernel. Walk embed, norm, QKV, RoPE, attention, o_proj, MLP, lm_head. The earliest layer wins.
3. Prove the replacement bitwise on that op, including the boundary lengths (255/256/257, 1023/1024/1025, 2047/2048/2049, a full page).
4. Run both sides through the real entry (ATOM `generate` and the Lumen teacher-forcing forward) on the same tokens. A unit probe does not replace `mismatch/abs_diff`.
5. Hold mismatch at 0 for a long run. Two steps do not clear a rare race. 100 steps is the gate used so far; 200 steps is the longest measured run.
6. Profile one decode step, take the top costs, and only then change launches. Re-check mismatch after each change.
7. Report rollout throughput against the frozen baseline on the same step count. For each step use `(sequences per GPU) * response_length/mean / timing_s/gen`, then average the steps. Single-seq (1 prompt × 8 replies on 8 GPUs) has 1 sequence per GPU. Multi-seq (8 × 16) has 16. Also report `1000 * timing_s/gen / response_length/max` averaged over steps, because reply lengths differ. Do not quote the logged `throughput/gen_tok_per_s` for this comparison. Keep wall time, host time, and GPU time in separate columns.

Acceptance for this path: `mismatch/abs_diff` is 0 on every step, `frac_abs_gt_0.1` is 0, and rollout throughput stays near 90% of the frozen baseline. Forward match, backward match, and speed are three results. One does not stand in for another.

## Attention

Decode, prefill, and the packed trainer must cut a row the same way. The cut depends only on how many keys that row has.

gfx950 fmha v3, bf16, head dim 128, groups its Q tile at 256 rows. An entry that fits in one tile takes the same per-row path as `q_len=1`. A longer causal entry schedules the second half of the tiles differently, so a single packed causal call diverges from decode once a query passes 2048 keys. `CAUSAL_CHUNK = 256` in `lumenrl/aiter_patch/qlen1_attention.py`. On 4096 tokens this is about 0.27 ms per layer against about 10.9 ms for one kernel per row, with 0 bad rows against `q_len=1`.

A row with more than 1024 keys is split into 1024-key blocks counted from the first key. 1024 is a multiple of 256, so a chunk never straddles a block. Each block is its own varlen entry. `_merge_partials` combines them in order: `w = exp(lse - max)`, `acc / s`, fp32 accumulation, stored back to bf16. Rows with at most 1024 keys are unsplit and match the old per-row kernel bitwise. Decode and training match each other bitwise at 512, 1024, and 2048. Against the unsplit per-row reference the split output differs by about 3e-2 absolute; a full causal call differs from that same reference by about 1.6e-2. Exact arithmetic makes the merge the same softmax. bf16 partials do not. Past 1024 keys the sampled trajectory leaves the old unsplit consistent run. The RL contract is that rollout and training still match each other.

`_varlen_fwd` calls `fmha_v3_varlen_fwd` directly for bf16 head dim 128, with the arguments the wrapper would have passed. Other dtypes go through `flash_attn_varlen_func`. The direct call matched the wrapper bitwise and dropped about 13 µs of Python per launch.

Prefill with a prefix cache has fewer query rows than key rows. Query row `j` attends through `prefix_len + j`. Build that plan with `causal_chunk_meta` and reuse it for every layer of the forward.

## Operators that moved the logprob

| What diverged | What matched |
| --- | --- |
| ATOM's asm paged-decode kernel returns a finite wrong value when the context fills exactly 16 pages and the last page is partial. The mean `mismatch/abs_diff` stays flat; `mismatch/chi2_token` jumps. | `ATOM_FORCE_ATTN_TRITON=1`. `run_dapo.sh` defaults this to 1 on every ATOM mode, not only the consistent path. A baseline script that exports 0 is a different control. Keep the prefill RoPE on aiter. The Triton flag otherwise switches prefill RoPE to Triton while training stays on aiter. |
| FlyDSL bf16 GEMM with splitK>1 is not repeatable on one input. | `LUMENRL_STABLE_GEMM=1` pads M to splitK=1 (`lumen/config.py` `_stable_tgemm_mm`). Pin the table with `AITER_CONFIG_GEMM_BF16`. The measured perf runs used `configs/qwen3_8b_bf16_tuned_gemm.opus.csv` (sha256 prefix `ea31df209345cc00` on the 200-step pair). |
| ATOM drops the padded lm_head tail before `log_softmax`. Training that keeps the pad scores every rollout token high. | `LUMENRL_TRAIN_VOCAB_MASK` equals the tokenizer length. For this Qwen3-8B-Base that length is 151669. |
| `tokenIDProcessor` copies the gathered logprob on a side stream and reads it next step. The caching allocator reuses that one-element block, so the copy observes the next forward's value. | `gpu_logprobs.record_stream(async_copy_stream)` while `LUMENRL_ALIGN_SYNC_OUTPUT=0`. `SYNC_OUTPUT=1` turns deferred output off and is the diagnostic arm. |
| A residual parked on `backbone._atom_residual` is missing when activation checkpoint recomputes one layer. | The residual travels with the hidden state into the next layer (`lumen/config.py`, rank-4 leading 2 is hidden plus residual). |

`pool_write` is one Triton launch for K and V. It matched `index_copy_` bitwise. A row whose last two strides are not `(dim, 1)` stays on `index_copy_`.

Reading a grouped Q in place with `as_strided` fails: the kernel requires `out` to have Q's physical row count. Skipping ATOM's `reshape_and_cache` was left alone.

### GEMM table coverage

If `AITER_CONFIG_GEMM_BF16` is unset, `aiter/jit/core.py` `get_config_file` merges `configs/bf16_tuned_gemm.csv` with every `model_configs/*bf16_tuned_gemm*.csv`. If the env is set, only those files are used. `qwen3_8b_bf16_tuned_gemm.csv` has 10 gfx950 cu256 rows: gate_up `24576×4096` at M=1,2,4,8,16 (triton) and lm_head `151936×4096` at M=1,2,4,8 (triton) and M=16 (torch). Lookup pads M upward, so only M≤16 on those two shapes hit. qkv `6144×4096`, o_proj `4096×4096`, and down_proj `4096×12288` have no rows. A miss logs `not found tuned config` and falls back to torch. `get_GEMM_A16W16_config` is cached, so log lines are not invocation counts.

The 100-step baseline that pointed the env at `bf16_tuned_gemm.csv` plus that qwen3 file (Lumen `b816848`, no opus) missed 13824 unique `(M,N,K)` and covered 32. Mean step time was 109.7 s against 95.3 s for the opus multi-seq baseline. `LUMENRL_PRIVATE_GEMM` is not read anywhere in the current tree. `private.csv` and `private_gemm.py` are gone. Do not cite them as the current table.

## Speed

After the split, attention GPU time on a long decode fell below the baseline and the step became host-bound: a gather, a merge, and a pool write per layer, plus the Python around `flash_attn_varlen_func`.

Changes that kept the values and cut host time:

- Call the leaf `fmha_v3_varlen_fwd` for bf16 head dim 128.
- `warmup` the merge and the pool write once, then launch the compiled kernel. The cache key includes 16-byte pointer alignment, because Triton specializes on it. A compiled launch was about 5 µs against about 13 µs through the JIT path.
- Read each `LUMENRL_*` flag once (`_flag` in `align_decode.py`). They are fixed before the worker starts.
- Build the decode split plan and the training chunk plan once per step and share them across layers.

Profile a single decode before editing. Optimize the top costs. CUDA graphs stay off on this path.

On one GPU, bs=1, context 7168, the old per-row consistent decode was 16.82 ms per step against a 13.84 ms baseline. After the split and the host-launch cuts it was 15.15 ms. GPU-busy time was 13.2 ms against the baseline's 13.5 ms. Quote a ratio only from one clock.

## Backward

The packed trainer's backward is still one full causal varlen on the same Q, K, and V (`packing.py`, `_RowwiseAttention.backward`). Chunk key ranges overlap, so the varlen backward cannot run on the chunked launch. That gradient matches the forward on short sequences and diverges once a query reads past 2048 keys. A mismatch of 0 does not mean the actor update is the gradient of the forward that produced the logprob.

## Measured

Opus GEMM on both sides. Throughput is the work-order definition, in tok/GPU/s. Consistent `mismatch/abs_diff` is 0 on every step of each run below. The baseline stays nonzero because `LUMEN_ATOM_OPS` is empty (HF RMSNorm, HF RoPE, HF attention), the vocab mask is empty, and those scripts set `ATOM_FORCE_ATTN_TRITON=0`. GEMM on the opus baseline is the aligned part (`LUMENRL_TRAIN_ATOM_GEMM=1`).

| Run | Shape | Baseline | Consistent | Consistent / baseline |
| --- | --- | --- | --- | --- |
| 100-step, older consistent (`SYNC_OUTPUT=1`, no split-KV, Lumen `b954e71`) | 1×8 | 35.80 | 31.60 | 0.88 |
| 100-step, split-KV consistent on Lumen `b816848`; baseline still `b954e71` | 8×16 | 183.02 | 168.25 | 0.92 |
| 200-step, both on Lumen `b816848` | 8×16 | 253.8 | 227.7 | 0.90 |

The 200-step pair is the current gate. Logs: `logs/ex5-fig100-seed-lumennew-base-opus-p8g16-200step-run.log` and `logs/ex5-fig100-seed-lumennew-consistent-opus-p8g16-200step-run.log`. Pins: image `zhangdanyangamd/lumen-rl:dapo-gfx950-rocm7.2.3-260917`, Lumen `b816848` on `origin/main` `7b5931d`, Lumen-RL HEAD `a7f4886` plus dirty tree `504cc5cc` (the attention, packing, and align_decode edits now in the "Gate consistent decode" commit), aiter `4eb7dcac`, ATOM `6e9b3e1` plus the uncommitted sampler-seed patch. Steps 1–100 of that pair were 181.5 vs 182.3 tok/GPU/s. Steps 101–200 were 326.1 vs 273.1. In that second half the consistent replies were shorter (mean 1902 vs 2078) and `ms/maxlen` was higher (16.12 vs 14.56). Baseline mean `mismatch/abs_diff` over 200 steps was 0.0086, `frac_abs_gt_0.1` 0.0304, `abs_max` 0.907.

## Evidence

- Pin the image, Lumen-RL, Lumen, ATOM, and aiter commits, plus the working-tree diff hash, into the run's `.pins` file. The image's aiter can fail importing flydsl; the measured runs put host aiter `4eb7dcac` on `PYTHONPATH` and set `LUMENRL_ATOM_AITER_SRC`. `examples/DAPO/atom_aiter_shim/sitecustomize.py` fills the modules the image lacks.
- `atom_ray_server._build_sampling_params` passes `seed` only when `SamplingParams` has that field: the request seed, else `ATOM_SAMPLING_SEED`, else `base_seed + replica_rank`. The committed ATOM gitlink has no `seed` field, so the kwarg is dropped and the trainer seed does not enter sampling. The local uncommitted patch on `6e9b3e1` adds it; the 200-step pins record that patch. Reply lengths still differ across runs. Compare each run with the logprobs it saved.
- `kl_coeff=0.001` loads a second model and does not fit on 8 cards. Filter groups drop a step when the eight samples of a prompt have no accuracy variance. The aligned runs kept both off.
- Put `MODE` in `ATOM_TORCH_COMPILE_CACHE_ROOT`. An fp8 graph loaded by a later bf16 run dies in AOTAutograd.
- A source probe, a 2-step run, and a 100-step run answer different questions. Say which one a number came from.
- Re-measure when the dtype, the head dim, the GEMM table, or the attention kernel changes. The bf16 head-dim-128 tile result does not transfer to fp8 or mxfp4.

## Where to edit

| Piece | File |
| --- | --- |
| Chunk, split, merge, direct varlen, pool write | `lumenrl/aiter_patch/qlen1_attention.py` |
| Decode pool, prefill plan, logprob `record_stream`, flag cache | `lumenrl/aiter_patch/align_decode.py` |
| Training forward and the varlen backward stand-in | `lumenrl/engine/training/packing.py` |
| The one switch and the eager override | `examples/DAPO/run_dapo.sh` |
| `STABLE_GEMM` setdefault, `TRAIN_ATOM_GEMM` | `lumenrl/engine/training/fsdp_backend.py` |
| Sampler seed passthrough | `lumenrl/engine/inference/atom_ray_server.py` |
| splitK=1 pad and the fused residual | `lumen/config.py` (Lumen, not this repo) |
| Measurements and the 200-step result (current section first, older record after) | `examples/DAPO/qwen3-8b-dense-bf16-align.md` |
