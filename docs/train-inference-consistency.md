# Train-inference consistency

This page describes the Qwen3-8B dense BF16 path that makes a Lumen
training forward and an ATOM rollout agree on the same token. FP8 is out
of scope.

The longer measurement record, including the older protocol, is in
[the BF16 alignment record][bf16-record].

Training is one Lumen teacher-forcing forward (FSDP2). The online score is
the log-probability ATOM returns from `generate`. Position `i` scores token
`i+1`. The last token of each sequence is dropped. Both sides use GPU
float32 `log_softmax`. The logged field `mismatch/abs_diff` is the mean of
`|rollout − training|` on the response mask. `frac_abs_gt_0.1` is the
fraction of response tokens whose absolute gap exceeds 0.1.

`LUMENRL_PREFILL_RESCORE` stays 0, so the comparison is training against
online decode (s^D). Setting it to 1 replaces the log-probability that
enters the loss with one prefill of the finished sequence (s^P) and hides
a real decode gap.

## What the feature changes

`MODE` selects the engine and the numeric format. `LUMENRL_CONSISTENT=1`
turns the alignment on, and only `atombf16` and `atomfp8` accept it. The
gather inside decode allocates every step, so that path stays eager
(`enforce_eager=true`, `compilation_config.level=0`). A level-3 CUDA graph
replays fixed addresses and aborts.

- **Six training ops.** The switch sets `LUMEN_ATOM_OPS` to:

  `rmsnorm,residual_rmsnorm,qkv,rope,varlen_attn,gate_up`

  Training norm, QKV, RoPE, attention, and gate/up follow the ATOM kernels.
  An empty tag list leaves HF RMSNorm, HF RoPE, and HF attention in place.

- **Decode attention.** `LUMENRL_ALIGN_DECODE=1`, implemented in
  `lumenrl/aiter_patch/align_decode.py`. Decode uses aiter RoPE and gathers
  the paged cache into the varlen kernel. It does not run the assembly
  paged-decode kernel.

- **Same prefix reduction.** `LUMENRL_PREFILL_DECODE_ATTN=1` and
  `LUMENRL_ROWWISE_TRAIN_ATTN=1`, implemented in
  `lumenrl/aiter_patch/qlen1_attention.py` and
  `lumenrl/engine/training/packing.py`. Prefill, decode, and the packed
  trainer cut a row the same way. A row with more than 1024 keys is split
  into 1024-key blocks and merged in fp32. A single packed causal call
  diverges from decode once a query passes 2048 keys.

- **Batched decode.** `LUMENRL_ALIGN_BATCHED_DECODE=1`. One decode step
  attends every sequence in the pool together.

- **Contiguous K/V.** `LUMENRL_ALIGN_CONTIG_KV=1`. A single sequence reads
  a contiguous K/V instead of a paged gather.

- **Deferred output.** `LUMENRL_ALIGN_SYNC_OUTPUT=0`. The one-token logprob
  copy is kept alive with `record_stream` until the copy finishes. `=1`
  clones and synchronizes, and is the diagnostic arm.

- **Vocabulary tail.** `LUMENRL_TRAIN_VOCAB_MASK`, read from the tokenizer.
  Training blanks the padded lm_head columns ATOM drops before
  `log_softmax`. For Qwen3-8B-Base that width is 151669. Leaving it unset
  scores every rollout token high.

- **Repeatable GEMM.** `LUMENRL_STABLE_GEMM=1`, set by default once
  `LUMEN_ATOM_OPS` is non-empty (`fsdp_backend.py`). Pads M so the BF16
  GEMM runs at splitK=1. splitK>1 FlyDSL BF16 is not repeatable on one
  input.

- **Assembly decode.** `ATOM_FORCE_ATTN_TRITON=1`. The assembly paged-decode
  kernel returns a finite wrong value when a context fills exactly 16 pages
  and the last page is partial. The mean absolute gap stays flat;
  `mismatch/chi2_token` jumps. The flag is the default on every ATOM mode in
  `run_dapo.sh`. Keep prefill RoPE on aiter.

The switch does not set a GEMM table. The runs below point
`AITER_CONFIG_GEMM_BF16` at `configs/qwen3_8b_bf16_tuned_gemm.opus.csv`
(sha256 prefix `ea31df209345cc00`). Without that file, qkv, o_proj, and
down_proj miss the tuned table.

## How to run

```bash
MODE=atombf16 LUMENRL_CONSISTENT=1 bash examples/DAPO/run_dapo.sh
```

`MODE=atombf16` alone is the unaligned ATOM BF16 control. `MODE=bf16` or
`MODE=fp8` with the switch exits.

## Measured results

Hardware is 8× AMD Instinct MI355X (gfx950). Model is Qwen3-8B-Base, BF16.
Rollout is ATOM at tensor parallel 1. Training is Lumen FSDP2. Seed 1234,
`kl_coeff=0`, eager rollout, opus GEMM table. The tables are steps 1–100
of the 200-step runs.

| Pin | Value |
| --- | --- |
| Lumen | `b83e23c` on the new arms. Multi-seq baseline log is `b816848`. |
| Lumen-RL | `34ea62b` on the new arms |
| ATOM | `6e9b3e1` plus the uncommitted sampler-seed patch |
| aiter | `4eb7dcac` |
| Image | `zhangdanyangamd/lumen-rl:dapo-gfx950-rocm7.2.3-260917` |

The single-sequence pair and the multi-sequence consistent arm use the
Lumen and Lumen-RL pins above. The multi-sequence baseline is the existing
200-step log on Lumen `b816848`. It was not rerun.

Rollout throughput is the per-step mean of
`(sequences per GPU × mean response length ÷ generation time)`, in
tok/GPU/s. Single-sequence is 1 sequence per GPU. Multi-sequence is 16.

### Single sequence: 1 prompt × 8 responses

| | Baseline | Consistent |
| --- | --- | --- |
| Rollout (s/step) | 26.08 | 29.43 |
| old_logprob (s/step) | 0.31 | 0.30 |
| Actor update (s/step) | 0.72 | 0.68 |
| Rollout throughput (tok/GPU/s) | 35.4 | 35.6 (100.6%) |
| `mismatch/abs_diff` | 0.0143, nonzero on 100/100 steps | 0 on 100/100 steps |
| Mean response length (tok) | 762 | 888 |

### Multi sequence: 8 prompts × 16 responses

| | Baseline | Consistent |
| --- | --- | --- |
| Rollout (s/step) | 88.55 | 90.74 |
| old_logprob (s/step) | 2.11 | 1.92 |
| Actor update (s/step) | 4.78 | 4.28 |
| Rollout throughput (tok/GPU/s) | 181.5 | 177.3 (97.7%) |
| `mismatch/abs_diff` | 0.0108, nonzero on 100/100 steps | 0 on 100/100 steps |
| Mean response length (tok) | 986 | 968 |

The baseline stays nonzero because `LUMEN_ATOM_OPS` is empty, the vocab
mask is empty, and `ATOM_FORCE_ATTN_TRITON=0`. Both arms use the opus GEMM
table. `LUMENRL_TRAIN_ATOM_GEMM=1` puts the baseline linears on `tgemm.mm`.

## Limits

The packed trainer's backward is still one full causal varlen on the same
Q, K, and V (`packing.py`, `_RowwiseAttention.backward`). It matches the
forward on short sequences and diverges once a query reads past 2048 keys.
A forward gap of 0 does not mean the actor update is the gradient of the
forward that produced the log-probability.

Reply lengths still differ across runs. Each run is compared only with the
log-probabilities it saved. The 256-row bitwise result is BF16, head dim
128. Another dtype needs its own mismatch run.

[bf16-record]: ../examples/DAPO/qwen3-8b-dense-bf16-align.md
