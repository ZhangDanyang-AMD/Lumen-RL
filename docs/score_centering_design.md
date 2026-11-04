# Score Centering in Lumen-RL — Design & Implementation Plan

Status: proposal · 2026-11-03 · author: Wen Chen (drafted with Claude)

## Implementation status (branch `feat/score-centering`)

| Phase | State | Where |
|---|---|---|
| P0 config + guards | done | `core/config.py` `ScoreCenteringConfig`; `algorithms/score_centering.py` `validate_score_centering` (also requires `controller.ray.enabled` and `vllm_cfg.transport: ray_http`) |
| P1 estimator | done, 30 CPU tests | `score_centering_token_loss`, `score_centering_pg_loss` |
| P2 trainer op | op done (13 tests, GPU-checked); Megatron wiring pending the Megatron container | `engine/training/packing.py` `selected_log_probs`, `packed_selected_log_probs`, `pack_shifted` |
| P3 rollout capture + transport | done (24 tests + 2-GPU vLLM/HF parity) | `vllm_ray_server.extract_flat_logprobs`; `rl_trainer` engine kwargs, sampling params, `rollout_topk` ragged → dense `rollout_topk_ids/logprobs` |
| P4 Megatron loss wiring | done (7 CPU tests + 8B smoke in `rl-megatron`) | `megatron_base_engine._row_policy_loss` SC branch, `_token_head_logprob_train`; per-row, dynamic-batch and PP/MoE (`megatron_native_engine._pp_update_policy`) paths; `sc/*_sum` + `sc_tok` reduced on the controller; CP > 1 rejected; config `dapo_qwen3_8b_ray_megatron_sc_smoke.yaml` |
| P5 FSDP2 wiring | done (8 CPU tests + 8B smoke in `rl-vllm-fsdp`) | `fsdp_engine._prepare_packed_outputs` scores `[y, H]` with `packed_selected_log_probs` when the micro carries `rollout_topk_ids` (head dropped from per-step outputs after the loss); `workers/actor_worker._policy_loss_fn` SC branch; shared `score_centering_loss_from_config` used by both backends; config `dapo_qwen3_8b_ray_vllm_sc_smoke.yaml` |
| P6 ATOM rollout | **deferred** — needs an upstream ATOM change (see P6 below) | |
| P7 follow-ups | done in Lumen-RL: support replay (top-k), ρ pinning, `ppo_clip_additive`, host-resident head, vocab-parallel op (not wired); deferred: general top-p and FP32 rollout LM head (vLLM changes) | see P7 below |

P4 smoke (Qwen3-8B-Base, Megatron-Native, 7 × MI300X, BF16 vLLM rollout, 2 steps, same
config with SC off vs on): both exit 0. Step-2 `timing/train_s` 3.16 s → 3.31 s (+5 %),
peak actor memory 76.4 → 75.7 GB. `rollout_corr/kl` 0.0009/0.0011 (off) vs 0.0010/0.0014
(on). SC metrics at step 2: `sampler_head_mass` 0.989 = `train_head_mass` 0.989,
`frac_low_head_mass` (< 0.9) 2.6 %, `correction_abs` 2.2e-4, `sample_weight_mean` 0.9996,
`frac_truncated` 0. **Watch item:** `tail_ratio_mean` (ρ) is 0.97, not ≈ 1: on
near-deterministic tokens both tails sit at the 1e-6 floor and ρ is fp32 noise (the Miles
#4204 pattern). With TIS c = 2, α = min(1, 2ρ) is unaffected unless ρ < 0.5; measure the ρ
distribution before deciding whether to pin ρ = 1 when both tails are below a tolerance.
MoE / EP (example 7) path is wired and unit-tested but not smoke-tested here
(Qwen3-30B-A3B not downloaded).

P5 smoke (Qwen3-8B-Base, FSDP2, 7 × MI300X, BF16 vLLM rollout, 2 steps, SC off vs on): both
exit 0. Step-2 `timing/train_s` 16.59 s → 16.77 s (+1 %), peak actor memory 24.71 → 24.73 GB,
`rollout_corr/kl` 0.0015 (off) vs 0.0010 (on). SC metrics at step 2: `sampler_head_mass`
0.985 = `train_head_mass` 0.985, `frac_low_head_mass` 4.1 %, `correction_abs` 1.9e-4,
`frac_truncated` 2e-5, `tail_ratio_mean` 0.968 (same watch item as P4).

P3 measurements (Qwen3-8B-Base, vLLM 0.23 BF16, k = 32, 128-token responses):

* vLLM head vs trainer log-probs at the **same ids** agree up to BF16 logit rounding
  (~0.25-nat steps; sampled-token mean |Δ| 0.01–0.04); head mass on the same ids agrees to
  ≤ 8e-4. Top-1 ids agree on 96–99 % of positions.
* `processed_logprobs` is required: with vLLM's default `raw_logprobs` at T = 0.7 the head
  is off by 4–5.6 nats on average (pre-temperature).
* Head coverage at T = 1 on a DAPO-Math prompt: 1st-percentile mass 0.93 (min 0.89); on
  open-ended prose 0.74. So the paper's "≥ 99.9 %" does not hold for this base model at T = 1;
  the tail model carries real weight. Track `sc/head_coverage_p01` in the smoke and run the
  k sweep (§6.5).
* Capture sets `detokenize=False` (decoding k + 1 candidates per position is pure overhead;
  the trainer decodes token ids itself) and uses `flat_logprobs` for both k = 0 and k > 0.

## 0. TL;DR

* **What**: Score centering (SC; Marek & Ryabinin, arXiv 2609.20807) adds a correction to the
  policy-gradient that removes the *drift* term caused by sampler/trainer mismatch
  (FP8 rollout, ATOM vs. HF kernels, MoE routing, async staleness). It is a no-op on-policy,
  has no hyperparameters, and needs only the sampler's **top-k token ids + logprobs** per
  response token.
* **Why for us**: Lumen-RL's raison d'être is low-precision / alternative-engine rollout
  (examples 2–5: vLLM FP8, ATOM FP8/BF16). The paper's headline results are exactly that
  regime (FP8/INT8/FP4-KV samplers, MoE without router replay), where plain PG and PPO/DAPO
  clipping collapse and SC stays stable.
* **Decision**: implement SC the way verl, slime, Vime and Miles converged on — a
  **REINFORCE objective + additive SC correction**, with the IS rule (none / token-TIS /
  IcePop) applied *inside* the correction coefficients, k = 32 default, vLLM
  `processed_logprobs` + `flat_logprobs`, unfiltered sampling only in v1. **Order: Megatron +
  vLLM first** (examples 7 / MoE, where mismatch matters most), then FSDP2, then ATOM.
  All three Megatron engines build `GPTModel(parallel_output=False)`, so every TP rank already
  holds full-vocab logits and no vocab-parallel op is needed for correctness (it is a later
  memory optimization). Top-p support replay is a follow-up.
* **Design clarifications** (REINFORCE vs PPO, vocab parallelism, what vLLM's top-k head
  contains, how frameworks model the tail, top-p support) are collected in **Appendix A**.
* **Cost**: ~7 PRs, ≈ 2–3 engineer-weeks to a Megatron + vLLM A/B (example 7 / Qwen3-8B
  Megatron config); +~3 days for FSDP2 (op and loss are shared); +~1 week for ATOM capture.

---

## 1. The method in one page

Notation: sampler (rollout engine) distribution `q`, trainer distribution `p`, sampled token
`y`, advantage `A`, score `s_v = ∇ log p_v`.

Plain PG under the sampler decomposes as

    E_q[A · s_y] = E_q[A] · s̄   +   Cov_q(A, s_y),      s̄ = Σ_v q_v s_v
                   └─ drift ─┘      └── signal ──┘

`s̄ = -∇ CE(q ‖ p)`: the drift term **distils the trainer toward the (biased) sampler**,
and since the sampler is re-synced from the trainer each step, the bias compounds. GRPO
group-centering does *not* remove it (the expected advantage at a given prefix is not 0).
SC replaces `s_y` with `s_y − s̄`, leaving only the covariance; with constant reward the
expected update is exactly 0, on-policy it is unchanged.

Only the sampler's top-k head `H` is stored. The tail is modelled as the trainer's tail
rescaled: `q̂_v = ρ p_v` for `v ∉ H`, with `ρ = (1 − Σ_H q) / (1 − Σ_H p)`. Because
`E_p[s] = 0`, the tail contribution folds into the head and the loss is (paper eq. 14):

    L = −A · ( sg[w_y] · log p_y  −  Σ_{v∈H} sg[q_v · w_v − α · p_v] · log p_v )

    w_v = f(p_v / q_v),   α = ρ · f(1/ρ),   ρ floor eps = 1e-6
    f(r) = 1                      (plain SC)
         = min(r, c)              (token TIS, c = 2)
         = r · 1[lo ≤ r ≤ hi]     (IcePop / MIS, [0.5, 5])

Everything except `log p` is detached. With `f(r) = r` (exact IS) the correction vanishes.
k = 32 matched full-vocab centering in every paper setting; head mass ≥ 99.9 % except
extreme quantization.

---

## 2. Survey of implementations

| Framework | Status | Base objective | IS inside SC | Sampler capture | k default | Trainer top-k op | Filtered sampling |
|---|---|---|---|---|---|---|---|
| **verl** (#8010, #8196 merged; #8199 Megatron open) | merged 2026-10 | REINFORCE in bypass mode only | none / TIS min(r,c) / IcePop | vLLM `logprobs=k, flat_logprobs`, `processed_logprobs` | user | chunked `log_softmax.gather` (4096 rows), FSDP | auto "support alignment": renormalize over support when head proves it complete |
| **NeMo RL** #4818 (merged 2026-11-01) | merged | **keeps clipped PPO**, adds SC on top | off / is / tis / icepop | OpenAI `top_logprobs=K` via Gym, `processed_logprobs` if T≠1 | user | Megatron TP/CP chunked distributed gather | rejected (follow-up #5006 adds replay) |
| NeMo RL #4833 | closed unmerged | REINFORCE | same | `SamplingParams(logprobs=K)` | — | reuses OPD top-k op | vLLM sampling-mask replay |
| **slime** #2405/#2406 | merged 2026-09 | REINFORCE only (`--pg-loss-type`) | TIS (two-sided clamp) / IcePop | patched SGLang, base64 int32/fp32 | 128 | `_VocabParallelTopKLogProbs` (no bwd all-reduce) | #2406 exact top-p replay |
| **Vime** (vllm-project/vime, #441/#470) | on main 2026-10/11 | REINFORCE only | TIS / IcePop (slime rules) | vLLM `processed_logprobs`, `logprobs=k+1` | 128 | slime's op | exact top-p via vLLM patch `--return-sampling-mask` |
| **Miles** #3621–#3626 + many | merged 2026-10 | separate `--loss-type score_centering` (REINFORCE) | none / TIS min(r,c) / MIS | SGLang top-logprobs (selected) or sampling mask (support) | user (recipe 128) | `_SelectedLogProbs` scores `[y, H]` in one pass; FP32 LM head forced on rollout | support replay (ρ pinned to 1, #4310) |
| **torchtitan** #5760 | draft, opened 2026-11-03 | ScoreCenteringLoss (REINFORCE) | TIS min(r,c) or masked band | vLLM `logprobs=k`, reshape `[sampled, top1..k]` | 32 | compiled top-k logprobs, TP via 3 all-reduces | no |
| AReaL #1797 | open | `reinforce` or `ppo_additive` | — | vLLM `logprobs=k` (no `logprobs_mode` → **likely wrong at T≠1**) | 128 | ported from verl | rejected |
| Reef #673 | merged | correction term per loss family; PPO clip refused | per family | SGLang/vLLM `capture_topk` | 128 | vocab-parallel gather | no |

Consensus points we adopt:

1. **Same estimator everywhere** — `residual = q·f(p/q) − α·p`, `α = ρ f(1/ρ)`, eps 1e-6,
   correction added *after* the sampled score is IS-weighted, and **never weighted twice**.
2. **REINFORCE base** (verl, slime, Vime, Miles, torchtitan, Reef). PPO clip is a
   policy-movement device; the paper shows PPO/DAPO collapse under quantization noise; the
   only PPO-additive implementation (NeMo) documents that it does not exactly center the
   clipped gradient and merged without a convergence A/B.
3. **Sampler logprobs must be post-temperature** (`processed_logprobs` in vLLM) and the
   sampled-token `q_y` comes from the recorded rollout logprob even if `y ∉ H`.
4. **Unfiltered sampling** (top_p = 1, top_k = −1) for the tail model; filtered sampling
   needs *support replay* (exact, ρ ≡ 1) — a separate feature (Appendix A.5).
   All frameworks use the same proportional tail `q̂ = ρ·p`; they differ only in the ρ
   floor, the IS rule applied to α, and how they detect a complete support (Appendix A.4).
5. **Validate captured rows** (distinct ids, finite, ≤ 0, head mass ≤ 1 + 1e-4) and
   **disable capture for eval rollouts**.

Known pitfalls to avoid (from review threads/bugs):

* vLLM's default `logprobs_mode=raw_logprobs` is **pre-temperature** (AReaL's likely bug).
* Per-token Python `Logprob` dicts are slow: NeMo measured +25 % step time, cut to +1.6 % with
  `flat_logprobs` + vectorized extraction (#5034).
* Miles #4204: fp32 renormalization makes `1 − mass ≈ 1e-6` cross eps → ρ ≠ 1 under complete
  support; pin ρ = 1 when support is known complete.
* Miles #4489: empty CP ranks returning bf16 vs fp32 → collective hang. Return a consistent
  dtype from empty micro-batches (we already pad micro-batches with zero `response_mask`).
* verl: fused cross-entropy with `inplace_backward=True` overwrites the logits buffer that a
  second (top-k) gather re-reads in backward → must not mix the two.
* Miles #3998: an un-chunked torch top-k path keeps `[rows, V]` fp32 probs for backward
  (~20 GB at 32 K rows) — chunk and recompute in backward.

---

## 3. Lumen-RL today (gap analysis)

| Area | Today | Gap |
|---|---|---|
| Rollout request | `rl_trainer.py:764` sets `sp["logprobs"]=0`; `vllm_ray_server.py:224` builds `SamplingParams(logprobs=0)` | need `logprobs=k`, `flat_logprobs=True`, engine `max_logprobs≥k`, `logprobs_mode=processed_logprobs` |
| Rollout output | `vllm_ray_server.py:241` extracts sampled-token logprob only; returns `routed_experts` side payload | add `topk_ids [T,k] int32`, `topk_logprobs [T,k] fp32` (same side-payload pattern) |
| ATOM | `atom_ray_server.py:172` coerces `logprobs` to bool | no top-k at all → guard in v1, implement in P6 |
| Batch | `rl_trainer.py:676-736` left-padded `[B, S-1]` `rollout_log_probs` | add dense `rollout_topk_ids/logprobs [B, S-1, k]`, same alignment |
| Transport | `DataProto.tensors` sliced by `split`, `dispatch_proto`, `slice_rows` (dynamic batching) | dense tensors ride for free; avoid `ragged` (legacy `apply_rollout_correction` drops it, `rollout_correction.py:711`) |
| Worker config | workers see only `algorithm.*` via `meta["algo_config"]` | put SC config under `algorithm.score_centering` |
| Trainer logprobs (FSDP2) | `packing.py:194` fused Triton CE (`inplace_backward=True`), fallback chunked log_softmax; full logits on-rank | new op that scores `[y, H]` jointly (one LSE) |
| Trainer logprobs (Megatron) | `parallel_output=False` → full-vocab fp32 logits `[L, V]` (`megatron_engine.py:988`) | same op; vocab-parallel version only if we flip `parallel_output` |
| Loss | `asymmetric_clip_loss` (`loss_functions.py:156`), called from `actor_worker.py:896` (+ Megatron `:2080/:2291/:2491`, `megatron_base_engine.py:345`) | REINFORCE + SC loss; `rollout_correction.loss_type: reinforce` exists in config but is **not consumed anywhere** |
| Rollout IS | controller computes `rollout_is_weights` vs `old_log_probs` (`rollout_correction.py:288`, token TIS = `clamp(max=thr)`) | in SC mode, IS is computed **in-loss** from current `p` vs `q`; controller weights must not be applied again |

Observation that de-risks switching to REINFORCE: DAPO in Lumen-RL does **one optimizer
step per rollout batch** (no DAPO mini-batches/epochs), so `old_log_probs ≈ current` at the
update (smoke: `ppo_kl = 2.7e-4`). The PPO clip is essentially inactive in our examples;
the off-policyness is the sampler mismatch, which is what TIS / SC address. Only the
dual-clip (`clip_ratio_c`) disappears, and it only bounds negative-advantage tokens with
huge ratios — TIS already truncates those. (Verify in P0 by logging `pg_clipfrac`.)

---

## 4. Design

### 4.1 Config

```yaml
algorithm:
  score_centering:
    enabled: false
    base_objective: reinforce   # reinforce | ppo_clip_additive (NeMo-style, P7)
    top_k: 32                # sampler head size; 128 = paper's conservative default
    is_mode: tis             # none | tis | icepop
    tis_clip: 2.0            # f(r) = min(r, c)        (tis)
    icepop_low: 0.5          # f(r) = r·1[lo ≤ r ≤ hi] (icepop)
    icepop_high: 5.0
    eps: 1.0e-6              # tail-mass floor
    rho_pin_tail: 1.0e-5     # rho := 1 when both tails < this (fp32 noise; P7)
    support_replay: false    # filtered sampling with top_k < score_centering.top_k (P7)
    chunk_rows: 4096         # trainer top-k op row chunk (fp32 workspace = rows × V × 4 B)
    validate_rows: true      # check captured rows on the controller
```

New `ScoreCenteringConfig` dataclass in `lumenrl/core/config.py`, field on
`AlgorithmConfig` (so it reaches workers via `meta["algo_config"]` with no extra plumbing).
Validation in `__post_init__` + a cross-config check at trainer setup (§4.8).

Rationale for a dedicated `is_mode` rather than reusing
`quantization.rollout_correction.rollout_is`: in SC mode the IS rule lives inside the
correction and must use the *current* `p`; reusing the controller field invites applying it
twice. At setup we **reject** `rollout_is: sequence` and `rollout_is_batch_normalize`.
`rollout_is: token` is allowed but its weights are never applied by the SC loss; a warning
names `score_centering.is_mode` / `tis_clip` as the rule in effect (no implicit copying of
the controller threshold). `rollout_rs` (rejection masks) stays allowed: it only edits
`response_mask`. Implemented in `lumenrl/algorithms/score_centering.py`
(`validate_score_centering`, called from `LumenRLConfig.__post_init__`).

### 4.2 Objective

New `score_centering_pg_loss(...)` in `lumenrl/algorithms/loss_functions.py`:

```python
def score_centering_pg_loss(logp_y, head_logp, q_y, q_head, head_valid, adv, *,
                            mask, weight_fn, eps, batch_num_tokens, dp_size,
                            loss_agg_mode="token-mean", global_batch_size=None):
    """REINFORCE + score centering (paper eq. 14). head_logp has grad; all else detached.
    logp_y:[N,L]  head_logp:[N,L,k]  q_y:[N,L]  q_head:[N,L,k] (−inf where invalid)."""
    with torch.no_grad():
        lp = head_logp.float().masked_fill(~head_valid, -inf)
        p, q = lp.exp(), q_head.float().exp()              # 0 where invalid
        p_mass, q_mass = p.sum(-1), q.sum(-1)
        rho = (1 - q_mass).clamp_min(eps) / (1 - p_mass).clamp_min(eps)
        alpha = rho * weight_fn(rho.reciprocal())
        w_head = weight_fn((lp - q_head.float()).exp().nan_to_num(0.0))
        residual = (q * w_head - alpha[..., None] * p) * head_valid
        w_y = weight_fn((logp_y.detach().float() - q_y.float()).clamp(-20, 20).exp())
    correction = (residual * head_logp.float().masked_fill(~head_valid, 0)).sum(-1)
    pg = -adv.detach() * (w_y * logp_y - correction)
    return agg_loss(pg, mask, loss_agg_mode, dp_size=dp_size,
                    batch_num_tokens=batch_num_tokens, global_batch_size=global_batch_size), \
           metrics(p_mass, q_mass, rho, w_y, correction, mask)
```

* `weight_fn` = `ones_like` | `clamp(max=c)` | IcePop band — one-sided TIS as in the paper,
  Miles, torchtitan, Reef, and our existing controller TIS (`clamp(max=threshold)`).
* Aggregation reuses `agg_loss` (token-mean, `batch_num_tokens`, `dp_size`) so the loss
  scale is identical to today's DAPO loss.
* `logp_y` and `head_logp` come from one op (§4.5), so `p_y` is consistent with `p_head`.

Wiring: in `actor_worker._policy_loss_fn` (`:896`), `megatron_base_engine._row_policy_loss`
(`:345`) and the Megatron packed/PP paths (`:2080/:2291/:2491`), branch on
`algo_config["score_centering"]["enabled"]` and call the new loss instead of
`asymmetric_clip_loss`; `old_log_probs` and `rollout_is_weights` are ignored in that branch
(old-logprob recompute can then be skipped — see P5 perf item).

Not in v1: `ppo_additive` (NeMo style). It is easy to add later as
`base_objective: ppo_clip_additive` if we want an A/B, but it is not what the paper
analyses and NeMo merged it without convergence evidence.

### 4.3 Rollout capture (vLLM)

* Engine args (`vllm_ray_server.py:139`, built in `rl_trainer.py`): when SC on,
  `logprobs_mode="processed_logprobs"`, `max_logprobs=max(k, existing)`. Note: on ROCm this
  disables the aiter top-k/top-p sampler — irrelevant at `top_p=1, top_k=-1` (native path
  already), but log it.
* Request (`rl_trainer.py:764`): `sp["logprobs"] = k`, `sp["flat_logprobs"] = True`.
  Eval sampling params (`_ray_eval_sampling_params`) keep `logprobs=0`.
* `VLLMRayServer.generate` (`:211`): pass through `logprobs=k, flat_logprobs=True`. With flat
  logprobs each position is `[sampled, top1..topk]`; vectorize into
  `topk_ids: np.int32[T, k]`, `topk_logprobs: np.float32[T, k]`, assert entry 0 id == sampled
  token and keep entry 0 as the existing `logprobs` value (no behaviour change for k = 0).
  Return them in the output dict (same pattern as `routed_experts`); numpy across Ray.
* Also fix the k = 0 path to use `flat_logprobs` (pure perf win, same values).

### 4.4 Batch assembly & transport

* `rl_trainer.py:676-736`: allocate `rollout_topk_ids = zeros(B, S-1, k, int32)`,
  `rollout_topk_logprobs = full(B, S-1, k, -inf, fp32)`; write each response at
  `start + plen − 1` exactly like `rollout_lp`. `-inf` marks invalid slots (prompt/pad rows,
  short heads); `head_valid = isfinite(q)`.
* Controller-side validation (`validate_rows`): distinct ids per row, logprobs ≤ 1e-5,
  head mass ≤ 1.0001, response rows count == generated tokens. Fail fast with the sample id.
* They are plain `DataProto.tensors` keys → `split`, `dispatch_proto`, dynamic-batch
  `slice_rows`, padding micro-batches all work unchanged (verify with a unit test).
* Size: `B·(S−1)·k·8 B`. Longrun 8B (B = 512, S = 21 504, k = 32) ≈ 2.8 GB host memory per
  step (k = 128: 11 GB). Per GPU micro-batch it is `max_token_len_per_gpu·k·8 B` (< 10 MB).
  If host memory bites, store only response rows + offsets (ragged) — v2.
* MoE `routed_experts` / R3 and filter_groups operate on rows → unaffected; the DAPO
  over-sampling (`gen_batch_size`) multiplies the in-flight size by the oversample ratio
  before filtering — drop filtered rows' top-k immediately.

### 4.5 Trainer top-k log-prob op

One op computes, per response position, the log-prob of `[y, H]` (k + 1 ids) under the
full-vocab softmax of `z / T` with a single LSE:

```python
class PackedSelectedLogProbs(torch.autograd.Function):
    # fwd: for row-chunks of `chunk_rows`: zc = logits[chunk].float() / T
    #      lse = logsumexp(zc); out = zc.gather(ids) - lse        (ids: [rows, k+1], pad→0)
    #      save logits (not zc) + lse
    # bwd: per chunk: g_logits = scatter_add(zeros, ids, g_out)/T - softmax(zc)·g_out.sum(-1)/T
```

* Lives in `lumenrl/engine/training/packing.py` next to `packed_token_log_probs`; called from
  `fsdp_engine._prepare_packed_outputs` (`:381`) **instead of** the fused CE path when SC is
  enabled (no `inplace_backward` conflict, one softmax instead of two). Returns
  `log_probs` (column 0) and `head_log_probs` (columns 1..k); `unpack_log_probs` gets a
  trailing-dim-aware variant to produce `[B, S−1, k]`.
* Memory: fp32 workspace `chunk_rows × V × 4 B` = 2.5 GB at 4096 rows, Qwen3 V = 151 936;
  recomputed in backward, so peak ≈ 2 chunks. Duplicate ids (y ∈ H) handled by
  `scatter_add` — deterministic on a single device.
* Megatron: logits are already full-vocab fp32 (`parallel_output=False`), so the same op is
  called from `_forward_logits_packed` users (`megatron_engine.py:2226`). A vocab-parallel
  variant (3 all-reduces fwd: max, sumexp, picked logits; none in bwd — slime/Miles/
  torchtitan pattern) is only needed if we later switch to `parallel_output=True`.
* Old-logprob / ref passes do not need the head (it is only used with grad in the update).
* Ulysses SP is not wired in Lumen-RL → nothing to do; CP (Megatron) — the op is row-local,
  rows travel with their labels.
* Why no vocab-parallel op is needed in v1, and what vLLM actually returns in the top-k
  head: Appendix A.2 and A.3.

### 4.6 Interaction with the rest of the stack

| Feature | Behaviour with SC on |
|---|---|
| DAPO dynamic sampling, overlong shaping, GRPO advantages | unchanged (reward side). Optional later: `disable_std_normalization` (Miles recipe) |
| `rollout_is: token` | `reinforce`: allowed, weights not applied (warning; the IS rule is `is_mode`/`tis_clip`). `ppo_clip_additive`: applied to the clip part only |
| `rollout_is: sequence`, `batch_normalize` | rejected |
| `rollout_rs` | allowed (mask only) |
| `bypass_mode` | allowed; SC already uses `q` directly, so old-logprob recompute becomes optional |
| `clip_ratio_low/high/c`, `ppo_kl` | ignored (log once). Keep `ppo_kl` metric for diagnostics |
| Entropy, KL-in-reward (`kl_coeff=0` in all DAPO configs) | unchanged; KL loss term with SC → reject in v1 |
| FP8 training (`TRAIN_FP8=1`) | orthogonal — trainer logits are BF16/FP32 at the LM head |
| MoE router replay / `LUMENRL_FP32_MOE_ROUTER` | orthogonal; SC is expected to *reduce* sensitivity to router mismatch (paper's MoE result) |
| Checkpoint/resume | no state |

### 4.7 Metrics (`sc/` prefix, token-masked means)

`sc/sampler_head_mass`, `sc/train_head_mass`, `sc/tail_ratio_mean` (ρ),
`sc/correction_abs`, `sc/sample_weight_mean`, `sc/frac_truncated` (TIS) or
`sc/frac_masked` (IcePop), `sc/head_coverage_p01` (1st percentile of head mass — the paper's
health signal), plus capture timing `timing/sc_extract_s`. Keep existing `rollout_corr/*`
mismatch metrics (computed from `q_y` vs old logprobs) so runs remain comparable to
examples 1–5.

### 4.8 Guards (fail at setup, not at step 1)

Reject when `score_centering.enabled` and any of: `generation_backend != vllm` (ATOM until
P6) · `top_p != 1` or `top_k not in (-1, 0)` or `min_p != 0` · repetition/presence/frequency
penalties · `temperature <= 0` · `vllm_cfg.calculate_log_probs == false` ·
`algorithm.name not in {dapo, grpo}` · `loss_mode == gmpo` · OPD / spec-distill / SFT task ·
`rollout_is: sequence` / `batch_normalize` · KL loss term · `top_k > max_logprobs` if the user
pinned it. (Planned relaxation: allow sampler `top_k ≤ score_centering.top_k`, where the
support always fits in the head — Appendix A.5.) Also: copy `generation_config.json` defaults check (NeMo #5006: vLLM fills
`min_p`/`repetition_penalty` from it) — assert the effective SamplingParams, not just ours.

---

## 5. Implementation plan

Each phase is one PR with its own tests; P1–P4 land behind `enabled: false`.

**P0 — Baseline & config (2 d)**
* `ScoreCenteringConfig` + `AlgorithmConfig.score_centering`; setup-time guard function
  `validate_score_centering(cfg)` in `lumenrl/algorithms/score_centering.py`.
* Log `pg_clipfrac` / `ppo_kl` on examples 1–2 to confirm the PPO clip is inactive (§3).
* Tests: config round-trip from YAML; each guard raises.

**P1 — Estimator (2 d)** — `lumenrl/algorithms/score_centering.py`, `loss_functions.py`
* `weight_fn(mode, ...)`, `score_centering_correction(...)`, `score_centering_pg_loss(...)`.
* CPU tests (`tests/unit/algorithms/test_score_centering.py`, pytest), ported from the
  consensus suites:
    1. k = V equals exact full-vocab centering (`Σ_v q_v w_v ∇log p_v` via autograd on a
       reference);
    2. constant advantage ⇒ zero expected gradient for random p ≠ q (all three modes);
    3. p = q ⇒ gradient identical to plain REINFORCE (bitwise);
    4. `f(r) = r` ⇒ correction = 0;
    5. masked / −inf head slots, y ∉ H, empty rows, dtype of empty micro-batch;
    6. loss scale matches `asymmetric_clip_loss` at ratio = 1 with no clipping.

**P2 — Trainer op, Megatron first (3 d)** — `packing.py` (shared op), `megatron_engine.py`,
`megatron_native_engine.py`, `megatron_base_engine.py`
* `PackedSelectedLogProbs` + `unpack` for trailing dim. Replace `_FusedTokenLogProb` /
  `packed_token_log_probs` in the Megatron train paths (packed `:2226`, row-wise, PP last
  stage) when SC on. Logits are full-vocab fp32 on every TP rank (`parallel_output=False`),
  so the op is rank-local; CP: rows travel with their labels.
* Tests: gradcheck (fp64, small V); chunked == unchunked; matches
  `log_softmax(z/T).gather` and the existing token log-prob on column 0; duplicate ids;
  GPU memory test that peak ≈ 2 × chunk workspace; TP=2 run gives identical per-rank values.

**P3 — Rollout capture & transport (3 d)** — `vllm_ray_server.py`, `rl_trainer.py`,
`vllm_engine.py`
* Engine args, request params, flat-logprob vectorized extraction, batch tensors,
  controller validation, eval k = 0.
* Tests: unit test on a fake `FlatLogprobs` object; `DataProto` split / dispatch /
  dynamic-batch slicing keep `[B, S−1, k]` aligned with `response_mask`; GPU test (1 GPU,
  Qwen3-0.6B or the 8B on one card) that **trainer top-k logprobs at the captured ids match
  vLLM within 1e-2** and head mass ≥ 0.99 (mirrors Miles #4109 CI check).

**P4 — Loss wiring + metrics, Megatron (3 d)** — `megatron_base_engine._row_policy_loss`,
`megatron_engine.py` (`:2080` row, `:2291` packed, `:2491` PP), `megatron_native_engine.py`,
`rl_trainer.py`, callbacks
* SC branch next to `asymmetric_clip_loss`; skip controller IS weights; `sc/*` metrics
  reduced over DP like existing metrics; example configs
  `dapo_qwen3moe_a3b_ray_megatron_verlref_4k_sc_smoke.yaml` (+ longrun) and an 8B Megatron
  SC smoke (from `dapo_qwen3_8b_ray_megatron_smoke.yaml`) with FP8 vLLM rollout.
* Skip the old-logprob recompute when SC on and `bypass_mode` (saves a full forward —
  `timing_s/old_log_prob` was 74 s of 175 s in the BF16 FSDP smoke).
* Smoke: exit 0, `sc/sampler_head_mass ≥ 0.999`, `timing/step` overhead ≤ 10 % vs SC off,
  `MoE+EP spec` line unchanged.

**P5 — FSDP2 (2–3 d)** — `fsdp_engine._prepare_packed_outputs`, `actor_worker._policy_loss_fn`
* Route through the shared op instead of the fused CE (`inplace_backward` conflict); same
  loss function; example 1/2 SC smoke configs.

**P6 — ATOM (deferred: blocked on an upstream ATOM PR)** — examples 4/5, where our measured
rollout KL is highest (0.0047–0.0088 on this box).

What ATOM does today (pin `8b6d613`):

* `SamplingParams.logprobs` is effectively a flag: `Sequence.return_logprobs =
  bool(sampling_params.logprobs)`, one float per token in `Sequence.logprobs`. There is no
  top-k log-prob path in the Python engine (only the Rust `atom/mesh` router has a
  `top_logprobs` field). Lumen-RL's `atom_ray_server.py` coerces `logprobs=0` to `True`.
* `ModelRunner` already computes a full-vocab fp32 `log_softmax(logits)` whenever any
  sequence asks for log-probs and then gathers only the sampled token, so the head is one
  cheap `topk(k)` away.
* That `log_softmax` runs on the logits **before** the sampler applies temperature (the
  sampler takes `temperatures` separately): ATOM's log-probs are pre-temperature, like
  vLLM's `raw_logprobs`. Harmless at T = 1 (all our configs), wrong at T ≠ 1 — this also
  affects ATOM's existing token-TIS path.

Needed in ATOM (one upstream PR, ~150–250 lines + tests):

1. `SamplingParams`: integer `logprobs=k` (or a separate `top_logprobs`); `Sequence` stores
   per-token `[k]` ids and log-probs next to `logprobs`.
2. `ModelRunner`: `log_probs.topk(max_k_in_batch)` after the existing `log_softmax`; TP
   broadcast like `sampled_logprobs`.
3. Deferred output (the delicate part): `prepare_sampled_ids` / `send_to_cpu_async` copy
   the sampled log-prob to CPU one step late and pair it with the *previous* batch's
   request ids; the head tensors must ride the same async-copy queue in lockstep.
4. `Scheduler`: append the head per token; where it strips rejected tokens
   (`del seq.logprobs[-strip:]`), strip the head too.
5. `LLMEngine.generate`: return `topk_ids` / `topk_logprobs` per request.
6. Temperature: divide by the per-row temperature before `log_softmax` (post-temperature
   log-probs, matching vLLM `processed_logprobs`).
7. Speculative decoding / MTP: reject head capture in the first version (examples 4/5 do
   not use it).

Needed in Lumen-RL once ATOM ships it (small): `atom_ray_server.py` passes
`topk_logprobs=k` as an int and returns `topk_ids` / `topk_logprobs` in the vLLM server's
schema (the ATOM Ray path already shares `_rollout_with_ray_vllm`, validation,
densification and both trainer backends); the guard allows `generation_backend: atom`
with an `atom_cfg.transport == "ray_http"` check (today it checks `vllm_cfg.transport`),
and requires T = 1 until the temperature fix is in; a P3-style parity test (ATOM head vs
trainer at the same ids).

Before starting: our ATOM checkout (`8b6d613`) differs from doc 08's pin (`0795f0e`), and
examples 4/5 show 5–9× the vLLM BF16 rollout KL. Base the PR on the commit Lumen-RL will
pin, and settle whether that KL gap is the commit mismatch first. A Lumen-RL-side
monkeypatch of the deferred-output/scheduler internals is possible as a stopgap but would
break on the next ATOM bump.

**P7 — Follow-ups**

Done in Lumen-RL:

* **Support replay, step 1** (Appendix A.5): `score_centering.support_replay: true` allows
  sampler `top_k < score_centering.top_k` (strictly; any `top_p`). The trainer
  renormalizes over the captured support (sampled token included) and ρ ≡ 1. Completeness
  is proven per position by a `−inf` filler slot; positions with a full head are still
  renormalized over the head and counted in `sc/frac_support_unproven`. 2-GPU parity and an
  FSDP2 smoke (`top_k = 20`, `top_p = 0.9`) pass with 0 % unproven.
* **ρ pinning** (`score_centering.rho_pin_tail`, default 1e-5): ρ := 1 where both tail
  masses are below the threshold. Measured on 5,270 Qwen3-8B rollout tokens (DAPO-Math,
  T = 1, k = 32): 39 % of tokens have both tails < 1e-5, and *all* ρ < 0.5 (3.6 % of tokens)
  come from that group; elsewhere ρ has p1 = 0.75, p50 = 1.002, p99 = 1.31 and is never
  < 0.5. The tail term there is bounded by the tail mass (< 1e-5), so pinning only removes
  fp32 noise.
* **`base_objective: ppo_clip_additive`** (NeMo RL #4818): the algorithm's DAPO/GRPO clip
  loss, with controller token-IS weights, plus the correction term `+A·Σ sg[q·w − α·p]·log p`.
  Default stays `reinforce`.
* **Controller memory instead of ragged storage**: the dense `[B, S−1, k]` head is kept on
  the host (the controller only reorders, slices and ships rows; engines move it per
  micro-batch), so it costs no controller-GPU memory (~2.8 GB on long-run configs). At
  k = 32 host memory is fine; ragged storage is not needed.
* **Vocab-parallel op** `engine/training/vocab_parallel.py`
  (`vocab_parallel_selected_log_probs`): three forward all-reduces per row chunk (max,
  sum-exp, picked logit), no backward communication; matches the full-vocab op on 2 gloo
  ranks (values identical on every rank, local-shard gradients). **Not wired**: switching
  a Megatron engine to `parallel_output=True` also needs vocab-parallel versions of the
  token log-prob (`_token_logprob_train`, `_FusedTokenLogProb`), the no-grad log-prob +
  entropy path (`_logprob_entropy_nograd`, `_cp_row_logprob_entropy`) and the old/ref
  log-prob passes. Pure memory/perf win at TP > 1; every shipped config uses TP = 1, so this
  waits for a TP > 1 config.

**Finding: with `is_mode: tis` the correction is almost always zero.** One-sided TIS
`w = min(p/q, c)` makes `q·w = p` wherever `p/q < c`, so the head residual `q·w − α·p` is
zero whenever α = min(1, c·ρ) = 1 (ρ ≥ 1/c). The correction then acts only on truncated
tokens (`p/q > c`) and where ρ < 1/c. With BF16 rollouts at this mismatch level that is
~0: the support-replay smoke measured `sc/correction_abs` 3e-9, and in the unfiltered smokes
the ~2e-4 came from the ρ-noise tokens that `rho_pin_tail` now removes. In other words,
`is_mode: tis` ≈ token-TIS REINFORCE until mismatch is large enough to truncate. Plain
score centering (`is_mode: none`, w ≡ 1) removes the drift without IS variance and is where
the correction does real work, so the long-run A/B (§6.3) must include `is_mode: none`.

Deferred (need changes inside vLLM, like P6 needs ATOM):

* **Support replay, step 2 — general top-p** (Vime / slime #2406 / Miles support mode):
  without a sampler `top_k < k`, top-p supports often exceed any fixed head (hundreds to
  thousands of tokens at high entropy). Needed: a vLLM sampler change that returns, per
  generated token, the kept support (ragged ids + offsets) with log-probs renormalized over
  it (Vime's `--return-sampling-mask --return-sampling-mask-logprobs` patch is the
  reference); in Lumen-RL, ragged transport for that payload (the `rollout_topk` ragged
  column already survives filtering/slicing), a ragged trainer op, and per-row support
  sizes in the loss.
* **FP32 rollout LM head** (Miles #4031): reduces the sampler/trainer mismatch at its
  source (the BF16 logit rounding measured in P3). Needed in vLLM: keep `lm_head` in fp32
  and compute logits from fp32 hidden states (a `compute_logits` patch via the colocate
  worker extension), and make the IPC weight sync load the BF16 trainer weight into the fp32
  parameter; on the trainer side the matching FP32 output layer. Worth doing only if the
  A/B shows mismatch still matters after score centering.

## 6. Validation plan

1. **Unit + parity** (P1–P3) as above; run on CPU in CI, GPU parity test nightly.
2. **Smoke A/B on 8B** (each 3–30 steps): example 1 (BF16) SC on vs off — SC metrics should
   be ≈ 0 correction (near on-policy); example 2 (vLLM FP8 rollout, `rollout_corr/kl ≈
   0.004`) — correction non-trivial, no instability.
3. **Long-run A/B** (the decisive one): example 2 and example 3 (FP8 E2E) for 200–300 steps,
   `{DAPO+TIS (today), REINFORCE+SC+TIS, REINFORCE+SC}`, 2 seeds; watch
   `reward/accuracy`, AIME `mean@1`, `rollout_corr/kl` trend (must not climb), entropy,
   `seq/max_len`. Success = SC ≥ baseline accuracy with flatter `rollout_corr/kl`.
4. **Stress**: a deliberately bad sampler (e.g. FP8 KV cache or INT8 W/A rollout if
   available on MI300X) where the paper predicts baseline collapse.
5. **k sweep** {16, 32, 128} on example 2 for head mass, overhead, and accuracy.
6. **MoE** (example 6, BF16 router on both sides) once P5 lands.

## 7. Risks & open questions

* **Losing PPO clip**: negligible today (single update per batch) — but if someone sets
  multiple mini-batches, REINFORCE+SC relies on TIS for staleness. Recommend `is_mode: tis`
  as default and warn when mini-batches > 1.
* **Host memory** of dense `[B, S−1, k]` at 20 K responses — k = 32 default; ragged storage
  if needed.
* **vLLM top-k overhead** on ROCm (native sampler + k logprobs): measure in P3; NeMo saw
  +1.6 % at k = 32 after flat logprobs.
* **Temperature consistency**: rely on `processed_logprobs` + trainer `z/T`; the parity test
  in P3 is the guard. Our configs use T = 1 today.
* **Interaction with the rollout_corr KL criterion in docs/release checks**: SC does not
  change the measured mismatch at step 1, so the release table stays valid; long-run KL
  should improve.
* Open: should SC become the default for FP8/ATOM rollout examples after the A/B? Decide
  after §6.3.

## 8. References

* Paper: https://arxiv.org/abs/2609.20807 (reference code: github.com/martin-marek/score-centering)
* verl: verl-project/verl #8009 (RFC), #8010, #8196, #8199, #8206;
  `verl/trainer/ppo/score_centering.py`; docs https://verl.readthedocs.io/en/latest/algo/rollout_corr.html
* NeMo RL: NVIDIA-NeMo/RL #4818 (merged), #4833 (closed), #5006, #5034; swiss-ai/Nemo-RL #19
* slime: THUDM/slime #2405, #2406, #2457, #2487
* Vime: vllm-project/vime #441, #470; `vime/utils/score_centering.py`,
  `docker/patch/latest/vllm-score-centering.patch`
* Miles: radixark/miles #3621–#3626, #3732, #3952, #4031, #4109, #4529; open #3946, #3998,
  #4204/#4310, #4462, #4489
* torchtitan: pytorch/torchtitan #5760 (draft)
* AReaL: areal-project/AReaL #1797 · Reef: Human-Agent-Society/reef #672/#673

---

## Appendix A — Design clarifications

Questions raised during design review, with the answers that shaped the plan.

### A.1 REINFORCE vs PPO, and why SC uses REINFORCE

| | PPO / DAPO clip (today) | REINFORCE (+ IS) |
|---|---|---|
| Per-token loss | `−min(r·A, clip(r, 1−ε_lo, 1+ε_hi)·A)` | `−sg[w]·A·log p_y` |
| Ratio | `r = π_θ / π_old` (trainer now vs trainer at rollout) | `w = f(p/q)` (trainer vs **sampler**) |
| What it controls | policy movement across several updates on one batch (trust region) | sampler/trainer mismatch |
| Gradient | `A·r·∇log p` while inside the clip, **0** once clipped | `w·A·∇log p` always |

* PPO's clip is a trust region: it only acts when the same batch is reused for several
  optimizer steps and `π_θ` drifts away from `π_old`.
* Lumen-RL DAPO does **one optimizer step per rollout batch**. At the update `π_θ = π_old`,
  so `r ≈ 1` (smoke `ppo_kl = 2.7e-4`) and the clip is inactive. With controller TIS the
  current loss already *is* REINFORCE with a token IS weight. Switching objectives changes
  almost nothing except dropping the dual clip `clip_ratio_c`.
* SC's estimator is derived for the score-function gradient `E_q[A·s_y]`: it subtracts
  `s̄ = Σ q_v s_v` so only `Cov_q(A, s_y)` remains. Under PPO clipping, clipped tokens
  contribute zero gradient, so subtracting `s̄` from every token no longer centers the
  actual gradient. NeMo's PPO-additive variant documents this and merged without a
  convergence A/B.
* The paper's failure mode is *drift* (distilling toward a biased sampler), not step size.
  PPO/DAPO clip collapses in the paper's quantized-sampler runs; SC does not.
* If someone later enables several mini-batches per batch, staleness returns. Then TIS
  (`is_mode: tis`) bounds it; `ppo_clip_additive` is a possible follow-up for an A/B.

### A.2 Why vocabulary parallelism is not a concern in v1

* Under Megatron TP the LM head is sharded over the vocabulary. With
  `parallel_output=True` each rank holds `V/TP` logits, and both the log-softmax and a
  top-k gather need cross-rank reductions (max, sum-exp, owner of each id).
* All three Lumen-RL Megatron engines build `GPTModel(parallel_output=False)`
  (`megatron_engine.py:784`). The output layer all-gathers the logits, so **every TP rank
  already holds full-vocab `[L, V]` logits**. Our top-k op is therefore rank-local and gives
  identical results on every TP rank; the gather's backward (a split) routes gradients to
  the right shard.
* FSDP2 has no TP; logits are full-vocab on each rank by construction.
* The cost (full `[L, V]` fp32 per TP rank + the all-gather) is already paid today by
  `packed_token_log_probs`. SC adds nothing new here.
* A vocab-parallel op (forward all-reduces of max / sum-exp / picked logits, none in
  backward; slime, Miles, torchtitan) together with `parallel_output=True` is a memory and
  bandwidth optimization only. All shipped MoE configs use TP = 1, so it is P7.

### A.3 Does every token in vLLM's top-k head have a log-prob?

Yes. In vLLM 0.23 the sampler's `gather_logprobs` computes a **full-vocabulary fp32
log-softmax**, takes `topk(k)`, and returns, per position, `[sampled, top1 … topk]` with
log-probs normalized over the whole vocabulary (not over the k tokens).

* **Sampled token is always slot 0**, even when it is not in the top-k. So `q_y` is always
  available. If `y ∈ top-k` it appears twice (slot 0 and its rank slot); we keep `y` and the
  head `H = slots 1..k` separate, and the trainer op handles the duplicate id.
* **Unfiltered sampling** (our configs): all k entries are finite.
* **`logprobs_mode` matters.** The default `raw_logprobs` is taken *before* temperature and
  filtering; it is the wrong `q` at T ≠ 1 (AReaL's likely bug). `processed_logprobs` is
  after temperature and after top-k/top-p masking: masked tokens get `−inf`. We use
  `processed_logprobs`. On ROCm this disables the aiter top-k/top-p sampler; at
  `top_p = 1, top_k = −1` the native path is already used.
* **`max_logprobs` defaults to 20.** A request with `logprobs = 32` is rejected unless the
  engine is built with `max_logprobs ≥ k`.
* **Filler slots.** When fewer than k tokens are finite (filtered sampling), the remaining
  slots carry `−inf` and arbitrary ids. Mask by logprob, never by id.
* **Precision.** The log-softmax is fp32 over the engine's (BF16/FP8) logits; the rounding is
  part of the mismatch SC corrects. Miles also offers an FP32 rollout LM head (#4031).

### A.4 Do all frameworks use the same tail estimate?

Yes. Every implementation stores only the sampler head `H` and models the sampler's tail as
the trainer's tail rescaled to the missing mass:

    q̂_v = ρ · p_v  for v ∉ H,      ρ = (1 − Σ_H q) / (1 − Σ_H p)

Because `E_p[s] = 0`, the whole tail collapses into one coefficient `α = ρ·f(1/ρ)` on the
head. The differences are in the details:

| Framework | ρ flooring | α under TIS | When the support is known complete |
|---|---|---|---|
| Paper, verl, torchtitan, Reef | numerator and denominator `clamp_min(1e-6)` (Reef: `--score-centering-min-tail-mass`) | one-sided `min(r, c)` → `α = min(1, c·ρ)` | verl #8196: detect `−inf` slots or k = V, renormalize over support |
| slime, Vime | both floored at eps | two-sided clamp | exact top-p replay, no tail |
| AReaL #1797 | both floored at eps | two-sided clamp | filtered sampling rejected |
| NeMo RL #4818 | sampler tail clamped at **0**, trainer tail at 1e-6 → ρ can be exactly 0 | two-sided: `α = min(max(1, lo·ρ), hi·ρ)`; IcePop `α = 1[lo ≤ 1/ρ ≤ hi]` | #5006 (open): renormalize over the captured finite head (ρ = 0) |
| Miles | both floored at eps | one-sided `min(c·ρ, 1)` | support mode, no tail; #4310 pins ρ ≡ 1 |

* **Flooring rarely matters.** With head mass ≥ 99.9 % both tails are far above 1e-6.
  It matters when the head covers (almost) everything: Miles #4204 had fp32 rounding leave
  `1 − mass ≈ 1e-6`, ρ ≈ 0.97 instead of 1, and a 25× too large correction.
* **α follows each framework's TIS definition.** Lumen-RL's controller TIS is one-sided
  (`clamp(max=c)`), so we use `α = min(1, c·ρ)`, matching the paper and Miles.
* **Lumen-RL choice:** paper form with both sides floored at 1e-6; when the support is
  known complete, set ρ = 1 directly instead of computing it.

### A.5 Top-p support and support replay

**Support** = the set of tokens the sampler can actually draw at a position.

* Unfiltered sampling (`top_p = 1, top_k = −1`, all our DAPO configs): the full vocabulary.
* Sampler top-k: the k most likely tokens (unrelated to SC's head size).
* Top-p (nucleus): the smallest set `S` of top tokens whose probability reaches `p`.

The sampler then draws from `q_S(v) = q(v) / Σ_{u∈S} q(u)` on `S` and **exactly 0
elsewhere**. The proportional tail model (A.4) assumes the sampler's tail looks like `ρ·p`,
which is wrong here: the true tail is 0.

**Support replay** makes the correction exact:

1. Rollout records `S` for every position, with log-probs renormalized over `S`
   (vLLM `processed_logprobs` already returns renormalized values for kept tokens and
   `−inf` for masked ones).
2. The trainer renormalizes over the same set; tokens outside `S` get probability 0
   (formulas below).
3. Everything is computed over `S` with ρ ≡ 1 and no tail.

```
p_S(v)     = p(v) / Σ_{u∈S} p(u)                for v ∈ S
p_S(v)     = 0                                  for v ∉ S
log p_S(v) = z_v/T − logsumexp_{u∈S}(z_u/T)
```

Consequences:

* **Out-of-support logits get exactly zero gradient.** `log p_S` only involves logits in
  `S`, so the trainer never pushes their mass up or down. That is correct while rollouts
  and deployment use the same top-p. Evaluating at `top_p = 1` runs a different policy.
* **Why renormalize the trainer at all.** The objective is `E_{π_S}[R]`. Comparing a
  full-softmax `p` with a renormalized `q_S` biases `p_y/q_y` low on every token
  (`q_S ≥ q`), so TIS would "correct" a mismatch that is not real and `rollout_corr/kl`
  would read high.

When `|S| < k`, the head contains `S` plus `−inf` filler, so `S` is fully captured:

```python
valid   = torch.isfinite(q_head)            # slots in S; filler ids are garbage
q       = q_head.exp()                      # vLLM already renormalized over S
lse_S   = torch.logsumexp(head_logp.masked_fill(~valid, -inf), -1)
lp_S    = head_logp - lse_S[..., None]      # trainer renormalized over S
logp_y  = logp_y - lse_S                    # sampled token too (y is always in S)
rho     = 1.0                               # hard-set; never compute 0/0
alpha   = weight_fn(1.0)                    # tail term vanishes: Σ_S q = Σ_S p_S = 1
w_head  = weight_fn((lp_S - q_head).exp()) * valid
```

Pitfalls:

1. **Set ρ = 1, do not compute it.** `(1 − Σq)/(1 − Σp)` becomes a ratio of ~1e-7 fp32
   noise (Miles #4204).
2. **`S` is defined by the sampler's `q`, not the trainer's `p`.** Use the ids vLLM returned;
   never recompute top-p on the trainer.
3. **Apply the same temperature** before renormalizing (`processed_logprobs` is
   post-temperature).
4. **Mask filler slots by `−inf` logprob**, not by id.
5. **If `|S| ≥ k` the head has no `−inf` slot**, so completeness cannot be proven and exact
   renormalization is impossible. Options: fall back to the (wrong) tail model with a
   warning (verl #8196), drop or flag the row, or ragged support replay. At top-p 0.9–0.95,
   high-entropy positions often have `|S| > 32`, so this is common unless the sampler also
   uses `top_k ≤ k`.

How each framework records `S`:

| Framework | Mechanism |
|---|---|
| Vime | vLLM patch `--return-sampling-mask --return-sampling-mask-logprobs`: ragged ids + offsets, logprobs renormalized over `S` (sampled token too) |
| slime #2406 | same idea through SGLang, full support as ragged rows |
| Miles | support mode via SGLang PR #40932; ρ pinned to 1 (#4310) |
| verl #8196 | auto "support alignment" when the head proves the support complete; else tail model + warning |
| NeMo RL #5006 (open) | renormalize over the captured finite head |

**Lumen-RL plan.** By default score centering requires `top_p = 1, top_k = −1`, where `S` is
the full vocabulary and the tail model is correct. P7 step 1 (done) adds
`score_centering.support_replay`, which allows sampler `top_k < score_centering.top_k` (any
`top_p`). P7 step 2 adds general top-p via a ragged sampling-mask patch.

**Measured: vLLM's top-k keeps ties.** With `top_k = 20`, Qwen3-8B BF16 rollouts had
supports of up to 25 tokens: vLLM masks logits strictly below the k-th value, so every
token tied with it survives, and BF16 logits tie often (≈0.25-nat steps). So "`top_k ≤ k`"
does not guarantee the support fits in a k-token head. The implementation therefore
(a) requires `top_k < score_centering.top_k` strictly, (b) treats a position as proven
complete only when its head contains at least one `−inf` filler slot, and (c) still
renormalizes unproven positions over the head (NeMo #5006 behaviour) while reporting
their share as `sc/frac_support_unproven`. Parity check (`top_k = 20`, k = 32; also with
`top_p = 0.9`, T = 0.7): every position proven, sampled token always inside `S`, sampler
mass over `S` = 1.000000, trainer `p_S` vs sampler `q_S` mean |Δ| 0.007–0.022 on the
sampled token (BF16 noise). Note: under filtered sampling `rollout_corr/kl` compares the
renormalized rollout log-prob with the trainer's full-softmax log-prob, so it reads
higher than the mismatch score centering actually sees.
