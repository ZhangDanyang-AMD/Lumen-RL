# LumenRL release validation

Measurement record behind the reference table in
[`examples/docs/08-release.md`](../examples/docs/08-release.md) §8.5.1. Newest round
first.

---

# 2026-09-21 — image `260921`: ATOM #2267, aiter rebased onto upstream, asm decode dodged

| | |
|---|---|
| Image | `zhangdanyangamd/lumen-rl:dapo-gfx950-rocm7.2.3-260921`, digest `sha256:de47df3cf5d2…` |
| Lumen-RL | `7933758`, the parent of the commit this record ships in |
| ATOM | `0795f0eae2d6` on `main` (PR #2267 merge), baked |
| aiter | `c395c62886e2` on `lumen/moe`, rebased onto ROCm/aiter `main` `6a9a005b7`, baked |
| Hardware | one node, 8x MI355X (gfx950), whole-node allocation |
| Seed | `10086` |
| Metrics read at | step 1 (`step=1`) |

## What this round changed

**ATOM `8b6d61392b06` → `0795f0eae2d6`.** PR #2267 merged on 2026-09-20, bringing the
two rollout fixes this line had been carrying out of tree: a capture no longer writes
cache through the last served batch's rows, and a wake re-allocates the KV pool at the
size it slept at.

**aiter `lumen/moe` rebased onto ROCm/aiter `main` `6a9a005b7`.** Not optional: the new
ATOM imports `topk_select` from aiter at module scope in `atom/model_ops/sampler.py`
(also `embed_head`, `rejection_sampler`, `model_runner_ext`), and that symbol arrives
with ROCm/aiter#5499 — after the branch's old base, on which the rollout cannot import
at all. The rebase crosses ROCm/aiter#5149, which reverted Triton modules Lumen needs;
upstream has since re-landed most of them, and the 148 files plus
`arch_info.is_cdna4` it has not are restored at their pre-revert content. None of the
restored paths exists on `6a9a005b7`, so nothing upstream is overwritten. Checked by
importing all 137 aiter modules the three repos reference: 134 succeed, and the 3 that
do not were equally absent before the rebase and sit behind `try`/`except` probes.

**`ATOM_FORCE_ATTN_TRITON=1` on examples 4, 5 and 9.** ATOM's assembly paged-decode
kernel returns finite but wrong output when a sequence's context occupies exactly 16
pages with a partial last one. It is invisible in `abs_diff` and shows only in the
quadratic `chi2_token`:

| example 4, per step | `chi2_token` | `abs_diff` |
|---|---|---|
| without the variable | 0.147 / 0.0138 / **5761.69** | 0.0311 / 0.0370 / 0.0473 |
| with it | 5.39 / 0.0411 / 0.113 | 0.0344 / 0.0391 / 0.0303 |

ATOM's own fix (`fix/asm-paged-decode-partial-16th-page`) is not in the pinned commit.

**20 of the 25 baked aiter kernels were rebuilt.** 257 upstream commits changed the
`csrc` behind them. One was loudly stale — `module_deepgemm_opus` lost
`opus_gemm_a16w16_launch` when ROCm/aiter#4961 unified the OPUS GEMM interfaces, and
example 9 aborted on it — but the rest are the greater hazard: same symbol, older
kernel, no error. Examples 4 and 5 passed against the stale set, which is why "it still
passes" was not accepted as evidence here. Rebuilding used the incremental object cache,
7–47 s each except two CK blockscale modules at ~400 s.

## Runs

10 runs, all exit 0, all four error counts 0, every weight-sync bucket `skipped=0`,
`--check` 10/10 PASS against the table in `08-release.md` §8.5.1.

| # | span | `k3_kl` | Δ vs reference | `entropy` | `kl` |
|---|---|---|---|---|---|
| 1 | 191 s | 0.0011179 | +5.5% | 0.6143 | 0.00117633 |
| 2 | 161 s | 0.00471458 | −5.3% | 0.794294 | 0.00432843 |
| 3 | 161 s | 0.00480643 | +19.0% | 0.774225 | 0.00447058 |
| 4 a | 492 s | 0.00459888 | +12.7% | 0.570037 | 0.00427408 |
| 4 b | 552 s | 0.00338652 | −17.0% | 0.548062 | 0.00298126 |
| 4 c | 471 s | 0.00426389 | +4.5% | 0.714157 | 0.00385979 |
| 5 | 396 s | 0.000841038 | −10.1% | 0.531491 | 0.000893993 |
| 6 | 562 s | 0.00136204 | −13.8% | 0.538206 | 0.00159133 |
| 7 | 536 s | 0.00166141 | +5.2% | 0.631309 | 0.00155714 |
| 9 | 587 s | 0.00157512 | +14.1% | 0.682081 | 0.00118207 |

Example 4's Δ is against its new reference; every other row is against the reference it
already had.

## Example 4's reference was re-measured; the other seven were re-confirmed

The Triton decode path moves example 4: three runs at 0.00339 / 0.00426 / 0.00460, mean
**0.00408**, against the 0.00287 it replaces. The old reference rejects the high end at
+60.2%, so this is a shift rather than spread — the three new runs deviate at most 17.0%
from their own mean, which is tighter than the 0.00241 / 0.00243 / 0.00377 the old
reference was built from. `entropy` moves with it to **0.611**.

Examples 5 and 9 gained one sample each, −10.1% and +14.1%. Replacing a 4-run and a
3-run mean with a single observation is a worse estimator, so they keep their values —
the same argument this record made in 2026-09-10.

## Example 9's sleep path

48 releases and 48 resumes, balanced (8 replicas × 2 memory tags × 3 steps), zero
negative KV-pool derivations, zero `Memory access fault`, zero `grad_norm=nan` — the
three symptoms PR #2267 fixes, none of them present.

## Limits of this record

- Examples 1, 2, 3, 5, 6, 7 and 9 have **one run each on this image**. Only example 4
  has three, and only its reference was re-derived.
- The `chi2_token` comparison for `ATOM_FORCE_ATTN_TRITON` is **one run either side** on
  example 4. The direction is not in doubt — 5761.69 against a per-step maximum of 5.39
  — but the magnitude is a single observation, and step 1 is *higher* with the variable
  set (5.39 against 0.147), which one pair of runs cannot explain.
- The five kernels not rebuilt were judged unaffected by diffing their `csrc` sources,
  not by running anything that isolates them.
- Reference values remain smoke-scale (1–3 steps). They say a stack reproduces, not that
  it trains well.

---

# 2026-09-17 — image `260917`: ATOM main baked in, measured with nothing mounted

| | |
|---|---|
| Image | `zhangdanyangamd/lumen-rl:dapo-gfx950-rocm7.2.3-260917`, digest `sha256:51ecafa95d1a…` |
| Lumen-RL | `e514596`, **from the image** — only `$DATA_ROOT` was mounted, so this is what a `docker pull` gets |
| ATOM | `8b6d61392b06` on `main`, baked |
| Hardware | one node, 8x MI355X (gfx950), whole-node allocation |
| Command | `run_dapo.sh` from the image, judged with `run_example.sh <N> --check-only` |
| Seed | `10086` | 
| Metrics read at | step 1 (`step=1`) |

## How the image was made, and why not with `build_image.sh`

ATOM is a source tree on `PYTHONPATH` — no `pip install`, nothing compiled, not even in
the image's `ENV PYTHONPATH` (`run_dapo.sh` adds it at runtime). So re-pinning it is a
file replacement: a container off `260910`, ATOM re-created the way `release/Dockerfile`
does it (`git init`, fetch the SHA, `checkout FETCH_HEAD`), then `docker commit`. **28 s
to build, 20 s to push** — one new layer — against ~60 min for a full rebuild. Entrypoint,
cmd and workdir are restored explicitly on the commit.

Two things came along with it:

- **The baked Lumen-RL is `e514596`**, the rebased checkout, rather than the branch tip
  `build_image.sh` would have resolved. That is what made a no-mounts run meaningful.
- **24 aiter kernels instead of 16.** The published image bakes 16; running the examples
  needs 24, and the other 8 JIT-compile on first use. That is not new and not
  ATOM-specific — a container on the *old* image with the *old* ATOM also ends at 22 —
  but it undercuts the image's "no first-run compilation" claim, so the 8 were taken
  from a container that had built them in real runs and committed too. Verified: a run
  on `260917` compiles **zero** kernels and the count stays at 24.

## Runs

`skipped` is weight-sync buckets that did not account for every tensor, out of `buckets`.

| ex | span | exit | errors | skipped / buckets | `rollout_corr/k3_kl` | `entropy` | `rollout_corr/kl` | `ppl_ratio` |
|---|---|---|---|---|---|---|---|---|
| 1 | 163 s | 0 | 0 | 0 / 0 | 0.0010942 | 0.624976 | 0.0010722 | 1.00085 |
| 2 | 127 s | 0 | 0 | 0 / 0 | 0.0048248 | 0.770634 | 0.00473415 | 1.00397 |
| 3 | 137 s | 0 | 0 | 0 / 0 | 0.00355718 | 0.799204 | 0.00317713 | 1.00518 |
| 4 | 472 s | 0 | 0 | 0 / 336 | 0.00287993 | 0.477497 | 0.00268712 | 1.00198 |
| 5 | 412 s | 0 | 0 | 0 / 112 | 0.000739239 | 0.360857 | 0.000607351 | 1.00126 |
| 5 | 375 s | 0 | 0 | 0 / 112 | 0.00102973 | 0.660954 | 0.00124422 | 0.997257 |
| 5 | 377 s | 0 | 0 | 0 / 112 | 0.00090947 | 0.566824 | 0.000815409 | 1.00206 |
| 5 | 375 s | 0 | 0 | 0 / 112 | 0.00106485 | 0.870339 | 0.00104208 | 1.0003 |
| 6 | 527 s | 0 | 0 | 0 / 0 | 0.00147054 | 0.679457 | 0.00168567 | 1.00143 |
| 7 | 496 s | 0 | 0 | 0 / 0 | 0.00154856 | 0.57809 | 0.0012452 | 1.00072 |
| 9 | 575 s | 0 | 0 | 0 / 2352 | 0.00151614 | 0.605641 | 0.00125785 | 1.00158 |

**11/11 exit 0, zero error lines, zero skipped buckets, `--check` 11/11 PASS.**

## Example 5's reference, and a conclusion the round before got wrong

The previous round replaced example 5's `k3_kl` reference with `0.00143`, from three runs
that came in at 0.00133–0.00151, and read the gap from the old single-run `0.000930` as a
systematic shift caused by releasing sleep. **Four runs on this image average 0.000936**
— within 1% of the value that was replaced — and include a 0.000739. The three-run
cluster was luck, and "the spread within a mode is ±6%" was an artifact of it.

So `k3_kl` goes back to essentially where it was, now with four samples behind it. What
genuinely needed changing is `entropy`: the four runs span 0.361 to 0.870, and the old
`0.568` fails the high end at +53%. It becomes the 4-run mean `0.615` at the ±60% floor
the MoE examples already use, which the observed −41% / +42% fits with margin.

The same correction applies to examples 4 and 9: the previous round measured them 11%
above their references under releasing sleep and called it a shift. On this image example
4 lands at **+0.3%** and example 9 at **+9.9%**. Releasing sleep does not move these
metrics; three samples of a 4096-token example do not describe it.

| ex | reference (runs) | this round | delta | tolerance |
|---|---|---|---|---|
| 1 | 0.00106 (1) | 0.0010942 | +3.2% | ±30% |
| 2 | 0.00498 (1) | 0.0048248 | −3.1% | ±30% |
| 3 | 0.00404 (1) | 0.00355718 | −12.0% | ±30% |
| 4 | 0.00287 (3) | 0.00287993 | +0.3% | ±50% |
| 5 | **0.000936 (4, new)** | 0.000739–0.00106 | −21% / +14% | ±50% |
| 6 | 0.00158 (1) | 0.00147054 | −6.9% | ±50% |
| 7 | 0.00158 (1) | 0.00154856 | −2.0% | ±50% |
| 9 | 0.00138 (3) | 0.00151614 | +9.9% | ±50% |

## Limits of this record

- **The image was committed, not built.** Faithful for ATOM, which is not compiled, but
  it is not reproducible from `Dockerfile` + `versions.env` alone. A full
  `build_image.sh` against this pin has not been done.
- **The baked Lumen-RL `e514596` is not pushed**, so the image carries a SHA that does
  not exist on the remote. `run_example.sh` mounts over it anyway.
- **One sample each for examples 1, 2, 3, 4, 6, 7 and 9**, four for example 5. Example
  5's history is the argument for treating any single-run reference as provisional.
- **Long runs, wandb, gfx942 and a cold `docker pull` were not exercised.**

---

# 2026-09-16 — ATOM re-pinned to `main`, and sleep no longer pinned resident

> ⚠️ Two conclusions below were corrected by the 2026-09-17 round: that releasing sleep
> shifts `k3_kl` upward, and the example 5 reference of `0.00143` it led to. Both came
> from three runs that happened to cluster. The rest of the round stands.

| | |
|---|---|
| Image | `zhangdanyangamd/lumen-rl:dapo-gfx950-rocm7.2.3-260910`, digest `sha256:cc18a3f5ce16…` — **unchanged**; ATOM was bind-mounted over, not rebuilt |
| ATOM | `8b6d61392b0690ff338b6ccd6b864a63ecaf45c3` on `main`, bind-mounted at `/opt/lumenrl/ATOM` |
| Lumen-RL | `d6005c0` on `dev/dapo_release`, bind-mounted |
| Hardware | one node, 8x MI355X (gfx950), whole-node allocation |
| Command | `run_dapo.sh` directly (§8.4.5 manual path), judged with `run_example.sh <N> --check-only` |
| Seed | `10086`, fixed inside `run_dapo.sh` |
| Metrics read at | step 1 (`step=1`) |
| Date | 2026-09-16 |

## What this round changed

- **ATOM moved from the PR head to main**, `d6b9e147cbf6` → `8b6d61392b06`. PR #2028 was
  squash-merged on 2026-09-16, so the pin is now an ordinary commit on `main`. The
  difference is not only the squash: main had advanced 33 commits past the PR's rebase
  base, so this round measures our changes *plus* that advance.
- **`_pin_sleep_keeps_memory_resident` was removed**, so sleep follows ATOM's default and
  releases. The reason is below.

The image itself was not rebuilt for this round — ATOM is only put on `PYTHONPATH`, so a
bind mount is faithful. A rebuild against this pin is the next step, and until it lands
the published image still carries `d6b9e147cbf6`.

## The pin was removed because the failure it guarded against does not exist

`versions.env` justified `sleep_keeps_memory_resident=true` with "example 9 aborts all
eight replicas in `resume_memory` with a negative KV pool". That **did not reproduce**,
on either ATOM:

| | example 9, releasing sleep | result |
|---|---|---|
| ATOM `8b6d61392b06` (this pin) | run 1 | exit 0, `--check` PASS |
| ATOM `8b6d61392b06` | run 2 | exit 0, `--check` PASS |
| ATOM `d6b9e147cbf6` (image) | 2026-09-16, earlier | exit 0, `--check` PASS |

Every run released and recaptured 24 / 24 times (8 replicas x 3 steps) and derived
`available_for_kv < 0` exactly **zero** times. A VRAM sampler at 2 s resolution shows the
8-card total falling from a ~1137 GB peak into the single digits inside each sleep
window, so the release is real and not merely reported.

Why the original observation stood is not established. Lumen-RL moved from `8ca6bdd` to
`d6005c0` in between, and the margin was always thin (−1435 MB against an 86 GB budget,
1.7%), so it may have been a genuine edge that the framework drifted off. **The honest
statement is that the documented symptom cannot be produced today**, which is not a
reason to keep a pin.

What the PR *did* fix is the PIECEWISE half, which no example exercises: probe-to-probe,
the old ATOM released 0 graphs and faulted on wake, the new one releases 1332 and
survives two sleep/wake cycles. PIECEWISE still cannot run the RL loop at all — 8/8
replicas abort on the first replay — which is why `_pin_cudagraph_mode` stays.

## Runs

`span` is the first-to-last timestamp in the trainer log. `errors` is the combined count
of `Traceback`, `OutOfMemory`, `CUDA error` and `HSA_STATUS`. `skipped` is the number of
weight-sync buckets that did not account for every tensor, out of `buckets` total.

| ex | sleep | span | exit | errors | skipped / buckets | `rollout_corr/k3_kl` | `entropy` | `rollout_corr/kl` | `ppl_ratio` | `response_length/mean` |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | — | 157 s | 0 | 0 | 0 / 0 | 0.00117866 | 0.674346 | 0.000978041 | 1.00022 | 413.78 |
| 2 | — | 127 s | 0 | 0 | 0 / 0 | 0.00510776 | 0.780022 | 0.00492439 | 1.00414 | 426.86 |
| 3 | — | 137 s | 0 | 0 | 0 / 0 | 0.00368384 | 0.766134 | 0.00325046 | 1.00524 | 425.45 |
| 4 | resident | 472 s | 0 | 0 | 0 / 336 | 0.00328873 | 0.707288 | 0.00331959 | 0.999701 | 703.13 |
| 4 | releasing | 475 s | 0 | 0 | 0 / 336 | 0.00364331 | 0.654406 | 0.00308988 | 0.996492 | 860.22 |
| 5 | resident | 397 s | 0 | 0 | 0 / 112 | 0.00110076 | 0.629561 | 0.00108828 | 1.00193 | 721.11 |
| 5 | resident | 375 s | 0 | 0 | 0 / 112 | 0.000894251 | 0.452552 | 0.000821423 | 1.00002 | 660.13 |
| 5 | releasing | 380 s | 0 | 0 | 0 / 112 | 0.00150943 | 0.648098 | 0.00168579 | 1.00545 | 865.63 |
| 5 | releasing | 403 s | 0 | 0 | 0 / 112 | 0.00143788 | 0.699219 | 0.00124357 | 1.00108 | 763.63 |
| 5 | releasing | 402 s | 0 | 0 | 0 / 112 | 0.00133292 | 0.683473 | 0.00178288 | 1.00224 | 828.23 |
| 6 | — | 523 s | 0 | 0 | 0 / 0 | 0.00146238 | 0.566913 | 0.00157363 | 1.00187 | 676.14 |
| 7 | — | 507 s | 0 | 0 | 0 / 0 | 0.00150788 | 0.736595 | 0.00144256 | 1.00165 | 689.78 |
| 9 | resident | 678 s | 0 | 0 | 0 / 2352 | 0.00145721 | 0.745864 | 0.00101772 | 1.00229 | 813.55 |
| 9 | releasing | 570 s | 0 | 0 | 0 / 2352 | 0.00150982 | 0.778577 | 0.00140162 | 1.00151 | 846.41 |
| 9 | releasing | 572 s | 0 | 0 | 0 / 2352 | 0.0017224 | 0.858394 | 0.00155727 | 1.00177 | 758.70 |

**15/15 exited 0 with zero error lines and zero skipped buckets.** Example 9's expert
relayout reported `12288 expert slices across 96 fused buffers` x 24, identical to the
image's ATOM.

## Example 5's reference was re-measured; the other seven were re-confirmed

Releasing sleep shifts `k3_kl` upward on all three ATOM examples, by roughly the same
fraction:

| ex | resident | releasing | shift |
|---|---|---|---|
| 4 | 0.00328873 | 0.00364331 | +11% |
| 5 | 0.00100 (mean of 2) | 0.00143 (mean of 3) | +43% |
| 9 | 0.00145721 | 0.00162 (mean of 2) | +11% |

Only example 5 was pushed out of band by it: its reference was `0.000930` from a single
run, the smallest base in the table, so +43% consumed most of a ±50% tolerance and two of
the three releasing runs failed `--check`. The spread *within* each mode is ±6%, so this
is a shift and not jitter, and **example 5's reference is now the 3-run releasing mean**:
`k3_kl 0.00143`, `entropy 0.677`, `kl 0.00157`. All five example-5 runs of this round,
in both sleep modes, pass against it.

Examples 4 and 9 shift the same way but sit far enough inside their bands to absorb it
(+26.9% and +24.8% of a ±50% tolerance), so their references are **left alone** rather
than replaced by a 1- or 2-run mean — the same argument the previous round made.

| ex | reference (runs) | this round | delta | tolerance | verdict |
|---|---|---|---|---|---|
| 1 | 0.00106 (1) | 0.00117866 | +11.2% | ±30% | PASS |
| 2 | 0.00498 (1) | 0.00510776 | +2.6% | ±30% | PASS |
| 3 | 0.00404 (1) | 0.00368384 | −8.8% | ±30% | PASS |
| 4 | 0.00287 (3) | 0.00364331 | +26.9% | ±50% | PASS |
| 5 | **0.00143 (3, new)** | 0.00143788 | +0.6% | ±50% | PASS |
| 6 | 0.00158 (1) | 0.00146238 | −7.4% | ±50% | PASS |
| 7 | 0.00158 (1) | 0.00150788 | −4.6% | ±50% | PASS |
| 9 | 0.00138 (3) | 0.0017224 | +24.8% | ±50% | PASS |

## What releasing costs

One graph recapture per step, and it is cheaper than feared. Per-step `timing_s/step`:

| ex | resident | releasing | delta |
|---|---|---|---|
| 9 | 64.0 / 46.7 / 49.3 | 65.5 / 47.1 / 51.5 and 64.2 / 47.9 / 50.0 | +1.1 s/step (+2%) |
| 4 | 50.0 / 33.2 / 35.8 | 51.2 / 35.4 / 35.9 | +1.1 s/step (+3%) |

## Limits of this record

- **The image was not rebuilt.** Everything here is ATOM bind-mounted into the `260910`
  image. A build against `8b6d61392b06`, and a re-run of the three ATOM examples on it
  with no mounts, is still owed.
- **Lumen-RL's tree was not clean.** `d6005c0` plus four uncommitted files: the two MoE
  ATOM configs at `gpu_memory_utilization=0.45` instead of 0.30, and KV-pressure and
  alignment diagnostics in the two Ray servers. The previous round's ATOM numbers were
  taken on the same tree, so the comparison holds, but these are not `d6005c0`'s numbers.
- **One sample for examples 1, 2, 3, 6, 7 and for each of 4 and 9's two sleep modes.**
  Only example 5 has three per mode.
- **Compile caches must be cleared between models, not just between modes.** ATOM keys
  its torch.compile cache by `mode/actor_id/replica/rank` with no model identity, so an
  8B graph is happily reused for the 30B MoE and dies in `assert_size_stride` with
  `stride 2048==4096`. Three example-9 runs were lost to this before it was spotted;
  §8.6.2 documents the mode rule but not the model rule.
- **PIECEWISE was not re-probed on this ATOM.** The new-vs-old graph-release comparison
  quoted above was measured on the pre-merge branch, whose content is identical to this
  commit for the files in question but was not re-run here.

---

# 2026-09-10 — ATOM re-pinned to the head of PR #2028

| | |
|---|---|
| Image | `zhangdanyangamd/lumen-rl:dapo-gfx950-rocm7.2.3-260910`, digest `sha256:cc18a3f5ce16…` |
| Lumen-RL | `8ca6bdd` on `dev/dapo_release`, bind-mounted from a checkout — **not** taken from the image |
| | The tree also carried the `release/` and `examples/docs/` changes that the commit adding this record then made (this image tag, this ATOM SHA, this file). Nothing the trainer imports, so the runs are `8ca6bdd`'s. |
| Hardware | one node, 8x MI355X (gfx950), whole-node allocation |
| Command | `bash release/run_example.sh <N> --check` |
| Seed | `10086`, fixed inside `run_dapo.sh` |
| Metrics read at | step 1 (`step=1`) |
| Date | 2026-09-10 |
| Stack in image | `vllm 0.23.0`, `flydsl 0.3.2`, `transformers 5.12.0`, `aiter` resolving to `/opt/lumenrl/aiter/` |

**A result is identified by the image digest and the Lumen-RL commit together.** The
launcher mounts the checkout it lives in over `/opt/lumenrl/Lumen-RL`, so the image no
longer pins the framework code and quoting the digest alone does not describe a run.
Every run prints both, and the launcher recreates its container when either changes.

All 8 cards were at the ~298 MB idle baseline before each run, and the launcher
restarts the container between runs.

## What this round changed

Two things at once, which is why every example was re-run rather than the ATOM ones:

- **ATOM moved to the current head of PR #2028**, `28721a5094b9` → `d6b9e147cbf6`.
- **Lumen-RL was rebased onto `main`** (`7545e2b` → `3d736a2`), which is 12 commits of
  framework change under the same examples.

## Image build: this image was not built, it was re-pinned

ATOM is only put on `PYTHONPATH` — there is no `pip install` and nothing is compiled —
so the tree was replaced the way `release/Dockerfile` creates it (`git init`, fetch the
SHA, `checkout FETCH_HEAD`) inside a container off `260908`, and the result committed.
That takes 18 s and reuses the baked aiter kernels, against ~40 min for a rebuild.
The push uploaded one layer; every other layer was already in the registry.

`docker commit` restores `ENTRYPOINT`, `CMD` and `WORKDIR` explicitly, and the
entrypoint's own version report was checked against `versions.env` afterwards:

```
Lumen-RL    f4439f38b5dd      <- the baked fallback copy, not what runs
Lumen       e6379cbd9057
aiter       4ebe6d69c7f4
ATOM        d6b9e147cbf6
```

⚠️ The baked Lumen-RL copy is the tip `dev/dapo_release` had at build time and is now
an orphaned SHA, because the branch was rebased after the image was made. It only
exists so the editable install has a path to point at; `run_example.sh` mounts over it.
A bare `docker run` without that mount lands on pre-rebase code.

## Runs

`span` is the first-to-last timestamp in the trainer log. `errors` is the combined
count of `Traceback`, `OutOfMemory`, `CUDA error` and `HSA_STATUS`. `skipped` is the
number of weight-sync buckets that did not account for every tensor in them, out of
`buckets` total; examples on the vLLM path emit no such line and count 0 of 0.

| ex | span | exit | errors | skipped / buckets | `rollout_corr/k3_kl` | `entropy` | `rollout_corr/kl` | `ppl_ratio` | `response_length/mean` |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 144 s | 0 | 0 | 0 / 0 | 0.00110113 | 0.654183 | 0.000999021 | 1.00089 | 407.33 |
| 2 | 129 s | 0 | 0 | 0 / 0 | 0.00517972 | 0.782929 | 0.00507141 | 1.00427 | 425.63 |
| 3 | 142 s | 0 | 0 | 0 / 0 | 0.00396365 | 0.806861 | 0.00371948 | 1.00568 | 434.39 |
| 4 | 508 s | 0 | 0 | 0 / 336 | 0.00310421 | 0.631050 | 0.00364320 | 1.00406 | 803.36 |
| 5 | 402 s | 0 | 0 | 0 / 112 | 0.000955007 | 0.618662 | 0.000948457 | 1.00109 | 736.41 |
| 6 | 523 s | 0 | 0 | 0 / 0 | 0.00154296 | 0.720107 | 0.00151181 | 1.00167 | 812.75 |
| 7 | 499 s | 0 | 0 | 0 / 0 | 0.00145876 | 0.744816 | 0.00155241 | 1.00135 | 753.10 |
| 9 | 579 s | 0 | 0 | 0 / 2352 | 0.00143179 | 0.564593 | 0.00150571 | 1.00186 | 851.48 |

**8/8 exited 0 with zero error lines and zero skipped buckets, and `--check` is
8/8 PASS.**

## The references were not re-measured, they were re-confirmed

| ex | reference (runs, image `260908`) | this round | delta | tolerance |
|---|---|---|---|---|
| 1 | 0.00106 (1) | 0.00110113 | +3.9% | ±30% |
| 2 | 0.00498 (1) | 0.00517972 | +4.0% | ±30% |
| 3 | 0.00404 (1) | 0.00396365 | −1.9% | ±30% |
| 4 | 0.00287 (3) | 0.00310421 | +8.2% | ±50% |
| 5 | 0.000930 (1) | 0.000955007 | +2.7% | ±50% |
| 6 | 0.00158 (1) | 0.00154296 | −2.3% | ±50% |
| 7 | 0.00158 (1) | 0.00145876 | −7.7% | ±50% |
| 9 | 0.00138 (3) | 0.00143179 | +3.8% | ±50% |

`k3_kl` lands within ±8.2% on all eight, against tolerances of 30–50%, so **the
references in `run_example.sh` and §8.5.1 are left exactly as they were.** They are
means over the 12 runs on the previous image; replacing a 3-run mean with one run on
this image would be a worse estimator, not a fresher one, and example 4 is the standing
argument for that — its three runs came in at 0.00241, 0.00243 and 0.00377, so any one
of them alone would misjudge the others.

`entropy` moves more, from −18.4% (example 9) to +16.9% (example 4), and stays inside
its bands. That is the metric the chapter already describes as a coarse batch-level
sanity check on a 4k run; the eight `k3_kl` figures are what say the stack reproduces.

**No example changed which side of a tolerance it is on, in either metric.** ATOM
moving one PR head and the framework gaining 12 commits of `main` did not move a
published number out of band.

## Four defects on `main` that this sweep found

The first pass of the sweep failed five of the eight examples. None of the causes were
in the release path; all four were latent on `main` and are fixed in the commits above
this record.

| examples | symptom | cause |
|---|---|---|
| 1, 2, 3, 5, 6 | `TypeError: OptimizerConfig.__init__() got an unexpected keyword argument 'optimizer'` | the shared `optimizer_config` dict carried Megatron's spelling of the field, which the FSDP2 dataclass does not declare |
| 6, 7 | `ValueError: moe_backend='' is not supported for unquantized MoE` | `VLLMConfig` declared `moe_backend` twice; the second declaration replaced the `"auto"` default the trainer's guard checks for |
| 7 | `AttributeError: module 'lumenrl.engine.training.dsv4_megatron_bridge' has no attribute 'is_dsv4'` | dropped when that module was rewritten, while `MegatronNativeEngine` still calls it for every model |
| 7 | `RuntimeError: The size of tensor a (4254) must match the size of tensor b (4255)` | the position-bucket diagnostic subtracted three tensors that share a frame but not a width |

⚠️ **Superseded on 2026-09-16.** The fifth failure below is the one that stopped
reproducing; the pin it motivated has since been removed. The paragraph is kept as
written because it is what was observed at the time. See the 2026-09-16 round.

The fifth failure was the ATOM re-pin itself and is the reason
`sleep_keeps_memory_resident` is now pinned: see `versions.env` and §8.1.1. Example 9
reached step 0, synced weights with `skipped=0`, and then aborted all eight replicas in
`resume_memory` with a negative KV pool, because ATOM now re-derives the block count on
every wake and the colocated trainer is 52 GB of the budget it subtracts.

⚠️ **Releasing the actors' allocator cache before the wake does not fix that** — it was
tried and moves `non_torch` by 0.6 GB. `non_torch` is derived from device-used, and on
this ROCm version freed memory is not returned to the driver, so `empty_cache()` barely
moves it; the behaviour is version-dependent and some ROCm versions do return it. This
is the same effect §8.6.1 documents as a card still holding ~90.9 GB after a clean run.
Only keeping the pool resident works.

## Limits of this record

- **One sample per example this round.** The per-example spread is inherited from the
  12 runs on `260908`, not re-established here; this round tests whether the published
  references still hold, which is a weaker claim than establishing them.
- **Examples 1, 2, 3 and 6 were run at `8ca6bdd`, as were 4, 5, 7 and 9**, but in two
  batches — 4/5/7/9 first, then 1/2/3/6 — so page-cache state differs between them.
  It affects `span`, not the metrics.
- **`span` is not a benchmark.** It is dominated by how warm the kernel and page caches
  are. Example 6's 523 s here against 1202 s on `260908` is that single earlier run
  paying the first read of the 57 GB checkpoint, not a speedup.
- **Long runs were not carried to completion.** Only the smoke configs ran. `--longrun`
  is exercised by `--dry-run`, which selects the right yaml, sets `STEPS=1000` and
  handles a missing wandb key.
- **Logging to a real wandb project was not exercised**, only the no-key path.
- **gfx942 was not tried.** This image is built for gfx950.
- **The from-scratch `docker pull` path was not re-timed.** The pushed digest was
  confirmed against the local image and the manifest read back from the registry, but
  no run started from a cold pull.
- **`sleep_keeps_memory_resident` was not measured against releasing on the 8B
  examples.** Both work there; the pin is justified by example 9, where releasing does
  not, and by it being the behaviour the previous ATOM had unconditionally. What the
  per-step cost of a graph recapture actually is on 4 and 5 was not quantified.
  *(2026-09-16: both were measured. Releasing works on example 9 too, and costs about
  1 s/step. The pin is gone.)*
- **The pin sidesteps the KV budget problem rather than fixing it.** The `non_torch`
  figure that makes the budget negative is itself drift in ROCm/ATOM `main` between the
  PR's merge base and `8938787d`: a separate A/B on `main` plus only the `n>1` fan-out
  fix reproduces `non_torch=52.88GB` and `available_for_kv=-29.93GB`, i.e. the same
  numbers with the whole MoE weight-sync path out of the picture. So an ATOM that
  releases will keep needing the pin until that is found, and finding it is upstream
  work this record does not cover. Reproduction worktrees are kept outside git.
