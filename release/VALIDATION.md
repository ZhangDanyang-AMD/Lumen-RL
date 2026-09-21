# LumenRL release validation

Measurement record behind the reference table in
[`examples/docs/08-release.md`](../examples/docs/08-release.md) §8.5.1. Newest round
first.

---

# 2026-09-21 — image `260921b`: ATOM #2267, aiter rebased onto upstream, whole table re-measured

| | |
|---|---|
| Image | `zhangdanyangamd/lumen-rl:dapo-gfx950-rocm7.2.3-260921b`, digest `sha256:eede1d8fcdf5…` |
| Lumen-RL | `7933758`, the parent of the commit this record ships in |
| ATOM | `0795f0eae2d6` on `main` (PR #2267 merge), baked |
| aiter | `d6303e316f9e` on `lumen/moe`, rebased onto ROCm/aiter `main` `6a9a005b7`, baked |
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
at all.

**`ATOM_FORCE_ATTN_TRITON=1` for every `atom*` `MODE`,** set in `run_dapo.sh` and listed
again in the launcher's table. ATOM's assembly paged-decode kernel returns finite but
wrong output when a sequence's context occupies exactly 16 pages with a partial last
one. See below for what it does and does not move.

**20 of the 25 baked aiter kernels were rebuilt.** 257 upstream commits changed the
`csrc` behind them. One was loudly stale — `module_deepgemm_opus` lost
`opus_gemm_a16w16_launch` when ROCm/aiter#4961 unified the OPUS GEMM interfaces, and
example 9 aborted on it — but the rest are the greater hazard: same symbol, older
kernel, no error. Examples 4 and 5 passed against the stale set, which is why "it still
passes" was not accepted as evidence here.

## The rebase broke Lumen's import surface, and a module-level check could not see it

ROCm/aiter#5149 deleted the Triton tree #4978 added. Upstream re-landed most of it, and
the first restore covered the 148 files plus `arch_info.is_cdna4` it had not. Three more
files upstream re-landed **rewritten**, with the kernels Lumen imports renamed or
dropped. The check used at the time — import all 137 aiter modules the three repos name
— passes on those: the module imports and only the name is gone.

It was not cosmetic. `lumen/ops/quantize/ops.py` guards that import with
`except ModuleNotFoundError`, and a missing *name* raises plain `ImportError`, so
`lumen.ops.quantize`, `lumen.quantize`, `lumen.ops.attention` and
`lumen.kernels.attention.attention_impl` all stopped importing, and
`lumenrl/tests/test_lumen_attn.py` failed at collection.

The predicate that finds it resolves every `from aiter… import NAME` the three repos
write, against the old branch as a baseline:

| aiter | modules missing | names missing |
|---|---|---|
| `4ebe6d69c` (before the rebase) | 3 | 16 |
| `c395c6288` (rebased, first try) | 3 | 12, **5 of them new** |
| `d6303e316` (shipped) | 3 | 7 |

The seven are a subset of the original sixteen, so there is no regression left, and nine
names the old branch lacked — `topk_select` among them — now resolve. The three missing
modules are the same on both and sit behind `try`/`except` probes.

Restoring was additive rather than a revert of upstream's rewrite: upstream's public
wrappers call the private kernels it renamed these to, so replacing the files wholesale
would break those instead.

⚠️ `import lumen.ops.quantize` on its own still raises a circular-import `ImportError`.
That is **pre-existing** — identical on `260917` — and importing `lumen.quantize` first
works on both. It is not from this round.

## Runs

26 runs, all exit 0, all four error counts 0, every weight-sync bucket `skipped=0`,
`--check` 26/26 PASS against the table in `08-release.md` §8.5.1.

| # | n | `k3_kl` per run | mean | worst dev from mean |
|---|---|---|---|---|
| 1 | 3 | 0.0011179 / 0.00107158 / 0.00111735 | 0.00110 | 2.6% |
| 2 | 3 | 0.00471458 / 0.00485167 / 0.00485262 | 0.00481 | 2.0% |
| 3 | 3 | 0.00480643 / 0.00373042 / 0.00375428 | 0.00410 | 17.2% |
| 4 | 5 | 0.00459888 / 0.00338652 / 0.00426389 / 0.00499343 / 0.00270809 | 0.00399 | 32.1% |
| 5 | 3 | 0.000841038 / 0.00099301 / 0.0010288 | 0.000954 | 11.8% |
| 6 | 3 | 0.00136204 / 0.00157178 / 0.00138287 | 0.00144 | 9.2% |
| 7 | 3 | 0.00166141 / 0.00160454 / 0.00156759 | 0.00161 | 3.2% |
| 9 | 3 | 0.00157512 / 0.00145847 / 0.00154193 | 0.00153 | 4.7% |

## The table was re-measured; the values did not need it, the spread did

Every reference moved by between **−8.9% and +10.5%**, so the new ATOM, the rebased
aiter and this round's changes did not measurably alter train/rollout alignment. What
three runs per example bought is the spread, which most rows had never had: from 2.0%
(example 2) to 32.1% (example 4). Tolerances are unchanged.

⚠️ **Example 4 is why single observations are not enough.** Its five runs span
0.00271–0.00499, a factor of 1.8. Three consecutive runs mid-round read 0.00339 /
0.00426 / 0.00460 against a 0.00287 reference, which looked like
`ATOM_FORCE_ATTN_TRITON` shifting it systematically — and that is what an earlier draft
of this record and of `versions.env` said. The fifth run came back at 0.00271, inside
the old band, and the no-variable run on the same image was 0.00367, inside the
with-variable range. **It is spread, not a shift, and the variable is not implicated.**

## What the decode override does and does not move

Same image, same rebuilt kernels, only the variable differs.

| example 4, `chi2_token` per step | step 1 | step 2 | step 3 |
|---|---|---|---|
| without | 0.147 | 0.0138 | **5761.69** |
| with, run a | 5.39 | 0.0411 | 0.113 |
| with, run b | 0.0152 | 0.135 | 0.112 |
| with, run c | 4.31 | 0.00972 | 0.192 |

`abs_diff` over the same steps is 0.0311–0.0473 without and 0.0259–0.0390 with — it does
not move, which is the defect's signature and why `chi2_token` is the metric to judge it
by.

**It does not lower `k3_kl` here, and should not be expected to.** Example 5 went
−12%, example 9 +15%, example 4 within its own spread; all noise at this sample size. At
a 4k response the defect fires about once in nine steps, and the reference is read at
step 1 while example 4's event was at step 3 — this table's metrics structurally cannot
see it. The earlier report of the variable lowering KL was measured on the 30B MoE at a
20K response, where `chi2_token` runs in the thousands and the defect fires often enough
to dominate.

⚠️ With the variable set, `chi2_token` is not uniformly small either: two of six step-1
values are ~4–5 against ~0.01–0.15 for the clean ones. It dodges the catastrophic class,
it does not make the rollout exact.

## Example 9's sleep path

48 releases and 48 resumes, balanced (8 replicas × 2 memory tags × 3 steps), zero
negative KV-pool derivations, zero `Memory access fault`, zero `grad_norm=nan` — the
three symptoms PR #2267 fixes, none of them present.

## Limits of this record

- Three runs per example is enough to show a spread, not to bound one. Example 4 needed
  five before its low end appeared.
- The `chi2_token` comparison for `ATOM_FORCE_ATTN_TRITON` is **one run without the
  variable against three with it**, on one example. The direction is not in doubt —
  5761.69 against a maximum of 5.39 — but the magnitude rests on a single event.
- The five kernels not rebuilt were judged unaffected by diffing their `csrc` sources,
  not by running anything that isolates them.
- The seven aiter names still unresolved are unresolved on the old branch too; nothing
  here shows they are unused, only that this round did not change them.
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
