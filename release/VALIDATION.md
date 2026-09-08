# LumenRL release validation

Measurement record behind the reference table in
[`examples/docs/08-release.md`](../examples/docs/08-release.md) §8.5.1.

| | |
|---|---|
| Image | `zhangdanyangamd/lumen-rl:dapo-gfx950-rocm7.2.3-260908`, digest `sha256:41eeaf8d5db5…` |
| Lumen-RL | `f4439f3` on `dev/dapo_release`, bind-mounted from a checkout — **not** taken from the image, see §8.1.1 |
| Hardware | one node, 8x MI355X (gfx950), whole-node allocation |
| Command | `bash release/run_example.sh <N> --check` |
| Seed | `10086`, fixed inside `run_dapo.sh` |
| Metrics read at | step 1 (`step=1`) |
| Date | 2026-09-08 |
| Stack in image | `vllm 0.23.0`, `flydsl 0.3.2`, `transformers 5.12.0`, `aiter` resolving to `/opt/lumenrl/aiter/` |

**A result is identified by the image digest and the Lumen-RL commit together.** The
launcher mounts the checkout it lives in over `/opt/lumenrl/Lumen-RL`, so the image no
longer pins the framework code and quoting the digest alone does not describe a run.
Every run prints both, and the launcher recreates its container when either changes.

All 8 cards were at the ~298 MB idle baseline before each run, and the launcher
restarts the container between runs.

## Image build

`docker build` has no GPUs, so aiter's JIT kernels are baked in afterwards.

| stage | duration | result |
|---|---|---|
| `build_image.sh` | 11 min | four source trees cloned at the SHAs in `versions.env` |
| `precompile_kernels.sh` | 8 min | 5 of 16 aiter kernel objects |
| warm example 4, then `docker commit` | 21 min | 1240 s for a 1-step example 4, PASS, 16/16 objects afterwards |

The second bake stage is not optional: the synthetic warmup reaches 5 objects, and
example 4 is the only path wide enough to pull in the remaining 11, being ATOM FP8
rollout plus FSDP2 FP8 training.

`build_image.sh` resolves `LUMENRL_BRANCH` to the SHA it points at and passes that to
the Dockerfile. The clone layer is cached on the build args, so passing a branch name
meant a second build on the same machine silently re-used the first build's clone and
shipped whatever the tip was back then.

Checks on the published image: the three upstream trees at the SHAs in `versions.env`,
16 baked kernel objects, both MoE ATOM configs present, and no `DATA_ROOT` baked into
the environment. Run the launcher from the `release/` directory of a checkout, not from
inside the image.

## Runs

`span` is the first-to-last timestamp in the trainer log. `errors` is the combined
count of `Traceback`, `OutOfMemory`, `CUDA error` and `HSA_STATUS`. `skipped` is the
number of weight-sync buckets that did not account for every tensor in them; examples
on the vLLM path emit no such line and count as 0. Examples 4 and 9 are the ones
sampled repeatedly.

| ex | span | exit | errors | skipped | `rollout_corr/k3_kl` | `entropy` | `rollout_corr/kl` | `ppl_ratio` | `response_length/mean` |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 170 s | 0 | 0 | 0 | 0.00105635 | 0.581662 | 0.00105897 | 1.00013 | 411.20 |
| 2 | 129 s | 0 | 0 | 0 | 0.00498006 | 0.784457 | 0.00537692 | 1.00808 | 414.66 |
| 3 | 140 s | 0 | 0 | 0 | 0.00403682 | 0.831763 | 0.00419250 | 1.01128 | 418.32 |
| 4 | 564 s | 0 | 0 | 0 | 0.00243338 | 0.493383 | 0.00309054 | 1.00306 | 734.66 |
| 4 | 559 s | 0 | 0 | 0 | 0.00241102 | 0.521360 | 0.00210570 | 1.00272 | 780.14 |
| 4 | 629 s | 0 | 0 | 0 | 0.00376739 | 0.605261 | 0.00342991 | 1.00196 | 687.64 |
| 5 | 408 s | 0 | 0 | 0 | 0.000929704 | 0.567738 | 0.000880022 | 1.00119 | 719.95 |
| 6 | 1202 s | 0 | 0 | 0 | 0.00157877 | 0.679286 | 0.00153856 | 1.00147 | 807.45 |
| 7 | 501 s | 0 | 0 | 0 | 0.00157938 | 0.660039 | 0.00188473 | 1.00270 | 790.08 |
| 9 | 684 s | 0 | 0 | 0 | 0.00142840 | 0.671092 | 0.00119565 | 1.00099 | 757.06 |
| 9 | 562 s | 0 | 0 | 0 | 0.00129423 | 0.608896 | 0.00128308 | 1.00155 | 734.99 |
| 9 | 557 s | 0 | 0 | 0 | 0.00142258 | 0.797094 | 0.00164708 | 1.00175 | 744.15 |

**12/12 exited 0 with zero error lines and zero skipped buckets, and `--check` is
12/12 PASS.**

## References and tolerances

| ex | runs | `k3_kl` | spread | `entropy` | spread | `kl` |
|---|---|---|---|---|---|---|
| 1 | 1 | 0.00106 | — | 0.582 | — | 0.00106 |
| 2 | 1 | 0.00498 | — | 0.784 | — | 0.00538 |
| 3 | 1 | 0.00404 | — | 0.832 | — | 0.00419 |
| 4 | 3 | 0.00287 | ±31% | 0.540 | ±12% | 0.00288 |
| 5 | 1 | 0.000930 | — | 0.568 | — | 0.000880 |
| 6 | 1 | 0.00158 | — | 0.679 | — | 0.00154 |
| 7 | 1 | 0.00158 | — | 0.660 | — | 0.00188 |
| 9 | 3 | 0.00138 | ±6% | 0.692 | ±15% | 0.00138 |

Each reference is the mean over that example's runs on this image. The tolerance is a
per-group floor: `k3_kl` 30% at 512 tokens and 50% at 4096; `entropy` 25% at 512, 50%
at 4096, 60% for the three MoE examples. Example 4 has the widest spread and uses 62%
of its band, so the floors carry the observed spread with margin.

**Example 4 is the argument against tightening the floors to a single run.** Its three
runs came in at 0.00241, 0.00243 and 0.00377. Either of the first two, taken alone as
the reference, would put the third outside a ±50% band — a reproduction that is in fact
correct would be reported as a regression.

## Example 9 against example 6

The two differ only in the rollout engine, so subtracting them is meaningful.

| | example 6 (vLLM) | example 9 (ATOM) |
|---|---|---|
| `k3_kl` | 0.00158 | 0.00138 (−12%) |
| per step | 117.3 s | 64.3 s (1.8x faster) |
| setup | 884 s | 514 s cold, 388–390 s warm |

`k3_kl` moves by less than example 9's own run-to-run spread doubled, so **replacing
the rollout engine does not measurably change train/rollout alignment**. The per-step
figure is the real difference and the reason a long run favours ATOM; a 3-step smoke
does not, because setup dominates it.

Example 6's 884 s setup is not a fair comparison point: that single run paid the first
read of the 57 GB checkpoint, which example 7 then got from page cache and finished
setup in 191 s. Example 9's own cold/warm gap (514 s vs 389 s) is aiter's JIT cache
being rebuilt after the container was recreated, the same effect on a smaller scale.

## Limits of this record

- **Examples 1, 2, 3, 5, 6 and 7 have one sample each**, so their references are a
  single measurement and the spread column is empty. The tolerances are the group
  floors rather than something these runs established. Example 5 had three samples on
  the previous image and does not here.
- **Example 6 has one sample and an unrepresentative span**, for the page-cache reason
  above. Its metrics are unaffected; only the timing is.
- **Long runs were not carried to completion.** Only the smoke configs ran. `--longrun`
  is exercised by `--dry-run`, which selects the right yaml, sets `STEPS=1000` and
  handles a missing wandb key. This includes example 9's new
  `dapo_qwen3moe_a3b_ray_atom_bf16_longrun.yaml`.
- **Logging to a real wandb project was not exercised**, only the no-key path.
- **gfx942 was not tried.** This image is built for gfx950.
- **The from-scratch `docker pull` path was not re-timed.** The published digest was
  confirmed against the local image, but no run started from a cold pull.
- **Cold `docker pull` duration is network-bound** and not quoted; §8.3.2 gives the
  download size instead.
- **Two changes landed after these measurements**, both inert against the pinned
  ATOM: `ATOM_SRC` in the launcher, which is empty by default, and a
  `true_vocab_size` engine kwarg alongside the environment variable the pinned
  ATOM reads. ATOM filters engine kwargs against its `Config` dataclass fields
  and the pinned build has no such field, so it drops the kwarg — verified
  directly. The references stand.

## A/B against a rebased ATOM branch

The three ATOM examples were re-run against a cleaned-up ATOM branch mounted
over the image's copy (`ATOM_SRC`), to check that reorganising that work changed
no numbers. One run each, against the references above.

| ex | reference (runs) | rebased ATOM | delta |
|---|---|---|---|
| 4 | 0.00287 (3, ±31%) | 0.00229 | −20% |
| 5 | 0.000930 (1) | 0.000948 | +2% |
| 9 | 0.00138 (3, ±6%) | 0.00160 | +16% |

All three PASS, all `skipped=0`, and per-step time is unchanged (63.5 s against
64.3 s on example 9). Examples 4 and 5 sit inside the reference spread.

**Example 9 lands 12% above the highest of its three reference runs, and that is
not explained.** No code path in the branch should reach it: the routed-expert
routing only triggers on per-expert names, and this stack renames the fused ones
upstream of ATOM. The most likely reading is that three samples underestimate
the spread — the one reference run that was also a cold start after a container
recreate, like this one, was itself the highest of the three. Recorded rather
than resolved; it did not warrant more machine time at the tolerances in use.
