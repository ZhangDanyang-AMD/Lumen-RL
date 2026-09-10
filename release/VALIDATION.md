# LumenRL release validation

Measurement record behind the reference table in
[`examples/docs/08-release.md`](../examples/docs/08-release.md) §8.5.1.

| | |
|---|---|
| Image | `zhangdanyangamd/lumen-rl:dapo-gfx950-rocm7.2.3-260910`, digest `sha256:cc18a3f5ce16…` |
| Lumen-RL | `8ca6bdd` on `dev/dapo_release`, bind-mounted from a checkout — **not** taken from the image, see §8.1.1 |
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

The fifth failure was the ATOM re-pin itself and is the reason
`sleep_keeps_memory_resident` is now pinned: see `versions.env` and §8.1.2. Example 9
reached step 0, synced weights with `skipped=0`, and then aborted all eight replicas in
`resume_memory` with a negative KV pool, because ATOM now re-derives the block count on
every wake and the colocated trainer is 52 GB of the budget it subtracts.

⚠️ **Releasing the actors' allocator cache before the wake does not fix that** — it was
tried and moves `non_torch` by 0.6 GB, because after an optimizer step that memory is
live state rather than cache. Only keeping the pool resident works.

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
- **The pin sidesteps the KV budget problem rather than fixing it.** The `non_torch`
  figure that makes the budget negative is itself drift in ROCm/ATOM `main` between the
  PR's merge base and `8938787d`: a separate A/B on `main` plus only the `n>1` fan-out
  fix reproduces `non_torch=52.88GB` and `available_for_kv=-29.93GB`, i.e. the same
  numbers with the whole MoE weight-sync path out of the picture. So an ATOM that
  releases will keep needing the pin until that is found, and finding it is upstream
  work this record does not cover. Reproduction worktrees are kept outside git.
