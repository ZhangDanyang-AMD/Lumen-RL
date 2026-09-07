# LumenRL release validation

Measurement record behind the reference table in
[`examples/docs/08-release.md`](../examples/docs/08-release.md) §8.5.1.

| | |
|---|---|
| Image | `zhangdanyangamd/lumen-rl:dapo-gfx950-rocm7.2.3-260907`, digest `sha256:dc2209744fa0…` |
| Hardware | one node, 8x MI355X (gfx950), whole-node allocation |
| Command | `bash release/run_example.sh <N> --check` |
| Seed | `10086`, fixed inside `run_dapo.sh` |
| Metrics read at | step 1 (`step=1`) |
| Date | 2026-09-07 |
| Stack in image | `vllm 0.23.0`, `flydsl 0.3.2`, `transformers 5.12.0`, `aiter` resolving to `/opt/lumenrl/aiter/` |

All 8 cards were at the ~298 MB idle baseline before each run, and the launcher
restarts the container between runs.

## Image build

`docker build` has no GPUs, so aiter's JIT kernels are baked in afterwards.

| stage | duration | result |
|---|---|---|
| `build_image.sh` | 10 min | four source trees cloned at the SHAs in `versions.env` |
| `precompile_kernels.sh` | 8 min | 5 of 16 aiter kernel objects |
| warm example 4, then `docker commit` | 21 min | 1274 s for a 1-step example 4, PASS, 16/16 objects afterwards |

The second bake stage is not optional: the synthetic warmup reaches 5 objects,
and example 4 is the only path wide enough to pull in the remaining 11, being
ATOM FP8 rollout plus FSDP2 FP8 training. A 5-object image pays about 1250 s for
its first example 4; a fully baked one about 500 s.

Checks on the published image: the three upstream trees at the SHAs in
`versions.env`, Lumen-RL at the tip of its branch, 16 baked kernel objects, and
no `DATA_ROOT` baked into the environment. Run the launcher from the `release/`
directory of a checkout, not from inside the image.

## Runs

`span` is the first-to-last timestamp in the trainer log. `errors` is the
combined count of `Traceback`, `OutOfMemory`, `CUDA error` and `HSA_STATUS`.
Examples 4 and 5 are the ATOM ones and were sampled repeatedly to characterise
their spread; the widest path, example 4, was sampled six times.

| ex | span | exit | errors | `rollout_corr/k3_kl` | `entropy` | `rollout_corr/kl` | `ppl_ratio` | `response_length/mean` |
|---|---|---|---|---|---|---|---|---|
| 1 | 142 s | 0 | 0 | 0.00114166 | 0.636215 | 0.000858606 | 1.00214 | 395.66 |
| 2 | 129 s | 0 | 0 | 0.00519366 | 0.790776 | 0.00555642 | 1.00804 | 416.25 |
| 3 | 140 s | 0 | 0 | 0.00413398 | 0.807625 | 0.00405282 | 1.01123 | 414.35 |
| 4 | 538 s | 0 | 0 | 0.00293989 | 0.809143 | 0.00341165 | 1.00384 | 640.05 |
| 4 | 558 s | 0 | 0 | 0.00510902 | 0.718585 | 0.00496713 | 1.00899 | 644.11 |
| 4 | 495 s | 0 | 0 | 0.00362788 | 0.877335 | 0.00303843 | 1.00093 | 782.14 |
| 4 | 512 s | 0 | 0 | 0.00304025 | 0.702852 | 0.00325054 | 1.00764 | 641.56 |
| 4 | 561 s | 0 | 0 | 0.00436911 | 0.709418 | 0.00454808 | 1.00723 | 759.42 |
| 4 | 496 s | 0 | 0 | 0.00334394 | 0.521407 | 0.00419052 | 1.00453 | 886.03 |
| 5 | 401 s | 0 | 0 | 0.0007968 | 0.522858 | 0.000275452 | 1.0003 | 657.56 |
| 5 | 385 s | 0 | 0 | 0.000824371 | 0.45037 | 0.000761109 | 1.00126 | 660.64 |
| 5 | 384 s | 0 | 0 | 0.000958777 | 0.743243 | 0.00124844 | 1.00228 | 914.77 |
| 6 | 530 s | 0 | 0 | 0.00134321 | 0.629325 | 0.00128033 | 1.00117 | 696.35 |
| 7 | 503 s | 0 | 0 | 0.0013848 | 0.627709 | 0.00139737 | 1.00009 | 676.55 |

**14/14 exited 0 with zero error lines, and `--check` is 14/14 PASS.**

## References and tolerances

| ex | runs | `k3_kl` | spread | `entropy` | spread | `kl` |
|---|---|---|---|---|---|---|
| 1 | 1 | 0.00114 | — | 0.636 | — | 0.000859 |
| 2 | 1 | 0.00519 | — | 0.791 | — | 0.00556 |
| 3 | 1 | 0.00413 | — | 0.808 | — | 0.00405 |
| 4 | 6 | 0.00374 | ±37% | 0.723 | ±28% | 0.00390 |
| 5 | 3 | 0.000860 | ±11% | 0.572 | ±30% | 0.000762 |
| 6 | 1 | 0.00134 | — | 0.629 | — | 0.00128 |
| 7 | 1 | 0.00138 | — | 0.628 | — | 0.00140 |

Each reference is the mean over that example's runs on this image. The tolerance
is a per-group floor: `k3_kl` 30% at 512 tokens and 50% at 4096; `entropy` 25%
at 512, 50% at 4096, 60% for the two MoE examples. Example 4, the most sampled,
uses 73% of its band, so the floors carry the observed spread with margin.

## Limits of this record

- **Examples 1, 2, 3, 6 and 7 have one sample each**, so their references are a
  single measurement and the spread column is empty. The tolerances are the group
  floors rather than something these runs established.
- **Example 5 has three samples**, enough to see a spread but not to bound it.
- **Long runs were not carried to completion.** Only the smoke configs ran.
  `--longrun` is exercised by `--dry-run`: it selects the right yaml, sets
  `STEPS=1000` and handles a missing wandb key.
- **Logging to a real wandb project was not exercised**, only the no-key path.
- **gfx942 was not tried.** This image is built for gfx950.
- **Cold `docker pull` duration is network-bound** and not quoted; §8.3.2 gives
  the download size instead.
