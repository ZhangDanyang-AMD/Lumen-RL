# LumenRL release validation — 2026-09-04

Raw record behind the `dapo-gfx950-rocm7.2.3-260904` image and the reference
table in [`examples/docs/08-release.md`](../examples/docs/08-release.md) §8.5.1.
Supersedes nothing: [`VALIDATION-20260903.md`](VALIDATION-20260903.md) remains
the record for the `260902` image, and §8.5.1's references are the mean over
both files' runs.

## Why a new image

| | |
|---|---|
| Delta vs `260902` | Lumen-RL `6957ee9c1c79` → `1cd932aa5519`; the other three repos and every python pin are unchanged |
| Code change | `lumenrl/engine/inference/atom_ray_server.py`: pin `cudagraph_mode=FULL` for no-eager ATOM rollouts |
| Also picked up | `release/run_example.sh` now ships **inside** the image (the `260902` image's `release/` predates the launcher) |
| Also fixed | `260902` carried a baked `DATA_ROOT=/home/xysheng/rl_data`, leaked in from its own `docker commit`; `260904` ships it empty |

The code change is **inert on this stack** and was not the reason for the
rebuild — see "What the code change is for" below.

## Environment

| | |
|---|---|
| Node | `crsuse2-m2m-v2-035`, 8x MI355X (gfx950), whole-node allocation (job 4134) |
| Image | `zhangdanyangamd/lumen-rl:dapo-gfx950-rocm7.2.3-260904` (image id `da82db5b0e19`, 47.3 GB on disk) |
| Container | `lumenrl-release`, restarted by the launcher before every run |
| Launcher | `release/run_example.sh <N> --check --log <path>` |
| `DATA_ROOT` | `/home/xysheng/rl_data` |
| Seed | `10086` (hardcoded in `run_dapo.sh`) |
| Date | 2026-09-04, 12:17–14:15 UTC |
| Stack in image | `vllm 0.23.0`, `flydsl 0.3.2`, `transformers 5.12.0`, `aiter` resolving to `/opt/lumenrl/aiter/aiter/__init__.py` |

GPU state before the first run: all 8 cards at the ~298 MB idle baseline, no
containers on the node.

## Build pipeline

`docker build` has no GPUs, so aiter's JIT kernels are baked in afterwards. All
three stages ran on the node above.

| stage | command | duration | result |
|---|---|---|---|
| build | `TAG=…-260904 bash release/build_image.sh` | 15 min | most layers hit the `260902` build cache; only the Lumen-RL clone and the layers after it rebuilt |
| bake | `TAG=…-260904 bash release/precompile_kernels.sh` | 8 min | 5 of 16 kernel objects; the long pole is `module_gemm_a8w8_blockscale_cktile` at 402 s |
| warm | `TAG=…-260904-kernels EX=4 NAME=warm bash release/validate_image.sh`, then `docker commit` | 21 min | 1237 s for a 1-step example 4, `RESULT: PASS`, 16/16 objects afterwards |

The two-stage bake is not optional: the synthetic warmup reaches 5 objects, and
example 4 is the only path wide enough to pull in the remaining 11 (it is ATOM
FP8 rollout plus FSDP2 FP8 training). The 1237 s here matches the 1256 s the
`precompile_kernels.sh` header records for a 5-object image, against ~450 s once
all 16 are baked.

Post-commit checks on the final image:

```
baked objects: 16
launcher in image: yes
DATA_ROOT: [<empty>]
cudagraph fix present: True
```

## Results

`launcher` is what `run_example.sh` reports for the whole invocation: container
restart, GPU idle probe, the run itself and the metric check. `span` is the
first-to-last timestamp inside the trainer log, which is reproducible from the
artifact alone. `errors` is the combined count of `Traceback`, `OutOfMemory`,
`CUDA error` and `HSA_STATUS`.

| run | ex | config | steps | resp | exit | span | launcher | `rollout_corr/k3_kl` | `entropy` | `rollout_corr/kl` | `ppl_ratio` | `response_length/mean` | `perf/time_per_step` | errors |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1a | 1 | `dapo_qwen3_8b_ray_vllm_smoke.yaml` | 3 | 512 | 0 | 173 s | 196 s | 0.00107306 | 0.581881 | 0.0011476 | 1.00147 | 392.75 | 40.40 s | 0 |
| 2a | 2 | `dapo_qwen3_8b_ray_vllm_fp8_smoke.yaml` | 3 | 512 | 0 | 130 s | 165 s | 0.00527892 | 0.802578 | 0.00536311 | 1.00811 | 414.84 | 29.58 s | 0 |
| 3a | 3 | `dapo_qwen3_8b_ray_vllm_fp8_smoke.yaml`, `TRAIN_FP8=1` | 3 | 512 | 0 | 139 s | 171 s | 0.00374353 | 0.821747 | 0.00351206 | 1.01071 | 413.44 | 30.54 s | 0 |
| 4a | 4 | `dapo_qwen3_8b_ray_atom_fp8_4k_smoke.yaml` | 3 | 4096 | 0 | 485 s | 522 s | 0.00291013 | 0.693103 | 0.00276403 | 1.00344 | 729.25 | 51.02 s | 0 |
| 4b | 4 | same | 3 | 4096 | 0 | 493 s | 527 s | 0.00303403 | 0.525075 | 0.00243383 | 1.00273 | 583.05 | 44.00 s | 0 |
| 4c | 4 | same | 3 | 4096 | 0 | 561 s | 597 s | 0.00271429 | 0.589821 | 0.00294271 | 1.00541 | 738.55 | 81.36 s | 0 |
| 5a | 5 | `dapo_qwen3_8b_ray_atom_bf16_4k_smoke.yaml` | 1 | 4096 | 0 | 404 s | 437 s | 0.000896993 | 0.686584 | 0.00084303 | 1.00182 | 680.34 | 46.72 s | 0 |
| 5b | 5 | same | 1 | 4096 | 0 | 413 s | 446 s | 0.0011023 | 0.818122 | 0.000681218 | 1.00099 | 712.55 | 69.46 s | 0 |
| 5c | 5 | same | 1 | 4096 | 0 | 389 s | 422 s | 0.000894893 | 0.597785 | 0.00109489 | 1.00107 | 835.89 | 46.71 s | 0 |
| 6a | 6 | `dapo_qwen3moe_a3b_ray_vllm_verlref_4k_smoke.yaml` | 3 | 4096 | 0 | 524 s | 557 s | 0.00139201 | 0.568863 | 0.00137177 | 1.00198 | 787.00 | 117.75 s | 0 |
| 7a | 7 | `dapo_qwen3moe_a3b_ray_megatron_verlref_4k_smoke.yaml` | 3 | 4096 | 0 | 505 s | 537 s | 0.00139328 | 0.676593 | 0.00120232 | 1.00133 | 733.56 | 113.14 s | 0 |

**Conclusion: 11/11 runs exited 0 with zero error lines, and `--check` was
11/11 PASS — against the `260902` reference values, before any were touched.**
That is the regression test that matters: the new image reproduces the old one's
numbers without the table being adjusted to fit.

Examples 4 and 5 were sampled three times each because they carry §8.5.1's
references and example 5's published tolerance rested on only two samples.
Examples 6 and 7 again ran back to back with no isolation beyond the container
restart.

## Re-derived references and tolerances

Runs from both images are pooled: they share identical ATOM / Lumen / aiter /
vLLM pins, and 11/11 PASS against the old references is the evidence that the
Lumen-RL delta does not move the numbers on this stack. Runs from the
ROCm/ATOM#2028 experiment are deliberately excluded — different ATOM.

| ex | runs | `k3_kl` reference | observed spread | `entropy` reference | observed spread | `kl` reference |
|---|---|---|---|---|---|---|
| 1 | 7 (6+1) | 0.00108 | ±8% | 0.605 | ±7% | 0.000973 |
| 2 | 2 (1+1) | 0.00498 | ±6% | 0.796 | ±1% | 0.00502 |
| 3 | 2 (1+1) | 0.00392 | ±5% | 0.817 | ±1% | 0.00382 |
| 4 | 6 (3+3) | 0.00287 | ±17% | 0.600 | ±17% | 0.00270 |
| 5 | 5 (2+3) | 0.000974 | ±13% | 0.677 | **±21%** | 0.000852 |
| 6 | 3 (2+1) | 0.00149 | ±12% | 0.766 | **±35%** | 0.00164 |
| 7 | 3 (2+1) | 0.00151 | ±8% | 0.662 | ±17% | 0.00151 |

Every reference is the mean over that example's pooled runs. No reference moved
by more than 6%, and examples 1, 4 and 5 moved by less than 2% — the two images
agree.

**One tolerance changed: example 5's `entropy`, ±50% → ±65%.** With two samples
its observed spread looked like ±4%; with five it is ±21%, and three times that
is 63%. The published ±50% no longer covered its own measurements. Every other
tolerance is unchanged.

Two corrections to earlier claims, both from having more samples:

- `VALIDATION-20260903.md` derived example 5's tolerance from a ±4% spread. That
  was an artifact of two samples. Do not derive a tolerance from two runs of a
  4096-token config.
- A note written during the ROCm/ATOM#2028 work said example 5's `k3_kl` jitters
  by 2.3x. That was measured on the #2028 stack (0.00186 and 0.000812). On the
  released stack, five samples span 0.000895–0.00110, i.e. ±13%. The 2.3x is not
  a property of example 5 in general.

All 28 runs across both files pass against the table above, the worst using 57%
of its tolerance (example 6's entropy).

## What the code change is for

The Lumen-RL delta in this image, `cudagraph_mode=FULL`, does nothing here: the
pinned ATOM overwrites `cudagraph_mode` unconditionally and never reads the
field. Confirmed empirically — all 11 runs pass, examples 4 and 5 included.

It exists for the ATOM source move. The pinned ATOM is a personal fork whose
changes are being upstreamed as ROCm/ATOM#2028; that PR enables a per-piece
cudagraph wrapper the fork leaves disabled, and a level-3 rollout then defaults
to per-piece capture. Each piece asserts its inputs keep their capture-time
addresses, while the attention between two pieces runs eager and reallocates its
output every call, so the first replay aborts all eight rollout workers with
`Input addresses for cudagraphs are different during replay` — before a single
weight update. Whole-forward capture has no such boundary.

Measured on #2028 (not part of the table above, and not in this image):

| configuration | example 4 | example 5 |
|---|---|---|
| no-eager, `cudagraph_mode` left to ATOM's default | 8 workers abort | same abort |
| no-eager, `cudagraph_mode=FULL` | 526 / 519 s, `--check` PASS | 433 / 392 s |
| eager (`enforce_eager=true`, `level=0`) | 619 s | 402 s |

No-eager runs each step 1.4–2.2x faster than eager but pays 70–85 s more setup
for torch.compile and capture, so a 3-step example 4 nets 15–16% and a 1-step
example 5 roughly breaks even. The source pin moves to upstream once #2028
merges.

## What was not measured

- **Cold `docker pull` time**, and the image was never pushed: publishing
  `260904` needs the registry owner's credentials.
- **Long runs.** Only the smoke configs ran. `--longrun` is still verified by
  `--dry-run` only.
- **Logging to a real wandb project.**
- **gfx942.** This image is gfx950-only by construction.
- **A second sample for examples 6 and 7 on this image** — they have three
  pooled runs each, but only one is from `260904`.
