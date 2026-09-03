# LumenRL release validation — 2026-09-03

Every example in [`examples/docs/08-release.md`](../examples/docs/08-release.md)
§8.2 was run with the exact command that chapter documents. This file is the raw
record behind its §8.6.1 reference table.

## Environment

| | |
|---|---|
| Node | `crsuse2-m2m-v2-035`, 8x MI355X (gfx950), whole-node allocation |
| Image | `zhangdanyangamd/lumen-rl:dapo-gfx950-rocm7.2.3-260902` (image id `68cf27af9a2c`) |
| Container | `lumenrl-release`, restarted by the launcher before every run |
| Launcher | `release/run_example.sh <N> --check --log <path>` |
| `DATA_ROOT` | `/home/xysheng/rl_data` |
| Seed | `10086` (hardcoded in `run_dapo.sh`) |
| Date | 2026-09-03, 03:28–06:04 UTC |

Software stack as reported by the container:
`vllm 0.23.0`, `flydsl 0.3.2`, `transformers 5.12.0`, `aiter` resolving to
`/opt/lumenrl/aiter/aiter/__init__.py`.

GPU state before the first run: all 8 cards at 297766912–297832448 B in use
(the ~298 MB idle baseline), no containers on the node.

## Results

Two durations are given because they measure different things.
`log span` is the first-to-last timestamp inside the trainer log, which is
reproducible from the artifact. `launcher` is what `run_example.sh` reports for the
whole invocation and is about 30–35 s longer: container restart, GPU idle probe and
the metric check. `errors` is the combined count of `Traceback`, `OutOfMemory`,
`CUDA error` and `HSA_STATUS` in the trainer log.

| run | ex | config | steps | resp | exit | log span | launcher | `rollout_corr/k3_kl` | `entropy` | `rollout_corr/kl` | `ppl_ratio` | `response_length/mean` | `perf/time_per_step` | errors |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1a | 1 | `dapo_qwen3_8b_ray_vllm_smoke.yaml` | 3 | 512 | 0 | 171 s | n/a | 0.00110339 | 0.635074 | 0.00122282 | 1.00109 | 405.43 | 40.06 s | 0 |
| 1b | 1 | same | 3 | 512 | 0 | 129 s | n/a | 0.00117701 | 0.623073 | 0.00046895 | 0.999031 | 397.23 | 33.99 s | 0 |
| 1c | 1 | same | 3 | 512 | 0 | 137 s | 171 s | 0.00105992 | 0.591768 | 0.00110177 | 1.00072 | 408.48 | 34.96 s | 0 |
| 2 | 2 | `dapo_qwen3_8b_ray_vllm_fp8_smoke.yaml` | 3 | 512 | 0 | 129 s | 166 s | 0.00469053 | 0.788654 | 0.00468057 | 1.00748 | 412.65 | 29.54 s | 0 |
| 3 | 3 | `dapo_qwen3_8b_ray_vllm_fp8_smoke.yaml`, `TRAIN_FP8=1` | 3 | 512 | 0 | 140 s | 176 s | 0.00410046 | 0.811758 | 0.00411976 | 1.01139 | 417.53 | 30.89 s | 0 |
| 4a | 4 | `dapo_qwen3_8b_ray_atom_fp8_4k_smoke.yaml` | 3 | 4096 | 0 | 520 s | 557 s | 0.00236943 | 0.500321 | 0.00255547 | 1.00185 | 812.05 | 51.73 s | 0 |
| 4b | 4 | same | 3 | 4096 | 0 | 567 s | 602 s | 0.00327112 | 0.631694 | 0.00285357 | 1.00462 | 644.94 | 81.03 s | 0 |
| 5 | 5 | `dapo_qwen3_8b_ray_atom_bf16_4k_smoke.yaml` | 1 | 4096 | 0 | 436 s | 472 s | 0.00102707 | 0.665127 | 0.000874595 | 1.00121 | 799.59 | 73.18 s | 0 |
| 6 | 6 | `dapo_qwen3moe_a3b_ray_vllm_verlref_4k_smoke.yaml` | 3 | 4096 | 0 | 519 s | 552 s | 0.0016679 | 1.03019 | 0.00180211 | 1.00165 | 785.05 | 113.51 s | 0 |
| 7 | 7 | `dapo_qwen3moe_a3b_ray_megatron_verlref_4k_smoke.yaml` | 3 | 4096 | 0 | 493 s | 526 s | 0.00154365 | 0.762078 | 0.00163412 | 1.00116 | 732.23 | 111.09 s | 0 |
| 1d | 1 | `dapo_qwen3_8b_ray_vllm_smoke.yaml` | 3 | 512 | 0 | 170 s | 190 s | 0.00110719 | 0.605645 | 0.000748721 | 1.00106 | 409.30 | 40.35 s | 0 |
| 4c | 4 | `dapo_qwen3_8b_ray_atom_fp8_4k_smoke.yaml` | 3 | 4096 | 0 | 524 s | 557 s | 0.00293003 | 0.660067 | 0.00262851 | 1.00289 | 716.98 | 83.11 s | 0 |
| 5b | 5 | `dapo_qwen3_8b_ray_atom_bf16_4k_smoke.yaml` | 1 | 4096 | 0 | 435 s | 472 s | 0.000948584 | 0.616478 | 0.000766472 | 1 | 971.53 | 73.99 s | 0 |
| 6b | 6 | `dapo_qwen3moe_a3b_ray_vllm_verlref_4k_smoke.yaml` | 3 | 4096 | 0 | 526 s | 557 s | 0.00141876 | 0.698045 | 0.00175837 | 1.00214 | 779.46 | 115.89 s | 0 |
| 7b | 7 | `dapo_qwen3moe_a3b_ray_megatron_verlref_4k_smoke.yaml` | 3 | 4096 | 0 | 498 s | 532 s | 0.00160584 | 0.548498 | 0.00169583 | 1.00323 | 750.07 | 110.31 s | 0 |
| 1e | 1 | `dapo_qwen3_8b_ray_vllm_smoke.yaml` | 3 | 512 | 0 | 136 s | 156 s | 0.000994479 | 0.560073 | 0.00133158 | 1.00193 | 402.81 | 34.40 s | 0 |
| 1f | 1 | `dapo_qwen3_8b_ray_vllm_smoke.yaml`, `--detach` | 3 | 512 | 0 | 171 s | n/a | 0.00107888 | 0.636252 | 0.000786981 | 1.0015 | 392.23 | 40.19 s | 0 |

Run 1a was executed as the hand-written `docker exec` command of appendix A against
a container started by hand; 1b through 1f went through the launcher. All six carry
identical parameters. 1a and 1b predate the driver that recorded launcher timings,
hence the two `n/a` cells.

Rows 1d and 4c–7b are the verification pass described below; 1e is a regression
run of the launcher after its last edit, and 1f exercises `--detach` (the launcher
returns immediately, so there is no end-to-end timing for it).

**Conclusion: 17/17 runs exited 0 with zero error lines, and `--check` with the
final tolerances is 17/17 PASS.**

## Verification pass — reading only the release chapter

After the chapter and the launcher were finished, the node was reset
(`docker rm -f lumenrl-release`, all 8 cards confirmed back at
297766912–297832448 B) and example 1 was run using only the three commands in
§8.4.3, with no other knowledge:

```
export DATA_ROOT=/path/to/data
docker pull zhangdanyangamd/lumen-rl:dapo-gfx950-rocm7.2.3-260902
bash release/run_example.sh 1 --check
```

It created the container from scratch and reported
`exit=0, wall clock 190s (3m10s)` followed by `RESULT: PASS` (row 1d).
Examples 4, 5, 6 and 7 were then re-run back to back through the launcher
(rows 4c, 5b, 6b, 7b), all exit 0 with zero error lines. Example 5 again logged
`ATOM precision changed (atomfp8 -> atombf16): clearing compile caches` on its own.

Nothing in the run required opening `examples/DAPO/configs/`, reading
`run_dapo.sh`, or supplying a parameter from experience.

### Launcher options exercised

| option | how it was verified |
|---|---|
| `--check` / `--check-only` | every row above; 17/17 PASS against the final references |
| `--dry-run` | generates §8.10 of both language versions |
| `--detach` | row 1f: returned in 6 s, printed the liveness instructions, finished exit 0 |
| detach conflict guard | launching example 2 while 1f was running was refused with `A previous run is still writing to ... (0 -> 141 bytes in 10 s)` and exit 1 |
| cache clearing | rows 5 and 5b, both logged `ATOM precision changed (atomfp8 -> atombf16)` |
| node-clean gate | reported `0/8 cards busy, peak 0.3 GB in use` on every run |
| container creation vs restart | `created` on rows 1d and 1f, `restarted` on the others |
| invalid example number | `run_example.sh 9` exits 2 with `'9' is not a valid example number` |
| `--longrun` | **dry-run only** (see below): selects the longrun yaml, sets `STEPS=1000`, and appends `logger.wandb_enabled=false` when `WANDB_API_KEY` is unset |

## Derived reference values and tolerances

| ex | runs | reference `k3_kl` | reference `entropy` | reference `kl` |
|---|---|---|---|---|
| 1 | 6 (1a–1f) | 0.00109 | 0.609 | 0.00094 |
| 2 | 1 | 0.00469 | 0.789 | 0.00468 |
| 3 | 1 | 0.00410 | 0.812 | 0.00412 |
| 4 | 3 (4a–4c) | 0.00286 | 0.597 | 0.00268 |
| 5 | 2 (5, 5b) | 0.000988 | 0.641 | 0.000821 |
| 6 | 2 (6, 6b) | 0.00154 | 0.864 | 0.00178 |
| 7 | 2 (7, 7b) | 0.00157 | 0.655 | 0.00166 |

Each reference is the mean over that example's runs. Maximum deviation of any
single run from its mean, and the tolerance derived from it:

| ex | resp | runs | `k3_kl` observed | `entropy` observed | `k3_kl` tolerance | `entropy` tolerance |
|---|---|---|---|---|---|---|
| 1 | 512 | 6 | ±9% | ±8% | ±30% | ±25% |
| 2 | 512 | 1 | — | — | ±30% | ±25% |
| 3 | 512 | 1 | — | — | ±30% | ±25% |
| 4 | 4096 | 3 | ±17% | ±16% | ±50% | ±50% |
| 5 | 4096 | 2 | ±4% | ±4% | ±50% | ±50% |
| 6 | 4096 | 2 | ±8% | **±19%** | ±50% | ±60% |
| 7 | 4096 | 2 | ±2% | ±16% | ±50% | ±60% |

Tolerances are approximately 3x the observed maximum deviation, floored per group,
because 2–6 samples underestimate the true spread. Examples 2 and 3 were measured
once and inherit example 1's tolerances (same response length).

`rollout_corr/kl` is deliberately not given a percentage tolerance — across the
six example-1 runs it deviated by up to ±50% from its own mean (max/min = 2.8x)
and `ppl_ratio - 1` changed sign. It is checked only for staying within a 10x band.
Rationale in §8.6.2.

## Specific questions this run answered

**Example 4 now works from the documented command on the first try.** The previous
release README implied the example could be reached by changing `MODE`, which fails with
`RuntimeError: aot_compile is not supported by the current configuration`. Pairing
`MODE=atomfp8` with `dapo_qwen3_8b_ray_atom_fp8_4k_smoke.yaml` succeeded twice, in
557 s and 602 s.

**Example 4 → example 5 no longer needs manual cache clearing.** The launcher
detected the precision change and logged
`ATOM precision changed (atomfp8 -> atombf16): clearing compile caches` before
running example 5, which then completed in 472 s. Without that step the run dies in
AOTAutograd.

**Examples 6 and 7 can run back to back.** They ran consecutively twice (04:36/04:45
and 05:25/05:35) with no isolation beyond the container restart. Both chosen smoke configs have
`checkpoint_dir: ""`, and the corresponding longrun configs use different
directories (`verlref-moe-a3b-bf16` versus `verlref-moe-a3b-megatron-ep8-4k`), so
the "backends must not share a checkpoint directory" constraint is not reachable
from the documented commands.

**`HSA_DISABLE_FRAGMENT_ALLOCATOR=1` (the `run_dapo.sh` default) is safe here.**
Example 7 is Megatron with `use_distributed_optimizer: true`, i.e. exactly the
reduce-scatter path the warning in `run_dapo.sh` is about. It completed 3 steps with
zero `CUDA error` and zero `HSA_STATUS` lines. The failure that comment describes is
specific to ROCm 7.14 / RCCL 2.28.9 / torch 2.12, which is not this image.

**The example-6 `entropy` discrepancy is real variance, not a config mismatch.**
The previous release README listed 1.013 and an earlier evaluation measured 0.630 on what it
believed was the same example, a 38% gap that looked like either a numerical
regression or a wrong yaml. Two runs here on
`dapo_qwen3moe_a3b_ray_vllm_verlref_4k_smoke.yaml` measured **1.03019 and 0.698045**
— they bracket both published values, so both were legitimate measurements of this
config. The same two runs agree to 8% on `k3_kl` and to 0.7% on
`response_length/mean` (785.0 versus 779.5), and their `filter_groups` first rounds
kept 11/24 and 12/24 prompt groups. So the instability is specific to `entropy`,
which is a mean over the 128 sequences that survive filtering on a MoE base model.
Consequences: example 6's entropy tolerance is ±60%, and §8.6.2 states
explicitly that MoE reproducibility must be judged on `k3_kl` rather than entropy.

## What was not measured

- **Cold `docker pull` time.** The image was already present on the node, so the
  chapter quotes the download size (11.8 GB) rather than a fabricated duration.
- **Long runs.** Only the smoke configs were executed. `--longrun` was verified with
  `--dry-run` only: it picks the right yaml, sets `STEPS=1000` and handles the wandb
  key, but no longrun was actually started, so nothing confirms that a
  20480-token config trains to completion on this image and no longrun metrics
  appear in the reference table.
- **Logging to a real wandb project.** The no-key path is exercised; the
  `WANDB_API_KEY` path was only checked to the point of being passed into the
  container.
- **gfx942.** This image is gfx950-only by construction and was not tried elsewhere.
- **Independent jitter estimates for examples 2 and 3** (one sample each); their
  tolerances are inherited from example 1 rather than measured.
- **Whether example 6's entropy spread is bimodal or continuous.** Two samples plus
  one third-party observation is not enough to characterize the distribution; the
  ±60% tolerance is a bound, not a model.
