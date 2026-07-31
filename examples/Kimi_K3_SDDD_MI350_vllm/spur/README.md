# Spur cluster restart kit — Kimi K3 DSpark

Everything needed to bring a K3 DSpark training run back up on a **fresh**
`crsuse2-m2m-*` node. These scripts are not part of the LumenRL framework; they
lived in `/home/<user>/` during the July 2026 runs and are collected here so a
node swap only requires cloning this repo.

Only `/home/<user>` (NFS) survives a node change. `/mnt/m2m_nobackup` is local
XFS on NVMe and `/dev/shm` is tmpfs, so the 1.5 TB of teacher weights, the
Docker images and the dataset all have to be rebuilt from scratch — that is
what `bootstrap/` is for. Measured end-to-end on 2026-07-30: **1 h 45 min**
from an empty node to a running 9000-step job.

The watchdog (`k3_watchdog.sh`) is deliberately **not** here: it runs on the
login node, not on the compute node, and is maintained separately.

## Layout

| Path | What it does |
|---|---|
| `bootstrap/dl_k3.sh` | Download `moonshotai/Kimi-K3` (1.56 TB, 96 shards) to NVMe, 200 retries. Must go to `/mnt/m2m_nobackup`, **not** `/dev/shm` — it does not fit in 1.4 TB |
| `bootstrap/dl_dataset_102.sh` | Download `lightseekorg/kimi-mtp-dataset` into the **docker-host** `/dev/shm` and split it. Run it from inside a container with `-v /dev/shm:/dev/shm` |
| `bootstrap/pull_k3img_102.sh` | Pull the `vllm/vllm-openai-rocm:kimi-k3` pre-release base image (57.1 GB), 60 retries |
| `bootstrap/build_k3img_102.sh` | Build `kimi_k3_dspark_k3img:latest` from `../../Dockerfile.train.k3img` |
| `selfcheck/test_dspark_shapes.py` | Single-GPU forward+backward of the draft model, ~20 s. Reports peak VRAM (expect 55–58 GiB at B=1/T=2048/anchor=512). Run this before touching attention code |
| `selfcheck/test_offload_leak.py` | Checks that VRAM is actually returned to the driver after the Phase B offload |
| `selfcheck/test_k3_parser.py` | Parser unit check: out-of-range ids, all-zero loss masks, supervised ratio |
| `selfcheck/test_parser_compare.py` | Diffs this branch's `KimiK3Parser` against the upstream `dev/OPD` one (token lengths, loss mask, rendering vs the official chat template) |
| `selfcheck/test_parser_split.py` | Separates *rendering* differences from *encoding* differences — the tool that found the missing thinking-effort block |
| `selfcheck/test_span_start.py` | Prints the generation-prompt tail and the actual supervised span, to confirm where supervision starts |
| `selfcheck/their_parser_ref.py` | Frozen copy of the upstream parser; the two comparison scripts load it by path |
| `monitor/plot_progress.py` | 4-panel PNG from a training log: loss, per-position accuracy, accept length, eval accuracy |
| `monitor/watch_progress.sh` | Redraws that PNG every `STEP_EVERY` steps and archives it. Needs matplotlib, so run it **inside the training image** |

The `_102` suffix on three bootstrap scripts is historical (they were written on
`crsuse2-m2m-102`); nothing in them is node-specific.

## Restart order on a fresh node

Fix `JOBID` first and never allocate or cancel jobs yourself. All node access
goes through `spur exec "$JOBID" bash -lc '...'` — direct ssh to compute nodes
is denied, and `spur exec <id> bash` without `-c` hangs.

**Check the wall clock limit before planning step count.** `scontrol show job`
does not print `TimeLimit` on this cluster and `EndTime=N/A` does *not* mean
unlimited; use `squeue -u "$USER" -o '%.8i %.10l %.10L'` instead. A bare
`--time=360` means 360 *minutes*. A 9000-step run needs ~12.5 h.

1. **Confirm the node**: `hostname`, 8× gfx950, and whether
   `/mnt/m2m_nobackup/<user>/models` already holds the 1.5 TB. If it does, skip
   to step 5.
2. **Weights** — `bootstrap/dl_k3.sh`, ~13 min at 2 GB/s. Run it in a *detached
   docker container*, not under `spur exec`: the `spur exec` client disappears
   after ~10 minutes and silently kills anything on its process tree.
   Done when `DOWNLOAD_COMPLETE` is printed, 96 shards, zero `.incomplete`.
3. **Image** — `bootstrap/pull_k3img_102.sh` then `bootstrap/build_k3img_102.sh`,
   5–40 min total. The build ends with a version-lock assertion
   (`base pins intact; KimiK3 registered; CacheOnly* symbols present`).
4. **Dataset** — `bootstrap/dl_dataset_102.sh` from inside a container, ~5 min.
   Expect `Phase 1 (perfectblend): 296034` / `Phase 2 (mixed): 180870`.
5. **Shape self-check** — `selfcheck/test_dspark_shapes.py` in the training
   image with the GPU devices passed through, ~22 s.
6. **Smoke test** — `run_docker_spur.sh --smoke-test` with `DETACH=1`, ~9 min,
   5 steps. Verify from *inside a container* that
   `/dev/shm/teacher_cache_smoke` is empty afterwards; checking it from
   `spur exec` reads a different tmpfs and always looks empty.
7. **Short run** — 80 steps / `CACHE_BATCHES=20`, ~67 min. This is the only
   stage that exercises a vLLM restart across rounds, which is where the
   allocator bug used to strand VRAM. Free memory at round ≥ 1 must be
   > 259 GiB.
8. **Real run** — see HANDOFF §9.3. `DETACH=1` is mandatory, `resume=false` and
   a fresh `CKPT_DIR` are mandatory when you want a clean comparison, and
   `save_steps` bounds what a node failure costs.

## Traps these scripts already work around

- **Three filesystem views.** `spur exec`, the docker host and the training
  container each see a *different* `/dev/shm`. Anything training reads must be
  written from inside a container or live on `/mnt/m2m_nobackup`.
- **`HOME` is not writable** inside `spur exec` (`/opt/spur`), so every docker
  invocation needs `HOME=/home/<user> DOCKER_CONFIG=/home/<user>/.docker`.
- **Foreground `docker run` dies with the session.** Always `DETACH=1` for
  anything longer than a few minutes.
- **Bind mounts are restricted** by `spur-authz` to your own directories.

Full context, including the ten bugs behind the current code, is in
`HANDOFF-kimi-k3-dspark.md` (§4) and `amd-rl-runbook/`.
