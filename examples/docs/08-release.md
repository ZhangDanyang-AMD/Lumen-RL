> [Examples README](../README.md) > Running from the release image

# 8. Running the eight examples from the release image

> 中文版：[08-release_cn.md](08-release_cn.md)

This chapter runs the eight examples in §8.2 from the **published container image**:
the software stack is pinned and the aiter kernels are already compiled, so each
example is a single command with nothing to install. Chapters
[1](01-env-setup.md)–[4](04-launching.md) are the other path — building the
environment from source — which is what you need in order to swap models or run two
nodes (example 8, see [chapter 7](07-disaggregated-rdma.md)).

> ⚠️ **This image supports AMD gfx950 only** (Instinct MI350X / MI355X) and requires
> 8 cards. See §8.3.1.

```bash
git clone https://github.com/ZhangDanyang-AMD/Lumen-RL.git && cd Lumen-RL
export DATA_ROOT=/path/to/data
docker pull zhangdanyangamd/lumen-rl:dapo-gfx950-rocm7.2.3-260910
bash release/run_example.sh 1 --check
```

Four commands run the first example and verify the result automatically. To switch
examples, change the final digit.

**Changing the code does not mean leaving this chapter.** The image supplies the
environment; the checkout you just cloned supplies the code, bind-mounted over the
copy inside the image. Edit `lumenrl/`, run the same command again, and the change is
what runs — see §8.1.1.

---

## 8.1 What is in the image

| | |
|---|---|
| Task | DAPO math RL (GRPO-style, per-uid group normalization) |
| Models | Qwen3-8B-Base (dense), Qwen3-30B-A3B-Base (MoE, 128 experts) |
| Training backends | Lumen FSDP2 (BF16 / FP8 blockwise2d), Megatron-Native (EP=8) |
| Rollout engines | vLLM 0.23.0 (BF16 / `fp8_per_block`), ATOM (BF16 / `per_block_fp8`) |
| Topology | 8 training actors + 8 co-located rollout replicas (TP=1) inside one Ray driver |
| Weight sync | ZMQ CUDA-IPC, same-device transfer, with coverage assertions |
| Hardware | **AMD gfx950 only (MI350X / MI355X), 8 cards** |

On the algorithm side: clip-higher + dual-clip + token-mean policy loss, dynamic
sampling (`filter_groups`), an overlong reward buffer, and TIS rollout correction.

All aiter kernels in the image are **already compiled** (16 objects), so the first run
spends no time compiling them:

```bash
docker run --rm --entrypoint /bin/bash \
  zhangdanyangamd/lumen-rl:dapo-gfx950-rocm7.2.3-260910 \
  -lc 'ls /opt/lumenrl/aiter-jit/*.so | wc -l'     # 16
```

### 8.1.1 The image is the environment; your checkout is the code

`run_example.sh` bind-mounts the checkout it lives in over
`/opt/lumenrl/Lumen-RL` inside the container. So the image pins everything that is
expensive to build and awkward to get right — aiter, Lumen, ATOM, Megatron/Apex/TE,
and the compiled kernels — while Lumen-RL itself comes from your working tree:

```bash
vim lumenrl/trainer/rl_trainer.py
bash release/run_example.sh 1 --check        # runs the edit; no rebuild, no new tag
```

This is not a second copy that can disagree with the image. You already need this
checkout to have `run_example.sh` at all, so mounting it is what removes the skew: the
copy baked into the image exists only so the editable install has a path to point at,
and the mount covers it. Every run prints which checkout and which commit it used:

```
   code   /path/to/Lumen-RL @ 8ca6bdd
   image  zhangdanyangamd/lumen-rl:dapo-gfx950-rocm7.2.3-260910
```

Two consequences worth knowing:

- **A result is identified by the image digest *and* the commit**, not by the image
  alone. §8.5.1 records both, and so should you.
- **The container remembers its mount.** A bind-mount is fixed when the container is
  created, so pointing `LUMENRL_SRC` somewhere else — or coming from an older
  launcher — makes `run_example.sh` delete and recreate the container rather than
  restart it. It says so when it does.

Set `LUMENRL_SRC=/other/checkout` to run code from somewhere else.

The image only needs rebuilding when one of the three pinned upstreams below moves, or
when a system-level dependency changes. Lumen-RL changes never require it.

### 8.1.2 Pinned versions

Reproducing a result depends on the three upstream repositories below, which **cannot
be upgraded independently** and are therefore pinned by commit. Lumen-RL is this
repository and is **not pinned by the image at all** — it is whatever your checkout
holds, per §8.1.1. The copy inside the image is a fallback and is not what runs.

| Component | Repository | Branch | Commit |
|---|---|---|---|
| Lumen-RL | `ZhangDanyang-AMD/Lumen-RL` | `dev/dapo_release` | your checkout (mounted, not pinned) |
| Lumen | `ZhangDanyang-AMD/Lumen` | `amd-atom-rollout` | `e6379cbd9057` |
| aiter | `ZhangDanyang-AMD/aiter` | `lumen/moe` | `4ebe6d69c7f4` |
| ATOM | `ROCm/ATOM` | `refs/pull/2028/head` | `d6b9e147cbf6` |
| composable_kernel | aiter submodule | — | `af9e1d1f1ae3` |

ATOM is pinned at the head of upstream
[ROCm/ATOM PR #2028](https://github.com/ROCm/ATOM/pull/2028). That PR is **not merged
as of 2026-09-10**, so this is a PR head rather than a commit on main; it is fetchable
by SHA, which is how `release/Dockerfile` takes it.

⚠️ If you swap ATOM yourself, note that a no-eager ATOM rollout **depends on** two
settings that Lumen-RL's `atom_ray_server.py` supplies rather than inherit:

- `compilation_config.cudagraph_mode=FULL`. Without it the rollout aborts every
  worker on the first CUDA graph replay with `Input addresses for cudagraphs are
  different during replay`.
- `sleep_keeps_memory_resident=true`. Without it ATOM re-derives the KV block count
  on every wake and charges the colocated trainer against the budget; example 9 then
  asks for a negative pool and aborts all eight replicas in `resume_memory`. The 8B
  examples survive it and pay a graph recapture per step instead, so an ATOM that
  releases is slower on 4 and 5 and broken on 9.

Both are inert on ATOM builds that predate the fields, which is how the same code
serves an older pin. The mechanism is documented in the comments of
[`release/versions.env`](../../release/versions.env).

Base image `vllm/vllm-openai-rocm:v0.23.0`, plus `flydsl 0.3.2`,
`megatron-core 0.18.2`, ROCm Apex `daed8525`, ROCm TransformerEngine `6e541a10`.
The full list is in [`release/versions.env`](../../release/versions.env).

The container prints the HEAD of all four source trees at startup. To verify the
software stack:

```bash
docker exec lumenrl-release bash -lc 'python3 -c "
import aiter, lumen, lumenrl, vllm, flydsl, transformers
print(vllm.__version__, flydsl.__version__, transformers.__version__)
print(aiter.__file__)"'
```

Expect `0.23.0 0.3.2 5.12.0`, with `aiter` resolving under `/opt/lumenrl/aiter/`.

---

## 8.2 The eight examples

All eight run training *and* inference on the same 8 cards.

### 8.2.1 Overview

| # | Example | Training | Rollout | Command |
|---|---------|----------|---------|---------|
| 1 | 8B BF16 baseline | FSDP2 BF16 | vLLM BF16 | `bash release/run_example.sh 1 --check` |
| 2 | 8B FP8 rollout | FSDP2 BF16 | vLLM `fp8_per_block` | `bash release/run_example.sh 2 --check` |
| 3 | 8B FP8 end-to-end | FSDP2 **FP8 blockwise2d** | vLLM `fp8_per_block` | `bash release/run_example.sh 3 --check` |
| 4 | 8B ATOM FP8 | FSDP2 **FP8 blockwise2d** | **ATOM** `per_block_fp8` | `bash release/run_example.sh 4 --check` |
| 5 | 8B ATOM BF16 | FSDP2 BF16 | **ATOM** BF16 | `bash release/run_example.sh 5 --check` |
| 6 | MoE FSDP2 | FSDP2 BF16 | vLLM BF16 | `bash release/run_example.sh 6 --check` |
| 7 | MoE Megatron EP=8 | **Megatron** TP=PP=CP=1, EP=8, DP=8 | vLLM BF16 | `bash release/run_example.sh 7 --check` |
| 9 | MoE ATOM BF16 | FSDP2 BF16 | **ATOM** BF16 | `bash release/run_example.sh 9 --check` |

**There is no example 8 here.** Example 8 in the
[examples README](../README.md) is the two-node disaggregated RDMA deployment, which
needs 2x8 gfx942 and is not covered by this image; see
[chapter 7](07-disaggregated-rdma.md). The numbering is shared across the whole
examples set, so this chapter runs 1–7 and 9.

- Examples 2 and 3 share one config and differ only in `TRAIN_FP8`: `0` quantizes the
  rollout only, `1` puts the training forward pass on FP8 as well.
- Example 5 is example 4's BF16 control: the same ATOM engine with the rollout online
  quantization and the training-side FP8 both switched off.
- Example 7 is example 6's Megatron twin: the two configs are field-for-field identical
  apart from `training_backend` and `megatron_cfg`, and EP=8 gives DP=8 to match FSDP2,
  so the two metric sets can be subtracted and the difference is the training backend.
- Example 9 is example 6's ATOM twin, and the pair answers "what does switching the
  rollout engine cost?": same model, same training config, `generation_backend` vllm
  → atom plus an `atom_cfg` block, nothing else. It is also the only example that
  exercises MoE expert weights over the ATOM weight-sync path — see §8.5.2 on why
  `skipped` is a health criterion.

### 8.2.2 Full parameters per example

This table *is* the launcher's internal table. When using the manual command (§8.4.5),
every column on the row has to be supplied.

| # | `MODE` | `TRAIN_FP8` | `CONFIG_OVERRIDE` (all under `examples/DAPO/configs/`) | `STEPS` | `max_response_length` | Model | Extra env |
|---|---|---|---|---|---|---|---|
| 1 | `bf16` | `0` | `dapo_qwen3_8b_ray_vllm_smoke.yaml` | 3 | 512 | Qwen3-8B-Base | — |
| 2 | `fp8` | `0` | `dapo_qwen3_8b_ray_vllm_fp8_smoke.yaml` | 3 | 512 | Qwen3-8B-Base | — |
| 3 | `fp8` | `1` | `dapo_qwen3_8b_ray_vllm_fp8_smoke.yaml` | 3 | 512 | Qwen3-8B-Base | — |
| 4 | `atomfp8` | `1` | `dapo_qwen3_8b_ray_atom_fp8_4k_smoke.yaml` | 3 | 4096 | Qwen3-8B-Base | — |
| 5 | `atombf16` | `0` | `dapo_qwen3_8b_ray_atom_bf16_4k_smoke.yaml` | 1 | 4096 | Qwen3-8B-Base | — |
| 6 | `bf16` | `0` | `dapo_qwen3moe_a3b_ray_vllm_verlref_4k_smoke.yaml` | 3 | 4096 | Qwen3-30B-A3B-Base | `LUMENRL_FP32_MOE_ROUTER=0` |
| 7 | `bf16` | `0` | `dapo_qwen3moe_a3b_ray_megatron_verlref_4k_smoke.yaml` | 3 | 4096 | Qwen3-30B-A3B-Base | `LUMENRL_FP32_MOE_ROUTER=0` |
| 9 | `atombf16` | `0` | `dapo_qwen3moe_a3b_ray_atom_bf16_4k_smoke.yaml` | 3 | 4096 | Qwen3-30B-A3B-Base | `LUMENRL_FP32_MOE_ROUTER=0` |

> ⚠️ **`MODE` and `CONFIG_OVERRIDE` must be given as a pair.** Besides selecting
> environment variables, `MODE` **appends a set of Hydra overrides**, and
> `CONFIG_OVERRIDE` only replaces the config file without cancelling them. The typical
> consequence of a mismatch: `MODE=atomfp8` unconditionally appends
> `compilation_config.level=3`, and combining that with a vLLM config yields
> `RuntimeError: aot_compile is not supported by the current configuration`.
> The launcher already pairs them correctly, so this is not a concern when using it.

All eight configs are `logger.wandb_enabled: false`, so **no wandb account is needed**
(see §8.4.6). `STEPS` is the command-line override for `num_training_steps`. None of
the eight smoke configs writes a checkpoint, so the examples can be run back to back in
any order.

---

## 8.3 Requirements

### 8.3.1 Hardware and drivers

- **8x AMD gfx950** (Instinct MI350X or MI355X), all cards idle
- Host ROCm 7.2, with `/dev/kfd` and `/dev/dri` accessible
- Docker (if your user is not in the docker group, see the `DOCKER` variable in §8.4.4)

> ⚠️ **This image only runs on gfx950.** TransformerEngine and Apex are compiled with
> `NVTE_ROCM_ARCH=gfx950` / `PYTORCH_ROCM_ARCH=gfx950`, and the 16 aiter kernels
> compiled into the image were built on gfx950. Those JIT artifacts do not carry an
> architecture tag in their filenames, so on gfx942 (MI300X / MI308X / MI325X) they are
> loaded as-is and fail at runtime instead of being rebuilt. For gfx942, build
> separately: `PYTORCH_ROCM_ARCH=gfx942 bash release/build_image.sh`.

### 8.3.2 Disk

| Item | Measured |
|---|---|
| Image download size (compressed layers, summed from the registry manifest) | **11.8 GB** |
| Cold pull | **85 s** |
| Image unpacked on disk | **47.3 GB** |

⚠️ **If you only pull the release image, `docker system df -v` reports all 47.3 GB as
unique.** It splits into "46.7 GB shared + 607.7 MB unique" only when the base image
`vllm/vllm-openai-rocm:v0.23.0` is also present locally. A reader following this chapter
sees the former, and the table is not wrong.

The 85 s cold pull was measured once after clearing every local image and the build
cache. It is **network-bound** (about 139 MB/s here) and does not transfer to another
machine; the portable number is the 11.8 GB download size.

**Recommended budget**: 60 GB for the image (47.3 GB unpacked plus 11.8 GB of
compressed layers retained in the content store) plus 74 GB of models and data, so
about **134 GB**. All eight examples are smokes and write no checkpoints. A long run
(`--longrun`) needs checkpoint space on top — a single 30B-A3B FSDP2 checkpoint
(fp32 weights plus optimizer) is about 342 GB, and `save_total_limit` decides how many
are kept.

> When checking the table above against `docker system df -v`, **match on IMAGE ID, not
> on REPOSITORY/TAG**: the same image may carry a different local tag. Get the ID with
> `docker image inspect <tag> --format '{{.Id}}'`.

### 8.3.3 Models and data

The following must exist under `$DATA_ROOT`. This is exactly the list the launcher
preflights:

| Path (relative to `$DATA_ROOT`) | Size | Needed by |
|---|---|---|
| `models/Qwen3-8B-Base/` | 16 GB | examples 1–5, plus the tokenizer for all of them |
| `models/Qwen3-30B-A3B-Base/` | 57 GB | examples 6, 7, 9 |
| `data_cached/qwen3-8b-maxprompt1024/dapo-math-17k.filtered.parquet` | 1.02 GB | all (train) |
| `data_cached/qwen3-8b-maxprompt1024/aime-2024.filtered.parquet` | 892 KB | all (val) |
| `logs/` | — | created by the launcher |

To prepare them from scratch, see §8.4.3.

---

## 8.4 Usage

### 8.4.1 Confirm the cards are idle before starting

```bash
docker ps -a                                    # is someone else's container holding cards
rocm-smi --showmeminfo vram | grep -i used      # works directly on the host
```

All eight cards should sit at the **idle baseline of about 298 MB**
(297766912–297832448 B measured on MI355X). Anything above that means a co-tenant or an
orphan process from a previous run. The launcher makes this a hard gate: it refuses to
start if any card is above 2 GB and prints what to do about it; `--force` skips the gate.

### 8.4.2 Get the image

```bash
docker pull zhangdanyangamd/lumen-rl:dapo-gfx950-rocm7.2.3-260910
```

You can also build it yourself; these are all the steps:

```bash
git clone -b <branch> <lumen-rl-repo> && cd Lumen-RL
bash release/build_image.sh                  # 45-60 min, mostly TransformerEngine
TAG=lumenrl:release-$(date +%Y%m%d) bash release/precompile_kernels.sh
```

`precompile_kernels.sh` needs a GPU: aiter kernels are only compiled on first use and
`docker build` has no devices, so they have to be compiled in a container with cards
attached and committed into the image. That script's synthetic warmup covers 5 of the
16 kernels; see its header for how to cover all 16.

### 8.4.3 Prepare the data

```bash
export DATA_ROOT=/path/to/data
```

Check against the list in §8.3.3. Downloading from scratch takes two steps, both inside
the container (start it as in §8.4.4 first):

```bash
# 1) models and raw datasets
docker exec -e DATA_ROOT="$DATA_ROOT" lumenrl-release bash -lc '
python3 - <<PY
from huggingface_hub import snapshot_download
import os; D = os.environ["DATA_ROOT"]
snapshot_download("Qwen/Qwen3-8B-Base", local_dir=f"{D}/models/Qwen3-8B-Base",
                  allow_patterns=["*.json","*.txt","*.safetensors","*.model","tokenizer*"])
snapshot_download("BytedTsinghua-SIA/DAPO-Math-17k", repo_type="dataset",
                  local_dir=f"{D}/raw/DAPO-Math-17k")
snapshot_download("BytedTsinghua-SIA/AIME-2024", repo_type="dataset",
                  local_dir=f"{D}/raw/AIME-2024")
PY'

# extra for examples 6, 7 and 9 (about 57 GB)
docker exec -e DATA_ROOT="$DATA_ROOT" lumenrl-release bash -lc '
hf download Qwen/Qwen3-30B-A3B-Base \
  --local-dir "$DATA_ROOT/models/Qwen3-30B-A3B-Base" --max-workers 8'
```

```bash
# 2) filter out prompts longer than 1024 tokens, producing the two parquet files in §8.3.3
docker exec -e DATA_ROOT="$DATA_ROOT" lumenrl-release bash -lc '
python3 - <<PY
import os, glob, datasets
from transformers import AutoTokenizer
D = os.environ["DATA_ROOT"]; MAXLEN = 1024
OUT = f"{D}/data_cached/qwen3-8b-maxprompt1024"
tok = AutoTokenizer.from_pretrained(f"{D}/models/Qwen3-8B-Base")
def first(g): return sorted(glob.glob(g, recursive=True))[0]
jobs = [(first(f"{D}/raw/DAPO-Math-17k/**/*.parquet"), f"{OUT}/dapo-math-17k.filtered.parquet"),
        (first(f"{D}/raw/AIME-2024/**/*.parquet"),     f"{OUT}/aime-2024.filtered.parquet")]
os.makedirs(OUT, exist_ok=True)
nproc = max(1, min(64, (os.cpu_count() or 8) // 4))
for src, dst in jobs:
    ds = datasets.Dataset.from_parquet(src); n0 = len(ds)
    ds = ds.filter(lambda d: len(tok.apply_chat_template(d["prompt"], add_generation_prompt=True,
                                                        tokenize=True)) <= MAXLEN, num_proc=nproc)
    ds.to_parquet(dst); print(src, "->", dst, n0, "->", len(ds))
PY'
```

> The data only has to be filtered once and is shared by all eight examples: the two
> models have identical `tokenizer.json` / `vocab.json` / `merges.txt` (vocab 151936),
> so a filter computed with the 8B tokenizer is valid for the MoE model too.
>
> **The MoE model must be the Base variant.** The instruct / thinking Qwen3-30B-A3B does
> not close `</think>` within `max_response_length`, so every sample is truncated,
> reward is stuck at -1, `filter_groups` comes up empty for 10 rounds, and the run
> raises `RuntimeError: filter_groups collected no valid groups`.
>
> For ModelScope mirrors (same repo IDs, same local paths) see
> [`03-data.md`](03-data.md).

### 8.4.4 The launcher

`release/run_example.sh` is a host-side script. It checks that the cards are idle,
manages the container, assembles every environment variable, writes predictable log
paths, and after the run compares each metric against built-in reference values.

The launcher and this chapter are versioned together, so **use the copy in the host-side
`release/` directory**. The image ships one too, but it is fixed at the commit the image
was built from and its built-in references may predate the table in this chapter.

```bash
bash release/run_example.sh <1..7> [options]
bash release/run_example.sh --help
```

The launcher creates and reuses the container, named `lumenrl-release` by default; if
it already exists the launcher runs `docker restart` first, because after a finished run
each card may still hold about 90.9 GB (see §8.5.2). The equivalent manual form:

```bash
docker run -d --name lumenrl-release \
  --network=host --ipc=host \
  --device=/dev/kfd --device=/dev/dri --group-add=video \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --shm-size 64G \
  -v "$DATA_ROOT":"$DATA_ROOT" -e DATA_ROOT="$DATA_ROOT" \
  -v "$PWD":/opt/lumenrl/Lumen-RL \
  zhangdanyangamd/lumen-rl:dapo-gfx950-rocm7.2.3-260910 sleep infinity
```

The second mount is the code (§8.1.1), with `$PWD` being the root of this checkout.
Leave it out and the container runs the copy baked into the image instead, which is a
different commit as soon as you change anything.

Log paths are fixed:

```
$DATA_ROOT/logs/example-<N>-<timestamp>.log           # trainer log
$DATA_ROOT/logs/example-<N>-<timestamp>.launcher.log  # wrapper output and exit code
```

| Option / variable | Effect |
|---|---|
| `--check` | after the run, compare metrics and report PASS / FAIL |
| `--check-only --log PATH` | do not run, only validate an existing log |
| `--steps N` | override the number of training steps |
| `--longrun` | use the example's longrun config instead (see §8.4.6) |
| `--detach` | start and return immediately, for long runs; prints how to check liveness |
| `--dry-run` | only print the commands that would be issued |
| `--force` | auto-remediate busy cards or leftover containers instead of failing |
| `--no-restart` | do not restart the container (to reuse compile caches) |
| `--keep-cache` | do not clear compile caches between examples 4 and 5 (see §8.5.2) |
| `--verbose` | print the full log in the foreground rather than the highlights |
| `DATA_ROOT` | **required**, host data directory |
| `IMAGE` / `CONTAINER` | image tag / container name |
| `DOCKER` | e.g. `DOCKER="sudo docker"` |
| `EXTRA_OVERRIDE` | extra Hydra overrides, space separated |
| `WANDB_API_KEY` | only needed with `--longrun` |
| `STALL_LIMIT` | seconds of log silence before declaring a hang, default 2400 |

Running your own Lumen-RL needs nothing extra — that is the default, see §8.1.1. To
run it from a *different* checkout than the one holding the launcher:

```bash
LUMENRL_SRC=/other/Lumen-RL bash release/run_example.sh <N>
```

The other three trees are editable installs too, so the same mount works for them:

```bash
docker run -d --name lumenrl-dev ... \
  -v "$PWD/ATOM":/opt/lumenrl/ATOM \
  zhangdanyangamd/lumen-rl:dapo-gfx950-rocm7.2.3-260910 sleep infinity
```

then `CONTAINER=lumenrl-dev bash release/run_example.sh <N>`. Unlike Lumen-RL, those
three are what the reference values are pinned to, so a result from a swapped ATOM,
Lumen or aiter is no longer comparable with §8.5.1.

### 8.4.5 The manual command, without the launcher

Below is the complete command for example 1. For the other examples, replace `MODE`,
`TRAIN_FP8`, `CONFIG_OVERRIDE`, `STEPS` and `MODEL_PATH` per the table in §8.2.2, and
for examples 6, 7 and 9 add `-e LUMENRL_FP32_MOE_ROUTER=0`.
`bash release/run_example.sh <N> --dry-run` prints this command for any example.

```bash
export DATA_ROOT=/path/to/data

docker exec \
  -e RL_ROOT=/opt/lumenrl \
  -e DATA_ROOT=$DATA_ROOT \
  -e SCRATCH_ROOT=$DATA_ROOT \
  -e PYTORCH_CUDA_ALLOC_CONF= \
  -e MODE=bf16 \
  -e TRAIN_FP8=0 \
  -e STEPS=3 \
  -e CONFIG_OVERRIDE=examples/DAPO/configs/dapo_qwen3_8b_ray_vllm_smoke.yaml \
  -e MODEL_PATH=$DATA_ROOT/models/Qwen3-8B-Base \
  -e LOG=$DATA_ROOT/logs/example-1.log \
  lumenrl-release bash -lc 'bash /opt/lumenrl/Lumen-RL/examples/DAPO/run_dapo.sh'
```

The log does not go to stdout; `run_dapo.sh` writes straight to `$LOG`. Follow it with
`tail -f "$LOG"` and extract metrics with
`grep -o 'step=[0-9]* .*rollout_corr/kl=[^ ]*' "$LOG"`.
Run `docker restart lumenrl-release` between examples, and also clear the compile caches
between examples 4 and 5 (§8.5.2).

Four things that are easy to miss:

- `CONFIG_OVERRIDE` is **relative to `$RL_ROOT/Lumen-RL`**; an absolute path is not found.
- Without `CONFIG_OVERRIDE`, `MODE` selects the **longrun** config
  (`wandb_enabled: true`, `max_response_length: 20480`), not the smoke one.
- `MODEL_PATH` defaults to the 8B model, so examples 6, 7 and 9 silently run the wrong
  model if it is not given.
- The empty value after `PYTORCH_CUDA_ALLOC_CONF=` is not a typo: only an explicitly
  empty string turns off `expandable_segments`.

### 8.4.6 wandb

| | smoke configs (the eight in §8.2.2) | longrun configs (`--longrun`) |
|---|---|---|
| `logger.wandb_enabled` | `false` | `true` |
| Account needed | **no** | yes, `WANDB_API_KEY` |
| `max_response_length` | 512 / 4096 | 20480 (4096 for example 7) |

So the eight examples in §8.2 need no wandb account. Only `--longrun` uses it:

```bash
WANDB_API_KEY=xxxx bash release/run_example.sh 1 --longrun --detach

# without an account, switch it off; the launcher adds this itself when it sees no key
EXTRA_OVERRIDE=logger.wandb_enabled=false bash release/run_example.sh 1 --longrun --detach
```

> The Hydra key is `logger.wandb_enabled`, not a top-level `wandb_enabled`; getting it
> wrong yields `ConfigKeyError: Key 'wandb_enabled' not in 'LumenRLConfig'`.
> A missing key fails *after* `RLTrainer.setup ... complete`, so the first few minutes
> look entirely normal.

---

## 8.5 Verifying the result

`--check` performs the judgement described in this section: it extracts the four step-1
metrics and compares them against the built-in reference values, counts occurrences of
`Traceback` / `OutOfMemory` / `CUDA error` / `HSA_STATUS`, counts weight-sync buckets
that skipped a tensor, and reports PASS or FAIL. For reading the numbers yourself, see
below.

```bash
bash release/run_example.sh 1 --check
bash release/run_example.sh 1 --check-only --log $DATA_ROOT/logs/example-1-xxx.log
```

### 8.5.1 Reference values

**Measurement conditions**: 8x MI355X (gfx950), image
`dapo-gfx950-rocm7.2.3-260910` (digest `sha256:cc18a3f5ce16…`) with Lumen-RL at
`8ca6bdd` — both matter, see §8.1.1 — the command being
`bash release/run_example.sh <N>` (equivalent to a full row of §8.2.2),
**`seed=10086`** (fixed inside `run_dapo.sh`), and metrics read at **step 1**
(`step=1`).

| # | config (`examples/DAPO/configs/`) | steps | resp | trainer-log span | `rollout_corr/k3_kl` | `entropy` | `rollout_corr/kl` (signed) | runs |
|---|---|---|---|---|---|---|---|---|
| 1 | `dapo_qwen3_8b_ray_vllm_smoke.yaml` | 3 | 512 | 144 s | **0.00106** ±30% | **0.582** ±25% | 0.00106 | 1 |
| 2 | `dapo_qwen3_8b_ray_vllm_fp8_smoke.yaml` | 3 | 512 | 129 s | **0.00498** ±30% | **0.784** ±25% | 0.00538 | 1 |
| 3 | `dapo_qwen3_8b_ray_vllm_fp8_smoke.yaml` (`TRAIN_FP8=1`) | 3 | 512 | 142 s | **0.00404** ±30% | **0.832** ±25% | 0.00419 | 1 |
| 4 | `dapo_qwen3_8b_ray_atom_fp8_4k_smoke.yaml` | 3 | 4096 | 508 s | **0.00287** ±50% | **0.540** ±50% | 0.00288 | 3 |
| 5 | `dapo_qwen3_8b_ray_atom_bf16_4k_smoke.yaml` | 1 | 4096 | 402 s | **0.000930** ±50% | **0.568** ±50% | 0.000880 | 1 |
| 6 | `dapo_qwen3moe_a3b_ray_vllm_verlref_4k_smoke.yaml` | 3 | 4096 | 523 s | **0.00158** ±50% | **0.679** ±60% | 0.00154 | 1 |
| 7 | `dapo_qwen3moe_a3b_ray_megatron_verlref_4k_smoke.yaml` | 3 | 4096 | 499 s | **0.00158** ±50% | **0.660** ±60% | 0.00188 | 1 |
| 9 | `dapo_qwen3moe_a3b_ray_atom_bf16_4k_smoke.yaml` | 3 | 4096 | 579 s | **0.00138** ±50% | **0.692** ±60% | 0.00138 | 3 |

The two bold columns with tolerances are what `--check` turns into PASS / FAIL; each
reference is the mean over the number of runs in the `runs` column. The span column is
this image's single run per example.

The references are means established on the previous image, `260908`, over **12 runs**
— examples 4 and 9 three times each, one each for the rest. This image and this commit
were then run once per example: **8/8 with exit code 0**, all four error counts
**zero**, every weight-sync bucket at `skipped=0`, `--check` **8/8 PASS**, and `k3_kl`
within **±8.2%** of every reference above. They are therefore left as they stand rather
than replaced by one measurement each. The per-run record, the tolerance derivation and
what moved between the two images are in
[`VALIDATION.md`](../../release/VALIDATION.md).

Where the `runs` column reads 1, the reference is that single measurement and the
tolerance is its group's floor. Example 4 is the reason the floors are not tightened
to fit a single run: its three came in at 0.00241, 0.00243 and 0.00377, so either of
the first two, alone, would have put the third outside a ±50% band around it.

**Example 9 versus example 6 — what the rollout engine costs.** Same model, same
training config, ATOM instead of vLLM: `k3_kl` is 0.00138 against 0.00158, i.e. ATOM
is 12% *lower*, which is inside example 9's own ±6% run-to-run band doubled and well
inside the ±50% tolerance. **Switching the rollout engine does not move
train/rollout alignment measurably.** What it does move is time: 56.7 s per step
against 106.7 s, so ATOM's steps are about 1.9x faster. Its setup is the offsetting
cost — 409 s against 203 s, spent on torch.compile and capturing the graphs — so a
3-step smoke reports ATOM as slower end to end (579 s against 523 s) while the per-step
figure is what matters for a real run. Example 7 pairs the same vLLM rollout with a
Megatron actor and lands in the same place, 103.0 s per step.

**The span column carries no tolerance and is not part of the verdict**: it is the
difference between the first and last timestamp in the trainer log, is dominated by how
warm the kernel caches are, and has been measured up to ±15% off on the same machine
with the same image. The launcher's end-to-end wall clock is about 20–35 s longer
(container restart, VRAM probe and the metric check).

### 8.5.2 Criteria

- **`rollout_corr/k3_kl` is the primary criterion**, with a tolerance of ±30% for the
  512 group (examples 1/2/3) and ±50% for the 4096 group (examples 4–7 and 9). It is
  the k3 estimator of the train/rollout distribution gap: non-negative, with no
  cancellation between positive and negative contributions, far steadier than
  `entropy`, and the metric to judge a reproduction on. Measured run-to-run spread on
  the two examples that were sampled several times: ±31% for example 4 and ±6% for
  example 9.
- **`entropy` is the secondary criterion**, with a tolerance of ±25% for the 512 group,
  ±50% for the 4096 group and ±60% for the three MoE examples (6, 7 and 9).
  It is a mean over the batch that survives `filter_groups`, so the sample is small and
  the variance high — **especially on MoE, where it was measured between 0.512 and
  1.030, i.e. ±47%** — which is why MoE reproducibility should be judged on `k3_kl`.
- **`rollout_corr/kl` is only an order-of-magnitude criterion.** It is a signed mean, so
  symmetric disagreement cancels inside it and repeated runs of the same command can
  differ by 2.8x; only the absolute value is checked, against a band from one tenth to
  ten times the reference. **One order of magnitude above the reference is what counts
  as wrong**, and the usual cause is model-sensitive RMSNorm not being enabled on one
  side.
- **`rollout_corr/ppl_ratio` is informational** and not part of the verdict.
- **Every weight-sync bucket must report `skipped=0`.** This is the one criterion the
  four error counts cannot stand in for. A rollout engine that silently fails to
  update some of its weights exits 0, logs every step, and raises nothing — it just
  serves a mix of current and stale weights, which reads as a slow accuracy
  regression rather than a fault. It has happened: an ATOM MoE rollout dropped all 96
  routed expert weights per replica on every sync, 2304 times over one run, and
  passed all four counts. Read it off the log with

  ```bash
  grep -oE 'bucket done - updated=[0-9]+, skipped=[0-9]+' <log> | sort | uniq -c
  ```

  A healthy run is `skipped=0` on every line. Examples 1, 2, 3, 6 and 7 use the vLLM
  path and print no such line at all, which also counts as clean.

The two 8B BF16 rollouts (0.00106 for example 1, 0.000930 for example 5) sit around
1e-3; the three FP8 ones (0.00498, 0.00404, 0.00287) are 3–5x larger, which is the
price of quantization and is expected. The three MoE runs (0.00158, 0.00158, 0.00138)
fall in between and close to each other, so FSDP2, Megatron and the ATOM rollout all
deliver comparable train/rollout alignment on this model.

> When the numbers do not match, **first confirm you ran the same config**:
> `grep -m1 'CONFIG=' $DATA_ROOT/logs/example-<N>-*.launcher.log`
> prints the config, `MODE`, `TRAIN_FP8` and `STEPS` actually used. A config mismatch
> and a numerical regression look identical in the metrics, but the former is far more
> common.

---

## 8.6 Problem handling

**1. VRAM is not released when a run finishes.** After a smoke ends normally each card
may still hold about 90.9 GB (measured 89960382464–90905997312 B): the Ray workers have
exited but the memory has not been returned, and no matching process is visible inside
the container. Restart the container between runs, otherwise the next run gets a smaller
KV cache budget. The launcher does this before every run.

```bash
docker restart lumenrl-release
```

**2. Switching ATOM precision requires clearing the compile caches.** The torch inductor
cache is not isolated per run, so going from example 4 straight to example 5 (or back)
fails in AOTAutograd. The launcher records the previous ATOM precision and clears only
when it changed (`--keep-cache` disables this).

```bash
docker exec lumenrl-release bash -lc \
  'rm -rf /tmp/aiter_configs /tmp/atom_torch_compile_cache /tmp/torchinductor_root'
```

**3. Judge a long run's liveness from the log, not with `pgrep`.** Processes started via
`docker exec` do not share a process tree with your shell, so `pgrep` returns 0 across
sessions. **Watch whether the log file is still growing:**

```bash
watch -n 30 'ls -l $DATA_ROOT/logs/example-4-xxx.log'
```

**4. `docker restart` terminates a `--detach`ed run.** Before restarting, the launcher
checks whether the previous log is still growing; if it is, it refuses to start and says
what to do. `--force` means terminate it anyway.

**5. `waiting for baton release` in the log is not a hang.** The 8 training actors are
waiting for one of them to finish a JIT compile, serialized behind a lock. The release
image has every kernel precompiled so this should not appear; it can if you mount your
own aiter source. The launcher's `STALL_LIMIT` (default 2400 s of log silence before
giving up) leaves room for it.

**6. `filter_groups round N` does not appear for every example.** That log line is only
emitted by configs with dynamic sampling enabled. **Examples 2 and 3 do not print it** —
the config they share explicitly sets `dynamic_sampling: false` and
`filter_groups.enable: false`, because at `max_response_length: 512` a base model rarely
finishes a problem and dynamic sampling would filter out every group. This is not a
fault: both still complete all 3 steps with a full metric line and pass `--check`.

**7. When overriding the `aiter` source, change `AITER_JIT_DIR` too.** Compiled kernels
are bound to the aiter revision that produced them, and reusing the old directory fails
at import time with a message that mentions neither aiter nor the branch:

```
AttributeError: module 'aiter.jit.module_aiter_core' has no attribute 'MlaVersion'
```

Add `-e AITER_JIT_DIR=/tmp/aiter-jit-<your-branch>`.

**8. `flydsl` must be upgraded together with `aiter`.** The base image ships 0.1.4.2
while `aiter/lumen/moe` requires `>= 0.2.4` (this image pins 0.3.2). A mismatch shows up
as the import-time error below, raised from ATOM's `model_ops/moe.py` with no mention of
aiter at all:

```
ImportError: Unsupported `flydsl` version: expected >=`0.2.4`, got `0.1.8`.
```

**9. FP8 training divergence** (very low entropy, `grad_norm` and `rollout_corr/kl` both
around 1e4): see [6. Troubleshooting](06-troubleshooting.md).

**10. A negative KV pool on the wake after a weight sync** means the ATOM rollout is
releasing its memory on sleep instead of keeping it resident. It only bites the MoE
example, and only after step 0 has already succeeded:

```
ATOM/atom/rollout/memory_manager.py:137  resume_memory
AssertionError: Not enough memory for KV cache with block size(16). At least 1 block
  (1.50MB) is required, but available_for_kv=-30805.98MB (budget=86.40GB,
   peak_torch=57.68GB, non_torch=52.88GB, safety=5.76GB, free=177.39GB)
```

ATOM re-derives the block count on every wake as `gpu_memory_utilization x total` minus
everything the card reports as in use, and `non_torch` here is the rest of the node —
mostly the colocated trainer. `free=177.39GB` is the tell: the memory exists, it is just
not credited to the rollout engine.

Raising `gpu_memory_utilization` is not the fix, and neither is freeing the actors'
allocator cache — that moves `non_torch` by 0.6 GB, because `non_torch` is derived from
device-used and this platform does not return freed memory to the driver (the same
effect as item 1 above, and version-dependent — some ROCm versions do return it). Keep
the pool resident, which is what §8.1.2's second setting does; if you are on your own
ATOM branch, check that `sleep_keeps_memory_resident` reaches it.

---

## 8.7 Further reading

| Need | Where |
|---|---|
| Changing the source, swapping models, not using the image | [1. Environment setup](01-env-setup.md) → [2. Dependencies](02-dependencies.md) → [4. Launching](04-launching.md) |
| Rebuilding the data (§8.4.3 is the condensed version) | [3. Models and data](03-data.md) |
| A failure not covered by §8.6 | [6. Troubleshooting](06-troubleshooting.md) |
| Two-node disaggregated serving (example 8, not covered by the image) | [5. Multi-node RDMA](05-multinode-rdma.md), [7. Disaggregated two-node RDMA](07-disaggregated-rdma.md) |

How the image is built and how the versions are pinned lives in
[`release/`](../../release/) at the repository root.
