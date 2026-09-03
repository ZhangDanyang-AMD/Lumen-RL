> [Examples README](../README.md) > Running from the release image

# 8. Running the seven examples from the release image

> 中文版：[08-release_cn.md](08-release_cn.md)

This chapter runs the seven examples in §8.2 from the **published container image**:
the software stack is pinned and the aiter kernels are already compiled, so each
example is a single command with nothing to install and no config or launch script to
open. Chapters [1](01-env-setup.md)–[4](04-launching.md) are the other path — building
an environment from source — which is what you need in order to change the code, swap
models, or run two nodes (example 8, see [chapter 7](07-disaggregated-rdma.md)).

> ⚠️ **This image supports AMD gfx950 only** (Instinct MI350X / MI355X) and requires
> 8 cards. See §8.3.1.

```bash
export DATA_ROOT=/path/to/data
docker pull zhangdanyangamd/lumen-rl:dapo-gfx950-rocm7.2.3-260902
bash release/run_example.sh 1 --check
```

Three commands run the first example and verify the result automatically. To switch
examples, change the final digit.

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
  zhangdanyangamd/lumen-rl:dapo-gfx950-rocm7.2.3-260902 \
  -lc 'ls /opt/lumenrl/aiter-jit/*.so | wc -l'     # 16
```

### 8.1.1 Pinned versions

Reproducing a result means reproducing four repositories together; they **cannot be
upgraded independently**.

| Component | Repository | Branch | Commit |
|---|---|---|---|
| Lumen-RL | `ZhangDanyang-AMD/Lumen-RL` | `dev/dsv4-dapo` | `6957ee9c1c79` |
| Lumen | `ZhangDanyang-AMD/Lumen` | `amd-atom-rollout` | `e6379cbd9057` |
| aiter | `ZhangDanyang-AMD/aiter` | `lumen/moe` | `4ebe6d69c7f4` |
| ATOM | `xysheng-AMD/ATOM` | `lumen-rl` | `7173f5b8f758` |
| composable_kernel | aiter submodule | — | `af9e1d1f1ae3` |

Base image `vllm/vllm-openai-rocm:v0.23.0`, plus `flydsl 0.3.2`,
`megatron-core 0.18.2`, ROCm Apex `daed8525`, ROCm TransformerEngine `6e541a10`.
The full list is in [`release/versions.env`](../../release/versions.env).

The container prints those four SHAs at startup. To verify the software stack:

```bash
docker exec lumenrl-release bash -lc 'python3 -c "
import aiter, lumen, lumenrl, vllm, flydsl, transformers
print(vllm.__version__, flydsl.__version__, transformers.__version__)
print(aiter.__file__)"'
```

Expect `0.23.0 0.3.2 5.12.0`, with `aiter` resolving under `/opt/lumenrl/aiter/`.

---

## 8.2 The seven examples

All seven run training *and* inference on the same 8 cards.

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

- Examples 2 and 3 share one config and differ only in `TRAIN_FP8`: `0` quantizes the
  rollout only, `1` puts the training forward pass on FP8 as well.
- Example 5 is example 4's BF16 control: the same ATOM engine with the rollout online
  quantization and the training-side FP8 both switched off.
- Example 7 is example 6's Megatron twin: the two configs are field-for-field identical
  apart from `training_backend` and `megatron_cfg`, and EP=8 gives DP=8 to match FSDP2,
  so the two metric sets can be subtracted and the difference is the training backend.

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

> ⚠️ **`MODE` and `CONFIG_OVERRIDE` must be given as a pair.** Besides selecting
> environment variables, `MODE` **appends a set of Hydra overrides**, and
> `CONFIG_OVERRIDE` only replaces the config file without cancelling them. The typical
> consequence of a mismatch: `MODE=atomfp8` unconditionally appends
> `compilation_config.level=3`, and combining that with a vLLM config yields
> `RuntimeError: aot_compile is not supported by the current configuration`.
> The launcher already pairs them correctly, so this is not a concern when using it.

All seven configs are `logger.wandb_enabled: false`, so **no wandb account is needed**
(see §8.4.6). `STEPS` is the command-line override for `num_training_steps`. None of
the seven smoke configs writes a checkpoint, so the examples can be run back to back in
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
| Image download size (compressed layers) | **11.8 GB** |
| Image unpacked on disk | **47.3 GB** |
| of which shared with `vllm/vllm-openai-rocm:v0.23.0` | 46.69 GB |
| unique to this image | 603.4 MB |

**Recommended budget**: 60 GB for the image (47.3 GB unpacked plus 11.8 GB of
compressed layers retained in the content store) plus 74 GB of models and data, so
about **134 GB**. All seven examples are smokes and write no checkpoints. A long run
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
| `models/Qwen3-30B-A3B-Base/` | 57 GB | examples 6, 7 |
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
docker pull zhangdanyangamd/lumen-rl:dapo-gfx950-rocm7.2.3-260902
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

# extra for examples 6 and 7 (about 57 GB)
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

> The data only has to be filtered once and is shared by all seven examples: the two
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
  zhangdanyangamd/lumen-rl:dapo-gfx950-rocm7.2.3-260902 sleep infinity
```

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

To run your own code against the image (all four source trees are editable installs):

```bash
docker run -d --name lumenrl-dev ... \
  -v "$PWD/Lumen-RL":/opt/lumenrl/Lumen-RL \
  zhangdanyangamd/lumen-rl:dapo-gfx950-rocm7.2.3-260902 sleep infinity
```

then `CONTAINER=lumenrl-dev bash release/run_example.sh <N>`.

### 8.4.5 The manual command, without the launcher

Below is the complete command for example 1. For the other examples, replace `MODE`,
`TRAIN_FP8`, `CONFIG_OVERRIDE`, `STEPS` and `MODEL_PATH` per the table in §8.2.2, and
for examples 6 and 7 add `-e LUMENRL_FP32_MOE_ROUTER=0`.
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
- `MODEL_PATH` defaults to the 8B model, so examples 6 and 7 silently run the wrong
  model if it is not given.
- The empty value after `PYTORCH_CUDA_ALLOC_CONF=` is not a typo: only an explicitly
  empty string turns off `expandable_segments`.

### 8.4.6 wandb

| | smoke configs (the seven in §8.2.2) | longrun configs (`--longrun`) |
|---|---|---|
| `logger.wandb_enabled` | `false` | `true` |
| Account needed | **no** | yes, `WANDB_API_KEY` |
| `max_response_length` | 512 / 4096 | 20480 (4096 for example 7) |

So the seven examples in §8.2 need no wandb account. Only `--longrun` uses it:

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
`Traceback` / `OutOfMemory` / `CUDA error` / `HSA_STATUS`, and reports PASS or FAIL.
For reading the numbers yourself, see below.

```bash
bash release/run_example.sh 1 --check
bash release/run_example.sh 1 --check-only --log $DATA_ROOT/logs/example-1-xxx.log
```

### 8.5.1 Reference values

**Measurement conditions**: 8x MI355X (gfx950), image
`dapo-gfx950-rocm7.2.3-260902`, the command being `bash release/run_example.sh <N>`
(equivalent to a full row of §8.2.2), **`seed=10086`** (fixed inside `run_dapo.sh`), and
metrics read at **step 1** (`step=1`).

| # | config (`examples/DAPO/configs/`) | steps | resp | end-to-end wall clock | `rollout_corr/k3_kl` | `entropy` | `rollout_corr/kl` (signed) | runs |
|---|---|---|---|---|---|---|---|---|
| 1 | `dapo_qwen3_8b_ray_vllm_smoke.yaml` | 3 | 512 | 156–190 s | **0.00109** ±30% | **0.609** ±25% | 0.00094 | 6 |
| 2 | `dapo_qwen3_8b_ray_vllm_fp8_smoke.yaml` | 3 | 512 | 166 s | **0.00469** ±30% | **0.789** ±25% | 0.00468 | 1 |
| 3 | `dapo_qwen3_8b_ray_vllm_fp8_smoke.yaml` (`TRAIN_FP8=1`) | 3 | 512 | 176 s | **0.00410** ±30% | **0.812** ±25% | 0.00412 | 1 |
| 4 | `dapo_qwen3_8b_ray_atom_fp8_4k_smoke.yaml` | 3 | 4096 | 557–602 s | **0.00286** ±50% | **0.597** ±50% | 0.00268 | 3 |
| 5 | `dapo_qwen3_8b_ray_atom_bf16_4k_smoke.yaml` | 1 | 4096 | 472 s | **0.000988** ±50% | **0.641** ±50% | 0.000821 | 2 |
| 6 | `dapo_qwen3moe_a3b_ray_vllm_verlref_4k_smoke.yaml` | 3 | 4096 | 552–557 s | **0.00154** ±50% | **0.864** ±60% | 0.00178 | 2 |
| 7 | `dapo_qwen3moe_a3b_ray_megatron_verlref_4k_smoke.yaml` | 3 | 4096 | 526–532 s | **0.00157** ±50% | **0.655** ±60% | 0.00166 | 2 |

The two bold columns with tolerances are what `--check` turns into PASS / FAIL; each
reference is the mean over the number of runs in the `runs` column. Across 17 runs the
**exit code was 0 every time**, all four error counts were **zero**, and `--check` is
**17/17 PASS**. The per-run record is in
[`VALIDATION-20260903.md`](../../release/VALIDATION-20260903.md).

**The wall-clock column carries no tolerance and is not part of the verdict**: it is
dominated by how warm the kernel caches are and has been measured up to ±15% off on the
same machine with the same image. Read it as a rough duration only.

### 8.5.2 Criteria

- **`rollout_corr/k3_kl` is the primary criterion**, with a tolerance of ±30% for the
  512 group (examples 1/2/3) and ±50% for the 4096 group (examples 4–7). It is the k3
  estimator of the train/rollout distribution gap: non-negative, with no cancellation
  between positive and negative contributions, and the most stable number in the table
  (worst measured deviation across all seven examples is ±17%).
- **`entropy` is the secondary criterion**, with a tolerance of ±25% for the 512 group,
  ±50% for the ATOM 4096 pair (examples 4/5) and ±60% for the MoE pair (examples 6/7).
  It is a mean over the batch that survives `filter_groups`, so the sample is small and
  the variance high — **especially on MoE, where it was measured between 0.698 and
  1.030** — which is why MoE reproducibility should be judged on `k3_kl`.
- **`rollout_corr/kl` is only an order-of-magnitude criterion.** It is a signed mean, so
  symmetric disagreement cancels inside it and repeated runs of the same command can
  differ by 2.8x; only the absolute value is checked, against a band from one tenth to
  ten times the reference. **One order of magnitude above the reference is what counts
  as wrong**, and the usual cause is model-sensitive RMSNorm not being enabled on one
  side.
- **`rollout_corr/ppl_ratio` is informational** and not part of the verdict.

The two BF16 rollouts (0.00109 for example 1, 0.000988 for example 5) sit around 1e-3;
the three FP8 ones (0.00469, 0.00410, 0.00286) are 3–4x larger, which is the price of
quantization and is expected. The two MoE runs (0.00154, 0.00157) fall in between, and
FSDP2 versus Megatron differ by only 2%, so changing the training backend introduces no
additional train/rollout drift.

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
