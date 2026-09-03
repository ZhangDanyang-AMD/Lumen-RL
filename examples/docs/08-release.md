> [Examples README](../README.md) > Running from the release image

# 8. Running the seven examples from the release image

> 中文版：[08-release_cn.md](08-release_cn.md)

**This chapter is the other entry point into this directory.** Chapters
[1](01-env-setup.md)–[4](04-launching.md) build an environment **from source**:
install dependencies, patch vLLM, precompile the ATOM JIT, then assemble the launch
command yourself. This chapter uses the **published container image**: the software
stack is already pinned with its kernels baked in, each of the seven examples is one
command, nothing has to be installed, and no config or `run_dapo.sh` has to be opened.

Both paths run the same examples — 1–7 in §8.2 are 1–7 in the
[Examples README](../README.md) "validated examples" table, so the metrics are
directly comparable. Use this chapter to qualify a new machine, reproduce the
published numbers, or hand the stack to a customer. Go back to chapters 1–4 to change
the source, swap models, or run two nodes (example 8, see
[7. Disaggregated two-node RDMA](07-disaggregated-rdma.md)).

This release of LumenRL on AMD GPUs: **training and inference share the same 8
cards**, the training backend is FSDP2 or Megatron, the rollout engine is vLLM or
ATOM, and the precision is BF16 or FP8. Everything in this chapter was run end to end
on the versions pinned in [`versions.env`](../../release/versions.env); nothing is
"expected to work". Every number in the §8.6 reference table is annotated with the
command, the config and the step count it came from.

**Shortest path** (details in §8.4):

```bash
export DATA_ROOT=/path/to/data
docker pull zhangdanyangamd/lumen-rl:dapo-gfx950-rocm7.2.3-260902
bash release/run_example.sh 1 --check
```

`release/run_example.sh` is a host-side launcher: it clears the node, starts the
container, assembles every environment variable, and after the run compares the
metrics against built-in reference values and prints `PASS` / `FAIL`.
**All seven examples differ by one digit.**

---

## 8.1 What is in this release

| | |
|---|---|
| Task | DAPO math RL (GRPO-style, per-uid group normalization) |
| Models | Qwen3-8B-Base (dense), Qwen3-30B-A3B-Base (MoE, 128 experts) |
| Training | Lumen FSDP2 (BF16 / FP8 blockwise2d), Megatron-Native (EP=8) |
| Rollout | vLLM 0.23.0 (BF16 / `fp8_per_block`), ATOM (BF16 / `per_block_fp8`) |
| Topology | 8 training actors + 8 co-located rollout replicas (TP=1) inside one Ray driver |
| Weight sync | ZMQ CUDA-IPC, same-device transfer, with coverage assertions |
| Hardware | **gfx950 only** — MI350X / MI355X, 8 cards |

Algorithm side: clip-higher + dual-clip + token-mean policy loss, dynamic sampling
(`filter_groups`), overlong reward buffer, TIS rollout correction.

---

## 8.2 The seven validated examples

All seven run training *and* inference on the same 8 cards. Measured on
8x MI355X (gfx950), ROCm 7.2, image `dapo-gfx950-rocm7.2.3-260902`, versions per
[`versions.env`](../../release/versions.env).

### 8.2.1 Overview

| # | Example | Training | Rollout | Run it |
|---|---------|----------|---------|--------|
| 1 | 8B BF16 baseline | FSDP2 BF16 | vLLM BF16 | `bash release/run_example.sh 1 --check` |
| 2 | 8B FP8 rollout | FSDP2 BF16 | vLLM `fp8_per_block` | `bash release/run_example.sh 2 --check` |
| 3 | 8B FP8 end-to-end | FSDP2 **FP8 blockwise2d** | vLLM `fp8_per_block` | `bash release/run_example.sh 3 --check` |
| 4 | 8B ATOM FP8 | FSDP2 **FP8 blockwise2d** | **ATOM** `per_block_fp8` | `bash release/run_example.sh 4 --check` |
| 5 | 8B ATOM BF16 | FSDP2 BF16 | **ATOM** BF16 | `bash release/run_example.sh 5 --check` |
| 6 | MoE FSDP2 | FSDP2 BF16 | vLLM BF16 | `bash release/run_example.sh 6 --check` |
| 7 | MoE Megatron EP=8 | **Megatron** TP=PP=CP=1, EP=8, DP=8 | vLLM BF16 | `bash release/run_example.sh 7 --check` |

### 8.2.2 Full parameters per example

This table *is* the launcher's internal table. When running by hand (§8.10),
**every column on the row has to be supplied.**

| # | `MODE` | `TRAIN_FP8` | `CONFIG_OVERRIDE` (relative to `$RL_ROOT/Lumen-RL`, all under `examples/DAPO/configs/`) | `STEPS` | `max_response_length` | Model | Extra env |
|---|---|---|---|---|---|---|---|
| 1 | `bf16` | `0` | `dapo_qwen3_8b_ray_vllm_smoke.yaml` | 3 | 512 | Qwen3-8B-Base | — |
| 2 | `fp8` | `0` | `dapo_qwen3_8b_ray_vllm_fp8_smoke.yaml` | 3 | 512 | Qwen3-8B-Base | — |
| 3 | `fp8` | `1` | `dapo_qwen3_8b_ray_vllm_fp8_smoke.yaml` | 3 | 512 | Qwen3-8B-Base | — |
| 4 | `atomfp8` | `1` | `dapo_qwen3_8b_ray_atom_fp8_4k_smoke.yaml` | 3 | 4096 | Qwen3-8B-Base | — |
| 5 | `atombf16` | `0` | `dapo_qwen3_8b_ray_atom_bf16_4k_smoke.yaml` | 1 | 4096 | Qwen3-8B-Base | — |
| 6 | `bf16` | `0` | `dapo_qwen3moe_a3b_ray_vllm_verlref_4k_smoke.yaml` | 3 | 4096 | Qwen3-30B-A3B-Base | `LUMENRL_FP32_MOE_ROUTER=0` |
| 7 | `bf16` | `0` | `dapo_qwen3moe_a3b_ray_megatron_verlref_4k_smoke.yaml` | 3 | 4096 | Qwen3-30B-A3B-Base | `LUMENRL_FP32_MOE_ROUTER=0` |

All six configs (examples 2 and 3 share one) are `logger.wandb_enabled: false`, so
**no wandb account is needed** (see §8.4.6). The `STEPS` column is the command-line `num_training_steps`
override, not the yaml default.

> ⚠️ **`MODE` and `CONFIG_OVERRIDE` must be given as a pair; changing one alone
> does not work.** `MODE` selects a set of environment variables *and* a batch of
> **appended Hydra overrides**; `CONFIG_OVERRIDE` only replaces the yaml. The
> canonical mismatch is example 4: `MODE=atomfp8` unconditionally appends
> `compilation_config.level=3`, and combining that with a vLLM smoke yaml yields
> `RuntimeError: aot_compile is not supported by the current configuration`.
> See §8.4.5 for the full `CONFIG_OVERRIDE` semantics.

### 8.2.3 How the examples relate to each other

- **Examples 2 and 3 share one yaml** and differ only in `TRAIN_FP8`: `0` means
  "rollout quantization only", `1` additionally exports
  `LUMEN_FP8=1 LUMEN_FP8_SCALING=blockwise2d` so the training forward is FP8 too.
- **Example 5 is example 4's BF16 control**: same ATOM engine, same no-eager
  level=3 + sleep2, with the rollout online quantization and the training-side FP8
  both switched off.
- **Example 7 is example 6's Megatron twin.** `configs/` offers two Megatron MoE
  smoke configs; this document picks
  `dapo_qwen3moe_a3b_ray_megatron_verlref_4k_smoke.yaml` over
  `dapo_qwen3moe_a3b_ray_megatron_smoke.yaml` for three reasons:
  1. It is **field-for-field identical to example 6's yaml except for
     `training_backend` and the `megatron_cfg` block** (same 4096 response, same
     global batch of 128, same 16 generations, same data), so the two metric sets
     can be subtracted and the difference is the training backend itself.
     `megatron_smoke` is a 512-response config and is not comparable to example 6.
  2. TP=1 / PP=1 / CP=1 / EP=8 gives DP=8, matching FSDP2's DP8, so each rank sees
     the same number of sequences per step.
  3. Its `checkpoint_dir` is an empty string, so it writes nothing (see below).

**On "the two training backends must not share a checkpoint directory"**: the
constraint still holds, but **none of the seven commands in this document hits it**
and you do not need to isolate anything by hand. The actual paths are:

| Example | smoke config `checkpoint_dir` | longrun config `checkpoint_dir` |
|---|---|---|
| 1 | `$DATA_ROOT/ckpts/lumenrl-dapo/ray-vllm-smoke` | `$DATA_ROOT/ckpts/lumenrl-dapo/longrun-ray-vllm-8b` |
| 2 / 3 | `$DATA_ROOT/ckpts/lumenrl-dapo/ray-vllm-fp8-smoke` | `$DATA_ROOT/ckpts/lumenrl-dapo/longrun-ray-vllm-fp8-8b` |
| 4 | `""` (no write) | `$DATA_ROOT/ckpts/lumenrl-dapo/longrun-ray-atom-fp8-8b` |
| 5 | `""` (no write) | `$DATA_ROOT/ckpts/lumenrl-dapo/longrun-ray-atom-bf16-8b` |
| 6 | `""` (no write) | `$SCRATCH_ROOT/ckpts/lumenrl-dapo/verlref-moe-a3b-bf16` |
| 7 | `""` (no write) | `$SCRATCH_ROOT/ckpts/lumenrl-dapo/verlref-moe-a3b-megatron-ep8-4k` |

Examples 6 and 7 both write nothing in smoke mode, and their longrun directories
differ, so **example 7 can run straight after example 6** (verified back to back,
see §8.6). The problem only appears if you point both backends at the same
`checkpoint_dir`: the two formats are mutually unreadable, and each engine sizes
its KV cache budget as a fraction of the whole card.

---

## 8.3 Requirements

- 8x gfx950 (MI350X or MI355X) with all cards idle
- Host ROCm 7.2, `/dev/kfd` and `/dev/dri` accessible
- Docker (the `docker` command must work; if it needs sudo see the `DOCKER`
  variable in §8.4.4)
- Disk: see below
- Models and data: about **74 GB**, itemized in §8.4.2

### 8.3.1 Disk (measured on node 035)

| Item | Measured | How |
|---|---|---|
| Image download size (content store, compressed) | **11.8 GB** | `docker image inspect <tag> --format '{{.Size}}'` = 11821982364 |
| Image unpacked on disk | **47.3 GB** | `SIZE` column of `docker system df -v` |
| of which shared with `vllm/vllm-openai-rocm:v0.23.0` | 46.69 GB | `SHARED SIZE` column |
| unique to this image | 603.4 MB | `UNIQUE SIZE` column |

> **Find that row in `docker system df -v` by IMAGE ID, not by REPOSITORY/TAG.**
> The same image may carry a different local tag (on the machine that built this
> release it shows up as `lumenrl release-20260902`, not as the
> `zhangdanyangamd/lumen-rl:dapo-gfx950-rocm7.2.3-260902` you pulled).
> Match on the ID: `docker image inspect <tag> --format '{{.Id}}'`, which agrees
> with the digest `docker pull` echoes back.

**Recommended budget:**

- Only the seven smokes in §8.2: **60 GB for the image** (47.3 GB unpacked plus the
  11.8 GB of compressed layers retained in the content store) **+ 74 GB of models
  and data** ≈ 134 GB. The seven examples use six distinct smoke configs
  (examples 2 and 3 share one); four have `checkpoint_dir: ""` and the other two
  name a directory but use `save_steps: 1000` while running only 3 steps, so
  **the smokes write no checkpoints at all**.
- A long run (`--longrun`) needs checkpoint space on top. An 8B FSDP2 checkpoint
  (fp32 weights + optimizer) is several times the model size, and a 30B-A3B one is
  about 342 GB (see the `_prune_old_checkpoints` comment in
  `lumenrl/trainer/callbacks.py`). `save_total_limit` sets how many are kept;
  peak usage during a write is one checkpoint.

> An earlier version of this document said "about 120 GB for the image", which is
> wrong; the numbers above were measured on 035.
> **Cold-pull time was not measured on this node** (the image was already local).
> Estimate it from the 11.8 GB download — a couple of minutes on a saturated
> 1 GbE link, depending on your registry. When the image is already present
> `docker pull` returns in seconds; that is a *warmed* machine, not the cold-node
> expectation.

### 8.3.2 gfx950 only

> **This image only runs on gfx950.** TransformerEngine and Apex are compiled with
> `NVTE_ROCM_ARCH=gfx950` / `PYTORCH_ROCM_ARCH=gfx950`, and the 16 baked-in aiter
> kernels were built on gfx950. JIT artifacts do not carry an architecture tag in
> their filename, so on gfx942 (MI300X / MI308X / MI325X) they are loaded as-is and
> fail at runtime instead of being rebuilt. Supporting gfx942 requires a separate
> build: `PYTORCH_ROCM_ARCH=gfx942 bash release/build_image.sh`.

---

## 8.4 Quick start

### 8.4.1 Get the image

```bash
docker pull zhangdanyangamd/lumen-rl:dapo-gfx950-rocm7.2.3-260902
```

Or build it yourself; these are all the steps, there is no extra manual work:

```bash
git clone -b <branch> <lumen-rl-repo> && cd Lumen-RL
bash release/build_image.sh                  # 45-60 min, mostly TransformerEngine
TAG=lumenrl:release-$(date +%Y%m%d) bash release/precompile_kernels.sh
```

`precompile_kernels.sh` needs a GPU: aiter kernels are compiled on first use and
`docker build` has no devices. Baking them in saves real time — example 4's smoke
takes **447 s** on an image with all **16** kernel objects baked in versus
**1256 s** on one with only **5**, and almost all of the gap is in large kernels like
`module_gemm_a8w8_blockscale_bpreshuffle_cktile`. A synthetic warmup only covers
5 of the 16 kernels; see the header of that script for full coverage.

> **Those two numbers are not the same measurement as the 557–602 s given for
> example 4 in §8.6.1; do not subtract one from the other.**
> 447 / 1256 s come from `release/validate_image.sh`, which runs **`STEPS=1`** and
> times only the single `docker exec` of `run_dapo.sh` — it excludes creating the
> container, probing the idle baseline and checking the software stack.
> The 557–602 s in §8.6.1 is `run_example.sh`'s end-to-end figure at **`STEPS=3`**,
> including the container restart and preflight. The difference is exactly the two
> extra steps (example 4 measured `perf/time_per_step` at 51.7–83.1 s) plus about
> 15 s of restart.
>
> **This release image is the one with all 16 objects baked in**, so 447 s is the
> figure that applies to it. The 1256 s image is the output of
> `precompile_kernels.sh`'s synthetic warmup (`<tag>-kernels`, 5 objects only) and
> is **not** an image with nothing baked in at all. Count them yourself:

```bash
docker run --rm --entrypoint /bin/bash \
  zhangdanyangamd/lumen-rl:dapo-gfx950-rocm7.2.3-260902 \
  -lc 'ls /opt/lumenrl/aiter-jit/*.so | wc -l'     # measured: 16
```

### 8.4.2 Prepare the data (self-check against this table before running)

```bash
export DATA_ROOT=/path/to/data
```

These must exist under `$DATA_ROOT`. This is exactly the list the launcher
preflights:

| Path (relative to `$DATA_ROOT`) | Measured size | Needed by |
|---|---|---|
| `models/Qwen3-8B-Base/` | 16 GB | examples 1–5, plus the tokenizer for all of them |
| `models/Qwen3-30B-A3B-Base/` | 57 GB | examples 6, 7 |
| `data_cached/qwen3-8b-maxprompt1024/dapo-math-17k.filtered.parquet` | 1.02 GB (1069626101 B) | all (train) |
| `data_cached/qwen3-8b-maxprompt1024/aime-2024.filtered.parquet` | 892 KB (913104 B) | all (val) |
| `logs/` | — | created by the launcher |

One command to self-check:

```bash
ls -la "$DATA_ROOT/models/Qwen3-8B-Base" \
       "$DATA_ROOT/data_cached/qwen3-8b-maxprompt1024"
du -sh "$DATA_ROOT/models"/*
```

#### Downloading from scratch (two steps, both inside the container)

Start the container first (§8.4.4), then:

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
# 2) filter out prompts longer than 1024 tokens, producing the two parquet files
#    above. Without this, startup spends a long time scanning overlong prompts.
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

> **Filter once, use for all seven examples.** Qwen3-8B-Base and
> Qwen3-30B-A3B-Base have byte-identical `tokenizer.json` / `vocab.json` /
> `merges.txt` (vocab 151936), so a filter computed with the 8B tokenizer is valid
> for the MoE model as well.
>
> **The MoE model must be the Base variant.** The instruct/thinking Qwen3-30B-A3B
> never closes `</think>` within `max_response_length`, so every sample is
> truncated, reward is stuck at -1, `filter_groups` keeps 0 groups for 10 rounds in
> a row, and the run raises
> `RuntimeError: filter_groups collected no valid groups`.
>
> For ModelScope mirrors (same repo IDs, same local paths) see
> [`03-data.md`](03-data.md).

### 8.4.3 Run the first example

```bash
export DATA_ROOT=/path/to/data
docker pull zhangdanyangamd/lumen-rl:dapo-gfx950-rocm7.2.3-260902
bash release/run_example.sh 1 --check
```

That is all three commands. The third prints:

```
== example 1 — 8B BF16 baseline
   config examples/DAPO/configs/dapo_qwen3_8b_ray_vllm_smoke.yaml   steps 3   model Qwen3-8B-Base
   container lumenrl-release created
   GPU idle check: 0/8 cards busy, peak 0.3 GB in use
   log: /path/to/data/logs/example-1-20260903-050508.log
   ...
== example 1 finished: exit=0, wall clock 190s (3m10s)

===== CHECK: example 1 (8B BF16 baseline) =====
  Traceback      0
  OutOfMemory    0
  CUDA error     0
  HSA_STATUS     0
  step-1 metrics:
  rollout_corr/k3_kl     0.00110719   +1.6%_vs_0.00109_(tol_30%)         PASS
  entropy                0.605645     -0.6%_vs_0.609_(tol_25%)           PASS
  rollout_corr/kl        0.000748721  |x|in[9.4e-05,0.0094]              PASS
  rollout_corr/ppl_ratio 1.00106      (informational, not checked)       INFO
RESULT: PASS
```

To switch examples, change the digit: `bash release/run_example.sh 4 --check`.
**Do not edit `MODE` / `TRAIN_FP8`** — those go together with the config, and the
launcher has already paired them for you (§8.2.2 is its internal table; `--dry-run`
expands it into the full command).

### 8.4.4 What those three commands actually do

The order is **clear the node → start the container → run → judge health**. The
launcher does all of it; the manual equivalents are given here because when
something breaks you need to run one of these steps on its own.

#### Step 1 · Clear the node (**do not skip**)

This was measured once: a leftover container from a previous tenant was holding
about 90.9 GB on each of GPUs 4–7, and because `docker ps` appeared nowhere in the
documentation the whole investigation went the wrong way.

```bash
docker ps -a                                                    # is someone else's container still there
docker exec lumenrl-release bash -lc 'rocm-smi --showmeminfo vram | grep -i used'
```

The second command needs the container to exist. **Once you have removed it, query
the host directly** — `rocm-smi` works on the host and reads the same devices, and
this is the form you need to confirm the VRAM came back after a `docker rm`:

```bash
rocm-smi --showmeminfo vram | grep -i used
```

> On the host this may also print
> `WARNING: AMD GPU device(s) is/are in a low-power state. Check power control/runtime_status`.
> Idle cards dropping into a low-power state is normal and does not affect the reading.

All eight cards should sit at the **idle baseline of about 298 MB** (297766912–297832448 B
measured on MI355X). Anything above that means a co-tenant, or an orphan process
from the previous run. The launcher makes this a hard gate: it refuses to start if
any card is above 2 GB, and prints the three likely causes with the corresponding
commands. `--force` means "I know, run anyway".

#### Step 2 · Start the container

```bash
docker run -d --name lumenrl-release \
  --network=host --ipc=host \
  --device=/dev/kfd --device=/dev/dri --group-add=video \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --shm-size 64G \
  -v "$DATA_ROOT":"$DATA_ROOT" -e DATA_ROOT="$DATA_ROOT" \
  zhangdanyangamd/lumen-rl:dapo-gfx950-rocm7.2.3-260902 sleep infinity
```

Do not name the container `lumenrl`; it collides too easily. The launcher defaults
to `lumenrl-release` and honours `CONTAINER=...`. **If the container already exists
the launcher runs `docker restart`**, because after a finished run each card may
still be holding about 90.9 GB (§8.7 item 1) and without a restart the next run gets a
smaller KV cache budget.

The container prints four fixed SHAs at startup. To verify the software stack:

```bash
docker exec lumenrl-release bash -lc 'python3 -c "
import aiter, lumen, lumenrl, vllm, flydsl, transformers
print(\"vllm\", vllm.__version__, \"flydsl\", flydsl.__version__, \"transformers\", transformers.__version__)
print(\"aiter\", aiter.__file__)"'
```

Expect `vllm 0.23.0 flydsl 0.3.2 transformers 5.12.0`, with `aiter` resolving under
`/opt/lumenrl/aiter/`.

> **The `release/` directory is not in the image.** The Dockerfile only `COPY`s
> three files and the four source trees are cloned by the image itself at fixed
> SHAs, so `/opt/lumenrl/Lumen-RL/release/` does not exist inside the container.
> `run_example.sh` is a **host-side** script: run it from the `release/` directory
> you already have, no need to enter the container.

#### Step 3 · Run

The launcher starts the job with `docker exec -d` and then polls the log file from
the host. Log paths are fixed and predictable:

```
$DATA_ROOT/logs/example-<N>-<timestamp>.log          # trainer log (written by run_dapo.sh)
$DATA_ROOT/logs/example-<N>-<timestamp>.launcher.log  # wrapper stdout + exit code
```

`--log PATH` overrides it. The manual equivalents are in **§8.10** (seven
self-contained blocks), or let the launcher generate them:

```bash
bash release/run_example.sh 4 --dry-run     # print the commands, run nothing
```

#### Step 4 · Judge health

```bash
bash release/run_example.sh 4 --check-only --log $DATA_ROOT/logs/example-4-xxx.log
```

`--check` / `--check-only` does four things: extracts step-1
`rollout_corr/k3_kl` / `entropy` / `rollout_corr/kl` / `rollout_corr/ppl_ratio`,
compares each against the built-in reference values (the §8.6 table), counts
occurrences of `Traceback` / `OutOfMemory` / `CUDA error` / `HSA_STATUS`, and prints
`PASS` / `FAIL`. For reading the numbers yourself, see §8.6.

#### All launcher options

```bash
bash release/run_example.sh --help
```

| Option / variable | Effect |
|---|---|
| `--check` | after the run, compare metrics and print PASS/FAIL |
| `--check-only --log PATH` | do not run, only validate an existing log |
| `--steps N` | override the number of training steps |
| `--longrun` | use the example's longrun config instead (see §8.4.6) |
| `--detach` | `docker exec -d` and return immediately; recommended on gated clusters (§8.11) |
| `--dry-run` | only print the `docker run` / `docker exec` it would issue |
| `--force` | auto-remediate a dirty node instead of failing |
| `--no-restart` | do not restart the container (to reuse compile caches) |
| `--keep-cache` | do not clear compile caches when switching between examples 4 and 5 (see §8.7) |
| `--verbose` | stream the whole log in the foreground instead of the highlights |
| `DATA_ROOT` | **required**. Host data directory |
| `IMAGE` / `CONTAINER` | image tag / container name |
| `DOCKER` | e.g. `DOCKER="sudo docker"` if your user is not in the docker group |
| `EXTRA_OVERRIDE` | extra Hydra overrides, space separated, appended verbatim |
| `WANDB_API_KEY` | only needed with `--longrun` |
| `STALL_LIMIT` | seconds without a new log line before declaring a hang, default 2400 |

### 8.4.5 The two semantics of `CONFIG_OVERRIDE` (read this before running by hand)

Line 156 of `run_dapo.sh` is `CONFIG="${CONFIG_OVERRIDE:-$CONFIG}"`, therefore:

1. **`CONFIG_OVERRIDE` replaces the yaml that `MODE` picked**, but
   **`MODE`'s appended Hydra overrides are still applied.**
   `MODE=atomfp8` / `atombf16` unconditionally append
   `policy.generation.vllm_cfg.enforce_eager=false`,
   `policy.generation.atom_cfg.engine_kwargs.enforce_eager=false`,
   `policy.generation.atom_cfg.engine_kwargs.compilation_config.level=3`,
   `enable_sleep_mode=true` and `sleep_level=2`.
   Stacking those on a **vLLM** smoke yaml is exactly the example-4 failure,
   `RuntimeError: aot_compile is not supported by the current configuration` —
   aot_compile is requested while torch.compile is not really on.
2. **The path is relative to `$RL_ROOT/Lumen-RL`** (the script does
   `cd "$RL_ROOT/Lumen-RL"`), so write `examples/DAPO/configs/xxx.yaml`.
   An absolute path will not be found.

Two more defaults that are easy to miss:

- **Without a config override, `MODE` selects the longrun yaml, not the smoke one.**
  `MODE=atomfp8` defaults to `dapo_qwen3_8b_ray_atom_fp8_longrun.yaml`
  (`wandb_enabled: true`, `max_response_length: 20480`). This is the second reason
  "just change `MODE` per the §8.2 table" fails.
- **`MODEL_PATH` defaults to the 8B model.** Examples 6 and 7 silently run the
  wrong model if it is not passed explicitly.

The empty value after `PYTORCH_CUDA_ALLOC_CONF=` is not a typo: the script reads it
as `${VAR-default}`, so only an explicitly empty string turns off
`expandable_segments`.

### 8.4.6 wandb

| | smoke configs (the six in §8.2.2) | longrun configs (`--longrun`) |
|---|---|---|
| `logger.wandb_enabled` | `false` | `true` |
| Account needed | **no** | yes, `WANDB_API_KEY` |
| `max_response_length` | 512 / 4096 | 20480 (4096 for example 7) |

So the seven examples in §8.2 **need no wandb account**. Only `--longrun` runs into it:

```bash
# with a key
WANDB_API_KEY=xxxx bash release/run_example.sh 1 --longrun --detach

# without a key: turn it off. The launcher adds this itself when it sees no key.
EXTRA_OVERRIDE=logger.wandb_enabled=false bash release/run_example.sh 1 --longrun --detach
```

> **The Hydra key is `logger.wandb_enabled`, not a top-level `wandb_enabled`.**
> Guessing wrong yields `ConfigKeyError: Key 'wandb_enabled' not in 'LumenRLConfig'`.
>
> ⚠️ **A missing key fails *after* `RLTrainer.setup ... complete`.** For the first
> several minutes all 8 cards are fully loaded and the log looks healthy, and only
> then does it raise `wandb.errors.errors.UsageError: No API key configured`.
> Do not read those first minutes as "it works".
>
> `run_dapo.sh` also picks up `$RL_ROOT/wandb.key` or `$RL_ROOT/../wandb.key`
> (format `KEY=xxxx`), in which case no environment variable is needed.

---

## 8.5 Developing on top of the image

The image provides the environment; your code does not have to live inside it.
Bind-mount over any of the four source trees and it takes effect immediately — all
four are editable installs.

```bash
docker run -d --name lumenrl-dev ... \
  -v "$PWD/Lumen-RL":/opt/lumenrl/Lumen-RL \
  zhangdanyangamd/lumen-rl:dapo-gfx950-rocm7.2.3-260902 sleep infinity
```

Then run the examples with `CONTAINER=lumenrl-dev bash release/run_example.sh <N>`.

**If you override `aiter`, point `AITER_JIT_DIR` at a fresh directory too.**
Compiled kernels are bound to the aiter revision that produced them, and reusing
old ones fails at import time with a message that mentions neither aiter nor the
branch:

```
AttributeError: module 'aiter.jit.module_aiter_core' has no attribute 'MlaVersion'
```

```bash
-e AITER_JIT_DIR=/tmp/aiter-jit-mybranch
```

---

## 8.6 How to tell whether the run is healthy

Check these four in order. Every one of them has cost somebody an hour.

**1. Every card is at the idle baseline before you start** (about 298 MB measured on
MI355X). See §8.4.4 step 1.

**2. The source installs win over the wheels in the image.** `import aiter` must
resolve under `/opt/lumenrl/aiter/`, not site-packages.

**3. During the run:** the log contains `RLTrainer.setup ... complete` and per-step
metrics, and contains no `Traceback`, `OutOfMemory`, `CUDA error` or `HSA_STATUS`.
`--check` counts exactly those four words.

> ⚠️ **`filter_groups round N` is not present in every example; do not treat it as
> a required marker.** That line is only emitted by configs with dynamic sampling
> enabled — when the trainer's `use_filter = bool(fg is not None and fg.enable)` is
> false it takes the single-round "no dynamic sampling" branch, which never logs it.
> **Examples 2 and 3 are in that category**: the
> `dapo_qwen3_8b_ray_vllm_fp8_smoke.yaml` they share explicitly sets
> `dynamic_sampling: false` and `filter_groups.enable: false`, and the comment in
> that file gives the reason — at `max_response_length: 512` a base model rarely
> finishes a DAPO math problem, so dynamic sampling would filter out every group
> (the longrun configs at 20480 response keep it on).
> Measured, `grep -c filter_groups` is **0** for examples 2 and 3, yet both complete
> all 3 steps with a full metric line and `--check` PASS. Examples 1, 4, 5, 6 and 7
> have it enabled, so they do log the line.

**4. The numbers line up.** See the reference table and the criteria below.

### 8.6.1 Reference values

**Measurement conditions** (every number below was taken under these; change any
one of them and the table no longer applies):

- 8x MI355X (gfx950), image `zhangdanyangamd/lumen-rl:dapo-gfx950-rocm7.2.3-260902`
- the command is exactly `bash release/run_example.sh <N>`, i.e. the full row from
  the §8.2.2 table
- **`seed=10086`** (hardcoded in `run_dapo.sh`; identical for all seven examples,
  which is why it does not get its own column); `STEPS` and `max_response_length`
  are columns in the table below
- metrics are read at **step 1** (`step=1`, 1-based)
- measured 2026-09-03 on node `crsuse2-m2m-v2-035`

| # | config (`examples/DAPO/configs/`) | steps | resp | end-to-end wall clock | `rollout_corr/k3_kl` | `entropy` | `rollout_corr/kl` (signed) | runs |
|---|---|---|---|---|---|---|---|---|
| 1 | `dapo_qwen3_8b_ray_vllm_smoke.yaml` | 3 | 512 | 156–190 s | **0.00109** ±30% | **0.609** ±25% | 0.00094 | 6 |
| 2 | `dapo_qwen3_8b_ray_vllm_fp8_smoke.yaml` | 3 | 512 | 166 s | **0.00469** ±30% | **0.789** ±25% | 0.00468 | 1 |
| 3 | `dapo_qwen3_8b_ray_vllm_fp8_smoke.yaml` (`TRAIN_FP8=1`) | 3 | 512 | 176 s | **0.00410** ±30% | **0.812** ±25% | 0.00412 | 1 |
| 4 | `dapo_qwen3_8b_ray_atom_fp8_4k_smoke.yaml` | 3 | 4096 | 557–602 s | **0.00286** ±50% | **0.597** ±50% | 0.00268 | 3 |
| 5 | `dapo_qwen3_8b_ray_atom_bf16_4k_smoke.yaml` | 1 | 4096 | 472 s | **0.000988** ±50% | **0.641** ±50% | 0.000821 | 2 |
| 6 | `dapo_qwen3moe_a3b_ray_vllm_verlref_4k_smoke.yaml` | 3 | 4096 | 552–557 s | **0.00154** ±50% | **0.864** ±60% | 0.00178 | 2 |
| 7 | `dapo_qwen3moe_a3b_ray_megatron_verlref_4k_smoke.yaml` | 3 | 4096 | 526–532 s | **0.00157** ±50% | **0.655** ±60% | 0.00166 | 2 |

The two bold columns with tolerances are what `--check` turns into PASS/FAIL; each
reference is the **mean** over the number of runs in the `runs` column, and the
tolerances are derived in §8.6.2. `rollout_corr/kl` is checked by order of magnitude
only, so it carries no percentage tolerance. "End-to-end wall clock" includes the
container restart and preflight, i.e. it is what `run_example.sh` reports.
**That column carries no tolerance and does not affect PASS/FAIL**: it is dominated
by how warm the aiter kernel and torch inductor caches are, and on the same node
with the same image it has been measured up to **±15%** off, systematically on the
fast side (one independent repeat of all seven measured, in example order,
185 / 145 / 155 / 601 / 426 / 546 / 516 s: examples 1 and 4 landed inside the ranges
above, the other five came in 1–13% lower). Read it as a rough "how long will this take", not as a
criterion. Examples 1, 4, 6 and 7 give a range because they were measured several
times; 2 and 3 show a single value because they were measured once (see the `runs`
column); example 5 was measured twice but both runs reported the same 472 s, so the
table still shows one value.
Across all 17 runs the **exit code was 0 every time**, the counts of
`Traceback` / `OutOfMemory` / `CUDA error` / `HSA_STATUS` were **all zero**, and
`--check` with the tolerances above is **17/17 PASS**.
The raw per-run record is in [`VALIDATION-20260903.md`](../../release/VALIDATION-20260903.md).

### 8.6.2 How to use this table (important)

**`rollout_corr/kl` cannot be used as a reproducibility criterion.** It is the
**signed** token mean of `(rollout_logp - train_logp)`, so symmetric disagreement
cancels inside it and what is left is mostly noise. Running **example 1 six times**
with the same command and the same `seed=10086` measured:

| | 1 | 2 | 3 | 4 | 5 | 6 | max deviation from the mean |
|---|---|---|---|---|---|---|---|
| `rollout_corr/kl` | 0.00122282 | 0.00046895 | 0.00110177 | 0.000748721 | 0.00133158 | 0.000786981 | **±50%**, max/min = 2.8x |
| `rollout_corr/ppl_ratio` | 1.00109 | 0.999031 | 1.00072 | 1.00106 | 1.00193 | 1.0015 | **the deviation from 1 flipped sign** |
| `rollout_corr/k3_kl` | 0.00110339 | 0.00117701 | 0.00105992 | 0.00110719 | 0.000994479 | 0.00107888 | ±9% |
| `entropy` | 0.635074 | 0.623073 | 0.591768 | 0.605645 | 0.560073 | 0.636252 | ±8% |

The jitter comes from the rollout side: vLLM and ATOM batch continuously and batch
composition differs between runs, which `seed` does not pin down. **It grows with
response length**, because dynamic sampling (`filter_groups`) then selects a visibly
different set of prompts. Maximum deviation from the mean, measured per example:

| example | resp | runs | `k3_kl` | `entropy` |
|---|---|---|---|---|
| 1 | 512 | 6 | ±9% | ±8% |
| 4 | 4096 | 3 | ±17% | ±16% |
| 5 | 4096 | 2 | ±4% | ±4% |
| 6 | 4096 | 2 | ±8% | **±19%** |
| 7 | 4096 | 2 | ±2% | ±16% |

(Examples 2 and 3 were measured once; their tolerances are inherited from example 1,
which shares the 512 response length.)

Hence the criteria:

- **`rollout_corr/k3_kl` is the primary criterion.** It is the k3 estimator of the
  same train/rollout gap: non-negative, no cancellation, and its worst measured
  deviation across all seven examples is ±17%. Tolerance **±30% for the 512 group
  (examples 1/2/3) and ±50% for the 4096 group (examples 4–7)**.
- **`entropy` at 4096 is only a coarse screen.** Tolerance **±25% at 512, ±50% for
  the ATOM 4k pair (examples 4/5) and ±60% for the MoE pair (examples 6/7)**.
  ⚠️ **Example 6's `entropy` is the least reproducible number in the whole table**:
  two runs measured 1.03019 and 0.698045, while the same two runs differed by only
  8% in `k3_kl` and 0.7% in `response_length/mean` (785.0 versus 779.5). Entropy is
  a mean over the 128 sequences that survive `filter_groups`, and on a MoE model the
  per-prompt spread is large, so it moves more than anything else.
  **Judge MoE reproducibility on `k3_kl`, not on entropy.**
- All tolerances are about 3x the measured maximum deviation, to allow for the
  underestimate that comes from having only 2–6 samples.
- **`rollout_corr/kl` is only checked by order of magnitude**: `--check` verifies
  that the measured absolute value lies between **1/10 and 10 times** the
  reference. **One order of magnitude above the reference is what counts as bad**,
  and the usual cause is model-sensitive RMSNorm not being enabled on one side.
- **`rollout_corr/ppl_ratio` is informational** and does not affect PASS/FAIL.

The two BF16 rollouts (0.00109 for example 1, 0.000988 for example 5) sit around
1e-3, and the three FP8 ones (0.00469, 0.00410, 0.00286 for examples 2, 3, 4) are
3–4x larger — that gap is the price of quantization and is normal. The two MoE runs
(0.00154 and 0.00157) fall in between, and FSDP2 versus Megatron differ by only 2%,
so switching the training backend introduces no extra train/rollout drift.

> **When the numbers do not match, suspect the config before the math.**
> `grep -m1 'CONFIG=' $DATA_ROOT/logs/example-<N>-*.launcher.log`
> prints the config file, `MODE`, `TRAIN_FP8` and `STEPS` actually used for that
> run. A config mismatch and a numerical regression look identical in the metrics,
> but the former is far more common.

---

## 8.7 Known issues

**1. VRAM is not released when a run finishes.** After a clean smoke each card may
still hold about 90.9 GB (measured on MI355X immediately after example 4:
89960382464–90905997312 B): the Ray workers have exited, the memory has not been
returned, and no matching process is visible from inside the container. Restart the
container between runs, otherwise the next run gets a smaller KV cache budget.

```bash
docker restart lumenrl-release   # VRAM returns to the idle baseline
```

The launcher does this before every run, so you do not have to remember it.

**2. Switching ATOM precision requires clearing the compile caches.** The torch
inductor cache is not isolated per run, so going from example 4 straight to example
5 (or back) dies in AOTAutograd:

```bash
docker exec lumenrl-release bash -lc \
  'rm -rf /tmp/aiter_configs /tmp/atom_torch_compile_cache /tmp/torchinductor_root'
```

The launcher remembers the previous ATOM example's precision and clears only when it
actually changed, so running the same example twice still reuses the cache.
`--keep-cache` disables this.

**3. `pgrep` cannot tell you whether a long run is alive.** Processes started via
`docker exec` do not share a process tree with your shell, so `pgrep` returns 0
across sessions and will tempt you into starting a second instance. **Watch whether
the log file is growing:**

```bash
watch -n 30 'ls -l $DATA_ROOT/logs/example-4-xxx.log'
```

**4. `docker restart` kills a `--detach`ed run.** The two pieces of advice conflict
(§8.7 item 1 wants a restart, `--detach` wants the container left alone). The launcher
handles it by checking whether the previous log is still growing before restarting;
if it is, it refuses to start and tells you what to do. `--force` means "kill it".

**5. `waiting for baton release` in the log is not a hang.** The 8 training actors
are waiting on one of them to JIT-compile a kernel, serialized behind a baton lock.
If you skipped `precompile_kernels.sh`, that is where roughly 20 minutes go. The
launcher's `STALL_LIMIT` (default 2400 s without a new log line before giving up)
exists to cover this stretch.

**6. `RUN_ID` has `-ray-vllm-8b-` hardcoded.** Without an explicit `LOG`, the log
filename carries that fragment even for the MoE and ATOM examples, which makes it
easy to read the wrong file. The launcher always passes `LOG` explicitly, so this
only bites when running by hand; every block in §8.10 sets `LOG`.

**7. Leaving `HSA_DISABLE_FRAGMENT_ALLOCATOR` at its default of 1 is safe on this
image.** `run_dapo.sh` exports `HSA_DISABLE_FRAGMENT_ALLOCATOR=1` by default, and
its own comment records that on **ROCm 7.14 / RCCL 2.28.9 / torch 2.12
(`rocm/primus:v26.4`)** this knob breaks intra-node reduce-scatter — which is how
Megatron's distributed optimizer reduces gradients — and the step then dies in an
**unrelated one-element allocation** with `CUDA error: invalid argument` (do not go
looking at `clip_grads`). **That combination is not this release image.** Example 7
(Megatron, `use_distributed_optimizer: true`) completed all 3 steps on this image
with the default of 1 and zero `CUDA error` / `HSA_STATUS` occurrences, so none of
the seven examples needs an explicit empty value. If you bind-mount the source onto
a different ROCm/torch combination and hit the failure, turn it off by passing an
explicit empty string: `-e HSA_DISABLE_FRAGMENT_ALLOCATOR=`.

---

## 8.8 Version pinning

Reproducing a result means reproducing four repositories together. They
**cannot be upgraded independently**; the aiter note below explains why.

| Component | Repository | Branch | Commit |
|---|---|---|---|
| Lumen-RL | `ZhangDanyang-AMD/Lumen-RL` | `dev/dsv4-dapo` | `6957ee9c1c79` |
| Lumen | `ZhangDanyang-AMD/Lumen` | `amd-atom-rollout` | `e6379cbd9057` |
| aiter | `ZhangDanyang-AMD/aiter` | `lumen/moe` | `4ebe6d69c7f4` |
| ATOM | `xysheng-AMD/ATOM` | `lumen-rl` | `7173f5b8f758` |
| composable_kernel | aiter submodule | — | `af9e1d1f1ae3` |

Base image `vllm/vllm-openai-rocm:v0.23.0`, plus `flydsl 0.3.2`,
`megatron-core 0.18.2`, ROCm Apex `daed8525`, ROCm TransformerEngine `6e541a10`.

### Why aiter is pinned to a fork

`aiter/lumen/moe` branches off `ROCm/aiter` main at `f2f8ed9b2`, the last commit
before [ROCm/aiter#5149](https://github.com/ROCm/aiter/pull/5149) reverted the batch
of Triton kernels that [#4978](https://github.com/ROCm/aiter/pull/4978) had merged
the day before. The revert covers `cross_entropy` (including the chunked variant),
`moe_aux_loss`, `moe_gemm_mxfp8`, `moe_gemm_per_token`, `fast_transpose`,
`quant_mxfp8`, `gemm_mxfp8`, `requant_fp8_row_to_col`, and the gfx942 tuning tables.
Lumen imports most of those directly, so current `ROCm/aiter` main is unusable.

The same branch also carries the newer `aiter.ops.shuffle` API
(`interleave_gate_up_rows`, `moe_shuffle_weight`), which ATOM has imported at module
top level since 2026-07-11. Pinning further back breaks ATOM instead. This branch is
the only point that satisfies both.

Two Lumen-specific changes were never upstreamed and are not on this branch either:
the `output_amax` / `output_rsigma` parameters of
`fused_rms_fp8_per_tensor_static_quant`, used only by Megatron's
`LumenLayerNormLinear` FP8 fusion path. None of the seven examples above reach it.

### If you upgrade aiter

`flydsl` has to move with it. The base image ships 0.1.4.2 while `aiter/lumen/moe`
requires `>= 0.2.4` and is pinned at 0.3.2. Getting it wrong looks like:

```
ImportError: Unsupported `flydsl` version: expected >=`0.2.4`, got `0.1.8`.
```

and it is raised from ATOM's `model_ops/moe.py` at import time, with no mention of
aiter anywhere in the message.

---

## 8.9 Further reading

Outside this chapter, the rest of this directory covers building **from source**, plus
the scenarios the image does not cover:

| When you need it | Where |
|---|---|
| Changing the source, swapping models, not using the image | [1. Environment setup](01-env-setup.md) → [2. Dependencies](02-dependencies.md) → [4. Launching](04-launching.md) |
| Rebuilding the data (§8.4.2 is the condensed version) | [3. Models and data](03-data.md) |
| A failure §8.7 does not list | [6. Troubleshooting](06-troubleshooting.md) |
| Two-node disaggregated serving (example 8, not covered by the image) | [5. Multi-node RDMA](05-multinode-rdma.md), [7. Disaggregated two-node RDMA](07-disaggregated-rdma.md) |

How the image itself is built and how the versions are pinned lives in
[`release/`](../../release/) at the repository root (`Dockerfile`, `versions.env`,
`build_image.sh`, `precompile_kernels.sh`) — those are what §8.4.1 refers to.

---

## 8.10 Manual equivalents, without the launcher

These seven blocks are the output of `run_example.sh <N> --dry-run`, one per
example, **sharing no variables**: each carries its own complete `MODE` /
`TRAIN_FP8` / `CONFIG_OVERRIDE` / `STEPS` / `MODEL_PATH` / `LOG`.
Passing environment variables via `docker exec -e` is deliberate: it removes the
need to nest quotes inside `bash -lc "..."`, and nested quoting is the main way
`CONFIG_OVERRIDE` gets swallowed or a script path gets torn apart (inside
`spur exec bash -lc '…'` you would be escaping three levels deep — **§8.11
item 6 gives a form that has been measured to work there**, do not improvise the
quoting).

Start the container as in §8.4.4 step 2 (called `lumenrl-release` below), then export
the data directory — **the `$DATA_ROOT` in the seven blocks below expands from this
one line, so there are no paths to substitute per block**:

```bash
export DATA_ROOT=/path/to/data
```

(`docker exec -e DATA_ROOT=$DATA_ROOT` is expanded by your host shell, so what
reaches the container is the resolved absolute path. That is intended.)

The log does not go to stdout; `run_dapo.sh` writes straight to `$LOG`. Follow it
with `tail -f "$LOG"` and extract metrics with
`grep -o 'step=[0-9]* .*rollout_corr/kl=[^ ]*' "$LOG"`.
**Run `docker restart lumenrl-release` between blocks** (§8.7 item 1), and
**also clear the compile caches between examples 4 and 5** (§8.7 item 2).

### Example 1 · 8B BF16 baseline

```bash
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

### Example 2 · 8B FP8 rollout

```bash
docker exec \
  -e RL_ROOT=/opt/lumenrl \
  -e DATA_ROOT=$DATA_ROOT \
  -e SCRATCH_ROOT=$DATA_ROOT \
  -e PYTORCH_CUDA_ALLOC_CONF= \
  -e MODE=fp8 \
  -e TRAIN_FP8=0 \
  -e STEPS=3 \
  -e CONFIG_OVERRIDE=examples/DAPO/configs/dapo_qwen3_8b_ray_vllm_fp8_smoke.yaml \
  -e MODEL_PATH=$DATA_ROOT/models/Qwen3-8B-Base \
  -e LOG=$DATA_ROOT/logs/example-2.log \
  lumenrl-release bash -lc 'bash /opt/lumenrl/Lumen-RL/examples/DAPO/run_dapo.sh'
```

### Example 3 · 8B FP8 end-to-end

```bash
docker exec \
  -e RL_ROOT=/opt/lumenrl \
  -e DATA_ROOT=$DATA_ROOT \
  -e SCRATCH_ROOT=$DATA_ROOT \
  -e PYTORCH_CUDA_ALLOC_CONF= \
  -e MODE=fp8 \
  -e TRAIN_FP8=1 \
  -e STEPS=3 \
  -e CONFIG_OVERRIDE=examples/DAPO/configs/dapo_qwen3_8b_ray_vllm_fp8_smoke.yaml \
  -e MODEL_PATH=$DATA_ROOT/models/Qwen3-8B-Base \
  -e LOG=$DATA_ROOT/logs/example-3.log \
  lumenrl-release bash -lc 'bash /opt/lumenrl/Lumen-RL/examples/DAPO/run_dapo.sh'
```

### Example 4 · 8B ATOM FP8

> This config is the **ATOM** 4k smoke. Taking example 1's vLLM yaml and
> changing `MODE=atomfp8` gives
> `RuntimeError: aot_compile is not supported by the current configuration`.

```bash
docker exec \
  -e RL_ROOT=/opt/lumenrl \
  -e DATA_ROOT=$DATA_ROOT \
  -e SCRATCH_ROOT=$DATA_ROOT \
  -e PYTORCH_CUDA_ALLOC_CONF= \
  -e MODE=atomfp8 \
  -e TRAIN_FP8=1 \
  -e STEPS=3 \
  -e CONFIG_OVERRIDE=examples/DAPO/configs/dapo_qwen3_8b_ray_atom_fp8_4k_smoke.yaml \
  -e MODEL_PATH=$DATA_ROOT/models/Qwen3-8B-Base \
  -e LOG=$DATA_ROOT/logs/example-4.log \
  lumenrl-release bash -lc 'bash /opt/lumenrl/Lumen-RL/examples/DAPO/run_dapo.sh'
```

### Example 5 · 8B ATOM BF16

> If the previous run was example 4, clear the compile caches first (§8.7 item 2):
> `docker exec lumenrl-release bash -lc 'rm -rf /tmp/aiter_configs /tmp/atom_torch_compile_cache /tmp/torchinductor_root'`

```bash
docker exec \
  -e RL_ROOT=/opt/lumenrl \
  -e DATA_ROOT=$DATA_ROOT \
  -e SCRATCH_ROOT=$DATA_ROOT \
  -e PYTORCH_CUDA_ALLOC_CONF= \
  -e MODE=atombf16 \
  -e TRAIN_FP8=0 \
  -e STEPS=1 \
  -e CONFIG_OVERRIDE=examples/DAPO/configs/dapo_qwen3_8b_ray_atom_bf16_4k_smoke.yaml \
  -e MODEL_PATH=$DATA_ROOT/models/Qwen3-8B-Base \
  -e LOG=$DATA_ROOT/logs/example-5.log \
  lumenrl-release bash -lc 'bash /opt/lumenrl/Lumen-RL/examples/DAPO/run_dapo.sh'
```

### Example 6 · MoE FSDP2

> `MODEL_PATH` must be given explicitly (the default is the 8B model), and
> `LUMENRL_FP32_MOE_ROUTER=0` is required so that FSDP2 and vLLM round to the same
> top-8 experts; the log should show
> `MoE router patched on 48 gates (fp32=False)`.
> `SCRATCH_ROOT` is only read by the MoE **longrun** configs, which resolve
> `checkpoint_dir` from `${oc.env:SCRATCH_ROOT}`; the smokes do not need it, but
> exporting it anyway costs nothing.

```bash
docker exec \
  -e RL_ROOT=/opt/lumenrl \
  -e DATA_ROOT=$DATA_ROOT \
  -e SCRATCH_ROOT=$DATA_ROOT \
  -e PYTORCH_CUDA_ALLOC_CONF= \
  -e MODE=bf16 \
  -e TRAIN_FP8=0 \
  -e STEPS=3 \
  -e CONFIG_OVERRIDE=examples/DAPO/configs/dapo_qwen3moe_a3b_ray_vllm_verlref_4k_smoke.yaml \
  -e MODEL_PATH=$DATA_ROOT/models/Qwen3-30B-A3B-Base \
  -e LOG=$DATA_ROOT/logs/example-6.log \
  -e LUMENRL_FP32_MOE_ROUTER=0 \
  lumenrl-release bash -lc 'bash /opt/lumenrl/Lumen-RL/examples/DAPO/run_dapo.sh'
```

### Example 7 · MoE Megatron EP=8

> `MODEL_PATH` must be given explicitly (the default is the 8B model), and
> `LUMENRL_FP32_MOE_ROUTER=0` is required so that FSDP2 and vLLM round to the same
> top-8 experts; the log should show
> `MoE router patched on 48 gates (fp32=False)`.
> `SCRATCH_ROOT` is only read by the MoE **longrun** configs, which resolve
> `checkpoint_dir` from `${oc.env:SCRATCH_ROOT}`; the smokes do not need it, but
> exporting it anyway costs nothing.

```bash
docker exec \
  -e RL_ROOT=/opt/lumenrl \
  -e DATA_ROOT=$DATA_ROOT \
  -e SCRATCH_ROOT=$DATA_ROOT \
  -e PYTORCH_CUDA_ALLOC_CONF= \
  -e MODE=bf16 \
  -e TRAIN_FP8=0 \
  -e STEPS=3 \
  -e CONFIG_OVERRIDE=examples/DAPO/configs/dapo_qwen3moe_a3b_ray_megatron_verlref_4k_smoke.yaml \
  -e MODEL_PATH=$DATA_ROOT/models/Qwen3-30B-A3B-Base \
  -e LOG=$DATA_ROOT/logs/example-7.log \
  -e LUMENRL_FP32_MOE_ROUTER=0 \
  lumenrl-release bash -lc 'bash /opt/lumenrl/Lumen-RL/examples/DAPO/run_dapo.sh'
```


## 8.11 Running on a gated or scheduler-managed cluster

The commands above assume you can type directly on the GPU host. The cluster this
release was actually validated on (Crusoe m2m plus the spur scheduler) does not work
that way, and the differences below probably apply to your environment too.

**1. The only entry point is the scheduler's `exec`; you cannot ssh to a compute
node** (`sshd` rejects ordinary users via `AllowUsers ubuntu root`):

```bash
export SPUR_CONTROLLER_ADDR=http://<controller-host>:6817
spur exec <JobID> bash -lc '<your command>'
```

`release/run_example.sh` works unchanged in that environment — all it needs is a
shell that can run `docker`:

```bash
spur exec <JobID> bash -lc '
  mkdir -p /tmp/.docker; export DOCKER_CONFIG=/tmp/.docker
  cd /path/to/Lumen-RL
  DATA_ROOT=/path/to/data bash release/run_example.sh 1 --check'
```

**2. `HOME` is not writable.** Outside the container `HOME=/root/spur` is read-only
and every command prints `bash: /root/spur/.bash_profile: Permission denied` —
harmless, but with two consequences. Before using docker you must

```bash
mkdir -p /tmp/.docker; export DOCKER_CONFIG=/tmp/.docker
```

or Docker floods warnings because it cannot write `$HOME/.docker/config.json`. And
**always use absolute paths** in scripts; do not rely on `~`.

**3. `exec` buffers stdout until the command exits**, it does not stream. So
foreground mode shows no live progress there (everything arrives at once at the
end). Use `--detach` for long runs:

```bash
spur exec <JobID> bash -lc '
  mkdir -p /tmp/.docker; export DOCKER_CONFIG=/tmp/.docker
  cd /path/to/Lumen-RL
  DATA_ROOT=/path/to/data bash release/run_example.sh 1 --longrun --detach'
```

then **poll the log file size** for liveness (`pgrep` does not work, §8.7 item 3):

```bash
spur exec <JobID> bash -lc 'ls -l /path/to/data/logs/example-1-*.log'
```

**4. Do not leave a long job in `exec`'s foreground.** When the client disconnects
the process is killed, leaving an orphan container holding VRAM — exactly what §8.4.4
step 1 has to clean up. Either use `--detach`, or put the launcher itself into the
background on the node with `setsid nohup ... &`.

**5. The conflict between `docker restart` and `--detach`** is §8.7 item 4: the
launcher probes whether the previous log is still growing before deciding to refuse
or restart.

**6. Running the manual commands of §8.10 inside `spur exec`.** Every block there
ends with `bash -lc 'bash /opt/lumenrl/…/run_dapo.sh'`, in single quotes,
while item 1 above wants the whole thing wrapped in
`spur exec <JobID> bash -lc '…'`, also in single quotes. **Nesting two layers of
single quotes breaks at the first `'`.** The form that works is to make the **outer**
layer double-quoted and leave the inner single quotes untouched:

```bash
spur exec <JobID> bash -lc "
  mkdir -p /tmp/.docker; export DOCKER_CONFIG=/tmp/.docker
  export DATA_ROOT=/path/to/data
  docker restart lumenrl-release >/dev/null
  docker exec \
    -e RL_ROOT=/opt/lumenrl \
    -e DATA_ROOT=\$DATA_ROOT \
    -e SCRATCH_ROOT=\$DATA_ROOT \
    -e PYTORCH_CUDA_ALLOC_CONF= \
    -e MODE=bf16 -e TRAIN_FP8=0 -e STEPS=3 \
    -e CONFIG_OVERRIDE=examples/DAPO/configs/dapo_qwen3_8b_ray_vllm_smoke.yaml \
    -e MODEL_PATH=\$DATA_ROOT/models/Qwen3-8B-Base \
    -e LOG=\$DATA_ROOT/logs/example-1.log \
    lumenrl-release bash -lc 'bash /opt/lumenrl/Lumen-RL/examples/DAPO/run_dapo.sh'"
```

Compared with §8.10 there is **exactly one thing to change**: inside the outer
double quotes, `$DATA_ROOT` must be written `\$DATA_ROOT`, or it expands **on your
machine** and injects a local path instead of expanding on the node. The other two
points are "leave it alone":

- keep the inner pair of single quotes exactly as it is;
- keep the line-continuation backslashes as single `\` too. Your local shell folds
  them away, so the command arrives on the node as one line — harmless, and keeping
  them is purely for readability.

This block was measured on `crsuse2-m2m-v2-035` with `exit=0` (example 1):
**3m18s for the whole block including the leading `docker restart`, and 2m31s if you
time only the `docker exec` part.** Both are correct and the difference is the
restart — the same trap as the 447 s in §8.4.1, so state what a duration covers when
you quote one.

**If you would rather not deal with the quoting at all**, do not hand-write it:
`spur exec <JobID> bash -lc '… bash release/run_example.sh 1 --check'` (the form in
item 1 above) has no nesting problem at all, and is the recommended path.
