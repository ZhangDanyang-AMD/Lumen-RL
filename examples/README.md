# LumenRL Examples Runbook

Reproduce the DAPO math RL training examples in this directory from scratch on a
**fresh 8-GPU AMD machine**.

> 中文版见 [README_cn.md](README_cn.md) — a Chinese translation of this document is
> available at [README_cn.md](README_cn.md).

Every example shares one entrypoint (`lumenrl.trainer.main`, the Ray controller) and
one launch script (`examples/DAPO/run_dapo.sh`). All differences are expressed through
the config plus environment variables: **8 training actors and 8 co-located rollout
replicas (TP=1) inside a single Ray-driver process, with train→rollout weights synced
over ZMQ CUDA-IPC**.

On the algorithm side: clip-higher + dual-clip + token-mean policy loss, GRPO with
per-uid group normalization, dynamic sampling via `filter_groups`, an overlong reward
buffer, and TIS rollout correction.

**The whole thing in one line**: set the path variables → clone the repos → start the
container and install dependencies → (FP8 only) apply the patch → download models and
data → smoke → launch the long run with `docker exec -d`.

---

## 1. Verified examples

| # | Example | Model | Training backend | Rollout / precision | GPU | Runtime | Switch |
|---|---|---|---|---|---|---|---|
| 1 | 8B BF16 baseline | Qwen3-8B-Base | Lumen FSDP2, BF16 | vLLM / BF16 | 8× MI355X (gfx950) · 8× MI325X (gfx942) | `vllm/vllm-openai-rocm:v0.23.0` | `MODE=bf16` |
| 2 | 8B FP8 rollout | Qwen3-8B-Base | Lumen FSDP2, BF16 | vLLM / `fp8_per_block` | same | same | `MODE=fp8` |
| 3 | 8B FP8 E2E | Qwen3-8B-Base | Lumen FSDP2, **FP8 blockwise2d** | vLLM / `fp8_per_block` | same | same | `MODE=fp8 TRAIN_FP8=1` |
| 4 | 8B ATOM FP8 | Qwen3-8B-Base | Lumen FSDP2, **FP8 blockwise2d** | **ATOM** / `per_block_fp8` | same | same | `MODE=atomfp8 TRAIN_FP8=1` |
| 5 | 8B ATOM BF16 | Qwen3-8B-Base | Lumen FSDP2, BF16 (pure BF16, no Lumen norm patch) | **ATOM** / BF16 | same | same | `MODE=atombf16` |
| 6 | MoE FSDP2 | Qwen3-30B-A3B-Base | Lumen FSDP2, BF16 | vLLM / BF16 | same | same | `MODE=bf16` + MoE config |
| 7 | MoE Megatron EP=8 | Qwen3-30B-A3B-Base | **Megatron-Native**, TP=PP=CP=1 · EP=8 → DP=8 | vLLM / BF16 | same | same | `MODE=bf16` + Megatron config |

All seven examples have been run on **8× MI355X** and **8× MI325X**: smoke plus long
run, exit 0, no traceback, no OOM, no `HSA_STATUS`, weight-sync coverage assertions all
passing, and memory back to the ~298 MB/card idle baseline afterwards. Both cards use
the same configs — **no need to shorten sequences or touch any memory parameter**.

⚠️ **Example 5 is the BF16 control for example 4**: same ATOM rollout engine, same
no-eager level=3 + sleep2, with only the rollout's online quantization and the training
side's FP8 turned off. That makes it comparable to example 1 (vLLM BF16). `MODE=atombf16`
unsets every `LUMEN_FP8*` variable, so **`TRAIN_FP8` has no effect on it**.

⚠️ **You cannot run two training backends on the same cards at once**, and you cannot
share a node with someone else — the engine budgets KV cache as a fraction of the whole
card. Confirm memory is at the idle baseline before starting.

⚠️ **The two training backends cannot share a checkpoint directory**; the formats differ.

---

## 2. Path variables

Every command below uses these three variables. Moving to another machine means
changing only this block.

```bash
export RL_ROOT=/path/to/lumen_rl      # code root: Lumen-RL / Lumen / aiter / ATOM
export DATA_ROOT=/path/to/data        # models / data / logs / ckpt root
export CONTAINER=rl-vllm-fsdp
mkdir -p "$RL_ROOT" "$DATA_ROOT/logs"
```

Target layout:

```text
$RL_ROOT/
├── Lumen-RL/            # the RL framework (this repo)
├── Lumen/               # FSDP2 training backend (incl. FP8)
├── aiter/               # AMD kernels
└── ATOM/                # needed by examples 4 and 5

$DATA_ROOT/
├── models/Qwen3-8B-Base/
├── models/Qwen3-30B-A3B-Base/          # examples 6 and 7, ~57G
├── raw/                                # raw parquet
├── data_cached/qwen3-8b-maxprompt1024/ # filtered train/val
├── logs/
└── ckpts/
```

---

## 3. Clone the repos

```bash
cd "$RL_ROOT"
git clone -b dev/vllm-fsdp-dapo   https://github.com/ZhangDanyang-AMD/Lumen-RL.git
git clone -b amd-atom-rollout     https://github.com/ZhangDanyang-AMD/Lumen.git
git clone -b lumen/triton_kernels https://github.com/ZhangDanyang-AMD/aiter.git
git clone -b lumen-rl             https://github.com/xysheng-AMD/ATOM.git   # examples 4, 5

# aiter's JIT needs composable_kernel. Without it, examples 3/4/5 fail to find
# generate.py the moment they trigger module_rmsnorm.
cd "$RL_ROOT/aiter"
git submodule update --init --depth 1 3rdparty/composable_kernel
```

If GitHub is unreliable from your network, use a proxy mirror **for this command only**
— do not bake it into the repo remotes:

```bash
cd "$RL_ROOT"
GHP=https://gh-proxy.com/https://github.com
git -c http.version=HTTP/1.1 clone --depth 1 --single-branch -b dev/vllm-fsdp-dapo   "$GHP/ZhangDanyang-AMD/Lumen-RL.git"
git -c http.version=HTTP/1.1 clone --depth 1 --single-branch -b amd-atom-rollout     "$GHP/ZhangDanyang-AMD/Lumen.git"
git -c http.version=HTTP/1.1 clone --depth 1 --single-branch -b lumen/triton_kernels "$GHP/ZhangDanyang-AMD/aiter.git"
git -c http.version=HTTP/1.1 clone --depth 1 --single-branch -b lumen-rl             "$GHP/xysheng-AMD/ATOM.git"

cd "$RL_ROOT/aiter"
git -c http.version=HTTP/1.1 -c url."$GHP/".insteadOf=https://github.com/ \
  submodule update --init --depth 1 3rdparty/composable_kernel
```

Updating an existing checkout:

```bash
git -C "$RL_ROOT/Lumen-RL" pull --ff-only origin dev/vllm-fsdp-dapo
git -C "$RL_ROOT/Lumen"    pull --ff-only origin amd-atom-rollout
git -C "$RL_ROOT/aiter"    pull --ff-only origin lumen/triton_kernels
git -C "$RL_ROOT/aiter"    submodule update --init --depth 1 3rdparty/composable_kernel
```

---

## 4. Start the container

```bash
sudo docker pull vllm/vllm-openai-rocm:v0.23.0
# From a restricted network:
# sudo docker pull docker.m.daocloud.io/vllm/vllm-openai-rocm:v0.23.0
# sudo docker tag docker.m.daocloud.io/vllm/vllm-openai-rocm:v0.23.0 vllm/vllm-openai-rocm:v0.23.0

sudo docker rm -f "$CONTAINER" 2>/dev/null
sudo docker run -d --name "$CONTAINER" --entrypoint /bin/bash \
  --network=host --ipc=host \
  --device=/dev/kfd --device=/dev/dri --group-add=video \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --shm-size 64G \
  -v "$RL_ROOT":"$RL_ROOT" -v "$DATA_ROOT":"$DATA_ROOT" \
  -e RL_ROOT="$RL_ROOT" -e DATA_ROOT="$DATA_ROOT" -e HF_HOME="$DATA_ROOT/hf_home" \
  -e LUMEN_DIR="$RL_ROOT/Lumen" -e AITER_DIR="$RL_ROOT/aiter" \
  vllm/vllm-openai-rocm:v0.23.0 -lc 'sleep infinity'
```

> The `-e` variables are visible to every later `docker exec` session, so no script has
> to hardcode paths. Container `stop/start` keeps the installed dependencies; only
> `docker rm` loses them (after which §5 and §6 must be redone).

⚠️ **Confirm the container really has vLLM 0.23.0.** An existing
`vllm/vllm-openai-rocm:latest` on the machine proves nothing — it may be an older vLLM.

⚠️ **The base image must be py3.12.** The prebuilt `.so` files in the local aiter use
`_PyThreadState_UncheckedGet`, a private API that CPython 3.13 removed, so py3.13 /
py3.14 images fail immediately with
`ImportError: undefined symbol: _PyThreadState_UncheckedGet`.

---

## 5. Install dependencies

### 5.1 Base dependencies (all examples)

```bash
sudo docker exec "$CONTAINER" bash -lc '
set -e
# Allow git introspection when container root touches host-mounted repos
# (editable install / setuptools_scm).
git config --global --add safe.directory "$RL_ROOT/Lumen-RL" || true
git config --global --add safe.directory "$LUMEN_DIR" || true
git config --global --add safe.directory "$AITER_DIR" || true
git config --global --add safe.directory "$RL_ROOT/ATOM" || true

cd "$AITER_DIR" && AITER_USE_SYSTEM_TRITON=1 python3 setup.py develop || pip install -e .
pip install -e "$LUMEN_DIR" --no-deps || true
cd "$RL_ROOT/Lumen-RL" && pip install -e . --no-deps
pip install "ray[default]>=2.9" "accelerate>=0.28" datasets \
  "math_verify[antlr4_13_2]" "omegaconf>=2.3,<2.4" safetensors wandb
'
```

> Without `safe.directory`, `pip install -e` fails with
> `fatal: detected dubious ownership`. ATOM (examples 4 and 5) **needs no separate pip
> install** — `run_dapo.sh` puts `$RL_ROOT/ATOM` and `examples/DAPO/atom_aiter_shim` on
> `PYTHONPATH`, so `import atom` just works.

### 5.2 Additionally for examples 6 and 7: flydsl

```bash
sudo docker exec "$CONTAINER" bash -lc 'pip install "flydsl==0.1.8" && python3 -c "
import flydsl, transformers, vllm
print(\"flydsl\", flydsl.__version__, \"transformers\", transformers.__version__, \"vllm\", vllm.__version__)"'
```
> Expect `flydsl 0.1.8 transformers 5.12.0 vllm 0.23.0`.

**Why this is mandatory**: the image ships flydsl 0.1.4.2, which makes
`from aiter import flash_attn_varlen_func` raise a version-incompatibility error and
**kills the training forward pass outright**.

⚠️ **Do not downgrade back to the wheel's pin.** `run_dapo.sh` puts the local
`$AITER_DIR` first on `PYTHONPATH`, so what actually runs is the aiter source from the
repo, which requires flydsl ≥ `0.1.5.dev515`. The image's bundled `amd-aiter` wheel pins
`flydsl<0.1.5` and reports `cannot import name 'fly_values'` after the upgrade. The two
are mutually exclusive, and going through `run_dapo.sh`'s PYTHONPATH is the correct side.

⚠️ **transformers must be 5.x.** It fuses the Qwen3-MoE experts into 3D tensors, and the
repo's weight sync (`lumenrl/engine/inference/vllm_moe_weight_sync.py`) is written
against that layout.

### 5.3 Additionally for example 7: megatron-core / Apex / TransformerEngine

The three verified components and revisions:

- **megatron-core `0.18.2`**: `pip install --no-deps "megatron-core==0.18.2"`.
- **ROCm Apex `daed85255d51476425080e7e6203f0bee6d7e4cc`**: from source,
  `setup.py install --cpp_ext --cuda_ext`, with `PYTORCH_ROCM_ARCH=gfx950`.
- **ROCm TransformerEngine `6e541a10419a6e31bdc98b1516db04eb81a463b6`**
  (→ `2.15.0.dev0+6e541a1`): from source, `pip install -v . --no-build-isolation`,
  about 9 minutes.

```bash
sudo docker exec "$CONTAINER" bash -lc 'pip install --no-deps "megatron-core==0.18.2"'
```

Notes for building TE: it must be the **ROCm fork**, all submodules must be fetched
recursively (~5.1 GiB, including AOTriton / CK JIT / Composable Kernel), any existing
NVIDIA TE package must be uninstalled first, and the build needs
`TORCH_DONT_CHECK_COMPILER_ABI=1` — ROCm 7.2.3's `hipcc -v` returns 1 when given no
input files, and CK-JIT's compiler ABI probe reads that as "compiler unusable".

⚠️ **Never `pip install transformer_engine`.** That installs the NVIDIA build, which
fails on import with an undefined symbol.

⚠️ Do not install `megatron-bridge`. The Qwen3 HF ↔ Megatron conversion is handled by
`lumenrl/engine/training/qwen3_megatron_bridge.py`.

### 5.4 Verify

```bash
sudo docker exec "$CONTAINER" bash -lc '
python3 - <<PY
import torch, vllm, ray, lumenrl
from lumenrl.engine.inference.vllm_ray_server import VLLMRayServer
print("torch", torch.__version__, "vllm", vllm.__version__,
      "ray", ray.__version__, "GPUs", torch.cuda.device_count())
print("import OK")
PY
'
```
> Expect `GPUs 8`, `vllm 0.23.0`, `import OK`.

---

## 6. vLLM AITER RMSNorm patch (required for examples 2, 3, 4)

FP8 rollout runs with `VLLM_ROCM_USE_AITER=1`, so vLLM uses the AITER RMSNorm. It has to
be passed `use_model_sensitive_rmsnorm=1` to match the model-sensitive RMSNorm on the
Lumen training side; otherwise `rollout_corr/kl` comes out too high.

⚠️ This patch modifies **the vllm wheel inside the container**, so it is lost when the
container is recreated with `docker rm` and **must be applied once per new container**.
Examples 1, 5, 6 and 7 use BF16 rollout (`VLLM_ROCM_USE_AITER=0`) and can skip this
section.

The script is idempotent and touches only the two *plain* RMSNorm paths in
`kernels/aiter_ops.py` / `_aiter_ops.py`, never a quant-fusion path:

```bash
cat > "$RL_ROOT/patch_vllm_aiter_rmsnorm.py" <<'PYEOF'
#!/usr/bin/env python3
"""vLLM AITER RMSNorm Patch (model-sensitive / T5-like) for the rollout side.

Patches the *plain* (non-quant) AITER RMSNorm paths inside the container's
installed vLLM wheel so they pass ``use_model_sensitive_rmsnorm=1``. Quant-fusion
paths (rmsnorm2d_fwd_with_dynamicquant / *_fp8_group_quant / ...) are NOT touched.
Idempotent: safe to run repeatedly.
"""

import importlib.util
import os
import sys

RMS_OLD = """    if x.dim() > 2:
        x_original_shape = x.shape
        x = x.reshape(-1, x_original_shape[-1])
        x = rms_norm(x, weight, variance_epsilon)
        return x.reshape(x_original_shape)

    return rms_norm(x, weight, variance_epsilon)"""

RMS_NEW = """    if not getattr(_rms_norm_impl, "_lumen_logged", False):
        print("[vllm-aiter] rms_norm use_model_sensitive_rmsnorm=1", flush=True)
        _rms_norm_impl._lumen_logged = True

    if x.dim() > 2:
        x_original_shape = x.shape
        x = x.reshape(-1, x_original_shape[-1])
        x = rms_norm(x, weight, variance_epsilon, use_model_sensitive_rmsnorm=1)
        return x.reshape(x_original_shape)

    return rms_norm(x, weight, variance_epsilon, use_model_sensitive_rmsnorm=1)"""

ADD_OLD = """    rmsnorm2d_fwd_with_add(
        out,  # output
        x,  # input
        residual,  # residual input
        residual_out,  # residual output
        weight,
        variance_epsilon,
    )
    return out, residual_out"""

ADD_NEW = """    if not getattr(_rocm_aiter_rmsnorm2d_fwd_with_add_impl, "_lumen_logged", False):
        print(
            "[vllm-aiter] rmsnorm2d_fwd_with_add use_model_sensitive_rmsnorm=1",
            flush=True,
        )
        _rocm_aiter_rmsnorm2d_fwd_with_add_impl._lumen_logged = True
    rmsnorm2d_fwd_with_add(
        out,  # output
        x,  # input
        residual,  # residual input
        residual_out,  # residual output
        weight,
        variance_epsilon,
        use_model_sensitive_rmsnorm=1,
    )
    return out, residual_out"""

REPLACEMENTS = ((RMS_OLD, RMS_NEW), (ADD_OLD, ADD_NEW))


def _vllm_dir() -> str:
    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.submodule_search_locations:
        sys.exit("ERROR: vllm is not importable in this interpreter")
    return spec.submodule_search_locations[0]


def _patch_file(path: str) -> int:
    if not os.path.isfile(path):
        print(f"[skip] {path} (not found)")
        return 0
    with open(path, "r", encoding="utf-8") as fh:
        src = fh.read()
    changed = 0
    for old, new in REPLACEMENTS:
        if new in src:
            continue  # already patched
        if old in src:
            src = src.replace(old, new, 1)
            changed += 1
    if changed:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(src)
        print(f"[patched] {path} ({changed} site(s))")
    else:
        already = sum(new in src for _, new in REPLACEMENTS)
        print(f"[ok] {path} (already patched)" if already else f"[skip] {path} (no plain call sites)")
    return changed


def main() -> None:
    vdir = _vllm_dir()
    targets = [os.path.join(vdir, "kernels", "aiter_ops.py"), os.path.join(vdir, "_aiter_ops.py")]
    total = sum(_patch_file(p) for p in targets)
    print(f"[done] vLLM AITER RMSNorm patch: {total} replacement(s) applied")


if __name__ == "__main__":
    main()
PYEOF
sudo docker exec "$CONTAINER" bash -lc 'cd "$RL_ROOT" && python3 patch_vllm_aiter_rmsnorm.py'
```
> Expect `[patched] .../kernels/aiter_ops.py (2 site(s))`, or `[ok] ... (already patched)`
> if it was applied before.

Verify the patch took effect:

```bash
sudo docker exec "$CONTAINER" bash -lc '
python3 - <<PY
import inspect
from vllm.kernels import aiter_ops as k
ms = all("use_model_sensitive_rmsnorm=1" in inspect.getsource(getattr(k, a))
         for a in ["_rms_norm_impl", "_rocm_aiter_rmsnorm2d_fwd_with_add_impl"])
print("RMSNorm model-sensitive patch:", ms, "(must be True for FP8)")
PY
'
```

---

## 7. ATOM JIT precompilation (required for examples 4, 5)

Examples 4 and 5 use the local `ATOM` and local `aiter` sources rather than the kernels
built into the vLLM wheel. When ATOM starts a rollout it JIT-compiles aiter kernels on
demand, and `module_rmsnorm` is the slow one. **This is an environment setup cost, not
training performance** — compile it separately up front:

```bash
# Confirm the submodule is present (§3), otherwise generate.py will not be found.
test -f "$RL_ROOT/aiter/3rdparty/composable_kernel/example/ck_tile/10_rmsnorm2d/generate.py"

# The first run can take fifteen to twenty minutes; it is done only once
# PRECOMPILE_DONE appears.
sudo docker exec "$CONTAINER" bash -lc '
export PYTHONPATH="$AITER_DIR:${PYTHONPATH:-}"
python3 - <<PY
import torch
from aiter import rmsnorm2d_fwd
print("PRECOMPILE_START", flush=True)
x = torch.randn((1, 4096), device="cuda", dtype=torch.bfloat16)
w = torch.ones((4096,), device="cuda", dtype=torch.bfloat16)
y = rmsnorm2d_fwd(x, w, 1e-6, use_model_sensitive_rmsnorm=1)
torch.cuda.synchronize()
print("PRECOMPILE_DONE", y.shape, y.dtype, flush=True)
PY
'
```

A `Ctrl-C`, container restart or `pkill` can leave a stale lock behind. The symptom is a
process that keeps printing `waiting for baton release at .../lock_module_rmsnorm` while
no `ninja` / `hipcc` / `clang-22` compile process exists. Clean up and rerun the
precompile:

```bash
sudo docker exec "$CONTAINER" bash -lc '
rm -rf "$AITER_DIR/aiter/jit/build/lock_module_rmsnorm" \
       "$AITER_DIR/aiter/jit/build/module_rmsnorm"
'
```

> Examples 2 and 3 do not need this step; they use the vLLM/AITER path already installed
> in the image.

---

## 8. Models and data

### 8.1 Download

```bash
sudo docker exec "$CONTAINER" bash -lc '
python3 - <<PY
from huggingface_hub import snapshot_download
import os; D = os.environ["DATA_ROOT"]
snapshot_download("Qwen/Qwen3-8B-Base", local_dir=f"{D}/models/Qwen3-8B-Base",
                  allow_patterns=["*.json","*.txt","*.safetensors","*.model","tokenizer*"])
snapshot_download("BytedTsinghua-SIA/DAPO-Math-17k", repo_type="dataset",
                  local_dir=f"{D}/raw/DAPO-Math-17k")
snapshot_download("BytedTsinghua-SIA/AIME-2024", repo_type="dataset",
                  local_dir=f"{D}/raw/AIME-2024")
PY
'

# Additionally for examples 6 and 7 (~57G)
sudo docker exec "$CONTAINER" bash -lc '
hf download Qwen/Qwen3-30B-A3B-Base \
  --local-dir "$DATA_ROOT/models/Qwen3-30B-A3B-Base" --max-workers 8'
```

From a restricted network use ModelScope instead. The IDs are identical
(`Qwen/Qwen3-8B-Base`, `Qwen/Qwen3-30B-A3B-Base`, `BytedTsinghua-SIA/DAPO-Math-17k`,
`BytedTsinghua-SIA/AIME-2024`), everything lands in the same local paths, and no later
command changes:

```bash
sudo docker exec "$CONTAINER" bash -lc '
pip install modelscope
python3 - <<PY
from modelscope.hub.snapshot_download import snapshot_download
import os
D = os.environ["DATA_ROOT"]
snapshot_download("Qwen/Qwen3-8B-Base", local_dir=f"{D}/models/Qwen3-8B-Base",
    allow_patterns=["*.json","*.txt","*.safetensors","*.model","tokenizer*","*.py","*.tiktoken"],
    max_workers=8)
for rid, sub in (("BytedTsinghua-SIA/DAPO-Math-17k", "DAPO-Math-17k"),
                 ("BytedTsinghua-SIA/AIME-2024", "AIME-2024")):
    snapshot_download(repo_id=rid, repo_type="dataset", local_dir=f"{D}/raw/{sub}",
        allow_patterns=["*.parquet","*.json","*.jsonl","*.md","*.txt"], max_workers=4)
PY
'
```

⚠️ **The MoE example requires the Base model.** The instruct/thinking Qwen3-30B-A3B
**never closes `</think>`** within `max_response_length` (measured: still unclosed at
3072 tokens, and no `\boxed` either), so every sample gets truncated, reward is
permanently −1, `filter_groups` keeps 0 for 10 consecutive rounds, and the run dies with
`RuntimeError: filter_groups collected no valid groups`. The Base model emits `Answer:`
normally.

### 8.2 Filter prompts to ≤1024

Without pre-filtering, startup enters a slow overlong-prompt scan.

```bash
cat > "$RL_ROOT/filter_prompts.py" <<'PYEOF'
import os, glob
import datasets
from transformers import AutoTokenizer

DATA = os.environ["DATA_ROOT"]
MODEL_PATH = f"{DATA}/models/Qwen3-8B-Base"
MAX_PROMPT_LENGTH = 1024
PROMPT_KEY = "prompt"
OUT_DIR = f"{DATA}/data_cached/qwen3-8b-maxprompt1024"

def first_parquet(*dir_globs):
    for g in dir_globs:
        hits = sorted(glob.glob(g, recursive=True))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"no parquet under {dir_globs}")

JOBS = [
    (first_parquet(f"{DATA}/raw/DAPO-Math-17k/**/*.parquet"),
     os.path.join(OUT_DIR, "dapo-math-17k.filtered.parquet")),
    (first_parquet(f"{DATA}/raw/AIME-2024/**/*.parquet"),
     os.path.join(OUT_DIR, "aime-2024.filtered.parquet")),
]

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

def doc2len(doc) -> int:
    return len(tokenizer.apply_chat_template(doc[PROMPT_KEY], add_generation_prompt=True, tokenize=True))

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    nproc = max(1, min(64, (os.cpu_count() or 8) // 4))
    for src, dst in JOBS:
        ds = datasets.Dataset.from_parquet(src)
        before = len(ds)
        ds = ds.filter(lambda d: doc2len(d) <= MAX_PROMPT_LENGTH, num_proc=nproc,
                       desc=f"Filtering prompts > {MAX_PROMPT_LENGTH} tokens")
        ds.to_parquet(dst)
        print(f"[{src}] -> {dst}: {before} -> {len(ds)} (removed {before-len(ds)})")

if __name__ == "__main__":
    main()
PYEOF
sudo docker exec "$CONTAINER" bash -lc 'cd "$RL_ROOT" && python3 filter_prompts.py'
```

The outputs are exactly `run_dapo.sh`'s default `TRAIN_FILE` / `VAL_FILE`:

```text
$DATA_ROOT/data_cached/qwen3-8b-maxprompt1024/dapo-math-17k.filtered.parquet   # train
$DATA_ROOT/data_cached/qwen3-8b-maxprompt1024/aime-2024.filtered.parquet       # val
```

> **Filter once, share across all examples.** Qwen3-8B-Base and Qwen3-30B-A3B-Base have
> byte-identical `tokenizer.json` / `vocab.json` / `merges.txt` (vocab 151936), so the
> filtering done with the 8B tokenizer holds for the MoE model too.

---

## 9. Configs and scale

All under `examples/DAPO/configs/`:

```text
# 1  8B BF16
dapo_qwen3_8b_ray_vllm_smoke.yaml                     resp=512
dapo_qwen3_8b_ray_vllm_longrun.yaml

# 2, 3  8B vLLM FP8 (shared config; training precision selected by TRAIN_FP8)
dapo_qwen3_8b_ray_vllm_fp8_smoke.yaml                 resp=512
dapo_qwen3_8b_ray_vllm_fp8_4k_smoke.yaml              resp=4096
dapo_qwen3_8b_ray_vllm_fp8_longrun.yaml

# 4  8B ATOM FP8
dapo_qwen3_8b_ray_atom_fp8_4k_smoke.yaml              resp=4096
dapo_qwen3_8b_ray_atom_fp8_longrun.yaml

# 5  8B ATOM BF16 (field-for-field identical to example 4, quantization off)
dapo_qwen3_8b_ray_atom_bf16_4k_smoke.yaml             resp=4096
dapo_qwen3_8b_ray_atom_bf16_longrun.yaml

# 6  MoE FSDP2
dapo_qwen3moe_a3b_ray_vllm_verlref_4k_smoke.yaml
dapo_qwen3moe_a3b_ray_vllm_verlref_longrun.yaml

# 7  MoE Megatron EP=8
dapo_qwen3moe_a3b_ray_megatron_verlref_4k_smoke.yaml
dapo_qwen3moe_a3b_ray_megatron_verlref_longrun.yaml
dapo_qwen3moe_a3b_ray_megatron_verlref_4k_longrun.yaml   # compressed, conclusive in hours
```

**8B long-run scale** (identical for examples 1–5): 1000 steps,
`train_global_batch_size=512` (32 prompts × 16), `gen_batch_size=96`,
`max_response_length=20480`, `max_total_sequence_length=21504`, lr 1e-6 / warmup 10 /
wd 0.1 / clip_grad 1.0, clip 0.2/0.28/10 + token-mean, `overlong_buffer` 512/1.0,
`filter_groups` on acc with at most 10 rounds, `rollout_is=token` with threshold 2.0,
`val_steps=10` / `save_steps=50` / seed 10086.
The BF16 and FP8 configs **differ by exactly one line, `vllm_cfg.quantization`**.

**MoE long-run scale** (identical for examples 6 and 7): prompt=2048, resp=20480,
**128 prompts × 16 = 2048 sequences**, `gen_batch_size=384`, **lr warmup = 0**,
1000 steps. The two configs are **field-for-field identical** apart from
`policy.training_backend` and `megatron_cfg`, so any difference between the two lines can
only come from the training backend.

⚠️ **Mind the units**: `train_global_batch_size` counts **sequences** (2048) while
`gen_batch_size` counts **prompts** (384). The framework derives the prompt count as
`train_prompts = train_global_batch_size // num_generations`.

**Example 7's `megatron_cfg` (long run)**:

```yaml
use_distributed_optimizer: true
tensor_model_parallel_size: 1
pipeline_model_parallel_size: 1
context_parallel_size: 1
expert_model_parallel_size: 8       # 128 experts over 8 cards, 16 each
sequence_parallel: false
moe_grouped_gemm: true
moe_permute_fusion: true
moe_aux_loss_coeff: 0.0
moe_router_dtype: fp32              # pairs with LUMENRL_FP32_MOE_ROUTER=1
recompute_granularity: full         # required at resp=20480
recompute_method: uniform
recompute_num_layers: 1
log_probs_chunk_size: 1024
enable_dynamic_batch: true
max_tokens_per_gpu: 8192            # not 22528, see §13
```

**Why EP=8 and not something else**: `DP = 8 / (TP × PP × CP) = 8`, matching FSDP2's DP8,
so each rank still sees 2048/8 = 256 sequences. Anything that shrinks DP doubles the
distributed optimizer state per card (DP 8→4 costs about 8.5 GB more) and gives back what
was saved on activations — **CP=2 OOMs immediately**, dying earlier than CP=1.

---

## 10. Launching

Every `run_dapo.sh` switch is an environment variable; **the script itself never needs
editing**:

- `MODE` (default `bf16`): `bf16` / `fp8` / `atomfp8` / `atombf16`, selecting the config
  plus the rollout engine and precision.
- `TRAIN_FP8` (default `0`): `1` enables Lumen FP8 blockwise2d on the training side and
  sets `FP8_PARAM_MANAGER=0` automatically.
- `STEPS` (default `1000`): overrides `num_training_steps`.
- `CONFIG_OVERRIDE` (default: derived from `MODE`): names a config directly.
  **Required for smoke runs.**
- `EXTRA_OVERRIDE` (default empty): appends arbitrary Hydra overrides, space separated.
- `MODEL_PATH` / `TRAIN_FILE` / `VAL_FILE`: swap model or data; defaults follow the
  standard `$DATA_ROOT` layout.
- `LOG`: log path, default `$DATA_ROOT/logs/$RUN_ID.log`, also written to
  `/tmp/run_dapo_log.txt`.
- `LUMENRL_FP32_MOE_ROUTER` (default `1`): **must be passed explicitly for examples 6
  and 7**, see below.
- `PYTORCH_CUDA_ALLOC_CONF`: **set it to empty**; the ROCm/HIP allocator does not support
  `expandable_segments`.

⚠️ **The script is the single source of truth. If it gets modified by accident, restore
it**: `git -C "$RL_ROOT/Lumen-RL" checkout -- examples/DAPO/run_dapo.sh`.

All commands use this prefix. `export VAR=` means "set to empty", which the script reads
as a signal to `unset` it:

```bash
S=$RL_ROOT/Lumen-RL/examples/DAPO/run_dapo.sh
ENVX="export RL_ROOT='$RL_ROOT' DATA_ROOT='$DATA_ROOT' PYTORCH_CUDA_ALLOC_CONF=;"
```

⚠️ `run_dapo.sh` starts with `: "${RL_ROOT:?}"`, so an empty `RL_ROOT` inside the
container exits immediately. `ENVX` exists to avoid exactly that — a detached exec should
not rely on the `-e` injection from §4.

### Examples 1–5: Qwen3-8B-Base

Run the smoke first, in the foreground. ⚠️ **A smoke run must point `CONFIG_OVERRIDE` at
a `*_smoke.yaml`**; setting only `STEPS=1` still uses the long-run config
(resp=20480, batch=512), which is not a smoke run:

```bash
# Example 1
sudo docker exec "$CONTAINER" bash -lc "$ENVX \
  CONFIG_OVERRIDE=examples/DAPO/configs/dapo_qwen3_8b_ray_vllm_smoke.yaml \
  STEPS=1 MODE=bf16 LOG=$DATA_ROOT/logs/smoke-bf16.log bash '$S'; \
  tail -80 \"\$(cat /tmp/run_dapo_log.txt)\""

# Example 2 (TRAIN_FP8=0, verifying only the fp8_per_block rollout)
sudo docker exec "$CONTAINER" bash -lc "$ENVX \
  CONFIG_OVERRIDE=examples/DAPO/configs/dapo_qwen3_8b_ray_vllm_fp8_smoke.yaml \
  STEPS=1 MODE=fp8 LOG=$DATA_ROOT/logs/smoke-fp8.log bash '$S'; \
  tail -80 \"\$(cat /tmp/run_dapo_log.txt)\""

# Example 4 (4k config; finish the §7 precompile first)
sudo docker exec "$CONTAINER" bash -lc "$ENVX \
  CONFIG_OVERRIDE=examples/DAPO/configs/dapo_qwen3_8b_ray_atom_fp8_4k_smoke.yaml \
  STEPS=1 MODE=atomfp8 LOG=$DATA_ROOT/logs/smoke-atomfp8.log bash '$S'; \
  tail -80 \"\$(cat /tmp/run_dapo_log.txt)\""

# Example 5 (also needs the §7 precompile; does not need the §6 patch)
sudo docker exec "$CONTAINER" bash -lc "$ENVX \
  CONFIG_OVERRIDE=examples/DAPO/configs/dapo_qwen3_8b_ray_atom_bf16_4k_smoke.yaml \
  STEPS=1 MODE=atombf16 LOG=$DATA_ROOT/logs/smoke-atombf16.log bash '$S'; \
  tail -80 \"\$(cat /tmp/run_dapo_log.txt)\""
```

Then start the long run, detached so it survives disconnects:

```bash
sudo docker exec -d "$CONTAINER" bash -lc "$ENVX STEPS=1000 MODE=bf16                 bash '$S'"  # 1
sudo docker exec -d "$CONTAINER" bash -lc "$ENVX STEPS=1000 MODE=fp8                  bash '$S'"  # 2
sudo docker exec -d "$CONTAINER" bash -lc "$ENVX STEPS=1000 MODE=fp8      TRAIN_FP8=1 bash '$S'"  # 3
sudo docker exec -d "$CONTAINER" bash -lc "$ENVX STEPS=1000 MODE=atomfp8  TRAIN_FP8=1 bash '$S'"  # 4
sudo docker exec -d "$CONTAINER" bash -lc "$ENVX STEPS=1000 MODE=atombf16             bash '$S'"  # 5
```

> Do not pass `TRAIN_FP8=1` to example 5: `MODE=atombf16` unsets `LUMEN_FP8`,
> `FP8_PARAM_MANAGER`, `LUMEN_FP8_SCALING` and friends, so training is unconditionally
> BF16. It also **does not import the Lumen/AITER norm patch** (HF Qwen3's RMSNorm is
> already model-sensitive), which is what makes it strictly comparable to example 1.

> Consider starting with `STEPS=30` to confirm memory and metrics look healthy before
> committing to 1000 steps.
> W&B is optional: put `WANDB_API_KEY=xxxx` in `$RL_ROOT/wandb.key` and the script picks
> it up.
> To change checkpoint frequency use
> `EXTRA_OVERRIDE='checkpointing.save_steps=10 checkpointing.save_total_limit=2'`; one 8B
> FSDP2 checkpoint is about 90 GB, so check `df -h` first.

Confirm it is actually running:

```bash
sudo docker exec "$CONTAINER" bash -lc 'L=$(cat /tmp/run_dapo_log.txt); sleep 200
  grep -aE "setup .ray-controller. complete|filter_groups round|View run" "$L" | tail -3
  grep -aiE "Traceback|OutOfMemory|CUDA error" "$L" | tail'
```

### Example 6: MoE + FSDP2

```bash
ENVX_MOE="export RL_ROOT='$RL_ROOT' DATA_ROOT='$DATA_ROOT' SCRATCH_ROOT='$DATA_ROOT' \
LUMENRL_FP32_MOE_ROUTER=0 PYTORCH_CUDA_ALLOC_CONF=;"

# smoke (4k, 3 steps, ~10 min, of which ~5 min is 8 actors each loading the 57GB model)
sudo docker exec "$CONTAINER" bash -lc "$ENVX_MOE \
  CONFIG_OVERRIDE=examples/DAPO/configs/dapo_qwen3moe_a3b_ray_vllm_verlref_4k_smoke.yaml \
  MODEL_PATH=\$DATA_ROOT/models/Qwen3-30B-A3B-Base STEPS=3 MODE=bf16 \
  LOG=\$DATA_ROOT/logs/moe-4k-smoke.log bash '$S'; \
  tail -40 \"\$(cat /tmp/run_dapo_log.txt)\""

# long run (check the disk first: at least 400G free)
df -h "$DATA_ROOT"
sudo docker exec -d "$CONTAINER" bash -lc "$ENVX_MOE \
  CONFIG_OVERRIDE=examples/DAPO/configs/dapo_qwen3moe_a3b_ray_vllm_verlref_longrun.yaml \
  MODEL_PATH=\$DATA_ROOT/models/Qwen3-30B-A3B-Base STEPS=1000 MODE=bf16 \
  LOG=\$DATA_ROOT/logs/longrun-moe.log bash '$S'"
```

⚠️ **`MODEL_PATH` must be given explicitly** — `run_dapo.sh` defaults to the 8B model.

⚠️ **`LUMENRL_FP32_MOE_ROUTER=0` is mandatory here.** The framework defaults to fp32, but
this line wants the router in BF16: FSDP2 and vLLM run **the same PyTorch router op with
the same layout**, so BF16 rounding lands both sides on the same top-8 experts, and
agreeing with each other matters more than raising precision on one side. The log should
show `[lumenrl] MoE router patched on 48 gates (fp32=False)`; `True` means the variable
was forgotten.

⚠️ **`SCRATCH_ROOT` must be exported**: the config resolves `model_name` and
`checkpoint_dir` through `${oc.env:SCRATCH_ROOT}`, and omegaconf exits outright if it
cannot resolve. **This is required even when checkpointing is disabled.**

**On a new machine, verify weight sync end to end before the first real MoE run.** If
transformers 5.x's fused expert tensors (~57 GB, **93% of the parameters**) fail to match
vLLM's `expert_params_mapping`, vLLM takes a silent `continue` branch: no error, no load,
and the rollout engine's experts stay at their on-disk values forever. The coverage
assertion (`LUMENRL_WEIGHT_SYNC_CHECK=error`) is on by default; a bit-exact comparison on
top of it is safer still:

```bash
sudo docker exec "$CONTAINER" bash -lc "$ENVX_MOE LUMENRL_WEIGHT_SYNC_VERIFY=1 \
  CONFIG_OVERRIDE=examples/DAPO/configs/dapo_qwen3moe_a3b_ray_vllm_verlref_4k_smoke.yaml \
  MODEL_PATH=\$DATA_ROOT/models/Qwen3-30B-A3B-Base STEPS=3 MODE=bf16 \
  LOG=\$DATA_ROOT/logs/moe-verify.log bash '$S'"
```

> Passing means **no exception**: exit 0 says all 96 fused tensors × 8 replicas × 3 syncs
> matched bit for bit. Failure raises either
> `weight sync verify failed for ... shard w1/w3/w2` or
> `weight sync (colocate-ipc) left N/M rollout parameters untouched: ...`.

Also run the CPU-only unit tests to confirm the code is complete:

```bash
sudo docker exec "$CONTAINER" bash -lc 'cd "$RL_ROOT/Lumen-RL" &&
  python3 -m lumenrl.tests.test_moe_weight_sync &&      # 11 checks, fused expert sync
  python3 -m lumenrl.tests.test_rollout_routing &&      #  9 checks
  python3 -m lumenrl.tests.test_dataproto_ragged &&     # 10 checks
  python3 -m lumenrl.tests.test_mismatch_metrics'       #  4 checks
```

### Example 7: MoE + Megatron EP=8

```bash
# smoke: the config's moe_router_dtype is null, hence =0 here
sudo docker exec "$CONTAINER" bash -lc "$ENVX_MOE \
  CONFIG_OVERRIDE=examples/DAPO/configs/dapo_qwen3moe_a3b_ray_megatron_verlref_4k_smoke.yaml \
  MODEL_PATH=\$DATA_ROOT/models/Qwen3-30B-A3B-Base STEPS=3 MODE=bf16 \
  LOG=\$DATA_ROOT/logs/moe-4k-smoke-megatron.log bash '$S'; \
  tail -40 \"\$(cat /tmp/run_dapo_log.txt)\""

# long run: the config's moe_router_dtype is fp32, so this must flip to =1
ENVX_MEGA="export RL_ROOT='$RL_ROOT' DATA_ROOT='$DATA_ROOT' SCRATCH_ROOT='$DATA_ROOT' \
LUMENRL_FP32_MOE_ROUTER=1 PYTORCH_CUDA_ALLOC_CONF=;"

df -h "$DATA_ROOT"     # a Megatron dist-checkpoint is about 400GB
sudo docker exec -d "$CONTAINER" bash -lc "$ENVX_MEGA \
  CONFIG_OVERRIDE=examples/DAPO/configs/dapo_qwen3moe_a3b_ray_megatron_verlref_longrun.yaml \
  MODEL_PATH=\$DATA_ROOT/models/Qwen3-30B-A3B-Base STEPS=1000 MODE=bf16 \
  LOG=\$DATA_ROOT/logs/longrun-moe-megatron.log bash '$S'"
```

⚠️ **`LUMENRL_FP32_MOE_ROUTER` only affects the vLLM worker**; the Megatron training side
reads `megatron_cfg.moe_router_dtype`. **Both must be flipped together.**

**Why this line uses fp32 while example 6 uses BF16**: Megatron runs its own `TopKRouter`
feeding a grouped-GEMM, a different implementation from vLLM's. In BF16 the two pick
different experts on near-tied tokens, and flipping one expert moves that token's
log-prob a lot. Measured: with `moe_router_dtype: null`, `rollout_corr/kl` sat flat at
6.5e-4 through step 77 and then climbed about 16% per step to 2.4e-2 by step 110; with
fp32 it was still 7e-4 at step 185.

The long-run config's `save_steps: 5` is aggressive (9.3 min/step → a 400GB write roughly
every 46 minutes). Raise it to match your fault-tolerance needs:
`EXTRA_OVERRIDE='checkpointing.save_steps=20'`.

After launching, confirm three things before walking away:
`MoE+EP spec ... EP=8 ... router_dtype=fp32`, no `Traceback` / `HSA_STATUS`, and
`callbacks: step=1` within roughly 14 minutes.

### Disabling checkpoints (when disk is short)

```bash
EXTRA_OVERRIDE='checkpointing.save_steps=1000000 checkpointing.resume=false'
```

⚠️ **Do not write `checkpointing.checkpoint_dir=`.** Hydra parses the empty value as
`None` and omegaconf immediately fails with
`Incompatible value 'None' for field of type 'str'`. A `save_steps` large enough never to
be reached is the clean way. Think it through first — a crash then means starting over.

---

## 11. Health criteria

**Hard criteria for a passing smoke run**: exit 0, no `Traceback`, no `HSA_STATUS`, and
`RLTrainer.setup (ray-controller) complete: ... actor_workers=8` in the log.

Per-example `rollout_corr/kl` / memory / step time (at `resp=20480`) / checkpoint size:

- **Example 1**: kl ≈0.001, `mem/actor_allocated_gb` 11.6 GB, 4–5 min/step, ckpt ~90 GB.
  `grad_norm` ~0.85, `ppo_kl` ≈0.
- **Examples 2, 3**: kl **≈0.003–0.004** (the FP8 gap, expected; only worry as it
  approaches the TIS threshold of 2.0). Memory and step time as example 1, ckpt ~90 GB.
- **Example 4**: kl ≈0.004 (slightly above vLLM FP8), memory as above. no-eager level=3
  mainly speeds up rollout, but sleep/wake plus weight sync add fixed overhead.
  ckpt ~90 GB.
- **Example 5**: kl should land at example 1's magnitude (≈0.001), not example 4's
  ≈0.004 — quantization is off, so all that remains is the implementation difference
  between ATOM and the training side. **This is the check for whether ATOM is aligned
  correctly**: if ATOM BF16 also sits at 0.004, the gap is not from FP8, so go look at
  the ATOM RMSNorm alignment in §13. Memory, step time and ckpt match example 4.
- **Example 6**: kl ~1.5e-3, `mem/actor_max_reserved_gb` 75–115 GB, ~11 min/step,
  ckpt **~342 GB**. The `lr` at step 1 is already `9.99998e-07` (full value), confirming
  warmup is really 0; seeing `2e-07` means the wrong config is in use.
- **Example 7**: kl ~1.5e-3 (healthy band 6e-4 to 1.8e-3), allocated 72 GB (4k) /
  130 GB (20k), `max_reserved` 128–140 GB, ~9.3 min/step (first step ~14 min including
  vLLM load), ckpt **~400 GB**. The log should carry
  `MoE+EP spec: num_experts=128 topk=8 moe_ffn=768 | tp=1 pp=1 cp=1 EP=8 etp=1
  -> local_experts/rank=16 | grouped_gemm=True router_dtype=fp32 pre_softmax=False`.

Across all of them: `timing/weight_sync_s` stays at 1.1–1.7 s and **does not grow with
step count**; `mem/actor_allocated_gb` stays constant (`max_reserved` fluctuating with
each step's batch is normal — **live memory moving is what indicates a leak**).

**The single most important criterion: `rollout_corr/kl` must not climb monotonically
with step count.** Going down is normal (the policy converges and becomes more
deterministic, so divergence in log space shrinks). Going up has three causes, in order
of likelihood: MoE router precision mismatched between the two sides (§10), weight sync
missing parameters (recheck with `LUMENRL_WEIGHT_SYNC_VERIFY=1`), or a new alignment bug.

**Watch `seq/max_len` for length collapse.** Fluctuating near the budget ceiling is
healthy — it means some sequence hits the cap every step. Monotonically shrinking means
it has collapsed.

### Measured reference curves

**Example 6** (101 steps / 21.6 hours): `reward/accuracy` 0.136 → 0.494 (step 50) →
**0.581**. On AIME-2024 online validation (every 10 steps, greedy),
`val-core/acc/mean@1` rose from 0.041 at step 10 to **0.361** at step 90, and
`val/response_length_mean` from 2407 to 10389 — the model learned to think longer, which
is the evidence this line works.

**Example 7** (the `verlref_4k_longrun` compressed recipe, 91 steps):
`reward/accuracy` 0.168 → 0.42, `seq/mean_response_len` 773 → 925, `rollout_corr/kl`
0.00136 → 0.00060 (falling, which is correct), AIME `mean@1` 0.086 → 0.199.

⚠️ **The known entropy collapse is not a bug**: examples 6 and 7 are configured with
`entropy_coeff=0`, so monotonically falling entropy (0.844 → 0.094 over 101 steps) is a
direct consequence of that setting. Only worry when entropy drops below 0.05 **and**
length starts shrinking at the same time. Fixing it means adding `entropy_coeff` first.

---

## 12. Monitoring / stopping / resuming

```bash
# monitor
sudo docker exec "$CONTAINER" bash -lc 'L=$(cat /tmp/run_dapo_log.txt)
  grep -aE "callbacks: step=" "$L" | tail -5
  grep -aiE "Traceback|OutOfMemory|CUDA error|HSA_STATUS" "$L" | tail'

# stop (clearing the Ray actors too)
sudo docker exec "$CONTAINER" bash -lc '
  ray stop --force 2>/dev/null
  pkill -9 -f "[l]umenrl.trainer.main"; pkill -9 -f "[V]LLMRayServer"; pkill -9 -f "[E]ngineCore"
  sleep 10; rocm-smi --showmeminfo vram | grep -i used | head -1'   # expect ~298MB/card
```

**Resuming**: the configs set `resume: true`, so rerunning the same long-run command
picks up from the most recent checkpoint. On a new machine with an empty directory that
simply means starting at step 0.

---

## 13. Troubleshooting

**FP8 training diverges** (entropy ≈0.04 / `grad_norm` 1e4+ / `rollout_corr/kl` 1e4+):
essentially only two causes — `FP8_PARAM_MANAGER` was not set to 0 (it conflicts with
native FSDP2's fp32 master weights), or the §6 vLLM RMSNorm patch was never applied
(a new container needs it again).

**Falling back on memory (OOM)**:
- lower `policy.max_response_length=8192` + `max_total_sequence_length=9216` +
  `max_token_len_per_gpu=9216`;
- or lower `train_global_batch_size` / `gen_batch_size`;
- **do not** enable `fsdp_cfg.param_offload` / `optimizer_offload` on the Ray path; it
  fails with `parameters should be materialized on CPU`.

**Example 7's OOM is counterintuitive**: at `max_tokens_per_gpu: 22528` it dies in the
actor backward around step 14, but **the crash is not about peak allocated memory**
(~130 GB either way) — it is **fragmentation**. ROCm has no `expandable_segments`, and
roughly 7 bins per step each filled to 22.5k tokens repeatedly allocate and free huge
blocks, leaving reserved 42 GB above allocated. Capping bins at 8192 collapses the
fragmentation gap to 4–11 GB and drops peak reserved from 177 GB to 134 GB. **So that
8192 must not be casually raised back.**

**`weight sync (colocate-ipc) left N/M rollout parameters untouched`**: the sync missed
parameters. The exception lists the first 8 names; if they look like
`...experts.w13_weight` / `w2_weight`, the fused MoE routing did not take effect — either
the code is not up to date, or a vLLM/transformers upgrade invalidated the layout
assumption.

**ATOM rollout degradation** (with `MODE=atomfp8` / `atombf16`: `filter_groups: kept 0/96`
plus `Rollout reward: accuracy=0.0000` plus many `finished with reason max` and no `eos`
in the log): generation has broken down. Check first whether the plain RMSNorm in ATOM's
`atom/model_ops/layernorm.py` passes `use_model_sensitive_rmsnorm=1`; a misalignment
shows up first as an elevated `rollout_corr/kl` (~0.007 instead of ~0.004).
**Example 5 localizes this faster**: with quantization off, a still-elevated kl means the
problem is in the ATOM/training alignment rather than in FP8.

**Do not set `TORCHDYNAMO_DISABLE` by hand.** The script keeps it globally at `=1`
(dynamo off for training actors). The dynamo that examples 4 and 5 need for no-eager
level=3 rollout is injected by `ATOMReplicaManager` through `runtime_env` when it creates
the ATOM Ray actors, so `TORCHDYNAMO_DISABLE=0` applies **to the rollout processes only**.
A top-level `export TORCHDYNAMO_DISABLE=0` makes the training actors inherit it too,
which is pure side effect on the training side.

**Two more prerequisites for examples 4 and 5** (`MODE=atomfp8` / `atombf16` set both
automatically; do not drop them when overriding by hand):
`ATOM_ISOLATE_TORCH_COMPILE_CACHE=1` (otherwise 8 single-card replicas write the same
torch compile cache concurrently and trigger a `FileNotFoundError` in Inductor's
`write_atomic -> rename`), and `enable_sleep_mode=true` with `sleep_level=2` (releasing
KV cache / weights / CUDA graph after rollout, without which the training backward
easily hits `HSA_STATUS_ERROR_OUT_OF_RESOURCES`).

**Memory not released after a run finishes or is interrupted** (`rocm-smi` still shows
~90 GB per card while `ps` shows no trainer): `run_dapo.sh` cleans up processes **before**
starting, not afterwards, so ATOM EngineCore's `spawn_main` children (and their inductor
compile workers) become orphans still holding memory. When cleaning up manually, write
the patterns as `spawn[_]main` and so on, or `pkill -f` matches its own command line and
kills itself:

```bash
sudo docker exec "$CONTAINER" bash -lc '
  pkill -9 -f "compile_[w]orker"   || true
  pkill -9 -f "spawn[_]main"       || true
  pkill -9 -f "resource[_]tracker" || true
  sleep 8; rocm-smi --showmeminfo vram | grep -i used | head -3'
```

**Changing `DATA_ROOT` requires an unconditional assignment.** The
`docker run -e DATA_ROOT=...` in §4 baked the value into the container environment, so
`export DATA_ROOT="${DATA_ROOT:-/new/disk}"` in a wrapper **has no effect** (the variable
already exists) and checkpoints silently go back to the old disk. Write
`export DATA_ROOT=/new/disk` directly.

**`logger.info` inside the vLLM worker does not reach the driver log.** That is why the
`weight sync coverage` line is invisible on example 7. This is **not** evidence that the
assertion did not run — to know whether it fired, check whether it raised.
