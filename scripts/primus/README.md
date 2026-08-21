# Running LumenRL on `rocm/primus:v26.4` (Ubuntu 24.04 + RDMA)

Bring-up scripts and probes for the MI350X/gfx950 cluster whose inter-node
fabric is eight ionic RoCE HCAs. Validated 2026-08-20 on job 32407
(`crsuse2-m2m-031` / `-057`): RDMA baseline, vLLM + ray + Megatron installed,
Qwen3-8B DAPO smoke `exit 0`, and the DeepSeek-V4 4-layer-slice probes 67/68/70
all passing.

## Why this base image

`rocm/primus:v26.4` is the **only** base on which RDMA has been measured
working, and the constraint is narrower than it looks: the cluster's ANP plugin
segfaults on RCCL 2.26.6 and 2.27.7 and only runs on **2.28.9**. A clean
control settled it — `rocm/primus:v26.3` is also 24.04, the plugin loads there
unmodified, and changing only the RCCL version still SIGSEGVs all 16 ranks.
**When judging a candidate image, read `strings librccl.so | grep "RCCL version"`
and ignore the tag name.**

The cost is that primus ships **neither vLLM nor ray**, so everything below is
installed into one NFS tree that every node mounts:

```
/home/xysheng/vllm_primus/site        # override with PRIMUS_SITE
  vLLM 0.26.0+rocm714 (built from source), ray 2.57.0, megatron-core 0.18.2,
  apex 1.14.0a0 (28 compiled extensions), transformer_engine 2.15.0.dev0+6e541a10,
  datasets 4.0.0, flydsl 0.1.8, sitecustomize.py
```

## Usage

```bash
# once per node
bash rl24_container.sh                              # RL24_IMAGE / RL24_CONTAINER

# inside the container, before anything else
export RL_ROOT=... DATA_ROOT=...
source ray_env_primus.sh                            # Qwen3 / general
source ray_env_dsv4_primus.sh                       # DeepSeek-V4 (adds AITER, patched Megatron, vime)

# multi-node only; a single-node run lets the driver start its own Ray
bash ray_start_primus.sh head|worker                # LUMEN_CLUSTER_ENV=<your env.sh>
```

`ray_env_primus.sh` is the entry point that matters: **every pitfall below that
has an environment-level fix is already applied there**, each with the reason
next to it. The 22.04 files it replaces (`ray_env.sh`, `ray_env_dsv4*.sh`) are
deliberately left alone — they are still the validated recipe for those images.

Per-allocation settings (`JOBID` / `HEAD_NODE` / `HEAD_IP` / `NODES` / `NET_IF`)
live outside the repo; point `LUMEN_CLUSTER_ENV` at your copy.

## The eight things that bite

Every one of these fails somewhere far away from its cause, which is the only
reason they are worth writing down.

### 1. `HSA_DISABLE_FRAGMENT_ALLOCATOR=1` breaks intra-node reduce-scatter

The single most expensive one. It is a line inherited from the 22.04 recipe. On
ROCm 7.14 / RCCL 2.28.9 / torch 2.12 the coalesced reduce-scatter that Megatron's
distributed optimizer uses (`param_and_grad_buffer.start_grad_sync`) **hangs**,
and the plain form returns correct numbers while leaving the HIP context in an
error state — so the step dies later in an unrelated one-element allocation:

```
clip_grads.py:109  dummy_overflow_buf = torch.zeros(1, dtype=torch.int, device='cuda')
torch.AcceleratorError: CUDA error: invalid argument
```

Sometimes the actor just SIGSEGVs instead and Ray reports only
`ActorDiedError ... SYSTEM_ERROR`; the real frame is in
`/tmp/ray/session_*/logs/worker-*.err`.

Bisected one variable at a time with `probes/probe_rs_coalesced.py` (8 ranks, no
Megatron, Ray or vLLM needed): `HSA_NO_SCRATCH_RECLAIM`, `HIP_FORCE_DEV_KERNARG`,
`CUDA_DEVICE_MAX_CONNECTIONS`, `NCCL_CUMEM_ENABLE` and the whole ANP/RDMA group
are all **innocent**; this one reproduces on its own. **all-to-all is
unaffected**, which is why the RDMA probe never catches it.

⚠️ `examples/DAPO/run_dapo.sh` exports its own copy, so clearing it in the
environment is not enough — that file now takes `HSA_DISABLE_FRAGMENT_ALLOCATOR=`
(explicit empty) the same way it already took `PYTORCH_CUDA_ALLOC_CONF=`.

### 2. `amdsmi` is absent, and adding it naively makes torch see 0 GPUs

Without it, `vllm.platforms`' ROCm probe raises `ModuleNotFoundError`, vLLM
**silently** becomes `UnspecifiedPlatform`, and the rollout actor dies with
`RuntimeError: Device string must not be empty`.

Putting the SDK's `share/amd_smi` on `PYTHONPATH` fixes that and breaks
something worse: `torch.cuda.device_count()` prefers amdsmi over HIP on ROCm,
and torch guards its own `import amdsmi` with a ctypes hook that redirects
`libamd_smi.so` to whichever copy the loader finds first (its workaround for
[ROCm/amdsmi#72](https://github.com/ROCm/amdsmi/issues/72)). Here that redirect
makes `amdsmi_get_processor_handles()` return an empty list, with no warning:

```
import amdsmi; import torch   -> device_count 8, RocmPlatform
import torch;  import amdsmi  -> device_count 0, UnspecifiedPlatform
```

Downstream symptoms, none of which mention amdsmi: Ray's raylet comes up with no
`GPU` in `--static_resource_list` and all 8 `num_gpus=1` actors hang on
*"No available node types can fulfill resource request {'GPU': 1.0}"*; torchrun
children divide by zero on `rank % torch.cuda.device_count()`.

⚠️ The driver still logs `Ray cluster initialized: 1 nodes x 8 GPUs` — that is
read from the config and proves nothing. **Check `ray status` for `0.0/8.0 GPU`.**

Fix: `sitecustomize.py` (copy it into `$PRIMUS_SITE`) imports amdsmi before torch
can. `sitecustomize` is the only hook that runs early enough — `.pth` files are
not processed for plain `PYTHONPATH` entries.

Tried and rejected: `ROCM_PATH` pointing at the SDK (torch's hook prefers
`$ROCM_PATH/lib/libamd_smi.so`, which is the broken copy), putting the bindings'
own lib dir on `LD_LIBRARY_PATH` (breaks vLLM too), and `ray start --num-gpus=8`
(Ray overrides it). `RAY_OVERRIDE_RESOURCES='{"GPU":8}'` fixes the Ray half only
and is kept as a belt-and-braces setting.

### 3. `NVTE_FLASH_ATTN=0` is baked into the image

Megatron builds its `TransformerConfig` with the default
`attention_backend=auto`, which asserts all three `NVTE_*_ATTN` are unset or
already 1, so every actor dies in `GPTModel.__init__`. Fix: unset all three.
⚠️ Do not "correct" this to `fused` — the 22.04 images set none of these
variables, so `auto` is what the validated numbers were produced under.

### 4. `import megatron.training` dies on `libz3.so.4.15`

Via `megatron.core.ssm.mamba_mixer` → `mamba_ssm` → `tilelang` → `tvm`. primus
ships libz3 only inside the `z3_solver` egg. z3 appears in the very last frame
only, and `megatron.core` imports fine, so it reads like a broken megatron-core.

### 5. The filtered parquet needs `datasets>=4`

`ValueError: Feature type 'List' not found`; the image has 3.6.0. ⚠️ Do **not**
install `datasets>=5` — it drags numpy 2.5.2, pandas 3.0.5, pyarrow 25 and
huggingface_hub 1.28 into the tree. 4.0.0 runs on the existing dependencies.

### 6. `flydsl` must be 0.1.8 for the MoE path

`ImportError: cannot import name 'extract_to_ir_values' from
'flydsl.compiler.protocol'`, raised from aiter's flydsl GEMM kernels. The image
has 0.1.6.

### 7. `GPU_ARCHS=native` resolves to *nothing*, and the kernels build anyway

The image presets `GPU_ARCHS=native`; aiter's `csrc/cpp_itfs/utils.py` resolves
"native" by shelling out to `/opt/rocm/llvm/bin/amdgpu-arch`, and **primus has no
`/opt/rocm` at all** (torch and the ROCm SDK live under `/opt/venv`). The result
is an empty string, the build runs with a bare `--offload-arch=`, and hipcc falls
back to its own defaults — the object that came out held gfx906 and gfx1250 code
and no gfx950.

It fails at launch, not at build: aiter prints `finish build ... cost 19.1s` and
~20 s later vLLM's sampler raises `CUDA error: invalid device function`.

⚠️ `${GPU_ARCHS:-gfx950}` does **not** work, because the image already set it;
`ray_env_primus.sh` rewrites the value `native` explicitly. After changing it,
`rm -rf /root/.aiter/build/<op>_*` — `docker restart` does not clear that cache.
Diagnosis: `strings <build>/lib.so | grep -oE "gfx[0-9]+" | sort -u`.

### 8. The `vime` package is imported lazily by the DSv4 router

`from vime.utils.routing_replay import register_routing_replay` happens inside
`routing()`, so nothing fails until the first MoE layer runs a forward, and the
traceback bottoms out in `moe_layer.route` — it looks like a problem with the
patched Megatron. On the 22.04 line the vime image supplied the package.
`ray_env_dsv4_primus.sh` adds `VIME_SRC` to `PYTHONPATH`.

## One process discipline

**`docker restart <container>` before every run.** A failed run leaves hundreds
of orphaned ray/vLLM processes (822 processes / 660 orphans, measured). VRAM
returns to ~298 MB per card so `rocm-smi` looks perfectly clean, but new
processes then get `torch.cuda.device_count() == 0`. Three A/B comparisons in
this bring-up were run on a polluted container and produced **entirely wrong
conclusions**, which cost a full round on pitfall 1. Confirm with
`python3 -c "import torch;print(torch.cuda.device_count())"`, not with `rocm-smi`.

## Probes

| Probe | Question |
|---|---|
| `probes/probe_a2a_ens3.py` (via `run_a2a_anp.sh`) | Is RDMA up? 302 MB/rank all-to-all, 8 ionic HCAs, `ens3` idle |
| `probes/probe_rs_coalesced.py` | Does coalesced reduce-scatter survive? (pitfall 1) |
| `probes/probe_devcount.py` | What does each torchrun rank think `device_count()` is? (pitfall 2) |
| `probes/probe_ipc_primus.py` | Can one process open another's CUDA IPC handle? (the weight-sync path's foundation) |
| `verify_vllm_primus.py` | 5 compiled extensions + ops actually registered + DSv4 in the registry + RCCL still 2.28.9 |

⚠️ Two knobs are load-bearing for RDMA and their absence is fatal rather than
slow: `NCCL_IB_HCA=ionic` (mlx5_0's `gid[1]` is link-local, and it is the ens3
NIC anyway) and `NCCL_CROSS_NIC=0` (this is a rail-optimized fabric with no
routing between rails). Both are defaults in `ray_env_primus.sh`.
