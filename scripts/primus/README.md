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

Every one of them fails somewhere far away from its cause, which is the only
reason they are worth writing down:

1. `HSA_DISABLE_FRAGMENT_ALLOCATOR=1` breaks intra-node reduce-scatter
2. `amdsmi` is absent, and adding it naively makes torch see 0 GPUs
3. `NVTE_FLASH_ATTN=0` is baked into the image
4. `import megatron.training` dies on `libz3.so.4.15`
5. The filtered parquet needs `datasets>=4`
6. `flydsl` must be 0.1.8 for the MoE path
7. `GPU_ARCHS=native` resolves to *nothing*, and the kernels build anyway
8. The `vime` package is imported lazily by the DSv4 router

**Full write-up — symptoms, root causes, what was tried and rejected, and how
to diagnose each one — is in
[`docs/agent/06-primus-pitfalls.md`](../../docs/agent/06-primus-pitfalls.md).**
It is indexed by the symptom you actually see, not by the root cause. Kept
there rather than here so there is one copy to maintain; `ray_env_primus.sh`
carries the fixes with a one-line reason next to each.

## One process discipline

**`docker restart <container>` before every run.** A failed run leaves hundreds
of orphaned ray/vLLM processes (822 processes / 660 orphans, measured). VRAM
returns to ~298 MB per card so `rocm-smi` looks perfectly clean, but new
processes then get `torch.cuda.device_count() == 0`. Three A/B comparisons in
this bring-up were run on a polluted container and produced **entirely wrong
conclusions**. Confirm with
`python3 -c "import torch;print(torch.cuda.device_count())"`, not `rocm-smi`.

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
