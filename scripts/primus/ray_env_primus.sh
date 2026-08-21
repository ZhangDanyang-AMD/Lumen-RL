#!/usr/bin/env bash
# Raylet environment for the primus/24.04 stack, with RDMA on.
#
# Why a fourth ray_env file instead of editing ray_env.sh: that file (and both
# dsv4 copies) hard-set NCCL_IB_DISABLE=1, and the comment there says why -- on
# the 22.04 images the ANP plugin could not be dlopen'd, so inter-node
# collectives had to fall back to TCP on ens3. On primus the plugin loads and
# RCCL is 2.28.9, so the fallback is not only unnecessary, it costs 26x. The
# 22.04 recipes are left untouched because they are still the validated ones for
# those images.
#
# Two other differences from ray_env.sh, both consequences of how the primus
# stack was assembled:
#   - vLLM/ray/megatron-core/Apex/TE live in one NFS tree on PYTHONPATH, not in
#     the image (rocm/primus ships none of them).
#   - MODEL_NAME is left to the caller; this file is shared by the Qwen3 and
#     DSv4 lines, which disagree about it.
: "${RL_ROOT:?}"; : "${DATA_ROOT:?}"

PRIMUS_SITE=${PRIMUS_SITE:-/home/xysheng/vllm_primus/site}

export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false TORCHDYNAMO_DISABLE=1 HYDRA_FULL_ERROR=1
export NCCL_TIMEOUT=7200 NCCL_CUMEM_ENABLE=0
# ROCm/HIP has no expandable_segments; run_dapo.sh unsets it when handed an
# empty value, and the raylet environment must match.
unset PYTORCH_CUDA_ALLOC_CONF
export HIP_FORCE_DEV_KERNARG=1 HSA_NO_SCRATCH_RECLAIM=1

# ⚠️ HSA_DISABLE_FRAGMENT_ALLOCATOR=1 is deliberately NOT set here, unlike
# ray_env.sh. On primus it breaks intra-node reduce-scatter, which is what
# Megatron's distributed optimizer reduces gradients with: the coalesced form
# (param_and_grad_buffer.start_grad_sync) hangs, and the plain form returns the
# right numbers but leaves the HIP context in an error state, so the next
# allocation -- typically clip_grads.py's one-element dummy_overflow_buf --
# raises "AcceleratorError: CUDA error: invalid argument" and the whole step
# dies in a frame that has nothing to do with the cause.
# Bisected with probe_rs_coalesced.py on job 32407: of the three knobs on the
# line above plus NCCL_CUMEM_ENABLE, only this one reproduces, and it does so
# on its own with everything else at defaults. all-to-all is unaffected, which
# is why the RDMA probe never caught it.
export CUDA_DEVICE_MAX_CONNECTIONS=1
export VLLM_USE_V1=1 VLLM_ENABLE_V1_MULTIPROCESSING=1 VLLM_LOGGING_LEVEL=WARN
export ATOM_DISABLE_VLLM_PLUGIN=1
export VLLM_ROCM_USE_AITER=${VLLM_ROCM_USE_AITER:-0} VLLM_ROCM_USE_AITER_MHA=0
export VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=0 VLLM_ROCM_USE_AITER_LINEAR=0
export RAY_DEDUP_LOGS=0 RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0

# ---- state the GPU count; Ray's AMD detection is not reliable here ----
# Importing lumenrl.workers.base_worker calls ray.init() before the controller
# gets to, so the driver is already attached to a LOCAL instance by the time
# cluster.ray_address is honoured -- the second call just logs "Calling ray.init()
# again after it has already been called" and the address is ignored. That local
# instance sizes itself from AMDGPUAcceleratorManager, which on this image
# intermittently reports 0 even though the same call returns 8 from a plain shell.
# When it does, the raylet comes up with no GPU in its resource list and all 8
# num_gpus=1 actors hang behind "No available node types can fulfill resource
# request {'GPU': 1.0}" -- while the driver still logs "1 nodes x 8 GPUs", which
# is read from the config and proves nothing.
export RAY_OVERRIDE_RESOURCES='{"GPU":8}'
export LUMEN_DISABLE_HF_ATTN_PATCH=1
export HF_HOME="$DATA_ROOT/hf_home" WANDB_DIR="$DATA_ROOT/wandb" LUMENRL_LOG_LEVEL=INFO

# ---- amdsmi, without which vLLM does not know it is on AMD ----
# vllm.platforms probes ROCm by importing amdsmi. primus ships the bindings only
# inside the ROCm SDK wheel's share/ directory, which is on no sys.path, so the
# probe raises ModuleNotFoundError, vllm silently falls back to
# UnspecifiedPlatform, and the rollout actor dies much later with the
# unrecognisable "RuntimeError: Device string must not be empty". Point at the
# SDK copy rather than installing one: libamd_smi.so sits next to it there.
AMDSMI=$(ls -d /opt/venv/lib/python3.12/site-packages/_rocm_sdk_devel/share/amd_smi 2>/dev/null | head -1)

# The LumenRL repos come first so they win over anything in the shared tree.
export PYTHONPATH="$RL_ROOT/Lumen-RL:$RL_ROOT/aiter:$RL_ROOT/Lumen:$RL_ROOT/ATOM:$PRIMUS_SITE${AMDSMI:+:$AMDSMI}"
export PATH="$PRIMUS_SITE/bin:$PATH"

# ---- aiter's JIT needs the arch spelled out; "native" resolves to nothing ----
# The primus image presets GPU_ARCHS=native, and aiter's cpp_itfs resolves
# "native" by shelling out to /opt/rocm/llvm/bin/amdgpu-arch -- which does not
# exist here, because primus has no /opt/rocm at all (torch and the ROCm SDK are
# under /opt/venv). The result is an empty string, so the build runs with a bare
# `--offload-arch=` and hipcc falls back to its own defaults: the object that
# came out held gfx906 and gfx1250 code and no gfx950.
# It fails at launch, not at build -- "CUDA error: invalid device function" out
# of vLLM's sampler (top_k_top_p_sampling_from_probs), ~20 s after aiter printed
# "finish build". ⚠️ Overriding "native" explicitly, not `${GPU_ARCHS:-...}`:
# the image already set it, so a :- default never applies.
if [ -z "${GPU_ARCHS:-}" ] || [ "${GPU_ARCHS}" = native ]; then
  export GPU_ARCHS=gfx950
fi

export LUMENRL_FSDP_CHUNK_CAT_FALLBACK=1
export LUMENRL_FP32_MOE_ROUTER=0
export LUMENRL_WEIGHT_SYNC_VERIFY=1

# ---- TE attention backend selection, which primus pre-empts ----
# The primus image bakes NVTE_FLASH_ATTN=0 / NVTE_FUSED_ATTN=1 into its
# Dockerfile ENV. Megatron builds its TransformerConfig with the default
# attention_backend=auto (LumenRL never overrides it), and auto asserts that all
# three NVTE_*_ATTN are unset or already 1 -- so the baked-in 0 aborts
# GPTModel.__init__ in every actor with "NVTE_FLASH_ATTN set to 0, but expected
# 1". The 22.04 images set none of these, so auto is also what the validated
# FSDP2/Megatron numbers were produced under; unsetting restores that.
unset NVTE_FLASH_ATTN NVTE_FUSED_ATTN NVTE_UNFUSED_ATTN

# ---- libz3, which nothing puts on the loader path by itself ----
# megatron.training -> megatron.core.ssm.mamba_mixer -> mamba_ssm -> tilelang ->
# tvm, and tvm's libtvm dlopens libz3.so.4.15. primus ships it, but only inside
# the z3_solver egg. Without this, `import megatron.training` dies with
# "libz3.so.4.15: cannot open shared object file" and nothing in the traceback
# mentions z3 until the very last frame.
Z3_LIB=$(ls -d /opt/venv/lib/python3.12/site-packages/z3_solver-*/z3/lib 2>/dev/null | head -1)

# ---- RDMA over the eight ionic RoCE HCAs (run_a2a_anp.sh, verified on job
# ---- 32407: 302 MB/rank at 0.008-0.013 s/iter, 39.4 GB/s peak, ens3 idle)
export LD_LIBRARY_PATH=${ANP_DIR:-/opt/anp}:/opt/openmpi/lib:${Z3_LIB:+$Z3_LIB:}${LD_LIBRARY_PATH:-}
export NCCL_NET_PLUGIN=anp
export NCCL_DMABUF_ENABLE=1
export NCCL_IB_GID_INDEX=1          # gid[0] is link-local fe80:: and just hangs
# Both of the next two are load-bearing: without either one it does not run at
# all, it is not merely slower. NCCL_IB_HCA=ionic keeps mlx5_0 out (its gid[1]
# is link-local, and it is the ens3 NIC anyway); NCCL_CROSS_NIC=0 keeps both
# ends of a channel on one rail, because this fabric does not route between
# rails.
export NCCL_IB_HCA=${NCCL_IB_HCA:-ionic}
export NCCL_CROSS_NIC=${NCCL_CROSS_NIC:-0}
export NCCL_IB_TC=96
export NCCL_IB_FIFO_TC=184
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_USE_INLINE=1
export NCCL_PXN_DISABLE=0
export NCCL_GDR_FLUSH_DISABLE=1
export NCCL_GDRCOPY_ENABLE=0
export NCCL_IGNORE_CPU_AFFINITY=1
export IONIC_LOCKFREE=all

# Bootstrap/OOB stays on TCP; only the collectives go over IB.
export NCCL_SOCKET_IFNAME=ens3
export GLOO_SOCKET_IFNAME=ens3
unset NCCL_IB_DISABLE
