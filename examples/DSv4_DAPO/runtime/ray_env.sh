#!/usr/bin/env bash
# Raylet / driver environment for DeepSeek-V4 DAPO on the megatron_native backend.
# Sourced INSIDE the training container, before `ray start` and before the driver.
#
# Ray does not propagate the driver's environment to actors: every actor inherits
# what its raylet was started with. That is why this file is sourced before
# `ray start` on every node, and why the AITER switch below must be set here --
# run_dapo.sh's MODE=bf16 branch sets VLLM_ROCM_USE_AITER=0 for the driver process
# only, and the vLLM replicas (Ray actors) keep the raylet's value of 1.
#
# Values are the ones the 4-layer run was validated with. See the README section
# "不要改的环境项" for why each of them is load-bearing.
: "${RL_ROOT:?RL_ROOT: directory holding Lumen-RL, aiter, Lumen, ATOM}"
: "${DATA_ROOT:?DATA_ROOT: directory holding data_cached/ and hf_home/}"
: "${DSV4_EXAMPLE_DIR:?DSV4_EXAMPLE_DIR: examples/DSv4_DAPO of this repo}"
: "${MEGATRON_DSV4_DIR:?MEGATRON_DSV4_DIR: the patched Megatron built by build_megatron.sh}"
: "${NET_IF:?NET_IF: network interface used between nodes}"

export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false TORCHDYNAMO_DISABLE=1 HYDRA_FULL_ERROR=1
export NCCL_TIMEOUT=7200 NCCL_CUMEM_ENABLE=0
# ROCm/HIP has no expandable_segments.
unset PYTORCH_CUDA_ALLOC_CONF
export HIP_FORCE_DEV_KERNARG=1 HSA_NO_SCRATCH_RECLAIM=1 HSA_DISABLE_FRAGMENT_ALLOCATOR=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export VLLM_USE_V1=1 VLLM_ENABLE_V1_MULTIPROCESSING=1 VLLM_LOGGING_LEVEL=WARN
export ATOM_DISABLE_VLLM_PLUGIN=1

# DSv4's sparse-attention indexer has only an AITER implementation on ROCm and
# hard-raises when AITER is off. The other three AITER paths stay off.
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_MHA=0
export VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=0 VLLM_ROCM_USE_AITER_LINEAR=0

export RAY_DEDUP_LOGS=0 RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export LUMEN_DISABLE_HF_ATTN_PATCH=1
export HF_HOME="$DATA_ROOT/hf_home" WANDB_DIR="$DATA_ROOT/wandb" LUMENRL_LOG_LEVEL=INFO
# The original FP8 checkpoint. Read only by the FSDP/verl path, not by this one;
# kept for parity with the validated environment.
export MODEL_NAME="${DSV4_SRC_MODEL_DIR:-}"

# Replaces the image's PYTHONPATH (it is assigned, not prepended). vime and
# megatron-core stay importable through the image's editable installs.
export PYTHONPATH="$RL_ROOT/Lumen-RL:$RL_ROOT/aiter:$RL_ROOT/Lumen:$RL_ROOT/ATOM"

export LUMENRL_FSDP_CHUNK_CAT_FALLBACK=1
export LUMENRL_FP32_MOE_ROUTER=0
# vLLM's loader drops names it does not recognise silently; keep the coverage check on.
export LUMENRL_WEIGHT_SYNC_VERIFY=1

export NCCL_SOCKET_IFNAME="$NET_IF"
export GLOO_SOCKET_IFNAME="$NET_IF"
# TCP between nodes; RDMA was never part of this recipe.
export NCCL_IB_DISABLE=1

# The DSv4-patched Megatron and the kernel-free model plugin. `megatron` is an
# implicit namespace package on both sides, so the patched tree merges with the
# installed megatron-core: megatron.core resolves here first.
export PYTHONPATH="$MEGATRON_DSV4_DIR:$DSV4_EXAMPLE_DIR/plugins/vendored:$PYTHONPATH"

# Megatron's deterministic mode (on for DSv4) asserts on NCCL_ALGO. Without
# determinism ~1.6% of argmaxes flip between identical forwards.
export NCCL_ALGO=Ring
