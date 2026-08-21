#!/usr/bin/env bash
# DeepSeek-V4 + megatron_native environment on primus/24.04.
#
# Stands to ray_env_dsv4_megatron.sh as ray_env_primus.sh stands to ray_env.sh:
# same DSv4-specific additions, but on top of the primus base instead of the
# 22.04 one. The 22.04 files are left alone -- they are still the validated
# recipe for those images.
#
# Two of the differences are not optional on primus:
#   - RDMA is on (ray_env_dsv4.sh sets NCCL_IB_DISABLE=1, worth 26x here).
#   - HSA_DISABLE_FRAGMENT_ALLOCATOR is NOT set. On this stack it breaks
#     intra-node reduce-scatter, which is exactly what probe_70's 8-rank
#     Megatron forward and any distributed optimizer step depend on
#     (primus-24.04-rdma-dsv4-handoff.md 9.9).
: "${RL_ROOT:?}"; : "${DATA_ROOT:?}"

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/ray_env_primus.sh"

# ---- 1. DSv4's rollout needs AITER ----
# The sparse attention indexer has only an AITER implementation on ROCm and
# hard-raises when it is off. The other three AITER paths stay off, matching
# deepseek-v4-base-rl-train-handoff.md 3.1.
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_MHA=0
export VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=0 VLLM_ROCM_USE_AITER_LINEAR=0

# ---- 2. the DSv4-patched Megatron and the tilelang-free plugin ----
# Neither is vendored into any repo yet. ``megatron`` is an implicit namespace
# package on both sides, so these merge with the installed megatron-core rather
# than shadowing it: megatron.core resolves here, megatron.bridge still resolves
# from the shared tree.
DSV4_PROBE=/home/xysheng/dsv4/mhc_probe
# ⚠️ VIME_SRC is the third entry and it is easy to miss. The DSv4 router does a
# late `from vime.utils.routing_replay import register_routing_replay` inside
# routing(), so nothing fails until the first MoE layer runs a forward -- on the
# 22.04 line the vime image supplied that package in site-packages, and primus
# does not. Symptom: probe_70 dies with ModuleNotFoundError: No module named
# 'vime' deep inside moe_layer.route.
VIME_SRC=${VIME_SRC:-/home/xysheng/working/vime-rl/vime}
export PYTHONPATH="$DSV4_PROBE/megatron_dsv4:$DSV4_PROBE/vendored:$VIME_SRC:${PYTHONPATH:-}"

# ---- 3. deterministic training ----
# The engine turns on Megatron's deterministic mode for DSv4 (without it ~1.6%
# of argmaxes flip between identical forwards, which is larger than the
# train/rollout gap DAPO measures). That mode asserts on NCCL_ALGO.
export NCCL_ALGO=Ring

# ---- 4. diagnostics that are still earning their keep ----
# What has never run at 43 layers is the rest of _sync_weights_ipc -- the
# ZMQ/CUDA-IPC transfer and vLLM's receiving end. Turn MEM_DIAG off once a
# 4-node EP=32 step has been through those.
export LUMENRL_MEM_DIAG=1
export LUMENRL_WEIGHT_SYNC_VERIFY=1

# Node-local, not $DATA_ROOT: the checkpoint is 275 GB and /home is a shared
# volume that was already 97% full on job 32407.
export MODEL_NAME=${MODEL_NAME:-/mnt/m2m_nobackup/xysheng/models/DeepSeek-V4-Flash-Base}
