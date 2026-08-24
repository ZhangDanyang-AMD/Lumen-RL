#!/usr/bin/env bash
# Standalone RCCL all-reduce smoke test. Run this directly on the login node;
# this cluster's nested `sbatch -> srun` path does not reliably retire steps.
#
# Single node / 8 GPUs:
#   bash examples/Kimi_K3_SDDD_MI350_ATOM/run_rccl_smoke_slurm.sh
#
# Five nodes / 40 GPUs:
#   NODES=5 NODELIST='crsuse2-m2m-v2-[030,035,037-039]' \
#     bash examples/Kimi_K3_SDDD_MI350_ATOM/run_rccl_smoke_slurm.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SOURCE="${RCCL_SMOKE_SOURCE:-${SCRIPT_DIR}/rccl_allreduce_smoke.cpp}"
SHARED_BIN="${RCCL_SMOKE_BIN:-${REPO_ROOT}/.rccl_allreduce_smoke}"
UNIQUE_ID_FILE="${REPO_ROOT}/.rccl-smoke-${USER}.id"
HIPCC="${HIPCC:-/opt/rocm/bin/hipcc}"
PARTITION="${PARTITION:-default}"
NODES="${NODES:-1}"
NODELIST="${NODELIST:-crsuse2-m2m-v2-035}"
COMPILE_NODE="${COMPILE_NODE:-crsuse2-m2m-v2-035}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
ELEMENTS="${ELEMENTS:-16777216}" # 64 MiB of float32 per GPU
ITERATIONS="${ITERATIONS:-10}"

# MI355X GPU n is directly attached to ionic_n. GID 0 is link-local; GID 1 is
# the routable RoCE v2 fabric address. dmabuf is required for GPU memory
# registration on these AINIC nodes.
export NCCL_IB_HCA="${NCCL_IB_HCA:-ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7}"
export NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-1}"
export NCCL_DMABUF_ENABLE="${NCCL_DMABUF_ENABLE:-1}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-spur0}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export HSA_NO_SCRATCH_RECLAIM="${HSA_NO_SCRATCH_RECLAIM:-1}"

if [[ ! -x "${SHARED_BIN}" || "${SOURCE}" -nt "${SHARED_BIN}" ]]; then
    srun --partition="${PARTITION}" --nodes=1 --ntasks=1 \
        --nodelist="${COMPILE_NODE}" \
        "${HIPCC}" "${SOURCE}" -O2 -std=c++17 -o "${SHARED_BIN}" -lrccl
fi

rm -f "${UNIQUE_ID_FILE}" "${UNIQUE_ID_FILE}.tmp"
trap 'rm -f "${UNIQUE_ID_FILE}" "${UNIQUE_ID_FILE}.tmp"' EXIT

echo "nodes=${NODES} gpus_per_node=${GPUS_PER_NODE} nodelist=${NODELIST}"
echo "NCCL_IB_HCA=${NCCL_IB_HCA} GID=${NCCL_IB_GID_INDEX} dmabuf=${NCCL_DMABUF_ENABLE}"

srun --partition="${PARTITION}" --nodes="${NODES}" --ntasks="${NODES}" \
    --ntasks-per-node=1 --gres=gpu:8 --exclusive --mpi=none \
    --nodelist="${NODELIST}" \
    "${SHARED_BIN}" "${UNIQUE_ID_FILE}" \
    "${GPUS_PER_NODE}" "${ELEMENTS}" "${ITERATIONS}"
