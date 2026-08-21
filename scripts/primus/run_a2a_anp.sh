#!/usr/bin/env bash
# run_a2a_anp.sh <node_rank> [-- probe args]
# The same all-to-all probe as run_a2a_ens3.sh, but over RDMA instead of TCP.
# Runs INSIDE the anp-primus container (see anp_container.sh).
#
# ⚠️ This deliberately does NOT source ray_env.sh: that file sets
# NCCL_IB_DISABLE=1, which is the very thing being removed here.
#
# The env below is the cluster's own /etc/profile.d/99-rccl-anp.sh verbatim, plus
# NCCL_SOCKET_IFNAME for the bootstrap. Keeping the OOB/rendezvous on TCP while
# the collectives go over IB is also what miles' own multi-node script does
# (scripts/run-glm4.5-355B-A32B.sh puts only GLOO/TP_SOCKET_IFNAME on the socket).
#
#   bash ~/4node/run_a2a_anp.sh 0 --seqs 6144 --iters 5
#   bash ~/4node/run_a2a_anp.sh 1 --seqs 6144 --iters 5
#
# Read three things in the output:
#   1. "NET/Plugin ... anp"      -- the plugin actually loaded (the whole point)
#   2. no "ibv_reg_mr failed"    -- 6.2 pitfall 4 is gone
#   3. ens3 totals near zero     -- the bytes moved on the ionic netdevs instead,
#                                   which is the positive proof it was RDMA
set -uo pipefail
# Per-allocation settings; see ray_start_primus.sh.
source "${LUMEN_CLUSTER_ENV:-/home/xysheng/4node/env.sh}"

NODE_RANK="${1:?usage: run_a2a_anp.sh <node_rank> [probe args...]}"
shift

# ANP_DIR is where the plugin (under all four names RCCL might look for) lives.
# On a 22.04 image that is /opt/anp-shim -- the copy anp_glibc_shim.sh patched so
# it no longer demands GLIBC_2.38 -- and LD_PRELOAD must carry the isoc23 shim.
# /opt/openmpi only exists when the host's OpenMPI is bind-mounted; images that
# ship their own libmpi.so.40 do not need it, and a missing entry here is inert.
export LD_LIBRARY_PATH=${ANP_DIR:-/opt/anp}:/opt/openmpi/lib:${LD_LIBRARY_PATH:-}

# ---- the cluster's ANP recipe (99-rccl-anp.sh) ----
export NCCL_NET_PLUGIN=anp
export NCCL_DMABUF_ENABLE=1
export NCCL_IB_GID_INDEX=1          # gid[0] is link-local fe80:: and just hangs

# ⚠️ Restrict to the ionic HCAs. Two reasons, and the first one is fatal without
# this: NCCL_IB_GID_INDEX=1 is the ROUTABLE fc01:: gid on ionic but the
# LINK-LOCAL fe80:: one on mlx5_0 (whose routable gid is index 3), so leaving
# mlx5_0 in the mix makes it try to reach fe80:: across a routed fabric and
# ibv_modify_qp INIT->RTR times out with 110. Second, mlx5_0 IS the ens3 NIC --
# its routable gid is ::ffff:<the ens3 IPv4> -- so using it for RDMA buys
# nothing over the TCP path it already carries.
# miles' own docs say the same thing: "RoCEv2: works, configure NCCL_IB_HCA to
# your physical NICs" (docs/platforms/nvidia.md).
export NCCL_IB_HCA=${NCCL_IB_HCA:-ionic}

# ⚠️ The second thing that is fatal without it. This is a RAIL-OPTIMIZED fabric:
# ionic_N sits on the fc01:N00::/64 rail on EVERY node, and there is no routing
# BETWEEN rails. An all-to-all pairs rank i with rank j, so if each rank picks the
# HCA nearest its own GPU the two ends land on different rails; the QPs come up,
# then every completion returns status=12 (IBV_WC_RETRY_EXC_ERR) vendor err 11 --
# packets sent, nothing ever acked. CROSS_NIC=0 forces both ends of a channel onto
# the same NIC index, i.e. onto one rail.
# Measured at 302 MB/rank, 2 nodes: 0.26 s/iter on TCP -> 0.030 s with one rail
# -> 0.010 s with all eight. Without this it does not run at all.
export NCCL_CROSS_NIC=${NCCL_CROSS_NIC:-0}
export NCCL_IB_TC=96
export NCCL_IB_FIFO_TC=184
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_USE_INLINE=1
export NCCL_PXN_DISABLE=0
export NCCL_GDR_FLUSH_DISABLE=1
export NCCL_GDRCOPY_ENABLE=0
export NCCL_IGNORE_CPU_AFFINITY=1
export HSA_NO_SCRATCH_RECLAIM=1
export IONIC_LOCKFREE=all

# Bootstrap/OOB only. NCCL_IB_DISABLE is deliberately NOT set.
export NCCL_SOCKET_IFNAME=ens3
export GLOO_SOCKET_IFNAME=ens3
unset NCCL_IB_DISABLE

export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
export NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-INIT,NET}

NNODES=$(echo "$NODES" | wc -w)
mkdir -p "$LOG4N"

echo "=== a2a/ANP node_rank=$NODE_RANK/$NNODES on $(hostname) at $(date -Is) ==="
echo "    NCCL_NET_PLUGIN=$NCCL_NET_PLUGIN  NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-unset}"

# Byte counters for ens3 AND every ionic netdev. ens3 should stay quiet and the
# ionic ones should carry the traffic; that pair of facts is the verdict.
declare -A T0 R0
IFACES=(ens3)
for d in /sys/class/infiniband/ionic_*; do
  nd=$(ls "$d/device/net" 2>/dev/null | head -1); [ -n "$nd" ] && IFACES+=("$nd")
done
for i in "${IFACES[@]}"; do
  T0[$i]=$(cat "/sys/class/net/$i/statistics/tx_bytes" 2>/dev/null || echo 0)
  R0[$i]=$(cat "/sys/class/net/$i/statistics/rx_bytes" 2>/dev/null || echo 0)
done

python3 -m torch.distributed.run --nnodes="$NNODES" --node_rank="$NODE_RANK" \
  --master_addr="$HEAD_IP" --master_port="${PORT:-29550}" \
  --nproc_per_node="${NPROC:-8}" "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/probes/probe_a2a_ens3.py" "$@"
rc=$?

echo "=== per-interface bytes for the whole run ==="
for i in "${IFACES[@]}"; do
  t1=$(cat "/sys/class/net/$i/statistics/tx_bytes" 2>/dev/null || echo 0)
  r1=$(cat "/sys/class/net/$i/statistics/rx_bytes" 2>/dev/null || echo 0)
  awk -v n="$i" -v a="${T0[$i]}" -v b="$t1" -v c="${R0[$i]}" -v d="$r1" \
    'BEGIN {printf "  %-12s tx %8.2f GiB   rx %8.2f GiB\n", n, (b-a)/1073741824, (d-c)/1073741824}'
done
echo "=== a2a/ANP node_rank=$NODE_RANK exit=$rc ==="
exit $rc
