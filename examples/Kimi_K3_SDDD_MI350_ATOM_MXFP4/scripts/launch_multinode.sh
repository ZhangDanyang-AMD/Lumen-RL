#!/usr/bin/env bash
# 起多节点 MXFP4 正式训练（N-1 teacher + 1 draft）。
# 正式训练从 seed 42 随机初始化，按 seed 42 正常 shuffle，并完整执行 57,849 steps。
set -uo pipefail
: "${ONNODE_ENV:?Set ONNODE_ENV to a shell file that defines onnode()}"
source "${ONNODE_ENV}"

# ---------------------------------------------------------------- 站点配置
# 全部可用环境变量覆盖。onnode() 需要你自己提供（契约："在第 N 台机器上跑一条
# 命令"）；我们用 spur，换 ssh/srun/kubectl exec 都可以。
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
# NODES[0] 是 Ray head / driver；Draft 由 K3_DRAFT_NODE 显式指定，默认使用
# 列表最后一个节点。实际位置仍以 `Draft node:` 日志为准。
: "${K3_NODES:?Set K3_NODES to the node identifiers, Ray head first}"
read -r -a NODES <<<"${K3_NODES}"
NODE_COUNT="${#NODES[@]}"
if (( NODE_COUNT < 2 )); then
  echo "K3_NODES must contain at least two node identifiers" >&2
  exit 2
fi
DRAFT_NODE="${K3_DRAFT_NODE:-${NODES[$((NODE_COUNT - 1))]}}"
DRAFT_NODE_RANK=-1
for ((rank = 0; rank < NODE_COUNT; rank++)); do
  if [[ "${NODES[$rank]}" == "$DRAFT_NODE" ]]; then
    (( DRAFT_NODE_RANK == -1 )) || {
      echo "K3_DRAFT_NODE appears more than once in K3_NODES: $DRAFT_NODE" >&2
      exit 2
    }
    DRAFT_NODE_RANK=$rank
  fi
done
(( DRAFT_NODE_RANK >= 0 )) || {
  echo "K3_DRAFT_NODE is not present in K3_NODES: $DRAFT_NODE" >&2
  exit 2
}
RUN_TAG="${RUN_TAG:-run}"
# Include seconds and the launcher PID so two starts in the same minute cannot
# reuse coordination state. An explicit RUN_ID still wins for reproducibility.
RUN_ID="${RUN_ID:-$RUN_TAG-$(date -u +%m%d%H%M%S)-$$}"
SHARED_ROOT="${SHARED_ROOT:-/path/to/shared_nfs/lumenrl}"
LOG_ROOT="${LOG_ROOT:-${SHARED_ROOT}/logs}"
LOG_PHASE="${LOG_PHASE:-${RUN_TAG}}"
TRAIN_IMAGE="${TRAIN_IMAGE:-kimi_k3_dspark_mxfp4_atom:latest}"
: "${DATA_ROOT:?Set DATA_ROOT to node-local storage holding the model}"
: "${MODEL:?Set MODEL to the Kimi-K3 model directory}"
: "${DATASET_PATH:?Set DATASET_PATH to the training data train.jsonl}"
: "${TOKEN_CACHE_DIR:?Set TOKEN_CACHE_DIR}"
: "${EVAL_FIXED_PATH:?Set EVAL_FIXED_PATH}"
: "${CKPT_DIR:?Set CKPT_DIR}"
RUN_PREFIX=kimi_k3_dspark_mxfp4_atom
LOGDIR="${LOG_ROOT}/${LOG_PHASE}/${RUN_PREFIX}_${RUN_ID}"
COORD="${SHARED_ROOT}/coord/${RUN_PREFIX}_${RUN_ID}"
log() { echo "[$(date -u '+%F %T')] $*"; }

OVR="${EXTRA_OVERRIDES:-}"
CONFIG_NAME=train_from_scratch.yaml
log "=== MXFP4 正式训练 ==="

log "执行 $NODE_COUNT 节点 host preflight"
pids=()
for n in "${NODES[@]}"; do
  onnode "$n" "
    cd $REPO &&
    TRAIN_IMAGE=$TRAIN_IMAGE DATA_ROOT=$DATA_ROOT MODEL=$MODEL \
    DATASET_PATH=$DATASET_PATH EVAL_FIXED_PATH=$EVAL_FIXED_PATH \
      bash examples/Kimi_K3_SDDD_MI350_ATOM_MXFP4/selfcheck/host_preflight.sh
  " &
  pids+=("$!")
done
preflight_failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || preflight_failed=1
done
(( preflight_failed == 0 )) || {
  log "host preflight 失败，停止启动"
  exit 2
}

printf -v Q_LOGDIR '%q' "$LOGDIR"
printf -v Q_COORD '%q' "$COORD"
if ! onnode "${NODES[0]}" "
  if [[ -e $Q_COORD || -e $Q_LOGDIR ]]; then
    echo 'run_id already has coordination or log state: $RUN_ID' >&2
    exit 3
  fi
  mkdir -p $Q_LOGDIR
"; then
  log "run_id=$RUN_ID 已存在或无法创建日志目录，停止启动"
  exit 2
fi
log "run_id=$RUN_ID  日志=$LOGDIR"

# 必须并行起：spur exec 留下后台进程时约 100 秒才返回，顺序启动会让各节点的
# 协调文件差好几分钟，可能超出 run_multinode_rank.sh 里 600 秒的发现窗口。
launch_pids=()
for ((rank = 0; rank < NODE_COUNT; rank++)); do
  n="${NODES[$rank]}"
  onnode "$n" "
    cd $REPO &&
    SLURM_PROCID=$rank SLURM_NTASKS=$NODE_COUNT LUMENRL_RUN_ID=$RUN_ID \
    LUMENRL_DRAFT_NODE_RANK=$DRAFT_NODE_RANK \
    CONFIG_NAME=$CONFIG_NAME \
    DATA_ROOT=$DATA_ROOT \
    SHARED_ROOT=$SHARED_ROOT \
    LOG_ROOT=$LOG_ROOT \
    LOG_DIR=$LOGDIR \
    MODEL_PATH=$MODEL \
    DATASET_PATH=${DATASET_PATH:?需要 DATASET_PATH} \
    CKPT_DIR=${CKPT_DIR:?需要 CKPT_DIR} \
    TOKEN_CACHE_DIR=${TOKEN_CACHE_DIR:?需要 TOKEN_CACHE_DIR} \
    EVAL_FIXED_PATH=${EVAL_FIXED_PATH:?需要 EVAL_FIXED_PATH} \
    LUMEN_MXFP4_AUTOTUNE=${LUMEN_MXFP4_AUTOTUNE:-0} \
    LUMEN_MXFP4_AUTOTUNE_CACHE=${LUMEN_MXFP4_AUTOTUNE_CACHE:-$SHARED_ROOT/cache/dspark_mxfp4_autotune_gfx950.json} \
    LUMEN_MXFP4_GEMM_SHAPE_LOG=${LUMEN_MXFP4_GEMM_SHAPE_LOG:-} \
    LUMEN_MXFP4_WGRAD_TILE_PAD=${LUMEN_MXFP4_WGRAD_TILE_PAD:-1} \
    LUMEN_MXFP4_WGRAD_POWER2_BUCKET=${LUMEN_MXFP4_WGRAD_POWER2_BUCKET:-1} \
    LUMEN_MXFP4_FUSED_TARGET_ROWS=${LUMEN_MXFP4_FUSED_TARGET_ROWS:-1} \
    LUMENRL_BALANCE_DISAGG_ROWS=${LUMENRL_BALANCE_DISAGG_ROWS:-1} \
    TRITON_CACHE_ROOT=${TRITON_CACHE_ROOT:-} \
    EXTRA_OVERRIDES='$OVR' \
    DOCKER_IMAGE=$TRAIN_IMAGE \
    setsid nohup bash examples/Kimi_K3_SDDD_MI350_ATOM_MXFP4/run_multinode_rank.sh \
      >> $LOGDIR/rank-$rank-$n.out 2>&1 < /dev/null &
    sleep 1" >/dev/null 2>&1 &
  launch_pids+=("$!")
done
launch_failed=0
for pid in "${launch_pids[@]}"; do
  wait "$pid" || launch_failed=1
done
(( launch_failed == 0 )) || {
  log "至少一个 rank 提交失败，通知已启动的 rank 退出"
  abort_written=0
  for n in "${NODES[@]}"; do
    if onnode "$n" "
      mkdir -p $Q_COORD &&
      printf '2\n' > $Q_COORD/exit-code
    " >/dev/null 2>&1; then
      abort_written=1
      break
    fi
  done
  (( abort_written == 1 )) || {
    log "无法写入共享退出信号；已启动的 rank 将在发现超时后自行退出"
  }
  exit 2
}
log "$NODE_COUNT 个 rank 都已发出"

# 只有 rank 0 确认全部 Ray 节点已加入后才写 cluster-ready。node/image/ray
# 文件只能说明启动过程走到某一步，不能作为集群可用的判据。
ready=0
for i in $(seq 1 40); do
  sleep 30
  if onnode "${NODES[0]}" "test -f $Q_COORD/cluster-ready" \
      >/dev/null 2>&1; then
    observed="$(
      onnode "${NODES[0]}" "cat $Q_COORD/cluster-ready" 2>/dev/null \
        | tr -dc '0-9'
    )"
    [[ "$observed" == "$NODE_COUNT" ]] || {
      log "cluster-ready 节点数异常：${observed:-empty}，预期 $NODE_COUNT"
      exit 1
    }
    ready=1
    log "$NODE_COUNT 个 Ray 节点已就绪"
    break
  fi
  if onnode "${NODES[0]}" "test -f $Q_COORD/exit-code" \
      >/dev/null 2>&1; then
    rc="$(
      onnode "${NODES[0]}" "cat $Q_COORD/exit-code" 2>/dev/null \
        | tr -dc '0-9'
    )"
    log "集群在就绪前退出，exit=${rc:-unknown}"
    exit 1
  fi
  log "等待 Ray 集群就绪（$i/40）"
done
(( ready == 1 )) || {
  log "等待 cluster-ready 超时，$NODE_COUNT 节点启动失败"
  exit 1
}
echo "$RUN_ID" > "${RID_FILE:-/tmp/train_rid}"
log "接下来看 $LOGDIR/node-00-*.log 里的 step 行"
