#!/usr/bin/env bash
# 起五节点训练（4 teacher + 1 draft）。
#
# 两种模式：
#   VERIFY=1  ——  lr=0 跑几步，核对固定 eval 尺子的读数和上一轮记录逐位一致。
#                 一次验完三条链路：checkpoint 权重导入、teacher hidden state、
#                 eval 切片对齐，十几分钟。挡住过整轮白跑。
#                 /!\ 从零训没有"上一轮记录"可比，这个模式只对续训有意义。
#   默认      ——  正式训练。
set -uo pipefail
source "${ONNODE_ENV:-/home/jimguo12/k3_regen/scripts/env.sh}"

# ---------------------------------------------------------------- 站点配置
# 全部可用环境变量覆盖。onnode() 需要你自己提供（契约："在第 N 台机器上跑一条
# 命令"）；我们用 spur，换 ssh/srun/kubectl exec 都可以。
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
# /!\ NODES[0] 是 **Ray head / driver**，不是 draft 节点。draft 由 Ray 自己放置：
# gen-6 五轮下来 driver 一直在 039，而 torchrun 的 8 个本地 rank（也就是 draft）
# 落在 040，四个 teacher actor 在 039/041/042/043。看 draft 日志要看
# logs/<run>/node-00-crsuse2-m2m-v2-040.log，不是 rank-0-039.out。
NODES=(039 040 041 042 043)
RUN_TAG="${RUN_TAG:-run}"
RUN_ID="${RUN_ID:-$RUN_TAG-$(date -u +%m%d%H%M)}"
SHARED_ROOT="${SHARED_ROOT:-/shared_nfs/jimguo12/lumenrl}"
RUN_PREFIX="${RUN_PREFIX:-kimi_k3_dspark_atom}"
LOGDIR=$SHARED_ROOT/logs/${RUN_PREFIX}_$RUN_ID
log() { echo "[$(date -u '+%F %T')] $*"; }

OVR=""
if [[ "${VERIFY:-0}" == "1" ]]; then
  # lr=0：权重不动，只看 eval 读数。步数给到能跑出第一个 eval 点。
  OVR="policy.learning_rate=0.0 policy.min_lr=0.0 policy.warmup_ratio=0.0 \
       eval.interval=2 num_training_steps=${VERIFY_STEPS:-4} \
       checkpointing.save_steps=100000"
  log "=== 校验模式：lr=0，只看 eval 读数 ==="
else
  log "=== 正式训练 ==="
fi

# /!\ Mooncake 段大小必须从环境变量给。launcher 会用
# mooncake.global_segment_size=${MOONCAKE_GLOBAL_SEGMENT_SIZE:-128GB} 作为
# runtime override，**覆盖掉 YAML 里的值** —— 只改 YAML 是不生效的。
export MOONCAKE_GLOBAL_SEGMENT_SIZE=256GB
export LUMENRL_TEACHER_MOONCAKE_SEGMENT_SIZE=256GB

onnode "${NODES[0]}" "mkdir -p $LOGDIR" >/dev/null 2>&1
log "run_id=$RUN_ID  日志=$LOGDIR"

# 必须并行起，而且发现窗口要放宽：spur exec 留下后台进程时约 100 秒才返回，
# 顺序起五个会让第一个和最后一个节点的协调文件差好几分钟，而默认窗口只有 120 秒。
for rank in 0 1 2 3 4; do
  n="${NODES[$rank]}"
  onnode "$n" "
    cd $REPO &&
    SLURM_PROCID=$rank SLURM_NTASKS=5 LUMENRL_RUN_ID=$RUN_ID \
    CONFIG_NAME=${CONFIG_NAME:-train_from_scratch.yaml} \
    DATA_ROOT=${DATA_ROOT:-/mnt/m2m_nobackup/jimguo12} \
    SHARED_ROOT=$SHARED_ROOT \
    MODEL_PATH=$MODEL \
    DATASET_PATH=${DATASET_PATH:?需要 DATASET_PATH} \
    CKPT_DIR=${CKPT_DIR:?需要 CKPT_DIR} \
    TOKEN_CACHE_DIR=${TOKEN_CACHE_DIR:?需要 TOKEN_CACHE_DIR} \
    MOONCAKE_GLOBAL_SEGMENT_SIZE=256GB \
    LUMENRL_TEACHER_MOONCAKE_SEGMENT_SIZE=256GB \
    LUMENRL_NODE_DISCOVERY_TIMEOUT=600 \
    EXTRA_OVERRIDES='$OVR' \
    DOCKER_IMAGE=${TRAIN_IMAGE:-kimi_k3_dspark_atom:pinned} \
    setsid nohup bash examples/Kimi_K3_SDDD_MI350_ATOM/run_multinode_rank.sh \
      > $LOGDIR/rank-$rank-$n.out 2>&1 < /dev/null &
    sleep 1" >/dev/null 2>&1 &
done
wait
log "五个 rank 都已发出"

# /!\ 判断启动成功要看协调目录里的 node-*/ray-* 文件，不要看 stdout：
# spur exec 留下后台进程时返回很慢，实测 5 个 rank 全起来了但 stdout 只打了两行
# 就被外层 timeout 掐掉。
COORD="$SHARED_ROOT/coord/${RUN_PREFIX}_$RUN_ID"
for i in $(seq 1 40); do
  sleep 30
  c=$(onnode "${NODES[0]}" "ls $COORD 2>/dev/null | wc -l" 2>/dev/null | tr -dc '0-9')
  log "协调文件 ${c:-0}/10（node-0..4 + ray-0..4）"
  [[ "${c:-0}" -ge 10 ]] && { log "五节点已就绪"; break; }
done
echo "$RUN_ID" > "${RID_FILE:-/tmp/train_rid}"
log "接下来看 $LOGDIR/node-00-*.log 里的 step 行（draft 由 Ray 放置，见下面注释）"
