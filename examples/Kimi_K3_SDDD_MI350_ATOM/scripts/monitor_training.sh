#!/usr/bin/env bash
# gen-6 训练监控。政策：**诊断优先，不盲目重启。**
#
# 前一版是"掉了就自动重拉"，结果连拉三次、每次都被同一台机器（043 的 teacher
# NCCL 自旋死锁）在同一个位置打断，step 在 41,5xx 原地打转、白烧一个半小时。
# 盲目重启把"一次故障"放大成"每十几分钟一次故障"。
#
# 所以这一版每次检测到失败都先做三件事，再决定动不动手：
#   1. 认出失败类型（teacher 超时 / 容器消失 / 机器丢失 / step 停滞）
#   2. 如果是 teacher 超时，查出是**哪一台**的 teacher 卡的
#   3. 和上一次失败比：同一台机器 + 同一类型 = 老故障重演 -> **停下不重启**
#
# 只有"新故障"或"机器丢失"这类明确可恢复的才自动续跑，且最多两次。
set -uo pipefail
source "${ONNODE_ENV:-/home/jimguo12/k3_regen/scripts/env.sh}"

# ---------------------------------------------------------------- 站点配置
# 全部可用环境变量覆盖。onnode()/held() 需要你自己提供：契约分别是"在第 N 台机器
# 上跑一条命令"和"列出当前持有的节点号"。我们用 spur，换 ssh/srun 都可以。
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TARGET=68283
BASE=49000
NODES="039 040 041 042 043"
DRAFT=040
LOGROOT="${LOGROOT:-/shared_nfs/jimguo12/lumenrl/logs}"
RUN_PREFIX="${RUN_PREFIX:-kimi_k3_dspark_atom}"
RUN_GLOB="${RUN_GLOB:-gen7-*}"
RUN_TAG="${RUN_TAG:-gen7}"          # 自动重启时新 run 用这个前缀
LAUNCHER="${LAUNCHER:-$HERE/launch_multinode.sh}"
CKPT="${CKPT:-/shared_nfs/jimguo12/lumenrl/checkpoints/k3_dspark_gen5}"
INTERVAL="${INTERVAL:-600}"
STALL_LIMIT=1800          # step 停这么久才算死（一个 step 8-12s，eval 点 40s）
MAX_AUTO=2                # 自动重启上限，超过就停下等人
STATE="${STATE:-/tmp/monitor_training.state}"
log() { echo "[$(date -u '+%F %T')] $*"; }

cur_run() {
  onnode "$DRAFT" "ls -dt $LOGROOT/${RUN_PREFIX}_${RUN_GLOB}/ 2>/dev/null | grep -v verify | head -1" \
    2>/dev/null | tr -d '\r' | sed 's:/*$::'
}
cur_step() {  # 只看当前 run，跨 run 取最大值会让停滞判据永远成立
  local L="$1"; [[ -z "$L" ]] && { echo 0; return; }
  onnode "$DRAFT" "grep -hoE 'callbacks: step=[0-9]+' $L/node-00-*.log 2>/dev/null \
    | grep -oE '[0-9]+' | sort -n | tail -1" 2>/dev/null | tr -dc '0-9'
}
alive() {     # 五台一起数：训练真死是五台都没有容器
  local t=0 v
  for n in $NODES; do
    v=$(onnode "$n" 'docker ps --format "{{.Names}}" | grep -c kimi-k3-sddd' 2>/dev/null | tr -dc '0-9')
    t=$(( t + ${v:-0} ))
  done; echo "$t"
}
held() { squeue -u "$USER" -h -o '%N' 2>/dev/null | grep -oE '[0-9]+$' | sort -u | tr '\n' ' '; }

# teacher_timeout 的前兆：四个副本里一个先慢下来，几小时后才彻底不响应。
# gen-7 第一个 run 的实测形状（分副本 p50，近 1000 步）：
#   退化前  r0 2.90  r1 2.73  r2 3.00  r3 3.09
#   退化后  r0 1.42  r1 47.85  r2 1.34  r3 1.33     <- 慢 35 倍
# 注意另外三个副本**变快了**（队列被拖空、没有竞争），所以 teacher_s 的平均值
# 只从 2.7 涨到 13.3，看着像"整体慢了一点"；必须分副本才看得出是一个坏了。
# 它就这样又跑了七小时（step 50,320 -> 51,641，全程 2.3 倍慢）才被超时带走。
#
# 判据刻意保守，不想再出一次"监控误杀健康训练"：
#   慢副本 p50 > 3 倍基准 **且** 绝对值超基准 8 秒，连续 3 次检查（30 分钟）才算。
slow_teacher() {
  local L="$1"; [[ -z "$L" ]] && { echo ""; return; }
  onnode "$DRAFT" "python3 - <<'PY' 2>/dev/null
import glob, re, statistics
f = sorted(glob.glob('$L/node-00-*.log'))
if not f: raise SystemExit
rows = []
for line in open(f[0], errors='ignore'):
    if 'callbacks: step=' not in line: continue
    s = re.search(r'step=(\d+)', line)
    r = re.search(r'teacher/replica=(\d+)', line)
    t = re.search(r'timing/teacher_s=([\d.]+)', line)
    if s and r and t: rows.append((int(s.group(1)), int(r.group(1)), float(t.group(1))))
if not rows: raise SystemExit
cut = rows[-1][0] - 200
per = {}
for st, rep, v in rows:
    if st > cut: per.setdefault(rep, []).append(v)
if len(per) < 2: raise SystemExit
med = {r: statistics.median(v) for r, v in per.items()}
base = statistics.median(list(med.values()))
bad = [str(r) for r, m in med.items() if m > 3 * base and m > base + 8]
if bad: print(','.join(bad))
PY" 2>/dev/null | tr -d '\r' | tail -1
}

# 出事时把现场问清楚：是哪台的 teacher 卡的
diagnose() {
  local L="$1"
  [[ -z "$L" ]] && { echo "unknown|无日志"; return; }
  local out
  out=$(onnode "$DRAFT" "
    F=\$(ls $L/node-00-*.log 2>/dev/null | head -1)
    [[ -f \$F ]] || { echo 'nolog|'; exit; }
    if grep -q 'cmd=extract_hidden after' \$F 2>/dev/null; then
      ip=\$(grep -oE 'ip=10\.245\.[0-9.]+' \$F | grep -oE '10\.245\.[0-9.]+' \
            | sort | uniq -c | sort -rn | head -1 | awk '{print \$2}')
      echo \"teacher_timeout|\$ip\"
    elif grep -q 'ChildFailedError' \$F 2>/dev/null; then
      echo 'child_failed|'
    else
      echo \"other|\$(grep -oE 'RuntimeError: [^\\\"]{0,60}' \$F | tail -1)\"
    fi" 2>/dev/null | tr -d '\r' | tail -1)
  echo "${out:-unknown|}"
}
ip2node() {
  case "$1" in
    # /!\ 这张表是本集群特有的。teacher 超时的报错里只有 Ray actor 的 IP，
    # 而你要知道是哪台机器才能去查它。换集群就换这张表（`hostname -I` 采一遍）。
    # 表不对只影响可读性，不影响判定 —— 判定用的是 "kind:node" 签名的稳定性。
    10.245.149.227) echo 039 ;; 10.245.158.39) echo 040 ;;
    10.245.154.154) echo 041 ;; 10.245.149.97) echo 042 ;;
    10.245.155.134) echo 043 ;; *) echo "${1:-?}" ;;
  esac
}

auto=0; last=0; stall_acc=0; slow_acc=0
log "监控启动。目标 step $TARGET。政策：老故障重演就停下，不自动重启。"

while true; do
  RUN=$(cur_run); s=$(cur_step "$RUN"); c=$(alive); h=$(held)
  s=${s:-0}; c=${c:-0}

  (( s >= TARGET )) && { log "训练完成：step $s"; exit 0; }

  # 正在启动就别打扰（装 K3 权重约 5 分钟，此时还没有容器）
  lp=$(ps -eo pid,etimes,args --no-headers | grep '[l]aunch_stage3.sh' | awk '$2 < 1200 {print $1}' | head -1)
  [[ -n "$lp" ]] && { log "run 正在启动（pid $lp）"; sleep "$INTERVAL"; continue; }

  fail=""
  miss=""; for n in $NODES; do grep -qw "$n" <<<"$h" || miss="$miss $n"; done
  [[ -n "$miss" ]] && fail="nodes_lost"
  if [[ -z "$fail" && "$c" -eq 0 ]]; then
    sleep 90; c2=$(alive); [[ "${c2:-0}" -eq 0 ]] && fail="container_gone"
  fi
  if [[ -z "$fail" ]]; then
    if (( s <= last )); then
      stall_acc=$(( stall_acc + INTERVAL ))
      log "step 停在 $s（已 ${stall_acc}s）"
      (( stall_acc >= STALL_LIMIT )) && fail="stalled"
    else
      stall_acc=0
      pct=$(( (s-BASE)*100/(TARGET-BASE) ))
      log "step $s  进度 ${pct}%  容器 $c/5  run=$(basename "$RUN")"
    fi
  fi

  # 还活着但有副本在退化 —— 别等它挂，那要再烧七小时
  if [[ -z "$fail" ]]; then
    sl=$(slow_teacher "$RUN")
    if [[ -n "$sl" ]]; then
      slow_acc=$(( slow_acc + 1 ))
      log "/!\\ teacher 副本 $sl 明显偏慢（第 $slow_acc/3 次）"
      (( slow_acc >= 3 )) && fail="teacher_degraded"
    else
      slow_acc=0
    fi
  fi
  last=$s

  [[ -z "$fail" ]] && { sleep "$INTERVAL"; continue; }

  # ---- 出事了，先诊断 ----
  IFS='|' read -r kind detail <<<"$(diagnose "$RUN")"
  node=""; [[ "$kind" == "teacher_timeout" ]] && node=$(ip2node "$detail")
  sig="$kind:$node"
  prev=$(cat "$STATE" 2>/dev/null || echo "")
  log "/!\\ 失败：$fail   类型=$kind   ${node:+卡死的机器=$node}   ${detail:+详情=$detail}"
  echo "$sig" > "$STATE"

  if [[ -n "$prev" && "$prev" == "$sig" ]]; then
    log "/!\\ 和上次同一个故障（$sig）—— **不重启**。这是需要人查的："
    log "    日志: $RUN/node-00-*.log"
    [[ -n "$node" ]] && log "    嫌疑机器 $node：查它的 rocm-smi、dmesg、以及是否每次都是它"
    log "    checkpoint 停在 $(onnode "$DRAFT" "ls $CKPT/checkpoint_*.pt 2>/dev/null | grep -oE '[0-9]+' | sort -n | tail -1" 2>/dev/null | tr -dc '0-9')"
    exit 2
  fi
  if (( auto >= MAX_AUTO )); then
    log "/!\\ 已自动重启 $auto 次，不再自动处理，等人介入"; exit 2
  fi

  auto=$((auto+1))
  log "新故障，自动续跑第 $auto/$MAX_AUTO 次"
  for n in $NODES; do
    onnode "$n" 'for x in $(docker ps -aq --filter name=kimi-k3-sddd); do docker rm -f $x >/dev/null 2>&1; done' >/dev/null 2>&1 &
  done; wait
  before=$(onnode "$DRAFT" "ls $CKPT/checkpoint_*.pt 2>/dev/null | grep -oE '[0-9]+' | sort -n | tail -1" 2>/dev/null | tr -dc '0-9')
  rid="${RUN_TAG}-$(date -u +%m%d%H%M)"
  setsid nohup env RUN_ID="$rid" bash "$LAUNCHER" \
    > "${LAUNCH_LOG_DIR:-/tmp}/launch_$rid.log" 2>&1 < /dev/null & disown
  log "已发起 $rid（checkpoint 在 $before）"
  sleep 1800
  after=$(cur_step "$(cur_run)")
  if [[ "${after:-0}" -lt "${before:-0}" ]]; then
    log "/!\\ 重启后 step $after 低于 checkpoint $before —— resume 没接上，停下"; exit 1
  fi
  log "已接上：step $after"
  last=0; stall_acc=0
done
