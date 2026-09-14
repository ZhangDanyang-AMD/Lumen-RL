#!/usr/bin/env bash
# 通用 benchmark 驱动：一个 draft、一组题目集、一个窗口。
#
# benchmark_dspark.sh 自己没有重启续跑：客户端一旦以 42（单请求超时 = 引擎卡死）
# 或 43（主动轮转）退出，整条就停在那里。gen-3 那轮在 GSM8K 550/1319 处撞过一次
# aiter 跨 rank 广播死锁——8 个 rank 自旋、/health 仍返回 200（那是 API 前端，
# 和 engine core 是两个进程），客户端把 550 条已完成的工作全丢了。
#
# 这里补两件事：
#   1. 非零退出就重启 server 并带 --resume 续跑
#   2. 重启前把已经产出最终 JSON 的题目集摘掉 —— run_client.py 在最终 flush 时
#      删掉 .partial.json，对已完成的集子 --resume 无从下手，不摘就从头重跑
#
# 跨服务生命周期续跑是有效测量：接受长度分布在前向步上可加，run_client.py 把
# partial 里的 cumulative_distribution 读回当基线、服务端计数器只作增量起点，
# 且 prefix caching 关闭、采样贪心。
#
# 用法（全部走环境变量，便于并行起多路）：
#   NODE=039 DRAFT=... LABEL=gen5-t0 SETS="math500" bash run_bench.sh
set -uo pipefail

# 站点相关，用环境变量覆盖
BENCH="${BENCH_DIR_ROOT:-/home/jimguo12/k3-bench}"
RESULTS=$BENCH/results
DRAFT="${DRAFT:?需要 DRAFT}"
LABEL="${LABEL:?需要 LABEL}"
SETS="${SETS:?需要 SETS}"
MAX_TOKENS="${MAX_TOKENS:-12288}"
MAX_RESTARTS="${MAX_RESTARTS:-12}"
ROTATE_AFTER="${ROTATE_AFTER:-400}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-900}"

export TARGET_MODEL="${TARGET_MODEL:-/mnt/m2m_nobackup/jimguo12/models/Kimi-K3}"
# 钉死的 digest。latest 是滚动 tag，漂过至少两次。
export IMAGE=rocm/atom-dev:nightly_202608141640
export BENCH_DIR=$BENCH
export RESULT_DIR=$RESULTS
export MAX_NUM_SEQS=8
export KV_CACHE_DTYPE=fp8
export STARTUP_TIMEOUT=3600
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-10240}"

log() { echo "[$(date -u '+%F %T')] [$LABEL] $*"; }

for attempt in $(seq 0 "$MAX_RESTARTS"); do
  pending=""
  for s in $SETS; do
    if [[ -f "$RESULTS/bench_${LABEL}-${s}.json" && ! -f "$RESULTS/bench_${LABEL}-${s}.json.partial.json" ]]; then
      continue
    fi
    pending="$pending $s"
  done
  pending="${pending# }"

  [[ -z "$pending" ]] && { log "全部题目集已完成"; break; }
  log "第 $attempt 轮，待跑: $pending"

  qs=""
  for s in $pending; do qs="$qs $s:$BENCH/data/$s.jsonl"; done
  export QUESTION_SETS="${qs# }"
  export CLIENT_EXTRA_ARGS="--temperature 0 --top-p 1.0 --request-timeout $REQUEST_TIMEOUT --rotate-after $ROTATE_AFTER --resume"

  bash "${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}/output/benchmark_dspark.sh" "$DRAFT" "$LABEL" 0 "$MAX_TOKENS"
  rc=$?
  log "benchmark_dspark.sh 退出码 $rc"
  case "$rc" in
    0)  ;;
    42) log "引擎卡死，重启续跑" ;;
    43) log "主动轮转，重启续跑" ;;
    *)  log "非预期退出码 $rc，仍然重启续跑（看 $RESULTS/server_${LABEL}.log）" ;;
  esac
  sleep 30
done

log "=== 汇总 ==="
for s in $SETS; do
  f="$RESULTS/bench_${LABEL}-${s}.json"
  [[ -f "$f" ]] || { echo "  $s: 缺"; continue; }
  python3 - "$f" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print(f"  {d['label']:34} n={d['n']:5} tok/fwd={d['tok_per_fwd']:.4f} "
      f"accept={100*d['acceptance_rate']:.2f}% steps={d['forward_steps']}")
PY
done
