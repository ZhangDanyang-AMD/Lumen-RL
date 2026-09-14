#!/usr/bin/env bash
# 看 gen-7 的三把 AL 尺子和 loss。随时跑，输出一屏能看完。
#
# 用法:
#   bash show_progress.sh            # 当前状态 + 曲线
#   WATCH=1 bash show_progress.sh    # 常驻，默认 20 分钟刷一次
#   WATCH=1 WATCH_INTERVAL=120 bash show_progress.sh
#
# 为什么要三把尺子：
#   eval       —— gen-7 自己的尾部（36.7% 多语种、10.2% 编程），涨了只说明学会了
#                 本轮这个分布，而本轮这个分布里多语种回到了 36.7%
#   eval_gen5  —— gen-5 那 128 行，跨四代不变的基准，用来和历史曲线对齐
#   eval_gen6  —— gen-6 那 128 行，**这一轮真正要守的底线**。gen-6 的全部价值是
#                 长文本长代码（throughput16k 41.7% -> 104.5%），而 gen-7 把编程
#                 占比从 20% 降到 10.2%、多语种从 12.3% 抬回 36.7%。这把尺子掉
#                 下去就是在拿 gen-6 的成果换东西，该停下来，不要等 benchmark。
set -uo pipefail
source "${ONNODE_ENV:-/home/jimguo12/k3_regen/scripts/env.sh}"

# ---------------------------------------------------------------- 站点配置
# 下面这些是我们集群的值，全部可以用环境变量覆盖。脚本本身的价值在逻辑，不在路径：
#   - 每次刷新都重新解析 run 目录（自动重启会换目录，写死路径会一直读死日志）
#   - 三把 eval 尺子分列
#   - **分副本**看 teacher 耗时（平均值会掩盖"一个副本慢 35 倍"这种故障）
# onnode() 需要你自己提供：它的契约是"在第 N 台机器上跑一条命令"。我们用的是
# spur（onnode() 定义在 env.sh 里）；换成 ssh / srun / kubectl exec 都可以。
SHARED_ROOT="${SHARED_ROOT:-/shared_nfs/jimguo12/lumenrl}"
RUN_PREFIX="${RUN_PREFIX:-kimi_k3_dspark_atom}"
RUN_GLOB="${RUN_GLOB:-gen7-*}"          # 只看这一轮的 run 目录
DRAFT_LOG_PAT="${DRAFT_LOG_PAT:-node-00-*}"
# 起点基线：lr=0 校验跑出来的三把尺子读数。换轮次必须重填。
BASE_STEP="${BASE_STEP:-49000}"
TARGET_STEP="${TARGET_STEP:-68283}"
B_EVAL="${B_EVAL:-2.801}"
B_GEN5="${B_GEN5:-2.972}"
B_GEN6="${B_GEN6:-2.567}"
# ------------------------------------------------------------------------


NODE="${DRAFT_NODE:-040}"

# /!\ 每次刷新都要重新解析 run 目录，不能在脚本开头解析一次。
# 自动重启会换 run 目录（gen7-09131704 -> gen7-09140327），而 WATCH 模式的循环
# 体如果用的是启动时算好的路径，就会一直读那个已经不再增长的旧日志 ——
# 表现是"step 和 s/步 卡在一个值上一百分钟不动"，看起来像训练变慢或卡死，
# 实际训练在另一个目录里跑得很正常。这个坑在 gen-6 用 /tmp/gen6_rid 缓存
# run 名时踩过一次，那次是因为缓存文件过期。
cur_run() {
  onnode "$NODE" "ls -dt $SHARED_ROOT/logs/${RUN_PREFIX}_${RUN_GLOB} 2>/dev/null | grep -v verify | head -1" \
    | xargs -r basename | sed 's/^kimi_k3_dspark_atom_//'
}

cat > /tmp/gen7_show.py <<'PY'
import re, sys, os, statistics
F = sys.argv[1]
BASE, TARGET = int(sys.argv[2]), int(sys.argv[3])
# lr=0 校验跑出来的起点读数（step 49000，权重未动）
B_EVAL, B_GEN5, B_GEN6 = (float(x) for x in sys.argv[4:7])

if not os.path.exists(F):
    print(f"日志还没出现: {F}"); sys.exit(0)

steps, evals, tch = [], [], []
for line in open(F, errors="ignore"):
    if "callbacks: step=" in line:
        m = re.search(r"step=(\d+).*?ce_loss=([\d.]+).*?loss=([\d.]+).*?lr=([\d.e+-]+)", line)
        if m: steps.append((int(m.group(1)), float(m.group(2)), float(m.group(3)), m.group(4)))
        t = re.search(r"timing/step_s=([\d.]+)", line)
        if t and steps: steps[-1] = steps[-1] + (float(t.group(1)),)
        r = re.search(r"teacher/replica=(\d+)", line)
        ts = re.search(r"timing/teacher_s=([\d.]+)", line)
        if m and r and ts:
            tch.append((int(m.group(1)), int(r.group(1)), float(ts.group(1))))
    elif "eval step=" in line:
        m = re.search(r"eval step=(\d+)", line)
        a  = re.search(r"eval/simulated_acc_len=([\d.]+)", line)
        g5 = re.search(r"eval_gen5/simulated_acc_len=([\d.]+)", line)
        g6 = re.search(r"eval_gen6/simulated_acc_len=([\d.]+)", line)
        el = re.search(r"eval/loss=([\d.]+)", line)
        l6 = re.search(r"eval_gen6/loss=([\d.]+)", line)
        if m and a:
            evals.append((int(m.group(1)), float(a.group(1)),
                          float(g5.group(1)) if g5 else None,
                          float(g6.group(1)) if g6 else None,
                          float(el.group(1)) if el else None,
                          float(l6.group(1)) if l6 else None))
if not steps:
    print("还没有 step 行（可能还在加载 K3 权重，约 5 分钟）"); sys.exit(0)

s = steps[-1]
done, total = s[0]-BASE, TARGET-BASE
st = [x[4] for x in steps[-50:] if len(x) > 4]
sps = sum(st)/len(st) if st else 0
eta = (TARGET-s[0])*sps/3600 if sps else 0
bar = "█"*int(done/total*32) + "·"*(32-int(done/total*32))
print(f"step {s[0]:,} / {TARGET:,}   [{bar}] {100*done/total:.1f}%")
print(f"  lr={s[3]}   {sps:.2f} s/步   预计还要 {eta:.1f} 小时")
print(f"  train: ce_loss={s[1]:.4f}  loss={s[2]:.4f}")

# 四个 teacher 副本分开看。gen-7 第一个 run 就是这么烂掉的：replica 1 从 2.7s
# 退化到 p50 49.5s（另外三个反而快到 1.4s，因为队列被拖空、没有竞争），这样
# 又跑了七小时才彻底不响应、被 600 秒超时带走。**平均值看不出来**，必须分副本。
REP2NODE = {0: "039?", 1: "041?", 2: "042?", 3: "043?"}
if tch:
    recent = [x for x in tch if x[0] > s[0] - 200]
    per = {}
    for _, rep, v in recent:
        per.setdefault(rep, []).append(v)
    if len(per) >= 2:
        med = {r: statistics.median(v) for r, v in sorted(per.items())}
        base = statistics.median(list(med.values()))
        cells = "  ".join(f"r{r}={m:.1f}s" for r, m in med.items())
        print(f"  teacher 各副本 p50（近 200 步）: {cells}")
        bad = [r for r, m in med.items() if m > max(3 * base, base + 8)]
        if bad:
            print(f"  /!\\ 副本 {bad} 明显慢于其他（基准 {base:.1f}s）—— 这是 "
                  f"teacher_timeout 的前兆，别等它挂，直接重启这一轮")

if evals:
    print(f"\n{'step':>8}{'eval AL':>10}{'gen5 AL':>10}{'gen6 AL':>10}"
          f"{'eval loss':>11}{'gen6 loss':>11}")
    print(f"{'起点':>8}{B_EVAL:>10.3f}{B_GEN5:>10.3f}{B_GEN6:>10.3f}"
          f"{'—':>11}{'—':>11}   <- lr=0 校验读数")
    show = evals if len(evals) <= 14 else evals[::max(1, len(evals)//12)] + [evals[-1]]
    seen = set()
    for e in show:
        if e[0] in seen: continue
        seen.add(e[0])
        f = lambda v, w=10, p=3: (f"{v:>{w}.{p}f}" if v is not None else f"{'—':>{w}}")
        print(f"{e[0]:>8,}{e[1]:>10.3f}{f(e[2])}{f(e[3])}"
              f"{f(e[4], 11, 4)}{f(e[5], 11, 4)}")

    last = evals[-1]
    print(f"\n  eval（gen-7 分布）  {B_EVAL:.3f} -> {last[1]:.3f}   {last[1]-B_EVAL:+.3f}")
    if last[2] is not None:
        print(f"  eval_gen5（基准）    {B_GEN5:.3f} -> {last[2]:.3f}   {last[2]-B_GEN5:+.3f}")
    if last[3] is not None:
        d = last[3] - B_GEN6
        # 阈值 -0.02：lr=0 校验时同一把尺子在四个读数间的抖动约 0.01（2.567/2.558），
        # 所以 -0.02 已经在噪声之外。
        flag = "长上下文没退化，好" if d >= -0.02 else "/!\\ 在拿 gen-6 的长上下文换东西，该查了"
        print(f"  eval_gen6（底线）    {B_GEN6:.3f} -> {last[3]:.3f}   {d:+.3f}   {flag}")
PY

show() {
  local run="${RUN_ID:-$(cur_run)}"
  local f
  f=$(onnode "$NODE" "ls $SHARED_ROOT/logs/${RUN_PREFIX}_${run}/${DRAFT_LOG_PAT}.log 2>/dev/null | head -1" | tr -d "\r")
  echo "run=$run"
  onnode "$NODE" "cat > /tmp/gen7_show.py <<'EOF'
$(cat /tmp/gen7_show.py)
EOF
python3 /tmp/gen7_show.py $f $BASE_STEP $TARGET_STEP $B_EVAL $B_GEN5 $B_GEN6"
}

if [[ "${WATCH:-0}" == "1" ]]; then
  while true; do
    # 只有输出是终端时才 clear。重定向到文件时 clear 会写进一堆 ANSI 转义码，
    # 文件就没法读了 —— 常驻监控几乎总是重定向的。
    [[ -t 1 ]] && clear || echo "================================================"
    date -u '+%F %T UTC'; echo; show
    sleep "${WATCH_INTERVAL:-1200}"
  done
else
  date -u '+%F %T UTC'; echo; show
fi
