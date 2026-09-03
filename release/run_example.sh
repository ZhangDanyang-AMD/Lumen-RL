#!/usr/bin/env bash
# LumenRL release launcher: run one of the seven validated examples with a
# single argument. Runs on the GPU *host* (not inside the container) and drives
# the published image through `docker run` / `docker exec`.
#
#   DATA_ROOT=/path/to/data bash release/run_example.sh 1 --check
#
# See release/README.md §4 for the full user-facing documentation.
set -euo pipefail

SCRIPT_NAME="$(basename "$0")"

# Set DOCKER="sudo docker" if your user is not in the docker group.
DOCKER_CLI="${DOCKER:-docker}"
read -r -a DOCKER <<<"$DOCKER_CLI"
IMAGE="${IMAGE:-zhangdanyangamd/lumen-rl:dapo-gfx950-rocm7.2.3-260902}"
CONTAINER="${CONTAINER:-lumenrl-release}"
RL_ROOT_IN_IMAGE="${RL_ROOT_IN_IMAGE:-/opt/lumenrl}"
RUN_DAPO="$RL_ROOT_IN_IMAGE/Lumen-RL/examples/DAPO/run_dapo.sh"
CFG_DIR="examples/DAPO/configs"

# GPU is considered idle below this many bytes of VRAM in use. The MI355X
# baseline measured on an empty node is ~298 MB per card; 2 GB leaves room for
# other tenants' bookkeeping without hiding a leaked 90 GB engine.
IDLE_VRAM_BYTES=2147483648

# ---------------------------------------------------------------------------
# Example table
# ---------------------------------------------------------------------------
# Fields, '|'-separated:
#   1 title
#   2 MODE
#   3 TRAIN_FP8
#   4 smoke config (relative to $RL_ROOT/Lumen-RL)
#   5 longrun config
#   6 default STEPS for the smoke
#   7 model subdirectory under $DATA_ROOT/models
#   8 extra env, space separated K=V (empty = none)
#   9 max_response_length of the smoke config (documentation only)
#  10 reference rollout_corr/k3_kl at step 1     (checked, relative tolerance)
#  11 reference entropy at step 1                (checked, relative tolerance)
#  12 reference rollout_corr/kl at step 1        (order-of-magnitude check only)
#  13 relative tolerance for k3_kl (fraction)
#  14 relative tolerance for entropy (fraction)
#
# Why k3_kl and not kl: `rollout_corr/kl` is a SIGNED token mean of
# (rollout_logp - train_logp). Symmetric disagreement cancels inside it, so it
# is noise around zero — five runs of example 1 with the same seed spread it by
# +-52% around its own mean, and the sign of ppl_ratio-1 flipped between them.
# `k3_kl` is the non-negative k3 estimator of the same gap and stayed within 9%
# of its mean over those same five runs. So k3_kl and entropy carry
# the tolerances; kl is only checked for staying inside a 10x band, which is
# the actual published criterion ("one order of magnitude above is bad").
#
# Why the tolerances differ per example: run-to-run jitter grows with
# max_response_length, because dynamic sampling (filter_groups) then selects a
# visibly different batch each time. Measured maximum deviation from the mean,
# over the runs recorded in release/VALIDATION-20260903.md:
#
#   example 1 (512,  5 runs):  k3_kl +-9%,  entropy +-7%
#   example 4 (4096, 3 runs):  k3_kl +-17%, entropy +-16%
#   example 5 (4096, 2 runs):  k3_kl +-4%,  entropy +-4%
#   example 6 (4096, 2 runs):  k3_kl +-8%,  entropy +-19%
#   example 7 (4096, 2 runs):  k3_kl +-2%,  entropy +-16%
#
# Tolerances are ~3x the observed deviation, floored per group: k3_kl 30% at
# 512 and 50% at 4096; entropy 25% at 512, 50% for the ATOM 4k pair and 60%
# for the MoE pair. Entropy on a 4k run is therefore only a coarse sanity
# check -- k3_kl is the metric that actually tracks train/rollout alignment,
# and it stayed within +-17% everywhere.
#
# Reference values: see release/README.md §6. Every number was measured on
# 8x MI355X with this script; "na" means not measured.
declare -A EX
EX[1]='8B BF16 baseline|bf16|0|dapo_qwen3_8b_ray_vllm_smoke.yaml|dapo_qwen3_8b_ray_vllm_longrun.yaml|3|Qwen3-8B-Base||512|0.00109|0.603|0.00097|0.30|0.25'
EX[2]='8B FP8 rollout|fp8|0|dapo_qwen3_8b_ray_vllm_fp8_smoke.yaml|dapo_qwen3_8b_ray_vllm_fp8_longrun.yaml|3|Qwen3-8B-Base||512|0.00469|0.789|0.00468|0.30|0.25'
EX[3]='8B FP8 end-to-end|fp8|1|dapo_qwen3_8b_ray_vllm_fp8_smoke.yaml|dapo_qwen3_8b_ray_vllm_fp8_longrun.yaml|3|Qwen3-8B-Base||512|0.00410|0.812|0.00412|0.30|0.25'
EX[4]='8B ATOM FP8|atomfp8|1|dapo_qwen3_8b_ray_atom_fp8_4k_smoke.yaml|dapo_qwen3_8b_ray_atom_fp8_longrun.yaml|3|Qwen3-8B-Base||4096|0.00286|0.597|0.00268|0.50|0.50'
EX[5]='8B ATOM BF16|atombf16|0|dapo_qwen3_8b_ray_atom_bf16_4k_smoke.yaml|dapo_qwen3_8b_ray_atom_bf16_longrun.yaml|1|Qwen3-8B-Base||4096|0.000988|0.641|0.000821|0.50|0.50'
EX[6]='MoE FSDP2|bf16|0|dapo_qwen3moe_a3b_ray_vllm_verlref_4k_smoke.yaml|dapo_qwen3moe_a3b_ray_vllm_verlref_longrun.yaml|3|Qwen3-30B-A3B-Base|LUMENRL_FP32_MOE_ROUTER=0|4096|0.00154|0.864|0.00178|0.50|0.60'
EX[7]='MoE Megatron EP=8|bf16|0|dapo_qwen3moe_a3b_ray_megatron_verlref_4k_smoke.yaml|dapo_qwen3moe_a3b_ray_megatron_verlref_4k_longrun.yaml|3|Qwen3-30B-A3B-Base|LUMENRL_FP32_MOE_ROUTER=0|4096|0.00157|0.655|0.00166|0.50|0.60'

field() { echo "${EX[$1]}" | cut -d'|' -f"$2"; }

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
usage() {
  cat <<EOF
$SCRIPT_NAME — run one validated LumenRL example end to end.

USAGE
  DATA_ROOT=/path/to/data bash release/$SCRIPT_NAME <1..7> [options]

EXAMPLES (smoke defaults; all of them fit on one 8x gfx950 node)
  #  name                 MODE      TRAIN_FP8  config                                              steps  resp
EOF
  local n
  for n in 1 2 3 4 5 6 7; do
    printf '  %s  %-19s %-9s %-10s %-51s %-6s %s\n' \
      "$n" "$(field "$n" 1)" "$(field "$n" 2)" "$(field "$n" 3)" \
      "$(field "$n" 4)" "$(field "$n" 6)" "$(field "$n" 9)"
  done
  cat <<EOF

OPTIONS
  --check          after the run, extract step-1 metrics, compare against the
                   built-in reference values and print PASS/FAIL per metric;
                   also count Traceback / OutOfMemory / CUDA error / HSA_STATUS
  --check-only     skip the run, only check an existing log (needs --log)
  --log PATH       log file to write (default \$DATA_ROOT/logs/example-<N>-<ts>.log)
  --steps N        override the number of training steps
  --longrun        use the example's longrun config instead of the smoke config
                   (needs WANDB_API_KEY, or the launcher disables wandb for you)
  --detach         start with 'docker exec -d' and return immediately
  --dry-run        print what would be executed and exit
  --force          auto-remediate a dirty node (docker rm -f leftovers) instead
                   of only reporting it
  --no-restart     do not restart the container before the run
  --keep-cache     do not clear the ATOM/inductor compile caches when switching
                   ATOM precision (examples 4 <-> 5)
  --verbose        stream the whole log instead of the filtered highlights
  -h, --help       this text

ENVIRONMENT
  DATA_ROOT   required. Host directory holding models/, data_cached/, logs/.
  IMAGE       default $IMAGE
  CONTAINER   default $CONTAINER
  WANDB_API_KEY  only needed with --longrun.
  EXTRA_OVERRIDE  extra Hydra overrides, space separated, appended verbatim,
                  e.g. EXTRA_OVERRIDE='logger.wandb_enabled=false policy.learning_rate=1e-6'
EOF
}

if [ $# -eq 0 ]; then usage; exit 2; fi

N=""; DO_CHECK=0; CHECK_ONLY=0; LOG=""; STEPS=""; LONGRUN=0; DETACH=0
DRY_RUN=0; FORCE=0; NO_RESTART=0; KEEP_CACHE=0; VERBOSE=0

while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --check) DO_CHECK=1 ;;
    --check-only) DO_CHECK=1; CHECK_ONLY=1 ;;
    --log) LOG="${2:?--log needs a path}"; shift ;;
    --steps) STEPS="${2:?--steps needs a number}"; shift ;;
    --longrun) LONGRUN=1 ;;
    --detach) DETACH=1 ;;
    --dry-run) DRY_RUN=1 ;;
    --force) FORCE=1 ;;
    --no-restart) NO_RESTART=1 ;;
    --keep-cache) KEEP_CACHE=1 ;;
    --verbose) VERBOSE=1 ;;
    [1-7]) N="$1" ;;
    [0-9]*)
      echo "ERROR: '$1' is not a valid example number. Valid values are 1..7; see --help." >&2
      exit 2 ;;
    *) echo "ERROR: unknown argument '$1'. Try --help." >&2; exit 2 ;;
  esac
  shift
done

if [ -z "$N" ]; then
  echo "ERROR: no example number given. Valid values are 1..7; see --help." >&2
  exit 2
fi

die() { echo "ERROR: $*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Resolve the example
# ---------------------------------------------------------------------------
TITLE="$(field "$N" 1)"
MODE="$(field "$N" 2)"
TRAIN_FP8="$(field "$N" 3)"
CFG_SMOKE="$(field "$N" 4)"
CFG_LONG="$(field "$N" 5)"
DEF_STEPS="$(field "$N" 6)"
MODEL_SUBDIR="$(field "$N" 7)"
EXTRA_ENV="$(field "$N" 8)"
RESP_LEN="$(field "$N" 9)"
REF_K3="$(field "$N" 10)"
REF_ENT="$(field "$N" 11)"
REF_KL="$(field "$N" 12)"
TOL_K3="$(field "$N" 13)"
TOL_ENT="$(field "$N" 14)"

if [ "$LONGRUN" = 1 ]; then
  CONFIG="$CFG_DIR/$CFG_LONG"
  STEPS="${STEPS:-1000}"
else
  CONFIG="$CFG_DIR/$CFG_SMOKE"
  STEPS="${STEPS:-$DEF_STEPS}"
fi

: "${DATA_ROOT:?DATA_ROOT is not set. Export it to the host directory that holds models/ and data_cached/.}"
DATA_ROOT="${DATA_ROOT%/}"

TS="$(date +%Y%m%d-%H%M%S)"
LOG="${LOG:-$DATA_ROOT/logs/example-$N-$TS.log}"
OUTER_LOG="${LOG%.log}.launcher.log"
DONE_MARK="=== LAUNCHER_EXIT="

# ---------------------------------------------------------------------------
# --check-only short-circuits everything below
# ---------------------------------------------------------------------------
check_log() {
  local log="$1"
  [ -f "$log" ] || die "log not found: $log"
  echo
  echo "===== CHECK: example $N ($TITLE) ====="
  echo "log: $log"

  local errs=0 rc
  for pat in 'Traceback' 'OutOfMemory' 'CUDA error' 'HSA_STATUS'; do
    rc="$(grep -c -- "$pat" "$log" || true)"
    printf '  %-14s %s\n' "$pat" "$rc"
    errs=$((errs + rc))
  done

  # Two different lines start with "step=1 "; the one we want is the metrics
  # dump from LoggingCallback, identified by rollout_corr/kl.
  local line
  line="$(grep -F 'step=1 ' "$log" | grep -F 'rollout_corr/kl=' | head -n 1 || true)"
  if [ -z "$line" ]; then
    echo "  step-1 metrics: NOT FOUND — the run did not reach the first optimizer step."
    echo "RESULT: FAIL"
    return 1
  fi

  get() { sed -n "s|.* $1=\([^ ]*\).*|\1|p" <<<"$line"; }
  local k3 ent kl ppl
  k3="$(get 'rollout_corr/k3_kl')"
  ent="$(get 'entropy')"
  kl="$(get 'rollout_corr/kl')"
  ppl="$(get 'rollout_corr/ppl_ratio')"

  local verdict=0
  cmp_metric() {  # name measured reference tolerance mode
    local name="$1" got="$2" ref="$3" tol="$4" mode="$5"
    if [ -z "$got" ]; then
      printf '  %-22s %-12s %-34s FAIL\n' "$name" "-" "(not in log)"
      verdict=1; return
    fi
    if [ "$mode" = info ]; then
      printf '  %-22s %-12s %-34s INFO\n' "$name" "$got" "(informational, not checked)"
      return
    fi
    if [ "$ref" = "na" ]; then
      printf '  %-22s %-12s %-34s SKIP\n' "$name" "$got" "(no reference recorded)"
      return
    fi
    local out status
    out="$(python3 - "$got" "$ref" "$tol" "$mode" <<'PY'
import sys
got, ref, tol, mode = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
if mode == "band":
    # order-of-magnitude band: |got| must stay within [ref/tol, ref*tol]
    lo, hi = abs(ref) / tol, abs(ref) * tol
    ok = lo <= abs(got) <= hi
    print(f"|x|in[{lo:.2g},{hi:.2g}] {'PASS' if ok else 'FAIL'}")
else:
    dev = (got - ref) / abs(ref) if ref else 0.0
    print(f"{dev*100:+.1f}%_vs_{ref}_(tol_{tol*100:.0f}%) {'PASS' if abs(dev) <= tol else 'FAIL'}")
PY
)"
    local detail
    read -r detail status <<<"$out"
    printf '  %-22s %-12s %-34s %s\n' "$name" "$got" "$detail" "$status"
    [ "$status" = PASS ] || verdict=1
  }

  echo "  step-1 metrics:"
  cmp_metric "rollout_corr/k3_kl"     "$k3"  "$REF_K3"  "$TOL_K3"  rel
  cmp_metric "entropy"                "$ent" "$REF_ENT" "$TOL_ENT" rel
  cmp_metric "rollout_corr/kl"        "$kl"  "$REF_KL"  "10"   band
  cmp_metric "rollout_corr/ppl_ratio" "$ppl" ""         ""     info

  if [ "$errs" -ne 0 ]; then
    echo "RESULT: FAIL ($errs error lines in the log)"
    return 1
  fi
  if [ "$verdict" -ne 0 ]; then
    echo "RESULT: FAIL (metric outside tolerance)"
    echo "HINT: first check you really ran $CONFIG"
    echo "      (grep -m1 'CONFIG=' $OUTER_LOG), because a config mismatch looks"
    echo "      exactly like a numerical regression. Only then suspect the numbers."
    return 1
  fi
  echo "RESULT: PASS"
  return 0
}

if [ "$CHECK_ONLY" = 1 ]; then
  check_log "$LOG"
  exit $?
fi

# ---------------------------------------------------------------------------
# Build the container command
# ---------------------------------------------------------------------------
MODEL_PATH="$DATA_ROOT/models/$MODEL_SUBDIR"

DOCKER_ENV=(
  -e "RL_ROOT=$RL_ROOT_IN_IMAGE"
  -e "DATA_ROOT=$DATA_ROOT"
  -e "SCRATCH_ROOT=$DATA_ROOT"
  -e "PYTORCH_CUDA_ALLOC_CONF="
  -e "MODE=$MODE"
  -e "TRAIN_FP8=$TRAIN_FP8"
  -e "STEPS=$STEPS"
  -e "CONFIG_OVERRIDE=$CONFIG"
  -e "MODEL_PATH=$MODEL_PATH"
  -e "LOG=$LOG"
)
if [ -n "$EXTRA_ENV" ]; then
  for kv in $EXTRA_ENV; do DOCKER_ENV+=(-e "$kv"); done
fi

# wandb: smoke configs are wandb_enabled:false and need no account. Longrun
# configs are wandb_enabled:true; without a key the run dies *after*
# "RLTrainer.setup complete", so disable it up front rather than 10 minutes in.
OVERRIDES="${EXTRA_OVERRIDE:-}"
if [ "$LONGRUN" = 1 ] && [ -z "${WANDB_API_KEY:-}" ] && [[ "$OVERRIDES" != *logger.wandb_enabled* ]]; then
  echo "NOTE: --longrun selects a wandb_enabled:true config but WANDB_API_KEY is unset."
  echo "      Appending 'logger.wandb_enabled=false'. Export WANDB_API_KEY to log to wandb."
  OVERRIDES="${OVERRIDES:+$OVERRIDES }logger.wandb_enabled=false"
fi
if [ -n "${WANDB_API_KEY:-}" ]; then DOCKER_ENV+=(-e "WANDB_API_KEY=$WANDB_API_KEY"); fi
if [ -n "$OVERRIDES" ]; then DOCKER_ENV+=(-e "EXTRA_OVERRIDE=$OVERRIDES"); fi

INNER="bash $RUN_DAPO > $(printf '%q' "$OUTER_LOG") 2>&1; \
echo \"$DONE_MARK\$? ===\" | tee -a $(printf '%q' "$OUTER_LOG") >> $(printf '%q' "$LOG")"

if [ "$DRY_RUN" = 1 ]; then
  cat <<EOF
# ---------------------------------------------------------------------------
# example $N — $TITLE
#   config          : $CONFIG
#   steps           : $STEPS
#   response length : $RESP_LEN
#   model           : $MODEL_PATH
#   log             : $LOG
# ---------------------------------------------------------------------------

# 1. the container, created once and reused (restart it between runs):
$DOCKER_CLI run -d --name $CONTAINER \\
  --network=host --ipc=host \\
  --device=/dev/kfd --device=/dev/dri --group-add=video \\
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --shm-size 64G \\
  -v $DATA_ROOT:$DATA_ROOT -e DATA_ROOT=$DATA_ROOT \\
  $IMAGE sleep infinity

# 2. the example. Every variable below is required: MODE alone does NOT pick
#    the config, and CONFIG_OVERRIDE alone does NOT switch off MODE's extra
#    Hydra args.
$DOCKER_CLI exec \\
EOF
  for e in "${DOCKER_ENV[@]}"; do
    [ "$e" = "-e" ] && continue
    echo "  -e $e \\"
  done
  echo "  $CONTAINER bash -lc 'bash $RUN_DAPO'"
  cat <<EOF

# 3. the log is written by run_dapo.sh to \$LOG, not to stdout:
tail -f $LOG
grep -o 'step=[0-9]* .*rollout_corr/kl=[^ ]*' $LOG
EOF
  exit 0
fi

# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------
echo "== example $N — $TITLE"
echo "   config $CONFIG   steps $STEPS   model $MODEL_SUBDIR"

command -v "${DOCKER[0]}" >/dev/null || die "'${DOCKER[*]}' not found on this host. Set DOCKER='sudo docker' if needed."

# 1. data
missing=0
for p in "$MODEL_PATH" \
         "$DATA_ROOT/data_cached/qwen3-8b-maxprompt1024/dapo-math-17k.filtered.parquet" \
         "$DATA_ROOT/data_cached/qwen3-8b-maxprompt1024/aime-2024.filtered.parquet"; do
  if [ ! -e "$p" ]; then echo "   MISSING: $p"; missing=1; fi
done
if [ "$missing" = 1 ]; then
  echo
  echo "DATA_ROOT=$DATA_ROOT is incomplete. Required layout (see README §4.2):"
  echo "  \$DATA_ROOT/models/Qwen3-8B-Base/                                  ~16 GB (all examples)"
  echo "  \$DATA_ROOT/models/Qwen3-30B-A3B-Base/                             ~57 GB (examples 6, 7)"
  echo "  \$DATA_ROOT/data_cached/qwen3-8b-maxprompt1024/dapo-math-17k.filtered.parquet   ~1.0 GB"
  echo "  \$DATA_ROOT/data_cached/qwen3-8b-maxprompt1024/aime-2024.filtered.parquet       ~0.9 MB"
  exit 1
fi
mkdir -p "$DATA_ROOT/logs"

# 2. image
if ! "${DOCKER[@]}" image inspect "$IMAGE" >/dev/null 2>&1; then
  die "image $IMAGE not present. Run: $DOCKER_CLI pull $IMAGE"
fi

# 3. leftovers from someone else's run
others="$("${DOCKER[@]}" ps -a --filter "ancestor=$IMAGE" --format '{{.Names}}' | grep -vx "$CONTAINER" || true)"
if [ -n "$others" ]; then
  echo "   other containers on this image: $(echo "$others" | tr '\n' ' ')"
  if [ "$FORCE" = 1 ]; then
    echo "   --force: removing them"
    echo "$others" | xargs -r "${DOCKER[@]}" rm -f >/dev/null
  else
    echo
    echo "Those containers can be holding GPU memory. Remove them, then re-run:"
    echo "$others" | sed "s|^|  $DOCKER_CLI rm -f |"
    echo "  (or re-run this script with --force)"
    exit 1
  fi
fi

# 4. a previous detached run of ours still alive?
if [ -f "$DATA_ROOT/logs/.running-$CONTAINER" ]; then
  prev="$(cat "$DATA_ROOT/logs/.running-$CONTAINER")"
  if [ -f "$prev" ] && ! grep -q "$DONE_MARK" "$prev" 2>/dev/null; then
    s1=$(stat -c %s "$prev" 2>/dev/null || echo 0); sleep 10
    s2=$(stat -c %s "$prev" 2>/dev/null || echo 0)
    if [ "$s2" -gt "$s1" ]; then
      echo
      echo "A previous run is still writing to $prev ($s1 -> $s2 bytes in 10 s)."
      echo "Restarting the container now would kill it. Wait for it, or:"
      echo "  $DOCKER_CLI restart $CONTAINER   # kills it on purpose"
      echo "  bash release/$SCRIPT_NAME $N --force   # same thing, automatic"
      [ "$FORCE" = 1 ] || exit 1
      echo "--force: killing it."
    fi
  fi
fi

# ---------------------------------------------------------------------------
# Container lifecycle
# ---------------------------------------------------------------------------
if "${DOCKER[@]}" container inspect "$CONTAINER" >/dev/null 2>&1; then
  if [ "$NO_RESTART" = 1 ]; then
    "${DOCKER[@]}" start "$CONTAINER" >/dev/null 2>&1 || true
    echo "   container $CONTAINER reused (--no-restart)"
  else
    # Ray workers leak ~85 GB/card after a clean smoke and the processes are
    # invisible from inside the container; only a restart gives it back.
    "${DOCKER[@]}" restart "$CONTAINER" >/dev/null
    echo "   container $CONTAINER restarted"
  fi
else
  "${DOCKER[@]}" run -d --name "$CONTAINER" \
    --network=host --ipc=host \
    --device=/dev/kfd --device=/dev/dri --group-add=video \
    --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --shm-size 64G \
    -v "$DATA_ROOT":"$DATA_ROOT" -e DATA_ROOT="$DATA_ROOT" \
    "$IMAGE" sleep infinity >/dev/null
  echo "   container $CONTAINER created"
fi
sleep 5

# ---------------------------------------------------------------------------
# GPU idle check (after the restart, so we measure the state we will run in)
# ---------------------------------------------------------------------------
vram="$("${DOCKER[@]}" exec "$CONTAINER" bash -lc \
  'rocm-smi --showmeminfo vram 2>/dev/null | grep -i "Total Used Memory"' || true)"
if [ -z "$vram" ]; then
  echo "   WARNING: could not read VRAM usage (rocm-smi returned nothing)."
else
  busy="$(awk -v lim="$IDLE_VRAM_BYTES" '{v=$NF+0; if (v>lim) print}' <<<"$vram" | wc -l)"
  maxb="$(awk '{v=$NF+0; if (v>m) m=v} END {print m+0}' <<<"$vram")"
  printf '   GPU idle check: %s/8 cards busy, peak %.1f GB in use\n' \
    "$busy" "$(python3 -c "print($maxb/1e9)")"
  if [ "$busy" -gt 0 ]; then
    echo
    echo "$vram" | sed 's/^/     /'
    echo "Cards are not at the ~298 MB idle baseline. Most likely causes:"
    echo "  a) another container is training:   $DOCKER_CLI ps -a"
    echo "  b) an orphan from a killed run:     $DOCKER_CLI restart $CONTAINER"
    echo "  c) another tenant on this node — nothing you can do from here."
    [ "$FORCE" = 1 ] || exit 1
    echo "--force: continuing anyway."
  fi
fi

# ---------------------------------------------------------------------------
# ATOM compile cache hygiene (examples 4 <-> 5)
# ---------------------------------------------------------------------------
STATE="$DATA_ROOT/logs/.atom-precision-$CONTAINER"
if [ "$MODE" = atomfp8 ] || [ "$MODE" = atombf16 ]; then
  prev_atom="$(cat "$STATE" 2>/dev/null || true)"
  if [ "$KEEP_CACHE" != 1 ] && [ -n "$prev_atom" ] && [ "$prev_atom" != "$MODE" ]; then
    echo "   ATOM precision changed ($prev_atom -> $MODE): clearing compile caches"
    "${DOCKER[@]}" exec "$CONTAINER" bash -lc \
      'rm -rf /tmp/aiter_configs /tmp/atom_torch_compile_cache /tmp/torchinductor_root'
  fi
  echo "$MODE" > "$STATE"
fi

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
: > "$LOG"; : > "$OUTER_LOG"
echo "$LOG" > "$DATA_ROOT/logs/.running-$CONTAINER"

echo "   log: $LOG"
START=$(date +%s)

if [ "$DETACH" = 1 ]; then
  "${DOCKER[@]}" exec -d "${DOCKER_ENV[@]}" "$CONTAINER" bash -lc "$INNER"
  cat <<EOF

Started detached. It is alive as long as the log keeps growing
(pgrep is useless here: 'docker exec' sessions do not share a process tree
with your shell, so it returns 0 whatever the truth is).

  watch -n 30 'ls -l $LOG'                      # alive if the size grows
  tail -f $LOG                                  # follow
  grep -m1 '$DONE_MARK' $LOG                    # appears when it finishes
  grep -oE 'step=[0-9]+ .*' $LOG | tail -1      # latest metrics
  bash release/$SCRIPT_NAME $N --check-only --log $LOG

Do NOT run this launcher again before it finishes: it restarts the container,
which kills this run.
EOF
  exit 0
fi

# Foreground: run_dapo.sh writes the trainer output to $LOG, not to the exec's
# stdout, so follow the file. Done with a plain poll loop rather than `tail -F`
# on purpose: a backgrounded tail survives this script and keeps the pipe open,
# which hangs any non-interactive wrapper (`spur exec`, CI) forever.
HIGHLIGHT='RLTrainer\.setup|filter_groups round|^ *step=|Traceback|OutOfMemory|CUDA error|HSA_STATUS'

"${DOCKER[@]}" exec -d "${DOCKER_ENV[@]}" "$CONTAINER" bash -lc "$INNER"

STALL_LIMIT="${STALL_LIMIT:-2400}"   # seconds without a new log line before we give up
printed=0
last_change=$(date +%s)
while :; do
  total="$(wc -l < "$LOG" 2>/dev/null || echo 0)"
  if [ "$total" -gt "$printed" ]; then
    last_change=$(date +%s)
    if [ "$VERBOSE" = 1 ]; then
      sed -n "$((printed + 1)),${total}p" "$LOG"
    else
      sed -n "$((printed + 1)),${total}p" "$LOG" \
        | grep -E "$HIGHLIGHT" | cut -c1-200 || true
    fi
    printed="$total"
  fi
  grep -q "$DONE_MARK" "$LOG" 2>/dev/null && break
  if [ $(( $(date +%s) - last_change )) -gt "$STALL_LIMIT" ]; then
    echo
    echo "ERROR: no new log line for ${STALL_LIMIT}s. Giving up on following it."
    echo "  trainer log : $LOG"
    echo "  wrapper log : $OUTER_LOG"
    [ -s "$OUTER_LOG" ] && { echo "  --- last lines of the wrapper log ---"; tail -n 20 "$OUTER_LOG" | sed 's/^/  /'; }
    echo "  The run may still be alive inside the container; check with"
    echo "    $DOCKER_CLI exec $CONTAINER bash -lc 'ps -ef | grep lumenrl.trainer.main'"
    exit 1
  fi
  sleep 5
done

EXIT_CODE="$(sed -n "s/.*$DONE_MARK\([0-9]*\) ===.*/\1/p" "$LOG" | tail -n 1)"
ELAPSED=$(( $(date +%s) - START ))

echo
echo "== example $N finished: exit=$EXIT_CODE, wall clock ${ELAPSED}s ($((ELAPSED/60))m$((ELAPSED%60))s)"
echo "   full log     : $LOG"
echo "   launcher log : $OUTER_LOG"
echo "   metrics      : grep -o 'step=[0-9]* .*rollout_corr/kl=[^ ]*' $LOG"
echo "   errors       : grep -nE 'Traceback|OutOfMemory|CUDA error|HSA_STATUS' $LOG"
echo "   free the VRAM before the next run: $DOCKER_CLI restart $CONTAINER"

rc=0
if [ "$DO_CHECK" = 1 ]; then
  check_log "$LOG" || rc=1
fi
[ "${EXIT_CODE:-1}" = 0 ] || rc=1
exit $rc
