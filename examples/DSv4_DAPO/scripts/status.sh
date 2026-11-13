#!/usr/bin/env bash
# Report on a run and grade it against the four success criteria in the README.
#
#   bash scripts/status.sh            # the latest run
#   bash scripts/status.sh <RUN_ID>
#
# Verdict: PASS, FAIL, or INCONCLUSIVE when everything that was observable passed
# but no step carried a training signal (see criterion 4 in the README).
# shellcheck source=lib.sh
source "$(dirname "$0")/lib.sh"
load_env
RUN_ID=${1:-$(cat "$LOG_DIR/latest" 2>/dev/null || true)}
[ -n "$RUN_ID" ] || die "no run yet under $LOG_DIR"
LOG=$LOG_DIR/$RUN_ID.log
if [ ! -f "$LOG" ]; then
  log "$RUN_ID: no training log yet; launcher output:"
  tail -5 "$LOG.launch" 2>/dev/null || true
  exit 0
fi

steps_want="" nodes="" cfg="" max_resp=""
while IFS= read -r line; do
  case "$line" in
    STEPS=*) steps_want=${line#STEPS=} ;;
    CONFIG_OVERRIDE=*) cfg=${line#CONFIG_OVERRIDE=} ;;
    EXTRA_OVERRIDE=*) [[ $line =~ cluster\.num_nodes=([0-9]+) ]] && nodes=${BASH_REMATCH[1]} ;;
  esac
done <"$LOG.cmd"
# The response-length cap of the run's config (policy.max_response_length).
if [ -n "$cfg" ] && [ -f "$RL_ROOT/Lumen-RL/$cfg" ]; then
  while IFS= read -r line; do
    [[ $line =~ ^[[:space:]]+max_response_length:[[:space:]]*([0-9]+) ]] && { max_resp=${BASH_REMATCH[1]}; break; }
  done <"$RL_ROOT/Lumen-RL/$cfg"
fi
max_resp=${max_resp:-4096}

# Value of key=<value> in a line, or "".
field() { [[ $2 =~ (^|[[:space:]])$1=([^[:space:]]+) ]] && echo "${BASH_REMATCH[2]}"; }
# A decimal or e-notation number in units of 1e-6, as an integer.
micro() { local s; s=$(printf '%.6f' "$1" 2>/dev/null) || { echo -1; return; }; echo $((10#${s%.*} * 1000000 + 10#${s#*.})); }

fin="" tb=0 rep="" finished=0 kls="" nsteps=0 signal=0 inconsistent=""
echo "run      $RUN_ID"
echo "log      $LOG ($(wc -l <"$LOG") lines, last write $(date -u -r "$LOG" +%H:%M:%SZ))"
if [ -f "$LOG.exit" ]; then echo "exit     $(cat "$LOG.exit")"; else echo "exit     (still running)"; fi
echo "steps    step grad_norm loss rollout_corr/kl mean_response_len actor_max_reserved_gb step_s"
while IFS= read -r line; do
  case "$line" in
    *Traceback*) tb=$((tb + 1)) ;;
    *"LumenRL finished"*) finished=1 ;;
    *"finished after "*" steps"*) [[ $line =~ finished\ after\ ([0-9]+)\ steps ]] && fin=${BASH_REMATCH[1]} ;;
    *"rollout ready: "*" replicas"*) [[ $line =~ ready:\ ([0-9]+)\ (colocated\ )?replicas ]] && rep=${BASH_REMATCH[1]} ;;
    *"callbacks: step="*)
      s=$(field step "$line") g=$(field grad_norm "$line") k=$(field rollout_corr/kl "$line")
      len=$(field seq/mean_response_len "$line")
      echo "         $s $g $(field loss "$line") $k $len $(field mem/actor_max_reserved_gb "$line") $(field timing/step_s "$line")"
      nsteps=$((nsteps + 1))
      [ -n "$k" ] && kls+="$k "
      # A step has a training signal iff some response stopped before the length
      # cap: on this slice every reward is -1, so only the overlong penalty varies.
      has_signal=0; [ -n "$len" ] && [ "$(micro "$len")" -lt "$(micro "$max_resp")" ] && has_signal=1
      has_grad=0; [ -n "$g" ] && [ "$(micro "$g")" -gt 0 ] && has_grad=1
      [ "$has_signal" = 1 ] && signal=$((signal + 1))
      [ "$has_signal" = "$has_grad" ] || inconsistent+="$s " ;;
  esac
done <"$LOG"

kl_ok=0
if [ -n "$kls" ]; then
  kl_ok=1
  for k in $kls; do v=$(micro "$k"); { [ "$v" -ge 2000 ] && [ "$v" -le 4000 ]; } || kl_ok=0; done
fi
exit_code=$(cat "$LOG.exit" 2>/dev/null || echo "")
pass() { if "$@"; then echo PASS; else echo "--"; fi; }
c1() { [ -n "$fin" ] && [ "$fin" = "$steps_want" ] && [ "$finished" = 1 ] && [ "$exit_code" = 0 ]; }
c2() { [ "$tb" = 0 ]; }
c3() { [ -n "$rep" ] && [ "$rep" = "$nodes" ]; }
c4() { [ "$kl_ok" = 1 ] && [ "$nsteps" -gt 0 ] && [ -z "$inconsistent" ]; }
echo "criteria"
echo "  1 finished after STEPS steps, exit 0      $(pass c1)  (finished after ${fin:-?}, STEPS=${steps_want:-?}, exit ${exit_code:-?})"
echo "  2 no Traceback                            $(pass c2)  ($tb)"
echo "  3 rollout replicas == nodes               $(pass c3)  (${rep:-?} vs ${nodes:-?})"
echo "  4 kl in [2.0e-3, 4.0e-3]; grad_norm>0 exactly on steps with a response < $max_resp tokens"
echo "                                            $(pass c4)  (kl: ${kls:-none}; steps with a training signal: $signal/$nsteps${inconsistent:+; inconsistent at step(s) $inconsistent})"
if c1 && c2 && c3 && c4; then
  if [ "$signal" -gt 0 ]; then
    echo "verdict  PASS"
  else
    echo "verdict  INCONCLUSIVE: every response of every step hit $max_resp tokens, so no step had a gradient to check."
    echo "         Expected now and then (about one group per step stops early); rerun, or STEPS=6 bash scripts/run.sh."
  fi
elif [ -f "$LOG.exit" ]; then
  echo "verdict  FAIL"
else
  echo "verdict  (running)"
fi
