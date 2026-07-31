#!/bin/bash
# Regenerate the 4-panel figure each time the run advances STEP_EVERY steps.
# Runs inside the training image (needs matplotlib); launched as a detached
# named container so it survives the ssh session and can be stopped by name.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

LOG="${LOG:-/home/jimguo12/train_ref36k.log}"
OUT="${OUT:-/home/jimguo12/progress.png}"
STEP_EVERY="${STEP_EVERY:-250}"
POLL="${POLL:-300}"
ARCHIVE="${ARCHIVE:-/home/jimguo12/progress_history}"
PLOT="${PLOT:-$HERE/plot_progress.py}"

mkdir -p "$ARCHIVE"
last_plotted=-1

while true; do
  step=$(grep -oE 'callbacks: step=[0-9]+' "$LOG" 2>/dev/null | tail -1 | grep -oE '[0-9]+$')
  if [ -n "${step:-}" ] && [ "$step" -ge $((last_plotted + STEP_EVERY)) ]; then
    if python3 "$PLOT" "$OUT" "$LOG" > /home/jimguo12/.plot_last.txt 2>&1; then
      cp "$OUT" "$ARCHIVE/step_$(printf '%06d' "$step").png"
      last_plotted=$step
      echo "[$(date -u +%T)] plotted step=$step"
    fi
  fi
  sleep "$POLL"
done
