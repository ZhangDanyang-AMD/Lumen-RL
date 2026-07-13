#!/bin/bash
# Watchdog for gpt_oss_120b_eagle3_vllm_mi355 training
# Checks if training is still producing new steps; restarts if stalled.

CONTAINER="gpt_oss_120b_eagle3_vllm_mi355"
LAST_STEP_FILE="/tmp/.watchdog_last_step"
RUN_SCRIPT="/home/danyzhan/Lumen-RL/examples/GPT_OSS_120b_MI355_vllm/run_docker.sh"
LOG="/tmp/watchdog_gpt_oss.log"

# Get latest step from container logs
CURRENT_STEP=$(docker logs "$CONTAINER" 2>&1 | grep "callbacks: step=" | tail -1 | grep -oP 'step=\K[0-9]+')

if [ -z "$CURRENT_STEP" ]; then
    # Container might not exist or no steps yet
    RUNNING=$(docker ps --filter "name=$CONTAINER" --format "{{.Names}}" 2>/dev/null)
    if [ -z "$RUNNING" ]; then
        echo "$(date): Container not running, starting..." >> "$LOG"
        nohup bash "$RUN_SCRIPT" >> "$LOG" 2>&1 &
        echo "0" > "$LAST_STEP_FILE"
    fi
    exit 0
fi

# Read previous step
LAST_STEP=$(cat "$LAST_STEP_FILE" 2>/dev/null || echo "0")

if [ "$CURRENT_STEP" = "$LAST_STEP" ]; then
    # No progress — check if process is actually dead
    TAIL=$(docker logs "$CONTAINER" 2>&1 | tail -5)
    if echo "$TAIL" | grep -qE "EngineDeadError|Error|Traceback|shutting down"; then
        echo "$(date): Training crashed at step $CURRENT_STEP, restarting..." >> "$LOG"
        docker stop "$CONTAINER" 2>/dev/null
        sleep 5
        nohup bash "$RUN_SCRIPT" >> "$LOG" 2>&1 &
        echo "0" > "$LAST_STEP_FILE"
    else
        echo "$(date): Step $CURRENT_STEP unchanged but no crash detected, waiting..." >> "$LOG"
    fi
else
    echo "$(date): Training healthy at step $CURRENT_STEP (was $LAST_STEP)" >> "$LOG"
    echo "$CURRENT_STEP" > "$LAST_STEP_FILE"
fi
