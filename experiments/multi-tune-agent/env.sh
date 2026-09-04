#!/usr/bin/env bash

# Source this file; do not execute it. Paths remain portable after cloning.
export LUMEN_CODE_ROOT="$(
    cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd
)"
export LUMEN_CODE_CONFIG="${LUMEN_CODE_CONFIG:-$LUMEN_CODE_ROOT/configs/mi300x.yaml}"
export LUMEN_CODE_SESSION="${LUMEN_CODE_SESSION:-multi-tune-cli}"
export LUMEN_CODE_CONTAINER="${LUMEN_CODE_CONTAINER:-geak-phase1-vllm}"
export GEAK_CONTAINER_NAME="${GEAK_CONTAINER_NAME:-$LUMEN_CODE_CONTAINER}"
export GEAK_HOME="${GEAK_HOME:-/home/danyzhan/GEAK}"
export LUMEN_CODE_GEAK_ROOT="${LUMEN_CODE_GEAK_ROOT:-$GEAK_HOME}"
export LUMEN_CODE_SERVER_HOST="${LUMEN_CODE_SERVER_HOST:-10.194.134.84}"
export LUMEN_CODE_VLLM_HOST="${LUMEN_CODE_VLLM_HOST:-127.0.0.1}"
export LUMEN_CODE_VLLM_PORT="${LUMEN_CODE_VLLM_PORT:-8000}"
export LUMEN_CODE_TUNNEL_PORT="${LUMEN_CODE_TUNNEL_PORT:-18000}"
export LUMEN_CODE_BASE_URL="${LUMEN_CODE_BASE_URL:-http://$LUMEN_CODE_VLLM_HOST:$LUMEN_CODE_VLLM_PORT/v1}"
export LUMEN_CODE_REMOTE_BASE_URL="${LUMEN_CODE_REMOTE_BASE_URL:-http://$LUMEN_CODE_SERVER_HOST:$LUMEN_CODE_VLLM_PORT/v1}"

_lumen_code_prepend_pythonpath() {
    case ":${PYTHONPATH:-}:" in
        *":$1:"*) ;;
        *) export PYTHONPATH="$1${PYTHONPATH:+:$PYTHONPATH}" ;;
    esac
}
_lumen_code_prepend_pythonpath "$LUMEN_CODE_ROOT"
_lumen_code_prepend_pythonpath "$LUMEN_CODE_ROOT/src"
unset -f _lumen_code_prepend_pythonpath

_lumen_code_python() {
    if [ -x "$LUMEN_CODE_ROOT/.venv/bin/python" ] \
        && "$LUMEN_CODE_ROOT/.venv/bin/python" -c \
            'import geak_utils, requests, yaml' >/dev/null 2>&1; then
        printf '%s\n' "$LUMEN_CODE_ROOT/.venv/bin/python"
        return
    fi
    if "${PYTHON_BIN:-python3}" -c \
        'import geak_utils, requests, yaml' >/dev/null 2>&1; then
        printf '%s\n' "${PYTHON_BIN:-python3}"
        return
    fi
    echo "lumen-code: no Python environment provides geak_utils, requests, and PyYAML." >&2
    echo "Install the project dependencies described in $LUMEN_CODE_ROOT/README.md." >&2
    return 1
}

_lumen_code_cli_loop() {
    local python_bin choice rc
    python_bin="$(_lumen_code_python)" || return 1
    while true; do
        "$python_bin" -m multi_tune_agent.cli \
            --config "$LUMEN_CODE_CONFIG" interactive
        rc=$?
        printf '\nLumen Code CLI exited with code %s.\n' "$rc"
        printf 'Choose [r] restart CLI or [q] close tmux session: '
        if ! read -r choice; then
            return "$rc"
        fi
        case "${choice,,}" in
            r|restart|"") ;;
            q|quit|exit) return "$rc" ;;
            *) echo "Please enter r or q." ;;
        esac
    done
}

lumen-code() {
    local source_command tmux_command
    for executable in docker tmux; do
        if ! command -v "$executable" >/dev/null 2>&1; then
            echo "lumen-code: missing required command: $executable" >&2
            return 1
        fi
    done
    if ! docker inspect "$LUMEN_CODE_CONTAINER" >/dev/null 2>&1; then
        echo "lumen-code: Docker container '$LUMEN_CODE_CONTAINER' does not exist." >&2
        echo "Create it using the command in $LUMEN_CODE_ROOT/README.md." >&2
        return 1
    fi
    if [ "$(docker inspect -f '{{.State.Running}}' "$LUMEN_CODE_CONTAINER")" != "true" ]; then
        echo "Starting $LUMEN_CODE_CONTAINER..."
        docker start "$LUMEN_CODE_CONTAINER" >/dev/null || return 1
    fi

    printf -v source_command 'source %q; _lumen_code_cli_loop' \
        "$LUMEN_CODE_ROOT/env.sh"
    printf -v tmux_command 'exec bash -lc %q' "$source_command"

    if ! tmux has-session -t "$LUMEN_CODE_SESSION" 2>/dev/null; then
        tmux new-session -d -s "$LUMEN_CODE_SESSION" \
            -c "$LUMEN_CODE_ROOT" "$tmux_command" || return 1
    fi
    tmux set-option -g mouse on
    if [ -n "${TMUX:-}" ]; then
        tmux switch-client -t "$LUMEN_CODE_SESSION"
    else
        tmux attach-session -t "$LUMEN_CODE_SESSION"
    fi
}

lumen-code-status() {
    docker ps -a --filter "name=^/${LUMEN_CODE_CONTAINER}$" \
        --format '{{.Names}}  {{.Image}}  {{.Status}}'
    echo "local endpoint   $LUMEN_CODE_BASE_URL"
    echo "remote endpoint  $LUMEN_CODE_REMOTE_BASE_URL"
    echo "SSH tunnel port  $LUMEN_CODE_TUNNEL_PORT"
    if tmux has-session -t "$LUMEN_CODE_SESSION" 2>/dev/null; then
        echo "$LUMEN_CODE_SESSION  tmux session running"
    else
        echo "$LUMEN_CODE_SESSION  tmux session stopped"
    fi
}

lumen-code-stop() {
    if tmux has-session -t "$LUMEN_CODE_SESSION" 2>/dev/null; then
        tmux kill-session -t "$LUMEN_CODE_SESSION"
        echo "Stopped Lumen Code tmux session: $LUMEN_CODE_SESSION"
    else
        echo "Lumen Code tmux session is already stopped."
    fi
}

kill-lumen-code() {
    lumen-code-stop
}

if [[ $- == *i* ]]; then
    echo "Lumen Code environment loaded. Run: lumen-code"
    echo "Stop the CLI with: kill-lumen-code"
fi
