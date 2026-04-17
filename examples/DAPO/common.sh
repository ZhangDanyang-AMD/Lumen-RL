#!/usr/bin/env bash
# Common DAPO configuration for LumenRL FP8 training alignment experiments.
# Adapted from Lumen/examples/rl/verl/dapo/common.sh for the LumenRL framework.
#
# Source this file from individual run scripts.

set -euo pipefail

# ─── Paths ────────────────────────────────────────────────────────────────────
MODEL_PATH="${MODEL_PATH:?MODEL_PATH must be set}"
TRAIN_FILE="${TRAIN_FILE:-/dev/shm/data/dapo-math-17k.parquet}"
TEST_FILE="${TEST_FILE:-/dev/shm/data/aime-2024.parquet}"
CKPTS_DIR="${CKPTS_DIR:-/root/ckpts/${PROJECT_NAME}/${EXP_NAME}}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/output/DAPO}"

# ─── Prerequisite validation ─────────────────────────────────────────────────
_preflight_errors=0
if [ ! -d "${MODEL_PATH}" ]; then
    echo "ERROR: MODEL_PATH does not exist: ${MODEL_PATH}" >&2
    _preflight_errors=1
fi
if [ ! -f "${TRAIN_FILE}" ]; then
    echo "ERROR: Training data not found: ${TRAIN_FILE}" >&2
    _preflight_errors=1
fi
if [ ! -f "${TEST_FILE}" ]; then
    echo "ERROR: Validation data not found: ${TEST_FILE}" >&2
    _preflight_errors=1
fi
if ! python3 -c "import torch" 2>/dev/null; then
    echo "ERROR: PyTorch not importable" >&2
    _preflight_errors=1
fi
if [ "${_preflight_errors}" -ne 0 ]; then
    echo "Aborting due to preflight errors." >&2
    exit 1
fi

# ─── Hardware ─────────────────────────────────────────────────────────────────
NUM_GPUS="${NUM_GPUS:-8}"
ROLLOUT_TP="${ROLLOUT_TP:-1}"

# ─── DAPO algorithm ──────────────────────────────────────────────────────────
ADV_ESTIMATOR="grpo"
CLIP_RATIO_LOW="${CLIP_RATIO_LOW:-0.2}"
CLIP_RATIO_HIGH="${CLIP_RATIO_HIGH:-0.28}"
LOSS_AGG_MODE="token-mean"

# ─── Data ─────────────────────────────────────────────────────────────────────
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-20480}"
TRAIN_BSZ="${TRAIN_BSZ:-32}"
GEN_BSZ="${GEN_BSZ:-96}"
MINI_BSZ="${MINI_BSZ:-${TRAIN_BSZ}}"
N_RESP="${N_RESP:-16}"

# ─── Overlong buffer ─────────────────────────────────────────────────────────
OVERLONG_BUFFER_ENABLE="${OVERLONG_BUFFER_ENABLE:-True}"
OVERLONG_BUFFER_LEN="${OVERLONG_BUFFER_LEN:-512}"
OVERLONG_PENALTY="${OVERLONG_PENALTY:-1.0}"

# ─── Rollout correction (TIS) ────────────────────────────────────────────────
ROLLOUT_IS="${ROLLOUT_IS:-null}"
ROLLOUT_IS_THRESHOLD="${ROLLOUT_IS_THRESHOLD:-2.0}"

# ─── FP8 rollout ─────────────────────────────────────────────────────────────
ROLLOUT_QUANTIZATION="${ROLLOUT_QUANTIZATION:-null}"

# ─── Lumen FP8 training ──────────────────────────────────────────────────────
export FP8_PARAM_MANAGER="${FP8_PARAM_MANAGER:-0}"
export LUMEN_FP8="${LUMEN_FP8:-0}"
export LUMEN_NORM="${LUMEN_NORM:-0}"
export LUMEN_FP8_ACTIVATION_STORE="${LUMEN_FP8_ACTIVATION_STORE:-0}"
export LUMEN_FP8_PARAM_GATHER="${LUMEN_FP8_PARAM_GATHER:-0}"

# ─── Lumen FP8 scaling (blockwise / MXFP8 / per-tensor) ─────────────────────
export LUMEN_FP8_SCALING="${LUMEN_FP8_SCALING:-delayed}"
export LUMEN_FP8_FORMAT="${LUMEN_FP8_FORMAT:-fp8_e4m3}"
export LUMEN_FP8_BLOCK_SIZE="${LUMEN_FP8_BLOCK_SIZE:-128}"

# ─── Lumen FP8 attention ─────────────────────────────────────────────────────
export LUMEN_FP8_ATTN="${LUMEN_FP8_ATTN:-none}"
export LUMEN_FP8_QUANT_TYPE="${LUMEN_FP8_QUANT_TYPE:-blockwise}"
export LUMEN_ATTN_KERNEL_BACKEND="${LUMEN_ATTN_KERNEL_BACKEND:-auto}"

# ─── Lumen fused kernel optimizations ────────────────────────────────────────
if [ "${LUMEN_FP8}" = "1" ]; then
    export LUMEN_FUSED_QUANT_AMAX="${LUMEN_FUSED_QUANT_AMAX:-1}"
    export LUMEN_FUSED_CAST_TRANSPOSE="${LUMEN_FUSED_CAST_TRANSPOSE:-1}"
    export LUMEN_FUSED_QUANT_SCALE="${LUMEN_FUSED_QUANT_SCALE:-1}"
    export LUMEN_TRANSPOSE_CACHE="${LUMEN_TRANSPOSE_CACHE:-1}"
else
    export LUMEN_FUSED_QUANT_AMAX="${LUMEN_FUSED_QUANT_AMAX:-0}"
    export LUMEN_FUSED_CAST_TRANSPOSE="${LUMEN_FUSED_CAST_TRANSPOSE:-0}"
    export LUMEN_FUSED_QUANT_SCALE="${LUMEN_FUSED_QUANT_SCALE:-0}"
fi

# ─── Performance ──────────────────────────────────────────────────────────────
OFFLOAD="${OFFLOAD:-true}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.5}"
SP_SIZE="${SP_SIZE:-1}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
TOTAL_STEPS="${TOTAL_STEPS:-275}"
TEST_FREQ="${TEST_FREQ:-5}"
SAVE_FREQ="${SAVE_FREQ:-20}"
PPO_MICRO_BSZ="${PPO_MICRO_BSZ:-1}"
LOG_PROB_MICRO_BSZ="${LOG_PROB_MICRO_BSZ:-1}"
USE_DYNAMIC_BSZ="${USE_DYNAMIC_BSZ:-True}"
PPO_MAX_TOKEN_LEN="${PPO_MAX_TOKEN_LEN:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}"
LOG_PROB_MAX_TOKEN_LEN="${LOG_PROB_MAX_TOKEN_LEN:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}"

# ─── Environment ──────────────────────────────────────────────────────────────
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export TORCHDYNAMO_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_USE_V1=1

# ─── Entry point selection ────────────────────────────────────────────────────
# Uses Lumen's VERL entry point when any Lumen feature is enabled,
# otherwise falls back to stock VERL.
VERL_ENTRY="verl.trainer.main_ppo"
if [ "${FP8_PARAM_MANAGER}" = "1" ] || [ "${LUMEN_FP8}" = "1" ] \
   || [ "${LUMEN_NORM}" = "1" ] || [ "${LUMEN_FP8_ACTIVATION_STORE}" = "1" ] \
   || [ "${LUMEN_FP8_PARAM_GATHER}" = "1" ]; then
    VERL_ENTRY="lumen.rl.verl.verl_entry"
fi

# ─── Patches ─────────────────────────────────────────────────────────────────
# Apply LumenRL/Lumen-specific patches to VERL and vLLM if present.
PATCHES_DIR="${SCRIPT_DIR}/patches"
if [ -d "${PATCHES_DIR}" ] && [ "$(ls -A "${PATCHES_DIR}" 2>/dev/null)" ]; then
    echo ">>> Applying LumenRL patches from ${PATCHES_DIR}"
    for patch_file in "${PATCHES_DIR}"/*.py; do
        [ -f "${patch_file}" ] || continue
        base="$(basename "${patch_file}")"
        case "${base}" in
            verl_fsdp_workers.py)
                target="$(python3 -c 'import verl.workers.fsdp_workers as m; print(m.__file__)')"
                ;;
            verl_ray_trainer.py)
                target="$(python3 -c 'import verl.trainer.ppo.ray_trainer as m; print(m.__file__)')"
                ;;
            verl_vllm_async_server.py)
                target="$(python3 -c 'import verl.workers.rollout.vllm_rollout.vllm_async_server as m; print(m.__file__)')"
                ;;
            verl_vllm_rollout.py)
                target="$(python3 -c 'import verl.workers.rollout.vllm_rollout as m; print(m.__file__)')"
                ;;
            vllm_v1_worker_gpu_worker.py)
                target="$(python3 -c 'import vllm.v1.worker.gpu_worker as m; print(m.__file__)')"
                ;;
            vllm_platforms_interface.py)
                target="$(python3 -c 'import vllm.platforms.interface as m; print(m.__file__)')"
                ;;
            *)
                echo "  [skip] Unknown patch: ${base}"
                continue
                ;;
        esac
        if [ -n "${target}" ]; then
            cp "${patch_file}" "${target}"
            echo "  [ok] ${base} → ${target}"
        fi
    done
fi

# ─── Banner ───────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  LumenRL DAPO — ${EXP_NAME}"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Model:           ${MODEL_PATH}"
echo "║  GPUs:            ${NUM_GPUS}"
echo "║  Train BSZ:       ${TRAIN_BSZ} × n=${N_RESP}"
echo "║  Rollout quant:   ${ROLLOUT_QUANTIZATION}"
echo "║  Rollout IS:      ${ROLLOUT_IS} (threshold=${ROLLOUT_IS_THRESHOLD})"
echo "║  FP8 PM:          ${FP8_PARAM_MANAGER}"
echo "║  LUMEN FP8:       ${LUMEN_FP8}"
echo "║  FP8 Scaling:     ${LUMEN_FP8_SCALING} / ${LUMEN_FP8_FORMAT} / block=${LUMEN_FP8_BLOCK_SIZE}"
echo "║  FP8 Attention:   ${LUMEN_FP8_ATTN} (quant=${LUMEN_FP8_QUANT_TYPE})"
echo "║  LUMEN Norm:      ${LUMEN_NORM}"
echo "║  FP8 Act Store:   ${LUMEN_FP8_ACTIVATION_STORE}"
echo "║  Entry:           ${VERL_ENTRY}"
echo "║  Steps:           ${TOTAL_STEPS}"
echo "║  Output:          ${OUTPUT_DIR}"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ─── Launch ───────────────────────────────────────────────────────────────────
launch_training() {
    local QUANT_OVERRIDE=""
    if [ "${ROLLOUT_QUANTIZATION}" != "null" ]; then
        QUANT_OVERRIDE="actor_rollout_ref.rollout.quantization=${ROLLOUT_QUANTIZATION}"
    fi

    mkdir -p "${OUTPUT_DIR}"

    python3 -m ${VERL_ENTRY} \
        data.train_files="${TRAIN_FILE}" \
        data.val_files="${TEST_FILE}" \
        data.prompt_key=prompt \
        data.truncation=left \
        data.return_raw_chat=True \
        data.filter_overlong_prompts=True \
        data.max_prompt_length=${MAX_PROMPT_LENGTH} \
        data.max_response_length=${MAX_RESPONSE_LENGTH} \
        data.train_batch_size=${TRAIN_BSZ} \
        +data.gen_batch_size=${GEN_BSZ} \
        \
        algorithm.adv_estimator=${ADV_ESTIMATOR} \
        algorithm.use_kl_in_reward=False \
        algorithm.kl_ctrl.kl_coef=0.0 \
        algorithm.rollout_correction.rollout_is=${ROLLOUT_IS} \
        algorithm.rollout_correction.rollout_is_threshold=${ROLLOUT_IS_THRESHOLD} \
        algorithm.rollout_correction.rollout_rs=null \
        algorithm.rollout_correction.rollout_rs_threshold=null \
        \
        actor_rollout_ref.actor.use_kl_loss=False \
        actor_rollout_ref.actor.kl_loss_coef=0.0 \
        actor_rollout_ref.actor.clip_ratio_low=${CLIP_RATIO_LOW} \
        actor_rollout_ref.actor.clip_ratio_high=${CLIP_RATIO_HIGH} \
        actor_rollout_ref.actor.clip_ratio_c=10.0 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BSZ} \
        actor_rollout_ref.actor.use_dynamic_bsz=${USE_DYNAMIC_BSZ} \
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN} \
        actor_rollout_ref.actor.ppo_mini_batch_size=${MINI_BSZ} \
        actor_rollout_ref.actor.loss_agg_mode=${LOSS_AGG_MODE} \
        actor_rollout_ref.actor.entropy_coeff=0 \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.actor.optim.weight_decay=0.1 \
        actor_rollout_ref.actor.optim.clip_grad=1.0 \
        actor_rollout_ref.actor.fsdp_config.param_offload=${OFFLOAD} \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=${OFFLOAD} \
        actor_rollout_ref.actor.ulysses_sequence_parallel_size=${SP_SIZE} \
        \
        actor_rollout_ref.model.path="${MODEL_PATH}" \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.mode=async \
        actor_rollout_ref.rollout.n=${N_RESP} \
        actor_rollout_ref.rollout.dtype=bfloat16 \
        actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEM_UTIL} \
        actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP} \
        actor_rollout_ref.rollout.enable_chunked_prefill=True \
        actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
        actor_rollout_ref.rollout.max_num_seqs=${MAX_NUM_SEQS} \
        actor_rollout_ref.rollout.temperature=1.0 \
        actor_rollout_ref.rollout.top_p=1.0 \
        actor_rollout_ref.rollout.top_k=-1 \
        actor_rollout_ref.rollout.calculate_log_probs=True \
        actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
        actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
        actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
        actor_rollout_ref.rollout.val_kwargs.do_sample=True \
        actor_rollout_ref.rollout.val_kwargs.n=1 \
        actor_rollout_ref.rollout.enforce_eager=True \
        actor_rollout_ref.rollout.free_cache_engine=${FREE_CACHE_ENGINE:-True} \
        \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BSZ} \
        actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${USE_DYNAMIC_BSZ} \
        actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${LOG_PROB_MAX_TOKEN_LEN} \
        actor_rollout_ref.ref.fsdp_config.param_offload=${OFFLOAD} \
        actor_rollout_ref.ref.ulysses_sequence_parallel_size=${SP_SIZE} \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BSZ} \
        actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${USE_DYNAMIC_BSZ} \
        actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${LOG_PROB_MAX_TOKEN_LEN} \
        \
        reward_model.reward_manager=dapo \
        +reward_model.reward_kwargs.overlong_buffer_cfg.enable=${OVERLONG_BUFFER_ENABLE} \
        +reward_model.reward_kwargs.overlong_buffer_cfg.len=${OVERLONG_BUFFER_LEN} \
        +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=${OVERLONG_PENALTY} \
        +reward_model.reward_kwargs.overlong_buffer_cfg.log=False \
        +reward_model.reward_kwargs.max_resp_len=${MAX_RESPONSE_LENGTH} \
        \
        trainer.logger='["console"]' \
        trainer.project_name="${PROJECT_NAME}" \
        trainer.experiment_name="${EXP_NAME}" \
        trainer.n_gpus_per_node=${NUM_GPUS} \
        trainer.nnodes=1 \
        trainer.val_before_train=False \
        trainer.test_freq=${TEST_FREQ} \
        trainer.save_freq=${SAVE_FREQ} \
        trainer.total_epochs=100 \
        trainer.total_training_steps=${TOTAL_STEPS} \
        trainer.default_local_dir="${CKPTS_DIR}" \
        trainer.resume_mode=auto \
        trainer.log_val_generations=1 \
        trainer.max_actor_ckpt_to_keep=2 \
        ${QUANT_OVERRIDE} \
        ${EXTRA_OVERRIDES:-} \
        2>&1 | tee "${OUTPUT_DIR}/${EXP_NAME}.log"
}
