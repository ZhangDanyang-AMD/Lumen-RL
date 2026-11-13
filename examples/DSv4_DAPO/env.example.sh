#!/usr/bin/env bash
# The only file to edit on a new machine. Copy it to env.sh (git-ignored) in this
# directory and fill it in; every script under scripts/ reads env.sh.
#
#   cp env.example.sh env.sh && $EDITOR env.sh

# ---- required ------------------------------------------------------------------
# Working directory on a disk with enough space (see README). Models, logs, the
# patched Megatron, code checkouts and data all go under it unless overridden below.
export WORK_DIR="<absolute path on a disk with enough free space>"

# ---- optional (defaults shown) ---------------------------------------------------
# Layer count: 4 = the validated slice; 43 = the full model (in development).
# export DSV4_LAYERS=4
# Where aiter / Lumen / ATOM are checked out (and Lumen-RL, as a link to this clone).
# export RL_ROOT="$WORK_DIR/src"
# Where data_cached/ (training parquet) and hf_home/ live.
# export DATA_ROOT="$WORK_DIR"
# Name of the training container this example creates on every node.
# export CONTAINER=dsv4-dapo
# Training / rollout image, pinned by digest.
# export IMAGE=vllm/vime-rocm@sha256:3de38e0566effb9d97e14cd2412fa97467db96401776afb71fd424b09d9dfda9
# Weight-conversion image (FP8 -> BF16 and torch_dist), pinned by digest.
# export MILES_IMAGE=rocm/sgl-dev:miles-rocm720-mi35x-20260803@sha256:fd662fc0ef3dbe99fb49f53fa6b393a921f50248f68d370655ac6562dc7f1301
# Docker command, split on spaces, e.g. "sudo docker".
# export DOCKER=docker
# Docker daemon to talk to, if not the default socket.
# export DOCKER_HOST=<unix:///path/to/docker.sock>
# Interface used between nodes. Default: the interface of the default route.
# export NET_IF=<interface>
# IPv4 the Ray head listens on. Default: the IPv4 of NET_IF on this host.
# export HEAD_IP=<head IPv4>
# Space-separated hosts, head first; run scripts on the head. Default: this host only.
# export NODES="<head host> <worker host>"
# How to run a command on another node; {host} is replaced, the command arrives on
# stdin as `bash -s`. Replace with your scheduler's equivalent if there is no ssh.
# export REMOTE_EXEC="ssh {host}"
# Ray GCS port and dashboard port on the head.
# export RAY_PORT=6379
# export RAY_DASHBOARD_PORT=8265
# Hugging Face endpoint (mirror) and token; neither is needed for the public repos used here.
# export HF_ENDPOINT=https://huggingface.co
# export HF_TOKEN=<token>
# A DeepSeek-V4-Flash-Base directory downloaded beforehand (revision
# 8855555deef230a27a21a8d6f294b7b7497759b6); skips the download.
# export MODEL_SRC_DIR=<path>
# Training steps for scripts/run.sh.
# export STEPS=3
# Continue on hardware that differs from the validated 8 x gfx950.
# export FORCE=0
