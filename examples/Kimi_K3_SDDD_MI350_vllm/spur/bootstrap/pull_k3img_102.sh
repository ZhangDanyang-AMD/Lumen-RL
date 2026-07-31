#!/usr/bin/env bash
# Pull the K3 pre-release base image with retries (layers already fetched are reused).
export HOME=/home/jimguo12
export DOCKER_CONFIG=/home/jimguo12/.docker
IMG=vllm/vllm-openai-rocm:kimi-k3
for attempt in $(seq 1 60); do
  echo "===== pull attempt $attempt $(date -Is) ====="
  if docker pull "$IMG"; then
    echo "PULL_COMPLETE"
    break
  fi
  echo "retry in 20s"
  sleep 20
done
