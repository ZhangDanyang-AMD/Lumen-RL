#!/usr/bin/env bash
# HANDOFF 3.4 build. HOME/DOCKER_CONFIG must be set or buildx/build fails with
# `mkdir /opt/spur/.docker: permission denied` (HANDOFF 1.3).
export HOME=/home/jimguo12
export DOCKER_CONFIG=/home/jimguo12/.docker
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/../../../.." || exit 1
echo "===== build start $(date -Is) ====="
docker build -f examples/Kimi_K3_SDDD_MI350_vllm/Dockerfile.train.k3img \
    -t kimi_k3_dspark_k3img:latest .
rc=$?
echo "===== build end $(date -Is) rc=$rc ====="
[ $rc -eq 0 ] && echo "BUILD_COMPLETE"
