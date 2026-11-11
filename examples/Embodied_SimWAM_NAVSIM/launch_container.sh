#!/usr/bin/env bash
# Start the embodied MI350X container with the Lumen-RL tree mounted in place.
# Usage (host): bash examples/Embodied_SimWAM_NAVSIM/launch_container.sh
set -euo pipefail

IMAGE="${IMAGE:-zhangdanyangamd/lumen-rl:embodied-MI350X}"
NAME="${CONTAINER_NAME:-simwam-rl}"
ROOT="${ROOT:-/home/danyzhan}"
GPUS="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"

if docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
    docker rm -f "$NAME" >/dev/null
fi

docker run -d --init --name "$NAME" \
    --network host --ipc host --shm-size 256g \
    --device /dev/kfd --device /dev/dri \
    --group-add video --group-add render \
    --security-opt seccomp=unconfined --cap-add SYS_PTRACE \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -v "$ROOT:$ROOT" \
    -v /mnt/k3:/mnt/k3 \
    -e CUDA_VISIBLE_DEVICES="$GPUS" \
    -e HIP_VISIBLE_DEVICES="$GPUS" \
    -e LUMEN_DIR="$ROOT/Lumen" \
    -e AITER_SRC="$ROOT/aiter" \
    -e HF_HOME="$ROOT/.cache/huggingface" \
    -e MIOPEN_USER_DB_PATH="$ROOT/.cache/miopen" \
    -e MIOPEN_CUSTOM_CACHE_DIRECTORY="$ROOT/.cache/miopen" \
    -w "$ROOT/Lumen-RL" \
    --entrypoint "" \
    "$IMAGE" sleep infinity

mkdir -p "$ROOT/.cache/miopen"
# The image links /root/.cache/miopen at /work/cache/miopen, which this
# container does not mount. A real directory is required for torch Conv3d.
docker exec "$NAME" bash -lc 'if [ -L /root/.cache/miopen ]; then rm -f /root/.cache/miopen; fi; mkdir -p /root/.cache/miopen '"$ROOT"'/.cache/miopen'

docker inspect -f 'started {{.Name}} image={{.Config.Image}} gpus='"$GPUS" "$NAME"
