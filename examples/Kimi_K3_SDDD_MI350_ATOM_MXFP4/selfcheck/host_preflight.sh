#!/usr/bin/env bash
# Fail before a 20-minute K3 load when a host cannot run one full TP=8 teacher.
set -euo pipefail

: "${TRAIN_IMAGE:?Set TRAIN_IMAGE}"
: "${DATA_ROOT:?Set DATA_ROOT}"
: "${MODEL:?Set MODEL}"
: "${DATASET_PATH:?Set DATASET_PATH}"
: "${EVAL_FIXED_PATH:?Set EVAL_FIXED_PATH}"

command -v docker >/dev/null
command -v rocm-smi >/dev/null
[[ -e /dev/kfd && -d /dev/dri ]]
[[ -d "${MODEL}" ]]
[[ -f "${DATASET_PATH}" ]]
[[ -f "${EVAL_FIXED_PATH}" ]]
docker image inspect "${TRAIN_IMAGE}" >/dev/null

model_real="$(realpath "${MODEL}")"
data_root_real="$(realpath "${DATA_ROOT}")"
if [[ "${model_real}" != "${data_root_real}"/* ]]; then
    echo "Kimi-K3 must be loaded from node-local DATA_ROOT" \
        "(${data_root_real}), got ${model_real}" >&2
    exit 2
fi
python3 - "${model_real}" <<'PY'
import json
import sys
from pathlib import Path

model = Path(sys.argv[1])
revision = "a590ce090cb049c93a33dfe8c208ec652aa20503"
expected_tensor_bytes = 1_560_860_324_864
expected_file_bytes = 1_560_936_091_448

index = json.loads(
    (model / "model.safetensors.index.json").read_text(encoding="utf-8")
)
shards = sorted(set(index["weight_map"].values()))
missing = [name for name in shards if not (model / name).is_file()]
actual_file_bytes = sum((model / name).stat().st_size for name in shards)
if (
    len(shards) != 96
    or missing
    or index["metadata"]["total_size"] != expected_tensor_bytes
    or actual_file_bytes != expected_file_bytes
):
    raise SystemExit(
        "invalid local Kimi-K3 checkpoint: "
        f"shards={len(shards)}, missing={missing}, "
        f"tensor_bytes={index['metadata']['total_size']}, "
        f"file_bytes={actual_file_bytes}"
    )

marker_path = model / ".kimi_k3_download_complete.json"
if marker_path.is_file():
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    actual_revision = marker.get("revision")
else:
    metadata_path = model / ".cache/huggingface/download/config.json.metadata"
    if not metadata_path.is_file():
        raise SystemExit("missing Kimi-K3 revision metadata")
    actual_revision = metadata_path.read_text(encoding="utf-8").splitlines()[0]
if actual_revision != revision:
    raise SystemExit(f"Kimi-K3 revision {actual_revision} != {revision}")
print(
    f"Local Kimi-K3 verified: revision={revision}, "
    f"shards={len(shards)}, bytes={actual_file_bytes}"
)
PY

gpu_count=0
for gpu_id_file in /sys/class/kfd/kfd/topology/nodes/*/gpu_id; do
    [[ -f "${gpu_id_file}" ]] || continue
    read -r gpu_id <"${gpu_id_file}"
    (( gpu_id != 0 )) && (( gpu_count += 1 ))
done
if (( gpu_count != 8 )); then
    echo "Expected 8 AMD GPUs, found ${gpu_count}" >&2
    exit 2
fi

if [[ "$(cat /proc/sys/kernel/numa_balancing)" != "0" ]]; then
    echo "NUMA auto-balancing must be disabled on the host" >&2
    exit 2
fi

shopt -s nullglob
kfd_processes=(/sys/class/kfd/kfd/proc/[0-9]*)
if (( ${#kfd_processes[@]} )); then
    echo "GPU processes already own /dev/kfd:" >&2
    printf '  %s\n' "${kfd_processes[@]##*/}" >&2
    exit 2
fi

product_info="$(rocm-smi --showproductname 2>&1)"
gfx950_count="$(
    awk '/GFX Version:.*gfx950/ { count += 1 } END { print count + 0 }' \
        <<<"${product_info}"
)"
if (( gfx950_count != 8 )); then
    echo "Expected 8 gfx950 GPUs, found ${gfx950_count}" >&2
    exit 2
fi
echo "Host preflight passed on $(hostname): gpus=8 arch=gfx950"
