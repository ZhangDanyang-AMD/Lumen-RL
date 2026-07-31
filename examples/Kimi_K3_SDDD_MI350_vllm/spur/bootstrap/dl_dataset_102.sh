#!/usr/bin/env bash
# Runs INSIDE the docker container: download kimi-mtp-dataset into the docker-host
# /dev/shm (see HANDOFF 1.2 three-container-layers trap) and split into phase1/phase2.
set -x
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export HF_HOME=/mnt/m2m_nobackup/jimguo12/hf_home
export HF_HUB_DOWNLOAD_TIMEOUT=60
HF=/home/jimguo12/hfvenv/bin/hf
for attempt in $(seq 1 50); do
  echo "===== dataset attempt $attempt $(date -Is) ====="
  "$HF" download lightseekorg/kimi-mtp-dataset --repo-type dataset \
      --local-dir /dev/shm/kimi-mtp-dataset --max-workers 8 && break
  echo "retry in 20s"; sleep 20
done
ls -la /dev/shm/kimi-mtp-dataset/data/
python3 "$HERE/../../split_dataset.py"
wc -l /dev/shm/kimi-mtp-dataset-phase1/train.jsonl /dev/shm/kimi-mtp-dataset-phase2/train.jsonl
echo "DATASET_COMPLETE"
