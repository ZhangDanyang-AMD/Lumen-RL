#!/usr/bin/env bash
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HOME=/mnt/m2m_nobackup/jimguo12/hf_home
export HF_HUB_DOWNLOAD_TIMEOUT=60
DEST=/mnt/m2m_nobackup/jimguo12/models/Kimi-K3
mkdir -p "$DEST"
for attempt in $(seq 1 200); do
  echo "===== attempt $attempt $(date -Is) ====="
  /home/jimguo12/hfvenv/bin/hf download moonshotai/Kimi-K3 \
     --local-dir "$DEST" --max-workers 16 && { echo "DOWNLOAD_COMPLETE"; break; }
  echo "retry in 30s"; sleep 30
done
