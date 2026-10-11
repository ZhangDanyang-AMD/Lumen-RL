"""Upload AITER held-out dataset to HuggingFace."""
import json
from pathlib import Path
from huggingface_hub import HfApi, create_repo

DATASET_DIR = Path("/home/danyzhan/held-out-benchmark-aiter")
REPO_ID = "Zhangdanyang/agent-phase1-held-out-aiter"

api = HfApi()
info = api.whoami()
print("Logged in as:", info.get("name", "unknown"))

create_repo(REPO_ID, repo_type="dataset", private=True, exist_ok=True)
print("Repo ready:", REPO_ID)

print("Uploading dataset...")
api.upload_folder(
    folder_path=str(DATASET_DIR),
    repo_id=REPO_ID,
    repo_type="dataset",
    commit_message="AITER-baseline held-out dataset: 96/120 tasks with AITER kernel baselines, provenance metadata",
)
print("Done:", "https://huggingface.co/datasets/" + REPO_ID)
