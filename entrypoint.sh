#!/usr/bin/env bash
set -e

echo "🚀 Voice-assistant container starting…"
if [[ -z "${HF_TOKEN}" ]]; then
  echo "⚠️  HF_TOKEN not set – gated models may fail to download."
else
  echo "🔑 HF_TOKEN detected – downloading/caching models."
fi

python - <<'PY'
import os
from huggingface_hub import snapshot_download

token = os.getenv("HF_TOKEN")
models = [
    "nvidia/parakeet-tdt-0.6b-v2",
    "google/gemma-3-1b-it",
    "sesame/csm-1b",
]

for repo in models:
    print(f"⬇  Ensuring local cache for {repo}")
    snapshot_download(
        repo_id=repo,
        token=token,
        local_dir=f"/opt/models/{repo.replace('/','_')}",
        local_dir_use_symlinks=False,
        revision="main",
    )
print("✅ Model cache ready")
PY

exec python /app/handler.py
