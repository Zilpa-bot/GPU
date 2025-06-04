# syntax=docker/dockerfile:1

############################################
# GPU-optimised voice-assistant  • RunPod  #
#   – Parakeet-TDT-0.6B-v2  (ASR)          #
#   – Gemma-3-1B-IT        (LLM)           #
#   – Sesame CSM-1B         (TTS)          #
############################################

# ===== Base =====
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# ----- TOKEN INJECTION -----
# (build-time ARG; value copied into an env-var the hub lib will read)
ARG HUGGINGFACE_HUB_TOKEN
ENV HUGGINGFACE_HUB_TOKEN=${HUGGINGFACE_HUB_TOKEN}
# ---------------------------

ARG DEBIAN_FRONTEND=noninteractive
ENV TORCH_COMPILE_DISABLE=1 \
    TORCH_DYNAMO_DISABLE=1 \
    PYTHONUNBUFFERED=1 \
    # path caches so model files survive layer-cache
    TRANSFORMERS_CACHE=/opt/models \
    HF_HOME=/opt/models \
    HF_DATASETS_CACHE=/opt/datasets

# ===== OS deps =====
RUN apt-get update && apt-get install -y --no-install-recommends \
        git ffmpeg wget curl ca-certificates \
        python3.11 python3.11-venv python3-pip \
    && ln -s /usr/bin/python3.11 /usr/local/bin/python \
    && ln -s /usr/bin/python3.11 /usr/bin/python \
    && python3.11 -m pip install --upgrade pip \
    && rm -rf /var/lib/apt/lists/*

# ===== Python libs (latest stable 4 Jun 2025) =====
RUN pip install --extra-index-url https://download.pytorch.org/whl/cu121 \
        torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 && \
    pip install \
        transformers==4.52.4 accelerate==1.7.0 \
        nemo_toolkit[asr]==2.3.1 silero-vad==5.1.2 \
        sounddevice==0.5.2 websockets==15.0.1 \
        numpy==1.26.4 scipy==1.13.0 \
        runpod

# ===== Pre-download HF models (best practice for RunPod) =====
RUN python - <<'PY'
from huggingface_hub import snapshot_download
models = [
    "nvidia/parakeet-tdt-0.6b-v2",
    "google/gemma-3-1b-it",
    "sesame/csm-1b",
]
for m in models:
    print(f"→ downloading {m}")
    snapshot_download(repo_id=m,
                      local_dir=f"/opt/models/{m.replace('/','_')}",
                      local_dir_use_symlinks=False,
                      revision="main")
PY

# ===== Copy application code =====
WORKDIR /app
COPY handler.py .
COPY server.py .

# ===== Runtime settings =====
ENV SAMPLE_RATE=16000 PORT=8000
EXPOSE 8000

CMD ["python", "handler.py"]
