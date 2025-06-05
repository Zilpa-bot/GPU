# syntax=docker/dockerfile:1
############################################################
# GPU Voice-Assistant – Runtime-only model downloads       #
############################################################
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ENV TORCH_COMPILE_DISABLE=1 \
    TORCH_DYNAMO_DISABLE=1 \
    PYTHONUNBUFFERED=1 \
    TRANSFORMERS_CACHE=/opt/models \
    HF_HOME=/opt/models \
    HF_DATASETS_CACHE=/opt/datasets

# ---------- system packages ----------
RUN apt-get update && apt-get install -y --no-install-recommends \
        git ffmpeg wget curl ca-certificates \
        build-essential g++ \
        python3.11 python3.11-dev python3.11-venv python3-pip libasound2-dev \
    && ln -s /usr/bin/python3.11 /usr/local/bin/python \
    && ln -s /usr/bin/python3.11 /usr/bin/python \
    && python3.11 -m pip install --upgrade pip \
    && rm -rf /var/lib/apt/lists/*

# ---------- Python libs ----------
RUN pip install --extra-index-url https://download.pytorch.org/whl/cu121 \
        torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 && \
    pip install \
        transformers==4.52.0 accelerate==1.7.0 \
        nemo_toolkit[asr]==2.3.1 silero-vad==5.1.2 \
        sounddevice==0.5.2 websockets==15.0.1 \
        numpy==1.26.4 scipy==1.13.0 \
        runpod

# ---------- application ----------
WORKDIR /app
COPY handler.py .
COPY server.py  .
COPY entrypoint.sh .
RUN chmod +x /app/entrypoint.sh

# ---------- runtime ----------
ENV SAMPLE_RATE=16000 PORT=8000
EXPOSE 8000
ENTRYPOINT ["/app/entrypoint.sh"]
