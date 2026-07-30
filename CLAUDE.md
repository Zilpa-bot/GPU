# CLAUDE.md

Guidance for AI assistants (and humans) working in this repository.

## What this is

A **GPU-optimized voice assistant** packaged for **RunPod Serverless**. It chains three
models into a single speech-in → speech-out pipeline:

| Stage | Model | Purpose |
|-------|-------|---------|
| STT (ASR) | `nvidia/parakeet-tdt-0.6b-v2` | Speech → text |
| LLM | `google/gemma-3-1b-it` | Text → chat response |
| TTS | `sesame/csm-1b` | Response text → speech (WAV) |

All models are loaded on GPU (`cuda`, device 0) in `float16` and require ~14 GB VRAM.
`google/gemma-3-1b-it` and `sesame/csm-1b` are **gated** on Hugging Face and need an
`HF_TOKEN` with access granted.

## Repository layout

The entire codebase is six files at the repo root — there is no package structure, no
tests, and no build tooling beyond Docker.

```
Dockerfile      CUDA 12.1 + Python 3.11 image; installs deps, copies the 3 scripts
entrypoint.sh   Container entrypoint: pre-downloads model snapshots, then launches handler.py
handler.py      RunPod Serverless handler (the production entrypoint)
server.py       Standalone WebSocket streaming server (local dev / real-time alternative)
get-docker.sh   Upstream Docker convenience installer (vendored; do not edit)
README.md       Deployment + API usage docs
```

## The two entrypoints

`handler.py` and `server.py` share **identical model bootstrap and the three helper
functions** (`transcribe`, `generate`, `synthesize`). They differ only in transport:

- **`handler.py`** — the deployed path. Registers a `handler(event)` with
  `runpod.serverless.start(...)`. Accepts one request via `event["input"]` with one of:
  `{"audio": "<b64-wav>", "format": "wav"}`, `{"audio_array": [...], "sample_rate": N}`,
  or `{"text": "..."}`. Returns `{input_text, response_text, audio_base64, sample_rate,
  format}`. Errors are returned as `{"error": "..."}` (it never raises out of the handler).
- **`server.py`** — a `websockets` server on `:8000/stream` for real-time streaming. The
  client pushes raw 16 kHz float32 PCM frames; silero-VAD gates speech, and on
  end-of-speech the server runs the pipeline and sends back a JSON `{text}` message
  followed by WAV bytes.

**When editing the pipeline, keep the two files in sync.** The `transcribe`/`generate`/
`synthesize` helpers and the model-loading block are duplicated verbatim; a change to one
almost always belongs in the other.

## Key conventions

- **Audio format is 16 kHz mono PCM everywhere** (`SAMPLE_RATE = 16_000`). Inputs at
  other rates are resampled via `scipy.signal.resample`; multi-channel is averaged to mono.
- **Models load once at cold start** (module import time), not per request. Keep heavy
  init at module scope so RunPod reuses it across warm invocations.
- **`HF_TOKEN`** is read from the environment (`os.getenv("HF_TOKEN")`) and passed to every
  `from_pretrained(..., token=HF_TOKEN)` call and to `snapshot_download`. Never hardcode a
  token; never commit one.
- **Prompt format** for the LLM is `f"User: {text}\nAssistant:"`, and the response is
  truncated at the next `"User:"` to stop the model from continuing the dialogue.
- Generation params: `max_new_tokens=120`, `temperature=0.7`, under
  `torch.no_grad()` + `torch.cuda.amp.autocast()`.
- Inference disables torch.compile / dynamo (`TORCH_COMPILE_DISABLE=1`,
  `TORCH_DYNAMO_DISABLE=1` in the Dockerfile) — these models are run eager-mode.

## Build & run

Everything runs inside the Docker image; there is no local Python venv workflow.

```bash
# Build (HF_TOKEN needed at runtime, not build time — models download on first start)
docker build -t voice-server .

# Run the RunPod handler locally (needs an NVIDIA GPU)
docker run --gpus all -e HF_TOKEN=hf_xxx voice-server

# Run the WebSocket server instead (override the entrypoint's final command)
docker run --gpus all -e HF_TOKEN=hf_xxx -p 8000:8000 voice-server python /app/server.py
# then connect a client to ws://localhost:8000/stream
```

`entrypoint.sh` pre-caches all three model snapshots into `/opt/models` before exec'ing
`python /app/handler.py`. Model caches live under `/opt/models` (`HF_HOME`,
`TRANSFORMERS_CACHE`); datasets under `/opt/datasets`.

## Deployment (RunPod Serverless)

Deployed from GitHub: RunPod builds this repo's `Dockerfile` and runs the container as a
serverless endpoint. Set `HUGGINGFACE_HUB_TOKEN` / `HF_TOKEN` in the endpoint config so
gated models download. Recommended GPU: RTX 4090+ with ≥16 GB VRAM. See `README.md` for
step-by-step console instructions and `curl` examples against
`https://api.runpod.ai/v2/<ENDPOINT_ID>/runsync`.

## Dependency versions (pinned in the Dockerfile)

`torch==2.7.1` (cu121), `torchaudio==2.7.1`, `transformers==4.52.0`, `accelerate==1.7.0`,
`nemo_toolkit[asr]==2.3.1`, `silero-vad==5.1.2`, `numpy==1.26.4`, `scipy==1.13.0`,
plus `runpod`. Base image: `nvidia/cuda:12.1.1-runtime-ubuntu22.04`, Python 3.11.
`CsmForConditionalGeneration` requires a recent `transformers`, so avoid downgrading it.

## Gotchas to know before editing

- **`soundfile` (`import soundfile as sf`) is used by both scripts but is not explicitly
  pinned in the Dockerfile's `pip install`** — it currently arrives transitively. If you
  touch dependencies, add it explicitly rather than relying on that.
- **`README.md` says the Dockerfile lives at `voice-server/Dockerfile`**, but it is at the
  repo root. If you reorganize, update the README (and RunPod's configured path) to match.
- The handler swallows all exceptions into `{"error": ...}` — when debugging a failing
  request, check the RunPod worker logs, not the HTTP response alone.
- `get-docker.sh` is the vendored upstream Docker installer; it is not part of the app —
  don't modify it.

## Working in this repo

- Keep changes minimal and match the existing terse, comment-lite style of the scripts.
- There is no test suite or linter configured; validate changes by building the image and
  exercising the handler / WebSocket path against a real GPU.
- Develop on the designated feature branch, commit with clear messages, and push to that
  branch (do not open a PR unless explicitly asked).
