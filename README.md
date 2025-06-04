# GPU Voice Assistant - RunPod Serverless

A GPU-optimized voice assistant that combines:
- **Parakeet-TDT-0.6B-v2** for speech-to-text (ASR)
- **Gemma-3-1B-IT** for language modeling (LLM) 
- **Sesame CSM-1B** for text-to-speech (TTS)

## RunPod Deployment

### Prerequisites
1. A [RunPod account](https://runpod.io)
2. A [Hugging Face account](https://huggingface.co) with an access token
3. This repository pushed to GitHub

### Setup Steps

1. **Connect GitHub to RunPod:**
   - Go to [RunPod Settings → Connections](https://www.runpod.io/console/user/settings)
   - Click "Connect" under the GitHub card
   - Authorize RunPod to access your repositories

2. **Create Serverless Endpoint:**
   - Navigate to the [Serverless section](https://www.runpod.io/console/serverless)
   - Click "New Endpoint"
   - Select "GitHub Repo" as the source
   - Choose this repository from the dropdown
   - Configure deployment options:
     - **Branch:** `main` (or your preferred branch)
     - **Dockerfile:** `voice-server/Dockerfile`

3. **Configure Environment:**
   - Set your Hugging Face token as a build argument:
     ```
     HUGGINGFACE_HUB_TOKEN=your_token_here
     ```

4. **Select GPU Configuration:**
   - Recommended: RTX 4090 or better
   - Minimum VRAM: 16GB
   - Workers: 1-3 (based on expected load)

### API Usage

Once deployed, you can send requests to your endpoint:

#### Text Input
```bash
curl -X POST https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "input": {
      "text": "Hello, how are you today?"
    }
  }'
```

#### Audio Input (Base64)
```bash
curl -X POST https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "input": {
      "audio": "base64_encoded_wav_data",
      "format": "wav"
    }
  }'
```

#### Response Format
```json
{
  "id": "request_id",
  "status": "COMPLETED",
  "output": {
    "input_text": "transcribed or input text",
    "response_text": "AI generated response",
    "audio_base64": "base64_encoded_response_audio",
    "sample_rate": 16000,
    "format": "wav"
  }
}
```

### Local Development

For local testing with the WebSocket server:

```bash
# Build the container
docker build --build-arg HUGGINGFACE_HUB_TOKEN=your_token \
  -t voice-server .

# Run locally  
docker run --gpus all -p 8000:8000 voice-server
```

Connect to `ws://localhost:8000/stream` for real-time audio streaming.

### Model Details

- **ASR Model:** NVIDIA Parakeet-TDT-0.6B-v2 (English STT)
- **LLM Model:** Google Gemma-3-1B-IT (Instruction-tuned chat)
- **TTS Model:** Sesame CSM-1B (English voice synthesis)
- **Audio Format:** 16kHz mono PCM
- **GPU Memory:** ~14GB VRAM required

### Troubleshooting

- **Build timeouts:** Optimize Dockerfile layers or use smaller models
- **CUDA errors:** Ensure GPU-enabled RunPod instances
- **Model download failures:** Check Hugging Face token permissions
- **Audio format issues:** Ensure 16kHz mono PCM input 