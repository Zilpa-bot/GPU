"""
Standalone WebSocket server:
  • ws://HOST:8000/stream  – send raw 16-kHz float32 PCM frames
    (client keeps flushing frames; server streams back JSON + wav)
"""
import asyncio, json, uuid, os
import numpy as np, torch, soundfile as sf, websockets
import nemo.collections.asr as nemo_asr
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoProcessor,
    CsmForConditionalGeneration,
)

SAMPLE_RATE = 16_000
DEVICE      = "cuda"
HF_TOKEN    = os.getenv("HF_TOKEN")

# ---------- model bootstrap ----------
torch.cuda.set_device(0)
vad_model, _ = torch.hub.load("snakers4/silero-vad",
                              "silero_vad", force_reload=False)

asr_model = (
    nemo_asr.models.ASRModel
    .from_pretrained("nvidia/parakeet-tdt-0.6b-v2")
    .to(DEVICE)
    .eval()
)

tok = AutoTokenizer.from_pretrained("google/gemma-3-1b-it", token=HF_TOKEN)
if tok.pad_token_id is None:
    tok.pad_token_id = tok.eos_token_id

llm = (
    AutoModelForCausalLM
    .from_pretrained("google/gemma-3-1b-it",
                     torch_dtype=torch.float16,
                     device_map="cuda",
                     token=HF_TOKEN)
    .eval()
)

tts_proc = AutoProcessor.from_pretrained("sesame/csm-1b", token=HF_TOKEN)
tts_mod  = (
    CsmForConditionalGeneration
    .from_pretrained("sesame/csm-1b",
                     torch_dtype=torch.float16,
                     device_map="cuda",
                     token=HF_TOKEN)
    .eval()
)

# ---------- helpers ----------
def transcribe(audio: np.ndarray) -> str:
    tmp = f"/tmp/{uuid.uuid4()}.wav"
    sf.write(tmp, audio, SAMPLE_RATE, subtype="PCM_16")
    out = asr_model.transcribe([tmp])[0]
    os.remove(tmp)
    return out.text.strip() if hasattr(out, "text") else str(out)

def generate(text: str) -> str:
    prompt = f"User: {text}\nAssistant:"
    inp    = tok(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad(), torch.cuda.amp.autocast():
        out = llm.generate(**inp, max_new_tokens=120, temperature=0.7)
    ans = tok.decode(out[0][inp["input_ids"].shape[1]:],
                     skip_special_tokens=True)
    return ans.split("User:")[0].strip()

def synthesize(text: str) -> bytes:
    conv = [{"role": "0",
             "content": [{"type": "text", "text": text}]}]
    inp  = tts_proc.apply_chat_template(conv, tokenize=True,
                                        return_dict=True).to(DEVICE)
    with torch.no_grad(), torch.cuda.amp.autocast():
        audio = tts_mod.generate(**inp, output_audio=True)
    buf = sf.BytesIO()
    sf.write(buf, audio, SAMPLE_RATE, subtype="PCM_16")
    return buf.getvalue()

# ---------- websocket handler ----------
async def stream(ws):
    buf, speaking = [], False
    while True:
        msg = await ws.recv()
        if isinstance(msg, str):        # ignore ping-text
            continue
        pcm = np.frombuffer(msg, dtype=np.float32)
        rms = np.sqrt(np.mean(pcm ** 2))
        vad = vad_model(torch.tensor(pcm), SAMPLE_RATE).item() if rms > 5e-4 else 0.
        if vad > 0.01:                  # speech detected
            speaking = True
            buf.append(pcm)
        elif speaking:                  # end-of-speech
            audio = np.concatenate(buf)
            user  = transcribe(audio)
            resp  = generate(user)
            wav   = synthesize(resp)
            await ws.send(json.dumps({"text": resp}))
            await ws.send(wav)
            speaking, buf = False, []
        # else: silence – keep listening

async def main():
    async with websockets.serve(stream, "0.0.0.0", 8000, max_size=2**20):
        print("🛰  WebSocket server on :8000")
        await asyncio.Future()

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    asyncio.run(main())
