"""
RunPod Serverless Handler for GPU Voice-Assistant
  • STT:  Parakeet-TDT-0.6B-v2
  • LLM:  Gemma-3-1B-IT
  • TTS:  Sesame CSM-1B
"""
import base64, os, uuid, json
import runpod

import numpy as np, torch, soundfile as sf
import nemo.collections.asr as nemo_asr
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoProcessor,
    CsmForConditionalGeneration,
)

SAMPLE_RATE = 16_000
DEVICE      = "cuda"
HF_TOKEN    = os.getenv("HF_TOKEN")        # injected by RunPod

# -------------------------------------------------------------------
#  Model bootstrap (runs once per cold-start)
# -------------------------------------------------------------------
print("🔄 Loading models…")
torch.cuda.set_device(0)

vad_model, _ = torch.hub.load(
    "snakers4/silero-vad", "silero_vad", force_reload=False
)

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
print("✅ Models loaded")

# -------------------------------------------------------------------
#  Helper functions
# -------------------------------------------------------------------
def transcribe(audio: np.ndarray) -> str:
    """Parakeet-TDT transcription"""
    tmp = f"/tmp/{uuid.uuid4()}.wav"
    sf.write(tmp, audio, SAMPLE_RATE, subtype="PCM_16")
    out = asr_model.transcribe([tmp])[0]
    os.remove(tmp)
    return out.text.strip() if hasattr(out, "text") else str(out)

def generate(text: str) -> str:
    """Gemma-3-1B-IT response"""
    prompt = f"User: {text}\nAssistant:"
    inp    = tok(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad(), torch.cuda.amp.autocast():
        out = llm.generate(**inp, max_new_tokens=120, temperature=0.7)
    ans = tok.decode(out[0][inp["input_ids"].shape[1]:],
                     skip_special_tokens=True)
    return ans.split("User:")[0].strip()

def synthesize(text: str) -> bytes:
    """Sesame CSM-1B speech synthesis → WAV bytes"""
    conv = [{"role": "0",
             "content": [{"type": "text", "text": text}]}]
    inp  = tts_proc.apply_chat_template(conv, tokenize=True,
                                        return_dict=True).to(DEVICE)
    with torch.no_grad(), torch.cuda.amp.autocast():
        audio = tts_mod.generate(**inp, output_audio=True)
    buf = sf.BytesIO()
    sf.write(buf, audio, SAMPLE_RATE, subtype="PCM_16")
    return buf.getvalue()

# -------------------------------------------------------------------
#  RunPod handler
# -------------------------------------------------------------------
def handler(event):
    """
    Accepted inputs:
      • {"audio": "<b64-wav>", "format": "wav"}
      • {"audio_array": [floats], "sample_rate": 16000}
      • {"text": "hello"}
    Returns JSON with text + base64 wav.
    """
    try:
        inp = event["input"]

        # ----- audio (base64) -----
        if "audio" in inp:
            data   = base64.b64decode(inp["audio"])
            audio, sr = sf.read(sf.BytesIO(data))
            if sr != SAMPLE_RATE:
                from scipy import signal
                audio = signal.resample(audio,
                        int(len(audio) * SAMPLE_RATE / sr))
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            user_text = transcribe(audio.astype(np.float32))

        # ----- audio array -----
        elif "audio_array" in inp:
            audio = np.asarray(inp["audio_array"], dtype=np.float32)
            sr    = inp.get("sample_rate", SAMPLE_RATE)
            if sr != SAMPLE_RATE:
                from scipy import signal
                audio = signal.resample(audio,
                        int(len(audio) * SAMPLE_RATE / sr))
            user_text = transcribe(audio)

        # ----- direct text -----
        elif "text" in inp:
            user_text = str(inp["text"])

        else:
            return {"error":
                    "Provide 'audio', 'audio_array', or 'text' in input"}

        if not user_text.strip():
            return {"error": "No speech detected / empty text."}

        assistant_text = generate(user_text)
        wav_bytes      = synthesize(assistant_text)
        wav_b64        = base64.b64encode(wav_bytes).decode()

        return {
            "input_text":     user_text,
            "response_text":  assistant_text,
            "audio_base64":   wav_b64,
            "sample_rate":    SAMPLE_RATE,
            "format":         "wav"
        }

    except Exception as e:
        return {"error": f"Processing failed: {e}"}

# Register with RunPod
runpod.serverless.start({"handler": handler})
