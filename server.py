"""
GPU Voice-Assistant server (Linux, RunPod)
  • Receives raw 16 kHz float32 PCM frames over WebSocket `/stream`
  • Uses silero-vad to segment, Parakeet-TDT 0.6b-v2 for STT,
    Gemma-3-1B-IT for LLM reply, Sesame CSM-1B for TTS
  • Returns JSON: {"text": "...", "wav": <bytes>}  (16 kHz, mono)
"""

import asyncio, os, json, wave, uuid, logging
import numpy as np, torch, websockets, soundfile as sf
import nemo.collections.asr as nemo_asr
from transformers import AutoTokenizer, AutoModelForCausalLM, \
                         AutoProcessor, CsmForConditionalGeneration

SAMPLE_RATE = 16_000
DEVICE      = "cuda"

# --------- Model boot ---------- #
torch.cuda.set_device(0)
vad_model,_ = torch.hub.load('snakers4/silero-vad','silero_vad',force_reload=False)
asr_model   = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v2").to(DEVICE).eval()

tok  = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")
if tok.pad_token_id is None: tok.pad_token_id = tok.eos_token_id
llm  = AutoModelForCausalLM.from_pretrained("google/gemma-3-1b-it",
        torch_dtype=torch.float16, device_map="cuda").eval()

tts_proc = AutoProcessor.from_pretrained("sesame/csm-1b")
tts_mod  = CsmForConditionalGeneration.from_pretrained("sesame/csm-1b",
        torch_dtype=torch.float16, device_map="cuda").eval()
# -------------------------------- #

async def handle(ws):
    """Stream endpoint: receive audio frames → reply wav"""
    buf, speak = [], False
    while True:
        frame = await ws.recv()
        if isinstance(frame, str): continue  # ignore text pings
        pcm  = np.frombuffer(frame, dtype=np.float32)
        level= np.sqrt(np.mean(pcm**2))
        vad  = vad_model(torch.tensor(pcm), SAMPLE_RATE).item() if level>5e-4 else 0.
        if vad>0.01:
            speak = True
            buf.append(pcm)
        elif speak:            # end-of-speech
            audio = np.concatenate(buf)
            text  = transcribe(audio)
            resp  = generate(text)
            wav   = synthesize(resp)
            await ws.send(json.dumps({"text": resp}))
            await ws.send(wav)
            speak, buf = False, []
        else:
            pass  # waiting for speech

def transcribe(audio:np.ndarray)->str:
    tmp = f"/tmp/{uuid.uuid4()}.wav"
    sf.write(tmp, audio, SAMPLE_RATE, subtype='PCM_16')
    r   = asr_model.transcribe([tmp])[0]
    return r.text.strip() if hasattr(r,'text') else str(r)

def generate(txt:str)->str:
    prompt = f"User: {txt}\nAssistant:"
    inp    = tok(prompt, return_tensors='pt').to(DEVICE)
    with torch.no_grad(), torch.cuda.amp.autocast():
        out = llm.generate(**inp, max_new_tokens=120, temperature=0.7)
    ans = tok.decode(out[0][inp['input_ids'].shape[1]:], skip_special_tokens=True)
    return ans.split("User:")[0].strip()

def synthesize(txt:str)->bytes:
    conv = [{"role":"0","content":[{"type":"text","text":txt}]}]
    inp  = tts_proc.apply_chat_template(conv, tokenize=True, return_dict=True).to(DEVICE)
    with torch.no_grad(), torch.cuda.amp.autocast():
        audio = tts_mod.generate(**inp, output_audio=True)
    out_wav = sf.write(sf.BytesIO(), audio, SAMPLE_RATE, subtype='PCM_16')
    return out_wav.getvalue()

async def main():
    async with websockets.serve(handle, "0.0.0.0", 8000, max_size=2**20):
        print("🛰  ready on :8000")
        await asyncio.Future()

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    asyncio.run(main())
