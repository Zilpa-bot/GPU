"""
RunPod Serverless Handler for GPU Voice-Assistant
Compatible with RunPod's serverless platform requirements
"""

import runpod
import os, json, wave, uuid, logging, base64
import numpy as np, torch, soundfile as sf
import nemo.collections.asr as nemo_asr
from transformers import AutoTokenizer, AutoModelForCausalLM, \
                         AutoProcessor, CsmForConditionalGeneration

SAMPLE_RATE = 16_000
DEVICE = "cuda"

# Global models (loaded once at startup)
vad_model = None
asr_model = None
tok = None
llm = None
tts_proc = None
tts_mod = None

def load_models():
    """Load all models once at startup"""
    global vad_model, asr_model, tok, llm, tts_proc, tts_mod
    
    print("🔄 Loading models...")
    torch.cuda.set_device(0)
    
    # Load VAD model
    vad_model, _ = torch.hub.load('snakers4/silero-vad', 'silero_vad', force_reload=False)
    
    # Load ASR model
    asr_model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v2").to(DEVICE).eval()
    
    # Load LLM
    tok = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")
    if tok.pad_token_id is None: 
        tok.pad_token_id = tok.eos_token_id
    llm = AutoModelForCausalLM.from_pretrained("google/gemma-3-1b-it",
            torch_dtype=torch.float16, device_map="cuda").eval()
    
    # Load TTS
    tts_proc = AutoProcessor.from_pretrained("sesame/csm-1b")
    tts_mod = CsmForConditionalGeneration.from_pretrained("sesame/csm-1b",
            torch_dtype=torch.float16, device_map="cuda").eval()
    
    print("✅ All models loaded successfully")

def transcribe(audio: np.ndarray) -> str:
    """Transcribe audio to text using Parakeet-TDT"""
    tmp = f"/tmp/{uuid.uuid4()}.wav"
    sf.write(tmp, audio, SAMPLE_RATE, subtype='PCM_16')
    r = asr_model.transcribe([tmp])[0]
    os.remove(tmp)  # cleanup
    return r.text.strip() if hasattr(r, 'text') else str(r)

def generate(txt: str) -> str:
    """Generate response using Gemma-3-1B-IT"""
    prompt = f"User: {txt}\nAssistant:"
    inp = tok(prompt, return_tensors='pt').to(DEVICE)
    with torch.no_grad(), torch.cuda.amp.autocast():
        out = llm.generate(**inp, max_new_tokens=120, temperature=0.7)
    ans = tok.decode(out[0][inp['input_ids'].shape[1]:], skip_special_tokens=True)
    return ans.split("User:")[0].strip()

def synthesize(txt: str) -> bytes:
    """Synthesize text to speech using Sesame CSM-1B"""
    conv = [{"role": "0", "content": [{"type": "text", "text": txt}]}]
    inp = tts_proc.apply_chat_template(conv, tokenize=True, return_dict=True).to(DEVICE)
    with torch.no_grad(), torch.cuda.amp.autocast():
        audio = tts_mod.generate(**inp, output_audio=True)
    
    # Convert to bytes
    buffer = sf.BytesIO()
    sf.write(buffer, audio, SAMPLE_RATE, subtype='PCM_16')
    return buffer.getvalue()

def handler(event):
    """
    RunPod serverless handler function
    
    Expected input formats:
    1. Audio file (base64 encoded): {"audio": "base64_string", "format": "wav"}
    2. Text input: {"text": "Hello, how are you?"}
    3. Audio array: {"audio_array": [float32_values], "sample_rate": 16000}
    """
    try:
        # Parse input
        if "audio" in event["input"]:
            # Handle base64 encoded audio
            audio_b64 = event["input"]["audio"]
            audio_format = event["input"].get("format", "wav")
            
            # Decode base64 audio
            audio_bytes = base64.b64decode(audio_b64)
            
            # Load audio
            audio, sr = sf.read(sf.BytesIO(audio_bytes))
            
            # Resample if necessary
            if sr != SAMPLE_RATE:
                from scipy import signal
                audio = signal.resample(audio, int(len(audio) * SAMPLE_RATE / sr))
            
            # Ensure mono
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            
            # Transcribe
            text = transcribe(audio.astype(np.float32))
            
        elif "audio_array" in event["input"]:
            # Handle raw audio array
            audio = np.array(event["input"]["audio_array"], dtype=np.float32)
            sr = event["input"].get("sample_rate", SAMPLE_RATE)
            
            # Resample if necessary
            if sr != SAMPLE_RATE:
                from scipy import signal
                audio = signal.resample(audio, int(len(audio) * SAMPLE_RATE / sr))
            
            text = transcribe(audio)
            
        elif "text" in event["input"]:
            # Handle direct text input
            text = event["input"]["text"]
            
        else:
            return {"error": "Invalid input. Provide 'audio', 'audio_array', or 'text'"}
        
        if not text.strip():
            return {"error": "No speech detected or empty text provided"}
        
        # Generate response
        response_text = generate(text)
        
        # Synthesize audio
        audio_bytes = synthesize(response_text)
        audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
        
        return {
            "input_text": text,
            "response_text": response_text,
            "audio_base64": audio_b64,
            "sample_rate": SAMPLE_RATE,
            "format": "wav"
        }
        
    except Exception as e:
        return {"error": f"Processing failed: {str(e)}"}

# Load models when the handler starts
load_models()

# Set up RunPod
runpod.serverless.start({"handler": handler}) 