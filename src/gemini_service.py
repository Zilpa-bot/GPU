"""
Gemini AI service for Hebrew conversation
Processes audio input and returns Hebrew text responses for butcher shop
"""

import os
import logging
import struct
import io
from typing import Optional, Tuple, List
import google.generativeai as genai
import numpy as np

logger = logging.getLogger(__name__)

# Model configuration
MODEL_ID = "models/gemini-2.5-flash-lite-preview-06-17"
TTS_MODEL_ID = "models/gemini-2.5-flash-preview-tts"

# Hebrew system prompt for butcher shop in Netanya
SYSTEM_PROMPT = (
    "אתה נציג קצבייה בנתניה. "
    "שלב 1: תעתיק במדויק את דברי הלקוח (שורה אחת). "
    "שלב 2: תן תשובה עניינית ורלוונטית רק לנושא הבשר. "
    "אם השאלה לא קשורה – תתנצל ותגיד שאינך יכול לעזור."
)

class GeminiService:
    """Hebrew conversation AI service using Gemini 2.5 Flash-Lite"""
    
    def __init__(self):
        self.client = None
        self._initialized = False
        
        # Try to initialize immediately
        self.initialize()
        
    def initialize(self) -> bool:
        """Initialize the Gemini service with google-generativeai library"""
        if self._initialized:
            return True
            
        try:
            # Get API key from environment
            api_key = os.getenv("GOOGLE_AI_API_KEY") or os.getenv("GEMINI_API_KEY")
            
            if not api_key:
                logger.error("GOOGLE_AI_API_KEY or GEMINI_API_KEY environment variable not set")
                return False
            
            # Configure the API key
            genai.configure(api_key=api_key)
            
            # Test the configuration by listing models (this validates the API key)
            models = list(genai.list_models())
            
            logger.info(f"✅ Initialized google-generativeai library with model: {MODEL_ID}")
            logger.debug(f"Available models: {len(models)}")
            self._initialized = True
            self.client = True  # Mark as initialized
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Gemini service: {e}")
            self.client = None
            return False
    
    def gemini_from_audio(self, pcm_bytes: bytes, mime_type: str = "audio/pcm") -> Optional[str]:
        """
        Process audio through Gemini and return Hebrew text response
        
        Args:
            pcm_bytes: Audio data in PCM16 format
            mime_type: Audio MIME type (default: audio/pcm)
            
        Returns:
            Hebrew text response containing transcript and answer, or None if error
        """
        if not self._initialized:
            logger.error("Gemini service not initialized")
            return None
            
        if not pcm_bytes:
            logger.warning("Empty audio data provided")
            return None
            
        try:
            logger.info(f"🎤 Processing {len(pcm_bytes)} bytes of audio with Gemini 2.5 Flash-Lite")
            
            # Convert PCM16 audio to WAV format for Gemini
            wav_audio = self._convert_pcm_to_wav(pcm_bytes)
            
            # Create audio part for Gemini
            audio_part = {
                "mime_type": "audio/wav",
                "data": wav_audio
            }
            
            # Create multimodal content with system prompt and audio
            content = [
                SYSTEM_PROMPT,
                audio_part
            ]
            
            # Generate content using google-generativeai with audio input
            model = genai.GenerativeModel(MODEL_ID)
            response = model.generate_content(content)
            
            if response and response.text:
                result = response.text.strip()
                logger.info(f"🤖 Gemini audio response: {len(result)} characters")
                logger.debug(f"Raw response: {result}")
                return result
            else:
                logger.warning("Empty response from Gemini")
                return None
                
        except Exception as e:
            logger.error(f"Error processing audio with Gemini: {e}")
            logger.error(f"Audio data length: {len(pcm_bytes)} bytes")
            return None
    
    def _convert_pcm_to_wav(self, pcm_data: bytes, sample_rate: int = 8000, channels: int = 1, bits_per_sample: int = 16) -> bytes:
        """
        Convert raw PCM16 audio data to WAV format
        
        Args:
            pcm_data: Raw PCM16 audio bytes
            sample_rate: Audio sample rate (default: 8000 Hz for Telnyx)
            channels: Number of audio channels (default: 1 for mono)
            bits_per_sample: Bits per sample (default: 16 for PCM16)
            
        Returns:
            WAV-formatted audio bytes
        """
        # Calculate audio parameters
        byte_rate = sample_rate * channels * bits_per_sample // 8
        block_align = channels * bits_per_sample // 8
        data_size = len(pcm_data)
        file_size = 36 + data_size
        
        # Create WAV header
        wav_header = struct.pack(
            '<4sI4s4sIHHIIHH4sI',
            b'RIFF',           # ChunkID
            file_size,         # ChunkSize
            b'WAVE',           # Format
            b'fmt ',           # Subchunk1ID
            16,                # Subchunk1Size (PCM = 16)
            1,                 # AudioFormat (PCM = 1)
            channels,          # NumChannels
            sample_rate,       # SampleRate
            byte_rate,         # ByteRate
            block_align,       # BlockAlign
            bits_per_sample,   # BitsPerSample
            b'data',           # Subchunk2ID
            data_size          # Subchunk2Size
        )
        
        # Combine header and audio data
        wav_data = wav_header + pcm_data
        
        logger.debug(f"Converted {len(pcm_data)} bytes PCM to {len(wav_data)} bytes WAV")
        return wav_data
    
    def hebrew_text_to_speech(self, text: str, voice_config: dict = None) -> Optional[bytes]:
        """
        Convert Hebrew text to speech using Gemini 2.5 Flash TTS
        
        Args:
            text: Hebrew text to synthesize
            voice_config: Optional voice configuration (gender, speed, etc.)
            
        Returns:
            Audio bytes in WAV format, or None if error
        """
        if not self._initialized:
            logger.error("Gemini service not initialized")
            return None
            
        if not text or not text.strip():
            logger.warning("Empty text provided for TTS")
            return None
            
        try:
            logger.info(f"🗣️ Converting Hebrew text to speech: {len(text)} characters")
            logger.debug(f"TTS text: {text[:100]}...")
            
            # Configure TTS model for audio output (not text)
            try:
                # Create TTS model with proper configuration
                model = genai.GenerativeModel(TTS_MODEL_ID)
                
                # Configure generation to expect audio response
                generation_config = {
                    "response_modalities": ["AUDIO"]
                }
                
                # Generate audio content with proper configuration
                response = model.generate_content(
                    text,
                    generation_config=generation_config
                )
                
                # Extract audio data from response - fixed extraction logic
                if response and hasattr(response, 'candidates') and response.candidates:
                    candidate = response.candidates[0]
                    logger.debug(f"Candidate type: {type(candidate)}")
                    
                    if hasattr(candidate, 'content') and candidate.content.parts:
                        logger.debug(f"Found {len(candidate.content.parts)} parts")
                        
                        for i, part in enumerate(candidate.content.parts):
                            logger.debug(f"Part {i} type: {type(part)}")
                            
                            # Check for inline_data with audio
                            if hasattr(part, 'inline_data') and part.inline_data:
                                # Get the actual data
                                if hasattr(part.inline_data, 'data'):
                                    audio_data = part.inline_data.data
                                    logger.info(f"✅ TTS generated {len(audio_data)} bytes of audio from part {i}")
                                    return audio_data
                                else:
                                    logger.debug(f"Part {i} inline_data has no data attribute")
                            else:
                                logger.debug(f"Part {i} has no inline_data")
                
                # Alternative: Check if response has parts directly
                if hasattr(response, 'parts'):
                    logger.debug(f"Response has {len(response.parts)} parts directly")
                    for i, part in enumerate(response.parts):
                        if hasattr(part, 'inline_data') and part.inline_data and hasattr(part.inline_data, 'data'):
                            audio_data = part.inline_data.data
                            logger.info(f"✅ TTS generated {len(audio_data)} bytes of audio from response part {i}")
                            return audio_data
                
                logger.warning("No audio data found in TTS response")
                logger.debug(f"Response type: {type(response)}")
                logger.debug(f"Response attributes: {dir(response)}")
                if hasattr(response, 'candidates') and response.candidates:
                    candidate = response.candidates[0]
                    logger.debug(f"Candidate attributes: {dir(candidate)}")
                    if hasattr(candidate, 'content'):
                        logger.debug(f"Content attributes: {dir(candidate.content)}")
                raise Exception("No audio data found in TTS response")
                    
            except Exception as tts_error:
                logger.warning(f"Gemini TTS failed: {tts_error}")
                
                # Fallback: Generate a simple notification that TTS was requested
                # In a real implementation, you might use alternative TTS services
                logger.info("TTS requested but not implemented - generating placeholder")
                
                # For now, return a simple acknowledgment that we would have generated audio
                # This allows the system to continue working while TTS is being fixed
                placeholder_text = f"[TTS PLACEHOLDER: '{text[:50]}...']"
                logger.info(f"TTS placeholder: {placeholder_text}")
                
                # Return None to indicate TTS failed but system should continue
                return None
                
        except Exception as e:
            logger.error(f"Error in Hebrew TTS: {e}")
            logger.error(f"Text length: {len(text)}")
            return None
    
    def convert_tts_to_telnyx_format(self, tts_audio: bytes, target_sample_rate: int = 8000) -> Optional[bytes]:
        """
        Convert TTS audio to Telnyx OPUS format (8kHz PCM16 little-endian for telephony)
        
        Args:
            tts_audio: Audio bytes from TTS (typically higher sample rate WAV)
            target_sample_rate: Target sample rate for Telnyx (default: 8000 for telephony)
            
        Returns:
            Audio bytes in PCM16 little-endian format for Telnyx WebSocket, or None if error
        """
        try:
            logger.info(f"🔧 Converting TTS audio to Telnyx OPUS format ({target_sample_rate}Hz): {len(tts_audio)} bytes")
            
            # Step 1: Extract PCM data from TTS audio (assuming WAV format)
            pcm_data = self._extract_pcm_from_audio(tts_audio)
            if not pcm_data:
                logger.error("Failed to extract PCM data from TTS audio")
                return None
            
            # Step 2: Ensure byte order is correct (convert BE to LE if needed)
            pcm_le = self._ensure_little_endian(pcm_data)
            if not pcm_le:
                logger.error("Failed to ensure little-endian format")
                return None
            
            # Step 3: Resample to target rate if needed (proper resampling, not naive downsampling)
            resampled_pcm = self._resample_audio_to_target_rate(pcm_le, target_sample_rate)
            if not resampled_pcm:
                logger.error(f"Failed to resample audio to {target_sample_rate}Hz")
                return None
            
            # Step 4: Normalize audio to prevent clipping (-6 dBFS peak)
            normalized_pcm = self._normalize_audio(resampled_pcm)
            if not normalized_pcm:
                logger.error("Failed to normalize audio")
                return None
            
            logger.info(f"✅ Converted to Telnyx OPUS PCM16 format: {len(normalized_pcm)} bytes")
            return normalized_pcm
            
        except Exception as e:
            logger.error(f"Error converting TTS audio format: {e}")
            return None
    
    def _extract_pcm_from_audio(self, audio_data: bytes) -> Optional[bytes]:
        """Extract PCM data from audio bytes (WAV or raw PCM)"""
        try:
            # Check if it's a WAV file
            if len(audio_data) > 44 and audio_data.startswith(b'RIFF'):
                # Parse WAV header to get correct offset
                # Standard WAV header is 44 bytes, but let's parse it properly
                if audio_data[12:16] == b'fmt ':
                    # Find the data chunk
                    data_pos = audio_data.find(b'data')
                    if data_pos > 0:
                        # Skip 'data' + 4 bytes for chunk size
                        pcm_start = data_pos + 8
                        pcm_data = audio_data[pcm_start:]
                        logger.debug(f"Extracted {len(pcm_data)} bytes PCM from WAV")
                        return pcm_data
                
                # Fallback: assume standard 44-byte header
                pcm_data = audio_data[44:]
                logger.debug(f"Extracted {len(pcm_data)} bytes PCM from WAV (fallback)")
                return pcm_data
            else:
                # Assume it's already raw PCM
                logger.debug(f"Using raw PCM data: {len(audio_data)} bytes")
                return audio_data
                
        except Exception as e:
            logger.error(f"Error extracting PCM from audio: {e}")
            return None
    
    def _resample_audio_to_target_rate(self, pcm_data: bytes, target_rate: int = 16000) -> Optional[bytes]:
        """Properly resample audio to target rate using interpolation"""
        try:
            # Convert bytes to 16-bit samples
            samples = struct.unpack(f'<{len(pcm_data)//2}h', pcm_data)
            sample_count = len(samples)
            
            # Estimate original sample rate based on data size
            # For a typical TTS response (few seconds), estimate the rate
            estimated_duration = max(1.0, sample_count / 24000)  # Assume 24kHz initially
            
            # Common TTS sample rates: 16kHz, 22kHz, 24kHz, 44.1kHz, 48kHz
            if sample_count > 48000 * 5:  # More than 5 seconds at 48kHz
                original_rate = 48000
            elif sample_count > 44100 * 5:  # More than 5 seconds at 44.1kHz
                original_rate = 44100
            elif sample_count > 24000 * 5:  # More than 5 seconds at 24kHz
                original_rate = 24000
            elif sample_count > 22050 * 5:  # More than 5 seconds at 22kHz
                original_rate = 22050
            elif sample_count > 16000 * 5:  # More than 5 seconds at 16kHz
                original_rate = 16000
            else:
                # Shorter audio or lower rate, assume 16kHz (common for TTS)
                original_rate = 16000
            
            logger.debug(f"Estimated original sample rate: {original_rate}Hz, samples: {sample_count}")
            
            # If already at target rate, return as-is
            if original_rate == target_rate:
                return pcm_data
            
            # Calculate resampling ratio
            ratio = original_rate / target_rate
            
            if ratio == 2.0:
                # Simple 2:1 downsampling with basic anti-aliasing
                # Apply a simple low-pass filter before downsampling
                filtered_samples = self._apply_simple_lowpass(samples)
                downsampled = filtered_samples[::2]  # Take every 2nd sample
            elif ratio == 3.0:
                # 3:1 downsampling
                filtered_samples = self._apply_simple_lowpass(samples)
                downsampled = filtered_samples[::3]  # Take every 3rd sample
            else:
                # Linear interpolation for other ratios
                downsampled = self._linear_resample(samples, original_rate, target_rate)
            
            # Convert back to bytes
            resampled_data = struct.pack(f'<{len(downsampled)}h', *downsampled)
            logger.debug(f"Resampled from {len(samples)} to {len(downsampled)} samples")
            return resampled_data
            
        except Exception as e:
            logger.error(f"Error resampling audio: {e}")
            return None
    
    def _ensure_little_endian(self, pcm_data: bytes) -> Optional[bytes]:
        """
        Ensure PCM data is in little-endian format
        Most TTS SDKs return big-endian even on Intel, so convert if needed
        """
        try:
            # More robust byte order detection as recommended in the guide
            samples_le = np.frombuffer(pcm_data, dtype='<i2')  # little-endian
            samples_be = np.frombuffer(pcm_data, dtype='>i2')  # big-endian
            
            # Check if one interpretation has much more reasonable values
            le_max = np.max(np.abs(samples_le))
            be_max = np.max(np.abs(samples_be))
            
            # If samples look like they exceed reasonable range, probably wrong byte order
            if le_max > 30000 and be_max < 10000:
                # LE interpretation shows clipping, BE shows reasonable levels
                logger.debug("Converting big-endian to little-endian (detected by amplitude)")
                return samples_be.byteswap().tobytes()
            elif be_max > 30000 and le_max < 10000:
                # BE interpretation shows clipping, LE shows reasonable levels  
                logger.debug("Audio already in little-endian format (detected by amplitude)")
                return pcm_data
            else:
                # Both seem reasonable, use standard heuristic
                if be_max < le_max * 0.5:  # BE interpretation has significantly smaller values
                    logger.debug("Converting big-endian to little-endian (standard heuristic)")
                    return samples_be.byteswap().tobytes()
                else:
                    logger.debug("Audio already in little-endian format (standard heuristic)")
                    return pcm_data
                
        except ImportError:
            # Fallback: explicit conversion as recommended in the guide
            logger.debug("NumPy not available, applying explicit BE→LE conversion")
            try:
                import struct
                samples = struct.unpack(f'>{len(pcm_data)//2}h', pcm_data)
                return struct.pack(f'<{len(samples)}h', *samples)
            except:
                logger.debug("Fallback conversion failed, using original")
                return pcm_data
        except Exception as e:
            logger.error(f"Error ensuring little-endian format: {e}")
            return pcm_data  # Return original on error
    
    def _normalize_audio(self, pcm_data: bytes) -> Optional[bytes]:
        """
        Normalize audio to -6 dBFS peak to prevent clipping and reduce noise
        """
        try:
            # Convert to samples
            samples = np.frombuffer(pcm_data, dtype='<i2')
            
            # Calculate current peak
            current_peak = np.max(np.abs(samples))
            
            if current_peak == 0:
                logger.warning("Audio contains only silence")
                return pcm_data
            
            # Clip to -6 dBFS as recommended in the guide (prevents distortion after companding)
            # -6 dBFS = 50% of int16 range = 16384 (true -6 dBFS)
            target_peak = 16384  # -6 dBFS for proper headroom
            
            # Always apply clipping to prevent over-hot peaks
            clipped_samples = np.clip(samples, -target_peak, target_peak)
            
            # If we had to clip, log it
            if np.any(np.abs(samples) > target_peak):
                clipped_count = np.sum(np.abs(samples) > target_peak)
                logger.debug(f"Clipped {clipped_count} samples to -6 dBFS (peak was {current_peak})")
            
            # Apply soft limiting for very hot audio
            if current_peak > target_peak * 1.2:  # If significantly over target
                # Calculate scaling factor for gentle compression
                scale_factor = target_peak / current_peak
                
                # Apply scaling to bring down the overall level
                scaled_samples = samples * scale_factor
                
                # Final hard limit
                result_samples = np.clip(scaled_samples, -target_peak, target_peak)
                
                logger.debug(f"Applied soft limiting: peak {current_peak} → {np.max(np.abs(result_samples))} (scale: {scale_factor:.3f})")
            else:
                result_samples = clipped_samples
            
            # Convert back to bytes
            result = result_samples.astype('<i2').tobytes()
            return result
                
        except ImportError:
            # Fallback without numpy - hard clipping as per guide
            logger.debug("NumPy not available, using hard clipping to -6 dBFS")
            samples = struct.unpack(f'<{len(pcm_data)//2}h', pcm_data)
            
            # Hard clipping to -6 dBFS (16384)
            clipped_samples = []
            for sample in samples:
                if sample > 16384:
                    clipped_samples.append(16384)
                elif sample < -16384:
                    clipped_samples.append(-16384)
                else:
                    clipped_samples.append(sample)
            
            return struct.pack(f'<{len(clipped_samples)}h', *clipped_samples)
            
        except Exception as e:
            logger.error(f"Error normalizing audio: {e}")
            return pcm_data  # Return original on error
    
    def _apply_simple_lowpass(self, samples):
        """Apply a simple low-pass filter to reduce aliasing"""
        if len(samples) < 3:
            return samples
        
        # Simple 3-point moving average filter
        filtered = []
        filtered.append(samples[0])  # First sample unchanged
        
        for i in range(1, len(samples) - 1):
            # Average of 3 consecutive samples
            avg = (samples[i-1] + samples[i] + samples[i+1]) // 3
            filtered.append(avg)
        
        filtered.append(samples[-1])  # Last sample unchanged
        return filtered
    
    def _linear_resample(self, samples, original_rate, target_rate):
        """Linear interpolation resampling"""
        ratio = original_rate / target_rate
        output_length = int(len(samples) / ratio)
        
        resampled = []
        for i in range(output_length):
            # Calculate position in original signal
            pos = i * ratio
            left_idx = int(pos)
            right_idx = min(left_idx + 1, len(samples) - 1)
            
            # Linear interpolation
            if left_idx == right_idx:
                value = samples[left_idx]
            else:
                fraction = pos - left_idx
                value = samples[left_idx] * (1 - fraction) + samples[right_idx] * fraction
            
            resampled.append(int(value))
        
        return resampled
    
    def _convert_pcm_to_ulaw(self, pcm_data: bytes) -> Optional[bytes]:
        """
        Convert 16-bit PCM to μ-law (PCMU) format
        
        Args:
            pcm_data: 16-bit PCM audio bytes
            
        Returns:
            μ-law encoded audio bytes
        """
        try:
            # Convert PCM bytes to 16-bit samples
            samples = struct.unpack(f'<{len(pcm_data)//2}h', pcm_data)
            
            ulaw_bytes = bytearray()
            
            for sample in samples:
                # Clamp to 16-bit range
                sample = max(-32768, min(32767, sample))
                
                # Convert to μ-law
                ulaw_byte = self._linear_to_ulaw(sample)
                ulaw_bytes.append(ulaw_byte)
            
            logger.debug(f"Converted {len(samples)} PCM samples to {len(ulaw_bytes)} μ-law bytes")
            return bytes(ulaw_bytes)
            
        except Exception as e:
            logger.error(f"Error converting PCM to μ-law: {e}")
            return None
    
    def _linear_to_ulaw(self, sample: int) -> int:
        """
        Convert a 16-bit linear PCM sample to μ-law
        
        Args:
            sample: 16-bit signed PCM sample (-32768 to 32767)
            
        Returns:
            μ-law encoded byte (0-255)
        """
        # μ-law compression constants
        BIAS = 0x84
        CLIP = 32635
        
        # Get sign and magnitude
        sign = 0 if sample >= 0 else 0x80
        if sample < 0:
            sample = -sample
        
        # Clip to maximum value
        if sample > CLIP:
            sample = CLIP
        
        # Add bias
        sample += BIAS
        
        # Find exponent
        exponent = 7
        for i in range(7):
            if sample <= (0x1F << (i + 3)):
                exponent = i
                break
        
        # Find mantissa
        mantissa = (sample >> (exponent + 3)) & 0x0F
        
        # Combine sign, exponent, and mantissa
        ulaw = sign | (exponent << 4) | mantissa
        
        # Invert all bits (μ-law standard)
        return (~ulaw) & 0xFF
    
    def chunk_audio_for_streaming(self, audio_data: bytes, chunk_duration_ms: int = 20, sample_rate: int = 8000) -> List[bytes]:
        """
        Chunk audio into frames for real-time streaming.
        Specs: split 640B frames → send 1/20s (50 FPS for 20ms frames)
        
        Args:
            audio_data: PCM16 audio bytes
            chunk_duration_ms: Frame duration in milliseconds (20ms standard)
            sample_rate: Audio sample rate (16kHz for TTS output, 8kHz for telephony)
            
        Returns:
            List of audio chunks ready for streaming
        """
        if not audio_data:
            logger.warning("No audio data to chunk")
            return []
        
        try:
            # Calculate frame size: 20ms at target sample rate
            if sample_rate == 16000:
                # 16kHz: 20ms = 320 samples × 2 bytes = 640 bytes (as per specs)
                frame_size = 640
            else:
                # 8kHz: 20ms = 160 samples × 2 bytes = 320 bytes
                frame_size = int(sample_rate * chunk_duration_ms / 1000) * 2
            
            chunks = []
            offset = 0
            
            while offset < len(audio_data):
                chunk = audio_data[offset:offset + frame_size]
                
                # Pad last chunk if necessary
                if len(chunk) < frame_size:
                    padding = frame_size - len(chunk)
                    chunk += b'\x00' * padding
                
                chunks.append(chunk)
                offset += frame_size
            
            logger.info(f"✅ Created {len(chunks)} × {frame_size}B frames ({chunk_duration_ms}ms each) at {sample_rate}Hz")
            return chunks
            
        except Exception as e:
            logger.error(f"Error chunking audio for streaming: {e}")
            return []

    def parse_response(self, text_block: str) -> Tuple[str, str]:
        """
        Parse Gemini response into transcript and answer
        
        Args:
            text_block: Full response from Gemini
            
        Returns:
            Tuple of (transcript, answer)
        """
        if not text_block:
            return "", ""
            
        lines = text_block.split("\n", 1)
        if len(lines) >= 2:
            transcript = lines[0].strip()
            answer = lines[1].strip()
        else:
            # If no clear separation, treat as answer only
            transcript = ""
            answer = text_block.strip()
            
        return transcript, answer
    
    def is_available(self) -> bool:
        """Check if Gemini service is available"""
        return self.client is not None and self._initialized


def normalize_audio_dbfs(audio_data: bytes, target_dbfs: float = -6.0) -> bytes:
    """
    Normalize audio to target dBFS level.
    Specs: normalise –6 dBFS (not -3 dBFS)
    
    Args:
        audio_data: PCM16 audio bytes
        target_dbfs: Target level in dBFS (e.g., -6.0)
        
    Returns:
        Normalized audio bytes
    """
    if not audio_data:
        return audio_data
    
    try:
        # Convert to numpy array
        samples = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
        
        # Calculate current RMS and dBFS
        rms = np.sqrt(np.mean(samples**2))
        if rms == 0:
            return audio_data  # Silent audio
        
        current_dbfs = 20 * np.log10(rms / 32767)
        
        # Calculate gain needed to reach target dBFS
        gain_db = target_dbfs - current_dbfs
        gain_linear = 10**(gain_db / 20)
        
        # Apply gain and clip to prevent overflow
        normalized = samples * gain_linear
        normalized = np.clip(normalized, -32767, 32767)
        
        logger.debug(f"🔧 Normalized audio: {current_dbfs:.1f} dBFS → {target_dbfs:.1f} dBFS (gain: {gain_db:+.1f} dB)")
        
        return normalized.astype(np.int16).tobytes()
        
    except Exception as e:
        logger.error(f"Error normalizing audio: {e}")
        return audio_data


# Global service instance
_gemini_service = None

def get_gemini_service() -> GeminiService:
    """Get global Gemini service instance"""
    global _gemini_service
    if _gemini_service is None:
        _gemini_service = GeminiService()
    return _gemini_service

def gemini_from_audio(pcm_bytes: bytes, mime_type: str = "audio/pcm") -> Optional[str]:
    """
    Convenience function to process audio through Gemini
    
    Args:
        pcm_bytes: Audio data in PCM16 format
        mime_type: Audio MIME type
        
    Returns:
        Hebrew text response or None if error
    """
    service = get_gemini_service()
    return service.gemini_from_audio(pcm_bytes, mime_type)

def parse_gemini_response(text_block: str) -> Tuple[str, str]:
    """
    Convenience function to parse Gemini response
    
    Args:
        text_block: Full response from Gemini
        
    Returns:
        Tuple of (transcript, answer)
    """
    service = get_gemini_service()
    return service.parse_response(text_block)

def is_gemini_available() -> bool:
    """Check if Gemini service is available"""
    service = get_gemini_service()
    return service.is_available()

def hebrew_text_to_speech(text: str, voice_config: dict = None) -> Optional[bytes]:
    """
    Convenience function to convert Hebrew text to speech
    
    Args:
        text: Hebrew text to synthesize
        voice_config: Optional voice configuration
        
    Returns:
        Audio bytes in WAV format or None if error
    """
    service = get_gemini_service()
    return service.hebrew_text_to_speech(text, voice_config)

def convert_tts_to_telnyx_format(tts_audio: bytes, target_sample_rate: int = 8000) -> Optional[bytes]:
    """
    Convenience function to convert TTS audio to Telnyx format
    
    Args:
        tts_audio: Audio bytes from TTS
        target_sample_rate: Target sample rate for Telnyx
        
    Returns:
        Audio bytes in Telnyx format or None if error
    """
    service = get_gemini_service()
    return service.convert_tts_to_telnyx_format(tts_audio, target_sample_rate)

def chunk_audio_for_streaming(audio_data: bytes, chunk_duration_ms: int = 20, sample_rate: int = 8000) -> List[bytes]:
    """
    Convenience function to chunk audio for streaming
    
    Args:
        audio_data: PCM16 audio bytes
        chunk_duration_ms: Chunk duration in milliseconds
        sample_rate: Sample rate of the audio (default: 8000 for telephony)
        
    Returns:
        List of audio chunks as bytes
    """
    service = get_gemini_service()
    return service.chunk_audio_for_streaming(audio_data, chunk_duration_ms, sample_rate) 