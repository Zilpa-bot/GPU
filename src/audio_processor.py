"""
Audio processing module for Hebrew voice recognition
Handles audio frame parsing, VAD, and speech segment detection
"""

import base64
import webrtcvad
import numpy as np
import logging
import struct
from typing import List, Optional, Tuple
from datetime import datetime, timedelta
import asyncio
import threading
from collections import deque

logger = logging.getLogger(__name__)


def ulaw_to_linear(ulaw_data: bytes) -> bytes:
    """
    Convert μ-law (PCMU) audio data to 16-bit linear PCM using the best available method.
    This ensures correct amplitude scaling for VAD and STT processing.
    """
    try:
        import audioop
        # Preferred method: use the highly optimized, standard library C module.
        pcm_data = audioop.ulaw2lin(ulaw_data, 2)  # 2 = 16-bit output
        return pcm_data
    except (ImportError, AttributeError):
        # Fallback for environments without audioop (e.g., some custom Docker images)
        logger.warning("audioop module not found. Using robust numpy-based fallback for µ-law decoding.")
        import numpy as np
        
        # G.711 µ-law decoding table for exponents. This is a standard, fast method.
        EXPONENT_LUT = np.array([0, 132, 396, 924, 1980, 4092, 8316, 16764], dtype=np.int32)
        
        ulaw_array = np.frombuffer(ulaw_data, dtype=np.uint8)
        
        # Invert all bits as per the G.711 standard for µ-law.
        ulaw_array = ~ulaw_array
        
        # Extract sign, exponent, and mantissa from each byte.
        sign = (ulaw_array & 0x80)
        exponent = (ulaw_array >> 4) & 0x07
        mantissa = ulaw_array & 0x0F
        
        # Calculate the linear PCM value.
        # 1. Use the exponent to look up the base value.
        # 2. Add the mantissa, shifted by the exponent's value.
        fragment = EXPONENT_LUT[exponent] + (mantissa << (exponent + 3))
        
        # Apply the sign to the calculated value.
        # Where sign bit is 0, value is positive. Where 1, it's negative.
        decoded = np.where(sign == 0, fragment, -fragment).astype(np.int16)
        
        return decoded.tobytes()


def upsample_8khz_to_16khz(pcm_8khz: bytes) -> bytes:
    """
    Upsample 8kHz PCM16 to 16kHz and normalize to -6 dBFS
    Follows the exact specification for proper VAD processing
    
    Args:
        pcm_8khz: 8kHz PCM16 little-endian audio bytes
        
    Returns:
        16kHz PCM16 little-endian audio bytes, normalized to -6 dBFS
    """
    # Convert bytes to numpy array
    samples_8khz = np.frombuffer(pcm_8khz, dtype=np.int16)
    
    # Simple 2x upsampling by linear interpolation (for production, use soxr)
    upsampled = np.zeros(len(samples_8khz) * 2, dtype=np.int16)
    upsampled[::2] = samples_8khz  # Original samples at even indices
    upsampled[1::2] = (samples_8khz + np.roll(samples_8khz, -1)) // 2  # Interpolated at odd indices
    upsampled[-1] = samples_8khz[-1]  # Fix last sample
    
    # Peak-limit to -6 dBFS (16384 = -6 dBFS for 16-bit audio, not 28000)
    # This ensures healthy amplitude for VAD detection
    normalized = np.clip(upsampled, -16384, 16384)  # True -6 dBFS
    
    return normalized.tobytes()


def linear_to_ulaw(pcm_data: bytes) -> bytes:
    """
    Convert 16-bit linear PCM audio at 16kHz to 8-bit µ-law at 8kHz.
    This function is resilient and uses numpy if the standard `audioop` module is not available.
    """
    try:
        import audioop
        pcm_8khz, _ = audioop.ratecv(pcm_data, 2, 1, 16000, 8000, None)
        return audioop.lin2ulaw(pcm_8khz, 2)
    except (ImportError, AttributeError):
        logger.warning("audioop not found. Using robust numpy fallback for PCM -> µ-law encoding.")
        import numpy as np
        
        samples_16k = np.frombuffer(pcm_data, dtype=np.int16)
        
        # Downsample to 8kHz by taking every other sample
        samples_8k = samples_16k[::2]
        
        # This is a standard, robust algorithm for PCM to µ-law conversion.
        # It's based on industry-standard C implementations (e.g., SoX).
        BIAS = 0x84  # Bias value for µ-law compression
        
        # Get the sign bit and the absolute value
        sign = np.bitwise_and(samples_8k, 0x8000)
        abs_val = np.abs(samples_8k)
        
        # Add the bias and clip to the max 16-bit value
        biased = np.add(abs_val, BIAS)
        clipped = np.clip(biased, 0, 0x7FFF)
        
        # Find the segment (exponent) for each sample
        # This determines the compression level for the sample's magnitude
        segment = np.searchsorted(
            np.array([0x100, 0x200, 0x400, 0x800, 0x1000, 0x2000, 0x4000], dtype=np.int16), 
            clipped
        )
        
        # Calculate the final µ-law byte
        # This combines the sign, exponent, and mantissa into a single byte
        ulaw_byte = np.bitwise_or(sign >> 8, segment << 4)
        ulaw_byte = np.bitwise_or(ulaw_byte, (clipped >> (segment + 4)) & 0x0F)
        
        # Invert the bits as per the standard
        return (~ulaw_byte.astype(np.uint8)).tobytes()


class AudioBuffer:
    """Manages audio buffering and VAD for Hebrew speech detection"""
    
    def __init__(self, 
                 sample_rate: int = 8000,
                 frame_duration_ms: int = 20,
                 vad_aggressiveness: int = 2,
                 silence_threshold_ms: int = 800,
                 max_utterance_length_ms: int = 10000):
        """
        Initialize audio buffer with VAD settings
        
        Args:
            sample_rate: Audio sample rate (8kHz for telephony)
            frame_duration_ms: VAD frame duration (20ms is optimal)
            vad_aggressiveness: VAD aggressiveness (0-3, higher = more aggressive)
            silence_threshold_ms: Silence duration to trigger end-of-utterance
            max_utterance_length_ms: Maximum utterance length before forced segmentation
        """
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.frame_size = int(sample_rate * frame_duration_ms / 1000)  # 160 samples for 8kHz/20ms
        self.silence_threshold_ms = silence_threshold_ms
        self.max_utterance_length_ms = max_utterance_length_ms
        
        # Initialize VAD - ensure sample rate is supported
        # WebRTC VAD supports: 8000, 16000, 32000, 48000 Hz
        if sample_rate not in [8000, 16000, 32000, 48000]:
            logger.warning(f"Sample rate {sample_rate}Hz not supported by WebRTC VAD, using 8000Hz")
            self.vad_sample_rate = 8000
        else:
            self.vad_sample_rate = sample_rate
            
        self.vad = webrtcvad.Vad(vad_aggressiveness)
        
        logger.info(f"🎤 AudioBuffer initialized: {sample_rate}Hz input, {self.vad_sample_rate}Hz VAD, {frame_duration_ms}ms frames")
        
        # Audio buffer and state
        self.audio_buffer = bytearray()
        self.speech_frames = deque()
        self.is_speaking = False
        self.last_speech_time = None
        self.utterance_start_time = None
        self.current_utterance_frames = []
        
        # Statistics
        self.total_frames_processed = 0
        self.speech_frames_detected = 0
        
    def process_audio_frame(self, audio_data: bytes) -> Optional[bytes]:
        """
        Process incoming 16kHz audio frame and return complete utterance if detected.
        Audio is already 16kHz, so no upsampling is required.
        """
        logger.debug(f"📥 AudioBuffer received {len(audio_data)} bytes of 16kHz audio")
        
        # Add to buffer
        self.audio_buffer.extend(audio_data)
        
        # Process complete frames (640 bytes for 20ms at 16kHz)
        frame_size_16khz = 640
        frames_processed = 0
        
        while len(self.audio_buffer) >= frame_size_16khz:
            frame_bytes = bytes(self.audio_buffer[:frame_size_16khz])
            self.audio_buffer = self.audio_buffer[frame_size_16khz:]
            
            utterance = self._process_frame_16khz(frame_bytes)
            frames_processed += 1
            
            if utterance:
                return utterance
                
        return None
    
    def _process_frame_16khz(self, frame_bytes: bytes) -> Optional[bytes]:
        """Process a 16kHz audio frame with VAD and return completed utterance if any"""
        try:
            current_time = datetime.now()
            
            # Basic validation - 640 bytes for 20ms at 16kHz (320 samples × 2 bytes)
            expected_frame_size = 640
            if len(frame_bytes) != expected_frame_size:
                logger.warning(f"⚠️ Frame size mismatch: got {len(frame_bytes)}, expected {expected_frame_size}")
                return None
            
            logger.debug(f"✅ 16kHz frame validation passed: {len(frame_bytes)} bytes for VAD")
            
            # Add energy-based pre-filter to prevent false positive on silence
            import numpy as np
            try:
                samples = np.frombuffer(frame_bytes, dtype=np.int16)
                
                # Calculate RMS energy
                rms_energy = np.sqrt(np.mean(samples.astype(np.float32) ** 2))
                energy_threshold = 50.0  # Minimum energy for non-silence
                
                logger.debug(f"🔊 Frame energy: {rms_energy:.1f} (threshold: {energy_threshold})")
                
                # If energy is too low, treat as silence regardless of VAD
                if rms_energy < energy_threshold:
                    logger.debug(f"🔇 Energy too low ({rms_energy:.1f} < {energy_threshold}) - treating as silence")
                    is_speech = False
                else:
                    # Call VAD with frame_bytes
                    logger.debug(f"🤖 Calling VAD.is_speech() with {len(frame_bytes)} bytes at {self.sample_rate}Hz")
                    is_speech = self.vad.is_speech(frame_bytes, self.sample_rate)
                    logger.debug(f"🤖 VAD result: is_speech={is_speech}")
            except ImportError:
                # Fallback to VAD only if numpy not available
                logger.debug(f"🤖 Calling VAD.is_speech() with {len(frame_bytes)} bytes at {self.sample_rate}Hz")
                is_speech = self.vad.is_speech(frame_bytes, self.sample_rate)
                logger.debug(f"🤖 VAD result: is_speech={is_speech}")
            
            # Update statistics
            self.total_frames_processed += 1
            if is_speech:
                self.speech_frames_detected += 1
                self.last_speech_time = current_time
            
            # Log periodic stats
            if self.total_frames_processed % 100 == 0:
                speech_percentage = (self.speech_frames_detected / self.total_frames_processed) * 100
                logger.info(f"🔊 VAD Stats (16kHz frame {self.total_frames_processed}): {self.speech_frames_detected}/{self.total_frames_processed} frames = {speech_percentage:.1f}% speech, speaking={self.is_speaking}, utterance_frames={len(self.current_utterance_frames)}")
            
            if is_speech:
                logger.info(f"🗣️ SPEECH DETECTED in 16kHz frame {self.total_frames_processed}!")
                
                # Start new utterance if not already speaking
                if not self.is_speaking:
                    self.is_speaking = True
                    self.utterance_start_time = current_time
                    self.current_utterance_frames = []
                    logger.info("🗣️ Speech detected - starting new utterance")
                
                # Add frame to current utterance
                self.current_utterance_frames.append(frame_bytes)
                
            else:
                # Silence frame
                if self.is_speaking:
                    # Add silence frame to utterance (helps with natural speech flow)
                    self.current_utterance_frames.append(frame_bytes)
                    
                    # Check if we've had enough silence to end utterance
                    silence_duration = current_time - self.last_speech_time
                    silence_ms = silence_duration.total_seconds() * 1000
                    logger.debug(f"🔇 Silence: {silence_ms:.0f}ms (threshold: {self.silence_threshold_ms}ms)")
                    
                    if silence_ms >= self.silence_threshold_ms:
                        logger.info(f"🔚 Silence threshold reached ({silence_ms:.0f}ms >= {self.silence_threshold_ms}ms) - finalizing utterance with {len(self.current_utterance_frames)} frames")
                        return self._finalize_utterance()
                    else:
                        logger.debug(f"⏳ Still waiting for silence: {silence_ms:.0f}ms < {self.silence_threshold_ms}ms")
            
            # Check for maximum utterance length
            if (self.is_speaking and self.utterance_start_time and 
                (current_time - self.utterance_start_time).total_seconds() * 1000 >= self.max_utterance_length_ms):
                logger.info("⏰ Maximum utterance length reached - forcing segmentation")
                return self._finalize_utterance()
                
        except Exception as e:
            logger.error(f"🚨 CRITICAL ERROR in 16kHz VAD _process_frame: {e}")
            logger.error(f"Frame details: {len(frame_bytes)} bytes, sample_rate: 16000Hz, aggressiveness: {self.vad.aggressiveness if hasattr(self.vad, 'aggressiveness') else 'unknown'}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            
        return None
    
    def _finalize_utterance(self) -> Optional[bytes]:
        """Finalize current utterance and return audio data"""
        if not self.current_utterance_frames:
            logger.warning("🔇 No utterance frames to finalize")
            return None
            
        # Combine all frames into single audio data
        utterance_audio = b''.join(self.current_utterance_frames)
        
        # Calculate duration
        duration_ms = len(self.current_utterance_frames) * self.frame_duration_ms
        
        logger.info(f"✅ Utterance completed: {duration_ms}ms, {len(utterance_audio)} bytes, {len(self.current_utterance_frames)} frames")
        
        # Reset state
        self.is_speaking = False
        self.utterance_start_time = None
        self.current_utterance_frames = []
        
        # Only return if utterance is long enough to be meaningful
        min_duration_ms = 100  # Reduced from 300ms for faster response
        if duration_ms >= min_duration_ms:
            logger.info(f"🎯 Utterance meets minimum duration ({duration_ms}ms >= {min_duration_ms}ms) - sending for processing")
            return utterance_audio
        else:
            logger.info(f"⏭️ Utterance too short ({duration_ms}ms < {min_duration_ms}ms) - discarding")
            
        return None
    
    def get_stats(self) -> dict:
        """Get buffer statistics"""
        return {
            "total_frames_processed": self.total_frames_processed,
            "speech_frames_detected": self.speech_frames_detected,
            "speech_ratio": self.speech_frames_detected / max(self.total_frames_processed, 1),
            "is_currently_speaking": self.is_speaking,
            "buffer_size": len(self.audio_buffer),
            "current_utterance_frames": len(self.current_utterance_frames)
        }


class AudioProcessor:
    """Main audio processing class for Hebrew voice recognition"""
    
    def __init__(self, call_control_id: str):
        """Initialize audio processor for specific call"""
        self.call_control_id = call_control_id
        # Proper VAD settings per fix plan
        self.audio_buffer = AudioBuffer(
            sample_rate=16000,
            frame_duration_ms=20,
            vad_aggressiveness=1,  # Telnyx IVR default
            silence_threshold_ms=600,  # Common IVR setting for Hebrew speech
            max_utterance_length_ms=10000  # Flush long rambling utterances
        )
        self.utterance_callback = None
        self.is_active = True
        
        logger.info(f"🎤 Audio processor initialized for call {call_control_id} - VAD aggressiveness=1, 600ms silence threshold")
    
    def set_utterance_callback(self, callback):
        """Set callback function for when utterances are detected"""
        self.utterance_callback = callback
    
    async def process_media_message(self, message_data: dict):
        """Process an incoming media message based on the detected codec."""
        from .connection_manager import connection_manager
        session = connection_manager.get_session(self.call_control_id)
        if not session:
            return

        # Pause VAD during playback to prevent self-barge-in
        if session.speaking:
            logger.debug(f"🔇 Skipping VAD processing - AI is speaking.")
            return

        # Skip processing if codec is not yet determined
        if session.codec == "UNKNOWN":
            return

        media_data = message_data.get('media', {})
        track = media_data.get('track', 'inbound')
        if track != 'inbound':
            return

        payload = media_data.get('payload')
        if not payload:
            return

        try:
            audio_bytes = base64.b64decode(payload)
            logger.debug(f"Received {len(audio_bytes)} bytes on {session.codec} track.")

            pcm_16khz_audio = None
            if session.codec == "OPUS":
                if len(audio_bytes) == 640:
                    pcm_16khz_audio = audio_bytes
                else:
                    logger.warning(f"Incorrect frame size for OPUS: {len(audio_bytes)} bytes.")
            
            elif session.codec == "PCMU":
                if len(audio_bytes) == 160:
                    pcm_8khz = ulaw_to_linear(audio_bytes)
                    pcm_16khz_audio = upsample_8khz_to_16khz(pcm_8khz)
                else:
                    logger.warning(f"Incorrect frame size for PCMU: {len(audio_bytes)} bytes.")

            if pcm_16khz_audio:
                # Log first 10 decoded samples for quality check
                try:
                    import numpy as np
                    samples = np.frombuffer(pcm_16khz_audio, dtype=np.int16)
                    logger.debug(f"Inbound samples (first 10): {samples[:10]}")
                except ImportError:
                    logger.debug("Numpy not found, skipping sample logging.")

                utterance = self.audio_buffer.process_audio_frame(pcm_16khz_audio)
                if utterance:
                    logger.info(f"🎤 VAD detected complete 16kHz utterance: {len(utterance)} bytes")
                    if self.utterance_callback:
                        await self.utterance_callback(self.call_control_id, utterance)
        except Exception as e:
            logger.error(f"Error processing media message: {e}")
            
    def _convert_audio_format(self, audio_bytes: bytes) -> bytes:
        # This function is now effectively deprecated by the dual-pipeline logic
        # but is kept to avoid breaking other parts of the code that might reference it.
        logger.warning("Bypassing _convert_audio_format; logic is now in process_media_message.")
        return audio_bytes
    
    def get_stats(self) -> dict:
        """Get processing statistics"""
        return {
            "call_control_id": self.call_control_id,
            "is_active": self.is_active,
            "audio_buffer_stats": self.audio_buffer.get_stats()
        }
    
    def stop(self):
        """Stop audio processing"""
        self.is_active = False
        logger.info(f"🛑 Audio processor stopped for call {self.call_control_id}")


# Global audio processor manager
audio_processors = {}


def get_audio_processor(call_control_id: str) -> AudioProcessor:
    """Get or create audio processor for call"""
    if call_control_id not in audio_processors:
        audio_processors[call_control_id] = AudioProcessor(call_control_id)
    return audio_processors[call_control_id]


def cleanup_audio_processor(call_control_id: str):
    """Clean up audio processor for call"""
    if call_control_id in audio_processors:
        audio_processors[call_control_id].stop()
        del audio_processors[call_control_id]
        logger.info(f"🧹 Audio processor cleaned up for call {call_control_id}") 