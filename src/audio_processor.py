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
    Convert μ-law (PCMU) audio data to 16-bit linear PCM
    
    Args:
        ulaw_data: μ-law encoded audio bytes
        
    Returns:
        Linear PCM audio bytes (16-bit)
    """
    # μ-law decompression table
    linear_data = bytearray()
    
    for ulaw_byte in ulaw_data:
        # Convert μ-law to linear PCM
        ulaw_byte = ~ulaw_byte & 0xFF  # Invert all bits
        
        sign = (ulaw_byte & 0x80) >> 7
        exponent = (ulaw_byte & 0x70) >> 4
        mantissa = ulaw_byte & 0x0F
        
        # Calculate linear value
        if exponent == 0:
            linear = (mantissa << 4) + 0x08
        else:
            linear = ((mantissa | 0x10) << (exponent + 3)) + 0x84
            
        if sign:
            linear = -linear
            
        # Clamp to 16-bit range
        linear = max(-32768, min(32767, linear))
        
        # Pack as 16-bit signed integer (little endian)
        linear_data.extend(struct.pack('<h', linear))
    
    return bytes(linear_data)


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
        
        # Initialize VAD
        self.vad = webrtcvad.Vad(vad_aggressiveness)
        
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
        Process incoming audio frame and return complete utterance if detected
        
        Args:
            audio_data: Raw audio bytes (PCM16 format)
            
        Returns:
            Complete utterance audio data if end-of-speech detected, None otherwise
        """
        # Add to buffer
        self.audio_buffer.extend(audio_data)
        
        # Process complete frames
        while len(self.audio_buffer) >= self.frame_size * 2:  # 2 bytes per sample (PCM16)
            # Extract one frame
            frame_bytes = bytes(self.audio_buffer[:self.frame_size * 2])
            self.audio_buffer = self.audio_buffer[self.frame_size * 2:]
            
            # Check if this frame contains speech
            utterance = self._process_frame(frame_bytes)
            if utterance:
                return utterance
                
        return None
    
    def _process_frame(self, frame_bytes: bytes) -> Optional[bytes]:
        """Process a single audio frame with VAD"""
        self.total_frames_processed += 1
        current_time = datetime.now()
        
        try:
            # Run VAD on the frame
            is_speech = self.vad.is_speech(frame_bytes, self.sample_rate)
            
            if is_speech:
                self.speech_frames_detected += 1
                self.last_speech_time = current_time
                
                # Start new utterance if not speaking
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
                    if silence_duration.total_seconds() * 1000 >= self.silence_threshold_ms:
                        return self._finalize_utterance()
            
            # Check for maximum utterance length
            if (self.is_speaking and self.utterance_start_time and 
                (current_time - self.utterance_start_time).total_seconds() * 1000 >= self.max_utterance_length_ms):
                logger.info("⏰ Maximum utterance length reached - forcing segmentation")
                return self._finalize_utterance()
                
        except Exception as e:
            logger.error(f"Error processing audio frame: {e}")
            
        return None
    
    def _finalize_utterance(self) -> Optional[bytes]:
        """Finalize current utterance and return audio data"""
        if not self.current_utterance_frames:
            return None
            
        # Combine all frames into single audio data
        utterance_audio = b''.join(self.current_utterance_frames)
        
        # Calculate duration
        duration_ms = len(self.current_utterance_frames) * self.frame_duration_ms
        
        logger.info(f"✅ Utterance completed: {duration_ms}ms, {len(utterance_audio)} bytes")
        
        # Reset state
        self.is_speaking = False
        self.utterance_start_time = None
        self.current_utterance_frames = []
        
        # Only return if utterance is long enough to be meaningful
        if duration_ms >= 300:  # At least 300ms of audio
            return utterance_audio
            
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
        self.audio_buffer = AudioBuffer()
        self.utterance_callback = None
        self.is_active = True
        
        logger.info(f"🎤 Audio processor initialized for call {call_control_id}")
    
    def set_utterance_callback(self, callback):
        """Set callback function for when utterances are detected"""
        self.utterance_callback = callback
    
    async def process_media_message(self, message_data: dict):
        """Process incoming media message from Telnyx WebSocket"""
        try:
            if not self.is_active:
                return
                
            # Extract audio data from message
            media_data = message_data.get('media', {})
            if not media_data:
                return
                
            # Get the audio payload (base64 encoded)
            payload = media_data.get('payload')
            if not payload:
                return
                
            # Decode base64 to raw bytes
            try:
                audio_bytes = base64.b64decode(payload)
            except Exception as e:
                logger.error(f"Failed to decode base64 audio: {e}")
                return
            
            # Convert PCMU (μ-law) to PCM16 if needed
            pcm_audio = self._convert_audio_format(audio_bytes)
            
            # Process audio through VAD
            utterance = self.audio_buffer.process_audio_frame(pcm_audio)
            
            if utterance and self.utterance_callback:
                # Call the callback with the detected utterance and call_control_id
                await self.utterance_callback(self.call_control_id, utterance)
                
        except Exception as e:
            logger.error(f"Error processing media message: {e}")
    
    def _convert_audio_format(self, audio_bytes: bytes) -> bytes:
        """Convert audio from PCMU (μ-law) to PCM16 format"""
        try:
            # Telnyx typically sends PCMU (G.711 μ-law) format
            # Convert to 16-bit linear PCM using our custom function
            pcm_audio = ulaw_to_linear(audio_bytes)
            return pcm_audio
        except Exception as e:
            # If conversion fails, assume it's already PCM16
            logger.debug(f"Audio conversion failed, assuming PCM16: {e}")
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