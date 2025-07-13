"""
Audio frame builder for Telnyx WebSocket streaming.

This module provides robust frame preparation for both OPUS (16kHz) and PCMU (8kHz μ-law) 
codecs, following the exact specifications for clean audio streaming without noise.

Kill-the-noise playbook implementation:
- OPUS: 16kHz PCM-16 little-endian, 640-byte frames (20ms)
- PCMU: 8kHz μ-law, 160-byte frames (20ms)
"""

import numpy as np
import base64
import logging
from typing import List, Tuple, Optional
import struct
import io
import wave

logger = logging.getLogger(__name__)

# Try to import soxr for high-quality resampling
try:
    import soxr
    SOXR_AVAILABLE = True
    logger.info("✅ soxr available for high-quality resampling")
except ImportError:
    SOXR_AVAILABLE = False
    logger.warning("⚠️ soxr not available, falling back to basic resampling")


def linear_to_ulaw(pcm_data: bytes) -> bytes:
    """
    Convert 16-bit linear PCM to μ-law (G.711) format.
    
    This is a pure Python implementation of μ-law encoding
    to replace the deprecated audioop module.
    
    Args:
        pcm_data: 16-bit linear PCM data (little-endian)
        
    Returns:
        μ-law encoded audio data
    """
    # μ-law constants
    BIAS = 0x84
    CLIP = 32635
    
    # Convert bytes to 16-bit signed integers (little-endian)
    samples = np.frombuffer(pcm_data, dtype='<i2')
    ulaw_samples = []
    
    for sample in samples:
        # Get absolute value and apply bias
        sample = int(sample)
        if sample < 0:
            sample = -sample
            sign = 0x80
        else:
            sign = 0x00
        
        # Clip to prevent overflow
        if sample > CLIP:
            sample = CLIP
        
        # Add bias
        sample += BIAS
        
        # Find the segment
        seg = 0
        if sample >= 0x100:
            seg = 1
            sample >>= 1
        if sample >= 0x200:
            seg = 2
            sample >>= 1
        if sample >= 0x400:
            seg = 3
            sample >>= 1
        if sample >= 0x800:
            seg = 4
            sample >>= 1
        if sample >= 0x1000:
            seg = 5
            sample >>= 1
        if sample >= 0x2000:
            seg = 6
            sample >>= 1
        if sample >= 0x4000:
            seg = 7
            sample >>= 1
        
        # Quantize the sample
        quantized = (sample >> 4) & 0x0F
        
        # Combine sign, segment, and quantized value
        ulaw_val = sign | (seg << 4) | quantized
        
        # Invert bits (G.711 standard)
        ulaw_val = (~ulaw_val) & 0xFF
        
        ulaw_samples.append(ulaw_val)
    
    return bytes(ulaw_samples)


class AudioFrameBuilder:
    """
    Robust audio frame builder for Telnyx WebSocket streaming.
    
    Following the kill-the-noise playbook for clean audio without distortion.
    """
    
    def __init__(self, codec: str = "OPUS"):
        """
        Initialize the frame builder for a specific codec.
        
        Args:
            codec: Target codec ("OPUS" for 16kHz or "PCMU" for 8kHz μ-law)
        """
        self.codec = codec.upper()
        logger.info(f"🎵 AudioFrameBuilder initialized for {self.codec} codec")
        
        # Frame size constants (20ms frames)
        self.frame_sizes = {
            "OPUS": 640,  # 16kHz × 20ms × 2 bytes = 640 bytes
            "PCMU": 160   # 8kHz × 20ms × 1 byte = 160 bytes
        }
        
        if self.codec not in self.frame_sizes:
            raise ValueError(f"Unsupported codec: {self.codec}")
    
    def prepare_frames_from_wav(self, wav_data: bytes) -> List[bytes]:
        """
        Prepare audio frames from WAV data for streaming.
        
        Following kill-the-noise playbook step 3: re-format/resample step (A)
        
        Args:
            wav_data: WAV audio data (should be mono 16kHz or 24kHz PCM-16)
            
        Returns:
            List of audio frames ready for streaming
        """
        try:
            # Step 3A: Extract PCM data and detect sample rate
            pcm_data, sample_rate = self._extract_pcm_with_rate(wav_data)
            if not pcm_data:
                logger.error("Failed to extract PCM data from WAV")
                return []
            
            logger.info(f"📊 Extracted {len(pcm_data)} bytes of PCM data from WAV (detected rate: {sample_rate}Hz)")
            
            # Step 3A: Process according to codec requirements
            if self.codec == "OPUS":
                return self._prepare_opus_frames(pcm_data, sample_rate)
            else:  # PCMU
                return self._prepare_pcmu_frames(pcm_data, sample_rate)
                
        except Exception as e:
            logger.error(f"Error preparing frames: {e}")
            return []
    
    def _extract_pcm_with_rate(self, wav_data: bytes) -> Tuple[Optional[bytes], int]:
        """
        Extract PCM data and sample rate from WAV file.
        
        Args:
            wav_data: WAV file data
            
        Returns:
            Tuple of (PCM data, sample rate)
        """
        try:
            # Use wave module to properly parse WAV headers
            with wave.open(io.BytesIO(wav_data), 'rb') as wav_file:
                # Verify format
                if wav_file.getnchannels() != 1:
                    logger.error(f"Expected mono audio, got {wav_file.getnchannels()} channels")
                    return None, 0
                
                if wav_file.getsampwidth() != 2:
                    logger.error(f"Expected 16-bit audio, got {wav_file.getsampwidth() * 8}-bit")
                    return None, 0
                
                sample_rate = wav_file.getframerate()
                logger.info(f"🎵 WAV format: {sample_rate}Hz, {wav_file.getsampwidth() * 8}-bit, {wav_file.getnchannels()} channel(s)")
                
                # Extract PCM data
                pcm_data = wav_file.readframes(wav_file.getnframes())
                
                return pcm_data, sample_rate
                
        except Exception as e:
            logger.error(f"Error extracting PCM with rate: {e}")
            return None, 0
    
    def _prepare_opus_frames(self, pcm_data: bytes, source_rate: int) -> List[bytes]:
        """
        Prepare OPUS frames (16kHz PCM16 little-endian, 640-byte frames).
        
        Following kill-the-noise playbook step 3.2 for OPUS/PCM-16 (16 kHz)
        
        Args:
            pcm_data: PCM16 audio data
            source_rate: Source sample rate
            
        Returns:
            List of 640-byte frames
        """
        try:
            # Step 3.2: Resample to 16kHz if needed
            if source_rate != 16000:
                logger.info(f"🔄 Resampling from {source_rate}Hz to 16kHz")
                pcm_16khz = self._resample_to_16khz(pcm_data, source_rate)
                if not pcm_16khz:
                    logger.error("Failed to resample to 16kHz")
                    return []
            else:
                pcm_16khz = pcm_data
                logger.info(f"✅ Audio already at 16kHz, no resampling needed")
            
            # Step 3.2: IMPORTANT: Ensure little-endian format
            pcm_16khz_le = self._ensure_little_endian(pcm_16khz)
            logger.info(f"✅ Ensured little-endian format: {len(pcm_16khz_le)} bytes")
            
            # Step 7: Volume normalization to prevent Telnyx transcoder clipping
            pcm_16khz_normalized = self._normalize_volume_to_minus_6dbfs(pcm_16khz_le)
            logger.info(f"✅ Applied volume normalization to -10 dBFS: {len(pcm_16khz_normalized)} bytes")
            
            # Step 4: Create 640-byte frames (20ms at 16kHz)
            frames = []
            frame_size = 640
            
            for i in range(0, len(pcm_16khz_normalized), frame_size):
                frame = pcm_16khz_normalized[i:i+frame_size]
                
                # Pad last frame if necessary with silence
                if len(frame) < frame_size:
                    padding = frame_size - len(frame)
                    frame += b'\x00' * padding
                    logger.debug(f"🔇 Padded last frame with {padding} bytes of silence")
                
                frames.append(frame)
            
            logger.info(f"✅ Created {len(frames)} × 640B OPUS frames for 16kHz streaming")
            return frames
            
        except Exception as e:
            logger.error(f"Error preparing OPUS frames: {e}")
            return []
    
    def _prepare_pcmu_frames(self, pcm_data: bytes, source_rate: int) -> List[bytes]:
        """
        Prepare PCMU frames (8kHz μ-law, 160-byte frames).
        
        FIXED: Now normalizes on NumPy int16 array and targets -3 dBFS for clearer µ-law encoding.
        
        Args:
            pcm_data: PCM16 audio data
            source_rate: Source sample rate
            
        Returns:
            List of 160-byte μ-law frames
        """
        try:
            # Convert PCM bytes to NumPy int16 array (little-endian)
            pcm16 = np.frombuffer(pcm_data, dtype='<i2')
            logger.info(f"📊 Starting PCMU processing: {len(pcm16)} samples at {source_rate}Hz")
            
            # Step 1: Normalize on int16 array (target peak -3 dBFS)
            peak = np.abs(pcm16).max()
            target = int(32767 * 10**(-3/20))  # -3 dBFS ≈ 23170
            gain = target / peak if peak > 0 else 1.0
            pcm16_normalized = (pcm16 * gain).astype('<i2')
            
            logger.info(f"🔧 Normalized to -3 dBFS: peak {peak} → {np.abs(pcm16_normalized).max()}, gain={gain:.3f}")
            
            # Step 2: Resample to 8kHz with soxr (high quality)
            if SOXR_AVAILABLE:
                logger.debug("🎵 Using soxr for high-quality resampling to 8kHz")
                pcm8_float = soxr.resample(
                    pcm16_normalized.astype(np.float32),
                    source_rate,
                    8000,
                    quality='HQ'
                )
                pcm8 = pcm8_float.astype('<i2')
            else:
                logger.warning("⚠️ soxr not available, using basic resampling")
                # Fallback to basic resampling
                if source_rate == 16000:
                    pcm8 = pcm16_normalized[::2]  # Simple decimation
                elif source_rate == 24000:
                    pcm8 = pcm16_normalized[::3]  # Simple decimation
                else:
                    logger.error(f"Unsupported source rate {source_rate}Hz without soxr")
                    return []
            
            # Step 3: Ensure little-endian format just before μ-law conversion
            if pcm8.dtype.byteorder != '<':
                pcm8 = pcm8.byteswap().view(pcm8.dtype.newbyteorder('<'))
            
            # Step 4: Log peak level before μ-law conversion for debugging
            peak_lin = np.abs(pcm8).max()
            logger.info(f"📊 Peak int16 before μ-law: {peak_lin} (expect >4000 for clear audio)")
            
            if peak_lin < 1000:
                logger.warning(f"⚠️ Signal too soft before μ-law: {peak_lin} < 1000")
            
            # Step 5: Convert to μ-law
            mulaw_data = linear_to_ulaw(pcm8.tobytes())
            logger.info(f"🔧 Converted to μ-law: {len(mulaw_data)} bytes")
            
            # Step 6: Debug first few decoded samples for verification
            try:
                # Use the existing fallback from audio_processor
                from audio_processor import ulaw_to_linear
                back_check = ulaw_to_linear(mulaw_data[:8])
                decoded_samples = np.frombuffer(back_check, dtype='<i2')
                logger.info(f"🔍 First 8 decoded samples: {decoded_samples.tolist()}")
            except:
                logger.debug("Could not decode μ-law samples for verification")
            
            # Step 7: Create 160-byte frames (20ms at 8kHz μ-law)
            frames = []
            frame_size = 160
            
            for i in range(0, len(mulaw_data), frame_size):
                frame = mulaw_data[i:i+frame_size]
                
                # Pad last frame if necessary with μ-law silence (0xFF)
                if len(frame) < frame_size:
                    padding = frame_size - len(frame)
                    frame += b'\xff' * padding  # μ-law silence is 0xFF
                    logger.debug(f"🔇 Padded last frame with {padding} bytes of μ-law silence")
                
                frames.append(frame)
            
            logger.info(f"✅ Created {len(frames)} × 160B PCMU frames for 8kHz streaming")
            return frames
            
        except Exception as e:
            logger.error(f"Error preparing PCMU frames: {e}")
            return []
    
    def _ensure_little_endian(self, pcm_data: bytes) -> bytes:
        """
        Ensure PCM data is in little-endian format.
        
        Following kill-the-noise playbook: "IMPORTANT: little-endian!"
        """
        try:
            # Read as 16-bit signed integers, auto-detect endianness
            samples = np.frombuffer(pcm_data, dtype=np.int16)
            
            # Force little-endian format
            le_samples = samples.astype('<i2')
            result = le_samples.tobytes()
            
            logger.debug(f"✅ Ensured little-endian format: {len(result)} bytes")
            return result
            
        except Exception as e:
            logger.error(f"Error ensuring little-endian format: {e}")
            return pcm_data
    
    def _resample_to_16khz(self, pcm_data: bytes, source_rate: int) -> Optional[bytes]:
        """Resample PCM data to 16kHz using high-quality resampling."""
        try:
            # Convert to numpy array (little-endian)
            samples = np.frombuffer(pcm_data, dtype='<i2')
            
            if SOXR_AVAILABLE:
                # High-quality resampling with soxr
                logger.debug("🎵 Using soxr for high-quality resampling to 16kHz")
                samples_16khz = soxr.resample(
                    samples.astype(np.float32),
                    source_rate,
                    16000,
                    quality='HQ'
                )
                # Convert back to int16 little-endian
                samples_16khz = samples_16khz.astype('<i2')
            else:
                # Basic resampling with anti-aliasing filter
                logger.warning("⚠️ soxr not available, using basic resampling with anti-aliasing filter")
                
                if source_rate == 24000:
                    # 24kHz -> 16kHz: Apply low-pass filter then resample (3:2 ratio)
                    logger.debug("🔧 Using basic resampling 24kHz->16kHz with anti-aliasing filter")
                    # 19-tap Hamming low-pass filter @8 kHz (Nyquist for 16kHz)
                    h = np.hamming(19)
                    h /= h.sum()
                    samples_filtered = np.convolve(samples.astype(np.float32), h, mode='valid')
                    # Use linear interpolation for 3:2 resampling
                    indices = np.arange(0, len(samples_filtered), 1.5)
                    samples_16khz = np.interp(indices, np.arange(len(samples_filtered)), samples_filtered).astype('<i2')
                elif source_rate == 8000:
                    # 8kHz -> 16kHz: duplicate every sample (upsampling - no anti-aliasing needed)
                    logger.debug("🔧 Using basic resampling 8kHz->16kHz (1:2 ratio)")
                    samples_16khz = np.repeat(samples, 2)
                else:
                    logger.warning(f"⚠️ Unsupported source rate {source_rate}Hz for basic resampling")
                    return None
            
            return samples_16khz.tobytes()
            
        except Exception as e:
            logger.error(f"Error resampling to 16kHz: {e}")
            return None
    
    def _resample_to_8khz(self, pcm_data: bytes, source_rate: int) -> Optional[bytes]:
        """Resample PCM data to 8kHz using high-quality resampling."""
        try:
            # Convert to numpy array (little-endian)
            samples = np.frombuffer(pcm_data, dtype='<i2')
            
            if SOXR_AVAILABLE:
                # High-quality resampling with soxr
                logger.debug("🎵 Using soxr for high-quality resampling to 8kHz")
                samples_8khz = soxr.resample(
                    samples.astype(np.float32),
                    source_rate,
                    8000,
                    quality='HQ'
                )
                # Convert back to int16 little-endian
                samples_8khz = samples_8khz.astype('<i2')
            else:
                # Basic resampling with anti-aliasing filter
                logger.warning("⚠️ soxr not available, using basic resampling with anti-aliasing filter")
                
                if source_rate == 16000:
                    # 16kHz -> 8kHz: Apply low-pass filter then decimate by 2
                    logger.debug("🔧 Using basic resampling 16kHz->8kHz with anti-aliasing filter")
                    # 17-tap Hamming low-pass filter @4 kHz (Nyquist for 8kHz)
                    h = np.hamming(17)
                    h /= h.sum()
                    samples_filtered = np.convolve(samples.astype(np.float32), h, mode='valid')
                    samples_8khz = samples_filtered[::2].astype('<i2')
                elif source_rate == 24000:
                    # 24kHz -> 8kHz: Apply low-pass filter then decimate by 3
                    logger.debug("🔧 Using basic resampling 24kHz->8kHz with anti-aliasing filter")
                    # 25-tap Hamming low-pass filter @4 kHz (Nyquist for 8kHz)
                    h = np.hamming(25)
                    h /= h.sum()
                    samples_filtered = np.convolve(samples.astype(np.float32), h, mode='valid')
                    samples_8khz = samples_filtered[::3].astype('<i2')
                else:
                    logger.warning(f"⚠️ Unsupported source rate {source_rate}Hz for basic resampling")
                    return None
            
            return samples_8khz.tobytes()
            
        except Exception as e:
            logger.error(f"Error resampling to 8kHz: {e}")
            return None
    
    def _trim_silence_from_start(self, pcm_data: bytes, silence_threshold: int = 100) -> bytes:
        """
        Trim silence from the beginning of audio to prevent μ-law hiss.
        
        The micro-tests showed that greeting.wav starts with very quiet samples
        (6, 7, 7, 7...) which become near-silence (0xf7) in μ-law, causing hiss.
        """
        try:
            # Convert to numpy array (little-endian 16-bit signed integers)
            samples = np.frombuffer(pcm_data, dtype='<i2')
            
            # Find the first sample above the silence threshold
            abs_samples = np.abs(samples)
            above_threshold = np.where(abs_samples > silence_threshold)[0]
            
            if len(above_threshold) > 0:
                # Trim from the first significant sample
                start_idx = above_threshold[0]
                trimmed_samples = samples[start_idx:]
                logger.debug(f"✂️ Trimmed {start_idx} samples of silence from start")
                return trimmed_samples.tobytes()
            else:
                logger.warning("⚠️ No samples found above silence threshold")
                return pcm_data
                
        except Exception as e:
            logger.error(f"Error trimming silence: {e}")
            return pcm_data

    def _normalize_volume_to_minus_6dbfs(self, pcm_data: bytes) -> bytes:
        """
        Normalize volume to -10 dBFS to prevent μ-law clipping and noise.
        
        Following kill-the-noise playbook step 7: Volume normalization
        Changed from -6 dBFS to -10 dBFS based on micro-test results showing
        that -10 dBFS produces cleaner μ-law with less clipping artifacts.
        """
        try:
            # Convert to numpy array (little-endian 16-bit signed integers)
            samples = np.frombuffer(pcm_data, dtype='<i2')
            
            # Trim silence from the beginning first
            trimmed_data = self._trim_silence_from_start(samples.tobytes())
            samples = np.frombuffer(trimmed_data, dtype='<i2')
            
            # Playbook step 7: Clip to -10 dBFS (approximately -10362 to 10362 for 16-bit)
            # -10 dBFS = 32767 * 10^(-10/20) ≈ 10362
            # This prevents μ-law clipping and reduces noise artifacts
            clipped_samples = np.clip(samples, -10362, 10362)
            
            # Convert back to bytes
            normalized_pcm = clipped_samples.astype('<i2').tobytes()
            
            logger.debug(f"✅ Normalized volume to -10 dBFS: {len(normalized_pcm)} bytes")
            return normalized_pcm
            
        except Exception as e:
            logger.error(f"Error normalizing volume: {e}")
            return pcm_data
    
    def encode_frames_for_streaming(self, frames: List[bytes]) -> List[str]:
        """
        Base64-encode frames for Telnyx WebSocket streaming.
        
        Following kill-the-noise playbook step 4: Frame & base-64 step (B)
        
        Args:
            frames: List of audio frames
            
        Returns:
            List of base64-encoded payloads
        """
        try:
            expected_size = self.frame_sizes[self.codec]
            
            b64_frames = []
            for i, frame in enumerate(frames):
                # Safety assert catches size mismatch before hitting network
                if len(frame) != expected_size:
                    logger.error(f"❌ Frame {i} size mismatch: expected {expected_size}, got {len(frame)}")
                    continue
                
                b64_payload = base64.b64encode(frame).decode('ascii')
                b64_frames.append(b64_payload)
                
                # Verify the base64 decode gives correct size
                decoded_size = len(base64.b64decode(b64_payload))
                if decoded_size != expected_size:
                    logger.error(f"❌ Frame {i} base64 decode size mismatch: expected {expected_size}, got {decoded_size}")
            
            logger.info(f"📤 Encoded {len(b64_frames)} frames for {self.codec} streaming")
            return b64_frames
            
        except Exception as e:
            logger.error(f"Error encoding frames: {e}")
            return []


def create_audio_frames(wav_data: bytes, codec: str = "OPUS") -> List[str]:
    """
    Convenience function to create ready-to-stream audio frames.
    
    Following kill-the-noise playbook implementation.
    
    Args:
        wav_data: WAV file data
        codec: Target codec ("OPUS" or "PCMU")
        
    Returns:
        List of base64-encoded audio frames ready for streaming
    """
    try:
        builder = AudioFrameBuilder(codec)
        frames = builder.prepare_frames_from_wav(wav_data)
        return builder.encode_frames_for_streaming(frames)
    except Exception as e:
        logger.error(f"Error creating audio frames: {e}")
        return []


def test_frame_builder():
    """Test the frame builder with the greeting.wav file."""
    try:
        import os
        greeting_path = "greeting.wav"
        
        if not os.path.exists(greeting_path):
            logger.warning(f"Greeting file not found: {greeting_path}")
            return
        
        # Load greeting WAV
        with open(greeting_path, 'rb') as f:
            wav_data = f.read()
        
        logger.info(f"🔍 Testing frame builder with {len(wav_data)} bytes of WAV data")
        
        # Test OPUS frames
        opus_frames = create_audio_frames(wav_data, "OPUS")
        logger.info(f"✅ OPUS: {len(opus_frames)} frames prepared")
        
        # Test PCMU frames
        pcmu_frames = create_audio_frames(wav_data, "PCMU")
        logger.info(f"✅ PCMU: {len(pcmu_frames)} frames prepared")
        
        # Verify frame sizes
        if opus_frames:
            opus_frame_data = base64.b64decode(opus_frames[0])
            logger.info(f"📊 OPUS frame size: {len(opus_frame_data)} bytes (expected: 640)")
        
        if pcmu_frames:
            pcmu_frame_data = base64.b64decode(pcmu_frames[0])
            logger.info(f"📊 PCMU frame size: {len(pcmu_frame_data)} bytes (expected: 160)")
        
        return True
        
    except Exception as e:
        logger.error(f"Error testing frame builder: {e}")
        return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_frame_builder() 