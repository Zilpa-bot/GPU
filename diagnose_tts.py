#!/usr/bin/env python3
"""
Diagnose Gemini TTS output format to fix the "noises" issue
"""

import os
import sys
import struct
import wave

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def analyze_wav_header(audio_data: bytes):
    """Analyze WAV header to extract exact format info"""
    if not audio_data.startswith(b'RIFF'):
        print("❌ Not a WAV file")
        return None
    
    try:
        # Parse WAV header
        chunk_size = struct.unpack('<I', audio_data[4:8])[0]
        format_type = audio_data[8:12]
        
        # Find fmt chunk
        pos = 12
        while pos < len(audio_data) - 8:
            chunk_id = audio_data[pos:pos+4]
            chunk_size = struct.unpack('<I', audio_data[pos+4:pos+8])[0]
            
            if chunk_id == b'fmt ':
                # Parse format chunk
                audio_format = struct.unpack('<H', audio_data[pos+8:pos+10])[0]
                channels = struct.unpack('<H', audio_data[pos+10:pos+12])[0]
                sample_rate = struct.unpack('<I', audio_data[pos+12:pos+16])[0]
                byte_rate = struct.unpack('<I', audio_data[pos+16:pos+20])[0]
                block_align = struct.unpack('<H', audio_data[pos+20:pos+22])[0]
                bits_per_sample = struct.unpack('<H', audio_data[pos+22:pos+24])[0]
                
                print(f"📊 WAV Format Analysis:")
                print(f"   Format: {audio_format} ({'PCM' if audio_format == 1 else 'Other'})")
                print(f"   Channels: {channels}")
                print(f"   Sample Rate: {sample_rate} Hz")
                print(f"   Byte Rate: {byte_rate}")
                print(f"   Block Align: {block_align}")
                print(f"   Bits per Sample: {bits_per_sample}")
                
                return {
                    'sample_rate': sample_rate,
                    'channels': channels,
                    'bits_per_sample': bits_per_sample,
                    'audio_format': audio_format
                }
                
            pos += 8 + chunk_size
            if chunk_size % 2:  # Word alignment
                pos += 1
                
        print("❌ No fmt chunk found")
        return None
        
    except Exception as e:
        print(f"❌ Error parsing WAV: {e}")
        return None


def main():
    """Test TTS and analyze output"""
    print("🔍 Diagnosing Gemini TTS Output")
    print("=" * 40)
    
    try:
        from gemini_service import hebrew_text_to_speech
        
        # Test short Hebrew phrase
        text = "שלום"
        print(f"Generating TTS for: '{text}'")
        
        # Generate TTS
        tts_audio = hebrew_text_to_speech(text)
        
        if not tts_audio:
            print("❌ TTS generation failed")
            return
            
        print(f"✅ Generated {len(tts_audio)} bytes")
        
        # Analyze the format
        format_info = analyze_wav_header(tts_audio)
        
        if format_info:
            # Save original
            with open("original_tts.wav", "wb") as f:
                f.write(tts_audio)
            print(f"💾 Saved original_tts.wav")
            
            # Test our conversion pipeline
            print("\n🔧 Testing conversion pipeline...")
            
            from gemini_service import GeminiService
            service = GeminiService()
            
            # Test PCM extraction
            pcm_data, detected_rate = service._extract_pcm_with_rate(tts_audio)
            if pcm_data and detected_rate:
                print(f"✅ Extracted {len(pcm_data)} bytes PCM, detected rate: {detected_rate}Hz")
                
                # Test resampling to 8kHz (for PCMU)
                resampled = service._resample_audio_to_target_rate_with_source(pcm_data, detected_rate, 8000)
                if resampled:
                    print(f"✅ Resampled to 8kHz: {len(resampled)} bytes")
                    
                    # Save resampled version
                    with wave.open("resampled_8khz.wav", "wb") as wav:
                        wav.setnchannels(1)
                        wav.setsampwidth(2)
                        wav.setframerate(8000)
                        wav.writeframes(resampled)
                    print(f"💾 Saved resampled_8khz.wav")
                    
                    # Test μ-law encoding
                    from audio_processor import linear_to_ulaw
                    ulaw_data = linear_to_ulaw(resampled)
                    print(f"✅ μ-law encoded: {len(ulaw_data)} bytes")
                    
                else:
                    print("❌ Resampling failed")
            else:
                print("❌ PCM extraction failed")
        
        print("\n💡 Files created:")
        print("1. original_tts.wav - Direct TTS output")
        print("2. resampled_8khz.wav - After our pipeline")
        print("\nCompare these files to identify quality issues.")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 