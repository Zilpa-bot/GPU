import json
import base64
import asyncio
import logging
from typing import Optional, Dict, Any
import httpx
from fastapi import HTTPException, WebSocket, WebSocketDisconnect
from datetime import datetime

from .models import (
    TelnyxWebhookEvent, 
    TelnyxEventType, 
    MediaStreamMessage,
    AnswerCallCommand,
    StartStreamCommand,
    HangupCommand
)
from .config import settings
from .connection_manager import connection_manager
from .audio_processor import get_audio_processor, cleanup_audio_processor, linear_to_ulaw
from .gemini_service import (
    gemini_from_audio, 
    parse_gemini_response, 
    is_gemini_available,
    hebrew_text_to_speech,
    convert_tts_to_telnyx_format,
    chunk_audio_for_streaming,
    normalize_audio_dbfs
)

logger = logging.getLogger(__name__)


class TelnyxHandler:
    """Handles Telnyx webhook events and WebSocket media streaming"""
    
    def __init__(self):
        self.telnyx_api_base = "https://api.telnyx.com/v2"
        self.headers = {
            "Authorization": f"Bearer {settings.telnyx_api_key}",
            "Content-Type": "application/json"
        }
    
    def _construct_stream_url(self, call_control_id: str) -> str:
        """
        Construct the WebSocket streaming URL for a call.
        
        Important: Telnyx requires a publicly accessible WSS (secure WebSocket) URL.
        For development, use ngrok to expose your local server.
        
        Example ngrok setup:
        1. Install: npm install -g ngrok
        2. Run: ngrok http 8000
        3. Set PUBLIC_DOMAIN=abc123.ngrok.io in your .env file
        """
        if settings.public_domain:
            # Use the configured public domain (ngrok URL or production domain)
            return f"wss://{settings.public_domain}/media-stream?call_control_id={call_control_id}"
        elif settings.debug:
            # Development fallback (will NOT work with real Telnyx calls)
            logger.warning(f"Using localhost URL for streaming - this will NOT work with real Telnyx calls!")
            logger.warning(f"Set PUBLIC_DOMAIN in your .env to your ngrok URL for testing")
            return f"wss://{settings.host}:{settings.port}/media-stream?call_control_id={call_control_id}"
        else:
            # Production fallback
            logger.error("PUBLIC_DOMAIN not configured for production deployment!")
            raise ValueError("PUBLIC_DOMAIN setting is required for production use")
    
    async def send_telnyx_command(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Send a command to Telnyx Call Control API"""
        url = f"{self.telnyx_api_base}/calls/{command['call_control_id']}/actions/{command['command']}"
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    url,
                    headers=self.headers,
                    json=command,
                    timeout=10.0
                )
                response.raise_for_status()
                return response.json()
        except httpx.HTTPError as e:
            logger.error(f"Error sending Telnyx command {command['command']}: {e}")
            raise HTTPException(status_code=500, detail=f"Telnyx API error: {e}")
    
    async def answer_call(self, call_control_id: str, webhook_url: str = None, 
                         stream_url: str = None, stream_track: str = "both_tracks") -> Dict[str, Any]:
        """Answer an incoming call with bidirectional media streaming, explicitly requesting OPUS."""
        command = AnswerCallCommand(
            call_control_id=call_control_id,
            webhook_url=webhook_url,
            stream_url=stream_url,
            stream_track=stream_track,
            stream_bidirectional_mode="rtp",
            # Explicitly request OPUS with all required parameters to avoid silent PCMU fallback
            stream_bidirectional_codec="OPUS",
            stream_bidirectional_sampling_rate=16000
        )
        return await self.send_telnyx_command(command.dict())
    
    async def start_streaming(self, call_control_id: str, stream_url: str) -> Dict[str, Any]:
        """Start media streaming for a call"""
        command = StartStreamCommand(
            call_control_id=call_control_id,
            stream_url=stream_url,
            stream_track="both"
        )
        return await self.send_telnyx_command(command.dict())
    
    async def hangup_call(self, call_control_id: str) -> Dict[str, Any]:
        """Hangup a call"""
        command = HangupCommand(call_control_id=call_control_id)
        return await self.send_telnyx_command(command.dict())
    
    async def handle_webhook_event(self, event_data: Dict[str, Any]) -> Dict[str, str]:
        """Process incoming Telnyx webhook events"""
        try:
            # Parse the webhook event
            event = TelnyxWebhookEvent(**event_data)
            event_type = event.data.event_type
            call_control_id = event.data.payload.call_control_id
            logger.info(f"Received Telnyx event: {event_type} for call {call_control_id}")
            
            # Handle different event types
            if event_type == "call.initiated":
                await self._handle_call_initiated(event)
            elif event_type == "call.answered":
                await self._handle_call_answered(event)
            elif event_type == "streaming.started":
                await self._handle_streaming_started(event)
            elif event_type == "streaming.stopped":
                await self._handle_streaming_stopped(event)
            elif event_type == "call.hangup":
                await self._handle_call_hangup(event)
            else:
                logger.info(f"Unhandled event type: {event_type}")
            
            return {"status": "success", "message": f"Processed {event_type}"}
            
        except Exception as e:
            logger.error(f"Error processing webhook event: {e}")
            logger.error(f"Event data: {event_data}")
            raise HTTPException(status_code=500, detail=f"Error processing webhook: {e}")
    
    async def _handle_call_initiated(self, event: TelnyxWebhookEvent):
        """Handle incoming call initiation - Answer call and start streaming in one step"""
        payload = event.data.payload
        call_control_id = payload.call_control_id
        
        # Create a new call session
        call_data = {
            "call_control_id": call_control_id,
            "call_leg_id": payload.call_leg_id or "",
            "call_session_id": payload.call_session_id or "",
            "from": payload.from_ or "",
            "to": payload.to or "",
            "direction": payload.direction or "incoming",
            "state": payload.state or "parked"
        }
        
        session = connection_manager.create_session(call_control_id, call_data)
        logger.info(f"Created session for incoming call from {payload.from_} to {payload.to}")
        
        # Answer the call AND start streaming in one API call (more efficient)
        try:
            stream_url = self._construct_stream_url(call_control_id)
            await self.answer_call(
                call_control_id=call_control_id,
                stream_url=stream_url,
                stream_track="both_tracks"
            )
            logger.info(f"Answered call {call_control_id} with streaming enabled to {stream_url}")
            
            # Update session state to indicate streaming will start
            connection_manager.update_session_state(call_control_id, "answered_with_streaming")
            
        except Exception as e:
            logger.error(f"Failed to answer call {call_control_id} with streaming: {e}")
    
    async def _handle_call_answered(self, event: TelnyxWebhookEvent):
        """Handle call answered event - streaming should already be starting"""
        call_control_id = event.data.payload.call_control_id
        
        # Update session state
        connection_manager.update_session_state(call_control_id, "answered")
        logger.info(f"Call {call_control_id} answered - waiting for streaming.started event")
    
    async def _handle_streaming_started(self, event: TelnyxWebhookEvent):
        """Handle streaming started event - confirms media streaming is active"""
        call_control_id = event.data.payload.call_control_id
        
        # Update session state to indicate streaming is active
        connection_manager.update_session_state(call_control_id, "streaming_active")
        logger.info(f"Media streaming started for call {call_control_id} - WebSocket should connect soon")
        
        # Add conversation entry to mark the start of the session
        connection_manager.add_conversation_entry(
            call_control_id, 
            "system", 
            "Media streaming started - ready for voice interaction"
        )
    
    async def _handle_streaming_stopped(self, event: TelnyxWebhookEvent):
        """Handle streaming stopped event"""
        call_control_id = event.data.payload.call_control_id
        
        # Update session state
        connection_manager.update_session_state(call_control_id, "streaming_stopped")
        logger.info(f"Media streaming stopped for call {call_control_id}")
        
        # Add conversation entry
        connection_manager.add_conversation_entry(
            call_control_id, 
            "system", 
            "Media streaming stopped"
        )
    
    async def _handle_call_hangup(self, event: TelnyxWebhookEvent):
        """Handle call hangup event"""
        call_control_id = event.data.payload.call_control_id
        
        # Update session state
        connection_manager.update_session_state(call_control_id, "hangup")
        
        # Clean up audio processor
        cleanup_audio_processor(call_control_id)
        
        # Clean up the session and WebSocket connection
        await connection_manager.cleanup_session(call_control_id)
        logger.info(f"Cleaned up session for hangup call {call_control_id}")
    
    async def handle_media_stream(self, websocket: WebSocket, call_control_id: str):
        """Handle incoming WebSocket media stream from Telnyx"""
        await websocket.accept()
 
        # Store WebSocket connection in session for outbound audio
        session = connection_manager.get_session(call_control_id)
        if session:
            session.websocket_connection = websocket
            logger.debug(f"�� WebSocket attached to session {call_control_id}")
        else:
            logger.error(f"No session found for {call_control_id} when establishing WebSocket")
        
        # Initialize audio processor for this call
        audio_processor = get_audio_processor(call_control_id)
        audio_processor.set_utterance_callback(self._process_utterance)
        
        # Connect the WebSocket to our manager
        connected = await connection_manager.connect(websocket, call_control_id)
        if not connected:
            logger.error(f"Failed to register WebSocket for call {call_control_id}")
            return
        
        # IMMEDIATE WELCOME: Send welcome message right after WebSocket connection
        # This ensures caller hears something even if Telnyx events are delayed/missing
        logger.info(f"🎯 WebSocket connected - sending immediate welcome message for call {call_control_id}")
        await asyncio.sleep(0.5)  # Brief delay to ensure connection is stable
        await self._send_welcome_message(call_control_id)
        
        try:
            while True:
                # Receive message from Telnyx
                message_text = await websocket.receive_text()
                message_data = json.loads(message_text)
                
                # Log the received message for debugging
                event_type = message_data.get('event', 'unknown')
                logger.debug(f"Received media message for call {call_control_id}: {event_type}")
                
                # DEBUG: Log first few events to see exactly what Telnyx sends
                if not hasattr(self, '_logged_events'):
                    self._logged_events = set()
                
                if event_type not in self._logged_events:
                    logger.info(f"🔍 FIRST TIME seeing event '{event_type}' for call {call_control_id}: {message_data}")
                    self._logged_events.add(event_type)
                
                await self._process_media_message(call_control_id, message_data)
                
        except WebSocketDisconnect:
            logger.info(f"WebSocket disconnected for call {call_control_id}")
        except Exception as e:
            logger.error(f"Error in media stream for call {call_control_id}: {e}")
        finally:
            # Clean up audio processor
            cleanup_audio_processor(call_control_id)
            await connection_manager.disconnect(call_control_id)
    
    async def _process_media_message(self, call_control_id: str, message_data: Dict[str, Any]):
        """Process a media message, detecting the codec and handling stream events."""
        event_type = message_data.get('event')
        session = connection_manager.get_session(call_control_id)
        if not session:
            logger.error(f"No session found for {call_control_id}, cannot process event '{event_type}'.")
            return

        if event_type in ("connected", "start") and not session.greeted:
            logger.info(f"📡 Media stream started for call {call_control_id}. Sending welcome message.")
            connection_manager.update_session_state(call_control_id, "media_streaming")
            await self._send_welcome_message(call_control_id)
            # NOTE: session.greeted will be set to True only after successfully streaming at least one frame
        
        elif event_type == "media":
            # Debug logging to understand welcome message flow
            attempts = getattr(session, '_welcome_attempts', 0)
            logger.debug(
                f"🔍 Processing media event: greeted={session.greeted}, attempts={attempts}, codec={session.codec}"
            )

            # If codec is now known and greeting not yet succeeded, attempt up to 2 times
            if session.codec != "UNKNOWN" and not session.greeted:
                if attempts < 2:
                    logger.info(
                        f"📡 Attempt {attempts + 1}: sending welcome message for {call_control_id} (codec={session.codec})"
                    )
                    session._welcome_attempts = attempts + 1
                    await self._send_welcome_message(call_control_id)
                else:
                    logger.warning(
                        f"⚠️ Welcome message already attempted {attempts} times for {call_control_id} without success. Skipping further attempts."
                    )
            else:
                logger.debug(
                    f"🔍 Skipping welcome: greeted={session.greeted}, attempts={attempts}"
                )

            payload_b64 = message_data.get('media', {}).get('payload')
            if not payload_b64:
                return

            # Codec detection on the first inbound media frame
            if session.codec == "UNKNOWN" and message_data.get('media', {}).get('track') == 'inbound':
                try:
                    decoded_payload = base64.b64decode(payload_b64)
                    frame_size = len(decoded_payload)
                    
                    if 600 <= frame_size <= 700:
                        session.codec = "OPUS"
                        logger.info(f"✅ Codec detected for {call_control_id}: OPUS (frame size: {frame_size} bytes)")
                    elif 150 <= frame_size <= 170:
                        session.codec = "PCMU"
                        logger.info(f"✅ Codec detected for {call_control_id}: PCMU (frame size: {frame_size} bytes)")
                    else:
                        logger.warning(f"⚠️ Unknown frame size {frame_size}, cannot determine codec yet.")
                except Exception as e:
                    logger.error(f"Error decoding payload for codec detection: {e}")

            # Route to the audio processor
            audio_processor = get_audio_processor(call_control_id)
            await audio_processor.process_media_message(message_data)

        elif event_type == "stop":
            logger.info(f"⏹️ Media stream stopped for call {call_control_id}")
            connection_manager.update_session_state(call_control_id, "media_stopped")
        else:
            logger.debug(f"Ignoring media event '{event_type}' for call {call_control_id}")
    
    async def _process_utterance(self, call_control_id: str, utterance_audio: bytes):
        """Process a complete audio utterance from the VAD"""
        logger.info(f"🎤 Processing Hebrew utterance for call {call_control_id}: {len(utterance_audio)} bytes")
        
        # Check if Gemini is available
        if not is_gemini_available():
            logger.error("Gemini service not available - cannot process utterance")
            connection_manager.add_conversation_entry(
                call_control_id, 
                "system", 
                "Error: Gemini AI service not available"
            )
            return
        
        try:
            # Process audio through Gemini 2.5 Flash-Lite (audio-in/text-out)
            logger.info(f"🤖 Sending {len(utterance_audio)} bytes of 16kHz audio to Gemini for Hebrew processing...")
            gemini_response = gemini_from_audio(utterance_audio, mime_type="audio/pcm")
            
            # Log Gemini reply each turn; alert if empty (as per specs)
            logger.info(f"GEMINI → {repr(gemini_response)}")
            
            if not gemini_response or not gemini_response.strip():
                logger.error("🚨 ALERT: Empty response from Gemini!")
                connection_manager.add_conversation_entry(
                    call_control_id, 
                    "system", 
                    "🚨 ALERT: Empty response from Gemini AI"
                )
                return
            
            # Parse the response into transcript and answer
            transcript, answer = parse_gemini_response(gemini_response)
            
            logger.info(f"📝 Transcript: {transcript}")
            logger.info(f"🪓 Answer: {answer}")
            
            # Add conversation entries for both transcript and answer
            if transcript:
                connection_manager.add_conversation_entry(
                    call_control_id, 
                    "user", 
                    transcript
                )
            
            if answer:
                connection_manager.add_conversation_entry(
                    call_control_id, 
                    "assistant", 
                    answer
                )
            else:
                logger.warning("🚨 ALERT: Gemini returned transcript but no answer!")
            
            # Update dashboard with both fields
            self._update_dashboard(call_control_id, transcript, answer)
            
            # Step 7: Convert AI response to Hebrew TTS and send back to caller
            if answer:
                await self._generate_and_send_tts_response(call_control_id, answer)
            else:
                logger.warning("🚨 ALERT: Gemini returned transcript but no answer!")
            
            logger.info(f"✅ Hebrew conversation turn completed for call {call_control_id}")
            
        except Exception as e:
            logger.error(f"Error processing utterance with Gemini: {e}")
            connection_manager.add_conversation_entry(
                call_control_id, 
                "system", 
                f"Error processing utterance: {str(e)}"
            )
    
    def _update_dashboard(self, call_control_id: str, transcript: str, answer: str):
        """Update dashboard with transcript and AI response"""
        # Log the conversation turn for dashboard display
        logger.info(f"📊 Dashboard Update - Call {call_control_id}:")
        logger.info(f"🧑‍💼 לקוח: {transcript}")
        logger.info(f"🪓 קצבאי: {answer}")
        
        # Store for dashboard API (this will be expanded in Step 7)
        session = connection_manager.get_session(call_control_id)
        if session:
            if not hasattr(session, 'conversation_turns'):
                session.conversation_turns = []
            
            turn = {
                "timestamp": datetime.now().isoformat(),
                "transcript": transcript,
                "answer": answer
            }
            session.conversation_turns.append(turn)
            
            logger.debug(f"Stored conversation turn {len(session.conversation_turns)} for dashboard")
    
    async def _generate_and_send_tts_response(self, call_control_id: str, answer_text: str):
        """Generate Hebrew TTS for AI response and stream back to caller at 16kHz"""
        try:
            logger.info(f"🗣️ Generating Hebrew TTS for call {call_control_id}: {len(answer_text)} chars")
            
            # Generate Hebrew TTS audio (16kHz PCM as per specs)
            tts_audio = hebrew_text_to_speech(answer_text)
            
            # Log when hebrew_text_to_speech() returns 0-bytes and retry once
            if not tts_audio:
                logger.warning(f"⚠️ hebrew_text_to_speech() returned 0 bytes for AI response. Retrying once...")
                tts_audio = hebrew_text_to_speech(answer_text)
                
                if not tts_audio:
                    logger.error(f"🚨 hebrew_text_to_speech() failed twice for AI response. Using silence frame fallback.")
                    # Generate a short silence frame so the pipeline clears the speaking flag
                    # Fix overflow issue: Create proper 16-bit PCM silence (not raw zeros)
                    # 0.5 seconds of 16kHz PCM16 silence = 8000 samples × 2 bytes = 16000 bytes
                    import struct
                    silence_samples = [0] * 8000  # 0.5 seconds at 16kHz
                    tts_audio = struct.pack('<' + 'h' * len(silence_samples), *silence_samples)
                    logger.info(f"💭 Generated proper PCM16 silence frame fallback for AI response: {len(tts_audio)} bytes")
            
            if not tts_audio:
                logger.error("❌ Failed to generate any audio for AI response, skipping audio streaming.")
                return
            
            logger.info(f"✅ Generated TTS audio: {len(tts_audio)} bytes (16kHz)")
            
            # Normalize to -6 dBFS (as per specs)
            normalized_audio = normalize_audio_dbfs(tts_audio, target_dbfs=-6.0)
            logger.info(f"🔧 Normalized to -6 dBFS: {len(normalized_audio)} bytes (16kHz)")

            # Audio is already 16kHz PCM, no conversion/downsampling needed for OPUS stream.
            
            # Chunk audio for 16kHz streaming (640-byte frames)
            audio_chunks = chunk_audio_for_streaming(normalized_audio, chunk_duration_ms=20, sample_rate=16000)
            
            if not audio_chunks:
                logger.error("Failed to chunk 16kHz audio for streaming")
                return
            
            logger.info(f"✅ Created {len(audio_chunks)} × 640B frames for 16kHz streaming")
            
            # Stream audio chunks back to caller
            await self._stream_audio_to_caller(call_control_id, audio_chunks)
            
            connection_manager.add_conversation_entry(
                call_control_id, "system", f"TTS audio sent: {len(audio_chunks)} chunks ({len(normalized_audio)} bytes, 16kHz)"
            )
            
        except Exception as e:
            logger.error(f"Error in TTS response generation for call {call_control_id}: {e}")
            # ... existing error handling ...

    async def _stream_audio_to_caller(self, call_control_id: str, audio_data: bytes):
        """Chunks and streams audio data to the caller, respecting the negotiated codec."""
        session = connection_manager.get_session(call_control_id)
        if not session or not session.websocket_connection:
            logger.error(f"No active WebSocket session for call {call_control_id} to stream audio.")
            return

        frames_sent = 0
        try:
            # Determine chunk size and format based on the detected codec
            if session.codec == "OPUS":
                chunk_size = 640  # 20ms of 16kHz 16-bit PCM
                sleep_interval = 0.02
                audio_to_send = audio_data
            elif session.codec == "PCMU":
                chunk_size = 160  # 20ms of 8kHz 8-bit µ-law
                sleep_interval = 0.02
                logger.info("Converting TTS audio from L16/16kHz to µ-law/8kHz for PCMU stream.")
                audio_to_send = linear_to_ulaw(audio_data)
            else:
                logger.error(f"Cannot stream audio: unknown codec '{session.codec}' for call {call_control_id}.")
                return

            if not audio_to_send:
                logger.error("Audio data is empty after codec conversion, cannot stream.")
                return

            # Hardening: Assert that we're not sending µ-law silence when we think it's PCM
            if session.codec == "OPUS":
                assert not audio_to_send.startswith(b'\xff\xff'), "CRITICAL: µ-law silence detected in OPUS stream!"

            # Set speaking flag only while actively streaming frames
            session.speaking = True
            logger.info(f"🎤 AI started speaking for call {call_control_id}. Codec: {session.codec}")

            # Stream the audio in chunks
            for i in range(0, len(audio_to_send), chunk_size):
                chunk = audio_to_send[i:i+chunk_size]
                if len(chunk) < chunk_size:
                    # Pad the last chunk if necessary
                    if session.codec == "PCMU":
                        chunk += b'\xff' * (chunk_size - len(chunk)) # µ-law silence
                    else:
                        chunk += b'\x00' * (chunk_size - len(chunk)) # PCM silence
                
                payload = base64.b64encode(chunk).decode('ascii')
                session.last_outbound_sequence += 1
                
                media_message = {
                    "event": "media",
                    "track": "outbound",
                    "media": {"payload": payload}
                }
                await session.websocket_connection.send_json(media_message)
                frames_sent += 1
                
                # Set session.greeted = True only after successfully streaming at least one frame
                if frames_sent == 1 and not session.greeted:
                    session.greeted = True
                    logger.info(f"✅ Successfully sent first audio frame - session greeted for {call_control_id}")
                
                await asyncio.sleep(sleep_interval)  # 20ms pace

            logger.info(f"🎤 AI finished speaking for call {call_control_id}. Sent {frames_sent} frames.")

        except Exception as e:
            logger.error(f"Error during audio streaming for call {call_control_id}: {e}")
        finally:
            # Ensure speaking flag is always cleared, even if zero frames are sent
            session.speaking = False
            logger.info(f"🔇 Speaking flag cleared for call {call_control_id} (sent {frames_sent} frames)")

    async def _send_welcome_message(self, call_control_id: str):
        """Send a Hebrew welcome message to the caller using TTS (only once per call)"""
        session = connection_manager.get_session(call_control_id)
        if not session:
            logger.error(f"No session found for {call_control_id}, cannot send welcome message.")
            return

        try:
            logger.info(f"🗣️ Generating Hebrew welcome message for call {call_control_id}")
            
            # Generate Hebrew welcome message
            welcome_text = "שלום וברוכים הבאים לקצבייה שלנו בנתניה. איך אני יכול לעזור לכם היום?"
            welcome_audio = hebrew_text_to_speech(welcome_text)
            
            # Log when hebrew_text_to_speech() returns 0-bytes and retry once
            if not welcome_audio:
                logger.warning(f"⚠️ hebrew_text_to_speech() returned 0 bytes for welcome message. Retrying once...")
                welcome_audio = hebrew_text_to_speech(welcome_text)
                
                if not welcome_audio:
                    logger.error(f"🚨 hebrew_text_to_speech() failed twice for welcome message. Using silence frame fallback.")
                    # Generate a short silence frame so the pipeline clears the speaking flag
                    # Fix overflow issue: Create proper 16-bit PCM silence (not raw zeros)
                    # 0.5 seconds of 16kHz PCM16 silence = 8000 samples × 2 bytes = 16000 bytes
                    import struct
                    silence_samples = [0] * 8000  # 0.5 seconds at 16kHz
                    welcome_audio = struct.pack('<' + 'h' * len(silence_samples), *silence_samples)
                    logger.info(f"💭 Generated proper PCM16 silence frame fallback: {len(welcome_audio)} bytes")
            
            if welcome_audio:
                logger.info(f"✅ Generated welcome TTS: {len(welcome_audio)} bytes")
                
                # Normalize and stream the welcome message
                normalized_audio = normalize_audio_dbfs(welcome_audio, target_dbfs=-6.0)
                await self._stream_audio_to_caller(call_control_id, normalized_audio)
                
                # Record the interaction
                connection_manager.add_conversation_entry(
                    call_control_id, 
                    "assistant", 
                    welcome_text
                )
            else:
                logger.error(f"❌ Failed to generate any audio for welcome message (call {call_control_id})")
                
        except Exception as e:
            logger.error(f"Error sending welcome message for call {call_control_id}: {e}")
    
    async def _process_incoming_audio(self, call_control_id: str, message: MediaStreamMessage):
        """Process incoming audio from the caller"""
        if not message.payload:
            return
        
        try:
            # Decode the base64 audio payload
            audio_data = base64.b64decode(message.payload)
            
            # Log audio reception
            logger.debug(f"Received {len(audio_data)} bytes of audio from call {call_control_id}")
            
            # TODO: Implement speech-to-text processing
            # This will be implemented in later steps with the actual STT service
            
            # For now, just acknowledge receipt
            session = connection_manager.get_session(call_control_id)
            if session:
                # Placeholder: In real implementation, you'll convert audio to text here
                # and then process with Gemini AI
                pass
                
        except Exception as e:
            logger.error(f"Error processing audio for call {call_control_id}: {e}")
    
    async def send_audio_response(self, call_control_id: str, audio_data: bytes):
        """Send audio response back to the caller"""
        try:
            # Encode audio as base64
            audio_base64 = base64.b64encode(audio_data).decode('utf-8')
            
            # Send via WebSocket
            success = await connection_manager.send_audio(call_control_id, audio_base64)
            
            if success:
                logger.debug(f"Sent {len(audio_data)} bytes of audio to call {call_control_id}")
            else:
                logger.error(f"Failed to send audio to call {call_control_id}")
                
        except Exception as e:
            logger.error(f"Error sending audio response for call {call_control_id}: {e}")


# Global handler instance
telnyx_handler = TelnyxHandler() 