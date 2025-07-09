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
from .audio_processor import get_audio_processor, cleanup_audio_processor

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
        """Answer an incoming call with optional media streaming"""
        command = AnswerCallCommand(
            call_control_id=call_control_id,
            webhook_url=webhook_url,
            stream_url=stream_url,
            stream_track=stream_track
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
        """Handle WebSocket media streaming from Telnyx"""
        logger.info(f"Starting media stream handler for call {call_control_id}")
        
        # Initialize audio processor for this call
        audio_processor = get_audio_processor(call_control_id)
        audio_processor.set_utterance_callback(self._process_utterance)
        
        # Connect the WebSocket
        connected = await connection_manager.connect(websocket, call_control_id)
        if not connected:
            logger.error(f"Failed to connect WebSocket for call {call_control_id}")
            return
        
        try:
            while True:
                # Receive message from Telnyx
                message_text = await websocket.receive_text()
                message_data = json.loads(message_text)
                
                # Log the received message for debugging
                logger.debug(f"Received media message for call {call_control_id}: {message_data.get('event', 'unknown')}")
                
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
        """Process incoming media stream messages"""
        try:
            message = MediaStreamMessage(**message_data)
            
            if message.event == "connected":
                # Telnyx sends {"event": "connected", "version": "1.0.0"} when WebSocket is established
                version = message_data.get("version", "unknown")
                logger.info(f"✅ Media stream WebSocket connected for call {call_control_id} (version: {version})")
                
                # Update session state to indicate WebSocket is connected
                connection_manager.update_session_state(call_control_id, "websocket_connected")
                
                # Add conversation entry
                connection_manager.add_conversation_entry(
                    call_control_id, 
                    "system", 
                    f"WebSocket connected (Telnyx version: {version})"
                )
                
                # Send a welcome message (you can implement TTS here later)
                await self._send_welcome_message(call_control_id)
                
            elif message.event == "start":
                logger.info(f"📡 Media stream data transmission started for call {call_control_id}")
                connection_manager.update_session_state(call_control_id, "media_streaming")
                
                # Add conversation entry
                connection_manager.add_conversation_entry(
                    call_control_id, 
                    "system", 
                    "Audio streaming started - ready to receive voice"
                )
                
            elif message.event == "media":
                # Handle incoming audio from the caller with VAD
                audio_processor = get_audio_processor(call_control_id)
                await audio_processor.process_media_message(message_data)
                
            elif message.event == "stop":
                logger.info(f"⏹️ Media stream stopped for call {call_control_id}")
                connection_manager.update_session_state(call_control_id, "media_stopped")
                
            else:
                logger.debug(f"Received unhandled media event '{message.event}' for call {call_control_id}")
                
        except Exception as e:
            logger.error(f"Error processing media message for call {call_control_id}: {e}")
            logger.error(f"Raw message data: {message_data}")
    
    async def _process_utterance(self, call_control_id: str, utterance_audio: bytes):
        """Process a complete utterance detected by VAD"""
        logger.info(f"🎤 Processing Hebrew utterance for call {call_control_id}: {len(utterance_audio)} bytes")
        
        # Add conversation entry
        connection_manager.add_conversation_entry(
            call_control_id, 
            "user", 
            f"[Audio utterance: {len(utterance_audio)} bytes]"
        )
        
        # TODO: Step 5 - Send to speech-to-text service for Hebrew recognition
        # For now, just log the utterance
        logger.info(f"📝 Hebrew utterance ready for STT processing")
        
        # TODO: Step 6 - Process with Gemini AI
        # TODO: Step 7 - Convert AI response to Hebrew TTS
        # TODO: Step 8 - Send audio response back to caller

    async def _send_welcome_message(self, call_control_id: str):
        """Send a welcome message to the caller"""
        # For now, just log. In later steps, you'll implement TTS
        session = connection_manager.get_session(call_control_id)
        if session:
            welcome_text = "שלום, אני עוזר וירטואלי. איך אני יכול לעזור לך היום?"  # Hebrew welcome
            connection_manager.add_conversation_entry(call_control_id, "assistant", welcome_text)
            logger.info(f"Welcome message queued for call {call_control_id}: {welcome_text}")
            
            # TODO: Convert text to speech and send audio
            # This will be implemented in later steps
    
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