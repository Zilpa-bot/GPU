from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class TelnyxEventType(str, Enum):
    CALL_INITIATED = "call.initiated"
    CALL_ANSWERED = "call.answered"
    CALL_BRIDGED = "call.bridged"
    CALL_HANGUP = "call.hangup"
    CALL_MACHINE_GREETING_ENDED = "call.machine_greeting_ended"
    STREAMING_STARTED = "streaming.started"
    STREAMING_STOPPED = "streaming.stopped"


class TelnyxEventPayload(BaseModel):
    """Telnyx webhook event payload (flexible for different event types)"""
    call_control_id: str
    call_leg_id: Optional[str] = None
    call_session_id: Optional[str] = None
    client_state: Optional[str] = None
    connection_id: Optional[str] = None
    from_: Optional[str] = Field(default=None, alias="from")
    to: Optional[str] = None
    direction: Optional[str] = None
    state: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    hangup_cause: Optional[str] = None
    hangup_source: Optional[str] = None
    # Allow any additional fields for different event types
    
    class Config:
        extra = "allow"


class TelnyxEventData(BaseModel):
    """Telnyx webhook event data structure"""
    event_type: str
    id: str
    occurred_at: str
    payload: TelnyxEventPayload
    record_type: str = "event"


class TelnyxWebhookEvent(BaseModel):
    """Telnyx webhook event structure"""
    data: TelnyxEventData
    meta: Optional[Dict[str, Any]] = None


class MediaStreamMessage(BaseModel):
    """WebSocket media stream message"""
    event: str
    stream_id: Optional[str] = None
    call_control_id: Optional[str] = None
    media: Optional[Dict[str, Any]] = None
    payload: Optional[str] = None  # Base64 encoded audio
    timestamp: Optional[str] = None


class CallSession(BaseModel):
    """Represents the state of a single call session"""
    call_control_id: str
    call_leg_id: str
    call_session_id: str
    from_number: str
    to_number: str
    direction: str
    state: str
    created_at: datetime
    websocket_connection: Optional[Any] = None
    conversation_history: list = []
    greeted: bool = False
    speaking: bool = False
    last_outbound_sequence: int = 0
    codec: str = "UNKNOWN"  # "UNKNOWN", "OPUS", or "PCMU"
    _welcome_attempts: int = 0  # Initialize to 0 for proper welcome message logic
    
    class Config:
        arbitrary_types_allowed = True


class TelnyxCommand(BaseModel):
    """Base class for Telnyx command requests"""
    command: str
    call_control_id: str
    client_state: Optional[str] = None


class AnswerCallCommand(TelnyxCommand):
    """Answer an incoming call with bidirectional media streaming"""
    command: str = "answer"
    webhook_url: Optional[str] = None
    webhook_url_method: Optional[str] = "POST"
    stream_url: Optional[str] = None
    stream_track: Optional[str] = None  # "inbound", "outbound", or "both_tracks"
    stream_bidirectional_mode: Optional[str] = "rtp"  # Enable bidirectional streaming
    stream_bidirectional_codec: Optional[str] = "OPUS"  # Default to OPUS for high quality
    stream_bidirectional_sampling_rate: Optional[int] = 16000  # 16kHz for OPUS


class StartStreamCommand(TelnyxCommand):
    """Start media streaming for a call"""
    command: str = "streaming_start"
    stream_url: str
    stream_track: str = "both"  # "inbound", "outbound", or "both"
    enable_dialogflow: bool = False


class HangupCommand(TelnyxCommand):
    """Hangup a call"""
    command: str = "hangup"


class AIResponse(BaseModel):
    """AI generated response"""
    text: str
    audio_data: Optional[bytes] = None
    language: str = "he"  # Hebrew
    
    class Config:
        arbitrary_types_allowed = True 