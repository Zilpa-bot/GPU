import asyncio
import json
import logging
from typing import Dict, Optional, Set
from fastapi import WebSocket, WebSocketDisconnect
from datetime import datetime

from .models import CallSession, MediaStreamMessage
from .config import settings

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages WebSocket connections for Telnyx media streaming"""
    
    def __init__(self):
        # Active WebSocket connections: {call_control_id: WebSocket}
        self.active_connections: Dict[str, WebSocket] = {}
        
        # Active call sessions: {call_control_id: CallSession}
        self.active_sessions: Dict[str, CallSession] = {}
        
        # Lock for thread-safe operations
        self._lock = asyncio.Lock()
    
    async def connect(self, websocket: WebSocket, call_control_id: str) -> bool:
        """Register a WebSocket connection (already accepted) and associate it with a call"""
        try:
            # Note: WebSocket should already be accepted by the handler
            
            async with self._lock:
                # Check if we've reached max connections
                if len(self.active_connections) >= settings.max_connections:
                    logger.warning(f"Max connections ({settings.max_connections}) reached. Rejecting connection for call {call_control_id}")
                    await websocket.close(code=1008, reason="Server at capacity")
                    return False
                
                # Store the connection
                self.active_connections[call_control_id] = websocket
                
                # Update session if it exists
                if call_control_id in self.active_sessions:
                    self.active_sessions[call_control_id].websocket_connection = websocket
                
                logger.info(f"WebSocket connected for call {call_control_id}. Active connections: {len(self.active_connections)}")
                return True
                
        except Exception as e:
            logger.error(f"Error connecting WebSocket for call {call_control_id}: {e}")
            return False
    
    async def disconnect(self, call_control_id: str):
        """Remove a WebSocket connection"""
        async with self._lock:
            if call_control_id in self.active_connections:
                del self.active_connections[call_control_id]
                logger.info(f"WebSocket disconnected for call {call_control_id}. Active connections: {len(self.active_connections)}")
            
            # Clean up session WebSocket reference
            if call_control_id in self.active_sessions:
                self.active_sessions[call_control_id].websocket_connection = None
    
    async def send_message(self, call_control_id: str, message: dict) -> bool:
        """Send a message to a specific WebSocket connection"""
        if call_control_id not in self.active_connections:
            logger.warning(f"No active connection for call {call_control_id}")
            return False
        
        try:
            websocket = self.active_connections[call_control_id]
            await websocket.send_text(json.dumps(message))
            return True
        except Exception as e:
            logger.error(f"Error sending message to call {call_control_id}: {e}")
            await self.disconnect(call_control_id)
            return False
    
    async def send_audio(self, call_control_id: str, audio_data: str, stream_id: str = None, sequence_number: int = None) -> bool:
        """Send audio data to Telnyx via WebSocket with proper frame structure"""
        message = {
            "event": "media",
            "track": "outbound",
            "media": {
                "payload": audio_data
            }
        }
        
        # Add stream_id if provided
        if stream_id:
            message["stream_id"] = stream_id
            
        # Add sequence_number for frame ordering (Telnyx requirement)
        if sequence_number is not None:
            message["sequence_number"] = str(sequence_number)
            
        return await self.send_message(call_control_id, message)
    
    def create_session(self, call_control_id: str, call_data: dict) -> CallSession:
        """Create a new call session"""
        session = CallSession(
            call_control_id=call_control_id,
            call_leg_id=call_data.get("call_leg_id", ""),
            call_session_id=call_data.get("call_session_id", ""),
            from_number=call_data.get("from", ""),
            to_number=call_data.get("to", ""),
            direction=call_data.get("direction", ""),
            state=call_data.get("state", ""),
            created_at=datetime.now(),
            conversation_history=[]
        )
        
        self.active_sessions[call_control_id] = session
        logger.info(f"Created session for call {call_control_id}")
        return session
    
    def get_session(self, call_control_id: str) -> Optional[CallSession]:
        """Get an active call session"""
        return self.active_sessions.get(call_control_id)
    
    def update_session_state(self, call_control_id: str, state: str):
        """Update the state of a call session"""
        if call_control_id in self.active_sessions:
            self.active_sessions[call_control_id].state = state
            logger.info(f"Updated session state for call {call_control_id}: {state}")
    
    def add_conversation_entry(self, call_control_id: str, role: str, content: str):
        """Add an entry to the conversation history"""
        if call_control_id in self.active_sessions:
            self.active_sessions[call_control_id].conversation_history.append({
                "role": role,
                "content": content,
                "timestamp": datetime.now().isoformat()
            })
    
    async def cleanup_session(self, call_control_id: str):
        """Clean up a call session and its WebSocket connection"""
        async with self._lock:
            # Close WebSocket if still connected
            if call_control_id in self.active_connections:
                try:
                    websocket = self.active_connections[call_control_id]
                    await websocket.close()
                except:
                    pass
                del self.active_connections[call_control_id]
            
            # Remove session
            if call_control_id in self.active_sessions:
                del self.active_sessions[call_control_id]
                
            logger.info(f"Cleaned up session for call {call_control_id}")
    
    def get_active_calls_count(self) -> int:
        """Get the number of active calls"""
        return len(self.active_sessions)
    
    def get_active_connections_count(self) -> int:
        """Get the number of active WebSocket connections"""
        return len(self.active_connections)
    
    async def broadcast_system_message(self, message: dict):
        """Broadcast a system message to all active connections"""
        if not self.active_connections:
            return
        
        disconnected = []
        for call_control_id, websocket in self.active_connections.items():
            try:
                await websocket.send_text(json.dumps(message))
            except:
                disconnected.append(call_control_id)
        
        # Clean up disconnected sessions
        for call_control_id in disconnected:
            await self.disconnect(call_control_id)


# Global connection manager instance
connection_manager = ConnectionManager() 