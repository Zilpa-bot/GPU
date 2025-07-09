import logging
import json
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, Request, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional

from .config import settings
from .telnyx_handler import telnyx_handler
from .connection_manager import connection_manager
from .models import TelnyxWebhookEvent

# Configure logging
logging.basicConfig(
    level=logging.INFO if not settings.debug else logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    logger.info("Starting Hebrew Voice AI Agent...")
    logger.info(f"Telnyx Phone Number: {settings.telnyx_phone_number}")
    logger.info(f"Server running on {settings.host}:{settings.port}")
    logger.info(f"Debug mode: {settings.debug}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Hebrew Voice AI Agent...")
    # Clean up any remaining connections
    for call_id in list(connection_manager.active_sessions.keys()):
        await connection_manager.cleanup_session(call_id)


# Create FastAPI app
app = FastAPI(
    title="Hebrew Voice AI Agent",
    description="A real-time Hebrew voice assistant using Telnyx, FastAPI, and Gemini 2.5",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Hebrew Voice AI Agent is running",
        "status": "healthy",
        "active_calls": connection_manager.get_active_calls_count(),
        "active_connections": connection_manager.get_active_connections_count()
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "settings": {
            "phone_number": settings.telnyx_phone_number,
            "debug_mode": settings.debug,
            "max_connections": settings.max_connections,
            "sample_rate": settings.sample_rate,
            "audio_encoding": settings.audio_encoding
        },
        "statistics": {
            "active_calls": connection_manager.get_active_calls_count(),
            "active_connections": connection_manager.get_active_connections_count()
        }
    }

@app.post("/telnyx/webhook")
async def telnyx_webhook(request: Request):
    """Handle incoming Telnyx webhook events"""
    try:
        # Get the raw request body
        body = await request.body()
        logger.info(f"Received webhook: {body}")
        
        # Parse JSON data
        event_data = json.loads(body)
        
        # Extract event info directly
        event_type = event_data.get("data", {}).get("event_type")
        call_control_id = event_data.get("data", {}).get("payload", {}).get("call_control_id")
        
        logger.info(f"Processing event: {event_type} for call: {call_control_id}")
        
        # Handle call.initiated directly
        if event_type == "call.initiated":
            payload = event_data["data"]["payload"]
            # Create call session
            call_data = {
                "call_control_id": call_control_id,
                "call_leg_id": payload.get("call_leg_id", ""),
                "call_session_id": payload.get("call_session_id", ""),
                "from": payload.get("from", ""),
                "to": payload.get("to", ""),
                "direction": payload.get("direction", "incoming"),
                "state": payload.get("state", "parked")
            }
            
            session = connection_manager.create_session(call_control_id, call_data)
            logger.info(f"Created session for call from {payload.get('from')} to {payload.get('to')}")
            
            # Answer the call immediately
            try:
                # Get the public domain from the webhook URL
                webhook_url = event_data.get("meta", {}).get("delivered_to", "")
                if webhook_url:
                    # Extract domain from webhook URL
                    domain = webhook_url.replace("https://", "").replace("/telnyx/webhook", "")
                    stream_url = f"wss://{domain}/media-stream?call_control_id={call_control_id}"
                else:
                    stream_url = f"wss://{request.headers.get('host', 'localhost:8000')}/media-stream?call_control_id={call_control_id}"
                
                await telnyx_handler.answer_call(
                    call_control_id=call_control_id,
                    stream_url=stream_url,
                    stream_track="both_tracks"
                )
                logger.info(f"✅ ANSWERED CALL {call_control_id} with streaming to {stream_url}")
                connection_manager.update_session_state(call_control_id, "answered_with_streaming")
            except Exception as e:
                logger.error(f"❌ Failed to answer call {call_control_id}: {e}")
        
        return {"status": "success", "message": f"Processed {event_type}"}
        
    except Exception as e:
        logger.error(f"Error handling webhook: {e}")
        logger.error(f"Request body: {body}")
        raise HTTPException(status_code=500, detail=str(e))

# @app.post("/telnyx/webhook")
# async def telnyx_webhook(request: Request):
#     """
#     Webhook endpoint for Telnyx Call Control events
#     This endpoint receives events like call.initiated, call.answered, call.hangup
#     """
#     try:
#         # Get the raw request body
#         raw_body = await request.body()
        
#         # Parse JSON
#         event_data = await request.json()
        
#         # Log the incoming event
#         logger.info(f"Received Telnyx webhook: {event_data.get('event_type', 'unknown')}")
        
#         # TODO: Verify webhook signature for security
#         # Telnyx provides webhook signature verification
#         # For now, we'll skip this for development
        
#         # Process the event
#         result = await telnyx_handler.handle_webhook_event(event_data)
        
#         return JSONResponse(content=result)
        
#     except Exception as e:
#         logger.error(f"Error processing webhook: {e}")
#         raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/media-stream")
async def media_stream_endpoint(
    websocket: WebSocket,
    call_control_id: Optional[str] = Query(None, description="Telnyx call control ID")
):
    """
    WebSocket endpoint for Telnyx media streaming
    This endpoint handles real-time audio streaming for voice calls
    """
    if not call_control_id:
        logger.error("WebSocket connection attempted without call_control_id")
        await websocket.close(code=1008, reason="call_control_id parameter required")
        return
    
    logger.info(f"WebSocket connection requested for call {call_control_id}")
    
    try:
        # Handle the media stream
        await telnyx_handler.handle_media_stream(websocket, call_control_id)
    except Exception as e:
        logger.error(f"Error in media stream for call {call_control_id}: {e}")
        try:
            await websocket.close(code=1011, reason="Internal server error")
        except:
            pass


@app.get("/calls")
async def get_active_calls():
    """Get information about active calls"""
    active_calls = []
    
    for call_id, session in connection_manager.active_sessions.items():
        call_info = {
            "call_control_id": session.call_control_id,
            "from_number": session.from_number,
            "to_number": session.to_number,
            "direction": session.direction,
            "state": session.state,
            "created_at": session.created_at.isoformat(),
            "has_websocket": call_id in connection_manager.active_connections,
            "conversation_length": len(session.conversation_history)
        }
        active_calls.append(call_info)
    
    return {
        "active_calls": active_calls,
        "total_count": len(active_calls)
    }


@app.get("/calls/{call_control_id}")
async def get_call_details(call_control_id: str):
    """Get detailed information about a specific call"""
    session = connection_manager.get_session(call_control_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Call not found")
    
    return {
        "call_control_id": session.call_control_id,
        "call_leg_id": session.call_leg_id,
        "call_session_id": session.call_session_id,
        "from_number": session.from_number,
        "to_number": session.to_number,
        "direction": session.direction,
        "state": session.state,
        "created_at": session.created_at.isoformat(),
        "has_websocket": call_control_id in connection_manager.active_connections,
        "conversation_history": session.conversation_history
    }


@app.post("/calls/{call_control_id}/hangup")
async def hangup_call(call_control_id: str):
    """Manually hang up a call"""
    try:
        result = await telnyx_handler.hangup_call(call_control_id)
        return {"status": "success", "message": f"Hangup command sent for call {call_control_id}"}
    except Exception as e:
        logger.error(f"Error hanging up call {call_control_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "type": str(type(exc).__name__)}
    )


def run_server():
    """Run the FastAPI server"""
    uvicorn.run(
        "src.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info" if not settings.debug else "debug"
    )


if __name__ == "__main__":
    run_server() 