#!/usr/bin/env python3
"""
Hebrew Voice AI Agent - Startup Script
Run this script to start the FastAPI server
"""

import sys
import os
import warnings

# Suppress specific deprecation warnings from webrtcvad
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API", category=UserWarning)

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.main import run_server

if __name__ == "__main__":
    print("Starting Hebrew Voice AI Agent...")
    print("Make sure you have configured your .env file with Telnyx and Gemini API keys")
    print("Server will start on http://localhost:8000")
    print("WebSocket endpoint: ws://localhost:8000/media-stream")
    print("Webhook endpoint: http://localhost:8000/telnyx/webhook")
    print("")
    
    try:
        run_server()
    except KeyboardInterrupt:
        print("\nShutting down Hebrew Voice AI Agent...")
    except Exception as e:
        print(f"Error starting server: {e}")
        sys.exit(1) 