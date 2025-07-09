# Hebrew Voice AI Agent

A real-time Hebrew voice assistant that handles telephone calls via Telnyx WebSocket media streaming, using FastAPI and Google Gemini 2.5 Flash-Lite for conversational AI.

## 🚀 Features

- **Real-time Voice Processing**: WebSocket-based audio streaming with Telnyx
- **Hebrew Language Support**: Native Hebrew conversation capabilities with Gemini AI
- **Call Management**: Automatic call answering, session tracking, and cleanup
- **RESTful API**: Comprehensive REST endpoints for call monitoring and control
- **Scalable Architecture**: Async FastAPI with connection management for multiple simultaneous calls
- **Production Ready**: Configurable settings, proper logging, and error handling

## 📋 Prerequisites

1. **Telnyx Account**: 
   - Sign up at [Telnyx](https://telnyx.com)
   - Obtain API Key from Mission Control Portal
   - Purchase a phone number (preferably Israeli for Hebrew context)
   - Set up a Call Control Application

2. **Google Gemini API**:
   - Get API key from [Google AI Studio](https://makersuite.google.com/app/apikey)

## 🛠️ Installation

1. **Clone and Setup**:
   ```bash
   git clone <your-repo>
   cd hebrew-voice-ai-agent
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment**:
   ```bash
   cp env.example .env
   # Edit .env file with your API keys and settings
   ```

4. **Required Environment Variables**:
   ```env
   TELNYX_API_KEY=your_telnyx_api_key
   TELNYX_PHONE_NUMBER=your_telnyx_phone_number
   GEMINI_API_KEY=your_gemini_api_key
   ```

## 🏃‍♂️ Running the Server

### Development Mode
```bash
python run.py
```

### Production Mode
```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000
```

The server will start on `http://localhost:8000`

## 🌐 Public Access Setup

For Telnyx to reach your WebSocket endpoints, you need a public HTTPS URL:

### Using ngrok (Recommended for Development)
```bash
# Install ngrok
npm install -g ngrok
# or download from https://ngrok.com/

# Expose your local server
ngrok http 8000

# Note the HTTPS URL (e.g., https://abc123.ngrok.io)
```

### Update Telnyx Configuration
1. Go to Telnyx Mission Control Portal
2. Navigate to your Call Control Application
3. Set webhook URL to: `https://your-ngrok-url.ngrok.io/telnyx/webhook`
4. Update the `stream_url` in `src/telnyx_handler.py` line 130 to use your ngrok URL

## 📡 API Endpoints

### Health & Status
- `GET /` - Basic health check
- `GET /health` - Detailed system status

### Call Management
- `GET /calls` - List active calls
- `GET /calls/{call_control_id}` - Get call details
- `POST /calls/{call_control_id}/hangup` - Manually hangup call

### Telnyx Integration
- `POST /telnyx/webhook` - Webhook for Telnyx events
- `WS /media-stream` - WebSocket for audio streaming

## 🔧 Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Telnyx PSTN   │───▶│   FastAPI Server │───▶│  Gemini AI API  │
│                 │    │                  │    │                 │
│ ┌─────────────┐ │    │ ┌──────────────┐ │    │ ┌─────────────┐ │
│ │   Webhook   │─┼────┼▶│   Webhook    │ │    │ │   Hebrew    │ │
│ │   Events    │ │    │ │   Handler    │ │    │ │Conversation │ │
│ └─────────────┘ │    │ └──────────────┘ │    │ │   Engine    │ │
│                 │    │                  │    │ └─────────────┘ │
│ ┌─────────────┐ │    │ ┌──────────────┐ │    └─────────────────┘
│ │  WebSocket  │◀┼────┼▶│   Media      │ │              │
│ │Media Stream │ │    │ │   Stream     │ │              │
│ └─────────────┘ │    │ │   Manager    │ │              │
└─────────────────┘    │ └──────────────┘ │              │
                       │                  │              │
                       │ ┌──────────────┐ │              │
                       │ │ Connection   │◀┼──────────────┘
                       │ │ Manager      │ │
                       │ └──────────────┘ │
                       └──────────────────┘
```

## 📁 Project Structure

```
hebrew-voice-ai-agent/
├── src/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   ├── config.py            # Configuration management
│   ├── models.py            # Pydantic models
│   ├── connection_manager.py # WebSocket connection handling
│   └── telnyx_handler.py     # Telnyx API integration
├── requirements.txt         # Python dependencies
├── env.example             # Environment variables template
├── run.py                  # Startup script
└── README.md              # This file
```

## 🔧 Configuration Options

| Variable | Description | Default |
|----------|-------------|---------|
| `TELNYX_API_KEY` | Your Telnyx API key | Required |
| `TELNYX_PHONE_NUMBER` | Your Telnyx phone number | Required |
| `GEMINI_API_KEY` | Google Gemini API key | Required |
| `HOST` | Server host | `0.0.0.0` |
| `PORT` | Server port | `8000` |
| `DEBUG` | Debug mode | `false` |
| `MAX_CONNECTIONS` | Max concurrent WebSocket connections | `100` |
| `SAMPLE_RATE` | Audio sample rate | `8000` |
| `SYSTEM_PROMPT` | Hebrew AI system prompt | Default Hebrew prompt |

## 🐛 Debugging

### Enable Debug Logging
```env
DEBUG=true
```

### Check Server Status
```bash
curl http://localhost:8000/health
```

### Monitor Active Calls
```bash
curl http://localhost:8000/calls
```

### Test Webhook Endpoint
```bash
curl -X POST http://localhost:8000/telnyx/webhook \
  -H "Content-Type: application/json" \
  -d '{"event_type": "call.initiated", "data": {...}}'
```

## 🚧 Current Implementation Status

### ✅ Completed (Steps 2-3)
- FastAPI server setup with WebSocket support
- Telnyx webhook event handling  
- **Enhanced call handling with combined answer+stream**
- **Optimized one-step call answering with streaming**
- **Streaming event processing (streaming.started/stopped)**
- **Improved WebSocket connection management**
- Call session tracking and management
- REST API for monitoring and control
- Proper logging and error handling
- Configuration management
- **Public domain configuration for ngrok/production**

### 🔄 Coming Next (Steps 4-6)
- Speech-to-Text integration
- Google Gemini AI conversation engine  
- Text-to-Speech for Hebrew responses
- Dashboard UI for live monitoring

## 🧪 Testing Step 3 Implementation

### Automated Testing
```bash
# Run the Step 3 test suite
python test_step3.py
```

This test validates:
- Combined answer+stream functionality
- Streaming event processing  
- Call session management
- Conversation history tracking

### Manual Testing with Telnyx

1. **Start the server**: `python run.py`
2. **Expose publicly**: Use ngrok to create public HTTPS URL
   ```bash
   ngrok http 8000
   # Copy the HTTPS URL (e.g., https://abc123.ngrok.io)
   ```
3. **Configure environment**: Set `PUBLIC_DOMAIN=abc123.ngrok.io` in `.env`
4. **Update Telnyx**: Set webhook URL to `https://abc123.ngrok.io/telnyx/webhook`
5. **Call your number**: The system will:
   - Automatically answer the call
   - Start media streaming immediately
   - Establish WebSocket connection
   - Log all events with Hebrew welcome message ready

## 📊 Step 3 Call Flow

```
📞 Incoming Call
    ↓
🎯 call.initiated webhook → Answer + Stream (1 API call)
    ↓
✅ call.answered webhook → Confirmation
    ↓  
📡 streaming.started webhook → Media streaming active
    ↓
🔌 WebSocket connected → Ready for audio
    ↓
🎤 Media streaming → Audio packets flow both ways
```

## 🔒 Security Notes

- Always use HTTPS in production
- Implement webhook signature verification (TODO)
- Restrict CORS origins in production
- Use environment variables for sensitive data
- Consider rate limiting for public endpoints

## 🤝 Contributing

This is part of a step-by-step implementation. Each step builds upon the previous one:

1. ✅ Telnyx Account Setup (Completed)
2. ✅ FastAPI Server & WebSocket Setup (Current)
3. 🔄 Speech Recognition Integration (Next)
4. 🔄 Gemini AI Integration
5. 🔄 Text-to-Speech Implementation
6. 🔄 Dashboard UI Development

## 📝 License

This project is for educational and development purposes. 