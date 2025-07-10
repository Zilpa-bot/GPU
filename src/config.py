import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field, ConfigDict


class Settings(BaseSettings):
    """Configuration settings for the Hebrew Voice AI Agent"""
    
    # Telnyx Configuration
    telnyx_api_key: str = Field(..., env="TELNYX_API_KEY")
    telnyx_webhook_secret: Optional[str] = Field(None, env="TELNYX_WEBHOOK_SECRET")
    telnyx_phone_number: str = Field(..., env="TELNYX_PHONE_NUMBER")
    
    # Google Gemini Configuration
    gemini_api_key: Optional[str] = Field(None, env="GEMINI_API_KEY")
    google_ai_api_key: Optional[str] = Field(None, env="GOOGLE_AI_API_KEY")
    gemini_model: str = Field("models/gemini-2.5-flash-lite-preview-06-17", env="GEMINI_MODEL")
    
    # Google Cloud / Vertex AI Configuration
    google_cloud_project: Optional[str] = Field(None, env="GOOGLE_CLOUD_PROJECT")
    google_application_credentials: Optional[str] = Field(None, env="GOOGLE_APPLICATION_CREDENTIALS")
    
    # Server Configuration
    host: str = Field("0.0.0.0", env="HOST")
    port: int = Field(8000, env="PORT")
    debug: bool = Field(False, env="DEBUG")
    public_domain: Optional[str] = Field(None, env="PUBLIC_DOMAIN")  # e.g., "your-app.com" or ngrok URL
    
    # WebSocket Configuration
    websocket_timeout: int = Field(300, env="WEBSOCKET_TIMEOUT")  # 5 minutes
    max_connections: int = Field(100, env="MAX_CONNECTIONS")
    
    # Audio Configuration
    sample_rate: int = Field(8000, env="SAMPLE_RATE")  # Telnyx uses 8kHz
    audio_encoding: str = Field("mulaw", env="AUDIO_ENCODING")  # Telnyx default
    
    # Hebrew AI Configuration
    system_prompt: str = Field(
        "אתה עוזר וירטואלי חכם שמדבר עברית. "
        "אתה מסוגל לעזור בשאלות שונות ולנהל שיחה טבעית בעברית. "
        "השב בקצרה ובבהירות.",
        env="SYSTEM_PROMPT"
    )
    
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"  # Ignore extra environment variables
    )


# Global settings instance
settings = Settings() 