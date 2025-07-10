#!/usr/bin/env python3
"""
Test Vertex AI Gemini integration with Hebrew text
"""

from vertexai.generative_models import GenerativeModel
import vertexai
import os

def test_gemini_hebrew():
    """Test Gemini with Hebrew text"""
    
    print("🧪 Testing Vertex AI Gemini integration...")
    
    # Initialize Vertex AI
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    print(f"📋 Project ID: {project_id}")
    
    if not project_id:
        print("❌ GOOGLE_CLOUD_PROJECT environment variable not set")
        return False
    
    try:
        # Try different regions with the correct Gemini 2.5 Flash-Lite model
        regions = ["us-central1", "us-east4", "us-west1", "europe-west1", "global"]
        models = [
            "gemini-2.5-flash-lite-preview",
            "gemini-2.5-flash-lite",
            "publishers/google/models/gemini-2.5-flash-lite-preview"
        ]
        
        for region in regions:
            for model_name in models:
                try:
                    print(f"🔄 Trying region: {region}, model: {model_name}")
                    vertexai.init(project=project_id, location=region)
                    model = GenerativeModel(model_name)
                    # Test if model works
                    test_response = model.count_tokens("test")
                    print(f"✅ Success! Using {model_name} in {region}")
                    break
                except Exception as e:
                    print(f"❌ Failed {model_name} in {region}: {str(e)[:100]}...")
                    continue
            else:
                continue
            break
        else:
            print("\n❌ Gemini 2.5 Flash-Lite not found in any region.")
            print("💡 Possible solutions:")
            print("   1. Check if your project has access to Gemini 2.5 Flash-Lite preview")
            print("   2. Request access to preview models in Google Cloud Console")
            print("   3. Enable Vertex AI API in your project")
            print("   4. Try using gemini-1.5-flash as fallback")
            raise Exception("Gemini 2.5 Flash-Lite not available")
        print("✅ Gemini model created")
        
        # Test token counting with Hebrew text
        hebrew_text = "שלום"
        token_count = model.count_tokens(hebrew_text).total_tokens
        print(f"🔤 Hebrew text: '{hebrew_text}'")
        print(f"🎯 Token count: {token_count}")
        
        # Test simple text generation
        response = model.generate_content(f"Translate this Hebrew word to English: {hebrew_text}")
        print(f"🤖 Gemini response: {response.text}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing Gemini: {e}")
        return False

if __name__ == "__main__":
    test_gemini_hebrew() 