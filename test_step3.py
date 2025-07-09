#!/usr/bin/env python3
"""
Test script for Step 3: Combined Answer + Stream functionality
This script tests the enhanced call handling with one-step answer+streaming
"""

import json
import asyncio
import httpx
from datetime import datetime

# Test data for a realistic Telnyx call.initiated event
CALL_INITIATED_EVENT = {
    "data": {
        "call_control_id": "step3-test-call-456",
        "call_leg_id": "step3-test-leg-789",
        "call_session_id": "step3-test-session-123",
        "connection_id": "test-connection-def",
        "from": "+972501234567",
        "to": "+972555078904",
        "direction": "inbound",
        "state": "ringing",
        "created_at": datetime.now().isoformat() + "Z",
        "updated_at": datetime.now().isoformat() + "Z"
    },
    "event_type": "call.initiated",
    "id": "step3-test-event-abc",
    "occurred_at": datetime.now().isoformat() + "Z",
    "record_type": "event"
}

# Test data for streaming.started event
STREAMING_STARTED_EVENT = {
    "data": {
        "call_control_id": "step3-test-call-456",
        "call_leg_id": "step3-test-leg-789",
        "call_session_id": "step3-test-session-123",
        "connection_id": "test-connection-def",
        "from": "+972501234567",
        "to": "+972555078904",
        "direction": "inbound",
        "state": "answered",
        "created_at": datetime.now().isoformat() + "Z",
        "updated_at": datetime.now().isoformat() + "Z"
    },
    "event_type": "streaming.started",
    "id": "step3-streaming-event-xyz",
    "occurred_at": datetime.now().isoformat() + "Z",
    "record_type": "event"
}

async def test_step3_functionality():
    """Test the Step 3 enhanced call handling"""
    
    print("🧪 Testing Step 3: Combined Answer + Stream Functionality")
    print("=" * 60)
    
    base_url = "http://localhost:8000"
    
    async with httpx.AsyncClient() as client:
        
        # Test 1: Health check
        print("\n1️⃣ Testing server health...")
        try:
            response = await client.get(f"{base_url}/health")
            if response.status_code == 200:
                health_data = response.json()
                print(f"   ✅ Server healthy - Version: {health_data.get('version')}")
                print(f"   📊 Active calls: {health_data['statistics']['active_calls']}")
            else:
                print(f"   ❌ Health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"   ❌ Cannot connect to server: {e}")
            print("   💡 Make sure to run 'python run.py' first")
            return False
        
        # Test 2: Send call.initiated event (should trigger combined answer+stream)
        print("\n2️⃣ Testing call.initiated webhook (combined answer+stream)...")
        try:
            response = await client.post(
                f"{base_url}/telnyx/webhook",
                json=CALL_INITIATED_EVENT,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"   ✅ Call initiated processed: {result['message']}")
            else:
                print(f"   ❌ Call initiated failed: {response.status_code}")
                print(f"   📄 Response: {response.text}")
                return False
        except Exception as e:
            print(f"   ❌ Error sending call.initiated: {e}")
            return False
        
        # Test 3: Check if call session was created
        print("\n3️⃣ Checking call session creation...")
        try:
            response = await client.get(f"{base_url}/calls")
            if response.status_code == 200:
                calls_data = response.json()
                active_calls = calls_data.get('active_calls', [])
                
                # Find our test call
                test_call = None
                for call in active_calls:
                    if call['call_control_id'] == 'step3-test-call-456':
                        test_call = call
                        break
                
                if test_call:
                    print(f"   ✅ Call session created successfully")
                    print(f"   📞 From: {test_call['from_number']}")
                    print(f"   📞 To: {test_call['to_number']}")
                    print(f"   📊 State: {test_call['state']}")
                    print(f"   🔌 WebSocket ready: {test_call['has_websocket']}")
                else:
                    print(f"   ❌ Test call session not found")
                    print(f"   📋 Found {len(active_calls)} other calls")
                    return False
            else:
                print(f"   ❌ Failed to get calls: {response.status_code}")
                return False
        except Exception as e:
            print(f"   ❌ Error checking calls: {e}")
            return False
        
        # Test 4: Send streaming.started event
        print("\n4️⃣ Testing streaming.started webhook...")
        try:
            response = await client.post(
                f"{base_url}/telnyx/webhook",
                json=STREAMING_STARTED_EVENT,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"   ✅ Streaming started processed: {result['message']}")
            else:
                print(f"   ❌ Streaming started failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"   ❌ Error sending streaming.started: {e}")
            return False
        
        # Test 5: Check call details with conversation history
        print("\n5️⃣ Checking call details and conversation history...")
        try:
            response = await client.get(f"{base_url}/calls/step3-test-call-456")
            if response.status_code == 200:
                call_details = response.json()
                print(f"   ✅ Call details retrieved")
                print(f"   📊 Current state: {call_details['state']}")
                
                conversation = call_details.get('conversation_history', [])
                print(f"   💬 Conversation entries: {len(conversation)}")
                
                for i, entry in enumerate(conversation):
                    print(f"      {i+1}. [{entry['role']}] {entry['content']}")
                
            else:
                print(f"   ❌ Failed to get call details: {response.status_code}")
                return False
        except Exception as e:
            print(f"   ❌ Error getting call details: {e}")
            return False
        
        # Test 6: Manual cleanup
        print("\n6️⃣ Cleaning up test call...")
        try:
            response = await client.post(f"{base_url}/calls/step3-test-call-456/hangup")
            if response.status_code == 200:
                print(f"   ✅ Test call cleaned up successfully")
            else:
                print(f"   ⚠️ Manual cleanup failed, but this is OK for testing")
        except Exception as e:
            print(f"   ⚠️ Cleanup error (not critical): {e}")
    
    print("\n🎉 Step 3 Enhanced Functionality Test Complete!")
    print("=" * 60)
    
    print("\n📋 What was tested:")
    print("  ✅ Server health and connectivity")  
    print("  ✅ Combined answer+stream call handling")
    print("  ✅ Call session creation and state management")
    print("  ✅ Streaming started event processing")
    print("  ✅ Conversation history tracking")
    print("  ✅ RESTful API endpoints")
    
    print("\n🚀 Step 3 Implementation Ready!")
    print("  📞 Calls will be answered automatically with streaming enabled")
    print("  🔌 WebSocket connections will be established immediately")
    print("  💬 Conversation history is tracked from the start")
    print("  🌐 Configure PUBLIC_DOMAIN for real Telnyx testing")
    
    return True

if __name__ == "__main__":
    asyncio.run(test_step3_functionality()) 