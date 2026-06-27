import requests
import json
import time

API_URL = "http://127.0.0.1:8000"

def test_pipeline():
    print("=== DermAI Integration Test ===")
    
    # 1. Register a new user
    email = f"agent_test_{int(time.time())}@gmail.com"
    register_payload = {
        "full_name": "Agent Validator",
        "email": email,
        "password": "Str0ngPass123"
    }
    
    print(f"1. Registering user {email}...")
    resp = requests.post(f"{API_URL}/api/v1/auth/register", json=register_payload)
    print("Status:", resp.status_code)
    if resp.status_code != 201:
        print("Failed to register:", resp.text)
        return
    user_data = resp.json()
    user_id = str(user_data["id"])
    print("Registered successfully. User ID:", user_id)
    
    # 2. Login to get JWT Token
    login_payload = {
        "email": email,
        "password": "Str0ngPass123"
    }
    print("2. Logging in...")
    resp = requests.post(f"{API_URL}/api/v1/auth/login", json=login_payload)
    print("Status:", resp.status_code)
    if resp.status_code != 200:
        print("Failed to login:", resp.text)
        return
    token_data = resp.json()
    token = token_data["access_token"]
    print("JWT Token acquired successfully.")
    
    # 3. Create mock analysis entry in mock-supabase
    analysis_id = f"test_analysis_{int(time.time())}"
    image_url = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    
    analysis_payload = {
        "id": analysis_id,
        "user_id": user_id,
        "image_url": image_url,
        "status": "processing",
        "age": 35,
        "skin_type": 3,
        "body_location": "back",
        "location_int": 3,
        "duration_weeks": 2,
        "symptoms": {
            "itching": True,
            "bleeding": False,
            "growing": True,
            "color_change": False,
            "painful": False
        },
        "changes_description": "Itches and grows slowly"
    }
    
    print(f"3. Inserting mock analysis '{analysis_id}' in local database...")
    resp = requests.post(f"{API_URL}/api/v1/mock-supabase/analyses", json=analysis_payload)
    print("Status:", resp.status_code)
    if resp.status_code != 200:
        print("Failed to create mock analysis:", resp.text)
        return
    print("Mock analysis created.")
    
    # 4. Trigger backend 11-step pipeline
    pipeline_payload = {
        "image_url": image_url,
        "analysis_id": analysis_id,
        "intake_form": {
            "age": 35,
            "skin_type": 3,
            "duration_weeks": 2,
            "body_location": "back",
            "location_int": 3,
            "symptoms": {
                "itching": True,
                "bleeding": False,
                "growing": True,
                "color_change": False,
                "painful": False
            }
        }
    }
    
    print("4. Triggering AI analysis pipeline (/analyze)...")
    resp = requests.post(f"{API_URL}/analyze", json=pipeline_payload)
    print("Status:", resp.status_code)
    if resp.status_code != 202:
        print("Pipeline failed to start:", resp.text)
        return
    print("Pipeline accepted task in background.")
    
    # 5. Poll the status from local mock database until complete or failed
    print("5. Polling mock analysis status...")
    max_attempts = 15
    for attempt in range(max_attempts):
        time.sleep(3)
        resp = requests.get(f"{API_URL}/api/v1/mock-supabase/analyses/{analysis_id}")
        if resp.status_code == 200:
            data = resp.json()
            status = data.get("status")
            print(f"Attempt {attempt + 1}: status = '{status}'")
            if status == "complete":
                print("\n=== Integration Test Succeeded! ===")
                print(f"Classified Lesion: {data.get('predicted_class')}")
                print(f"Confidence: {data.get('confidence')}")
                print(f"Risk Level: {data.get('risk_level')}")
                print(f"Compound Risk Score: {data.get('compound_risk_score')}")
                print(f"Referral Pathway: {data.get('referral', {}).get('referral_path')}")
                print(f"AI Clinical Notes: {data.get('referral', {}).get('ai_clinical_notes')}")
                return
            elif status == "failed" or status == "error":
                print("Pipeline failed in background.")
                return
        else:
            print(f"Failed to fetch mock analysis status: {resp.status_code}")
            
    print("Test timed out waiting for pipeline completion.")

if __name__ == "__main__":
    test_pipeline()
