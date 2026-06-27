"""
supabase_client.py

Client initializer and helper utility functions for Supabase integration.
Configured using SUPABASE_URL and SUPABASE_SERVICE_KEY from environment variables.
"""

from __future__ import annotations
import os
from dotenv import load_dotenv
from supabase import create_client, Client

# Ensure env variables are loaded
load_dotenv()

_SUPABASE_URL = os.getenv("SUPABASE_URL")
_SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

supabase: Client | None = None
if _SUPABASE_URL and _SUPABASE_KEY:
    try:
        supabase = create_client(_SUPABASE_URL, _SUPABASE_KEY)
    except Exception as e:
        print(f"Supabase client initialization failed: {e}")

import json
import base64

BASE_MOCK_DIR = os.path.join(os.getcwd(), "tmp", "mock-supabase")

def get_analysis(analysis_id: str) -> dict:
    """Fetch an analysis record by ID from local mock folder or 'analyses' table on Supabase."""
    # First, try to load from local mock directory
    local_path = os.path.join(BASE_MOCK_DIR, "analyses", f"{analysis_id}.json")
    if os.path.exists(local_path):
        try:
            with open(local_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error reading local mock analysis: {e}")

    # Fallback to Supabase
    if supabase is None:
        raise RuntimeError("Supabase client is not initialized and local mock file not found.")
    
    response = supabase.table("analyses").select("*").eq("id", analysis_id).execute()
    if not response.data:
        raise KeyError(f"Analysis with ID {analysis_id} not found.")
    return response.data[0]

import requests

def update_analysis(analysis_id: str, data: dict) -> None:
    """Update an analysis record by ID locally and on Supabase if credentials are set."""
    # 1. Save to local mock directory
    local_folder = os.path.join(BASE_MOCK_DIR, "analyses")
    os.makedirs(local_folder, exist_ok=True)
    local_path = os.path.join(local_folder, f"{analysis_id}.json")
    
    existing = {}
    if os.path.exists(local_path):
        try:
            with open(local_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except Exception:
            pass
            
    existing.update(data)
    try:
        with open(local_path, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2)
        print(f"Supabase mock updated locally — status: {data.get('status','?')}")
    except Exception as e:
        print(f"Failed to update mock analysis locally: {e}")

    # 2. Try to update Supabase if configured
    if _SUPABASE_URL and _SUPABASE_KEY:
        try:
            url = f"{_SUPABASE_URL}/rest/v1/analyses?id=eq.{analysis_id}"
            headers = {
                "apikey": _SUPABASE_KEY,
                "Authorization": f"Bearer {_SUPABASE_KEY}",
                "Content-Type": "application/json",
                "Prefer": "return=minimal"
            }
            resp = requests.patch(url, json=data, headers=headers, timeout=10)
            if resp.status_code not in (200, 204):
                print(f"Supabase update failed: {resp.status_code} {resp.text}")
            else:
                print("Supabase updated successfully.")
        except Exception as e:
            print(f"Supabase update exception (non-fatal): {e}")

def upload_to_storage(bucket: str, path: str, file_bytes: bytes) -> str:
    """Upload binary file bytes to Supabase Storage or return local base64 Data URI."""
    # If Supabase is available, try to upload
    if supabase is not None:
        try:
            supabase.storage.from_(bucket).upload(
                path=path,
                file=file_bytes,
                file_options={"content-type": "image/jpeg", "upsert": "true"}
            )
            return supabase.storage.from_(bucket).get_public_url(path)
        except Exception as e:
            print(f"Supabase storage upload failed: {e}")

    # Fallback to local Data URI representation
    encoded = base64.b64encode(file_bytes).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"
