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
    """
    Upload an image to the private skin-images Supabase bucket
    and return a temporary signed URL.
    """

    if supabase is None:
        raise RuntimeError(
            "Supabase is not configured. Check SUPABASE_URL "
            "and SUPABASE_SERVICE_KEY."
        )

    import uuid

    safe_path = path.replace("\\", "/").lstrip("/")
    object_path = f"{bucket}/{safe_path}"

    try:
        supabase.storage.from_("skin-images").upload(
            path=object_path,
            file=file_bytes,
            file_options={
                "content-type": "image/jpeg",
                "upsert": "true"
            }
        )

        result = supabase.storage.from_("skin-images").create_signed_url(
            object_path,
            3600
        )

        signed_url = result.get("signedURL") or result.get("signedUrl")

        if not signed_url:
            raise RuntimeError("Could not generate signed URL.")

        if signed_url.startswith("/"):
            signed_url = f"{_SUPABASE_URL}/storage/v1{signed_url}"

        return signed_url

    except Exception as e:
        raise RuntimeError(f"Private Supabase image upload failed: {e}") from e

def upload_image_to_storage(
    object_path: str,
    file_bytes: bytes,
    content_type: str,
) -> str:
    """
    Upload an original user image to the private skin-images bucket.
    Returns the permanent storage object path, not a temporary URL.
    """
    if supabase is None:
        raise RuntimeError(
            "Supabase is not configured. Check SUPABASE_URL "
            "and SUPABASE_SERVICE_KEY."
        )

    safe_path = object_path.replace("\\", "/").lstrip("/")

    if not safe_path or ".." in safe_path.split("/"):
        raise ValueError("Invalid Supabase storage path.")

    try:
        supabase.storage.from_("skin-images").upload(
            path=safe_path,
            file=file_bytes,
            file_options={
                "content-type": content_type,
                "upsert": "false",
            },
        )
        return safe_path

    except Exception as e:
        raise RuntimeError(
            f"Original image upload to Supabase failed: {e}"
        ) from e


def delete_from_storage(object_path: str) -> None:
    """
    Delete an object from the private skin-images bucket.
    """
    if not object_path:
        return

    if supabase is None:
        raise RuntimeError(
            "Supabase is not configured. Cannot delete stored image."
        )

    safe_path = object_path.replace("\\", "/").lstrip("/")

    if not safe_path or ".." in safe_path.split("/"):
        raise ValueError("Invalid Supabase storage path.")

    try:
        supabase.storage.from_("skin-images").remove([safe_path])

    except Exception as e:
        raise RuntimeError(
            f"Supabase image deletion failed: {e}"
        ) from e

