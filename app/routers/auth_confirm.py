"""
Adds /auth/confirm-user endpoint to auto-confirm new Supabase users.
Called by the frontend registration page after successful signup.
"""
import os
import logging
import requests as http_requests
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Auth"])

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")


class ConfirmUserRequest(BaseModel):
    user_id: str


@router.post("/auth/confirm-user", status_code=status.HTTP_200_OK)
async def confirm_user(payload: ConfirmUserRequest) -> dict:
    """
    Auto-confirms a newly registered user's email using the Supabase Admin API.
    Called by the frontend immediately after successful signUp() so users
    don't need to verify their email during development/demo.
    """
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Supabase admin credentials not configured."
        )

    headers = {
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
        "apikey": SUPABASE_SERVICE_KEY,
        "Content-Type": "application/json"
    }

    url = f"{SUPABASE_URL}/auth/v1/admin/users/{payload.user_id}"
    resp = http_requests.put(url, headers=headers, json={"email_confirm": True}, timeout=10)

    if resp.status_code == 200:
        logger.info(f"Auto-confirmed user: {payload.user_id}")
        return {"confirmed": True, "user_id": payload.user_id}
    else:
        logger.error(f"Failed to confirm user {payload.user_id}: {resp.text}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to confirm user: {resp.text[:200]}"
        )
