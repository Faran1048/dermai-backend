"""
app/core/security.py

JWT token creation/verification and password hashing utilities.
Uses bcrypt directly (not via passlib) for Python 3.14 / bcrypt 4.x compatibility.
Uses python-jose for JWT signing/verification.
"""

from datetime import UTC, datetime, timedelta
from typing import Any

# pyrefly: ignore [missing-import]
import bcrypt
from jose import JWTError, jwt

from app.core.config import get_settings

settings = get_settings()

def hash_password(plain_password: str) -> str:
    """Return a bcrypt hash of the given plain-text password."""
    return bcrypt.hashpw(plain_password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Return True if the plain-text password matches the bcrypt hash."""
    return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))


# ── JWT tokens ────────────────────────────────────────────────────────────────
def create_access_token(
    subject: str | int,
    extra_claims: dict[str, Any] | None = None,
    expires_delta: timedelta | None = None,
) -> str:
    """
    Create a signed JWT access token.

    Args:
        subject: Usually the user's ID or email (stored as the `sub` claim).
        extra_claims: Any additional claims to embed in the token payload.
        expires_delta: Custom TTL. Falls back to settings.access_token_expire_minutes.

    Returns:
        Encoded JWT string.
    """
    expire = datetime.now(UTC) + (
        expires_delta or timedelta(minutes=settings.access_token_expire_minutes)
    )
    payload: dict[str, Any] = {
        "sub": str(subject),
        "exp": expire,
        "iat": datetime.now(UTC),
        **(extra_claims or {}),
    }
    return jwt.encode(payload, settings.secret_key, algorithm=settings.algorithm)


def decode_token(token: str) -> dict[str, Any]:
    """
    Decode and validate a JWT access token.

    Raises:
        JWTError: If the token is invalid, expired, or tampered with.
    """
    return jwt.decode(token, settings.secret_key, algorithms=[settings.algorithm])


def extract_user_id(token: str) -> int:
    """
    Decode the token and return the user ID stored in the `sub` claim.

    Raises:
        JWTError: If the token is invalid.
        ValueError: If `sub` is missing or not a valid integer.
    """
    payload = decode_token(token)
    sub = payload.get("sub")
    if sub is None:
        raise JWTError("Token is missing the 'sub' claim.")
    return int(sub)
