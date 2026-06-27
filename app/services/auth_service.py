"""
app/services/auth_service.py

Authentication business logic: user registration, login, and
current-user retrieval from a JWT Bearer token.
"""

import logging

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError
# pyrefly: ignore [missing-import]
from sqlalchemy import select
# pyrefly: ignore [missing-import]
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.security import (
    create_access_token,
    hash_password,
    verify_password,
    extract_user_id,
)
from app.core.config import get_settings
from app.models.user import User
from app.schemas.user import UserRegisterRequest, TokenResponse, UserResponse

logger = logging.getLogger(__name__)
settings = get_settings()

_bearer_scheme = HTTPBearer(auto_error=True)


# ── Registration ──────────────────────────────────────────────────────────────

async def register_user(
    payload: UserRegisterRequest,
    db: AsyncSession,
) -> User:
    """
    Create a new user account.

    Raises:
        HTTPException 409: If the email is already registered.
    """
    # Check for duplicate email
    existing = await db.scalar(select(User).where(User.email == payload.email))
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email address already exists.",
        )

    user = User(
        email=payload.email,
        full_name=payload.full_name,
        hashed_password=hash_password(payload.password),
        role=payload.role,
    )
    db.add(user)
    await db.flush()  # Flush to get the generated ID without committing yet

    # If the registered user is a dermatologist, initialize their profile record
    if user.role == "dermatologist":
        from app.models.consultation import Dermatologist
        dermatologist_profile = Dermatologist(
            id=user.id,
            specialization="General Dermatology",
            biography="Professional profile pending completion.",
            experience_years=0,
            rating=5.0,
            languages="English",
            availability="[]",
            price=150.0,
            is_approved=False
        )
        db.add(dermatologist_profile)
        await db.flush()

    await db.refresh(user)

    logger.info("New user registered: id=%d email=%s role=%s", user.id, user.email, user.role)
    return user


# ── Login ─────────────────────────────────────────────────────────────────────

async def login_user(
    email: str,
    password: str,
    db: AsyncSession,
) -> TokenResponse:
    """
    Authenticate a user and return a JWT token response.

    Raises:
        HTTPException 401: If credentials are invalid.
    """
    user = await db.scalar(select(User).where(User.email == email))

    # Use constant-time comparison to prevent timing attacks
    if user is None or not verify_password(password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is disabled. Please contact support.",
        )

    token = create_access_token(subject=user.id)
    logger.info("User logged in: id=%d", user.id)

    return TokenResponse(
        access_token=token,
        expires_in=settings.access_token_expire_minutes * 60,
        user=UserResponse.model_validate(user),
    )


# ── Current user dependency ───────────────────────────────────────────────────

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer_scheme),
    db: AsyncSession = Depends(get_db),
) -> User:
    """
    FastAPI dependency that extracts and validates the Bearer token,
    then loads and returns the authenticated user from the database.

    Raises:
        HTTPException 401: If the token is missing, invalid, or expired.
        HTTPException 404: If the user no longer exists in the database.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        user_id = extract_user_id(credentials.credentials)
    except (JWTError, ValueError):
        raise credentials_exception

    user = await db.get(User, user_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Authenticated user not found.",
        )
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is disabled.",
        )
    return user
