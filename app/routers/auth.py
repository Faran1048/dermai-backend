"""
app/routers/auth.py

Authentication endpoints:
  POST /auth/register  — Create a new user account
  POST /auth/login     — Authenticate and receive a JWT token
  GET  /auth/me        — Get the currently authenticated user's profile
"""

from fastapi import APIRouter, Depends, status
# pyrefly: ignore [missing-import]
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.user import User
from app.schemas.user import (
    TokenResponse,
    UserLoginRequest,
    UserRegisterRequest,
    UserResponse,
)
from app.services.auth_service import (
    get_current_user,
    login_user,
    register_user,
)

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user",
)
async def register(
    payload: UserRegisterRequest,
    db: AsyncSession = Depends(get_db),
) -> UserResponse:
    """
    Create a new DermAI user account.

    - **full_name**: User's display name (2–255 characters)
    - **email**: Must be a valid, unique email address
    - **password**: Minimum 8 characters, must contain at least one digit
    """
    user = await register_user(payload, db)
    return UserResponse.model_validate(user)


@router.post(
    "/login",
    response_model=TokenResponse,
    summary="Login and receive a JWT token",
)
async def login(
    payload: UserLoginRequest,
    db: AsyncSession = Depends(get_db),
) -> TokenResponse:
    """
    Authenticate with email and password.

    Returns a Bearer JWT token valid for the duration set in `ACCESS_TOKEN_EXPIRE_MINUTES`.
    Include the token in the `Authorization: Bearer <token>` header for protected endpoints.
    """
    return await login_user(payload.email, payload.password, db)


@router.get(
    "/me",
    response_model=UserResponse,
    summary="Get current user profile",
)
async def get_me(
    current_user: User = Depends(get_current_user),
) -> UserResponse:
    """Return the profile of the currently authenticated user."""
    return UserResponse.model_validate(current_user)
