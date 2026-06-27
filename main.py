"""
main.py

DermAI Backend - FastAPI entry point.
Flat project layout (no app/ package) for direct, simple deployment.
Run with: uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# ── Load environment variables ────────────────────────────────────────────────
load_dotenv()
print("DermAI Backend starting...")

# ── App ───────────────────────────────────────────────────────────────────────
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from app.core.database import init_db

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    print("Initializing database...")
    await init_db()
    print("Database ready.")
    print("Pre-loading local CNN classifier model...")
    try:
        from algorithms.cnn_classifier import _load_local_model
        _load_local_model()
        print("CNN classifier model pre-loaded successfully.")
    except Exception as e:
        print(f"Warning: CNN classifier model pre-load failed: {e}")
    yield

app = FastAPI(
    title="DermAI Backend",
    version="1.0.0",
    description="AI-powered dermatology analysis API.",
    docs_url="/docs",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health", tags=["System"])
async def health() -> dict:
    """Returns service status. No auth required."""
    return {"status": "ok", "service": "DermAI Backend"}


# ── AI Pipeline Router ────────────────────────────────────────────────────────
from app.routers.analyze import router as analyze_router
app.include_router(analyze_router)

# ── Skin Age & Change Detection Routers ───────────────────────────────────────
from app.routers.skin_age import router as skin_age_router
app.include_router(skin_age_router)
app.include_router(skin_age_router, prefix="/api/v1")

from app.routers.change_detection import router as change_detection_router
app.include_router(change_detection_router)
app.include_router(change_detection_router, prefix="/api/v1")

# ── Auth Admin Router (auto-confirm new users) ────────────────────────────────
from app.routers.auth_confirm import router as auth_confirm_router
app.include_router(auth_confirm_router)

# ── Consultation & Telemedicine Router ────────────────────────────────────────
from app.routers.consultation import router as consultation_router
app.include_router(consultation_router, prefix="/api/v1")

# ── Billing & Subscription Router ─────────────────────────────────────────────
from app.routers.billing import router as billing_router
app.include_router(billing_router, prefix="/api/v1")

# ── Other Platform Routers ───────────────────────────────────────────────────
from app.routers import auth, analysis, history, reports, mock_supabase
app.include_router(auth.router, prefix="/api/v1")
app.include_router(analysis.router, prefix="/api/v1")
app.include_router(history.router, prefix="/api/v1")
app.include_router(reports.router, prefix="/api/v1")
app.include_router(mock_supabase.router, prefix="/api/v1")

