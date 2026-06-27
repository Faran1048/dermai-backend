"""
app/main.py

DermAI Backend — FastAPI application entry point.

Startup sequence:
  1. Load settings from .env
  2. Initialise the database (create tables if they don't exist)
  3. Pre-load the ViT ML model into memory
  4. Register all API routers under /api/v1
  5. Start serving requests

Run with:
  uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings
from app.core.database import init_db
from app.ml.model_loader import get_model_loader
from app.routers import auth, analysis, history, reports

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
settings = get_settings()


# ── Lifespan (startup / shutdown) ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Run startup tasks before the server accepts traffic, and cleanup on shutdown.
    Using the modern lifespan context manager instead of deprecated @app.on_event.
    """
    # ── Startup ──────────────────────────────────────────────────────────────
    logger.info("=== DermAI Backend starting up ===")

    logger.info("Initialising database...")
    await init_db()
    logger.info("Database ready.")

    logger.info("Loading ML model in background thread (server accepts requests immediately)...")
    loader = get_model_loader()
    loader.load()
    logger.info("ML model background load initiated.")

    logger.info("=== Server ready to accept requests ===")

    yield  # Server is now running

    # ── Shutdown ─────────────────────────────────────────────────────────────
    logger.info("DermAI Backend shutting down.")


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "AI-powered dermatology analysis API. "
        "Provides skin condition classification from images and "
        "rule-based symptom analysis. All AI runs locally — no external APIs."
    ),
    contact={
        "name": "DermAI Team",
        "url": "https://github.com/your-org/dermai-backend",
    },
    license_info={
        "name": "MIT",
    },
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# ── CORS Middleware ───────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get_allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
API_PREFIX = "/api/v1"

app.include_router(auth.router, prefix=API_PREFIX)
app.include_router(analysis.router, prefix=API_PREFIX)
app.include_router(history.router, prefix=API_PREFIX)
app.include_router(reports.router, prefix=API_PREFIX)

from app.routers.consultation import router as consultation_router
app.include_router(consultation_router, prefix=API_PREFIX)

from app.routers.analyze import router as analyze_router
app.include_router(analyze_router)



# ── Health check ──────────────────────────────────────────────────────────────
@app.get(
    "/health",
    tags=["System"],
    summary="Health check",
    response_model=dict,
)
async def health_check() -> dict:
    """
    Returns the server status and version. Use this to verify the API is running.
    Does not require authentication.
    """
    return {
        "status": "ok",
        "app": settings.app_name,
        "version": settings.app_version,
    }


# ── Root redirect ─────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def root() -> dict:
    return {
        "message": f"Welcome to {settings.app_name}",
        "docs": "/docs",
        "health": "/health",
    }
