"""
app/routers/analysis.py

Skin analysis endpoints:
  POST /analysis/image    — Upload a skin image for AI analysis
  POST /analysis/symptoms — Submit symptoms for rule-based analysis
  GET  /analysis/symptoms/known — List all recognised symptom keywords
"""

import logging

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.database import get_db
from app.models.user import User
from app.schemas.analysis import AnalysisResponse, SymptomAnalysisRequest
from app.services.analysis_service import run_image_analysis, run_symptom_analysis
from app.services.auth_service import get_current_user
from app.services.symptom_service import get_all_known_symptoms

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(prefix="/analysis", tags=["Analysis"])

# Allowed MIME types for image uploads
_ALLOWED_IMAGE_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
    "image/bmp",
    "image/tiff",
}


@router.post(
    "/image",
    response_model=AnalysisResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Analyse a skin image with the AI model",
)
async def analyse_image(
    file: UploadFile = File(..., description="Skin image (JPEG, PNG, WebP, BMP, TIFF)"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> AnalysisResponse:
    """
    Upload a skin photo for AI-powered analysis.

    The image is processed by a Vision Transformer (ViT) model to identify
    potential skin conditions and their confidence scores.

    - Accepted formats: JPEG, PNG, WebP, BMP, TIFF
    - Maximum file size: controlled by `MAX_UPLOAD_SIZE_MB` in `.env`
    - Requires: `Authorization: Bearer <token>` header
    """
    # Validate content type
    if file.content_type not in _ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type '{file.content_type}'. "
                   f"Accepted: {', '.join(sorted(_ALLOWED_IMAGE_TYPES))}",
        )

    # Read and size-check
    image_bytes = await file.read()
    if len(image_bytes) > settings.max_upload_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum upload size of {settings.max_upload_size_mb} MB.",
        )
    if len(image_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded file is empty.",
        )

    logger.info(
        "Image upload: user_id=%d filename=%s size=%d bytes",
        current_user.id,
        file.filename,
        len(image_bytes),
    )

    try:
        result = await run_image_analysis(
            image_bytes=image_bytes,
            original_filename=file.filename or "upload.jpg",
            user=current_user,
            db=db,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )

    return result


@router.post(
    "/symptoms",
    response_model=AnalysisResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Analyse symptoms with the rule-based checker",
)
async def analyse_symptoms(
    payload: SymptomAnalysisRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> AnalysisResponse:
    """
    Submit a list of skin symptoms for rule-based analysis.

    The engine matches symptoms against a curated dermatology knowledge base
    and returns ranked condition matches with clinical recommendations.

    - Requires: `Authorization: Bearer <token>` header
    - `symptoms`: 1–20 symptom strings (e.g. `["itching", "scaling", "redness"]`)
    - `body_area`: Optional — e.g. `"forearm"`, `"face"`
    - `duration_days`: Optional — how many days the patient has had symptoms
    """
    result = await run_symptom_analysis(
        request=payload,
        user=current_user,
        db=db,
    )
    return result


@router.get(
    "/symptoms/known",
    response_model=list[str],
    summary="List all recognised symptom keywords",
)
async def list_known_symptoms() -> list[str]:
    """
    Return a sorted list of all symptom keywords recognised by the
    rule-based engine. Useful for building autocomplete UIs.
    This endpoint does NOT require authentication.
    """
    return get_all_known_symptoms()
