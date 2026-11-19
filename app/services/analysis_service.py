
"""
app/services/analysis_service.py

Orchestrates image and symptom analyses:
  1. Runs ML inference or symptom matching
  2. Determines overall severity
  3. Stores original images in private Supabase Storage
  4. Persists results to the database
  5. Returns structured AnalysisResponse objects
"""

from __future__ import annotations

import logging
import uuid

# pyrefly: ignore [missing-import]
from sqlalchemy.ext.asyncio import AsyncSession
# pyrefly: ignore [missing-import]
from sqlalchemy import select, func

from app.core.config import get_settings
from app.models.analysis import Analysis, AnalysisType, SeverityLevel
from app.models.user import User
from app.schemas.analysis import (
    AnalysisResponse,
    ConditionResult,
    Recommendation,
    SymptomAnalysisRequest,
)
from app.services import symptom_service
from supabase_client import upload_image_to_storage, delete_from_storage

logger = logging.getLogger(__name__)
settings = get_settings()


# ── Severity helpers ──────────────────────────────────────────────────────────

_SEVERITY_ORDER = {
    SeverityLevel.LOW: 0,
    SeverityLevel.MODERATE: 1,
    SeverityLevel.HIGH: 2,
    SeverityLevel.URGENT: 3,
}


def _resolve_severity(hints: list[str]) -> SeverityLevel:
    """Return the highest severity found in a list of hint strings."""
    mapping = {
        "low": SeverityLevel.LOW,
        "moderate": SeverityLevel.MODERATE,
        "high": SeverityLevel.HIGH,
        "urgent": SeverityLevel.URGENT,
    }

    resolved = [mapping.get(h.lower(), SeverityLevel.LOW) for h in hints]

    return max(
        resolved,
        key=lambda s: _SEVERITY_ORDER[s],
        default=SeverityLevel.LOW,
    )


def _build_summary(
    conditions: list[ConditionResult],
    severity: SeverityLevel,
) -> str:
    top = conditions[0] if conditions else None

    if not top:
        return "No significant skin conditions detected."

    return (
        f"The analysis detected '{top.condition}' as the most likely finding "
        f"(confidence: {top.confidence * 100:.1f}%). "
        f"Overall severity is assessed as '{severity.value}'."
    )


# ── Image analysis ────────────────────────────────────────────────────────────

async def run_image_analysis(
    image_bytes: bytes,
    original_filename: str,
    content_type: str,
    user: User,
    db: AsyncSession,
) -> AnalysisResponse:
    """
    Perform image-based skin analysis.

    Workflow:
      1. Validate image format.
      2. Save a temporary local copy.
      3. Run EfficientNet-B4 inference.
      4. Remove the temporary local copy.
      5. Upload the original image to private Supabase Storage.
      6. Save the permanent storage path in the database.
      7. Return AnalysisResponse.
    """

    # 1. Validate image format.
    extension_by_type = {
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
        "image/bmp": ".bmp",
        "image/tiff": ".tiff",
    }

    ext = extension_by_type.get(content_type)

    if ext is None:
        raise ValueError("Unsupported image format.")

    if not image_bytes:
        raise ValueError("Uploaded image is empty.")

    # 2. Save a temporary image for model inference.
    saved_filename = f"{uuid.uuid4().hex}{ext}"
    save_path = settings.upload_path / saved_filename

    try:
        save_path.write_bytes(image_bytes)

        # 3. Run EfficientNet-B4 inference.
        from algorithms.cnn_classifier import classify_image

        cnn_res = classify_image(str(save_path))
        all_scores = cnn_res.get("all_scores", {})

        if not all_scores:
            raise RuntimeError(
                "The image classifier did not return valid prediction scores."
            )

    finally:
        # 4. Always remove the temporary local image.
        save_path.unlink(missing_ok=True)

    # 5. Map model classes to display information.
    META_MAPPING = {
        "melanoma": {
            "condition": "Melanoma",
            "description": "Potential high-risk skin finding. Requires professional assessment.",
            "icd10": "C43.9",
            "severity": "urgent",
        },
        "basal_cell_carcinoma": {
            "condition": "Basal Cell Carcinoma",
            "description": "Potential basal cell carcinoma finding. Requires dermatologist assessment.",
            "icd10": "C44.91",
            "severity": "high",
        },
        "actinic_keratosis": {
            "condition": "Actinic Keratosis",
            "description": "Potential actinic keratosis finding. Professional assessment is recommended.",
            "icd10": "L57.0",
            "severity": "high",
        },
        "melanocytic_nevi": {
            "condition": "Melanocytic Nevus (Mole)",
            "description": "Potential melanocytic nevus finding.",
            "icd10": "D22.9",
            "severity": "low",
        },
        "benign_keratosis": {
            "condition": "Benign Keratosis-like Lesion",
            "description": "Potential benign keratosis-like finding.",
            "icd10": "L82.0",
            "severity": "low",
        },
        "vascular_lesion": {
            "condition": "Vascular Lesion",
            "description": "Potential vascular lesion finding.",
            "icd10": "D18.01",
            "severity": "low",
        },
        "petechiae_purpura": {
            "condition": "Petechiae/Purpura",
            "description": "Potential petechiae or purpura finding.",
            "icd10": "L81.8",
            "severity": "moderate",
        },
        "leukemia_cutis": {
            "condition": "Leukemia Cutis",
            "description": "Potential high-risk skin finding. Requires urgent professional assessment.",
            "icd10": "C95.9",
            "severity": "urgent",
        },
        "ecchymosis": {
            "condition": "Ecchymosis",
            "description": "Potential bruising or bleeding under the skin.",
            "icd10": "R58",
            "severity": "moderate",
        },
        "normal_skin": {
            "condition": "Normal Skin",
            "description": "The model classified this image as normal skin. This is not a medical diagnosis.",
            "icd10": "L98.9",
            "severity": "low",
        },
    }

    # 6. Sort scores and select the top five.
    sorted_scores = sorted(
        all_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )[:5]

    # 7. Build structured conditions.
    conditions = []

    for cls_name, score in sorted_scores:
        meta = META_MAPPING.get(
            cls_name,
            {
                "condition": cls_name,
                "description": "Unverified model classification.",
                "icd10": None,
                "severity": "low",
            },
        )

        conditions.append(
            ConditionResult(
                condition=meta["condition"],
                confidence=round(float(score), 4),
                description=meta["description"],
                icd10_code=meta["icd10"],
            )
        )

    if not conditions:
        raise RuntimeError("No usable image classification results were produced.")

    # 8. Determine severity.
    severity_hints = [
        META_MAPPING.get(cls_name, {}).get("severity", "low")
        for cls_name, _ in sorted_scores[:2]
    ]

    severity = _resolve_severity(severity_hints)

    # 9. Build recommendations.
    recommendations = [
        Recommendation(
            recommendation=(
                "Consult a licensed dermatologist for an accurate diagnosis."
            ),
            urgency=(
                "routine"
                if severity in (SeverityLevel.LOW, SeverityLevel.MODERATE)
                else "urgent"
            ),
        ),
        Recommendation(
            recommendation=(
                "Photograph the affected area regularly to track any changes."
            ),
            urgency="routine",
        ),
    ]

    if severity == SeverityLevel.URGENT:
        recommendations.insert(
            0,
            Recommendation(
                recommendation=(
                    "Seek medical attention as soon as possible."
                ),
                urgency="urgent",
            ),
        )

    summary = _build_summary(conditions, severity)

    # 10. Upload the original image to private Supabase Storage.
    object_path = (
        f"originals/{user.id}/{uuid.uuid4().hex}{ext}"
    )

    stored_image_path = upload_image_to_storage(
        object_path=object_path,
        file_bytes=image_bytes,
        content_type=content_type,
    )

    # 11. Create the database record using the permanent object path.
    record = Analysis(
        user_id=user.id,
        analysis_type=AnalysisType.IMAGE,
        severity=severity.value,
        image_filename=original_filename,
        image_path=stored_image_path,
        conditions=[condition.model_dump() for condition in conditions],
        recommendations=[
            recommendation.model_dump()
            for recommendation in recommendations
        ],
        summary=summary,
    )

    db.add(record)

    try:
        await db.flush()
        await db.refresh(record)

    except Exception:
        # Clean up the uploaded image if database persistence fails.
        try:
            delete_from_storage(stored_image_path)
        except Exception:
            logger.exception(
                "Could not clean up Supabase image after database failure: %s",
                stored_image_path,
            )

        raise

    logger.info(
        "Image analysis created: user_id=%s analysis_id=%s storage_path=%s",
        user.id,
        record.id,
        stored_image_path,
    )

    return AnalysisResponse.model_validate(record)


# ── Symptom analysis ──────────────────────────────────────────────────────────

async def run_symptom_analysis(
    request: SymptomAnalysisRequest,
    user: User,
    db: AsyncSession,
) -> AnalysisResponse:
    """
    Perform rule-based symptom analysis and persist the result.
    """

    results = symptom_service.analyze_symptoms(
        symptoms=request.symptoms,
        body_area=request.body_area,
        duration_days=request.duration_days,
    )

    conditions = [
        ConditionResult(
            condition=result.condition,
            confidence=min(round(result.score / 20.0, 4), 1.0),
            description=result.description,
            icd10_code=result.icd10_code,
        )
        for result in results
    ]

    severity = _resolve_severity(
        [result.severity for result in results[:2]]
    )

    recommendations: list[Recommendation] = []

    if results:
        top_urgency = (
            "urgent" if results[0].urgency_flag else "routine"
        )

        for rec_text in results[0].recommendations:
            recommendations.append(
                Recommendation(
                    recommendation=rec_text,
                    urgency=top_urgency,
                )
            )

    summary = _build_summary(conditions, severity)

    record = Analysis(
        user_id=user.id,
        analysis_type=AnalysisType.SYMPTOM,
        severity=severity.value,
        symptoms=request.symptoms,
        conditions=[
            condition.model_dump()
            for condition in conditions
        ],
        recommendations=[
            recommendation.model_dump()
            for recommendation in recommendations
        ],
        summary=summary,
        notes=request.notes,
    )

    db.add(record)
    await db.flush()
    await db.refresh(record)

    return AnalysisResponse.model_validate(record)


# ── History queries ───────────────────────────────────────────────────────────

async def get_user_analyses(
    user: User,
    db: AsyncSession,
    page: int = 1,
    page_size: int = 20,
) -> tuple[int, list[Analysis]]:
    """Return paginated analysis history for the current user."""

    total: int = await db.scalar(
        select(func.count(Analysis.id)).where(
            Analysis.user_id == user.id
        )
    )

    offset = (page - 1) * page_size

    rows = (
        await db.scalars(
            select(Analysis)
            .where(Analysis.user_id == user.id)
            .order_by(Analysis.created_at.desc())
            .offset(offset)
            .limit(page_size)
        )
    ).all()

    return total, list(rows)


async def get_analysis_by_id(
    analysis_id: int,
    user: User,
    db: AsyncSession,
) -> Analysis:
    """
    Retrieve a single analysis record belonging to the current user.
    """

    from fastapi import HTTPException, status

    record = await db.get(Analysis, analysis_id)

    if record is None or record.user_id != user.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Analysis #{analysis_id} not found.",
        )

    return record
