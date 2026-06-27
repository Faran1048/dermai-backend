"""
app/services/analysis_service.py

Orchestrates image and symptom analyses:
  1. Runs ML inference or symptom matching
  2. Determines overall severity
  3. Persists results to the database
  4. Returns structured AnalysisResponse objects
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path

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
from app.services import ml_service, symptom_service

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
    return max(resolved, key=lambda s: _SEVERITY_ORDER[s], default=SeverityLevel.LOW)


def _build_summary(conditions: list[ConditionResult], severity: SeverityLevel) -> str:
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
    user: User,
    db: AsyncSession,
) -> AnalysisResponse:
    """
    Perform image-based skin analysis.

    Steps:
      1. Save the uploaded image to disk.
      2. Run ML inference.
      3. Build recommendations based on top condition.
      4. Persist to DB.
      5. Return AnalysisResponse.
    """
    # 1. Save image
    ext = Path(original_filename).suffix or ".jpg"
    saved_filename = f"{uuid.uuid4().hex}{ext}"
    save_path = settings.upload_path / saved_filename
    save_path.write_bytes(image_bytes)
    logger.info("Image saved: %s", save_path)

    # 2. ML inference using local EfficientNet-B4 model
    from algorithms.cnn_classifier import classify_image
    
    # Run prediction using single forward pass (cpu optimization default)
    cnn_res = classify_image(str(save_path))
    all_scores = cnn_res.get("all_scores", {})
    
    _META_MAPPING = {
        "melanoma": {"condition": "Melanoma", "description": "Malignant melanoma — high-risk skin cancer. Urgent referral required.", "icd10": "C43.9", "severity": "urgent"},
        "basal_cell_carcinoma": {"condition": "Basal Cell Carcinoma", "description": "Basal cell carcinoma — most common skin cancer. Dermatologist visit within 2 weeks.", "icd10": "C44.91", "severity": "high"},
        "actinic_keratosis": {"condition": "Actinic Keratosis", "description": "Actinic keratosis — pre-cancerous lesion from UV damage.", "icd10": "L57.0", "severity": "high"},
        "melanocytic_nevi": {"condition": "Melanocytic Nevus (Mole)", "description": "Melanocytic nevus (common mole) — benign.", "icd10": "D22.9", "severity": "low"},
        "benign_keratosis": {"condition": "Benign Keratosis-like Lesion", "description": "Benign keratosis-like lesion — benign.", "icd10": "L82.0", "severity": "low"},
        "vascular_lesion": {"condition": "Vascular Lesion", "description": "Vascular lesion — usually benign.", "icd10": "D18.01", "severity": "low"},
        "petechiae_purpura": {"condition": "Petechiae/Purpura", "description": "Petechiae/Purpura — minor hemorrhage. Monitor.", "icd10": "L81.8", "severity": "moderate"},
        "leukemia_cutis": {"condition": "Leukemia Cutis", "description": "Leukemia cutis — leukemia infiltration. Urgent referral.", "icd10": "C95.9", "severity": "urgent"},
        "ecchymosis": {"condition": "Ecchymosis", "description": "Ecchymosis — bruising or bleeding under the skin.", "icd10": "R58", "severity": "moderate"},
        "normal_skin": {"condition": "Normal Skin", "description": "Normal skin — no active lesions detected.", "icd10": "L98.9", "severity": "low"},
    }
    
    # Sort all scores descending and limit to top-5
    sorted_scores = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)[:5]
    
    # 3. Build structured conditions
    conditions = []
    for cls_name, score in sorted_scores:
        meta = _META_MAPPING.get(cls_name, {"condition": cls_name, "description": "Skin finding", "icd10": None, "severity": "low"})
        conditions.append(
            ConditionResult(
                condition=meta["condition"],
                confidence=round(score, 4),
                description=meta["description"],
                icd10_code=meta["icd10"],
            )
        )
        
    if not conditions:
        conditions.append(
            ConditionResult(
                condition="Unclassified skin finding",
                confidence=1.0,
                description="Skin finding",
                icd10_code=None,
            )
        )

    # 4. Determine severity
    severity_hints = [
        _META_MAPPING.get(cls_name, {}).get("severity", "low")
        for cls_name, _ in sorted_scores[:2]
    ]
    severity = _resolve_severity(severity_hints)


    # 5. Build generic recommendations
    recommendations = [
        Recommendation(
            recommendation="Consult a licensed dermatologist for an accurate diagnosis.",
            urgency="routine" if severity in (SeverityLevel.LOW, SeverityLevel.MODERATE) else "urgent",
        ),
        Recommendation(
            recommendation="Photograph the affected area regularly to track any changes.",
            urgency="routine",
        ),
    ]
    if severity == SeverityLevel.URGENT:
        recommendations.insert(
            0,
            Recommendation(
                recommendation="URGENT: Seek medical attention as soon as possible.",
                urgency="urgent",
            ),
        )

    summary = _build_summary(conditions, severity)

    # 6. Persist to DB
    record = Analysis(
        user_id=user.id,
        analysis_type=AnalysisType.IMAGE,
        severity=severity.value,
        image_filename=original_filename,
        image_path=str(save_path),
        conditions=[c.model_dump() for c in conditions],
        recommendations=[r.model_dump() for r in recommendations],
        summary=summary,
    )
    db.add(record)
    await db.flush()
    await db.refresh(record)

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
            condition=r.condition,
            # Normalize score to a 0-1 confidence range
            confidence=min(round(r.score / 20.0, 4), 1.0),
            description=r.description,
            icd10_code=r.icd10_code,
        )
        for r in results
    ]

    severity = _resolve_severity([r.severity for r in results[:2]])

    recommendations: list[Recommendation] = []
    if results:
        top_urgency = "urgent" if results[0].urgency_flag else "routine"
        for rec_text in results[0].recommendations:
            recommendations.append(
                Recommendation(recommendation=rec_text, urgency=top_urgency)
            )

    summary = _build_summary(conditions, severity)

    record = Analysis(
        user_id=user.id,
        analysis_type=AnalysisType.SYMPTOM,
        severity=severity.value,
        symptoms=request.symptoms,
        conditions=[c.model_dump() for c in conditions],
        recommendations=[r.model_dump() for r in recommendations],
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
        select(func.count(Analysis.id)).where(Analysis.user_id == user.id)
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
    Retrieve a single analysis record that belongs to the current user.

    Raises:
        ValueError: If the record is not found or does not belong to the user.
    """
    from fastapi import HTTPException, status

    record = await db.get(Analysis, analysis_id)
    if record is None or record.user_id != user.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Analysis #{analysis_id} not found.",
        )
    return record
