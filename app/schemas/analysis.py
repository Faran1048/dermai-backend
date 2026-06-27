"""
app/schemas/analysis.py

Pydantic v2 schemas for skin analysis requests and responses.
"""

from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, Field

from app.models.analysis import AnalysisType, SeverityLevel


# ── Shared sub-schemas ────────────────────────────────────────────────────────

class ConditionResult(BaseModel):
    """A single predicted skin condition with its confidence score."""
    condition: str = Field(..., examples=["Eczema"])
    confidence: float = Field(..., ge=0.0, le=1.0, examples=[0.87])
    description: str = Field(..., examples=["Chronic inflammatory skin condition."])
    icd10_code: str | None = Field(None, examples=["L20.9"])


class Recommendation(BaseModel):
    """A clinical recommendation linked to the analysis."""
    recommendation: str = Field(
        ..., examples=["Consult a dermatologist within 2 weeks."]
    )
    urgency: str = Field(..., examples=["routine"])  # "routine" | "soon" | "urgent"


# ── Request schemas ───────────────────────────────────────────────────────────

class SymptomAnalysisRequest(BaseModel):
    symptoms: list[str] = Field(
        ...,
        min_length=1,
        max_length=20,
        examples=[["itching", "redness", "scaling"]],
    )
    body_area: str | None = Field(
        None,
        max_length=100,
        examples=["forearm"],
    )
    duration_days: int | None = Field(
        None,
        ge=0,
        description="How long the patient has had the symptoms (in days).",
    )
    notes: str | None = Field(None, max_length=1000)


# ── Response schemas ──────────────────────────────────────────────────────────

class AnalysisResponse(BaseModel):
    model_config = {"from_attributes": True}

    id: int
    analysis_type: AnalysisType
    severity: SeverityLevel | None
    conditions: list[ConditionResult] | None
    symptoms: list[str] | None
    recommendations: list[Recommendation] | None
    summary: str | None
    image_filename: str | None
    created_at: datetime


class AnalysisListResponse(BaseModel):
    total: int
    page: int
    page_size: int
    items: list[AnalysisResponse]


class PaginationParams(BaseModel):
    page: Annotated[int, Field(ge=1, default=1)]
    page_size: Annotated[int, Field(ge=1, le=100, default=20)]
