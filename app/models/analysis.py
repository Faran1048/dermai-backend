"""
app/models/analysis.py

SQLAlchemy ORM model for skin analysis records.
Stores both image-based and symptom-based analyses in a unified table.
JSON columns hold lists/dicts and are stored as TEXT in SQLite.
"""

from datetime import datetime
from enum import Enum as PyEnum

# pyrefly: ignore [missing-import]
from sqlalchemy import DateTime, ForeignKey, Integer, JSON, String, Text, func
# pyrefly: ignore [missing-import]
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class AnalysisType(str, PyEnum):
    IMAGE = "image"
    SYMPTOM = "symptom"
    COMBINED = "combined"


class SeverityLevel(str, PyEnum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    URGENT = "urgent"


class Analysis(Base):
    __tablename__ = "analyses"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False
    )

    # ── Analysis metadata ─────────────────────────────────────────────────────
    analysis_type: Mapped[str] = mapped_column(
        String(20), nullable=False, default=AnalysisType.IMAGE
    )
    severity: Mapped[str] = mapped_column(
        String(20), nullable=True, default=SeverityLevel.LOW
    )

    # ── Image analysis fields ─────────────────────────────────────────────────
    image_filename: Mapped[str | None] = mapped_column(String(512), nullable=True)
    image_path: Mapped[str | None] = mapped_column(String(512), nullable=True)

    # ── Structured results (stored as JSON) ───────────────────────────────────
    # List of {"condition": str, "confidence": float, "description": str}
    conditions: Mapped[list | None] = mapped_column(JSON, nullable=True)

    # List of symptom strings submitted by the user
    symptoms: Mapped[list | None] = mapped_column(JSON, nullable=True)

    # List of {"recommendation": str, "urgency": str}
    recommendations: Mapped[list | None] = mapped_column(JSON, nullable=True)

    # Free-text summary of the analysis
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Any notes added by a clinician or the user
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    # ── Timestamps ────────────────────────────────────────────────────────────
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # ── Relationships ─────────────────────────────────────────────────────────
    user: Mapped["User"] = relationship("User", back_populates="analyses")  # noqa: F821

    def __repr__(self) -> str:
        return f"<Analysis id={self.id} type={self.analysis_type} user_id={self.user_id}>"
