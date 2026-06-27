"""
app/routers/reports.py

PDF report generation endpoint:
  GET /reports/{analysis_id}/pdf — Stream a PDF report for a specific analysis
"""

import logging

from fastapi import APIRouter, Depends
from fastapi.responses import Response
# pyrefly: ignore [missing-import]
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.user import User
from app.services.analysis_service import get_analysis_by_id
from app.services.auth_service import get_current_user
from app.services.report_service import generate_pdf_report

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/reports", tags=["Reports"])


@router.get(
    "/{analysis_id}/pdf",
    summary="Download a PDF report for an analysis",
    response_class=Response,
    responses={
        200: {
            "content": {"application/pdf": {}},
            "description": "PDF report file",
        }
    },
)
async def download_pdf_report(
    analysis_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Response:
    """
    Generate and stream a branded PDF report for the specified analysis.

    The report includes:
    - Patient and report metadata
    - Detected conditions with confidence scores and ICD-10 codes
    - Clinical recommendations
    - AI disclaimer footer

    Returns 404 if the analysis does not exist or does not belong to the current user.
    """
    record = await get_analysis_by_id(analysis_id, current_user, db)

    logger.info(
        "Generating PDF report for analysis_id=%d user_id=%d",
        analysis_id,
        current_user.id,
    )

    pdf_bytes = generate_pdf_report(analysis=record, user=current_user)

    filename = f"dermai_report_{analysis_id}.pdf"

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Content-Length": str(len(pdf_bytes)),
        },
    )
