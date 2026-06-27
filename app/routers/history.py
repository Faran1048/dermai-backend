"""
app/routers/history.py

Patient analysis history endpoints:
  GET /history        — Paginated list of the user's past analyses
  GET /history/{id}   — Single analysis detail
  DELETE /history/{id} — Delete a specific analysis record
"""

# pyrefly: ignore [missing-import]
from fastapi import APIRouter, Depends, HTTPException, Query, status
# pyrefly: ignore [missing-import]
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.user import User
from app.schemas.analysis import AnalysisListResponse, AnalysisResponse
from app.services.analysis_service import get_analysis_by_id, get_user_analyses
from app.services.auth_service import get_current_user

router = APIRouter(prefix="/history", tags=["History"])


@router.get(
    "",
    response_model=AnalysisListResponse,
    summary="List all past analyses for the current user",
)
async def list_history(
    page: int = Query(default=1, ge=1, description="Page number"),
    page_size: int = Query(default=20, ge=1, le=100, description="Items per page"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> AnalysisListResponse:
    """
    Return a paginated list of the authenticated user's analysis history,
    ordered by most recent first.
    """
    total, records = await get_user_analyses(
        user=current_user,
        db=db,
        page=page,
        page_size=page_size,
    )
    return AnalysisListResponse(
        total=total,
        page=page,
        page_size=page_size,
        items=[AnalysisResponse.model_validate(r) for r in records],
    )


@router.get(
    "/{analysis_id}",
    response_model=AnalysisResponse,
    summary="Get a single analysis by ID",
)
async def get_analysis(
    analysis_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> AnalysisResponse:
    """
    Retrieve the full details of a specific analysis record.
    Returns 404 if the record does not exist or does not belong to the current user.
    """
    record = await get_analysis_by_id(analysis_id, current_user, db)
    return AnalysisResponse.model_validate(record)


@router.delete(
    "/{analysis_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete an analysis record",
)
async def delete_analysis(
    analysis_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> None:
    """
    Permanently delete a specific analysis record belonging to the current user.
    Returns 204 No Content on success.
    """
    record = await get_analysis_by_id(analysis_id, current_user, db)
    await db.delete(record)
