"""
api/routes/evaluation.py — HTTP route handlers for evaluation submissions.

Each function in this module maps one API endpoint to service-layer logic.
Routes are grouped in an APIRouter that is registered in main.py with the
prefix /api/v1, so the final URLs are:

  POST   /api/v1/evaluations/
  GET    /api/v1/evaluations/
  GET    /api/v1/evaluations/{submission_id}
  GET    /api/v1/health
"""

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.schemas.evaluation import (
    EvaluationSubmitRequest,
    EvaluationResponse,
    PaginatedEvaluationsResponse,
)
from app.services import evaluation_service

# Create a router; the /api/v1 prefix is applied in main.py
router = APIRouter()


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@router.get(
    "/health",
    summary="Health check",
    tags=["Health"],
)
def health_check():
    """
    Simple liveness probe.

    Returns 200 OK with a JSON body so load balancers and monitoring tools
    can confirm the service is running.
    """
    return {"status": "healthy", "service": "Verity API"}


# ---------------------------------------------------------------------------
# Submit a new evaluation
# ---------------------------------------------------------------------------

@router.post(
    "/evaluations/",
    response_model=EvaluationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Submit an AI response for evaluation",
    tags=["Evaluations"],
)
def create_evaluation(
    request: EvaluationSubmitRequest,
    db: Session = Depends(get_db),
):
    """
    Accept a user's question and an AI-generated response, persist them to
    the database, and return the created record.

    The submission starts with status **pending**. Future milestones will
    update the status as the evaluation pipeline processes it.
    """
    submission = evaluation_service.create_evaluation(db, request)
    return submission


# ---------------------------------------------------------------------------
# Retrieve a single submission
# ---------------------------------------------------------------------------

@router.get(
    "/evaluations/{submission_id}",
    response_model=EvaluationResponse,
    summary="Get a single evaluation submission",
    tags=["Evaluations"],
)
def get_evaluation(
    submission_id: int,
    db: Session = Depends(get_db),
):
    """
    Fetch the full details of a specific evaluation submission by its ID.

    Returns 404 if no submission with the given ID exists.
    """
    submission = evaluation_service.get_evaluation(db, submission_id)
    if submission is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Evaluation submission with id={submission_id} not found.",
        )
    return submission


# ---------------------------------------------------------------------------
# List submissions (paginated)
# ---------------------------------------------------------------------------

@router.get(
    "/evaluations/",
    response_model=PaginatedEvaluationsResponse,
    summary="List evaluation submissions (paginated)",
    tags=["Evaluations"],
)
def list_evaluations(
    page: int = Query(default=1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(
        default=10, ge=1, le=100, description="Items per page (1–100)"
    ),
    db: Session = Depends(get_db),
):
    """
    Return a paginated list of evaluation submissions, newest first.

    The **question** field in each list item is truncated to 100 characters
    for readability. Use the detail endpoint to fetch the full content.
    """
    items, total = evaluation_service.list_evaluations(db, page, page_size)
    return PaginatedEvaluationsResponse(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
    )
