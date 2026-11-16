"""
services/evaluation_service.py — Business logic for evaluation submissions.

This service layer sits between the route handlers and the database. Keeping
database access here (rather than directly in routes) makes the code easier
to test and to extend when new features are added.
"""

from typing import Optional, Tuple, List

from sqlalchemy.orm import Session

from app.models.evaluation import EvaluationSubmission
from app.schemas.evaluation import EvaluationSubmitRequest, EvaluationListItem


def create_evaluation(
    db: Session, request: EvaluationSubmitRequest
) -> EvaluationSubmission:
    """
    Persist a new evaluation submission to the database.

    Parameters
    ----------
    db      : Active SQLAlchemy session (provided by FastAPI dependency).
    request : Validated request data from the client.

    Returns
    -------
    The newly created EvaluationSubmission ORM instance with its assigned ID.
    """
    submission = EvaluationSubmission(
        question=request.question,
        ai_response=request.ai_response,
        reference_answer=request.reference_answer,
        source_document=request.source_document,
        # status defaults to "pending" at the ORM level, but we set it
        # explicitly here to make the intent clear when reading this code.
        status="pending",
    )
    db.add(submission)
    db.commit()
    db.refresh(submission)  # Reload from DB so `id` and `created_at` are populated
    return submission


def get_evaluation(
    db: Session, submission_id: int
) -> Optional[EvaluationSubmission]:
    """
    Retrieve a single evaluation submission by its primary key.

    Parameters
    ----------
    db            : Active SQLAlchemy session.
    submission_id : The integer ID of the submission to look up.

    Returns
    -------
    The EvaluationSubmission if found, or None if no record matches.
    """
    return (
        db.query(EvaluationSubmission)
        .filter(EvaluationSubmission.id == submission_id)
        .first()
    )


def list_evaluations(
    db: Session, page: int, page_size: int
) -> Tuple[List[EvaluationListItem], int]:
    """
    Return a paginated list of evaluation submissions, newest first.

    The question is truncated to 100 characters so list responses stay compact.
    Pagination is 1-indexed (page=1 is the first page).

    Parameters
    ----------
    db        : Active SQLAlchemy session.
    page      : Page number to retrieve (must be ≥ 1).
    page_size : Number of records per page (1–100).

    Returns
    -------
    A tuple of:
      - List of EvaluationListItem (may be empty if the page is beyond the data)
      - Total count of all submissions in the database
    """
    # Count ALL records so the client knows how many pages exist
    total = db.query(EvaluationSubmission).count()

    # Calculate how many rows to skip for the requested page
    offset = (page - 1) * page_size

    rows = (
        db.query(EvaluationSubmission)
        .order_by(EvaluationSubmission.id.desc())  # Newest first
        .offset(offset)
        .limit(page_size)
        .all()
    )

    # Build lightweight list items, truncating the question preview
    items = [
        EvaluationListItem(
            id=row.id,
            question=row.question[:100],  # Truncate to 100 chars
            status=row.status,
            created_at=row.created_at,
        )
        for row in rows
    ]

    return items, total
