"""
models/evaluation.py — SQLAlchemy ORM model for evaluation submissions.

This module defines the `EvaluationSubmission` table that stores every piece
of data the user submits for evaluation. At Milestone 1, the backend only
handles ingestion; the actual AI evaluation pipeline comes later.
"""

from datetime import datetime, timezone
from sqlalchemy import Column, Integer, Text, String, DateTime

from app.core.database import Base


class EvaluationSubmission(Base):
    """
    Represents a single user submission to be evaluated.

    Columns
    -------
    id              : Auto-incrementing primary key.
    question        : The question that was asked to the AI model.
    ai_response     : The AI-generated answer that needs to be evaluated.
    reference_answer: Optional ground-truth answer (may be None).
    source_document : Optional source text for context (may be None).
    status          : Lifecycle state of the submission.
                      Allowed values: pending | running | completed | failed | partial
    created_at      : UTC timestamp when the row was inserted.
    """

    __tablename__ = "evaluation_submissions"

    id = Column(Integer, primary_key=True, autoincrement=True, index=True)

    question = Column(Text, nullable=False)
    ai_response = Column(Text, nullable=False)
    reference_answer = Column(Text, nullable=True)
    source_document = Column(Text, nullable=True)

    # Status tracks the evaluation pipeline state.
    # Default is "pending" because newly submitted records haven't been
    # picked up by any evaluation agent yet.
    status = Column(String(20), nullable=False, default="pending")

    # Store timestamps in UTC so they are timezone-independent.
    created_at = Column(
        DateTime,
        nullable=False,
        default=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
    )

    def __repr__(self) -> str:
        return (
            f"<EvaluationSubmission id={self.id} status={self.status!r} "
            f"created_at={self.created_at}>"
        )
