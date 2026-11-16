"""
schemas/evaluation.py — Pydantic request and response schemas for evaluations.

Pydantic schemas serve two purposes:
  1. Validate and coerce incoming JSON request bodies.
  2. Serialize database model instances into JSON responses.

We keep schemas separate from SQLAlchemy models so the two layers can evolve
independently.
"""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Request schema
# ---------------------------------------------------------------------------

class EvaluationSubmitRequest(BaseModel):
    """
    Payload a client sends when submitting an AI response for evaluation.

    Validation rules
    ----------------
    - question      : Required. 10–5000 chars after stripping whitespace.
    - ai_response   : Required. 10–20 000 chars after stripping whitespace.
    - reference_answer : Optional. ≤ 10 000 chars; empty string → None.
    - source_document  : Optional. ≤ 50 000 chars; empty string → None.
    """

    question: str = Field(
        ...,
        description="The question that was posed to the AI model.",
    )
    ai_response: str = Field(
        ...,
        description="The AI-generated answer to evaluate.",
    )
    reference_answer: Optional[str] = Field(
        default=None,
        description="Optional ground-truth answer for comparison.",
    )
    source_document: Optional[str] = Field(
        default=None,
        description="Optional source text that provides context.",
    )

    # ------------------------------------------------------------------
    # Field-level validators
    # ------------------------------------------------------------------

    @field_validator("question", mode="before")
    @classmethod
    def validate_question(cls, v: str) -> str:
        """Strip whitespace and enforce length bounds."""
        if not isinstance(v, str):
            raise ValueError("question must be a string")
        v = v.strip()
        if not v:
            raise ValueError("question must not be empty or whitespace-only")
        if len(v) < 10:
            raise ValueError("question must be at least 10 characters long")
        if len(v) > 5000:
            raise ValueError("question must not exceed 5000 characters")
        return v

    @field_validator("ai_response", mode="before")
    @classmethod
    def validate_ai_response(cls, v: str) -> str:
        """Strip whitespace and enforce length bounds."""
        if not isinstance(v, str):
            raise ValueError("ai_response must be a string")
        v = v.strip()
        if not v:
            raise ValueError("ai_response must not be empty or whitespace-only")
        if len(v) < 10:
            raise ValueError("ai_response must be at least 10 characters long")
        if len(v) > 20000:
            raise ValueError("ai_response must not exceed 20 000 characters")
        return v

    @field_validator("reference_answer", mode="before")
    @classmethod
    def validate_reference_answer(cls, v: Optional[str]) -> Optional[str]:
        """Strip whitespace; treat blank string as None; enforce max length."""
        if v is None:
            return None
        if not isinstance(v, str):
            raise ValueError("reference_answer must be a string or null")
        v = v.strip()
        if not v:
            return None  # Empty after stripping → treat as not provided
        if len(v) > 10000:
            raise ValueError("reference_answer must not exceed 10 000 characters")
        return v

    @field_validator("source_document", mode="before")
    @classmethod
    def validate_source_document(cls, v: Optional[str]) -> Optional[str]:
        """Strip whitespace; treat blank string as None; enforce max length."""
        if v is None:
            return None
        if not isinstance(v, str):
            raise ValueError("source_document must be a string or null")
        v = v.strip()
        if not v:
            return None  # Empty after stripping → treat as not provided
        if len(v) > 50000:
            raise ValueError("source_document must not exceed 50 000 characters")
        return v


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

class EvaluationResponse(BaseModel):
    """
    Full representation of an evaluation submission, returned after creation
    or when fetching a single submission by ID.
    """

    id: int
    question: str
    ai_response: str
    reference_answer: Optional[str]
    source_document: Optional[str]
    status: str
    created_at: datetime

    model_config = {"from_attributes": True}


class EvaluationListItem(BaseModel):
    """
    Compact representation used in list responses.
    The question is truncated to 100 characters in the service layer to keep
    list payloads small.
    """

    id: int
    question: str   # Truncated to 100 chars by the service layer
    status: str
    created_at: datetime

    model_config = {"from_attributes": True}


class PaginatedEvaluationsResponse(BaseModel):
    """
    Wraps a page of EvaluationListItem records together with pagination info.
    """

    items: List[EvaluationListItem]
    total: int      # Total number of records in the database (all pages)
    page: int       # Current page number (1-indexed)
    page_size: int  # Number of items per page
