"""
schemas/retrieval.py — Pydantic schemas for semantic retrieval API requests and responses.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator


class RetrievalQueryRequest(BaseModel):
    """Payload sent by client when requesting semantic reference evidence."""
    query: str = Field(
        ...,
        description="The question, statement, or claim to find evidence for.",
        examples=["What is the capital of France?"],
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of relevant reference passages to return (1–20).",
    )
    dataset_filter: Optional[str] = Field(
        default=None,
        description="Optional filter by dataset: 'truthfulqa' or 'squad'.",
    )

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        cleaned = v.strip() if v else ""
        if not cleaned:
            raise ValueError("Query string must not be empty or whitespace-only.")
        if len(cleaned) < 3:
            raise ValueError("Query must be at least 3 characters long.")
        if len(cleaned) > 2000:
            raise ValueError("Query must not exceed 2000 characters.")
        return cleaned


class EvidencePassageItem(BaseModel):
    """Structured representation of a single retrieved reference passage."""
    chunk_id: str
    text: str
    similarity_score: float
    dataset_name: str
    dataset_split: str
    source_record_id: str
    title: Optional[str] = None
    category: Optional[str] = None
    chunk_index: int = 0
    content_type: str = "context_chunk"
    extra_metadata: Dict[str, Any] = Field(default_factory=dict)


class RetrievalQueryResponse(BaseModel):
    """Response returned by the semantic retrieval endpoint."""
    query: str
    total_retrieved: int
    status: str  # "sufficient" | "insufficient_evidence"
    passages: List[EvidencePassageItem]
