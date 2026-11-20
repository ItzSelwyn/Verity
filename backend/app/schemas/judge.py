"""
judge.py — Pydantic schemas for the Judge Agent layer and evaluation outputs.
Defines strict input contracts, scoring bands, and structured output formats.
"""

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class RelevanceClassification(str, Enum):
    """Classification bands for question-answer relevance."""
    FULLY_RELEVANT = "fully_relevant"        # 9.0 - 10.0: Directly addresses core intent and constraints
    MOSTLY_RELEVANT = "mostly_relevant"      # 7.0 - 8.9: Addresses core question with minor tangents
    PARTIALLY_RELEVANT = "partially_relevant" # 4.0 - 6.9: Only answers sub-parts or significantly drifts
    OFF_TOPIC = "off_topic"                  # 1.0 - 3.9: Related concepts mentioned, but question unaddressed
    UNRELATED = "unrelated"                  # 0.0 - 0.9: Completely different topic or non-responsive


class ConfidenceLevel(str, Enum):
    """Confidence level of the evaluation judgment."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class RelevanceJudgeInput(BaseModel):
    """Input contract for the Relevance Judge Agent."""
    question: str = Field(
        ...,
        min_length=5,
        max_length=5000,
        description="The original prompt or question submitted by the user."
    )
    ai_response: str = Field(
        ...,
        min_length=5,
        max_length=20000,
        description="The candidate AI response to evaluate."
    )
    context: Optional[str] = Field(
        None,
        max_length=50000,
        description="Optional reference context or source text."
    )


class RelevanceJudgeOutput(BaseModel):
    """Structured output contract for the Relevance Judge Agent."""
    agent: str = Field(
        default="relevance_judge",
        description="Identifier of the judge agent."
    )
    score: float = Field(
        ...,
        ge=0.0,
        le=10.0,
        description="Relevance score on a 0.0 to 10.0 scale."
    )
    classification: RelevanceClassification = Field(
        ...,
        description="Categorical relevance classification."
    )
    reasoning: str = Field(
        ...,
        min_length=10,
        description="Detailed explanation of the assigned score and classification."
    )
    key_points_addressed: List[str] = Field(
        default_factory=list,
        description="Core components or aspects of the question that the response addressed."
    )
    key_points_missed: List[str] = Field(
        default_factory=list,
        description="Core components or aspects of the question that the response ignored or drifted from."
    )
    confidence: ConfidenceLevel = Field(
        default=ConfidenceLevel.HIGH,
        description="Confidence level in the assessment."
    )
    evaluator_model: str = Field(
        default="deterministic-semantic",
        description="Name of the model or algorithm used to produce the judgment."
    )

    @field_validator("score")
    @classmethod
    def round_score(cls, v: float) -> float:
        return round(v, 2)


class EvaluationRunStatus(str, Enum):
    """Status of an orchestrated evaluation run."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"


class RetrievedEvidenceItem(BaseModel):
    """Structured representation of reference evidence extracted from ChromaDB."""
    chunk_id: str
    text: str
    similarity_score: float
    dataset_name: str
    source_record_id: str
    title: Optional[str] = None
    category: Optional[str] = None


class OrchestratedEvaluationResult(BaseModel):
    """Overall orchestrated evaluation package containing individual judge results."""
    submission_id: str
    status: EvaluationRunStatus
    relevance: Optional[RelevanceJudgeOutput] = None
    retrieved_evidence: List[RetrievedEvidenceItem] = Field(default_factory=list)
    # Accuracy, Hallucination, and Completeness will be added in M2.2, M2.3, and beyond
    accuracy: Optional[dict] = None
    hallucination: Optional[dict] = None
    completeness: Optional[dict] = None
    overall_score: Optional[float] = None
    verdict: Optional[str] = None
    created_at: str
