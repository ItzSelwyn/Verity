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


class AccuracyClassification(str, Enum):
    """Classification bands for factual accuracy against references."""
    FULLY_ACCURATE = "fully_accurate"        # 9.0 - 10.0: Fully supported, no factual contradictions
    MOSTLY_ACCURATE = "mostly_accurate"      # 7.0 - 8.9: Core facts correct with minor imprecision
    PARTIALLY_ACCURATE = "partially_accurate"# 4.0 - 6.9: Mixed verified facts and inaccuracies
    LARGELY_INACCURATE = "largely_inaccurate"# 1.0 - 3.9: Major factual conflicts with references
    CONTRADICTORY = "contradictory"          # 0.0 - 0.9: Directly and critically contradicted


class EvidenceCitation(BaseModel):
    """Citation linking a claim to supporting or contradicting source evidence."""
    source_id: str = Field(..., description="ID or label of source ('reference_answer', 'source_doc', chunk ID)")
    excerpt: str = Field(..., description="Relevant text excerpt from the reference")
    relation: str = Field(..., description="'supports' | 'contradicts' | 'insufficient'")
    claim: Optional[str] = Field(None, description="The specific candidate claim evaluated")


class AccuracyJudgeInput(BaseModel):
    """Input contract for the Accuracy Judge Agent."""
    question: str = Field(..., min_length=5, max_length=5000)
    ai_response: str = Field(..., min_length=5, max_length=20000)
    reference_answer: Optional[str] = Field(None, max_length=10000)
    source_document: Optional[str] = Field(None, max_length=50000)
    retrieved_evidence: List["RetrievedEvidenceItem"] = Field(default_factory=list)


class AccuracyJudgeOutput(BaseModel):
    """Structured output contract for the Accuracy Judge Agent."""
    agent: str = Field(default="accuracy_judge")
    score: float = Field(..., ge=0.0, le=10.0)
    classification: AccuracyClassification
    reasoning: str = Field(..., min_length=10)
    correct_claims: List[str] = Field(default_factory=list)
    contradictory_claims: List[str] = Field(default_factory=list)
    supporting_evidence: List[EvidenceCitation] = Field(default_factory=list)
    confidence: ConfidenceLevel = Field(default=ConfidenceLevel.HIGH)
    evaluator_model: str = Field(default="deterministic-fact-check")

    @field_validator("score")
    @classmethod
    def round_score(cls, v: float) -> float:
        return round(v, 2)


# ---------------------------------------------------------------------------
# Milestone 2.3: Hallucination Detection Schemas
# ---------------------------------------------------------------------------
class ClaimVerificationStatus(str, Enum):
    """Verification verdict for an atomic claim against reference evidence."""
    SUPPORTED = "supported"          # Claim is fully confirmed by available reference evidence
    UNSUPPORTED = "unsupported"      # Claim makes an assertion with zero backing in reference evidence
    CONTRADICTED = "contradicted"    # Claim directly conflicts with reference evidence


class ClaimVerification(BaseModel):
    """Fine-grained verification record for an individual atomic claim."""
    claim: str = Field(..., description="The individual factual claim extracted from the response")
    status: ClaimVerificationStatus = Field(..., description="'supported' | 'unsupported' | 'contradicted'")
    explanation: str = Field(..., description="Reasoning explaining why the claim is supported, unsupported, or contradicted")
    evidence_excerpt: Optional[str] = Field(None, description="Direct supporting or contradicting reference excerpt, if available")
    source_id: Optional[str] = Field(None, description="Identifier of the reference source")


class HallucinationSeverity(str, Enum):
    """Categorical severity band for hallucination risk."""
    NONE = "none"            # 0.0 - 0.9: No hallucination detected; all claims grounded
    LOW = "low"              # 1.0 - 3.5: Minor unsupported background nuance; core facts grounded
    MODERATE = "moderate"    # 3.6 - 6.5: Significant unverified or fabricated assertions
    HIGH = "high"            # 6.6 - 8.9: Major factual fabrication or multiple ungrounded assertions
    SEVERE = "severe"        # 9.0 - 10.0: Critical factual contradictions or pervasive fabrication


class HallucinationJudgeInput(BaseModel):
    """Input contract for the Hallucination Detection Agent."""
    question: str = Field(..., min_length=5, max_length=5000)
    ai_response: str = Field(..., min_length=5, max_length=20000)
    reference_answer: Optional[str] = Field(None, max_length=10000)
    source_document: Optional[str] = Field(None, max_length=50000)
    retrieved_evidence: List["RetrievedEvidenceItem"] = Field(default_factory=list)


class HallucinationJudgeOutput(BaseModel):
    """Structured output contract for the Hallucination Detection Agent."""
    agent: str = Field(default="hallucination_detector")
    score: float = Field(..., ge=0.0, le=10.0, description="Hallucination risk score (0.0 = zero hallucination, 10.0 = completely hallucinated)")
    hallucination_detected: bool = Field(..., description="Whether any claim was flagged as unsupported or contradicted")
    severity: HallucinationSeverity
    reasoning: str = Field(..., min_length=10)
    total_claims: int = Field(default=0)
    supported_claims_count: int = Field(default=0)
    unsupported_claims_count: int = Field(default=0)
    contradicted_claims_count: int = Field(default=0)
    claims: List[ClaimVerification] = Field(default_factory=list)
    confidence: ConfidenceLevel = Field(default=ConfidenceLevel.HIGH)
    evaluator_model: str = Field(default="deterministic-claim-checker")

    @field_validator("score")
    @classmethod
    def round_score(cls, v: float) -> float:
        return round(v, 2)


class OrchestratedEvaluationResult(BaseModel):
    """Overall orchestrated evaluation package containing individual judge results."""
    submission_id: str
    status: EvaluationRunStatus
    relevance: Optional[RelevanceJudgeOutput] = None
    accuracy: Optional[AccuracyJudgeOutput] = None
    hallucination: Optional[HallucinationJudgeOutput] = None
    retrieved_evidence: List[RetrievedEvidenceItem] = Field(default_factory=list)
    completeness: Optional[dict] = None
    overall_score: Optional[float] = None
    verdict: Optional[str] = None
    created_at: str
