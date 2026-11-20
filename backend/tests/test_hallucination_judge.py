"""
test_hallucination_judge.py — Comprehensive tests for Milestone 2.3 (Hallucination Detection Agent).
Tests atomic claim decomposition, claim verification against reference evidence,
unsupported claim flagging, contradiction detection, severity bands, and orchestrator integration.
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.database import Base, get_db
from app.main import app
from app.models.evaluation import EvaluationSubmission
from app.schemas.judge import (
    HallucinationJudgeInput,
    HallucinationSeverity,
    ClaimVerificationStatus,
    RetrievedEvidenceItem,
)
from app.agents.hallucination_judge import (
    HallucinationDetectionAgent,
    _classify_hallucination_severity,
)
from app.agents.llm_client import BaseLLMClient
from app.services.orchestrator import EvaluationOrchestrator


# ---------------------------------------------------------------------------
# Test database setup (In-Memory SQLite with StaticPool)
# ---------------------------------------------------------------------------
TEST_DATABASE_URL = "sqlite:///:memory:"

test_engine = create_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture(autouse=True)
def setup_test_db():
    Base.metadata.create_all(bind=test_engine)
    app.dependency_overrides[get_db] = override_get_db
    yield
    Base.metadata.drop_all(bind=test_engine)
    app.dependency_overrides.clear()


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


# ---------------------------------------------------------------------------
# Mock LLM Client for testing Hallucination LLM path
# ---------------------------------------------------------------------------
class MockHallucinationLLMClient(BaseLLMClient):
    def __init__(self, score: float, severity: str, reasoning: str):
        self.score = score
        self.severity = severity
        self.reasoning = reasoning
        self.model = "mock-gpt-4o"

    def generate(self, system_prompt: str, user_prompt: str, temperature: float = 0.0) -> str:
        return f"""
        {{
            "score": {self.score},
            "hallucination_detected": {str(self.score > 0.0).lower()},
            "severity": "{self.severity}",
            "reasoning": "{self.reasoning}",
            "claims": [
                {{
                    "claim": "Sample candidate statement.",
                    "status": "supported",
                    "explanation": "Verified against reference text.",
                    "evidence_excerpt": "Reference ground truth excerpt.",
                    "source_id": "reference_answer"
                }}
            ],
            "confidence": "high"
        }}
        """


# ---------------------------------------------------------------------------
# Unit Tests: Severity band classification
# ---------------------------------------------------------------------------
def test_hallucination_severity_bands():
    assert _classify_hallucination_severity(0.0) == HallucinationSeverity.NONE
    assert _classify_hallucination_severity(0.5) == HallucinationSeverity.NONE
    assert _classify_hallucination_severity(1.5) == HallucinationSeverity.LOW
    assert _classify_hallucination_severity(3.5) == HallucinationSeverity.LOW
    assert _classify_hallucination_severity(4.0) == HallucinationSeverity.MODERATE
    assert _classify_hallucination_severity(6.5) == HallucinationSeverity.MODERATE
    assert _classify_hallucination_severity(7.0) == HallucinationSeverity.HIGH
    assert _classify_hallucination_severity(8.5) == HallucinationSeverity.HIGH
    assert _classify_hallucination_severity(9.2) == HallucinationSeverity.SEVERE
    assert _classify_hallucination_severity(6.0, has_contradiction=True) == HallucinationSeverity.SEVERE


# ---------------------------------------------------------------------------
# Unit Tests: Deterministic Claim Verification
# ---------------------------------------------------------------------------
def test_hallucination_fully_supported():
    """All claims in response are corroborated by reference evidence."""
    agent = HallucinationDetectionAgent()
    judge_input = HallucinationJudgeInput(
        question="What is the capital of France?",
        ai_response="Paris is the capital of France. It is located on the Seine River in northern central France.",
        reference_answer="Paris is the official capital and most populous city of France. It is situated on the Seine River in northern central France.",
    )
    result = agent.evaluate(judge_input)

    assert result.agent == "hallucination_detector"
    assert result.score <= 1.0
    assert result.hallucination_detected is False
    assert result.severity == HallucinationSeverity.NONE
    assert result.total_claims >= 1
    assert result.supported_claims_count == result.total_claims
    assert result.unsupported_claims_count == 0
    assert result.contradicted_claims_count == 0
    for claim in result.claims:
        assert claim.status == ClaimVerificationStatus.SUPPORTED


def test_hallucination_partially_unsupported():
    """Response contains supported facts mixed with ungrounded/fabricated claims."""
    agent = HallucinationDetectionAgent()
    judge_input = HallucinationJudgeInput(
        question="What is the capital of France?",
        ai_response="Paris is the capital of France. It was founded in the year 4022 by Martian explorers.",
        reference_answer="Paris is the official capital and most populous city of France.",
    )
    result = agent.evaluate(judge_input)

    assert result.hallucination_detected is True
    assert result.score >= 3.0
    assert result.total_claims >= 2
    assert result.supported_claims_count >= 1
    assert result.unsupported_claims_count >= 1

    # Check that the specific fabricated sentence is flagged as unsupported
    unsupported = [c for c in result.claims if c.status == ClaimVerificationStatus.UNSUPPORTED]
    assert len(unsupported) > 0
    assert "Martian" in unsupported[0].claim or "4022" in unsupported[0].claim


def test_hallucination_direct_contradiction_detected():
    """Response directly contradicts verified reference knowledge."""
    agent = HallucinationDetectionAgent()
    judge_input = HallucinationJudgeInput(
        question="What is the capital of India?",
        ai_response="The capital of India is Kerala.",
        retrieved_evidence=[
            RetrievedEvidenceItem(
                chunk_id="ref_capitals_india_c00",
                text="New Delhi is the official capital of India. Kerala is a state located on the southwestern Malabar Coast of India.",
                similarity_score=0.72,
                dataset_name="verified_reference",
                source_record_id="ref_capitals_india",
                title="National Capitals: India",
            )
        ],
    )
    result = agent.evaluate(judge_input)

    assert result.hallucination_detected is True
    assert result.score >= 7.0
    assert result.contradicted_claims_count >= 1
    contradicted = [c for c in result.claims if c.status == ClaimVerificationStatus.CONTRADICTED]
    assert len(contradicted) > 0
    assert "Kerala" in contradicted[0].claim


def test_hallucination_no_evidence_available():
    """Zero reference evidence results in unverified status with low confidence."""
    agent = HallucinationDetectionAgent()
    judge_input = HallucinationJudgeInput(
        question="What is quantum computing?",
        ai_response="Quantum computing utilizes qubits that can exist in superposition.",
        reference_answer=None,
        source_document=None,
        retrieved_evidence=[],
    )
    result = agent.evaluate(judge_input)

    assert result.hallucination_detected is True
    assert result.score == 5.0
    assert result.severity == HallucinationSeverity.MODERATE
    assert "No reference evidence" in result.reasoning


def test_hallucination_with_llm_client():
    """Tests execution through an active LLM client."""
    mock_llm = MockHallucinationLLMClient(
        score=0.0,
        severity="none",
        reasoning="All candidate claims verified and supported by reference.",
    )
    agent = HallucinationDetectionAgent(llm_client=mock_llm)
    judge_input = HallucinationJudgeInput(
        question="What is water made of?",
        ai_response="Water is made of two hydrogen atoms and one oxygen atom.",
        reference_answer="Water consists of hydrogen and oxygen molecules in a 2:1 ratio.",
    )
    result = agent.evaluate(judge_input)

    assert result.score == 0.0
    assert result.hallucination_detected is False
    assert result.severity == HallucinationSeverity.NONE
    assert result.evaluator_model == "mock-gpt-4o"


# ---------------------------------------------------------------------------
# Integration Tests: Evaluation Orchestrator running all three judges
# ---------------------------------------------------------------------------
def test_orchestrator_runs_all_three_judges():
    """Orchestrator coordinates Relevance, Accuracy, and Hallucination judges."""
    db = TestingSessionLocal()
    submission = EvaluationSubmission(
        question="What is the capital of France?",
        ai_response="Paris is the capital of France.",
        reference_answer="Paris is the capital of France.",
        status="pending",
    )
    db.add(submission)
    db.commit()
    db.refresh(submission)

    orchestrator = EvaluationOrchestrator()
    result = orchestrator.evaluate_submission(db=db, submission_id=submission.id)

    assert result.submission_id == str(submission.id)
    assert result.status == "completed"
    assert result.relevance is not None
    assert result.relevance.score >= 8.5
    assert result.accuracy is not None
    assert result.accuracy.score >= 8.5
    assert result.hallucination is not None
    assert result.hallucination.score <= 1.5
    assert result.overall_score is not None
    assert result.overall_score >= 8.5
    assert "Reliable" in result.verdict or "Accuracy" in result.verdict

    # Check persistence in database
    updated = db.query(EvaluationSubmission).filter_by(id=submission.id).first()
    assert updated.status == "completed"
    assert "hallucination" in updated.evaluation_result
    db.close()


# ---------------------------------------------------------------------------
# API Tests: FastAPI Endpoints
# ---------------------------------------------------------------------------
def test_api_judge_hallucination_endpoint(client):
    payload = {
        "question": "What is the capital of Japan?",
        "ai_response": "The capital of Japan is Tokyo. It was founded in 1603.",
        "reference_answer": "Tokyo is the official capital and largest city of Japan.",
    }
    response = client.post("/api/v1/judge/hallucination", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["agent"] == "hallucination_detector"
    assert "score" in data
    assert "severity" in data
    assert "claims" in data
    assert len(data["claims"]) >= 1
    assert "supported_claims_count" in data
