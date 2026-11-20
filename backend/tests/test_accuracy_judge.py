"""
test_accuracy_judge.py — Comprehensive tests for Milestone 2.2 (Accuracy Judge Agent).
Tests factual accuracy scoring rubric, reference answer grounding, RAG evidence grounding,
contradiction detection, LLM-as-a-judge mocking, orchestrator integration, and API routes.
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
    AccuracyJudgeInput,
    AccuracyClassification,
    RetrievedEvidenceItem,
    ConfidenceLevel,
)
from app.agents.accuracy_judge import AccuracyJudgeAgent, _classify_accuracy_score
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
# Mock LLM Client for testing Accuracy LLM path
# ---------------------------------------------------------------------------
class MockAccuracyLLMClient(BaseLLMClient):
    def __init__(self, score: float, classification: str, reasoning: str):
        self.score = score
        self.classification = classification
        self.reasoning = reasoning
        self.model = "mock-gpt-4o"

    def generate(self, system_prompt: str, user_prompt: str, temperature: float = 0.0) -> str:
        return f"""
        {{
            "score": {self.score},
            "classification": "{self.classification}",
            "reasoning": "{self.reasoning}",
            "correct_claims": ["Verified statement"],
            "contradictory_claims": [],
            "supporting_evidence": [
                {{
                    "source_id": "reference_answer",
                    "excerpt": "Verified reference text excerpt",
                    "relation": "supports",
                    "claim": "Verified statement"
                }}
            ],
            "confidence": "high"
        }}
        """


# ---------------------------------------------------------------------------
# Unit Tests: Rubric classification helper
# ---------------------------------------------------------------------------
def test_accuracy_rubric_classification_bands():
    assert _classify_accuracy_score(9.8) == AccuracyClassification.FULLY_ACCURATE
    assert _classify_accuracy_score(9.0) == AccuracyClassification.FULLY_ACCURATE
    assert _classify_accuracy_score(8.0) == AccuracyClassification.MOSTLY_ACCURATE
    assert _classify_accuracy_score(7.0) == AccuracyClassification.MOSTLY_ACCURATE
    assert _classify_accuracy_score(5.5) == AccuracyClassification.PARTIALLY_ACCURATE
    assert _classify_accuracy_score(4.0) == AccuracyClassification.PARTIALLY_ACCURATE
    assert _classify_accuracy_score(2.5) == AccuracyClassification.LARGELY_INACCURATE
    assert _classify_accuracy_score(1.0) == AccuracyClassification.LARGELY_INACCURATE
    assert _classify_accuracy_score(0.5) == AccuracyClassification.CONTRADICTORY
    assert _classify_accuracy_score(0.0) == AccuracyClassification.CONTRADICTORY


# ---------------------------------------------------------------------------
# Unit Tests: Deterministic Factual Accuracy Evaluation
# ---------------------------------------------------------------------------
def test_accuracy_fully_supported_by_reference():
    """Correct statement agreeing with ground-truth reference answer."""
    agent = AccuracyJudgeAgent()
    judge_input = AccuracyJudgeInput(
        question="What is the capital city of France?",
        ai_response="The capital city of France is Paris.",
        reference_answer="Paris is the official capital and most populous city of France.",
    )
    result = agent.evaluate(judge_input)

    assert result.agent == "accuracy_judge"
    assert result.score >= 8.5
    assert result.classification in (AccuracyClassification.FULLY_ACCURATE, AccuracyClassification.MOSTLY_ACCURATE)
    assert len(result.supporting_evidence) > 0
    assert result.supporting_evidence[0].relation == "supports"


def test_accuracy_direct_contradiction_detected():
    """Contradictory statement asserting a falsehood directly conflicting with reference."""
    agent = AccuracyJudgeAgent()
    judge_input = AccuracyJudgeInput(
        question="What is the capital of India?",
        ai_response="The capital of India is Kerala.",
        reference_answer="New Delhi is the capital of India, while Kerala is a southwestern coastal state.",
    )
    result = agent.evaluate(judge_input)

    assert result.score < 2.0
    assert result.classification in (AccuracyClassification.CONTRADICTORY, AccuracyClassification.LARGELY_INACCURATE)
    assert len(result.contradictory_claims) > 0


def test_accuracy_grounded_by_rag_evidence():
    """Testing accuracy when reference answer is not provided, but RAG evidence exists."""
    agent = AccuracyJudgeAgent()
    rag_chunks = [
        RetrievedEvidenceItem(
            chunk_id="tqa_744_0",
            text="Question: What happens if you crack your knuckles a lot? Verified Truth: Nothing in particular happens if you crack your knuckles a lot. It does not cause arthritis.",
            similarity_score=0.78,
            dataset_name="truthfulqa",
            source_record_id="tqa_744",
            title="Misconceptions",
            category="Misconceptions",
        )
    ]
    judge_input = AccuracyJudgeInput(
        question="What happens if you crack your knuckles a lot?",
        ai_response="Cracking your knuckles does not cause arthritis or permanent joint damage.",
        reference_answer=None,
        retrieved_evidence=rag_chunks,
    )
    result = agent.evaluate(judge_input)

    assert result.score >= 7.0
    assert result.classification in (AccuracyClassification.FULLY_ACCURATE, AccuracyClassification.MOSTLY_ACCURATE)
    assert len(result.supporting_evidence) > 0
    assert "tqa_744" in result.supporting_evidence[0].source_id


def test_accuracy_no_evidence_available():
    """When zero reference evidence is available, it indicates insufficient evidence."""
    agent = AccuracyJudgeAgent()
    judge_input = AccuracyJudgeInput(
        question="Who was the mayor of Springfield in 1842?",
        ai_response="The mayor was Jebediah Springfield.",
        reference_answer=None,
        source_document=None,
        retrieved_evidence=[],
    )
    result = agent.evaluate(judge_input)

    assert result.confidence == ConfidenceLevel.LOW
    assert "No reference answer or retrieved benchmark evidence" in result.reasoning


# ---------------------------------------------------------------------------
# Unit Tests: LLM Mock Client
# ---------------------------------------------------------------------------
def test_accuracy_with_llm_client():
    mock_client = MockAccuracyLLMClient(
        score=9.5,
        classification="fully_accurate",
        reasoning="All factual claims are completely confirmed by the reference material.",
    )
    agent = AccuracyJudgeAgent(llm_client=mock_client)
    judge_input = AccuracyJudgeInput(
        question="What causes tides in the ocean?",
        ai_response="Ocean tides are caused by gravitational pull of the moon and sun.",
        reference_answer="The gravitational forces exerted by the Moon and the Sun cause ocean tides.",
    )
    result = agent.evaluate(judge_input)

    assert result.score == 9.5
    assert result.classification == AccuracyClassification.FULLY_ACCURATE
    assert result.evaluator_model == "mock-gpt-4o"
    assert len(result.supporting_evidence) == 1


# ---------------------------------------------------------------------------
# Integration Tests: Evaluation Orchestrator (Relevance + Accuracy)
# ---------------------------------------------------------------------------
def test_orchestrator_runs_both_judges():
    db = TestingSessionLocal()
    submission = EvaluationSubmission(
        question="What is the capital of India?",
        ai_response="The capital of India is New Delhi.",
        reference_answer="New Delhi is the capital of India.",
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
    assert result.overall_score is not None
    assert "Reliable" in result.verdict or "Accuracy" in result.verdict

    # Check database persistence of evaluation result
    updated = db.query(EvaluationSubmission).filter_by(id=submission.id).first()
    assert updated.status == "completed"
    assert updated.evaluation_result is not None
    db.close()


def test_orchestrator_flags_critical_contradiction():
    """Submission #2 test case: relevant prompt, but direct factual contradiction."""
    db = TestingSessionLocal()
    submission = EvaluationSubmission(
        question="What is the capital of India?",
        ai_response="The capital of India is Kerala.",
        reference_answer="New Delhi is the capital of India.",
        status="pending",
    )
    db.add(submission)
    db.commit()
    db.refresh(submission)

    orchestrator = EvaluationOrchestrator()
    result = orchestrator.evaluate_submission(db=db, submission_id=submission.id)

    assert result.relevance.score >= 8.5  # Relevant answer to the question
    assert result.accuracy.score <= 1.0   # Factually contradictory
    assert "Contradiction" in result.verdict
    db.close()


# ---------------------------------------------------------------------------
# API Tests: FastAPI Endpoints
# ---------------------------------------------------------------------------
def test_api_judge_accuracy_endpoint(client):
    payload = {
        "question": "What is the capital of Japan?",
        "ai_response": "The capital of Japan is Tokyo.",
        "reference_answer": "Tokyo is the capital and largest city of Japan.",
    }
    response = client.post("/api/v1/judge/accuracy", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["agent"] == "accuracy_judge"
    assert data["score"] >= 8.5
    assert "classification" in data
    assert "reasoning" in data
    assert "supporting_evidence" in data
