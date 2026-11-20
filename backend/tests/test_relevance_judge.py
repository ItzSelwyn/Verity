"""
test_relevance_judge.py — Comprehensive tests for Milestone 2.1 (Relevance Judge Agent).
Tests scoring rubric, categorical classifications, reasoning outputs, deterministic fallback,
LLM-as-a-judge mocking, orchestrator integration, and FastAPI endpoints.
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
    RelevanceJudgeInput,
    RelevanceClassification,
    ConfidenceLevel,
)
from app.agents.relevance_judge import RelevanceJudgeAgent, _classify_score
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
    return TestClient(app)


# ---------------------------------------------------------------------------
# Mock LLM Client for testing LLM evaluation path
# ---------------------------------------------------------------------------
class MockLLMClient(BaseLLMClient):
    def __init__(self, return_score: float, return_reasoning: str, classification: str = "fully_relevant"):
        self.return_score = return_score
        self.return_reasoning = return_reasoning
        self.classification = classification
        self.model = "mock-gpt-4o"

    def generate(self, system_prompt: str, user_prompt: str, temperature: float = 0.0) -> str:
        return f"""
        {{
            "score": {self.return_score},
            "classification": "{self.classification}",
            "reasoning": "{self.return_reasoning}",
            "key_points_addressed": ["Primary prompt addressed"],
            "key_points_missed": [],
            "confidence": "high"
        }}
        """


# ---------------------------------------------------------------------------
# Unit Tests: Rubric classification helper
# ---------------------------------------------------------------------------
def test_rubric_classification_bands():
    assert _classify_score(9.5) == RelevanceClassification.FULLY_RELEVANT
    assert _classify_score(9.0) == RelevanceClassification.FULLY_RELEVANT
    assert _classify_score(8.0) == RelevanceClassification.MOSTLY_RELEVANT
    assert _classify_score(7.0) == RelevanceClassification.MOSTLY_RELEVANT
    assert _classify_score(6.0) == RelevanceClassification.PARTIALLY_RELEVANT
    assert _classify_score(4.0) == RelevanceClassification.PARTIALLY_RELEVANT
    assert _classify_score(3.0) == RelevanceClassification.OFF_TOPIC
    assert _classify_score(1.0) == RelevanceClassification.OFF_TOPIC
    assert _classify_score(0.5) == RelevanceClassification.UNRELATED
    assert _classify_score(0.0) == RelevanceClassification.UNRELATED


# ---------------------------------------------------------------------------
# Unit Tests: Deterministic Semantic Relevance Evaluation
# ---------------------------------------------------------------------------
def test_relevance_fully_relevant_benchmark():
    """TruthfulQA benchmark pair: directly answering the prompt."""
    agent = RelevanceJudgeAgent()
    judge_input = RelevanceJudgeInput(
        question="What happens if you crack your knuckles a lot?",
        ai_response="Nothing in particular happens to your joints if you crack your knuckles a lot. It does not cause arthritis.",
    )
    result = agent.evaluate(judge_input)

    assert result.agent == "relevance_judge"
    assert result.score >= 8.5
    assert result.classification in (RelevanceClassification.FULLY_RELEVANT, RelevanceClassification.MOSTLY_RELEVANT)
    assert len(result.reasoning) > 10
    assert len(result.key_points_addressed) > 0


def test_relevance_partially_relevant():
    """Question has multiple parts, answer only touches on one part broadly."""
    agent = RelevanceJudgeAgent()
    judge_input = RelevanceJudgeInput(
        question="What is the capital of France and what are its top three tourist attractions?",
        ai_response="France is a country located in Western Europe known for its wine and cuisine.",
    )
    result = agent.evaluate(judge_input)

    assert result.score < 8.5
    assert result.classification in (RelevanceClassification.PARTIALLY_RELEVANT, RelevanceClassification.MOSTLY_RELEVANT)


def test_relevance_off_topic_drift():
    """Mentions keywords but completely drifts from the actual question."""
    agent = RelevanceJudgeAgent()
    judge_input = RelevanceJudgeInput(
        question="How do nuclear power plants generate electricity?",
        ai_response="Nuclear weapons were first developed during World War II in the Manhattan Project led by J. Robert Oppenheimer.",
    )
    result = agent.evaluate(judge_input)

    assert result.score < 6.0
    assert result.classification in (RelevanceClassification.OFF_TOPIC, RelevanceClassification.PARTIALLY_RELEVANT)


def test_relevance_completely_unrelated():
    """Completely unrelated response."""
    agent = RelevanceJudgeAgent()
    judge_input = RelevanceJudgeInput(
        question="What are the health benefits of drinking water every day?",
        ai_response="Python 3.11 introduced substantial performance improvements including adaptive bytecode specialization.",
    )
    result = agent.evaluate(judge_input)

    assert result.score < 2.0
    assert result.classification in (RelevanceClassification.UNRELATED, RelevanceClassification.OFF_TOPIC)


def test_relevance_refusal_response():
    """Refusal responses ('I don't know') should receive low relevance."""
    agent = RelevanceJudgeAgent()
    judge_input = RelevanceJudgeInput(
        question="What is the chemical formula for photosynthesis?",
        ai_response="I don't know the answer to this question.",
    )
    result = agent.evaluate(judge_input)

    assert result.score <= 3.0
    assert result.classification == RelevanceClassification.OFF_TOPIC


# ---------------------------------------------------------------------------
# Unit Tests: LLM-backed evaluation with Mock Client
# ---------------------------------------------------------------------------
def test_relevance_with_llm_client():
    mock_client = MockLLMClient(
        return_score=9.8,
        return_reasoning="Directly answers the query with clear precision.",
        classification="fully_relevant",
    )
    agent = RelevanceJudgeAgent(llm_client=mock_client)
    judge_input = RelevanceJudgeInput(
        question="What causes tides in the ocean?",
        ai_response="Ocean tides are caused primarily by the gravitational pull of the Moon and the Sun on the Earth.",
    )
    result = agent.evaluate(judge_input)

    assert result.score == 9.8
    assert result.classification == RelevanceClassification.FULLY_RELEVANT
    assert result.reasoning == "Directly answers the query with clear precision."
    assert result.evaluator_model == "mock-gpt-4o"


# ---------------------------------------------------------------------------
# Integration Tests: Evaluation Orchestrator Service
# ---------------------------------------------------------------------------
def test_orchestrator_evaluate_submission():
    db = TestingSessionLocal()
    submission = EvaluationSubmission(
        question="Why is the sky blue?",
        ai_response="The sky appears blue due to Rayleigh scattering of sunlight by particles in the atmosphere.",
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
    assert result.relevance.score >= 8.0

    # Ensure DB record status was updated
    updated = db.query(EvaluationSubmission).filter_by(id=submission.id).first()
    assert updated.status == "completed"
    db.close()


def test_orchestrator_submission_not_found():
    db = TestingSessionLocal()
    orchestrator = EvaluationOrchestrator()
    with pytest.raises(ValueError, match="not found"):
        orchestrator.evaluate_submission(db=db, submission_id=99999)
    db.close()


# ---------------------------------------------------------------------------
# API Tests: FastAPI Endpoints
# ---------------------------------------------------------------------------
def test_api_judge_relevance_valid(client):
    payload = {
        "question": "What is machine learning?",
        "ai_response": "Machine learning is a subfield of artificial intelligence focused on building algorithms that learn from data.",
    }
    response = client.post("/api/v1/judge/relevance", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["agent"] == "relevance_judge"
    assert "score" in data
    assert 0.0 <= data["score"] <= 10.0
    assert "classification" in data
    assert "reasoning" in data


def test_api_judge_relevance_validation_error(client):
    payload = {
        "question": "too",  # Min length is 5
        "ai_response": "Short answer",
    }
    response = client.post("/api/v1/judge/relevance", json=payload)
    assert response.status_code == 422


def test_api_evaluate_stored_submission(client):
    # 1. Create a submission
    create_resp = client.post(
        "/api/v1/evaluations/",
        json={
            "question": "How do vaccines work to protect human health?",
            "ai_response": "Vaccines stimulate the human immune system by introducing a harmless antigen, enabling the body to produce antibodies.",
        },
    )
    assert create_resp.status_code == 201
    submission_id = create_resp.json()["id"]

    # 2. Trigger evaluation on stored submission
    eval_resp = client.post(f"/api/v1/judge/submissions/{submission_id}/evaluate")
    assert eval_resp.status_code == 200
    eval_data = eval_resp.json()

    assert eval_data["submission_id"] == str(submission_id)
    assert eval_data["status"] == "completed"
    assert eval_data["relevance"] is not None
    assert eval_data["relevance"]["score"] >= 8.0
    assert "verdict" in eval_data


def test_api_evaluate_nonexistent_submission(client):
    response = client.post("/api/v1/judge/submissions/999999/evaluate")
    assert response.status_code == 404
