"""
tests/test_evaluation.py — pytest test suite for the Verity API backend.

We use FastAPI's built-in TestClient (backed by httpx) to make HTTP requests
against the application without starting a real server. Each test uses a
completely isolated in-memory SQLite database so tests never interfere with
one another and never touch the development database (verity_dev.db).

How the DB isolation works
--------------------------
1. We create a fresh SQLAlchemy engine pointing to `sqlite:///:memory:`.
2. We create all tables on that engine.
3. We override the `get_db` FastAPI dependency so every request during the
   test uses a session bound to the in-memory engine instead of the real one.
4. The `client` fixture is function-scoped, so each test starts with a
   completely empty database.
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.main import app
from app.core.database import Base, get_db


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Valid sample payloads reused across multiple tests
VALID_QUESTION = "What is the capital of France and why is it important?"
VALID_AI_RESPONSE = (
    "The capital of France is Paris. It is important because it is the "
    "political, cultural, and economic centre of the country."
)


@pytest.fixture()
def client():
    """
    Provide a TestClient backed by an isolated in-memory SQLite database.

    A new engine and session factory are created for every test function,
    ensuring a clean slate each time.
    """
    # Create an in-memory SQLite engine for this test.
    # We use StaticPool so all sessions share the exact same in-memory connection;
    # otherwise sqlite:///:memory: wipes all tables on every new connection.
    test_engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    # Create all tables defined by our ORM models
    Base.metadata.create_all(bind=test_engine)

    # Session factory bound to the in-memory engine
    TestingSessionLocal = sessionmaker(
        autocommit=False, autoflush=False, bind=test_engine
    )

    def override_get_db():
        """Replacement for the real get_db dependency during tests."""
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    # Register the override with the FastAPI application
    app.dependency_overrides[get_db] = override_get_db

    # Yield the test client; cleanup happens after each test
    with TestClient(app) as test_client:
        yield test_client

    # Remove the override so other tests (or real usage) are not affected
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# 1. Health check
# ---------------------------------------------------------------------------

def test_health_check(client):
    """GET /api/v1/health should return 200 with status=healthy."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "Verity API"


# ---------------------------------------------------------------------------
# 2. Create evaluation — valid minimal payload
# ---------------------------------------------------------------------------

def test_create_evaluation_valid(client):
    """POST with the two required fields should return 201 and status=pending."""
    payload = {
        "question": VALID_QUESTION,
        "ai_response": VALID_AI_RESPONSE,
    }
    response = client.post("/api/v1/evaluations/", json=payload)
    assert response.status_code == 201
    data = response.json()
    assert "id" in data
    assert data["id"] > 0
    assert data["status"] == "pending"
    assert data["question"] == VALID_QUESTION
    assert data["ai_response"] == VALID_AI_RESPONSE
    assert data["reference_answer"] is None
    assert data["source_document"] is None


# ---------------------------------------------------------------------------
# 3. Create evaluation — all four fields
# ---------------------------------------------------------------------------

def test_create_evaluation_with_all_fields(client):
    """POST with all four fields should return 201 and preserve every field."""
    payload = {
        "question": VALID_QUESTION,
        "ai_response": VALID_AI_RESPONSE,
        "reference_answer": "Paris is the capital of France.",
        "source_document": "France is a country in Western Europe. Its capital is Paris.",
    }
    response = client.post("/api/v1/evaluations/", json=payload)
    assert response.status_code == 201
    data = response.json()
    assert data["reference_answer"] == "Paris is the capital of France."
    assert "France is a country" in data["source_document"]


# ---------------------------------------------------------------------------
# 4. Missing required field — question
# ---------------------------------------------------------------------------

def test_create_evaluation_missing_question(client):
    """POST without 'question' should return 422 Unprocessable Entity."""
    payload = {"ai_response": VALID_AI_RESPONSE}
    response = client.post("/api/v1/evaluations/", json=payload)
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# 5. Missing required field — ai_response
# ---------------------------------------------------------------------------

def test_create_evaluation_missing_ai_response(client):
    """POST without 'ai_response' should return 422 Unprocessable Entity."""
    payload = {"question": VALID_QUESTION}
    response = client.post("/api/v1/evaluations/", json=payload)
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# 6. Whitespace-only question
# ---------------------------------------------------------------------------

def test_create_evaluation_whitespace_question(client):
    """POST with a whitespace-only question should return 422."""
    payload = {
        "question": "     \t\n   ",
        "ai_response": VALID_AI_RESPONSE,
    }
    response = client.post("/api/v1/evaluations/", json=payload)
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# 7. Question too short (< 10 chars)
# ---------------------------------------------------------------------------

def test_create_evaluation_too_short_question(client):
    """POST with a question shorter than 10 characters should return 422."""
    payload = {
        "question": "Short?",   # Only 6 characters
        "ai_response": VALID_AI_RESPONSE,
    }
    response = client.post("/api/v1/evaluations/", json=payload)
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# 8. Get evaluation — found
# ---------------------------------------------------------------------------

def test_get_evaluation_found(client):
    """GET /api/v1/evaluations/{id} for an existing record should return 200."""
    # First create a submission
    create_payload = {
        "question": VALID_QUESTION,
        "ai_response": VALID_AI_RESPONSE,
    }
    create_response = client.post("/api/v1/evaluations/", json=create_payload)
    assert create_response.status_code == 201
    submission_id = create_response.json()["id"]

    # Now fetch it
    get_response = client.get(f"/api/v1/evaluations/{submission_id}")
    assert get_response.status_code == 200
    data = get_response.json()
    assert data["id"] == submission_id
    assert data["question"] == VALID_QUESTION
    assert data["status"] == "pending"


# ---------------------------------------------------------------------------
# 9. Get evaluation — not found
# ---------------------------------------------------------------------------

def test_get_evaluation_not_found(client):
    """GET /api/v1/evaluations/9999 for a non-existent record should return 404."""
    response = client.get("/api/v1/evaluations/9999")
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# 10. List evaluations — empty database
# ---------------------------------------------------------------------------

def test_list_evaluations_empty(client):
    """GET /api/v1/evaluations/ on an empty DB should return items=[] and total=0."""
    response = client.get("/api/v1/evaluations/")
    assert response.status_code == 200
    data = response.json()
    assert data["items"] == []
    assert data["total"] == 0
    assert data["page"] == 1
    assert data["page_size"] == 10


# ---------------------------------------------------------------------------
# 11. List evaluations — with data
# ---------------------------------------------------------------------------

def test_list_evaluations_with_data(client):
    """After creating 2 submissions, the list endpoint should return both."""
    for i in range(2):
        client.post(
            "/api/v1/evaluations/",
            json={
                "question": f"Question number {i + 1} — asking about topic {i + 1}",
                "ai_response": f"This is the AI response for question number {i + 1} and it is detailed enough.",
            },
        )

    response = client.get("/api/v1/evaluations/")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 2
    assert len(data["items"]) == 2


# ---------------------------------------------------------------------------
# 12. List evaluations — pagination
# ---------------------------------------------------------------------------

def test_list_evaluations_pagination(client):
    """With page_size=1, only 1 item should be returned per page."""
    # Create 2 submissions
    for i in range(2):
        client.post(
            "/api/v1/evaluations/",
            json={
                "question": f"Pagination test question number {i + 1} — topic details here",
                "ai_response": f"Detailed AI response for pagination test question {i + 1} goes here.",
            },
        )

    response = client.get("/api/v1/evaluations/?page=1&page_size=1")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 2       # Both records exist
    assert len(data["items"]) == 1  # But only 1 is returned on this page
    assert data["page"] == 1
    assert data["page_size"] == 1


# ---------------------------------------------------------------------------
# 13. Root health check & root evaluations endpoints
# ---------------------------------------------------------------------------

def test_root_health_check(client):
    """GET /health should return 200 with status=healthy."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "Verity API"


def test_root_evaluations_endpoints(client):
    """POST /evaluations/ and GET /evaluations/ should work directly."""
    payload = {
        "question": VALID_QUESTION,
        "ai_response": VALID_AI_RESPONSE,
    }
    create_res = client.post("/evaluations/", json=payload)
    assert create_res.status_code == 201
    created_id = create_res.json()["id"]

    get_res = client.get(f"/evaluations/{created_id}")
    assert get_res.status_code == 200
    assert get_res.json()["id"] == created_id

    list_res = client.get("/evaluations/")
    assert list_res.status_code == 200
    assert list_res.json()["total"] == 1


# ---------------------------------------------------------------------------
# 14. Input validation: blank optional fields coerced to None
# ---------------------------------------------------------------------------

def test_create_evaluation_blank_optional_coerced_to_none(client):
    """Whitespace-only reference_answer and source_document should become None."""
    payload = {
        "question": VALID_QUESTION,
        "ai_response": VALID_AI_RESPONSE,
        "reference_answer": "   \n\t  ",
        "source_document": "      ",
    }
    response = client.post("/api/v1/evaluations/", json=payload)
    assert response.status_code == 201
    data = response.json()
    assert data["reference_answer"] is None
    assert data["source_document"] is None


# ---------------------------------------------------------------------------
# 15. Input validation: AI response errors
# ---------------------------------------------------------------------------

def test_create_evaluation_whitespace_ai_response(client):
    """Whitespace-only ai_response should return 422."""
    payload = {
        "question": VALID_QUESTION,
        "ai_response": "     \n\t  ",
    }
    response = client.post("/api/v1/evaluations/", json=payload)
    assert response.status_code == 422


def test_create_evaluation_ai_response_too_short(client):
    """ai_response shorter than 10 characters should return 422."""
    payload = {
        "question": VALID_QUESTION,
        "ai_response": "Too short",  # 9 chars
    }
    response = client.post("/api/v1/evaluations/", json=payload)
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# 16. Input validation: Maximum length limits
# ---------------------------------------------------------------------------

def test_create_evaluation_question_too_long(client):
    """question exceeding 5000 characters should return 422."""
    payload = {
        "question": "A" * 5001,
        "ai_response": VALID_AI_RESPONSE,
    }
    response = client.post("/api/v1/evaluations/", json=payload)
    assert response.status_code == 422


def test_create_evaluation_ai_response_too_long(client):
    """ai_response exceeding 20,000 characters should return 422."""
    payload = {
        "question": VALID_QUESTION,
        "ai_response": "B" * 20001,
    }
    response = client.post("/api/v1/evaluations/", json=payload)
    assert response.status_code == 422


def test_create_evaluation_reference_answer_too_long(client):
    """reference_answer exceeding 10,000 characters should return 422."""
    payload = {
        "question": VALID_QUESTION,
        "ai_response": VALID_AI_RESPONSE,
        "reference_answer": "C" * 10001,
    }
    response = client.post("/api/v1/evaluations/", json=payload)
    assert response.status_code == 422


def test_create_evaluation_source_document_too_long(client):
    """source_document exceeding 50,000 characters should return 422."""
    payload = {
        "question": VALID_QUESTION,
        "ai_response": VALID_AI_RESPONSE,
        "source_document": "D" * 50001,
    }
    response = client.post("/api/v1/evaluations/", json=payload)
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# 17. Pagination query parameter validation
# ---------------------------------------------------------------------------

def test_list_evaluations_invalid_page(client):
    """page < 1 should return 422."""
    response = client.get("/api/v1/evaluations/?page=0")
    assert response.status_code == 422


def test_list_evaluations_invalid_page_size(client):
    """page_size > 100 or < 1 should return 422."""
    res_zero = client.get("/api/v1/evaluations/?page_size=0")
    assert res_zero.status_code == 422

    res_large = client.get("/api/v1/evaluations/?page_size=101")
    assert res_large.status_code == 422
