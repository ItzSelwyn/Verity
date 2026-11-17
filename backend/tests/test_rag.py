"""
test_rag.py — Automated test suite for Verity Reference Knowledge Base (Milestone 1.4).

Tests:
  1. DocumentChunker: 500-token chunking, 75-token overlap, boundary preservation.
  2. EmbeddingService: all-MiniLM-L6-v2 384-dim normalized vector generation.
  3. ChromaVectorStore: isolated temporary ChromaDB persistence and nearest-neighbor search.
  4. SemanticRetriever: query-time retrieval, similarity threshold, and provenance metadata.
  5. FastAPI Endpoints: /api/v1/retrieval/status and /api/v1/retrieval/query validation.
"""

import math
import shutil
import tempfile
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.rag.chunker import DocumentChunker, estimate_tokens
from app.rag.embedder import EmbeddingService
from app.rag.vector_store import ChromaVectorStore
from app.rag.retriever import SemanticRetriever


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def shared_embedder():
    """Shared EmbeddingService instance for the test module."""
    return EmbeddingService()


@pytest.fixture()
def temp_vector_store():
    """Isolated ChromaVectorStore in a temporary directory."""
    temp_dir = tempfile.mkdtemp(prefix="verity_test_chroma_")
    store = ChromaVectorStore(persist_directory=temp_dir, collection_name="test_collection")
    yield store
    # Cleanup temp directory
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture()
def client():
    """FastAPI TestClient for API tests."""
    with TestClient(app) as test_client:
        yield test_client


# ---------------------------------------------------------------------------
# 1. DocumentChunker Tests
# ---------------------------------------------------------------------------

def test_chunker_short_text():
    """Text shorter than target_tokens should return exactly one chunk."""
    chunker = DocumentChunker(target_tokens=500, overlap_tokens=75)
    short_text = "The quick brown fox jumps over the lazy dog. Paris is the capital of France."
    chunks = chunker.chunk_text(short_text)

    assert len(chunks) == 1
    assert chunks[0].chunk_index == 0
    assert chunks[0].total_chunks == 1
    assert chunks[0].text == short_text
    assert chunks[0].token_count == estimate_tokens(short_text)


def test_chunker_empty_or_whitespace():
    """Empty or whitespace text should return an empty chunk list."""
    chunker = DocumentChunker(target_tokens=500, overlap_tokens=75)
    assert chunker.chunk_text("") == []
    assert chunker.chunk_text("   \n\t  ") == []


def test_chunker_long_text_splits_with_overlap():
    """Long text should be partitioned into overlapping segments."""
    chunker = DocumentChunker(target_tokens=50, overlap_tokens=15)
    # Generate 15 distinct sentences (~150 words)
    sentences = [
        f"This is sentence number {i} containing some informative factual details about topic {i}."
        for i in range(1, 16)
    ]
    long_text = " ".join(sentences)

    chunks = chunker.chunk_text(long_text)

    assert len(chunks) > 1
    for i, c in enumerate(chunks):
        assert c.chunk_index == i
        assert c.total_chunks == len(chunks)
        assert len(c.text) > 0
        assert c.token_count <= 70  # roughly bounded around target


def test_chunker_invalid_overlap_raises():
    """overlap >= target should raise ValueError."""
    with pytest.raises(ValueError):
        DocumentChunker(target_tokens=50, overlap_tokens=50)


# ---------------------------------------------------------------------------
# 2. EmbeddingService Tests
# ---------------------------------------------------------------------------

def test_embed_text_dimensions_and_normalization(shared_embedder):
    """Embedding should return a 384-dimensional unit vector."""
    text = "The capital of Australia is Canberra."
    vec = shared_embedder.embed_text(text)

    assert len(vec) == 384
    # Check L2 normalization: sum(v^2) should be approximately 1.0
    norm = math.sqrt(sum(x * x for x in vec))
    assert abs(norm - 1.0) < 1e-3


def test_embed_documents_batch(shared_embedder):
    """Batch embedding should match input count and dimensions."""
    texts = [
        "First document about artificial intelligence.",
        "Second document discussing quantum computing.",
        "Third document analyzing climate patterns.",
    ]
    batch_vecs = shared_embedder.embed_documents(texts)

    assert len(batch_vecs) == 3
    for v in batch_vecs:
        assert len(v) == 384


# ---------------------------------------------------------------------------
# 3. ChromaVectorStore & SemanticRetriever Tests
# ---------------------------------------------------------------------------

def test_vector_store_add_and_retrieve(temp_vector_store, shared_embedder):
    """Indexed chunks should be retrievable via nearest-neighbor query."""
    docs = [
        "Canberra is the capital city of Australia, founded following federation.",
        "Tokyo is the bustling capital of Japan, known for high technology and finance.",
        "Paris is the capital of France, home to the Louvre and the Eiffel Tower.",
    ]
    ids = ["doc_australia", "doc_japan", "doc_france"]
    metas = [
        {"dataset_name": "squad", "dataset_split": "validation", "source_record_id": "aus_1", "title": "Australia"},
        {"dataset_name": "squad", "dataset_split": "validation", "source_record_id": "jpn_1", "title": "Japan"},
        {"dataset_name": "truthfulqa", "dataset_split": "validation", "source_record_id": "fra_1", "title": "France"},
    ]

    embeddings = shared_embedder.embed_documents(docs)
    temp_vector_store.add_chunks(ids=ids, documents=docs, embeddings=embeddings, metadatas=metas)

    assert temp_vector_store.count() == 3

    # Initialize retriever
    retriever = SemanticRetriever(vector_store=temp_vector_store, embedder=shared_embedder)

    # Query for France
    pkg = retriever.retrieve("What is the capital of France and what famous landmarks are there?", top_k=2)

    assert pkg.status == "sufficient"
    assert pkg.total_retrieved >= 1
    top_result = pkg.passages[0]
    assert top_result.chunk_id == "doc_france"
    assert top_result.dataset_name == "truthfulqa"
    assert top_result.source_record_id == "fra_1"
    assert top_result.similarity_score > 0.5
    assert "Eiffel Tower" in top_result.text


def test_retriever_empty_database():
    """An empty vector store should gracefully return insufficient_evidence."""
    temp_dir = tempfile.mkdtemp(prefix="verity_empty_chroma_")
    store = ChromaVectorStore(persist_directory=temp_dir, collection_name="empty_collection")
    retriever = SemanticRetriever(vector_store=store)

    pkg = retriever.retrieve("Any question here")
    assert pkg.status == "insufficient_evidence"
    assert pkg.total_retrieved == 0
    assert pkg.passages == []

    shutil.rmtree(temp_dir, ignore_errors=True)


def test_retriever_dataset_filter(temp_vector_store, shared_embedder):
    """Filtering by dataset_name should exclude other datasets."""
    docs = [
        "The Earth rotates around the Sun once every 365.25 days.",
        "The Moon orbits Earth every 27.3 days.",
    ]
    ids = ["earth_tqa", "moon_squad"]
    metas = [
        {"dataset_name": "truthfulqa", "dataset_split": "validation", "source_record_id": "tqa_1"},
        {"dataset_name": "squad", "dataset_split": "validation", "source_record_id": "sq_1"},
    ]
    embeddings = shared_embedder.embed_documents(docs)
    temp_vector_store.add_chunks(ids=ids, documents=docs, embeddings=embeddings, metadatas=metas)

    retriever = SemanticRetriever(vector_store=temp_vector_store, embedder=shared_embedder)

    # Search with squad filter
    res = retriever.retrieve("celestial orbits in solar system", top_k=5, dataset_filter="squad")
    for p in res.passages:
        assert p.dataset_name == "squad"


# ---------------------------------------------------------------------------
# 4. FastAPI Retrieval Endpoints
# ---------------------------------------------------------------------------

def test_api_retrieval_status(client):
    """GET /api/v1/retrieval/status should return 200 with online status."""
    response = client.get("/api/v1/retrieval/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "collection_name" in data
    assert "embedding_model" in data
    assert "total_chunks_indexed" in data


def test_api_retrieval_query_validation(client):
    """POST /api/v1/retrieval/query should reject invalid inputs."""
    # Too short (< 3 chars)
    res_short = client.post("/api/v1/retrieval/query", json={"query": "a"})
    assert res_short.status_code == 422

    # Whitespace only
    res_ws = client.post("/api/v1/retrieval/query", json={"query": "     "})
    assert res_ws.status_code == 422

    # Invalid top_k (> 20)
    res_k = client.post("/api/v1/retrieval/query", json={"query": "Valid question?", "top_k": 50})
    assert res_k.status_code == 422


def test_api_retrieval_query_valid(client):
    """POST /api/v1/retrieval/query should return 200 with structured response schema."""
    payload = {"query": "What is machine learning and neural networks?", "top_k": 3}
    response = client.post("/api/v1/retrieval/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "What is machine learning and neural networks?"
    assert "total_retrieved" in data
    assert "status" in data
    assert isinstance(data["passages"], list)
