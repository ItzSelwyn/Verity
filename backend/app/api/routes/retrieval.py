"""
api/routes/retrieval.py — HTTP route handlers for semantic retrieval & knowledge base.

Endpoints:
  POST /api/v1/retrieval/query   — Retrieve relevant reference passages for a question/claim
  GET  /api/v1/retrieval/status  — Inspect knowledge base status and indexed chunk count
"""

from fastapi import APIRouter, status
from app.schemas.retrieval import (
    RetrievalQueryRequest,
    RetrievalQueryResponse,
    EvidencePassageItem,
)
from app.rag.retriever import SemanticRetriever
from app.rag.vector_store import ChromaVectorStore
from app.core.config import EMBEDDING_MODEL_NAME, CHROMA_COLLECTION_NAME

router = APIRouter(prefix="/retrieval", tags=["Reference Knowledge Base"])

_retriever_instance = None


def get_retriever() -> SemanticRetriever:
    """Lazy initialization of the SemanticRetriever."""
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = SemanticRetriever()
    return _retriever_instance


@router.post(
    "/query",
    response_model=RetrievalQueryResponse,
    summary="Semantic retrieval of reference evidence",
)
def query_reference_evidence(request: RetrievalQueryRequest):
    """
    Search the ChromaDB reference knowledge base for the most semantically relevant
    passages supporting or contradicting the query.
    """
    retriever = get_retriever()
    pkg = retriever.retrieve(
        query=request.query,
        top_k=request.top_k,
        dataset_filter=request.dataset_filter,
    )

    items = [
        EvidencePassageItem(
            chunk_id=p.chunk_id,
            text=p.text,
            similarity_score=p.similarity_score,
            dataset_name=p.dataset_name,
            dataset_split=p.dataset_split,
            source_record_id=p.source_record_id,
            title=p.title,
            category=p.category,
            chunk_index=p.chunk_index,
            content_type=p.content_type,
            extra_metadata=p.extra_metadata,
        )
        for p in pkg.passages
    ]

    return RetrievalQueryResponse(
        query=pkg.query,
        total_retrieved=pkg.total_retrieved,
        status=pkg.status,
        passages=items,
    )


@router.get(
    "/status",
    summary="Knowledge Base Status",
)
def get_knowledge_base_status():
    """Return status of the ChromaDB reference knowledge base collection."""
    store = ChromaVectorStore()
    count = store.count()
    return {
        "status": "online",
        "collection_name": CHROMA_COLLECTION_NAME,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "total_chunks_indexed": count,
    }
