"""
retriever.py — Semantic retrieval service for Verity AI evaluation pipeline.

Accepts evaluation questions or candidate statements, computes semantic embeddings,
retrieves the top-k nearest reference passages from ChromaDB, and formats evidence
with complete source provenance.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from app.rag.embedder import EmbeddingService
from app.rag.vector_store import ChromaVectorStore


@dataclass
class RetrievedEvidence:
    """A single retrieved evidence passage with provenance metadata."""
    chunk_id: str
    text: str
    similarity_score: float  # Normalized cosine similarity: 0.0 to 1.0
    dataset_name: str
    dataset_split: str
    source_record_id: str
    title: Optional[str] = None
    category: Optional[str] = None
    chunk_index: int = 0
    content_type: str = "context_chunk"
    extra_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalPackage:
    """Complete package of retrieved evidence for an evaluation query."""
    query: str
    passages: List[RetrievedEvidence]
    total_retrieved: int
    status: str  # "sufficient" | "insufficient_evidence"


class SemanticRetriever:
    """
    Retrieves evidence passages from the ChromaDB knowledge base.

    Parameters
    ----------
    vector_store : ChromaVectorStore, optional
        Vector store client instance.
    embedder : EmbeddingService, optional
        Embedding generator instance.
    similarity_threshold : float
        Minimum similarity score (0.0 - 1.0) required to consider evidence relevant.
    """

    def __init__(
        self,
        vector_store: Optional[ChromaVectorStore] = None,
        embedder: Optional[EmbeddingService] = None,
        similarity_threshold: float = 0.25,
    ):
        self.vector_store = vector_store or ChromaVectorStore()
        self.embedder = embedder or EmbeddingService()
        self.similarity_threshold = similarity_threshold

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        dataset_filter: Optional[str] = None,
    ) -> RetrievalPackage:
        """
        Retrieve the top-k most semantically relevant reference passages.

        Parameters
        ----------
        query : str
            The question, claim, or prompt to search evidence for.
        top_k : int
            Number of evidence passages to return (default: 5).
        dataset_filter : str, optional
            Filter search to a specific dataset ('truthfulqa' or 'squad').

        Returns
        -------
        RetrievalPackage
            Structured evidence package containing ranked passages and provenance.
        """
        cleaned_query = query.strip() if query else ""
        if not cleaned_query:
            return RetrievalPackage(
                query=query,
                passages=[],
                total_retrieved=0,
                status="insufficient_evidence",
            )

        # 1. Embed the query using all-MiniLM-L6-v2
        query_embedding = self.embedder.embed_text(cleaned_query)

        # 2. Build metadata filter if specified
        where_filter = None
        if dataset_filter:
            where_filter = {"dataset_name": dataset_filter}

        # 3. Query ChromaDB
        raw_results = self.vector_store.query(
            query_embedding=query_embedding,
            n_results=top_k,
            where=where_filter,
        )

        passages: List[RetrievedEvidence] = []
        ids_list = raw_results.get("ids", [[]])[0]
        docs_list = raw_results.get("documents", [[]])[0]
        meta_list = raw_results.get("metadatas", [[]])[0]
        dist_list = raw_results.get("distances", [[]])[0]

        for chunk_id, doc, meta, dist in zip(ids_list, docs_list, meta_list, dist_list):
            # ChromaDB cosine distance: 0 = identical, 1 = orthogonal, 2 = opposite
            # Convert to similarity score: max(0.0, 1.0 - distance)
            similarity = max(0.0, min(1.0, 1.0 - float(dist)))

            # Filter out chunks below relevance threshold
            if similarity < self.similarity_threshold:
                continue

            # Extract provenance
            dataset_name = meta.get("dataset_name", "unknown")
            dataset_split = meta.get("dataset_split", "unknown")
            source_record_id = str(meta.get("source_record_id", ""))
            title = meta.get("title")
            category = meta.get("category")
            chunk_index = int(meta.get("chunk_index", 0))
            content_type = meta.get("content_type", "context_chunk")

            passages.append(
                RetrievedEvidence(
                    chunk_id=chunk_id,
                    text=doc,
                    similarity_score=round(similarity, 4),
                    dataset_name=dataset_name,
                    dataset_split=dataset_split,
                    source_record_id=source_record_id,
                    title=title,
                    category=category,
                    chunk_index=chunk_index,
                    content_type=content_type,
                    extra_metadata={k: v for k, v in meta.items() if k not in {
                        "dataset_name", "dataset_split", "source_record_id",
                        "title", "category", "chunk_index", "content_type"
                    }},
                )
            )

        # Sort by similarity descending
        passages.sort(key=lambda p: p.similarity_score, reverse=True)

        status = "sufficient" if passages else "insufficient_evidence"

        return RetrievalPackage(
            query=cleaned_query,
            passages=passages,
            total_retrieved=len(passages),
            status=status,
        )
