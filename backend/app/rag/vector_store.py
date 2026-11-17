"""
vector_store.py — ChromaDB integration for Verity Reference Knowledge Base.

Provides persistent storage, cosine-distance indexing, and fast nearest-neighbor
retrieval of chunk embeddings and their provenance metadata.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Any
import chromadb
from chromadb.config import Settings

from app.core.config import VECTOR_DB_DIR, CHROMA_COLLECTION_NAME


class ChromaVectorStore:
    """
    Manages persistent ChromaDB indexing and vector search.

    Parameters
    ----------
    persist_directory : str, optional
        Disk path where ChromaDB saves collections (default: configured VECTOR_DB_DIR).
    collection_name : str, optional
        Name of the Chroma collection (default: verity_knowledge_base).
    """

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        collection_name: Optional[str] = None,
    ):
        self.persist_directory = persist_directory or VECTOR_DB_DIR
        self.collection_name = collection_name or CHROMA_COLLECTION_NAME

        # Ensure directory exists on disk
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)

        # Initialize persistent Chroma client
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(anonymized_telemetry=False),
        )

        # Use cosine distance space for normalized embeddings
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(
        self,
        ids: List[str],
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
    ) -> int:
        """
        Upsert a batch of text chunks with precomputed embeddings and metadata.

        Parameters
        ----------
        ids : List[str]
            Unique identifiers for each chunk.
        documents : List[str]
            Raw text content of each chunk.
        embeddings : List[List[float]]
            384-dimensional dense vectors.
        metadatas : List[Dict[str, Any]]
            Provenance metadata (dataset_name, source_record_id, title, etc.).

        Returns
        -------
        int
            Number of chunks successfully added.
        """
        if not ids:
            return 0

        # Sanitize metadata (ChromaDB requires primitives: str, int, float, bool)
        sanitized_metadatas = []
        for meta in metadatas:
            clean_meta = {}
            for k, v in meta.items():
                if v is None:
                    continue
                if isinstance(v, (str, int, float, bool)):
                    clean_meta[k] = v
                else:
                    clean_meta[k] = str(v)
            sanitized_metadatas.append(clean_meta)

        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=sanitized_metadatas,
        )
        return len(ids)

    def query(
        self,
        query_embedding: List[float],
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Perform nearest-neighbor search for a query embedding vector.

        Parameters
        ----------
        query_embedding : List[float]
            The 384-dimensional query vector.
        n_results : int
            Number of top matching chunks to retrieve (default: 5).
        where : Dict[str, Any], optional
            Chroma metadata filter (e.g. {"dataset_name": "squad"}).

        Returns
        -------
        Dict[str, Any]
            Dictionary containing 'ids', 'documents', 'metadatas', and 'distances'.
        """
        total = self.count()
        if total == 0:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

        k = min(n_results, total)
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

    def count(self) -> int:
        """Return total number of indexed chunks in the collection."""
        return self.collection.count()

    def get_chunk(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a single chunk by ID."""
        res = self.collection.get(ids=[chunk_id], include=["documents", "metadatas"])
        if res and res["ids"]:
            return {
                "id": res["ids"][0],
                "document": res["documents"][0] if res["documents"] else None,
                "metadata": res["metadatas"][0] if res["metadatas"] else {},
            }
        return None

    def reset(self) -> None:
        """Clear all records from the collection."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
