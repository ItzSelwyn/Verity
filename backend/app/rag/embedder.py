"""
embedder.py — Embedding generation module for Verity Knowledge Base.

Uses sentence-transformers/all-MiniLM-L6-v2 to produce 384-dimensional
dense vector representations of text passages and evaluation queries.
"""

from typing import List, Union
from sentence_transformers import SentenceTransformer
from app.core.config import EMBEDDING_MODEL_NAME

_MODEL_INSTANCE: Union[SentenceTransformer, None] = None
_SERVICE_INSTANCE: Union["EmbeddingService", None] = None


def get_embedding_model(model_name: str = EMBEDDING_MODEL_NAME) -> SentenceTransformer:
    """
    Lazy singleton loader for the SentenceTransformer embedding model.
    Avoids loading model weights during module import.
    """
    global _MODEL_INSTANCE
    if _MODEL_INSTANCE is None:
        _MODEL_INSTANCE = SentenceTransformer(model_name)
    return _MODEL_INSTANCE


def get_embedding_service() -> "EmbeddingService":
    """
    Lazy singleton provider for the EmbeddingService.
    """
    global _SERVICE_INSTANCE
    if _SERVICE_INSTANCE is None:
        _SERVICE_INSTANCE = EmbeddingService()
    return _SERVICE_INSTANCE


class EmbeddingService:
    """
    Service for generating vector embeddings from text strings or batches.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.model_name = model_name

    @property
    def model(self) -> SentenceTransformer:
        return get_embedding_model(self.model_name)

    def embed_text(self, text: str) -> List[float]:
        """
        Generate a normalized 384-dim embedding vector for a single string.

        Parameters
        ----------
        text : str
            The input query or text passage.

        Returns
        -------
        List[float]
            Embedding vector as a list of floats.
        """
        cleaned = text.strip() if text else ""
        if not cleaned:
            return [0.0] * 384
        embedding = self.model.encode(cleaned, normalize_embeddings=True)
        return embedding.tolist()

    def embed_documents(self, texts: List[str], batch_size: int = 64) -> List[List[float]]:
        """
        Generate normalized embedding vectors for a batch of documents.

        Parameters
        ----------
        texts : List[str]
            List of passage texts to embed.
        batch_size : int
            Batch size for inference.

        Returns
        -------
        List[List[float]]
            List of 384-dim embedding vectors.
        """
        if not texts:
            return []
        cleaned_texts = [t.strip() if t else "" for t in texts]
        embeddings = self.model.encode(
            cleaned_texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return embeddings.tolist()
