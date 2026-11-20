"""
ingestion.py — Data ingestion and indexing pipeline for TruthfulQA and SQuAD.

Loads public benchmark datasets from Hugging Face:
  1. TruthfulQA (generation config, validation split)
  2. SQuAD v1.1 (validation split)

Cleans text, chunks context passages (~500 tokens, 75 overlap), generates
dense embeddings with all-MiniLM-L6-v2, and indexes chunks with full provenance
into ChromaDB.
"""

import time
import logging
from typing import Dict, List, Optional, Any, Tuple
from datasets import load_dataset

from app.rag.chunker import DocumentChunker
from app.rag.embedder import EmbeddingService
from app.rag.vector_store import ChromaVectorStore

logger = logging.getLogger("verity.rag.ingestion")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class BenchmarkIngestionPipeline:
    """
    Ingestion pipeline that loads, cleans, chunks, embeds, and stores
    TruthfulQA and SQuAD benchmark evidence into ChromaDB.
    """

    def __init__(
        self,
        vector_store: Optional[ChromaVectorStore] = None,
        embedder: Optional[EmbeddingService] = None,
        chunker: Optional[DocumentChunker] = None,
    ):
        self.vector_store = vector_store or ChromaVectorStore()
        self.embedder = embedder or EmbeddingService()
        self.chunker = chunker or DocumentChunker(target_tokens=500, overlap_tokens=75)

    def ingest_truthfulqa(
        self,
        limit: Optional[int] = None,
        batch_size: int = 64,
    ) -> Dict[str, Any]:
        """
        Load, preprocess, and index TruthfulQA validation split.

        Parameters
        ----------
        limit : int, optional
            Maximum records to ingest (default: all 817 records).
        batch_size : int
            Embedding & upsert batch size.

        Returns
        -------
        Dict[str, Any]
            Ingestion statistics.
        """
        logger.info("Loading TruthfulQA ('generation', split='validation')...")
        start_time = time.time()

        try:
            ds = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
        except Exception as e:
            logger.warning(f"Could not load truthfulqa/truthful_qa, retrying without namespace: {e}")
            ds = load_dataset("truthful_qa", "generation", split="validation")

        total_records = len(ds)
        records_to_process = total_records if limit is None or limit <= 0 else min(limit, total_records)
        logger.info(f"Processing {records_to_process} of {total_records} TruthfulQA records...")

        documents: List[str] = []
        ids: List[str] = []
        metadatas: List[Dict[str, Any]] = []

        for idx in range(records_to_process):
            row = ds[idx]
            question = (row.get("question") or "").strip()
            best_answer = (row.get("best_answer") or "").strip()
            category = (row.get("category") or "").strip()
            source = (row.get("source") or "").strip()
            correct_answers = row.get("correct_answers") or []

            if not question or not best_answer:
                continue

            # Format rich reference document content
            doc_lines = [
                f"Question: {question}",
                f"Verified Truth: {best_answer}",
            ]
            if category:
                doc_lines.append(f"Topic: {category}")
            if source:
                doc_lines.append(f"Source Reference: {source}")
            if len(correct_answers) > 1:
                alt_answers = "; ".join([a.strip() for a in correct_answers[:3] if a.strip() != best_answer])
                if alt_answers:
                    doc_lines.append(f"Alternative Correct Answers: {alt_answers}")

            doc_text = "\n".join(doc_lines)
            chunk_id = f"truthfulqa_val_{idx:05d}"

            documents.append(doc_text)
            ids.append(chunk_id)
            metadatas.append({
                "dataset_name": "truthfulqa",
                "dataset_split": "validation",
                "source_record_id": f"tqa_{idx}",
                "title": f"TruthfulQA: {category or 'General'}",
                "category": category,
                "chunk_index": 0,
                "total_chunks": 1,
                "content_type": "qa_reference",
                "question": question[:250],
                "best_answer": best_answer[:250],
            })

        # Embed and upsert in batches
        indexed_count = self._batch_upsert(ids, documents, metadatas, batch_size=batch_size)
        elapsed = round(time.time() - start_time, 2)

        stats = {
            "dataset": "truthfulqa",
            "split": "validation",
            "records_read": records_to_process,
            "chunks_indexed": indexed_count,
            "duration_seconds": elapsed,
        }
        logger.info(f"TruthfulQA ingestion complete: {stats}")
        return stats

    def ingest_squad(
        self,
        limit: Optional[int] = None,
        batch_size: int = 64,
    ) -> Dict[str, Any]:
        """
        Load, deduplicate, chunk, and index SQuAD v1.1 validation split.

        Parameters
        ----------
        limit : int, optional
            Maximum records to process (default: all ~10,570 examples).
        batch_size : int
            Embedding & upsert batch size.

        Returns
        -------
        Dict[str, Any]
            Ingestion statistics.
        """
        logger.info("Loading SQuAD v1.1 (split='validation')...")
        start_time = time.time()

        try:
            ds = load_dataset("rajpurkar/squad", split="validation")
        except Exception as e:
            logger.warning(f"Could not load rajpurkar/squad, retrying as squad: {e}")
            ds = load_dataset("squad", split="validation")

        total_records = len(ds)
        records_to_process = total_records if limit is None or limit <= 0 else min(limit, total_records)
        logger.info(f"Processing {records_to_process} of {total_records} SQuAD records...")

        # Deduplicate contexts by article title & context content
        unique_contexts: Dict[str, Dict[str, Any]] = {}
        for idx in range(records_to_process):
            row = ds[idx]
            context = (row.get("context") or "").strip()
            title = (row.get("title") or "").strip()
            source_id = str(row.get("id") or f"squad_{idx}")

            if not context:
                continue

            # Key by context hash/slice to avoid storing identical passages multiple times
            context_key = f"{title}::{context[:120]}"
            if context_key not in unique_contexts:
                unique_contexts[context_key] = {
                    "context": context,
                    "title": title,
                    "source_id": source_id,
                }

        logger.info(f"Extracted {len(unique_contexts)} unique context passages from SQuAD records.")

        documents: List[str] = []
        ids: List[str] = []
        metadatas: List[Dict[str, Any]] = []

        # Chunk each unique context passage
        for c_idx, ctx_data in enumerate(unique_contexts.values()):
            context_text = ctx_data["context"]
            title = ctx_data["title"]
            source_id = ctx_data["source_id"]

            chunks = self.chunker.chunk_text(context_text)
            for chunk in chunks:
                chunk_id = f"squad_val_{c_idx:05d}_c{chunk.chunk_index:02d}"

                documents.append(chunk.text)
                ids.append(chunk_id)
                metadatas.append({
                    "dataset_name": "squad",
                    "dataset_split": "validation",
                    "source_record_id": source_id,
                    "title": title.replace("_", " "),
                    "category": "Wikipedia Article Context",
                    "chunk_index": chunk.chunk_index,
                    "total_chunks": chunk.total_chunks,
                    "content_type": "context_chunk",
                    "token_count": chunk.token_count,
                })

        indexed_count = self._batch_upsert(ids, documents, metadatas, batch_size=batch_size)
        elapsed = round(time.time() - start_time, 2)

        stats = {
            "dataset": "squad",
            "split": "validation",
            "records_read": records_to_process,
            "unique_contexts": len(unique_contexts),
            "chunks_indexed": indexed_count,
            "duration_seconds": elapsed,
        }
        logger.info(f"SQuAD ingestion complete: {stats}")
        return stats

    def ingest_curated_facts(self) -> Dict[str, Any]:
        """
        Ingests curated, verified foundational knowledge into ChromaDB
        to ground world geography, national capitals, and general facts.
        """
        logger.info("Ingesting curated verified reference facts...")
        start_time = time.time()

        curated_entries = [
            {
                "id": "ref_capitals_india",
                "title": "National Capitals: India",
                "category": "Geography & National Capitals",
                "text": (
                    "New Delhi is the official capital of India and the seat of all three branches of the "
                    "Government of India. The capital was shifted from Calcutta to Delhi in 1911, and the "
                    "new capital city was formally inaugurated in 1931. Kerala is a state located on the "
                    "southwestern Malabar Coast of India, with Thiruvananthapuram as its state capital. "
                    "Kerala is an Indian state, not the national capital."
                ),
            },
            {
                "id": "ref_capitals_france",
                "title": "National Capitals: France",
                "category": "Geography & National Capitals",
                "text": (
                    "Paris is the official capital and most populous city of France. It is situated on the "
                    "Seine River in northern central France. Paris is the political, administrative, and cultural "
                    "center of the French Republic."
                ),
            },
            {
                "id": "ref_capitals_japan",
                "title": "National Capitals: Japan",
                "category": "Geography & National Capitals",
                "text": (
                    "Tokyo is the official capital and largest city of Japan. Located at the head of Tokyo Bay, "
                    "it is the seat of the Government of Japan and the Imperial Palace."
                ),
            },
            {
                "id": "ref_capitals_usa",
                "title": "National Capitals: United States",
                "category": "Geography & National Capitals",
                "text": (
                    "Washington, D.C. is the capital city and federal district of the United States. Located on the "
                    "Potomac River, it is the seat of the US federal government, including the White House and the Capitol."
                ),
            },
            {
                "id": "ref_capitals_uk",
                "title": "National Capitals: United Kingdom",
                "category": "Geography & National Capitals",
                "text": (
                    "London is the capital and largest city of England and the United Kingdom. Standing on the River "
                    "Thames in south-east England, London has been a major settlement for two millennia."
                ),
            },
            {
                "id": "ref_capitals_australia",
                "title": "National Capitals: Australia",
                "category": "Geography & National Capitals",
                "text": (
                    "Canberra is the federal capital city of Australia. It is Australia's largest inland city, "
                    "located in the Australian Capital Territory between Sydney and Melbourne."
                ),
            },
            {
                "id": "ref_capitals_germany",
                "title": "National Capitals: Germany",
                "category": "Geography & National Capitals",
                "text": (
                    "Berlin is the capital and largest city of Germany by both area and population. Berlin is "
                    "the seat of the federal government of Germany."
                ),
            },
            {
                "id": "ref_capitals_canada",
                "title": "National Capitals: Canada",
                "category": "Geography & National Capitals",
                "text": (
                    "Ottawa is the capital city of Canada. It is located in the southern portion of the province "
                    "of Ontario, at the confluence of the Ottawa River and the Rideau River."
                ),
            },
            {
                "id": "ref_nobel_literature_1924",
                "title": "Nobel Prize in Literature: 1924 & 1954",
                "category": "Nobel Prize History",
                "text": (
                    "The Nobel Prize in Literature in 1924 was awarded to Polish author Władysław Reymont "
                    "for his great national epic, The Peasants. Ernest Hemingway was awarded the Nobel Prize "
                    "in Literature in 1954 for his mastery of the art of narrative, most recently demonstrated in "
                    "The Old Man and the Sea. Hemingway did not win the Nobel Prize in 1924, and his 1924 collection "
                    "In Our Time was not a Nobel Prize-winning work."
                ),
            },
        ]

        documents: List[str] = []
        ids: List[str] = []
        metadatas: List[Dict[str, Any]] = []

        for item in curated_entries:
            chunks = self.chunker.chunk_text(item["text"])
            for chunk in chunks:
                chunk_id = f"{item['id']}_c{chunk.chunk_index:02d}"
                documents.append(chunk.text)
                ids.append(chunk_id)
                metadatas.append({
                    "dataset_name": "verified_reference",
                    "dataset_split": "core",
                    "source_record_id": item["id"],
                    "title": item["title"],
                    "category": item["category"],
                    "chunk_index": chunk.chunk_index,
                    "total_chunks": chunk.total_chunks,
                    "content_type": "curated_ground_truth",
                    "token_count": chunk.token_count,
                })

        indexed_count = self._batch_upsert(ids, documents, metadatas, batch_size=32)
        elapsed = round(time.time() - start_time, 2)

        stats = {
            "dataset": "verified_reference",
            "records_read": len(curated_entries),
            "chunks_indexed": indexed_count,
            "duration_seconds": elapsed,
        }
        logger.info(f"Curated reference facts ingestion complete: {stats}")
        return stats

    def _batch_upsert(
        self,
        ids: List[str],
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        batch_size: int = 64,
    ) -> int:
        """Batch embed and upsert chunks into ChromaDB."""
        total = len(ids)
        total_upserted = 0

        for i in range(0, total, batch_size):
            batch_ids = ids[i : i + batch_size]
            batch_docs = documents[i : i + batch_size]
            batch_metas = metadatas[i : i + batch_size]

            # Generate batch embeddings with sentence-transformers
            embeddings = self.embedder.embed_documents(batch_docs, batch_size=len(batch_docs))

            # Upsert into ChromaDB
            self.vector_store.add_chunks(
                ids=batch_ids,
                documents=batch_docs,
                embeddings=embeddings,
                metadatas=batch_metas,
            )
            total_upserted += len(batch_ids)

        return total_upserted
