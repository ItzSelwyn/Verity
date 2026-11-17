"""
ingest_knowledge_base.py — CLI tool to ingest TruthfulQA and SQuAD into ChromaDB.

Usage:
    python -m scripts.ingest_knowledge_base --dataset all --limit 200
    python -m scripts.ingest_knowledge_base --dataset truthfulqa
    python -m scripts.ingest_knowledge_base --dataset squad --limit 500
"""

import sys
import argparse
from pathlib import Path

# Ensure backend root is on sys.path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.rag.ingestion import BenchmarkIngestionPipeline
from app.rag.vector_store import ChromaVectorStore
from app.rag.retriever import SemanticRetriever


def main():
    parser = argparse.ArgumentParser(
        description="Ingest TruthfulQA and SQuAD benchmark datasets into ChromaDB."
    )
    parser.add_argument(
        "--dataset",
        choices=["all", "truthfulqa", "squad"],
        default="all",
        help="Which dataset to ingest (default: all)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum records to process per dataset (default: all records in split)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset existing vector collection before ingesting",
    )
    parser.add_argument(
        "--test-query",
        type=str,
        default="What happens if you crack your knuckles?",
        help="Sample question to run retrieval test after ingestion",
    )

    args = parser.parse_args()

    pipeline = BenchmarkIngestionPipeline()

    if args.reset:
        print("[*] Resetting ChromaDB vector collection...")
        pipeline.vector_store.reset()

    print(f"[*] Starting ingestion pipeline for dataset='{args.dataset}', limit={args.limit}...")

    stats = []
    if args.dataset in ["all", "truthfulqa"]:
        print("\n--- 1. Ingesting TruthfulQA (validation split) ---")
        tqa_stats = pipeline.ingest_truthfulqa(limit=args.limit)
        stats.append(tqa_stats)
        print(f"    ✓ TruthfulQA indexed: {tqa_stats['chunks_indexed']} chunks in {tqa_stats['duration_seconds']}s")

    if args.dataset in ["all", "squad"]:
        print("\n--- 2. Ingesting SQuAD v1.1 (validation split) ---")
        squad_stats = pipeline.ingest_squad(limit=args.limit)
        stats.append(squad_stats)
        print(f"    ✓ SQuAD indexed: {squad_stats['chunks_indexed']} chunks in {squad_stats['duration_seconds']}s")

    total_chunks = pipeline.vector_store.count()
    print(f"\n[✓] Knowledge Base Ingestion Finished! Total Chunks in ChromaDB: {total_chunks}")

    # Validation retrieval test
    if args.test_query and total_chunks > 0:
        print(f"\n[*] Running Semantic Retrieval Validation on query: '{args.test_query}'...")
        retriever = SemanticRetriever(vector_store=pipeline.vector_store, embedder=pipeline.embedder)
        result = retriever.retrieve(args.test_query, top_k=3)

        print(f"    Status: {result.status} | Passages retrieved: {result.total_retrieved}")
        for i, p in enumerate(result.passages, 1):
            print(f"\n    [Result #{i}] Similarity: {p.similarity_score:.4f} | Dataset: {p.dataset_name} ({p.title})")
            print(f"    Source ID: {p.source_record_id} | Chunk: #{p.chunk_index}")
            preview = p.text.replace("\n", " ")[:140]
            print(f"    Passage: {preview}...")


if __name__ == "__main__":
    main()
