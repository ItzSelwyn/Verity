"""
orchestrator.py — Evaluation Orchestrator Service for Verity.
Coordinates data flow from user submissions through the Reference Knowledge Base (RAG)
and dispatches to specialized Judge Agents.
"""

import logging
from datetime import datetime, timezone
from typing import Optional
from sqlalchemy.orm import Session

from app.schemas.judge import (
    RelevanceJudgeInput,
    RelevanceJudgeOutput,
    RetrievedEvidenceItem,
    OrchestratedEvaluationResult,
    EvaluationRunStatus,
)
from app.agents.relevance_judge import RelevanceJudgeAgent
from app.services.evaluation_service import get_evaluation
from app.rag.retriever import get_semantic_retriever

logger = logging.getLogger(__name__)


class EvaluationOrchestrator:
    """
    Central coordinator for the multi-agent evaluation workflow.
    Manages context preparation, RAG evidence retrieval, and judge agent execution.
    """

    def __init__(self, relevance_judge: Optional[RelevanceJudgeAgent] = None):
        self.relevance_judge = relevance_judge or RelevanceJudgeAgent()

    def evaluate_relevance(
        self,
        question: str,
        ai_response: str,
        context: Optional[str] = None,
    ) -> RelevanceJudgeOutput:
        """
        Directly evaluates relevance for an ad-hoc question and response pair.
        """
        judge_input = RelevanceJudgeInput(
            question=question,
            ai_response=ai_response,
            context=context,
        )
        return self.relevance_judge.evaluate(judge_input)

    def evaluate_submission(
        self,
        db: Session,
        submission_id: int,
    ) -> OrchestratedEvaluationResult:
        """
        Runs the evaluation pipeline for a stored evaluation submission.
        Fetches the submission from the database, runs the active judge agents,
        updates the database status, and returns the structured evaluation result.
        """
        submission = get_evaluation(db, submission_id)
        if not submission:
            raise ValueError(f"Submission with ID {submission_id} not found.")

        # Update status to running
        submission.status = "running"
        db.commit()
        db.refresh(submission)

        try:
            # 1. Prepare evaluation context: combine user source document with RAG retrieved evidence
            combined_context_parts = []
            if submission.source_document:
                combined_context_parts.append(f"User Source Document:\n{submission.source_document}")
            if submission.reference_answer:
                combined_context_parts.append(f"Reference Answer:\n{submission.reference_answer}")

            # Also attempt semantic retrieval for factual context from ChromaDB
            retrieved_items = []
            try:
                retriever = get_semantic_retriever()
                retrieval_pkg = retriever.retrieve(query=submission.question, top_k=3)
                if retrieval_pkg.passages:
                    for p in retrieval_pkg.passages:
                        retrieved_items.append(
                            RetrievedEvidenceItem(
                                chunk_id=p.chunk_id,
                                text=p.text,
                                similarity_score=p.similarity_score,
                                dataset_name=p.dataset_name,
                                source_record_id=p.source_record_id,
                                title=p.title,
                                category=p.category,
                            )
                        )
                    rag_evidence = "\n\n".join([p.text for p in retrieval_pkg.passages])
                    combined_context_parts.append(f"Retrieved Knowledge Base Evidence:\n{rag_evidence}")
            except Exception as rag_err:
                logger.warning(f"RAG retrieval skipped during orchestration: {rag_err}")

            full_context = "\n\n---\n\n".join(combined_context_parts) if combined_context_parts else None

            # 2. Run Relevance Judge Agent (M2.1)
            relevance_output = self.evaluate_relevance(
                question=submission.question,
                ai_response=submission.ai_response,
                context=full_context,
            )

            result = OrchestratedEvaluationResult(
                submission_id=str(submission.id),
                status=EvaluationRunStatus.COMPLETED,
                relevance=relevance_output,
                retrieved_evidence=retrieved_items,
                overall_score=relevance_output.score,
                verdict=f"Relevance: {relevance_output.classification.value} ({relevance_output.score}/10)",
                created_at=datetime.now(timezone.utc).isoformat(),
            )

            # Update submission status and persist result JSON in database
            submission.status = "completed"
            submission.evaluation_result = result.model_dump_json()
            db.commit()

            return result

        except Exception as err:
            logger.error(f"Error during evaluation orchestration: {err}", exc_info=True)
            submission.status = "failed"
            db.commit()
            raise


# Singleton instance
_orchestrator: Optional[EvaluationOrchestrator] = None


def get_evaluation_orchestrator() -> EvaluationOrchestrator:
    """Provides a singleton instance of the EvaluationOrchestrator."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = EvaluationOrchestrator()
    return _orchestrator
