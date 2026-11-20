"""
judge.py — FastAPI routes for AI Response Evaluation and Judge Agents.
Exposes endpoints for ad-hoc judge evaluations and orchestrated submission runs.
"""

import logging
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.schemas.judge import (
    RelevanceJudgeInput,
    RelevanceJudgeOutput,
    AccuracyJudgeInput,
    AccuracyJudgeOutput,
    HallucinationJudgeInput,
    HallucinationJudgeOutput,
    OrchestratedEvaluationResult,
)
from app.services.orchestrator import get_evaluation_orchestrator, EvaluationOrchestrator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/judge", tags=["Judge Agents"])


@router.post(
    "/relevance",
    response_model=RelevanceJudgeOutput,
    status_code=status.HTTP_200_OK,
    summary="Evaluate Prompt-Response Relevance",
    description="Evaluates whether an AI-generated response directly and appropriately addresses the submitted question.",
)
def evaluate_relevance(
    input_data: RelevanceJudgeInput,
    orchestrator: EvaluationOrchestrator = Depends(get_evaluation_orchestrator),
) -> RelevanceJudgeOutput:
    """
    Directly evaluates the relevance of an ad-hoc question and AI response pair.
    Returns structured scoring (0-10), classification, reasoning, and key points analysis.
    """
    try:
        return orchestrator.evaluate_relevance(
            question=input_data.question,
            ai_response=input_data.ai_response,
            context=input_data.context,
        )
    except Exception as e:
        logger.error(f"Relevance evaluation failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Relevance evaluation encountered an unexpected error: {str(e)}",
        )


@router.post(
    "/accuracy",
    response_model=AccuracyJudgeOutput,
    status_code=status.HTTP_200_OK,
    summary="Evaluate Factual Accuracy",
    description="Evaluates whether an AI-generated response is factually supported or contradicted by reference evidence.",
)
def evaluate_accuracy(
    input_data: AccuracyJudgeInput,
    orchestrator: EvaluationOrchestrator = Depends(get_evaluation_orchestrator),
) -> AccuracyJudgeOutput:
    """
    Directly evaluates the factual correctness of an ad-hoc AI response
    against reference answers, source text, and RAG evidence.
    """
    try:
        return orchestrator.evaluate_accuracy(
            question=input_data.question,
            ai_response=input_data.ai_response,
            reference_answer=input_data.reference_answer,
            source_document=input_data.source_document,
            retrieved_evidence=input_data.retrieved_evidence,
        )
    except Exception as e:
        logger.error(f"Accuracy evaluation failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Accuracy evaluation encountered an unexpected error: {str(e)}",
        )


@router.post(
    "/hallucination",
    response_model=HallucinationJudgeOutput,
    status_code=status.HTTP_200_OK,
    summary="Detect Hallucinations & Verify Claims",
    description="Deconstructs an AI response into atomic claims and cross-references each claim against reference evidence.",
)
def evaluate_hallucination(
    input_data: HallucinationJudgeInput,
    orchestrator: EvaluationOrchestrator = Depends(get_evaluation_orchestrator),
) -> HallucinationJudgeOutput:
    """
    Directly evaluates hallucination risk by verifying individual atomic claims
    against reference answers, source text, and RAG evidence.
    """
    try:
        return orchestrator.evaluate_hallucination(
            question=input_data.question,
            ai_response=input_data.ai_response,
            reference_answer=input_data.reference_answer,
            source_document=input_data.source_document,
            retrieved_evidence=input_data.retrieved_evidence,
        )
    except Exception as e:
        logger.error(f"Hallucination detection failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Hallucination detection encountered an unexpected error: {str(e)}",
        )


@router.post(
    "/submissions/{submission_id}/evaluate",
    response_model=OrchestratedEvaluationResult,
    status_code=status.HTTP_200_OK,
    summary="Orchestrate Evaluation for Stored Submission",
    description="Loads a stored submission by ID, retrieves reference evidence, and executes judge evaluation.",
)
def evaluate_stored_submission(
    submission_id: int,
    db: Session = Depends(get_db),
    orchestrator: EvaluationOrchestrator = Depends(get_evaluation_orchestrator),
) -> OrchestratedEvaluationResult:
    """
    Runs the multi-agent evaluation pipeline on a stored evaluation submission.
    """
    try:
        return orchestrator.evaluate_submission(db=db, submission_id=submission_id)
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(val_err),
        )
    except Exception as e:
        logger.error(f"Orchestration failed for submission #{submission_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Evaluation orchestration failed: {str(e)}",
        )
