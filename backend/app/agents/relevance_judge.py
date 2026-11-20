"""
relevance_judge.py — Relevance Judge Agent for Verity.
Evaluates whether an AI-generated response directly and appropriately addresses
the submitted question on a clear, calibrated 0.0 to 10.0 scale.
"""

import logging
import re
from typing import List, Tuple

from app.schemas.judge import (
    RelevanceJudgeInput,
    RelevanceJudgeOutput,
    RelevanceClassification,
    ConfidenceLevel,
)
from app.agents.llm_client import BaseLLMClient, get_llm_client
from app.rag.embedder import get_embedding_service

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are Verity's Relevance Judge Agent.
Your SOLE responsibility is to evaluate whether an AI-generated response directly, appropriately, and thoroughly addresses the submitted question.

IMPORTANT EVALUATION PRINCIPLES:
1. Relevance ONLY: Do NOT evaluate factual correctness, truthfulness, or hallucination. An answer can be completely factually wrong (e.g., claiming the Earth is flat when asked for its shape) but still be 100% relevant because it directly answers the question asked.
2. Directness: Does the response answer the specific prompt, or does it dance around the question?
3. Constraints: Did the user ask for specific aspects, constraints, or formats? Are they addressed?
4. Extraneous Drift: Does the response digress into unrelated topics?

SCORING RUBRIC (0.0 to 10.0):
- 9.0 - 10.0 (fully_relevant): The response directly, clearly, and completely addresses what was asked. No distracting digressions.
- 7.0 - 8.9 (mostly_relevant): Directly answers the core question, but includes minor tangential commentary or slight verbosity.
- 4.0 - 6.9 (partially_relevant): Addresses only a sub-part of a compound question, provides an overly high-level/vague reply, or heavily drifts.
- 1.0 - 3.9 (off_topic): Mentions keywords from the prompt or shares general domain, but fails to answer the question posed.
- 0.0 - 0.9 (unrelated): Completely unrelated subject, non-responsive text, or incoherent generation.

OUTPUT FORMAT:
You MUST respond with a valid JSON object matching this schema:
{
  "score": <float between 0.0 and 10.0>,
  "classification": "<fully_relevant | mostly_relevant | partially_relevant | off_topic | unrelated>",
  "reasoning": "<concise explanation detailing why this score and classification were assigned>",
  "key_points_addressed": ["<point 1>", "<point 2>"],
  "key_points_missed": ["<point missed 1>"],
  "confidence": "<high | medium | low>"
}
"""


def _classify_score(score: float) -> RelevanceClassification:
    """Maps a 0.0-10.0 score to its formal RelevanceClassification category."""
    if score >= 9.0:
        return RelevanceClassification.FULLY_RELEVANT
    elif score >= 7.0:
        return RelevanceClassification.MOSTLY_RELEVANT
    elif score >= 4.0:
        return RelevanceClassification.PARTIALLY_RELEVANT
    elif score >= 1.0:
        return RelevanceClassification.OFF_TOPIC
    else:
        return RelevanceClassification.UNRELATED


class RelevanceJudgeAgent:
    """
    Relevance Judge Agent.
    Evaluates prompt-response alignment using an active LLM client or local semantic analysis.
    """

    def __init__(self, llm_client: BaseLLMClient = None):
        self.llm_client = llm_client

    def evaluate(self, judge_input: RelevanceJudgeInput) -> RelevanceJudgeOutput:
        """
        Main entry point for evaluating relevance.
        Dispatches to LLM judge if available, otherwise executes local semantic evaluation.
        """
        client = self.llm_client or get_llm_client()

        if client is not None:
            try:
                return self._evaluate_with_llm(client, judge_input)
            except Exception as e:
                logger.warning(f"LLM Judge call failed ({e}). Falling back to local semantic evaluator.")

        return self._evaluate_deterministic(judge_input)

    def _evaluate_with_llm(self, client: BaseLLMClient, judge_input: RelevanceJudgeInput) -> RelevanceJudgeOutput:
        """Executes LLM-as-a-judge evaluation with strict JSON schema parsing."""
        user_prompt = f"""EVALUATE THIS SUBMISSION:

[QUESTION / PROMPT]:
{judge_input.question}

[AI-GENERATED RESPONSE]:
{judge_input.ai_response}
"""
        if judge_input.context:
            user_prompt += f"\n[OPTIONAL CONTEXT]:\n{judge_input.context}\n"

        result_dict = client.generate_json(SYSTEM_PROMPT, user_prompt, temperature=0.0)

        raw_score = float(result_dict.get("score", 5.0))
        clamped_score = max(0.0, min(10.0, raw_score))
        classification = _classify_score(clamped_score)

        return RelevanceJudgeOutput(
            agent="relevance_judge",
            score=clamped_score,
            classification=classification,
            reasoning=result_dict.get("reasoning", "Evaluated via LLM judge."),
            key_points_addressed=result_dict.get("key_points_addressed", []),
            key_points_missed=result_dict.get("key_points_missed", []),
            confidence=ConfidenceLevel(result_dict.get("confidence", "high").lower()),
            evaluator_model=getattr(client, "model", "llm-judge"),
        )

    def _evaluate_deterministic(self, judge_input: RelevanceJudgeInput) -> RelevanceJudgeOutput:
        """
        Deterministic, local fallback evaluator using sentence embeddings,
        question intent parsing, and lexical keyword coverage.
        Runs entirely offline without requiring third-party API keys.
        """
        question = judge_input.question.strip()
        response = judge_input.ai_response.strip()

        embedder = get_embedding_service()
        
        # 1. Cosine similarity between question and candidate response
        q_emb = embedder.embed_text(question)
        r_emb = embedder.embed_text(response)
        
        # Vectors are already L2 normalized, so dot product = cosine similarity
        cosine_sim = float(sum(q * r for q, r in zip(q_emb, r_emb)))
        
        # 2. Token / keyword coverage analysis
        # Extract meaningful word tokens (length >= 4) from question
        q_words = set(re.findall(r"\b[a-zA-Z]{4,}\b", question.lower()))
        r_words = set(re.findall(r"\b[a-zA-Z]{4,}\b", response.lower()))
        
        overlap_words = q_words.intersection(r_words)
        coverage_ratio = len(overlap_words) / max(len(q_words), 1)

        # 3. Non-responsive / refusal signals
        refusal_patterns = [
            r"\bi (don't|do not) know\b",
            r"\bi cannot (answer|help)\b",
            r"\bas an ai language model\b",
            r"\bno information provided\b",
        ]
        is_refusal = any(re.search(pat, response.lower()) for pat in refusal_patterns)

        # 4. Score calibration based on semantic similarity and lexical overlap
        # Base score from cosine similarity scaled to 0-10
        # Cosine similarity for all-MiniLM-L6-v2 generally ranges from 0.0 to 0.85
        if is_refusal and len(response.split()) < 20:
            score = 1.5
            classification = RelevanceClassification.OFF_TOPIC
            reasoning = "Response expresses refusal or inability to answer the question."
            addressed = []
            missed = list(q_words)[:3]
        elif cosine_sim >= 0.65:
            # High semantic alignment
            score = min(10.0, 8.5 + (cosine_sim - 0.65) * 7.5 + (coverage_ratio * 0.5))
            classification = RelevanceClassification.FULLY_RELEVANT if score >= 9.0 else RelevanceClassification.MOSTLY_RELEVANT
            reasoning = f"The response directly and closely addresses the prompt with high semantic alignment (similarity: {cosine_sim:.2f})."
            addressed = list(overlap_words)[:4] if overlap_words else ["Primary subject addressed"]
            missed = []
        elif cosine_sim >= 0.45:
            # Moderate alignment
            score = 5.0 + (cosine_sim - 0.45) * 15.0
            score = max(4.0, min(8.4, score))
            classification = RelevanceClassification.MOSTLY_RELEVANT if score >= 7.0 else RelevanceClassification.PARTIALLY_RELEVANT
            reasoning = f"The response is related to the question domain with moderate topical overlap (similarity: {cosine_sim:.2f})."
            addressed = list(overlap_words)[:3]
            missed = [w for w in q_words if w not in r_words][:2]
        elif cosine_sim >= 0.25:
            # Low alignment / off-topic
            score = 1.5 + (cosine_sim - 0.25) * 12.0
            score = max(1.0, min(3.9, score))
            classification = RelevanceClassification.OFF_TOPIC
            reasoning = f"The response touches on peripheral keywords but fails to directly answer the question (similarity: {cosine_sim:.2f})."
            addressed = list(overlap_words)[:2]
            missed = [w for w in q_words if w not in r_words][:3]
        else:
            # Unrelated
            score = max(0.0, min(0.9, cosine_sim * 3.0))
            classification = RelevanceClassification.UNRELATED
            reasoning = f"The response does not address the question and appears completely unrelated (similarity: {cosine_sim:.2f})."
            addressed = []
            missed = list(q_words)[:4]

        return RelevanceJudgeOutput(
            agent="relevance_judge",
            score=round(score, 2),
            classification=classification,
            reasoning=reasoning,
            key_points_addressed=addressed,
            key_points_missed=missed,
            confidence=ConfidenceLevel.HIGH if (score >= 9.0 or score <= 2.0) else ConfidenceLevel.MEDIUM,
            evaluator_model="deterministic-semantic-v1",
        )
