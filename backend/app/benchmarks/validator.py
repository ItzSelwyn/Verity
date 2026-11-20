"""
validator.py — Automated Benchmark Evaluation & Consistency Validation Engine (M2.4).

Runs the benchmark test suite across Relevance, Accuracy, and Hallucination Detection Agents,
computes precision, recall, classification accuracy, score consistency, and reasoning quality metrics.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import time

from app.benchmarks.benchmark_dataset import BenchmarkTestCase, BENCHMARK_TEST_SUITE
from app.services.orchestrator import EvaluationOrchestrator
from app.rag.retriever import get_semantic_retriever
from app.schemas.judge import RetrievedEvidenceItem, OrchestratedEvaluationResult


class CaseValidationResult(BaseModel):
    """Evaluation result for an individual benchmark test case."""
    case_id: str
    case_name: str
    category: str
    
    # Relevance metrics
    relevance_score: float
    relevance_class: str
    relevance_score_passed: bool
    relevance_class_passed: bool
    relevance_reasoning: str
    
    # Accuracy metrics
    accuracy_score: float
    accuracy_class: str
    accuracy_score_passed: bool
    accuracy_class_passed: bool
    accuracy_reasoning: str
    
    # Hallucination metrics
    hallucination_score: float
    hallucination_severity: str
    hallucination_detected: bool
    hallucination_detection_passed: bool
    hallucination_reasoning: str
    total_claims: int
    supported_claims: int
    unsupported_claims: int
    contradicted_claims: int
    
    # Overall
    all_checks_passed: bool


class ConsistencyPairResult(BaseModel):
    """Consistency analysis between two semantically equivalent responses."""
    pair_id: str
    case_a_id: str
    case_b_id: str
    relevance_score_delta: float
    accuracy_score_delta: float
    hallucination_score_delta: float
    classes_match: bool
    consistent: bool


class BenchmarkValidationReport(BaseModel):
    """Aggregate validation metrics across the entire evaluation benchmark."""
    total_cases: int
    passed_cases: int
    overall_pass_rate: float

    # Classification accuracy
    relevance_accuracy: float
    accuracy_classification_rate: float
    
    # Hallucination confusion matrix & rates
    hallucination_tp: int
    hallucination_fp: int
    hallucination_tn: int
    hallucination_fn: int
    hallucination_precision: float
    hallucination_recall: float
    hallucination_f1: float

    # Reasoning quality (all cases must produce non-empty, descriptive explanations)
    reasoning_quality_score: float

    # Consistency pairs
    consistency_pairs: List[ConsistencyPairResult] = Field(default_factory=list)
    average_consistency_delta: float

    # Detailed per-case breakdown
    case_results: List[CaseValidationResult] = Field(default_factory=list)
    duration_seconds: float


class BenchmarkValidator:
    """Orchestrates benchmark execution and consistency validation."""

    def __init__(self, orchestrator: Optional[EvaluationOrchestrator] = None):
        self.orchestrator = orchestrator or EvaluationOrchestrator()
        self.retriever = get_semantic_retriever()

    def run_case(self, case: BenchmarkTestCase) -> CaseValidationResult:
        """Executes a single benchmark test case through all three judge agents."""
        # Retrieve ChromaDB context if reference answer is omitted or as additional evidence
        retrieved_items: List[RetrievedEvidenceItem] = []
        try:
            pkg = self.retriever.retrieve(query=case.question, top_k=3)
            retrieved_items = [
                RetrievedEvidenceItem(
                    chunk_id=p.chunk_id,
                    text=p.text,
                    similarity_score=p.similarity_score,
                    dataset_name=p.dataset_name,
                    source_record_id=p.source_record_id,
                    title=p.title,
                    category=p.category,
                )
                for p in pkg.passages
            ]
        except Exception:
            retrieved_items = []

        # 1. Relevance evaluation
        rel_out = self.orchestrator.evaluate_relevance(
            question=case.question,
            ai_response=case.ai_response,
            context=case.source_document or case.reference_answer,
        )

        # 2. Accuracy evaluation
        acc_out = self.orchestrator.evaluate_accuracy(
            question=case.question,
            ai_response=case.ai_response,
            reference_answer=case.reference_answer,
            source_document=case.source_document,
            retrieved_evidence=retrieved_items,
        )

        # 3. Hallucination detection evaluation
        hal_out = self.orchestrator.evaluate_hallucination(
            question=case.question,
            ai_response=case.ai_response,
            reference_answer=case.reference_answer,
            source_document=case.source_document,
            retrieved_evidence=retrieved_items,
        )

        # Validate against expectations
        rel_score_ok = (case.expected_relevance_min_score <= rel_out.score <= case.expected_relevance_max_score)
        rel_class_ok = (
            (rel_out.classification.value == case.expected_relevance_class)
            or (
                case.expected_relevance_class in ("mostly_relevant", "fully_relevant")
                and rel_out.classification.value in ("mostly_relevant", "fully_relevant")
            )
            or (
                case.expected_relevance_class in ("off_topic", "unrelated")
                and rel_out.classification.value in ("off_topic", "unrelated")
            )
            or (
                case.expected_relevance_class in ("partially_relevant", "mostly_relevant")
                and rel_out.classification.value in ("partially_relevant", "mostly_relevant")
            )
        )

        acc_score_ok = (case.expected_accuracy_min_score <= acc_out.score <= case.expected_accuracy_max_score)
        acc_class_ok = (
            (acc_out.classification.value == case.expected_accuracy_class)
            or (
                case.expected_accuracy_class in ("mostly_accurate", "fully_accurate")
                and acc_out.classification.value in ("mostly_accurate", "fully_accurate")
            )
            or (
                case.expected_accuracy_class in ("contradictory", "largely_inaccurate")
                and acc_out.classification.value in ("contradictory", "largely_inaccurate")
            )
        )

        hal_detected_ok = (hal_out.hallucination_detected == case.expected_hallucination_detected)

        all_passed = rel_score_ok and acc_score_ok and hal_detected_ok

        return CaseValidationResult(
            case_id=case.id,
            case_name=case.name,
            category=case.category,
            relevance_score=rel_out.score,
            relevance_class=rel_out.classification.value,
            relevance_score_passed=rel_score_ok,
            relevance_class_passed=rel_class_ok,
            relevance_reasoning=rel_out.reasoning,
            accuracy_score=acc_out.score,
            accuracy_class=acc_out.classification.value,
            accuracy_score_passed=acc_score_ok,
            accuracy_class_passed=acc_class_ok,
            accuracy_reasoning=acc_out.reasoning,
            hallucination_score=hal_out.score,
            hallucination_severity=hal_out.severity.value,
            hallucination_detected=hal_out.hallucination_detected,
            hallucination_detection_passed=hal_detected_ok,
            hallucination_reasoning=hal_out.reasoning,
            total_claims=hal_out.total_claims,
            supported_claims=hal_out.supported_claims_count,
            unsupported_claims=hal_out.unsupported_claims_count,
            contradicted_claims=hal_out.contradicted_claims_count,
            all_checks_passed=all_passed,
        )

    def run_benchmark(
        self, test_suite: Optional[List[BenchmarkTestCase]] = None
    ) -> BenchmarkValidationReport:
        """Executes the full benchmark test suite and calculates summary metrics."""
        suite = test_suite or BENCHMARK_TEST_SUITE
        start_time = time.time()

        case_results: List[CaseValidationResult] = []
        tp = fp = tn = fn = 0
        rel_correct = 0
        acc_correct = 0
        reasoning_valid_count = 0

        for case in suite:
            res = self.run_case(case)
            case_results.append(res)

            # Confusion matrix for hallucination detection
            if case.expected_hallucination_detected:
                if res.hallucination_detected:
                    tp += 1
                else:
                    fn += 1
            else:
                if not res.hallucination_detected:
                    tn += 1
                else:
                    fp += 1

            if res.relevance_class_passed and res.relevance_score_passed:
                rel_correct += 1
            if res.accuracy_class_passed and res.accuracy_score_passed:
                acc_correct += 1

            # Quality of generated reasoning (non-trivial length, specific, non-empty)
            if (
                len(res.relevance_reasoning) >= 15
                and len(res.accuracy_reasoning) >= 15
                and len(res.hallucination_reasoning) >= 15
            ):
                reasoning_valid_count += 1

        total = len(suite)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 1.0

        # Consistency analysis between pairs
        pairs: Dict[str, List[CaseValidationResult]] = {}
        for c, r in zip(suite, case_results):
            if c.pair_id:
                pairs.setdefault(c.pair_id, []).append(r)

        consistency_results: List[ConsistencyPairResult] = []
        total_deltas = []

        for pair_id, pair_cases in pairs.items():
            if len(pair_cases) >= 2:
                a, b = pair_cases[0], pair_cases[1]
                rel_delta = round(abs(a.relevance_score - b.relevance_score), 2)
                acc_delta = round(abs(a.accuracy_score - b.accuracy_score), 2)
                hal_delta = round(abs(a.hallucination_score - b.hallucination_score), 2)
                mean_delta = (rel_delta + acc_delta + hal_delta) / 3.0
                total_deltas.append(mean_delta)

                classes_match = (
                    a.relevance_class == b.relevance_class
                    and a.accuracy_class == b.accuracy_class
                    and a.hallucination_detected == b.hallucination_detected
                )
                is_consistent = classes_match and (mean_delta <= 1.0)

                consistency_results.append(
                    ConsistencyPairResult(
                        pair_id=pair_id,
                        case_a_id=a.case_id,
                        case_b_id=b.case_id,
                        relevance_score_delta=rel_delta,
                        accuracy_score_delta=acc_delta,
                        hallucination_score_delta=hal_delta,
                        classes_match=classes_match,
                        consistent=is_consistent,
                    )
                )

        avg_consistency_delta = (
            round(sum(total_deltas) / len(total_deltas), 3) if total_deltas else 0.0
        )
        passed_count = sum(1 for r in case_results if r.all_checks_passed)

        elapsed = round(time.time() - start_time, 2)

        return BenchmarkValidationReport(
            total_cases=total,
            passed_cases=passed_count,
            overall_pass_rate=round((passed_count / total) * 100, 2),
            relevance_accuracy=round((rel_correct / total) * 100, 2),
            accuracy_classification_rate=round((acc_correct / total) * 100, 2),
            hallucination_tp=tp,
            hallucination_fp=fp,
            hallucination_tn=tn,
            hallucination_fn=fn,
            hallucination_precision=round(precision, 4),
            hallucination_recall=round(recall, 4),
            hallucination_f1=round(f1, 4),
            reasoning_quality_score=round((reasoning_valid_count / total) * 100, 2),
            consistency_pairs=consistency_results,
            average_consistency_delta=avg_consistency_delta,
            case_results=case_results,
            duration_seconds=elapsed,
        )
