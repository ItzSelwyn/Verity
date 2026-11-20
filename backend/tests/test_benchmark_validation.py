"""
test_benchmark_validation.py — Pytest suite for M2.4 Agent Evaluation & Consistency Validation.

Tests that the 26-case representative benchmark achieves:
- Overall pass rate >= 90%
- Hallucination precision >= 90%
- Hallucination recall >= 90%
- Reasoning quality score == 100%
- Semantically equivalent consistency pair invariance (mean delta <= 0.5)
"""

import pytest
from app.benchmarks.validator import BenchmarkValidator
from app.benchmarks.benchmark_dataset import BENCHMARK_TEST_SUITE


@pytest.fixture(scope="module")
def benchmark_report():
    validator = BenchmarkValidator()
    report = validator.run_benchmark(BENCHMARK_TEST_SUITE)
    return report


def test_benchmark_overall_pass_rate(benchmark_report):
    """M2.4: Ensure at least 90% of benchmark test cases pass all multi-agent criteria."""
    assert benchmark_report.overall_pass_rate >= 90.0, (
        f"Pass rate {benchmark_report.overall_pass_rate}% is below the 90% threshold."
    )
    assert benchmark_report.total_cases >= 20


def test_hallucination_detection_precision_and_recall(benchmark_report):
    """M2.4: Ensure hallucination detector has zero or minimal false positives and false negatives."""
    assert benchmark_report.hallucination_precision >= 0.90, (
        f"Precision {benchmark_report.hallucination_precision} is below 0.90."
    )
    assert benchmark_report.hallucination_recall >= 0.90, (
        f"Recall {benchmark_report.hallucination_recall} is below 0.90."
    )
    assert benchmark_report.hallucination_f1 >= 0.90
    assert benchmark_report.hallucination_fn == 0  # Zero missed hallucinations


def test_agent_reasoning_quality(benchmark_report):
    """M2.4: Verify that all agents generate non-empty, actionable explanations for their scores."""
    assert benchmark_report.reasoning_quality_score == 100.0
    for case in benchmark_report.case_results:
        assert len(case.relevance_reasoning) >= 15
        assert len(case.accuracy_reasoning) >= 15
        assert len(case.hallucination_reasoning) >= 15


def test_score_consistency_across_semantically_equivalent_responses(benchmark_report):
    """M2.4: Verify that semantically equivalent candidate responses receive consistent scores."""
    assert len(benchmark_report.consistency_pairs) >= 2
    for pair in benchmark_report.consistency_pairs:
        assert pair.consistent is True
        assert pair.classes_match is True
        assert pair.relevance_score_delta <= 0.5
        assert pair.accuracy_score_delta <= 0.5
        assert pair.hallucination_score_delta <= 0.5

    assert benchmark_report.average_consistency_delta <= 0.25


def test_relevance_and_accuracy_classification_rates(benchmark_report):
    """M2.4: Verify categorical classification stability across prompt response archetypes."""
    assert benchmark_report.relevance_accuracy >= 90.0
    assert benchmark_report.accuracy_classification_rate >= 90.0
