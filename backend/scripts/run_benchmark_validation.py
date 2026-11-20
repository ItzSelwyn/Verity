"""
run_benchmark_validation.py — CLI Runner for M2.4 Agent Evaluation & Consistency Validation.

Executes the 24-case benchmark test suite across Relevance, Accuracy, and Hallucination Judge Agents,
calculates Precision, Recall, F1, and Consistency metrics, and generates a structured summary.
"""

import sys
import os

# Ensure backend root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.benchmarks.validator import BenchmarkValidator
from app.benchmarks.benchmark_dataset import BENCHMARK_TEST_SUITE


def main():
    print("=" * 80)
    print("VERITY EVALUATION PLATFORM — MILESTONE 2.4 BENCHMARK VALIDATION")
    print("Running 24-case benchmark suite across Relevance, Accuracy & Hallucination Agents...")
    print("=" * 80)

    validator = BenchmarkValidator()
    report = validator.run_benchmark(BENCHMARK_TEST_SUITE)

    print("\n--- INDIVIDUAL CASE RESULTS ---")
    for r in report.case_results:
        status_flag = "PASS" if r.all_checks_passed else "FAIL"
        print(f"[{status_flag}] {r.case_id} ({r.category}) - {r.case_name}")
        print(f"       Relevance: {r.relevance_score:.2f} ({r.relevance_class}) | Passed: {r.relevance_score_passed and r.relevance_class_passed}")
        print(f"       Accuracy:  {r.accuracy_score:.2f} ({r.accuracy_class}) | Passed: {r.accuracy_score_passed and r.accuracy_class_passed}")
        print(f"       Hallucination: {r.hallucination_score:.2f} ({r.hallucination_severity}) | Detected: {r.hallucination_detected} | Passed: {r.hallucination_detection_passed}")
        print(f"       Claims: {r.total_claims} total (Supported: {r.supported_claims}, Unsupported: {r.unsupported_claims}, Contradicted: {r.contradicted_claims})")
        if not r.all_checks_passed:
            print(f"       Reasoning Sample: {r.accuracy_reasoning[:120]}...")

    print("\n" + "=" * 80)
    print("BENCHMARK AGGREGATE EVALUATION METRICS")
    print("=" * 80)
    print(f"Total Test Cases:            {report.total_cases}")
    print(f"Cases Passing All Checks:    {report.passed_cases} / {report.total_cases} ({report.overall_pass_rate}%)")
    print(f"Relevance Scoring Accuracy:  {report.relevance_accuracy}%")
    print(f"Accuracy Classification Rate:{report.accuracy_classification_rate}%")
    print(f"Reasoning Quality Score:     {report.reasoning_quality_score}% (valid, non-empty, actionable)")
    print(f"Execution Duration:          {report.duration_seconds}s")

    print("\nHALLUCINATION DETECTION CONFUSION MATRIX & ACCURACY:")
    print(f"  True Positives  (TP): {report.hallucination_tp}")
    print(f"  False Positives (FP): {report.hallucination_fp}")
    print(f"  True Negatives  (TN): {report.hallucination_tn}")
    print(f"  False Negatives (FN): {report.hallucination_fn}")
    print(f"  Precision:            {report.hallucination_precision * 100:.2f}%")
    print(f"  Recall:               {report.hallucination_recall * 100:.2f}%")
    print(f"  F1 Score:             {report.hallucination_f1 * 100:.2f}%")

    print("\nCONSISTENCY VALIDATION (SEMANTICALLY EQUIVALENT PAIRS):")
    for cp in report.consistency_pairs:
        status = "CONSISTENT" if cp.consistent else "DIVERGENT"
        print(f"  [{status}] Pair {cp.pair_id} ({cp.case_a_id} vs {cp.case_b_id}):")
        print(f"       Relevance Delta:     {cp.relevance_score_delta}")
        print(f"       Accuracy Delta:      {cp.accuracy_score_delta}")
        print(f"       Hallucination Delta: {cp.hallucination_score_delta}")
        print(f"       Classes Match:       {cp.classes_match}")
    print(f"  Mean Score Delta:    {report.average_consistency_delta}")
    print("=" * 80)


if __name__ == "__main__":
    main()
