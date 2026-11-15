# LLM Evaluation Techniques

## 1. Purpose

Large language models (LLMs) can produce fluent answers that are incomplete, irrelevant, unsupported, or factually incorrect. LLM evaluation measures the quality and reliability of generated responses against a question, reference answer, and/or supporting evidence.

VerityAI evaluates a response across four primary dimensions:

- **Relevance:** Does the response address the user's question?
- **Accuracy:** Are its factual claims correct according to available evidence?
- **Hallucination risk:** Does it contain unsupported, fabricated, or contradicted claims?
- **Completeness:** Does it cover the important parts needed to answer the question?

These dimensions are related but not interchangeable. A response can be relevant but inaccurate, accurate but incomplete, or well-written while containing unsupported claims.

## 2. Main evaluation approaches

### 2.1 Exact-match and token-overlap metrics

Traditional question-answering evaluation often compares a prediction with one or more reference answers.

- **Exact Match (EM):** 1 when the normalized prediction exactly matches a reference; otherwise 0.
- **Token F1:** Computes token-level precision and recall, then their harmonic mean.

These metrics are useful for short, constrained answers. They are less suitable for open-ended responses because different wording can express the same meaning.

### 2.2 Semantic similarity

Embedding models convert text into numerical vectors. Similarity between vectors can estimate whether two pieces of text express related meanings.

Cosine similarity is commonly used:

`cosine_similarity(a, b) = (a · b) / (||a|| ||b||)`

Semantic similarity can help compare a response with a reference answer, but similarity alone does not prove factual correctness. Two statements can be semantically similar while sharing the same factual error.

### 2.3 Model-based evaluation (LLM-as-a-judge)

An evaluator model receives a question, candidate answer, evidence, and a rubric. It returns structured judgments, such as a score and explanation.

Advantages:
- Can assess meaning rather than exact wording.
- Can evaluate nuanced dimensions such as relevance and completeness.
- Can provide claim-level explanations.

Limitations:
- The judge can also hallucinate or make mistakes.
- Results can vary with prompts and model versions.
- Judges may prefer longer or more confident-sounding answers.
- Evaluation can be costly and introduce latency.

VerityAI should use explicit rubrics, structured outputs, evidence references, and deterministic validation where possible. A judge must not be treated as unquestionable ground truth.

### 2.4 Reference-based evaluation

A candidate answer is compared with a trusted reference answer or reference document. This is useful when reliable reference material exists.

A reference answer is not always exhaustive. A candidate should not automatically be marked incorrect merely because it contains additional valid information.

### 2.5 Evidence-grounded evaluation

The system retrieves relevant passages from a knowledge base and checks whether answer claims are supported by those passages.

This approach is especially useful for detecting unsupported statements. However, **not retrieved** does not necessarily mean **false**: the knowledge base may be incomplete or retrieval may fail.

## 3. VerityAI scoring dimensions

Each dimension initially uses a 0–10 scale. The scoring rubric is a starting design and should be calibrated using labelled examples.

| Dimension | Meaning | Direction |
|---|---|---|
| Relevance (R) | How directly the answer addresses the question | Higher is better |
| Accuracy (A) | Factual correctness against references/evidence | Higher is better |
| Hallucination risk (H) | Severity and extent of unsupported or contradicted claims | Lower is better |
| Completeness (C) | Coverage of important information | Higher is better |

Provisional weighted score:

`S = 0.25R + 0.35A + 0.25(10 - H) + 0.15C`

The result is on a 0–10 scale. These weights are provisional, not empirically validated.

### Critical-failure safeguard

A weighted average can hide a severe factual error. The verdict layer should therefore apply explicit safeguards. For example, a clearly contradicted critical claim may force a **Needs Review** or **Unreliable** verdict even if the average score is high.

### Suggested initial score bands

| Score | Suggested interpretation |
|---|---|
| 8.5–10 | Strong |
| 7.0–8.49 | Generally reliable; inspect caveats |
| 5.0–6.99 | Mixed; human review recommended |
| Below 5.0 | Weak; do not rely on without correction |

These bands should be revised after testing. The final verdict must also consider critical contradictions and evidence coverage.

## 4. Evaluation quality and calibration

To assess whether VerityAI itself is working:

1. Build a small labelled test set containing correct, incorrect, incomplete, irrelevant, and unsupported answers.
2. Have humans label the expected outcome and, where possible, the incorrect claims.
3. Compare automated judgments with those labels.
4. Track precision, recall, F1, and confusion matrices for classification outcomes.
5. Track score agreement and disagreement between evaluators.
6. Review false positives and false negatives.
7. Version prompts, models, rubrics, and datasets so results can be reproduced.

For hallucination detection, false negatives are especially important: an unsupported claim that is incorrectly accepted can undermine trust.

## 5. Limitations

- A reference corpus may not contain every true fact.
- Retrieval can return irrelevant or incomplete passages.
- LLM judges can be biased or inconsistent.
- Numeric scores can imply more precision than the evidence supports.
- Domain-specific questions may require domain-specific sources and rubrics.

VerityAI should show evidence and reasoning alongside scores and clearly identify when the available evidence is insufficient.

## 6. References

- RAGAS documentation: https://docs.ragas.io/
- TruLens documentation: https://www.trulens.org/
- Hugging Face Evaluate: https://huggingface.co/docs/evaluate/
- Sentence Transformers: https://sbert.net/
