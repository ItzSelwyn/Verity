# Evaluation Frameworks: RAGAS and TruLens

## 1. Why use evaluation frameworks?

VerityAI can implement its own multi-agent evaluation pipeline, but established frameworks provide useful metrics, patterns, and ways to assess RAG applications. They should be treated as evaluation tools and references, not as automatic sources of ground truth.

## 2. RAGAS

RAGAS provides metrics and evaluation workflows for retrieval-augmented generation systems.

Relevant concepts include:

- **Faithfulness:** Whether answer claims are supported by the retrieved context.
- **Answer relevancy:** Whether the answer addresses the user's question.
- **Context precision:** Whether retrieved context is relevant, especially near the top of the ranking.
- **Context recall:** Whether the retrieved context contains information needed to answer the question.

RAGAS metric names, requirements, and APIs can change between versions. Consult the current documentation before implementation.

Official documentation: https://docs.ragas.io/

## 3. TruLens

TruLens provides tools for instrumenting and evaluating LLM applications. Its evaluation approach includes feedback functions and the commonly discussed RAG triad:

- **Context relevance:** Is the retrieved context relevant to the question?
- **Groundedness:** Is the response supported by the context?
- **Answer relevance:** Does the response answer the question?

TruLens can help inspect application traces and feedback results. The exact APIs and package organization depend on the installed version.

Official documentation: https://www.trulens.org/

## 4. Comparison

| Area | RAGAS | TruLens |
|---|---|---|
| Main orientation | RAG evaluation metrics | LLM application evaluation and observability |
| Useful concepts | Faithfulness, answer relevancy, context precision/recall | RAG triad, feedback functions, traces |
| VerityAI use | Benchmark selected RAG-related metrics | Study groundedness and application-level evaluation |
| Important caution | Metric scores depend on judge/model/configuration | Feedback functions also require validation |

This is a high-level comparison. Check each project's current documentation for supported integrations and version-specific features.

## 5. How VerityAI should use them

For Milestone 1:
1. Document the metrics and concepts relevant to the project's four dimensions.
2. Map RAGAS faithfulness to evidence support and answer relevancy to relevance.
3. Map TruLens groundedness to claim support and context relevance to retrieval quality.
4. Keep VerityAI's own scoring rubric explicit and independent.
5. Later, run the same test cases through VerityAI and selected framework metrics to compare behavior.

Do not simply average scores from different frameworks. Their scales, definitions, and judge configurations may differ.

## 6. References

- RAGAS: https://docs.ragas.io/
- TruLens: https://www.trulens.org/
