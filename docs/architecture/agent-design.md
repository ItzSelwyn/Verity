# Agent Responsibilities and Evaluation Design

## 1. Agent model

VerityAI uses an orchestrated set of specialized evaluators. Each evaluator receives a defined input contract and returns a structured result. The orchestrator coordinates execution; it does not replace the specialist judgments.

## 2. Responsibilities

### 2.1 Evaluation Orchestrator
- Receives a validated submission.
- Requests evidence retrieval.
- Builds a consistent evaluation context.
- Invokes the required judges.
- Validates outputs and handles failures.
- Sends results to the verdict component.

### 2.2 Relevance Judge
Question: **Does the response address what was asked?**

Checks:
- Whether the answer responds to the main question.
- Whether it stays on topic.
- Whether it answers the requested task or format.

Does not decide factual truth by itself.

### 2.3 Accuracy Judge
Question: **Are the factual statements correct according to the available references and evidence?**

Checks:
- Agreement with a supplied reference answer.
- Support from retrieved source passages.
- Numerical, temporal, entity, and attribution correctness.

If evidence is insufficient, it should say so rather than guess.

### 2.4 Hallucination Detection Agent
Question: **Does the response contain unsupported, fabricated, or contradicted claims?**

Checks:
- Atomic factual claims.
- Evidence support or contradiction.
- Fabricated citations or specific details.
- Severity and importance of problematic claims.

It must distinguish contradicted from insufficiently supported claims.

### 2.5 Completeness Judge
Question: **Does the response include the important information needed to answer the question?**

Checks:
- Coverage of key aspects.
- Missing constraints or requested sub-parts.
- Whether omissions change the usefulness of the answer.

Completeness should be judged relative to the question and task, not answer length.

### 2.6 Verdict Component
- Combines the specialist results.
- Applies the documented scoring formula.
- Applies critical-failure safeguards.
- Produces an overall verdict and concise explanation.
- Preserves disagreements and uncertainty.

The verdict is an aggregation component; it should not invent new evidence.

## 3. Scoring dimensions

All initial dimension scores use 0–10.

| Dimension | Score meaning |
|---|---|
| Relevance | 0 = unrelated; 10 = directly addresses the request |
| Accuracy | 0 = largely incorrect; 10 = fully supported/correct on checked facts |
| Hallucination risk | 0 = no detected risk; 10 = severe/pervasive risk |
| Completeness | 0 = essential information missing; 10 = all important parts covered |

Provisional overall score:

`S = 0.25R + 0.35A + 0.25(10 - H) + 0.15C`

Weights are initial design assumptions and must be calibrated. A critical contradiction may trigger a negative verdict or review flag independently of the weighted score.

## 4. Suggested structured agent result

```json
{
  "agent": "accuracy_judge",
  "score": 3,
  "summary": "The central factual claim conflicts with the supplied source.",
  "findings": [
    {
      "claim": "Sydney is the capital of Australia.",
      "status": "contradicted",
      "evidence_ids": ["chunk_001"],
      "explanation": "The cited passage identifies Canberra."
    }
  ],
  "confidence": "high"
}
```

This is a proposed output contract. Implement it as Pydantic models and validate every agent response. `confidence` should be a controlled enum, not arbitrary free text.

## 5. Failure handling

- If retrieval fails, do not pretend evidence was checked.
- If one judge fails, record the failure and mark the report partial or requiring review.
- Do not silently replace missing scores with perfect scores.
- Preserve raw judge output for debugging only when privacy and retention rules allow it.
- Record model and prompt versions.

## 6. Evaluation safeguards

- Require evidence IDs for evidence-based findings.
- Keep unsupported and contradicted labels separate.
- Detect disagreement between agents.
- Apply critical-failure rules after scoring.
- Make the rubric visible in documentation and the final report.
