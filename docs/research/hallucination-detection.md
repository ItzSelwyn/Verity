# Hallucination Detection Methods

## 1. Definition

An LLM hallucination is generated content that is not adequately supported by the available input, evidence, or reliable facts. In VerityAI, hallucination detection focuses on identifying claims that are unsupported, fabricated, or contradicted by the evidence available to the evaluation pipeline.

The system must distinguish between:
- **Supported:** Evidence directly supports the claim.
- **Contradicted:** Evidence conflicts with the claim.
- **Insufficient evidence:** The retrieved material does not establish whether the claim is true.
- **Not applicable:** The statement is opinion, style, or another type not assessed as a factual claim.

Absence from a retrieved passage is not proof that a claim is false.

## 2. Common categories

### 2.1 Factual hallucination
A response states a fact that conflicts with reliable information.

### 2.2 Unsupported claim
A response makes a factual assertion for which the supplied or retrieved evidence provides no adequate support.

### 2.3 Entity or attribution error
A response assigns an action, quotation, role, date, or property to the wrong person, organization, or object.

### 2.4 Numerical or temporal error
A response gives an incorrect number, measurement, date, duration, or sequence.

### 2.5 Causal overstatement
A response claims that one event caused another when the evidence only shows correlation or does not establish a causal link.

### 2.6 Fabricated citation
A response names a source, quotation, study, URL, or citation that cannot be verified.

## 3. Detection pipeline

A practical evidence-grounded pipeline is:

1. **Input validation:** Check that the question and candidate answer are present.
2. **Claim extraction:** Break the response into independently checkable factual claims.
3. **Evidence retrieval:** Retrieve relevant passages from the reference knowledge base and any user-provided source.
4. **Claim-evidence matching:** Compare each claim with the retrieved passages.
5. **Claim classification:** Label each claim as supported, contradicted, or insufficient evidence.
6. **Risk aggregation:** Consider severity, importance, and number of problematic claims.
7. **Explanation:** Return the claim, evidence, classification, and reason.

Claim extraction and classification may use an LLM, but the pipeline should validate the returned structure and preserve links to the source passages.

## 4. Claim-level output model

An initial claim assessment can use this structure:

```json
{
  "claim": "Sydney is the capital of Australia.",
  "status": "contradicted",
  "evidence": [
    {
      "text": "Canberra is the capital city of Australia.",
      "source": "user_provided_reference"
    }
  ],
  "reason": "The evidence identifies Canberra, not Sydney."
}
```

The example is illustrative. Production records should include stable source and chunk identifiers rather than relying only on copied text.

## 5. Hallucination risk score

A 0–10 risk score is proposed:

- 0: No issue identified in the checked claims.
- 1–3: Minor unsupported details or low-impact uncertainty.
- 4–6: Several unsupported claims or a material claim with weak evidence.
- 7–9: Major unsupported or contradicted claims.
- 10: Severe or pervasive fabrication/contradiction.

This rubric is provisional. Risk should reflect claim importance, not just claim count. One dangerous or central false claim may matter more than several harmless details.

## 6. Important safeguards

- Do not label every unsupported claim as false.
- Preserve the exact evidence passage and its provenance.
- Distinguish user-provided references from the system's general dataset.
- If retrieval returns no useful evidence, report insufficient evidence.
- Avoid evaluating opinions as factual claims unless the task asks for factual support.
- Include uncertainty and a human-review path for ambiguous cases.
- Test on deliberately misleading and adversarial examples.

## 7. References

- TruthfulQA paper and dataset: https://huggingface.co/datasets/truthfulqa/truthful_qa
- RAGAS documentation: https://docs.ragas.io/
- TruLens documentation: https://www.trulens.org/
