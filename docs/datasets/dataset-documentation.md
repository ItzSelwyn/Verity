# Reference Dataset Documentation

## 1. Purpose

VerityAI uses reference datasets to seed an initial knowledge base and to build evaluation test cases. Dataset records must retain their original identifiers and metadata so that retrieved evidence can be traced to its source.

The initial datasets are TruthfulQA and SQuAD.

## 2. TruthfulQA

**Purpose:** TruthfulQA was designed to test whether models reproduce common misconceptions rather than provide truthful answers.

Dataset page: https://huggingface.co/datasets/truthfulqa/truthful_qa

### Configuration and split decision

TruthfulQA has two configurations:

| Configuration | Schema | VerityAI use |
|---|---|---|
| `generation` | `question`, `best_answer`, `correct_answers`, `incorrect_answers`, `source`, `category` | **Selected** — provides ranked correct/incorrect answers for factuality evaluation |
| `multiple_choice` | `question`, `mc1_targets`, `mc2_targets` | Deferred — useful for future multiple-choice evaluation tasks |

**Selected split:** `validation` (the full dataset has 817 questions). There is no separate test split with public labels.

Load with: `load_dataset("truthfulqa/truthful_qa", "generation", split="validation")`

Potential VerityAI uses:
- Build test questions involving common misconceptions.
- Test whether an evaluator detects misleading or incorrect answers.
- Retain category and answer metadata for analysis.

Important caution:
- Do not flatten all fields into one undifferentiated text string.
- Preserve which answers are designated correct or incorrect.
- Check the dataset card and selected configuration for the exact schema before writing the loader.

## 3. SQuAD

**Purpose:** The Stanford Question Answering Dataset (SQuAD) contains questions posed on Wikipedia passages, with answer annotations.

Dataset page: https://huggingface.co/datasets/rajpurkar/squad

### Configuration and split decision

| Version | Notes | VerityAI use |
|---|---|---|
| SQuAD v1.1 (`rajpurkar/squad`) | All questions have answers in the passage | **Selected** — simpler answer structure, well-understood schema |
| SQuAD v2.0 (`rajpurkar/squad_v2`) | Adds unanswerable questions | Deferred — useful for future "no evidence" evaluation testing |

**Selected split:** `validation` (~10,570 examples). The training split (~87,599) can be added later if more coverage is needed.

Load with: `load_dataset("rajpurkar/squad", split="validation")`

Common fields include:
- `id`
- `title`
- `context`
- `question`
- `answers`

Potential VerityAI uses:
- Index source contexts as evidence passages.
- Use questions and answer annotations for retrieval/evaluation test cases.
- Check whether retrieved context contains the information needed to support an answer.

Important caution:
- The context is the evidence passage; the answer annotation is not a replacement for the context.
- Preserve article title, question, answer text, and answer offsets where available.
- Confirm the exact dataset version and schema before ingestion.

## 4. Initial ingestion design

1. Load the selected dataset configuration and split.
2. Inspect fields and sample records.
3. Remove empty or malformed records.
4. Normalize whitespace without changing factual meaning.
5. Create source records with stable IDs and provenance.
6. Chunk long contexts while preserving titles and source IDs.
7. Generate embeddings.
8. Upsert chunk text, embeddings, and metadata into ChromaDB.
9. Record ingestion counts and errors.
10. Run retrieval tests against a small set of known questions.

## 5. Proposed metadata

| Metadata | Description |
|---|---|
| dataset_name | `truthfulqa` or `squad` |
| dataset_split | Selected split, e.g. train or validation |
| source_record_id | Original dataset record identifier |
| title | Article title, when available |
| category | TruthfulQA category, when available |
| chunk_index | Position of chunk within source |
| content_type | context, question, answer, or other typed content |

Use separate content types or collections if necessary. Avoid mixing short answer strings and long evidence passages without metadata to distinguish them.

## 6. Data quality and validation

Before indexing:
- Inspect records from every selected split.
- Count records before and after cleaning.
- Check for missing IDs and empty contexts.
- Detect duplicate records.
- Confirm answer annotations remain linked to their source.
- Save a small ingestion report.

After indexing:
- Verify the expected number of chunks.
- Query sample questions.
- Inspect top-k results manually.
- Check that each result has source provenance.
- Record retrieval failures and adjust chunking or retrieval settings.

## 7. Dataset scope and limitations

These datasets are useful starting points, not a complete factual knowledge base. They are not guaranteed to cover current events, every domain, or all user questions. Dataset-derived evidence should be labelled as such and not presented as universally authoritative.

For real-world use, add curated, domain-appropriate sources and track their publication dates and trust levels.

## 8. References

- TruthfulQA: https://huggingface.co/datasets/truthfulqa/truthful_qa
- SQuAD: https://huggingface.co/datasets/rajpurkar/squad
- Hugging Face Datasets documentation: https://huggingface.co/docs/datasets/
