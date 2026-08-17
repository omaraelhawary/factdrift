# factdrift — spec

Build spec. One page plus the design note. If it grows past two pages, the project is in trouble.

## Problem

When the same fact lives in several documents with different values, retrieval returns whichever chunk scores highest on similarity, not whichever is current. The generated answer is then faithful to a stale chunk. The system is confidently wrong and every standard eval passes it.

What this tool measures: not "is the answer grounded in the retrieved text", but "does the corpus disagree with itself, and which version does retrieval serve".

## Why existing tools miss it

RAGAS, DeepEval, promptfoo and TruLens score faithfulness (does the answer match the retrieved context) and relevance (does the context match the question). A stale chunk quoted faithfully scores perfectly on both. The defect sits one level below the answer, in the corpus, and nothing in that stack looks there.

## Pipeline

1. **Ingest** markdown and mdx from a pinned corpus directory.
2. **Extract** atomic factual claims per document with an LLM pass.
3. **Normalize** entities and attributes to canonical forms.
4. **Cluster** claims on `entity + attribute`.
5. **Classify** each attribute as single-valued or free-choice (see design note).
6. **Flag** single-valued clusters whose values disagree.
7. **Generate** the natural user question that lands on each flagged cluster.
8. **Run** those questions through retrieval and record which document won and which value the answer stated.

Output: a contradiction inventory, and per contradiction, the value the system currently serves a user.

### Claim schema

```
entity          canonical entity id
attribute       canonical attribute id
value           normalized value
raw_value       value as written
source_path     file path
line_start      int
line_end        int
quote           the sentence or line it came from
context         "prose" | "code"
confidence      float from the extraction pass
```

`context` matters. A value inside a code block is an example. The same value in prose is an assertion. Both can conflict, at different severity.

## Design note: the two hard parts

Everything else in the pipeline is plumbing. These two decide whether the tool is useful or a noise generator.

**Normalization.** `max_tokens` and `maxTokens` are the same attribute. `Anthropic`, `ChatAnthropic` and `langchain-anthropic` are the same entity in some claims and different ones in others. Without an aliasing layer the clustering step finds nothing, because nothing collides.

**Attribute cardinality.** Most differing values are not contradictions. `temperature=0` in one example and `temperature=0.7` in another are both valid user choices. But one document saying the current model is `claude-3-sonnet-20240229` while another says `claude-sonnet-4-6` is a real contradiction. The detector needs to know which attributes have exactly one correct value (package name, env var name, current model, default) and which are free choices the user makes. Get this wrong and precision collapses, because the tool reports every example parameter in the corpus.

Precision is a function of these two steps. Budget accordingly.

## Corpus

Pinned, read-only:

```
repo: langchain-ai/docs
sha:  41abc08558036f8c99ce4b0150b99a0c1d364919
paths: src/oss/python/integrations/chat
       src/oss/javascript/integrations/chat
```

63 files, ~101k words. Fourteen providers are documented in both languages, which is the multi-location fact condition occurring naturally rather than synthetically.

Pin the SHA. Upstream moves — HEAD changed the same day this corpus was chosen. If the files shift after the gold set is labeled, the labels stop matching and precision and recall become uncomputable.

Never modify corpus files. The control set is a separate copy under `fixtures/`.

## Smoke test

Verified drift on the Anthropic provider before the corpus was chosen. Weekend 2 succeeds when the detector finds these without being told they exist:

- The Python doc leads with `claude-sonnet-4-6`. The JavaScript doc leads with `claude-haiku-4-5-20251001` and still carries `claude-3-sonnet-20240229`, an identifier from early 2024.
- `max_tokens` appears as 5000 and 4096 in Python; `maxTokens` as 1024 in JavaScript.
- The Python doc contains `temperature=75.0`, outside the valid range for that API.

These are the smoke test, not the gold set.

## Gold set and detector evaluation

This is what separates the project from a demo. It is not optional and it is not delegable.

- Hand-label every real contradiction across the 14 shared providers. Two passes, second pass a day later.
- Build a control set: a copy of ~10 documents with contradictions injected at known locations.
- Report precision and recall against both.
- List every false positive and why it fired. The false positives are the most interesting output.
- Record cost and wall-clock for a full run.

A detector with no measured precision is a suggestion, not an eval.

## Cost discipline

The extraction pass covers ~101k words and is the only expensive step.

- Never run the full corpus without first running 5 files and printing projected full-run cost and token counts.
- Cache extraction output to disk keyed by file content hash. A rerun with an unchanged prompt must cost nothing.
- Make the model and the file limit CLI flags, not constants.

## What it does not do

No fine-tuning. No abstraction layer over vector stores — one store, hardcoded. No web UI. No multi-language corpora. No agent framework. No answer-quality scoring; other tools do that and adding it here dilutes the point.

CLI in, JSON and markdown report out.

## Stack

Python 3.12+. uv. pytest. Chroma for embeddings, local, no infrastructure. Anthropic SDK for the extraction and question-generation passes. Public repo from the first commit.

## Milestones

**Weekend 1 — ingest and extract.** Claims dumped for all 63 files, schema above, cached to disk. Success: a hand-check of 3 files against the source finds no invented claims and no missed values in prose.

**Weekend 2 — normalize, cluster, detect.** Success: the smoke test findings surface without being told about them, and the report does not drown them in free-choice parameter noise.

**Weekend 3 — retrieval run and evaluation.** Questions generated, retrieval run, gold set labeled, precision and recall reported with the false-positive list.

Anything that does not serve one of these three is v2.

## Constraints

Nothing in this repo — code, README, fixtures, test data — comes from an employer system. The corpus above is the only source of examples.
