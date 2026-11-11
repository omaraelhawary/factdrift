# factdrift

Find conflicting factual claims in Markdown and MDX documents, and check which of the conflicting values a retrieval pipeline serves. Factdrift extracts claims with an LLM, assigns canonical entity and attribute IDs, reports disagreements with source paths, line numbers, and quotes, then asks retrieval the question each disagreement would answer. [SPEC.md](SPEC.md) describes the problem and the scope.

## Install

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```sh
git clone https://github.com/omaraelhawary/factdrift.git
cd factdrift
uv sync --locked
uv run factdrift --help
```

## Prepare the corpus

The reference corpus is pinned to commit `41abc08558036f8c99ce4b0150b99a0c1d364919` of `langchain-ai/docs`. Fetch that commit and copy only the two chat integration directories into the ignored `corpus/` directory:

```sh
git init corpus
git -C corpus fetch --depth 1 https://github.com/langchain-ai/docs.git 41abc08558036f8c99ce4b0150b99a0c1d364919
git -C corpus restore --source=FETCH_HEAD --worktree -- src/oss/python/integrations/chat src/oss/javascript/integrations/chat
uv run factdrift ingest --corpus corpus --limit 5
```

Keep these files read-only. For your own corpus, pass its root with `--corpus`; ingestion recursively loads `.md` and `.mdx` files.

## Run the pipeline

The numbers below came from OpenRouter `anthropic/claude-haiku-5.5` for extraction and normalization, and local Ollama `qwen3:8b` for questions and retrieval. Set `OPENROUTER_API_KEY` in your environment for the OpenRouter steps. The CLI does not load `.env` files. Extraction and normalization send document content or claims to the provider and cost money.

1. Extract claims from five files, check the projected full-run cost it prints, then rerun without `--limit`:

   ```sh
   uv run factdrift extract --backend openrouter --corpus corpus --limit 5 --out out/claims.json
   ```

2. Assign canonical IDs and classify each attribute as single-valued or free-choice:

   ```sh
   uv run factdrift normalize --backend openrouter --claims out/claims.json --out out/normalized.json
   ```

3. Detect disagreements. This step makes no model requests:

   ```sh
   uv run factdrift detect --claims out/normalized.json --out out/findings.json --report out/report.md
   ```

4. Write one user question per finding, then run each question through retrieval and record the value the answer states:

   ```sh
   uv run factdrift questions --backend ollama --findings out/findings.json --out out/questions.json
   uv run factdrift retrieve --backend ollama --corpus corpus --findings out/findings.json \
     --questions out/questions.json --out out/findings.json --report out/report.md
   ```

   Retrieval uses one local Chroma index with the default `all-MiniLM-L6-v2` embedder. Pass `--embed-model` to embed with an Ollama model instead.

5. Score the findings against a labeled set:

   ```sh
   uv run factdrift score --findings out/findings.json --labels gold/labels.csv --report out/report.md
   ```

Every model step caches its output under `.cache/`, so a rerun with unchanged inputs and prompts makes no requests. Each command's `--help` lists its flags and defaults. For models without built-in prices, pass `--price-in` and `--price-out` in dollars per million tokens.

## Read the report

A contradiction is a single-valued attribute with more than one value within one language. Python and JavaScript disagreeing on their own does not trigger it. A divergence is a free-choice attribute whose Python and JavaScript value sets share no value; `--no-divergence` drops them.

Severity counts prose evidence: high when at least two conflicting values appear in prose, medium when one does, low when all evidence is code. The detector does not decide which value is correct.

The "What retrieval serves" section lists, per finding, the question asked, the top chunk, the disputed values present in the top-k chunks, and the value the answer stated.

## Results on the pinned corpus

Extraction kept 2,417 claims from 63 files. Detection reported 34 findings:

| kind | high | medium | low |
| --- | --- | --- | --- |
| contradiction | 13 | 1 | 8 |
| divergence | 3 | 2 | 7 |

Retrieval results for the 34 findings: no answer for 13, one of the disputed values for 15, a value outside the disputed set for 4, and several values for 2.

Smoke test from SPEC.md:

- `anthropic` / `model` is reported: the JavaScript page uses `claude-sonnet-4-6` at line 263 and `claude-3-sonnet-20240229` at line 283.
- `max_tokens` is reported as a low divergence with 1024, 5000, and an unset value. The 4096 value was not extracted.
- `temperature=75.0` is not reported. In this corpus it appears as `Weather(temperature=75.0, ...)`, a tool output field, and extraction kept no claim for it.

## Precision and recall

### Gold set

Method:

1. `factdrift candidates` lists every group of disagreeing claims before cardinality is applied, on the 13 page names that have claims in both languages (`index.mdx` and `TEMPLATE.mdx` excluded). That gave 87 candidates in `gold/labels.csv`, 31 of them reported by detect.
2. Each candidate was labeled real or not-real with a one-line reason.
3. `factdrift score` counts a finding as correct when it shares a source line with a row labeled real and agrees on entity and attribute. Findings on files outside the labeled set are left out of precision.

The labels were written by Codex (an OpenAI model), not by hand. SPEC.md calls for two hand-labeled passes a day apart; the owner approved this substitution. Codex ran two passes on the same day without the detector and cardinality columns, the second in shuffled order. The passes agree on all 87 rows. That shows the labeler is consistent with itself. It is not an independent check of the labels.

Results, identical for both passes:

- Precision 9%: 2 of 22 scored findings match a real label. One finding (`chat_groq` / `model`) covers two real rows.
- Recall 60%: 3 of 5 real labels found.
- 12 findings sit on single-language pages outside the gold set and have no measured precision. The three HuggingFace feature-table findings are among them.

Recall here is relative to the claims that extraction and normalization produced. A contradiction that extraction missed never becomes a candidate and cannot lower this number. The control set measures that end to end.

The two real labels the detector missed are both a configured model that differs from the model named in the displayed response: `python/anthropic.mdx` line 86 against line 178, and `python/cerebras.mdx` lines 92 and 161 against lines 112 and 178.

### Control set

`fixtures/control/` holds copies of 10 corpus pages (DeepSeek, Fireworks, Groq, xAI, and Cerebras in both languages) with 12 contradictions injected. `fixtures/control/labels.jsonl` records each one's location, original value, and injected value.

```sh
uv run factdrift extract --backend ollama --corpus fixtures/control --out out/control/claims.json
uv run factdrift normalize --backend ollama --claims out/control/claims.json --out out/control/normalized.json
uv run factdrift detect --claims out/control/normalized.json --out out/control/findings.json
uv run factdrift score --findings out/control/findings.json --labels fixtures/control/labels.jsonl
```

This run used `qwen3:8b` on Ollama for extraction and normalization, not the Haiku model used on the corpus. Results:

- Precision 20%: 3 of 15 findings.
- Recall 25%: 3 of 12 injections.

All 9 misses happened before detection. Four sit in pages that timed out at 1,200 seconds (both Cerebras pages and JavaScript Groq). Three had no claim extracted from the injected line. Two had the wrong value extracted. All 3 injections that were extracted correctly were detected.

## Known false positives

From the gold set and control set:

- Values that depend on a condition: `ollama` / `model_storage_directory` gives one path for macOS and another for Linux; `google_genai` / `temperature` gives one default for Gemini 3.0+ and another for older models.
- Different tools or packages sharing one attribute: `code_execution_tool` / `minimum_version` merges two dated tool versions; `reasoning_effort` / `minimum_version` merges versions of several provider packages.
- Different parameters merged into one attribute: `bash_tool` / `required_parameter` merges `command` and `restart`; `app_attribution` / `env_var` merges `OPENROUTER_APP_URL` and `OPENROUTER_APP_TITLE`.
- Example values treated as fixed: `fireworks` / `model`, `anthropic` / `max_tokens`, and `chat_vertex_ai` / `max_retries` are choices made in separate examples.
- One value written two ways: `context_caching` / `ttl_seconds` is `300` and `300s`; `with_structured_output` / `method` is `jsonSchema` and `json_schema`.
- Set-valued attributes: `predicted_outputs` / `supported_model` and `ollama` / `install_command` list several members, and two pages listing different members do not disagree.
- Generic entity names: in the control set, 11 of 12 false positives came from names such as "chat model" merging claims from different providers.

## Retrieval limits

Each Markdown table row is indexed as its own chunk with the table's header row and section heading. The HuggingFace feature-table rows still rank 13th to 26th for their questions, outside the top 5, so the answer for image input is `false` without the table in context.

## Cost and run time

Full corpus run on OpenRouter `anthropic/claude-haiku-5.5`:

| step | tokens in / out | cost | wall clock |
| --- | --- | --- | --- |
| extract, 63 files | 453,395 / 421,492 | $0.26 | 371 s at concurrency 4 |
| normalize | 217,924 / 202,388 | $0.12 | not recorded |
| questions and retrieval | local Ollama | $0 | not recorded |

The control set extraction on `qwen3:8b` took 4,323 seconds at concurrency 1.

## Tests

```sh
uv run pytest
```

Tests use fixtures and fake model responses and make no API calls.
