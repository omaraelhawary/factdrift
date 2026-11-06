# factdrift

Find conflicting factual claims in Markdown and MDX documents. Factdrift extracts claims with an LLM, assigns canonical entity and attribute IDs, and reports disagreements with source paths, line numbers, and quotes.

The CLI currently supports ingestion, extraction, normalization, and detection. Question generation, retrieval checks, and precision/recall evaluation are planned in [SPEC.md](SPEC.md) and are not implemented yet.

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

The following commands use Anthropic for extraction and OpenRouter for normalization. Set `ANTHROPIC_API_KEY` and `OPENROUTER_API_KEY` in your environment before running them. The CLI does not load `.env` files automatically. Both stages send document content or extracted claims to the selected provider and can incur API charges.

1. Extract claims from five files first:

   ```sh
   uv run factdrift extract --corpus corpus --limit 5 --out out/claims.json
   ```

   Review the kept and rejected claims, token counts, and projected full-corpus cost before increasing the limit. The limit is a manual precaution; the CLI does not enforce a spending approval step. Use `--show-claims 10` to print a sample and `--model` to choose a model available to your account.

2. Normalize the extracted claims:

   ```sh
   uv run factdrift normalize --backend openrouter --claims out/claims.json --out out/normalized.json
   ```

   This assigns canonical IDs and classifies attributes as single-valued or free-choice. Normalization supports OpenRouter and Ollama; the Anthropic backend is currently rejected for this command.

3. Write the findings and report:

   ```sh
   uv run factdrift detect --claims out/normalized.json --out out/findings.json --report out/report.md
   ```

   Detection runs locally without model requests. Omit `--out` and `--report` to print the report to the terminal.

Extraction and normalization cache results under `.cache/`. Keep the cache to reuse completed work; changing inputs, prompts, or model settings can require new requests. Generated files under `out/` are ignored by Git.

## Other model backends

Extraction also supports `--backend openrouter` and `--backend ollama`. For local inference, start an Ollama server and pull the model first, then run:

```sh
uv run factdrift extract --backend ollama --model qwen3:8b --corpus corpus --limit 5 --out out/claims.json
uv run factdrift normalize --backend ollama --model qwen3:8b --claims out/claims.json --out out/normalized.json
```

Use `--ollama-url`, `--num-ctx`, or `--timeout` to adjust the connection and context window. Each command's `--help` lists its defaults and available flags. For cost reporting on models without built-in prices, supply both `--price-in` and `--price-out` in dollars per million tokens; OpenRouter extraction projections estimate token counts locally.

## Interpret the report

A contradiction is a single-valued attribute with disagreeing values within one language. Differences between Python and JavaScript alone do not trigger that rule. Free-choice attributes with disjoint Python and JavaScript values are reported separately as divergence; use `--no-divergence` to omit them.

Severity depends on the source context: high when at least two conflicting values have prose evidence, medium when only one does, and low when all evidence is code. Review the cited sources before treating a finding as an error. The detector does not decide which value is current or correct, and measured precision and recall are not yet available.

## Tests

```sh
uv run pytest
```

Tests use fixtures and fake model responses without calling the Anthropic API. See [SPEC.md](SPEC.md) for the corpus constraints and remaining evaluation work.
