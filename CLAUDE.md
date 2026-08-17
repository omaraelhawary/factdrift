# factdrift

A detector for corpus-level contradiction: find facts that appear in multiple documents with conflicting values, then measure which version a retrieval pipeline actually serves. Read `SPEC.md` before writing code. Do not expand scope beyond it.

## Stack

Python 3.12+, uv for dependencies, pytest for tests, Chroma for embeddings, Anthropic SDK for LLM passes. No web framework. No vector-store abstraction layer.

## How to work in this repo

Take the correct approach, not the quick one. If you find a bug while working on something else, fix it in the same pass rather than leaving a note.

Keep code modular and reusable. Each pipeline stage is independently callable and independently testable. No stage reaches into another's internals.

No hardcoded values that a caller might reasonably want to change. Model names, file limits, paths, thresholds and batch sizes are CLI flags or config, not constants buried in a function.

Guard inputs. Empty files, malformed frontmatter, values that fail to parse, and LLM responses that don't match the schema are expected conditions, not crashes.

Do not create files that were not asked for. No README beyond what the project needs to be run, no CONTRIBUTING, no docs directory, no example notebooks, no badges.

Never write marketing or positioning copy. Not in the README, not in docstrings. Describe what the code does.

## Cost discipline

The extraction pass is the only expensive step and it is easy to waste money on.

Cache extraction output to disk, keyed by a hash of file content plus prompt version. A rerun with an unchanged prompt and unchanged files must make zero API calls.

Never run the full corpus on a first attempt. Run a small subset, print token counts and projected full-run cost, and stop for a decision.

## Corpus rules

The corpus is pinned by SHA in `SPEC.md` and is read-only. Never edit a corpus file, never regenerate it from a newer upstream commit, and never let a test write into it. Injected-contradiction fixtures are separate copies under `fixtures/`.

## Testing

Every detection rule gets a test with a fixture that isolates it. Tests do not call the Anthropic API — record fixture responses instead.

Tests must fail for the right reason. A test that passes against a stub that returns nothing is not a test.

## Secrets

`ANTHROPIC_API_KEY` comes from the environment. Never write a key into a file, a test, or an example. `.env` is gitignored from the first commit.

## Commits

Conventional commits. One logical change per commit. Do not bundle a refactor with a behavior change.

## Writing style for anything human-readable

Applies to the README, CLI help text, report output, and commit bodies.

US spelling. Sentence case for headings. No emoji. No em-dash overuse. Plain, specific language: name the thing rather than describing it in the abstract. Cut any sentence that is trying to impress rather than inform.
