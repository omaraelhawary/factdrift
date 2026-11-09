"""Retrieval run, pipeline step 8.

Indexes the corpus in a local Chroma collection, retrieves the top k chunks
for each generated question, asks a model to answer from those chunks only,
and records which of the finding's disputed values the answer states.

Chunking: one chunk per markdown section, where a section runs from one
heading line to the next. A section longer than `max_chars` is split at
blank lines outside code fences, so a code block is never cut in half unless
it alone is longer than `max_chars`, in which case it is cut at line
boundaries. Each row of a markdown table is its own chunk, written as
`Column: value` pairs from the table's header row, with check marks spelled
out as yes and no, so a feature-table row is indexed with the names of its
columns. Frontmatter is not indexed. Each chunk's text starts with the
page title and the section heading, so a chunk from deep in a page still says
which integration it is about. Chunks keep their 1-indexed line range, which
is how a retrieved chunk is matched to the claims it contains.

Answers are cached on disk, keyed by prompt version, model, question and the
retrieved chunks, so a rerun with an unchanged index makes no requests.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from factdrift.detect import normalize_value
from factdrift.extract import Usage
from factdrift.ingest import FENCE, Document, frontmatter_keys
from factdrift.questions import finding_id, mentions

PROMPT_VERSION = "answer-v1"
DEFAULT_MAX_CHARS = 2000
DEFAULT_TOP_K = 5
DEFAULT_MAX_TOKENS = 1000
DEFAULT_INDEX_DIR = Path(".cache/chroma")
DEFAULT_CACHE_DIR = Path(".cache/answers")
DEFAULT_EMBED_BATCH = 50  # Ollama drops the connection on batches of 500
DEFAULT_EMBED_MODEL = "default"  # Chroma's built-in all-MiniLM-L6-v2

_HEADING = re.compile(r"^#{1,6}\s+(.*\S)")
_TABLE_SEPARATOR = re.compile(r"^\s*\|?(\s*:?-+:?\s*\|)+\s*:?-*:?\s*$")
_LINK = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")
_LEADING_YES_NO =re.compile(r"\s*(yes|no)\b", re.IGNORECASE)

SYSTEM = """\
Answer the developer's question using only the documentation excerpts given. \
Be short and specific: state the value they asked for. If the excerpts do not \
contain the answer, say so.

Return the answer, and separately the single value it states (a version, \
name, URL, number, or yes/no), copied as written. Use an empty string for the \
value when the answer states none.
"""

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}, "value": {"type": "string"}},
    "required": ["answer", "value"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class Chunk:
    source_path: str
    line_start: int
    line_end: int
    text: str

    @property
    def id(self) -> str:
        return f"{self.source_path}:{self.line_start}-{self.line_end}"


def _title(document: Document, keys: dict[int, str]) -> str:
    lines = document.text.splitlines()
    for number, key in sorted(keys.items()):
        if key == "title":
            return lines[number - 1].split(":", 1)[1].strip().strip("\"'")
    return document.path


def _pieces(lines: list[tuple[int, str]], max_chars: int) -> list[list[tuple[int, str]]]:
    """Split one section at blank lines outside fences, then at line
    boundaries for a single block that is still too long."""
    blocks: list[list[tuple[int, str]]] = [[]]
    fence: str | None = None
    for number, line in lines:
        match = FENCE.match(line)
        if fence is None and match:
            fence = match.group(1)
        elif fence is not None and line.strip().startswith(fence):
            fence = None
        blocks[-1].append((number, line))
        if fence is None and not line.strip():
            blocks.append([])
    pieces: list[list[tuple[int, str]]] = [[]]
    size = 0
    for block in (b for b in blocks if b):
        block_size = sum(len(line) + 1 for _, line in block)
        if pieces[-1] and size + block_size > max_chars:
            pieces.append([])
            size = 0
        if block_size > max_chars:
            for number, line in block:
                if pieces[-1] and size + len(line) + 1 > max_chars:
                    pieces.append([])
                    size = 0
                pieces[-1].append((number, line))
                size += len(line) + 1
        else:
            pieces[-1].extend(block)
            size += block_size
    return [p for p in pieces if any(line.strip() for _, line in p)]


def chunk_document(document: Document, max_chars: int = DEFAULT_MAX_CHARS) -> list[Chunk]:
    keys = frontmatter_keys(document.text)
    body_start = max(keys, default=0) + 1
    if keys:
        body_start += 1  # the closing --- line
    title = _title(document, keys)
    sections: list[tuple[str, list[tuple[int, str]]]] = [("", [])]
    fence: str | None = None
    for number, line in enumerate(document.text.splitlines(), start=1):
        if number < body_start:
            continue
        match = FENCE.match(line)
        if fence is None and match:
            fence = match.group(1)
        elif fence is not None and line.strip().startswith(fence):
            fence = None
        heading = _HEADING.match(line) if fence is None else None
        if heading:
            sections.append((heading.group(1), []))
        sections[-1][1].append((number, line))
    chunks = []
    for heading, lines in sections:
        header = f"{title} > {heading}" if heading else title
        rest, rows = _split_tables(lines)
        # Text either side of a table is chunked separately, so no chunk's
        # line range spans a row it does not contain.
        runs: list[list[tuple[int, str]]] = []
        for number, line in rest:
            if not runs or number != runs[-1][-1][0] + 1:
                runs.append([])
            runs[-1].append((number, line))
        for piece in (p for run in runs for p in _pieces(run, max_chars)):
            body = "\n".join(line for _, line in piece)
            chunks.append(Chunk(document.path, piece[0][0], piece[-1][0], f"{header}\n\n{body}"))
        for number, columns, separator, row in rows:
            body = _linearize(columns, row)
            chunks.append(Chunk(document.path, number, number, f"{header}\n\n{body}"))
    return sorted(chunks, key=lambda c: c.line_start)


_GLYPHS = {"✅": "yes", "❌": "no"}


def _cells(row: str) -> list[str]:
    cells = [_LINK.sub(r"\1", cell).strip() for cell in row.strip().strip("|").split("|")]
    return [_GLYPHS.get(cell, cell) for cell in cells]


def _linearize(columns: str, row: str) -> str:
    """`Column: value` pairs with link targets dropped and check marks
    spelled out, so the embedding sees what each cell says and which column
    it belongs to rather than a row of bare symbols."""
    return "; ".join(f"{c}: {v}" for c, v in zip(_cells(columns), _cells(row)) if c or v)


def _split_tables(
    lines: list[tuple[int, str]],
) -> tuple[list[tuple[int, str]], list[tuple[int, str, str, str]]]:
    """Take markdown tables out of a section. Returns the remaining lines and
    one (line number, header row, separator row, data row) per table row, so
    every row is indexed with the header that names its columns."""
    rest: list[tuple[int, str]] = []
    rows: list[tuple[int, str, str, str]] = []
    fence: str | None = None
    i = 0
    while i < len(lines):
        number, line = lines[i]
        match = FENCE.match(line)
        if fence is None and match:
            fence = match.group(1)
        elif fence is not None and line.strip().startswith(fence):
            fence = None
        if (fence is None and line.lstrip().startswith("|") and i + 1 < len(lines)
                and _TABLE_SEPARATOR.match(lines[i + 1][1])):
            columns, separator = line.strip(), lines[i + 1][1].strip()
            i += 2
            start = i - 2
            while i < len(lines) and lines[i][1].lstrip().startswith("|"):
                rows.append((lines[i][0], columns, separator, lines[i][1].strip()))
                i += 1
            if i == start + 2:  # a header with no rows stays as plain text
                rest.extend(lines[start:i])
            continue
        rest.append((number, line))
        i += 1
    return rest, rows


def _digest(*parts: Any) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(json.dumps(part, sort_keys=True, ensure_ascii=False).encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


def _embedding_function(embed_model: str, ollama_url: str) -> Any:
    from chromadb.utils import embedding_functions

    if embed_model == DEFAULT_EMBED_MODEL:
        return embedding_functions.DefaultEmbeddingFunction()
    return embedding_functions.OllamaEmbeddingFunction(url=ollama_url, model_name=embed_model)


class Index:
    """A Chroma collection over the corpus chunks.

    The collection name is a hash of every chunk and the embedding model, so
    a changed corpus, chunk size or embedding model builds a new collection
    and an unchanged one is reused.
    """

    def __init__(
        self,
        chunks: list[Chunk],
        *,
        index_dir: Path = DEFAULT_INDEX_DIR,
        embed_model: str = DEFAULT_EMBED_MODEL,
        ollama_url: str = "http://localhost:11434",
        client: Any = None,
        embedding_function: Any = None,
        batch_size: int = DEFAULT_EMBED_BATCH,
    ) -> None:
        import chromadb

        self.chunks = {c.id: c for c in chunks}
        name = "corpus-" + _digest(embed_model, [(c.id, c.text) for c in chunks])[:32]
        self.client = client or chromadb.PersistentClient(path=str(index_dir))
        self.collection = self.client.get_or_create_collection(
            name,
            embedding_function=embedding_function
            or _embedding_function(embed_model, ollama_url),
            metadata={"hnsw:space": "cosine"},
        )
        self.built = 0
        if self.collection.count() != len(chunks):
            ids = list(self.chunks)
            for start in range(0, len(ids), batch_size):
                batch = [self.chunks[i] for i in ids[start : start + batch_size]]
                self.collection.upsert(
                    ids=[c.id for c in batch],
                    documents=[c.text for c in batch],
                    metadatas=[
                        {"source_path": c.source_path, "line_start": c.line_start,
                         "line_end": c.line_end}
                        for c in batch
                    ],
                )
            self.built = len(ids)

    def query(self, question: str, k: int) -> list[tuple[Chunk, float]]:
        result = self.collection.query(query_texts=[question], n_results=k)
        return [
            (self.chunks[i], float(d))
            for i, d in zip(result["ids"][0], result["distances"][0])
        ]


def user_message(question: str, chunks: Iterable[Chunk]) -> str:
    parts = [f"Question: {question}", "", "Excerpts:"]
    for number, chunk in enumerate(chunks, start=1):
        parts += ["", f"[{number}] {chunk.source_path}", chunk.text]
    return "\n".join(parts)


def _contains(chunk: Chunk, site: dict[str, Any]) -> bool:
    return (
        chunk.source_path == site["source_path"]
        and chunk.line_start <= site["line_start"]
        and site["line_end"] <= chunk.line_end
    )


def served_value(finding: dict[str, Any], answer: str, value: str) -> str | None:
    """Which of the finding's values the answer states.

    The model's own `value` field is compared first, by the same
    normalization detect uses, but only when the answer text states it. A
    leading yes or no answers a true/false finding. Otherwise the answer text
    is searched for each value's spellings. Returns the value key, "several"
    when the text names more than one, "other" when it states a value outside
    the finding, or None when it states nothing.
    """
    keys = finding["values"]
    stated = value.strip()
    if stated and not mentions(answer, stated):
        stated = ""  # the model named a value its answer does not state
    normalized = normalize_value(stated) if stated else None
    if normalized in keys:
        return normalized
    polarity = _LEADING_YES_NO.match(answer)
    if polarity and normalize_value(polarity.group(1)) in keys:
        return normalize_value(polarity.group(1))
    named = {
        key
        for key, sites in keys.items()
        if any(mentions(answer, s["value"]) for s in sites) or mentions(answer, key)
    }
    if len(named) == 1:
        return named.pop()
    if named:
        return "several"
    return "other" if normalized else None


@dataclass
class RetrievalResult:
    served: list[dict[str, Any]] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    api_calls: int = 0
    cached: int = 0


def run(
    findings: list[dict[str, Any]],
    questions: dict[str, str],
    index: Index,
    backend: Any,
    *,
    model: str,
    top_k: int = DEFAULT_TOP_K,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> RetrievalResult:
    """Retrieve and answer for each finding that has a question.

    `questions` maps finding_id to question text. A finding without one is
    recorded with served None and a reason.
    """
    result = RetrievalResult()
    for finding in findings:
        fid = finding_id(finding)
        record: dict[str, Any] = {"finding_id": fid, "question": questions.get(fid)}
        result.served.append(record)
        if not record["question"]:
            record.update(served=None, error="no question")
            continue
        hits = index.query(record["question"], top_k)
        chunks = [chunk for chunk, _ in hits]
        record["top_k"] = [
            {"source_path": c.source_path, "line_start": c.line_start,
             "line_end": c.line_end, "distance": round(d, 4)}
            for c, d in hits
        ]
        record["winner"] = chunks[0].source_path if chunks else None
        record["values_in_context"] = sorted(
            key for key, sites in finding["values"].items()
            if any(_contains(c, s) for c in chunks for s in sites)
        )
        path = cache_dir / f"{_digest(PROMPT_VERSION, model, record['question'], [(c.id, c.text) for c in chunks])}.json"
        payload = None
        if path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                result.cached += 1
            except json.JSONDecodeError:
                path.unlink()
        if payload is None:
            text, usage, stop_reason = backend.complete_json(
                system=SYSTEM,
                user=user_message(record["question"], chunks),
                schema=ANSWER_SCHEMA,
                model=model,
                max_tokens=max_tokens,
                schema_name="answer",
            )
            result.api_calls += 1
            result.usage.add(usage)
            try:
                payload = json.loads(text) if stop_reason in (None, "end_turn") else None
            except json.JSONDecodeError:
                payload = None
            if not (
                isinstance(payload, dict)
                and isinstance(payload.get("answer"), str)
                and isinstance(payload.get("value"), str)
            ):
                record.update(served=None, error=f"unusable answer (stop: {stop_reason})")
                continue
            cache_dir.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        record["answer"] = payload["answer"]
        record["answer_value"] = payload["value"]
        record["served"] = served_value(finding, payload["answer"], payload["value"])
        record["error"] = None
    return result


def render_served(findings: list[dict[str, Any]], served: list[dict[str, Any]]) -> str:
    """The report section: per finding, the value a user is served."""
    lines = [
        "## What retrieval serves",
        "",
        "| finding | question | top chunk | values in top k | served |",
        "| --- | --- | --- | --- | --- |",
    ]
    for finding, record in zip(findings, served):
        name = f"{finding['kind']} {finding['entity_id']}/{finding['attribute_id']}"
        question = (record.get("question") or "").replace("|", "\\|")
        top = record.get("top_k") or []
        top_chunk = f"{top[0]['source_path']}:{top[0]['line_start']}" if top else ""
        in_context = ", ".join(f"`{v}`" for v in record.get("values_in_context", []))
        served_cell = record.get("served") or record.get("error") or "none"
        if served_cell in finding["values"]:
            served_cell = f"`{served_cell}`"
        lines.append(f"| {name} | {question} | {top_chunk} | {in_context} | {served_cell} |")
    counts: dict[str, int] = {}
    for record in served:
        if record.get("served") in (None,):
            bucket = "no answer"
        elif record["served"] in ("several", "other"):
            bucket = record["served"]
        else:
            bucket = "one disputed value"
        counts[bucket] = counts.get(bucket, 0) + 1
    summary = ", ".join(f"{n} {k}" for k, n in sorted(counts.items()))
    return "\n".join(lines[:2] + [f"{len(served)} findings: {summary}.", ""] + lines[2:]) + "\n"
