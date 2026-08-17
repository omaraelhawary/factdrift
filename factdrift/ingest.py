"""Load markdown and mdx files into an internal document representation.

A document is a sequence of spans. Each span is either prose or a fenced code
block, and carries the 1-indexed line range it occupied in the source file.
Line numbers survive into claim records, so nothing here reflows text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

FENCE = re.compile(r"^\s*(`{3,}|~{3,})[ \t]*([^\s`~]*)")
TAG = re.compile(r"</?([A-Za-z][A-Za-z0-9.:-]*)(?:\s[^<>]*?)?/?>")
MDX_IMPORT = re.compile(r"^(import|export)\s")

SOURCE_SUFFIXES = (".md", ".mdx")


@dataclass(frozen=True)
class Span:
    """A contiguous run of source lines of a single kind."""

    kind: str  # "prose" or "code"
    text: str
    line_start: int  # 1-indexed, inclusive
    line_end: int  # 1-indexed, inclusive
    lang: str | None = None


@dataclass(frozen=True)
class Document:
    path: str  # relative to the corpus root
    language: str  # "python", "javascript", or "unknown"
    text: str  # the file verbatim
    spans: tuple[Span, ...]

    @property
    def line_count(self) -> int:
        return len(self.text.splitlines())


def _is_component(name: str) -> bool:
    """True for mdx components and html tags, false for placeholders.

    `<Tabs>` and `<img>` are markup. `<YOUR_API_KEY>` is a value the author
    wrote in prose, so it must survive stripping.
    """
    return name != name.upper()


def _strip_tags(line: str) -> str:
    return TAG.sub(lambda m: "" if _is_component(m.group(1)) else m.group(0), line)


def _closes_fence(line: str, marker: str) -> bool:
    stripped = line.strip()
    return (
        len(stripped) >= len(marker)
        and set(stripped) == {marker[0]}
    )


def _prose_span(lines: list[str], first_lineno: int) -> list[Span]:
    """Build at most one prose span from `lines`, which start at `first_lineno`.

    Component syntax is stripped and mdx import/export lines are blanked, both
    in place, so the line offsets of everything that survives are unchanged.
    """
    cleaned = [
        "" if MDX_IMPORT.match(line) else _strip_tags(line) for line in lines
    ]
    start = 0
    end = len(cleaned)
    while start < end and not cleaned[start].strip():
        start += 1
    while end > start and not cleaned[end - 1].strip():
        end -= 1
    if start == end:
        return []
    return [
        Span(
            kind="prose",
            text="\n".join(cleaned[start:end]),
            line_start=first_lineno + start,
            line_end=first_lineno + end - 1,
        )
    ]


def parse_spans(text: str) -> tuple[Span, ...]:
    """Split raw file text into prose and code spans.

    Frontmatter is kept as prose: it carries assertions (package name, class
    name) that the detector needs, and treating it as text means a malformed
    block is inert rather than a parse error.
    """
    lines = text.splitlines()
    n = len(lines)
    spans: list[Span] = []
    i = 0

    if n and lines[0].strip() == "---":
        for j in range(1, n):
            if lines[j].strip() == "---":
                spans.extend(_prose_span(lines[1:j], 2))
                i = j + 1
                break

    buffer: list[str] = []
    buffer_start = i + 1

    while i < n:
        match = FENCE.match(lines[i])
        if not match:
            buffer.append(lines[i])
            i += 1
            continue

        spans.extend(_prose_span(buffer, buffer_start))
        buffer = []

        marker = match.group(1)
        lang = match.group(2) or None
        body_start = i + 2  # 1-indexed line of the first content line
        i += 1
        body: list[str] = []
        while i < n and not _closes_fence(lines[i], marker):
            body.append(lines[i])
            i += 1
        if body:
            spans.append(
                Span(
                    kind="code",
                    text="\n".join(body),
                    line_start=body_start,
                    line_end=body_start + len(body) - 1,
                    lang=lang,
                )
            )
        i += 1  # step over the closing fence (or past EOF if unterminated)
        buffer_start = i + 1

    spans.extend(_prose_span(buffer, buffer_start))
    return tuple(spans)


def _language(relative_path: str) -> str:
    parts = relative_path.split("/")
    for candidate in ("python", "javascript"):
        if candidate in parts:
            return candidate
    return "unknown"


def load_document(path: Path, root: Path) -> Document:
    text = path.read_text(encoding="utf-8", errors="replace")
    relative = path.relative_to(root).as_posix()
    return Document(
        path=relative,
        language=_language(relative),
        text=text,
        spans=parse_spans(text),
    )


def find_sources(root: Path) -> list[Path]:
    """Every markdown and mdx file under `root`, skipping dot directories."""
    return sorted(
        p
        for p in root.rglob("*")
        if p.is_file()
        and p.suffix in SOURCE_SUFFIXES
        and not any(part.startswith(".") for part in p.relative_to(root).parts)
    )


def load_corpus(root: Path, limit: int | None = None) -> list[Document]:
    """Load every markdown and mdx file under `root`, in sorted path order."""
    paths = find_sources(root)
    if limit is not None:
        paths = paths[:limit]
    return [load_document(p, root) for p in paths]
