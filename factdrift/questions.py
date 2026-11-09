"""Question generation, pipeline step 7.

One request per finding asks for the question a developer would type when
they need the disputed fact, without naming any of the disputed values.
Questions are cached on disk, keyed by the finding, prompt version, model and
effort, so a rerun over unchanged findings makes no requests. A question that
names a disputed value, or is not a question, is rejected and not cached.
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
from factdrift.normalize import language_of

PROMPT_VERSION = "questions-v1"
DEFAULT_MODEL = "claude-haiku-5-5"
DEFAULT_MAX_TOKENS = 2000
DEFAULT_CACHE_DIR = Path(".cache/questions")

# Values that only say yes or no. A question about them has to use the word
# ("Does X support Y?"), so naming them reveals nothing.
_POLARITY = {"true", "false"}

SYSTEM = """\
You write the question a developer would type into a documentation search box \
or a docs chat assistant when they need one fact.

You are shown one fact that the documentation states more than once, with the \
quotes that state it. Write one question whose answer is that fact.

Rules:
- Write it the way a developer would ask it: short, specific, plain. Name the \
library, class, provider or feature the quotes are about, so the question \
makes sense on its own.
- Never include any of the values listed as forbidden, in any spelling. The \
question must not hint at which answer is right.
- Ask about the fact itself, not about the documentation or about a \
disagreement.
- End with a question mark.
"""

QUESTION_SCHEMA = {
    "type": "object",
    "properties": {"question": {"type": "string"}},
    "required": ["question"],
    "additionalProperties": False,
}


def finding_id(finding: dict[str, Any]) -> str:
    """Stable id for a finding: a hash of everything detect reported."""
    canonical = json.dumps(finding, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def forbidden_values(finding: dict[str, Any]) -> list[str]:
    """Every spelling of a disputed value, except plain yes/no values."""
    spellings = set(finding["values"])
    for sites in finding["values"].values():
        spellings.update(site["value"] for site in sites)
    return sorted(
        v for v in spellings if v.strip() and normalize_value(v) not in _POLARITY
    )


def _languages(finding: dict[str, Any]) -> list[str]:
    paths = (s["source_path"] for sites in finding["values"].values() for s in sites)
    return sorted({language_of(p) for p in paths})


def user_message(finding: dict[str, Any]) -> str:
    lines = [
        f"Subject: {finding['entity_id']}",
        f"Fact: {finding['attribute_id']}",
        f"Language: {' and '.join(_languages(finding))}",
        "",
        "Quotes:",
    ]
    for sites in finding["values"].values():
        for site in sites:
            lines.append(f"- {site['source_path']}:{site['line_start']}: {site['quote']}")
    lines += ["", "Forbidden values:"]
    lines += [f"- {v}" for v in forbidden_values(finding)] or ["- (none)"]
    return "\n".join(lines)


def check_question(question: str, finding: dict[str, Any]) -> str | None:
    """Why `question` is unusable for `finding`, or None when it is usable."""
    text = question.strip()
    if not text:
        return "empty"
    if not text.endswith("?"):
        return "not a question"
    for value in forbidden_values(finding):
        if mentions(text, value):
            return f"names a disputed value: {value}"
    return None


def mentions(text: str, value: str) -> bool:
    """True when `value` appears in `text` as a whole token, ignoring case.

    `4.8` does not match inside `4.85`, and `v3` does not match inside `v30`.
    """
    pattern = r"(?<![\w.])" + re.escape(value.strip().casefold()) + r"(?!\w|\.\w)"
    return re.search(pattern, text.casefold()) is not None


@dataclass
class QuestionResult:
    questions: list[dict[str, Any]] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    api_calls: int = 0
    cached: int = 0


def _cache_key(finding: dict[str, Any], model: str, effort: str | None) -> str:
    digest = hashlib.sha256()
    for part in (PROMPT_VERSION, model, effort, finding):
        digest.update(json.dumps(part, sort_keys=True).encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


def generate_questions(
    findings: Iterable[dict[str, Any]],
    backend: Any,
    *,
    model: str,
    max_tokens: int,
    cache_dir: Path,
    effort: str | None = None,
) -> QuestionResult:
    """One question per finding, from the cache or from `backend`.

    `effort` is whatever setting the backend was built with; it only enters
    the cache key, so entries from different settings never mix.
    """
    result = QuestionResult()
    for finding in findings:
        record = {
            "finding_id": finding_id(finding),
            "kind": finding["kind"],
            "entity_id": finding["entity_id"],
            "attribute_id": finding["attribute_id"],
            "question": None,
            "error": None,
        }
        result.questions.append(record)
        path = cache_dir / f"{_cache_key(finding, model, effort)}.json"
        if path.exists():
            try:
                record["question"] = json.loads(path.read_text(encoding="utf-8"))[
                    "question"
                ]
                result.cached += 1
                continue
            except (json.JSONDecodeError, KeyError, TypeError):
                path.unlink()

        text, usage, stop_reason = backend.complete_json(
            system=SYSTEM,
            user=user_message(finding),
            schema=QUESTION_SCHEMA,
            model=model,
            max_tokens=max_tokens,
            schema_name="question",
        )
        result.api_calls += 1
        result.usage.add(usage)
        if stop_reason not in (None, "end_turn"):
            record["error"] = f"stopped: {stop_reason}"
            continue
        try:
            question = json.loads(text)["question"]
        except (json.JSONDecodeError, KeyError, TypeError):
            record["error"] = "response is not the question schema"
            continue
        if not isinstance(question, str):
            record["error"] = "response is not the question schema"
            continue
        reason = check_question(question, finding)
        if reason:
            record["error"] = f"rejected: {reason}: {question}"
            continue
        record["question"] = question.strip()
        cache_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"question": record["question"]}), encoding="utf-8")
    return result
