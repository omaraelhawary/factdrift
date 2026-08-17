"""The claim extraction pass.

One API call per document. Results are cached to disk, keyed by file content,
prompt version and model, so a rerun over unchanged files makes no API calls.
Every claim is checked against the source before it is kept: a claim whose
quote is not in the lines it cites is dropped and recorded as a rejection.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import anthropic

from factdrift.ingest import Document
from factdrift.prompts.extraction import (
    CLAIMS_SCHEMA,
    PROMPT_VERSION,
    SYSTEM,
    user_message,
)

DEFAULT_MODEL = "claude-opus-5"
DEFAULT_EFFORT = "low"
DEFAULT_MAX_TOKENS = 32000
DEFAULT_CACHE_DIR = Path(".cache/extract")

# US dollars per million tokens, by model. Override on the command line when a
# price changes or a model is missing.
PRICES = {
    "claude-opus-5": (5.0, 25.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-sonnet-5": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
}

CONTEXTS = ("prose", "code")
_WHITESPACE = re.compile(r"\s+")

# Credentials and permissions are wrong for every document, not just this one,
# so retrying the rest of the corpus wastes time. Everything else that the API
# can raise is recorded per document and the run continues.
FATAL_API_ERRORS = (
    anthropic.AuthenticationError,
    anthropic.PermissionDeniedError,
    anthropic.NotFoundError,
)


@dataclass(frozen=True)
class Claim:
    entity: str
    attribute: str
    value: str
    raw_value: str
    source_path: str
    line_start: int
    line_end: int
    quote: str
    context: str
    confidence: float


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0

    def add(self, other: "Usage") -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cache_read_input_tokens += other.cache_read_input_tokens
        self.cache_creation_input_tokens += other.cache_creation_input_tokens


@dataclass
class ExtractResult:
    path: str
    claims: list[Claim] = field(default_factory=list)
    rejected: list[dict[str, Any]] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    stop_reason: str | None = None
    error: str | None = None
    cached: bool = False


class ExtractionError(RuntimeError):
    pass


def _normalize(text: str) -> str:
    return _WHITESPACE.sub(" ", text).strip()


def cache_key(document: Document, model: str, prompt_version: str, effort: str) -> str:
    """Everything that changes the claims a document produces.

    Effort is in the key because it changes output quality: without it, an
    effort sweep would serve the first run's claims and appear to do nothing.
    """
    digest = hashlib.sha256()
    for part in (document.text, prompt_version, model, effort):
        digest.update(part.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


def _cache_path(cache_dir: Path, key: str) -> Path:
    return cache_dir / f"{key}.json"


def _reject(raw: Any, reason: str) -> dict[str, Any]:
    return {"reason": reason, "claim": raw}


def validate_claims(
    raw_claims: Iterable[Any], document: Document
) -> tuple[list[Claim], list[dict[str, Any]]]:
    """Keep only claims that are supported by the lines they cite.

    Rejection reasons are kept rather than counted, because the false
    positives are the interesting output.
    """
    lines = document.text.splitlines()
    line_count = len(lines)
    kept: list[Claim] = []
    rejected: list[dict[str, Any]] = []

    for raw in raw_claims:
        if not isinstance(raw, dict):
            rejected.append(_reject(raw, "not_an_object"))
            continue

        missing = [
            key
            for key in (
                "entity",
                "attribute",
                "value",
                "raw_value",
                "line_start",
                "line_end",
                "quote",
                "context",
                "confidence",
            )
            if key not in raw
        ]
        if missing:
            rejected.append(_reject(raw, f"missing_fields:{','.join(missing)}"))
            continue

        text_fields = ("entity", "attribute", "value", "raw_value", "quote")
        if not all(isinstance(raw[key], str) for key in text_fields):
            rejected.append(_reject(raw, "non_string_field"))
            continue
        if not all(raw[key].strip() for key in ("entity", "attribute", "quote")):
            rejected.append(_reject(raw, "empty_field"))
            continue

        if raw["context"] not in CONTEXTS:
            rejected.append(_reject(raw, "bad_context"))
            continue

        try:
            start = int(raw["line_start"])
            end = int(raw["line_end"])
            confidence = float(raw["confidence"])
        except (TypeError, ValueError):
            rejected.append(_reject(raw, "unparseable_number"))
            continue

        if not 0.0 <= confidence <= 1.0:
            rejected.append(_reject(raw, "confidence_out_of_range"))
            continue
        if not 1 <= start <= end <= line_count:
            rejected.append(_reject(raw, "line_range_out_of_bounds"))
            continue

        quote = _normalize(raw["quote"])
        cited = _normalize("\n".join(lines[start - 1 : end]))
        if quote not in cited:
            reason = (
                "line_range_mismatch"
                if quote in _normalize(document.text)
                else "quote_not_found"
            )
            rejected.append(_reject(raw, reason))
            continue

        raw_value = _normalize(raw["raw_value"])
        value = _normalize(raw["value"])
        if raw_value not in quote and value not in quote:
            rejected.append(_reject(raw, "value_not_in_quote"))
            continue

        kept.append(
            Claim(
                entity=raw["entity"].strip(),
                attribute=raw["attribute"].strip(),
                value=raw["value"].strip(),
                raw_value=raw["raw_value"],
                source_path=document.path,
                line_start=start,
                line_end=end,
                quote=raw["quote"],
                context=raw["context"],
                confidence=confidence,
            )
        )

    return kept, rejected


def _response_text(message: Any) -> str:
    for block in message.content:
        if block.type == "text":
            return block.text
    return ""


def call_model(
    client: Any,
    document: Document,
    *,
    model: str,
    effort: str,
    max_tokens: int,
) -> tuple[str, Usage, str | None]:
    """Run one extraction request and return the raw response text and usage."""
    with client.messages.stream(
        model=model,
        max_tokens=max_tokens,
        system=SYSTEM,
        output_config={
            "effort": effort,
            "format": {"type": "json_schema", "schema": CLAIMS_SCHEMA},
        },
        messages=[{"role": "user", "content": user_message(document)}],
    ) as stream:
        message = stream.get_final_message()

    usage = Usage(
        input_tokens=getattr(message.usage, "input_tokens", 0) or 0,
        output_tokens=getattr(message.usage, "output_tokens", 0) or 0,
        cache_read_input_tokens=getattr(message.usage, "cache_read_input_tokens", 0)
        or 0,
        cache_creation_input_tokens=getattr(
            message.usage, "cache_creation_input_tokens", 0
        )
        or 0,
    )
    return _response_text(message), usage, message.stop_reason


def extract_document(
    document: Document,
    client: Any,
    *,
    model: str = DEFAULT_MODEL,
    effort: str = DEFAULT_EFFORT,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    prompt_version: str = PROMPT_VERSION,
) -> ExtractResult:
    """Extract claims from one document, serving from the cache when possible."""
    key = cache_key(document, model, prompt_version, effort)
    path = _cache_path(cache_dir, key)

    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            path.unlink()  # a truncated cache entry is worth less than a rerun
        else:
            return ExtractResult(
                path=document.path,
                claims=[Claim(**c) for c in payload["claims"]],
                rejected=payload.get("rejected", []),
                usage=Usage(**payload.get("usage", {})),
                stop_reason=payload.get("stop_reason"),
                error=payload.get("error"),
                cached=True,
            )

    try:
        text, usage, stop_reason = call_model(
            client, document, model=model, effort=effort, max_tokens=max_tokens
        )
    except FATAL_API_ERRORS:
        raise
    except anthropic.APIError as exc:
        return ExtractResult(
            path=document.path,
            error=f"{type(exc).__name__}: {exc}",
            cached=False,
        )

    result = ExtractResult(
        path=document.path, usage=usage, stop_reason=stop_reason, cached=False
    )
    if stop_reason == "refusal":
        result.error = "refusal"
    elif stop_reason == "max_tokens":
        result.error = "max_tokens: response truncated, raise --max-tokens"
    else:
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            result.error = f"unparseable_response: {exc}"
        else:
            raw_claims = payload.get("claims") if isinstance(payload, dict) else None
            if not isinstance(raw_claims, list):
                result.error = "response_missing_claims_array"
            else:
                result.claims, result.rejected = validate_claims(raw_claims, document)

    if result.error:
        # A failure is not an answer. Caching one would make the fix it asks
        # for ("raise --max-tokens") do nothing on the rerun.
        return result

    cache_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "path": document.path,
                "model": model,
                "effort": effort,
                "prompt_version": prompt_version,
                "claims": [asdict(c) for c in result.claims],
                "rejected": result.rejected,
                "usage": asdict(result.usage),
                "stop_reason": result.stop_reason,
                "error": result.error,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return result


def count_input_tokens(client: Any, document: Document, model: str) -> int:
    """Count the input tokens one extraction request would send. Free to call."""
    response = client.messages.count_tokens(
        model=model,
        system=SYSTEM,
        messages=[{"role": "user", "content": user_message(document)}],
    )
    return response.input_tokens


def cost(usage: Usage, model: str, prices: dict[str, tuple[float, float]]) -> float | None:
    """Dollar cost of `usage`, or None when the model has no price on file."""
    if model not in prices:
        return None
    price_in, price_out = prices[model]
    billed_input = usage.input_tokens + usage.cache_creation_input_tokens * 1.25
    billed_input += usage.cache_read_input_tokens * 0.1
    return (billed_input * price_in + usage.output_tokens * price_out) / 1_000_000
