"""Normalization: canonical entity and attribute ids, and attribute cardinality.

Both passes run over distinct names, not over claims, so the cost grows with
the vocabulary of the corpus rather than its size.

1. Alias: map every distinct (entity, attribute) pair to a canonical
   entity_id and attribute_id, so `maxTokens` and `max_tokens`, or
   `ChatAnthropic` and `@langchain/anthropic`, land in the same cluster.
2. Cardinality: label every canonical attribute "single" (one correct value
   at a time) or "free" (a choice the user makes, such as an example
   temperature). Detection only treats disagreement on "single" attributes
   as a contradiction.

Model output is checked: an id that is missing or malformed falls back to a
slug of the original name and is counted, never raised.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

ALIAS_PROMPT_VERSION = "normalize-v1"  # unchanged since the first run; keeps its cache
CARDINALITY_PROMPT_VERSION = "cardinality-v2"
DEFAULT_BATCH_SIZE = 80
DEFAULT_MAX_TOKENS = 16000
DEFAULT_CACHE_DIR = Path(".cache/normalize")
CARDINALITIES = ("single", "free")

ALIAS_SYSTEM = """\
You normalize names extracted from the LangChain chat model integration docs, \
which exist in a Python and a JavaScript version.

Each item has an id, the entity and attribute as the extractor wrote them, an \
example value, and the language of the doc it came from. Return, for every id:

- entity_id: a canonical id for the real-world thing, lowercase snake_case. The \
same thing in the Python and JavaScript docs gets the same id. ChatAnthropic, \
@langchain/anthropic, langchain-anthropic and "Anthropic" are all `anthropic` \
when the claim is about that provider integration. Use a different id only \
for a genuinely different thing, such as `anthropic_bedrock` or a specific tool.
- attribute_id: a canonical, language-neutral property name, lowercase \
snake_case. max_tokens and maxTokens are both `max_tokens`. "model", \
"modelName" and "model in instantiation example" are all `model`. Drop notes \
about where the value appeared ("in example", "for streaming") unless they \
change what the property is, as in `default_model` versus `model`.

Keep ids short. Reuse an id from the known list whenever it names the same thing.\
"""

CARDINALITY_SYSTEM = """\
You classify properties of software integrations described in documentation.

single: at any one time the property has exactly one correct value for a \
given entity, so two documents disagreeing means at least one is wrong. \
Examples: package name, install command, class name, import path, environment \
variable name, API base URL, a default value, the minimum supported version.

free: the value is a choice the user makes, so different values in different \
examples are expected. Examples: temperature or max_tokens passed in an \
example, the prompt text, the model picked for an example, a timeout picked \
for an example. Also free: a property that is a set with many members, such \
as supported models, supported features, available methods or valid \
options, because different passages legitimately name different members.

Return one label per attribute id with a one-sentence reason.\
"""

ALIAS_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "entity_id": {"type": "string"},
                    "attribute_id": {"type": "string"},
                },
                "required": ["id", "entity_id", "attribute_id"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["items"],
    "additionalProperties": False,
}

CARDINALITY_SCHEMA = {
    "type": "object",
    "properties": {
        "attributes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "attribute_id": {"type": "string"},
                    "cardinality": {"type": "string", "enum": list(CARDINALITIES)},
                    "reason": {"type": "string"},
                },
                "required": ["attribute_id", "cardinality", "reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["attributes"],
    "additionalProperties": False,
}

_ID = re.compile(r"^[a-z0-9][a-z0-9_]*$")
_NON_ID = re.compile(r"[^a-z0-9]+")


def slug(name: str) -> str:
    """A canonical-looking id built from the raw name; used as the fallback."""
    return _NON_ID.sub("_", name.casefold()).strip("_") or "unknown"


def language_of(source_path: str) -> str:
    parts = source_path.split("/")
    for language in ("python", "javascript"):
        if language in parts:
            return language
    return "unknown"


@dataclass
class NormalizeResult:
    aliases: dict[tuple[str, str], tuple[str, str]] = field(default_factory=dict)
    cardinality: dict[str, str] = field(default_factory=dict)
    reasons: dict[str, str] = field(default_factory=dict)
    fallbacks: int = 0
    api_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


def distinct_pairs(claims: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per distinct (entity, attribute), with one example value."""
    seen: dict[tuple[str, str], dict[str, Any]] = {}
    for claim in claims:
        key = (claim["entity"], claim["attribute"])
        if key not in seen:
            seen[key] = {
                "entity": claim["entity"],
                "attribute": claim["attribute"],
                "example_value": claim["value"][:80],
                "language": language_of(claim["source_path"]),
            }
    return list(seen.values())


def _cache_key(*parts: Any) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(json.dumps(part, sort_keys=True).encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


def _cached_call(
    backend: Any,
    cache_dir: Path,
    key: str,
    result: NormalizeResult,
    **request: Any,
) -> Any:
    """Run one JSON request through the backend, or serve it from the cache.

    Returns the parsed object, or None when the response is unusable.
    Unusable responses are not cached.
    """
    path = cache_dir / f"{key}.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            path.unlink()
    text, usage, stop_reason = backend.complete_json(**request)
    result.api_calls += 1
    result.input_tokens += usage.input_tokens
    result.output_tokens += usage.output_tokens
    if stop_reason not in (None, "end_turn"):
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    cache_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def build_aliases(
    pairs: list[dict[str, Any]],
    backend: Any,
    result: NormalizeResult,
    *,
    model: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> None:
    """Fill result.aliases for every pair, batch by batch.

    Ids chosen in earlier batches are passed to later ones so the model can
    reuse them; this makes batches order-dependent, and the cache key
    includes the known ids for that reason.
    """
    known_entities: set[str] = set()
    known_attributes: set[str] = set()
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start : start + batch_size]
        items = [{"id": i, **row} for i, row in enumerate(batch)]
        known = {
            "entity_ids": sorted(known_entities),
            "attribute_ids": sorted(known_attributes),
        }
        user = (
            f"Known ids so far: {json.dumps(known)}\n\n"
            f"Items:\n{json.dumps(items, ensure_ascii=False)}"
        )
        key = _cache_key(ALIAS_PROMPT_VERSION, "alias", model, items, known)
        payload = _cached_call(
            backend,
            cache_dir,
            key,
            result,
            system=ALIAS_SYSTEM,
            user=user,
            schema=ALIAS_SCHEMA,
            schema_name="aliases",
            model=model,
            max_tokens=max_tokens,
        )
        answered: dict[int, tuple[str, str]] = {}
        if isinstance(payload, dict) and isinstance(payload.get("items"), list):
            for row in payload["items"]:
                if not isinstance(row, dict):
                    continue
                entity_id, attribute_id = row.get("entity_id"), row.get("attribute_id")
                if (
                    isinstance(row.get("id"), int)
                    and 0 <= row["id"] < len(batch)
                    and isinstance(entity_id, str)
                    and isinstance(attribute_id, str)
                    and _ID.match(entity_id)
                    and _ID.match(attribute_id)
                ):
                    answered[row["id"]] = (entity_id, attribute_id)
        for i, row in enumerate(batch):
            if i in answered:
                ids = answered[i]
            else:
                ids = (slug(row["entity"]), slug(row["attribute"]))
                result.fallbacks += 1
            result.aliases[(row["entity"], row["attribute"])] = ids
            known_entities.add(ids[0])
            known_attributes.add(ids[1])


def classify_attributes(
    attribute_ids: list[str],
    backend: Any,
    result: NormalizeResult,
    *,
    model: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> None:
    """Fill result.cardinality. An attribute the model skips is treated as
    "free", the label that cannot create a false contradiction."""
    for start in range(0, len(attribute_ids), batch_size):
        batch = attribute_ids[start : start + batch_size]
        key = _cache_key(CARDINALITY_PROMPT_VERSION, "cardinality", model, batch)
        payload = _cached_call(
            backend,
            cache_dir,
            key,
            result,
            system=CARDINALITY_SYSTEM,
            user=f"Attribute ids:\n{json.dumps(batch)}",
            schema=CARDINALITY_SCHEMA,
            schema_name="cardinality",
            model=model,
            max_tokens=max_tokens,
        )
        if isinstance(payload, dict) and isinstance(payload.get("attributes"), list):
            for row in payload["attributes"]:
                if (
                    isinstance(row, dict)
                    and row.get("attribute_id") in batch
                    and row.get("cardinality") in CARDINALITIES
                ):
                    result.cardinality[row["attribute_id"]] = row["cardinality"]
                    result.reasons[row["attribute_id"]] = str(row.get("reason", ""))
        for attribute_id in batch:
            if attribute_id not in result.cardinality:
                result.cardinality[attribute_id] = "free"
                result.reasons[attribute_id] = "not classified; defaulted to free"
                result.fallbacks += 1


def normalize_claims(
    claims: list[dict[str, Any]],
    backend: Any,
    *,
    model: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> tuple[list[dict[str, Any]], NormalizeResult]:
    """Return claims annotated with entity_id, attribute_id, language and
    cardinality, plus the run summary."""
    result = NormalizeResult()
    pairs = distinct_pairs(claims)
    build_aliases(
        pairs, backend, result,
        model=model, batch_size=batch_size, max_tokens=max_tokens, cache_dir=cache_dir,
    )
    attribute_ids = sorted({ids[1] for ids in result.aliases.values()})
    classify_attributes(
        attribute_ids, backend, result,
        model=model, batch_size=batch_size, max_tokens=max_tokens, cache_dir=cache_dir,
    )
    annotated = []
    for claim in claims:
        entity_id, attribute_id = result.aliases[(claim["entity"], claim["attribute"])]
        annotated.append(
            {
                **claim,
                "entity_id": entity_id,
                "attribute_id": attribute_id,
                "language": language_of(claim["source_path"]),
                "cardinality": result.cardinality[attribute_id],
            }
        )
    return annotated, result
