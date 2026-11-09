import json

from factdrift.extract import Usage
from factdrift.normalize import distinct_pairs, language_of, normalize_claims, slug


def claim(entity, attribute, value, path="src/oss/python/integrations/chat/a.mdx", context="prose"):
    return {
        "entity": entity,
        "attribute": attribute,
        "value": value,
        "raw_value": value,
        "source_path": path,
        "line_start": 1,
        "line_end": 1,
        "quote": value,
        "context": context,
        "confidence": 0.9,
    }


JS = "src/oss/javascript/integrations/chat/a.mdx"


class FakeBackend:
    """Answers alias and cardinality requests from fixed tables."""

    def __init__(self, aliases, cardinality, drop=()):
        self.aliases = aliases
        self.cardinality = cardinality
        self.drop = set(drop)
        self.calls = []

    def complete_json(self, *, system, user, schema, model, max_tokens, schema_name="claims"):
        self.calls.append(schema_name)
        if schema_name == "aliases":
            items = json.loads(user.split("Items:\n", 1)[1])
            rows = [
                {"id": it["id"], **dict(zip(("entity_id", "attribute_id"), self.aliases[(it["entity"], it["attribute"])]))}
                for it in items
                if (it["entity"], it["attribute"]) not in self.drop
            ]
            payload = {"items": rows}
        else:
            ids = json.loads(user.split("\n", 1)[1])
            payload = {
                "attributes": [
                    {"attribute_id": a, "cardinality": self.cardinality[a], "reason": "test"}
                    for a in ids
                    if a in self.cardinality
                ]
            }
        return json.dumps(payload), Usage(input_tokens=10, output_tokens=5), "end_turn"


CLAIMS = [
    claim("ChatAnthropic", "max_tokens", "4096"),
    claim("@langchain/anthropic", "maxTokens", "1024", path=JS),
    claim("ChatAnthropic", "package", "langchain-anthropic"),
]
ALIASES = {
    ("ChatAnthropic", "max_tokens"): ("anthropic", "max_tokens"),
    ("@langchain/anthropic", "maxTokens"): ("anthropic", "max_tokens"),
    ("ChatAnthropic", "package"): ("anthropic", "package_name"),
}


def test_slug_and_language():
    assert slug("@langchain/Anthropic") == "langchain_anthropic"
    assert slug("!!") == "unknown"
    assert language_of(JS) == "javascript"
    assert language_of("README.md") == "unknown"


def test_distinct_pairs_dedupes():
    pairs = distinct_pairs(CLAIMS + [claim("ChatAnthropic", "max_tokens", "5000")])
    assert len(pairs) == 3


def test_normalize_maps_aliases_across_languages(tmp_path):
    backend = FakeBackend(ALIASES, {"max_tokens": "free", "package_name": "single"})
    out, result = normalize_claims(CLAIMS, backend, model="m", cache_dir=tmp_path)
    assert {c["attribute_id"] for c in out[:2]} == {"max_tokens"}
    assert out[1]["language"] == "javascript"
    assert out[0]["cardinality"] == "free"
    assert out[2]["cardinality"] == "single"
    assert result.fallbacks == 0
    assert result.api_calls == 2


def test_normalize_uses_the_cache(tmp_path):
    backend = FakeBackend(ALIASES, {"max_tokens": "free", "package_name": "single"})
    normalize_claims(CLAIMS, backend, model="m", cache_dir=tmp_path)
    again = FakeBackend(ALIASES, {"max_tokens": "free", "package_name": "single"})
    _, result = normalize_claims(CLAIMS, again, model="m", cache_dir=tmp_path)
    assert again.calls == []
    assert result.api_calls == 0


def test_missing_answers_fall_back_safely(tmp_path):
    backend = FakeBackend(
        ALIASES, {"max_tokens": "free"}, drop={("ChatAnthropic", "package")}
    )
    out, result = normalize_claims(CLAIMS, backend, model="m", cache_dir=tmp_path)
    assert out[2]["entity_id"] == "chatanthropic"
    assert out[2]["attribute_id"] == "package"
    assert out[2]["cardinality"] == "free"
    assert result.fallbacks == 2


def test_malformed_ids_are_rejected(tmp_path):
    bad = dict(ALIASES)
    bad[("ChatAnthropic", "package")] = ("Anthropic Inc", "package_name")
    backend = FakeBackend(bad, {"max_tokens": "free", "package_name": "single"})
    out, result = normalize_claims(CLAIMS, backend, model="m", cache_dir=tmp_path)
    assert out[2]["entity_id"] == "chatanthropic"
    assert result.fallbacks >= 1


# Two JavaScript pages whose frontmatter the extractor attributed to the same
# raw entity, "document". The alias pass sees one (entity, attribute) pair,
# with azure.mdx's value as the example, and maps both pages to the Azure
# integration. Reproduces the azure_chat_openai/title finding on the corpus.
AZURE = "src/oss/javascript/integrations/chat/azure.mdx"
INDEX = "src/oss/javascript/integrations/chat/index.mdx"
PAGE_TEXT = {
    AZURE: "---\ntitle: AzureChatOpenAI integration\nnpm: \"@langchain/openai\"\n---\nBody.\n",
    INDEX: "---\ntitle: \"Chat model integrations\"\n---\nBody.\n",
}


def frontmatter_claim(path, attribute, value, line):
    return {**claim("document", attribute, value, path=path), "line_start": line, "line_end": line}


def test_frontmatter_page_metadata_stays_on_its_page(tmp_path):
    from factdrift.detect import detect
    from factdrift.ingest import frontmatter_keys

    claims = [
        frontmatter_claim(AZURE, "title", "AzureChatOpenAI integration", 2),
        frontmatter_claim(INDEX, "title", "Chat model integrations", 2),
        frontmatter_claim(AZURE, "npm", "@langchain/openai", 3),
    ]
    aliases = {
        ("document", "title"): ("azure_chat_openai", "title"),
        ("document", "npm"): ("azure_chat_openai", "npm_package"),
    }
    backend = FakeBackend(aliases, {"title": "single", "npm_package": "single"})
    out, _ = normalize_claims(
        claims,
        backend,
        model="m",
        cache_dir=tmp_path,
        frontmatter={path: frontmatter_keys(text) for path, text in PAGE_TEXT.items()},
    )
    assert out[0]["entity_id"] != out[1]["entity_id"]
    # The package key is a fact about the integration, not the page.
    assert out[2]["entity_id"] == "azure_chat_openai"
    assert detect(out) == []


def test_a_response_object_reports_on_the_call_above_it(tmp_path):
    def at(c, line):
        return {**c, "line_start": line, "line_end": line, "context": "code"}

    claims = [
        at(claim("ChatAnthropic", "model", "claude-sonnet-4-6", path=JS), 263),
        at(claim("custom headers output AIMessage", "model", "claude-3-sonnet-20240229", path=JS), 283),
        # Too far below any call, and on a page with no call: left alone.
        at(claim("AIMessage", "model", "m", path=JS), 900),
        at(claim("AIMessage", "model", "m"), 10),
        at(claim("AIMessage", "content", "hi", path=JS), 284),
    ]
    aliases = {
        ("ChatAnthropic", "model"): ("anthropic", "model"),
        ("custom headers output AIMessage", "model"): ("ai_message", "model"),
        ("AIMessage", "model"): ("ai_message", "model"),
        ("AIMessage", "content"): ("ai_message", "content"),
    }
    backend = FakeBackend(aliases, {"model": "free", "content": "free"})
    annotated, _ = normalize_claims(claims, backend, model="m", cache_dir=tmp_path)
    assert [(c["entity_id"], c.get("reports_on_line")) for c in annotated] == [
        ("anthropic", None),
        ("anthropic", 263),
        ("ai_message", None),
        ("ai_message", None),
        ("ai_message", None),
    ]
