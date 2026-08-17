import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from factdrift import extract as extract_mod
from factdrift.extract import Usage, cache_key, cost, extract_document, validate_claims
from factdrift.ingest import Document, load_document

FIXTURES = Path(__file__).parent / "fixtures"
RESPONSE = (FIXTURES / "response_sample.json").read_text()


@pytest.fixture
def sample() -> Document:
    return load_document(FIXTURES / "sample.mdx", FIXTURES)


@pytest.fixture
def validated(sample):
    return validate_claims(json.loads(RESPONSE)["claims"], sample)


def reasons(rejected) -> set[str]:
    return {item["reason"] for item in rejected}


def test_supported_claims_survive(validated, sample):
    kept, _ = validated
    assert [(c.attribute, c.value) for c in kept] == [
        ("pypi package", "langchain-example"),
        ("max_tokens", "5000"),
    ]
    assert all(c.source_path == sample.path for c in kept)


def test_a_fabricated_quote_is_rejected(validated):
    _, rejected = validated
    fabricated = [
        r for r in rejected if r["claim"]["value"] == "claude-opus-5"
    ]
    assert len(fabricated) == 1
    assert fabricated[0]["reason"] == "quote_not_found"


def test_a_quote_from_the_wrong_lines_is_rejected(validated):
    _, rejected = validated
    misplaced = [r for r in rejected if r["claim"]["attribute"] == "import"]
    assert misplaced[0]["reason"] == "line_range_mismatch"


def test_a_value_missing_from_its_quote_is_rejected(validated):
    _, rejected = validated
    assert "value_not_in_quote" in reasons(rejected)


def test_malformed_claims_are_rejected_not_raised(validated):
    _, rejected = validated
    assert reasons(rejected) >= {
        "line_range_out_of_bounds",
        "bad_context",
        "confidence_out_of_range",
    }


def test_missing_fields_are_rejected(sample):
    kept, rejected = validate_claims([{"entity": "x"}, "not an object", 7], sample)
    assert kept == []
    assert reasons(rejected) == {"missing_fields:attribute,value,raw_value,"
                                 "line_start,line_end,quote,context,confidence",
                                 "not_an_object"}


def test_quote_matching_ignores_whitespace_differences(sample):
    claim = {
        "entity": "ChatExample",
        "attribute": "pypi package",
        "value": "langchain-example",
        "raw_value": "langchain-example",
        "line_start": 4,
        "line_end": 4,
        "quote": "pypi:   langchain-example",
        "context": "prose",
        "confidence": 0.9,
    }
    kept, rejected = validate_claims([claim], sample)
    assert len(kept) == 1 and not rejected


class FakeClient:
    """Stands in for anthropic.Anthropic, counting requests."""

    def __init__(self, text: str = RESPONSE, stop_reason: str = "end_turn"):
        self.text = text
        self.stop_reason = stop_reason
        self.calls = 0
        self.messages = SimpleNamespace(stream=self._stream)

    def _stream(self, **kwargs):
        self.calls += 1
        client = self

        class Stream:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def get_final_message(self):
                return SimpleNamespace(
                    content=[SimpleNamespace(type="text", text=client.text)],
                    stop_reason=client.stop_reason,
                    usage=SimpleNamespace(
                        input_tokens=1000,
                        output_tokens=500,
                        cache_read_input_tokens=0,
                        cache_creation_input_tokens=0,
                    ),
                )

        return Stream()


def test_extraction_validates_the_response(sample, tmp_path):
    client = FakeClient()
    result = extract_document(sample, client, cache_dir=tmp_path)
    assert client.calls == 1
    assert result.error is None
    assert len(result.claims) == 2
    assert len(result.rejected) == 6
    assert result.usage.input_tokens == 1000


def test_a_rerun_over_unchanged_files_makes_no_api_calls(sample, tmp_path):
    client = FakeClient()
    first = extract_document(sample, client, cache_dir=tmp_path)
    second = extract_document(sample, client, cache_dir=tmp_path)
    assert client.calls == 1
    assert second.cached is True
    assert second.claims == first.claims
    assert second.rejected == first.rejected


def test_a_new_prompt_version_invalidates_the_cache(sample, tmp_path):
    client = FakeClient()
    extract_document(sample, client, cache_dir=tmp_path)
    extract_document(sample, client, cache_dir=tmp_path, prompt_version="v2")
    assert client.calls == 2


def test_cache_key_covers_content_prompt_model_and_effort(sample):
    base = cache_key(sample, "m", "v1", "low")
    changed = Document(
        path=sample.path,
        language=sample.language,
        text=sample.text + "\n",
        spans=sample.spans,
    )
    assert cache_key(changed, "m", "v1", "low") != base
    assert cache_key(sample, "m", "v2", "low") != base
    assert cache_key(sample, "other", "v1", "low") != base
    assert cache_key(sample, "m", "v1", "high") != base


def test_a_new_effort_level_invalidates_the_cache(sample, tmp_path):
    client = FakeClient()
    extract_document(sample, client, cache_dir=tmp_path, effort="low")
    extract_document(sample, client, cache_dir=tmp_path, effort="high")
    assert client.calls == 2


def test_a_failed_extraction_is_not_cached(sample, tmp_path):
    truncated = FakeClient(text='{"claims": [', stop_reason="max_tokens")
    assert extract_document(sample, truncated, cache_dir=tmp_path).error
    assert list(tmp_path.glob("*.json")) == []

    retry = FakeClient()
    result = extract_document(sample, retry, cache_dir=tmp_path)
    assert retry.calls == 1
    assert result.error is None
    assert len(result.claims) == 2


def test_a_truncated_response_is_reported_not_raised(sample, tmp_path):
    client = FakeClient(text='{"claims": [', stop_reason="max_tokens")
    result = extract_document(sample, client, cache_dir=tmp_path)
    assert result.claims == []
    assert "max_tokens" in result.error


def test_unparseable_json_is_reported_not_raised(sample, tmp_path):
    client = FakeClient(text="I could not find any claims.")
    result = extract_document(sample, client, cache_dir=tmp_path)
    assert result.claims == []
    assert result.error.startswith("unparseable_response")


def test_a_corrupt_cache_entry_is_discarded(sample, tmp_path):
    client = FakeClient()
    extract_document(sample, client, cache_dir=tmp_path)
    for entry in tmp_path.glob("*.json"):
        entry.write_text("{ truncated")
    result = extract_document(sample, client, cache_dir=tmp_path)
    assert client.calls == 2
    assert result.cached is False


def test_cost_uses_the_price_table():
    usage = Usage(input_tokens=1_000_000, output_tokens=1_000_000)
    assert cost(usage, "claude-opus-5", extract_mod.PRICES) == pytest.approx(30.0)
    assert cost(usage, "made-up-model", extract_mod.PRICES) is None
