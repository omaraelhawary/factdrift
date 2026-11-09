import json
from pathlib import Path
from types import SimpleNamespace

import anthropic
import httpx
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


class RaisingClient:
    """Stands in for a client whose requests fail."""

    def __init__(self, exc: Exception):
        self.exc = exc
        self.calls = 0
        self.messages = SimpleNamespace(stream=self._stream)

    def _stream(self, **kwargs):
        self.calls += 1
        raise self.exc


def _api_error(cls: type, status: int) -> Exception:
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(status, request=request, json={"error": {}})
    return cls("boom", response=response, body=None)


def test_a_transient_api_error_is_recorded_not_raised(sample, tmp_path):
    client = RaisingClient(_api_error(anthropic.RateLimitError, 429))
    result = extract_document(sample, client, cache_dir=tmp_path)
    assert result.claims == []
    assert result.error.startswith("RateLimitError")
    assert list(tmp_path.glob("*.json")) == []


def test_a_bad_key_stops_the_run(sample, tmp_path):
    client = RaisingClient(_api_error(anthropic.AuthenticationError, 401))
    with pytest.raises(anthropic.AuthenticationError):
        extract_document(sample, client, cache_dir=tmp_path)


def test_cost_uses_the_price_table():
    usage = Usage(input_tokens=1_000_000, output_tokens=1_000_000)
    assert cost(usage, "claude-opus-5", extract_mod.PRICES) == pytest.approx(30.0)
    assert cost(usage, "made-up-model", extract_mod.PRICES) is None


class FakeOllamaPost:
    """Stands in for the HTTP call to an Ollama server, recording payloads."""

    def __init__(self, response=None, exc: Exception | None = None):
        self.response = response
        self.exc = exc
        self.payloads = []

    def __call__(self, url, payload, timeout):
        self.payloads.append(payload)
        if self.exc is not None:
            raise self.exc
        return self.response


def _ollama_response(text: str = RESPONSE, done_reason: str = "stop") -> dict:
    return {
        "response": text,
        "done": True,
        "done_reason": done_reason,
        "prompt_eval_count": 900,
        "eval_count": 400,
    }


def test_ollama_responses_are_validated_like_anthropic_ones(sample, tmp_path):
    post = FakeOllamaPost(_ollama_response())
    backend = extract_mod.OllamaBackend(post=post)
    result = extract_document(sample, backend, model="qwen3:8b", cache_dir=tmp_path)
    assert result.error is None
    assert len(result.claims) == 2
    assert len(result.rejected) == 6
    assert result.usage.input_tokens == 900
    assert result.usage.output_tokens == 400


def test_ollama_request_is_schema_constrained_and_deterministic(sample, tmp_path):
    post = FakeOllamaPost(_ollama_response())
    backend = extract_mod.OllamaBackend(num_ctx=8192, post=post)
    extract_document(
        sample, backend, model="qwen3:8b", max_tokens=1234, cache_dir=tmp_path
    )
    payload = post.payloads[0]
    assert payload["format"] == extract_mod.CLAIMS_SCHEMA
    assert payload["options"] == {"temperature": 0, "num_ctx": 8192, "num_predict": 1234}
    assert payload["think"] is False


def test_an_ollama_rerun_makes_no_requests(sample, tmp_path):
    post = FakeOllamaPost(_ollama_response())
    backend = extract_mod.OllamaBackend(post=post)
    extract_document(sample, backend, model="qwen3:8b", cache_dir=tmp_path)
    second = extract_document(sample, backend, model="qwen3:8b", cache_dir=tmp_path)
    assert len(post.payloads) == 1
    assert second.cached is True


def test_an_ollama_length_stop_is_reported_and_not_cached(sample, tmp_path):
    post = FakeOllamaPost(_ollama_response(done_reason="length"))
    backend = extract_mod.OllamaBackend(post=post)
    result = extract_document(sample, backend, model="qwen3:8b", cache_dir=tmp_path)
    assert result.error.startswith("max_tokens")
    assert list(tmp_path.glob("*.json")) == []


def test_a_failed_ollama_request_is_recorded_not_raised(sample, tmp_path):
    post = FakeOllamaPost(exc=extract_mod.BackendRequestError("timed out after 900s"))
    backend = extract_mod.OllamaBackend(post=post)
    result = extract_document(sample, backend, model="qwen3:8b", cache_dir=tmp_path)
    assert result.claims == []
    assert "timed out" in result.error
    assert list(tmp_path.glob("*.json")) == []


def test_an_unreachable_ollama_server_stops_the_run(sample, tmp_path):
    post = FakeOllamaPost(exc=extract_mod.BackendUnavailable("connection refused"))
    backend = extract_mod.OllamaBackend(post=post)
    with pytest.raises(extract_mod.BackendUnavailable):
        extract_document(sample, backend, model="qwen3:8b", cache_dir=tmp_path)


def test_an_ollama_error_payload_is_recorded(sample, tmp_path):
    post = FakeOllamaPost({"error": "model requires more system memory"})
    backend = extract_mod.OllamaBackend(post=post)
    result = extract_document(sample, backend, model="qwen3:8b", cache_dir=tmp_path)
    assert "more system memory" in result.error


class FakeOpenRouterPost:
    """Stands in for the HTTP call to OpenRouter, recording requests."""

    def __init__(self, response=None, exc: Exception | None = None):
        self.response = response
        self.exc = exc
        self.requests = []

    def __call__(self, url, payload, timeout, headers=None):
        self.requests.append((url, payload, headers))
        if self.exc is not None:
            raise self.exc
        return self.response


def _openrouter_response(text: str = RESPONSE, finish_reason: str = "stop") -> dict:
    return {
        "choices": [{"message": {"content": text}, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 1100, "completion_tokens": 450},
    }


def test_openrouter_responses_are_validated_like_anthropic_ones(sample, tmp_path):
    post = FakeOpenRouterPost(_openrouter_response())
    backend = extract_mod.OpenRouterBackend("test-key", post=post)
    result = extract_document(sample, backend, model="m", cache_dir=tmp_path)
    assert result.error is None
    assert len(result.claims) == 2
    assert len(result.rejected) == 6
    assert (result.usage.input_tokens, result.usage.output_tokens) == (1100, 450)


def test_openrouter_request_is_schema_constrained_and_authenticated(sample, tmp_path):
    post = FakeOpenRouterPost(_openrouter_response())
    backend = extract_mod.OpenRouterBackend("test-key", post=post)
    extract_document(sample, backend, model="m", max_tokens=777, cache_dir=tmp_path)
    url, payload, headers = post.requests[0]
    assert url.endswith("/chat/completions")
    assert headers == {"Authorization": "Bearer test-key"}
    assert payload["response_format"]["json_schema"]["schema"] == extract_mod.CLAIMS_SCHEMA
    assert payload["response_format"]["json_schema"]["strict"] is True
    assert (payload["temperature"], payload["max_tokens"]) == (0, 777)


def test_an_openrouter_length_stop_is_reported_and_not_cached(sample, tmp_path):
    post = FakeOpenRouterPost(_openrouter_response(finish_reason="length"))
    backend = extract_mod.OpenRouterBackend("test-key", post=post)
    result = extract_document(sample, backend, model="m", cache_dir=tmp_path)
    assert result.error.startswith("max_tokens")
    assert list(tmp_path.glob("*.json")) == []


def test_a_response_without_choices_is_recorded(sample, tmp_path):
    post = FakeOpenRouterPost({"usage": {}})
    backend = extract_mod.OpenRouterBackend("test-key", post=post)
    result = extract_document(sample, backend, model="m", cache_dir=tmp_path)
    assert "no choices" in result.error


def test_running_out_of_openrouter_credit_stops_the_run(sample, tmp_path):
    post = FakeOpenRouterPost(exc=extract_mod.BackendUnavailable("out of credit"))
    backend = extract_mod.OpenRouterBackend("test-key", post=post)
    with pytest.raises(extract_mod.BackendUnavailable):
        extract_document(sample, backend, model="m", cache_dir=tmp_path)


def test_a_missing_openrouter_key_is_rejected_on_the_first_request(sample, tmp_path):
    post = FakeOpenRouterPost()
    backend = extract_mod.OpenRouterBackend("", post=post)
    with pytest.raises(extract_mod.BackendUnavailable):
        extract_document(sample, backend, model="m", cache_dir=tmp_path)
    assert post.requests == []
