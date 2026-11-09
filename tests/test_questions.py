import json

from factdrift.extract import Usage
from factdrift.questions import (
    check_question,
    finding_id,
    forbidden_values,
    generate_questions,
    user_message,
)


def site(value, path="src/oss/python/integrations/chat/together.mdx", line=10):
    return {"value": value, "source_path": path, "line_start": line, "line_end": line,
            "context": "prose", "quote": f"Sign up at {value} for a key."}


FINDING = {
    "kind": "contradiction",
    "severity": "high",
    "entity_id": "together_ai",
    "attribute_id": "signup_url",
    "values": {
        "https://api.together.xyz/": [site("https://api.together.xyz/")],
        "https://api.together.ai/": [site("https://api.together.ai/", line=40)],
    },
}

BOOLEAN = {
    "kind": "divergence",
    "severity": "high",
    "entity_id": "ollama",
    "attribute_id": "logprobs",
    "values": {
        "true": [site("Supported")],
        "false": [site("no", path="src/oss/javascript/integrations/chat/ollama.mdx")],
    },
}


class FakeBackend:
    def __init__(self, *answers, stop_reason="end_turn"):
        self.answers = list(answers)
        self.stop_reason = stop_reason
        self.requests = []

    def complete_json(self, *, system, user, schema, model, max_tokens, schema_name="claims"):
        self.requests.append(user)
        answer = self.answers.pop(0)
        text = answer if isinstance(answer, str) and answer.startswith("{") else json.dumps({"question": answer})
        return text, Usage(input_tokens=300, output_tokens=20), self.stop_reason


def run(backend, findings, tmp_path, **kw):
    return generate_questions(findings, backend, model="m", max_tokens=100,
                              cache_dir=tmp_path, **kw)


def test_a_question_is_generated_and_cached(tmp_path):
    q = "Where do I sign up for a Together AI API key?"
    first = run(FakeBackend(q), [FINDING], tmp_path)
    assert [r["question"] for r in first.questions] == [q]
    assert (first.api_calls, first.usage.input_tokens) == (1, 300)
    again = run(FakeBackend(), [FINDING], tmp_path)
    assert (again.api_calls, again.cached, again.questions[0]["question"]) == (0, 1, q)


def test_a_question_that_names_a_disputed_value_is_rejected_and_not_cached(tmp_path):
    bad = "Is the Together signup page at https://api.together.ai/?"
    result = run(FakeBackend(bad), [FINDING], tmp_path)
    assert result.questions[0]["question"] is None
    assert "names a disputed value" in result.questions[0]["error"]
    assert list(tmp_path.iterdir()) == []


def test_value_matching_ignores_case_but_not_substrings():
    finding = {**FINDING, "values": {"v3": [site("v3")], "4.8": [site("4.8")]}}
    assert check_question("Which API version does watsonx use, V3?", finding)
    assert check_question("Does watsonx 4.8 work?", finding)
    assert check_question("Which API version does watsonx use?", finding) is None
    assert check_question("Does watsonx 4.85 work?", finding) is None


def test_yes_no_values_may_appear_in_the_question():
    assert forbidden_values(BOOLEAN) == []
    assert check_question("Is logprobs supported by ChatOllama?", BOOLEAN) is None


def test_a_statement_is_rejected():
    assert check_question("Tell me the signup URL", FINDING) == "not a question"
    assert check_question("   ", FINDING) == "empty"


def test_the_prompt_carries_quotes_and_forbidden_values():
    message = user_message(FINDING)
    assert "together.mdx:40: Sign up at https://api.together.ai/" in message
    assert "Language: python" in message
    assert message.split("Forbidden values:")[1].count("- https://") == 2
    assert "Language: javascript and python" in user_message(BOOLEAN)


def test_bad_responses_are_recorded_not_raised(tmp_path):
    result = run(FakeBackend('{"wrong": 1}', '{"question": 5}', "nope"), [FINDING, BOOLEAN, FINDING], tmp_path)
    assert [r["error"] for r in result.questions] == [
        "response is not the question schema",
        "response is not the question schema",
        "rejected: not a question: nope",
    ]
    truncated = run(FakeBackend("Where?", stop_reason="max_tokens"), [FINDING], tmp_path)
    assert truncated.questions[0]["error"] == "stopped: max_tokens"
    assert list(tmp_path.iterdir()) == []


def test_the_cache_key_covers_model_effort_and_finding(tmp_path):
    q = "Where do I sign up for a Together AI API key?"
    run(FakeBackend(q), [FINDING], tmp_path)
    assert run(FakeBackend(q), [FINDING], tmp_path, effort="low").api_calls == 1
    changed = {**FINDING, "severity": "low"}
    assert finding_id(changed) != finding_id(FINDING)
    assert run(FakeBackend(q), [changed], tmp_path).api_calls == 1


def test_a_corrupt_cache_entry_is_discarded(tmp_path):
    q = "Where do I sign up for a Together AI API key?"
    run(FakeBackend(q), [FINDING], tmp_path)
    for path in tmp_path.iterdir():
        path.write_text("{", encoding="utf-8")
    assert run(FakeBackend(q), [FINDING], tmp_path).api_calls == 1


def test_a_value_at_the_end_of_a_sentence_is_mentioned():
    from factdrift.questions import mentions

    assert mentions("The default version is 4.8.", "4.8")
    assert not mentions("Use 4.8.1 instead.", "4.8")
    assert not mentions("Use 4.85 instead.", "4.8")
