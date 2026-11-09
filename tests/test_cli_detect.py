import json

from factdrift import cli
from factdrift.extract import Usage

from tests.test_detect import norm


def test_detect_writes_findings_and_report(tmp_path, capsys):
    claims = tmp_path / "normalized.json"
    claims.write_text(json.dumps({"claims": [
        norm("anthropic", "package_name", "langchain-anthropic"),
        norm("anthropic", "package_name", "anthropic-langchain"),
    ]}))
    out, report = tmp_path / "findings.json", tmp_path / "report.md"
    rc = cli.main(["detect", "--claims", str(claims), "--out", str(out), "--report", str(report)])
    assert rc == 0
    assert json.loads(out.read_text())[0]["kind"] == "contradiction"
    assert "contradiction (high)" in report.read_text()


def test_detect_refuses_unnormalized_claims(tmp_path):
    claims = tmp_path / "claims.json"
    claims.write_text(json.dumps({"claims": [{"entity": "a", "attribute": "b", "value": "c"}]}))
    assert cli.main(["detect", "--claims", str(claims)]) == 2


def test_normalize_defaults_to_openrouter_and_extract_does_not():
    parser = cli.build_parser()
    assert parser.parse_args(["normalize", "--claims", "a", "--out", "b"]).backend == "openrouter"
    assert parser.parse_args(["extract"]).backend == "anthropic"


def test_normalize_end_to_end_with_a_fake_backend(tmp_path, monkeypatch):
    class Fake:
        def complete_json(self, *, system, user, schema, model, max_tokens, schema_name="claims"):
            if schema_name == "aliases":
                items = json.loads(user.split("Items:\n", 1)[1])
                body = {"items": [{"id": i["id"], "entity_id": "anthropic", "attribute_id": "package_name"} for i in items]}
            else:
                body = {"attributes": [{"attribute_id": "package_name", "cardinality": "single", "reason": "r"}]}
            return json.dumps(body), Usage(input_tokens=1, output_tokens=1), "end_turn"

    monkeypatch.setattr(cli, "_make_backend", lambda args: (setattr(args, "model", "m"), Fake())[1])
    src = tmp_path / "claims.json"
    src.write_text(json.dumps({"claims": [
        {"entity": "ChatAnthropic", "attribute": "package", "value": "x", "raw_value": "x",
         "source_path": "src/oss/python/integrations/chat/a.mdx", "line_start": 1, "line_end": 1,
         "quote": "x", "context": "prose", "confidence": 1.0},
    ]}))
    dest = tmp_path / "normalized.json"
    rc = cli.main(["normalize", "--claims", str(src), "--out", str(dest), "--cache-dir", str(tmp_path / "c")])
    assert rc == 0
    payload = json.loads(dest.read_text())
    assert payload["claims"][0]["cardinality"] == "single"
    assert payload["cardinality"] == {"package_name": "single"}


def test_questions_runs_a_subset_and_projects_the_full_cost(tmp_path, monkeypatch, capsys):
    from tests.test_questions import BOOLEAN, FINDING, FakeBackend

    backend = FakeBackend("Where do I get a Together AI key?")
    monkeypatch.setattr(cli, "_make_backend", lambda args: backend)
    findings = tmp_path / "findings.json"
    findings.write_text(json.dumps([FINDING, BOOLEAN]))
    dest = tmp_path / "questions.json"
    rc = cli.main([
        "questions", "--backend", "openrouter", "--model", "claude-haiku-5-5",
        "--findings", str(findings), "--out", str(dest), "--limit", "1",
        "--cache-dir", str(tmp_path / "c"),
    ])
    assert rc == 0
    assert len(backend.requests) == 1
    out = capsys.readouterr().out
    assert "1 of 2 findings: 1 questions, 0 failed" in out
    assert "projected for all 2 findings: 600 in / 40 out" in out
    payload = json.loads(dest.read_text())
    assert payload["questions"][0]["question"] == "Where do I get a Together AI key?"


def test_questions_defaults_to_anthropic_at_low_effort():
    args = cli.build_parser().parse_args(["questions", "--findings", "a", "--out", "b"])
    assert (args.backend, args.effort) == ("anthropic", "low")
