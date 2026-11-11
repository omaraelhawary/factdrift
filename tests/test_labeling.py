import csv

from factdrift.cli import main
from factdrift.detect import candidates, detect
from factdrift.labeling import rows, write_csv
from factdrift.score import load_labels, score
from tests.test_detect import norm


def test_candidates_include_free_attributes_detect_skips():
    claims = [norm("groq", "temperature", "0", cardinality="free", line=1),
              norm("groq", "temperature", "0.7", cardinality="free", line=2)]
    assert detect(claims) == []
    (c,) = candidates(claims)
    assert (c["scope"], c["cardinality"], sorted(c["values"])) == ("python", "free", ["0", "0.7"])


def test_candidates_cover_single_attributes_and_twin_pages():
    claims = [norm("groq", "env_var", "GROQ_API_KEY", line=1),
              norm("groq", "env_var", "GROQ_TOKEN", line=2),
              norm("groq", "model", "a", cardinality="free", line=3),
              norm("groq", "model", "b", cardinality="free", language="javascript", line=3)]
    scopes = {(c["scope"], c["attribute_id"]) for c in candidates(claims)}
    assert scopes == {("python", "env_var"), ("twin x.mdx", "model")}


def test_agreeing_values_are_not_candidates():
    claims = [norm("groq", "env_var", "GROQ_API_KEY", line=1),
              norm("groq", "env_var", "`GROQ_API_KEY`", line=2)]
    assert candidates(claims) == []


def test_labeling_rows_round_trip_through_the_scorer(tmp_path):
    claims = [norm("groq", "env_var", "GROQ_API_KEY", line=1),
              norm("groq", "env_var", "GROQ_TOKEN", line=2),
              norm("groq", "temperature", "0", cardinality="free", line=3),
              norm("groq", "temperature", "0.7", cardinality="free", line=4)]
    findings = [f.as_dict() for f in detect(claims)]
    table = rows(candidates(claims), findings)
    by_attr = {r["attribute_id"]: r for r in table}
    assert by_attr["env_var"]["detector"] == "contradiction high"
    assert by_attr["temperature"]["detector"] == ""
    assert "python/x.mdx:2 [prose] GROQ_TOKEN" in by_attr["env_var"]["values"]
    # A reviewer marks the free example real: detect missed it.
    by_attr["env_var"]["label"] = "real"
    by_attr["temperature"]["label"] = "real"
    by_attr["env_var"]["label_2"] = "real"
    by_attr["temperature"]["label_2"] = "not-real"
    path = tmp_path / "gold.csv"
    write_csv(table, path)
    result = score(findings, load_labels(path))
    assert (result.precision, result.recall) == (1.0, 0.5)
    assert score(findings, load_labels(path, "label_2", "note_2")).recall == 1.0


def test_candidates_command_defaults_to_pages_in_both_languages(tmp_path, capsys):
    import json
    claims = [norm("groq", "env_var", "A", line=1), norm("groq", "env_var", "B", line=2)]
    other = dict(norm("groq", "env_var", "A", line=1, language="javascript"))
    lonely = {**norm("groq", "env_var", "Z", line=9),
              "source_path": "src/oss/python/integrations/chat/only.mdx"}
    path = tmp_path / "normalized.json"
    path.write_text(json.dumps({"claims": claims + [other, lonely]}))
    out = tmp_path / "gold.csv"
    assert main(["candidates", "--claims", str(path), "--out", str(out)]) == 0
    (row,) = list(csv.DictReader(out.open()))
    assert "only.mdx" not in row["sites"] and row["label"] == ""


def test_detector_column_needs_a_shared_site_not_just_the_same_attribute():
    python = [norm("groq", "temperature", "0", cardinality="free", line=1),
              norm("groq", "temperature", "0.7", cardinality="free", line=2)]
    elsewhere = {"kind": "contradiction", "severity": "low", "entity_id": "groq",
                 "attribute_id": "temperature",
                 "values": {"1": [norm("groq", "temperature", "1", language="javascript", line=7)]}}
    (row,) = rows(candidates(python), [elsewhere])
    assert row["detector"] == ""
