import json

import pytest

from factdrift.cli import _replace_section, main
from factdrift.score import LabelError, load_labels, render, score

PY = "src/oss/python/integrations/chat/groq.mdx"
JS = "src/oss/javascript/integrations/chat/groq.mdx"
OTHER = "src/oss/python/integrations/chat/xai.mdx"


def site(path, line, value):
    return {"value": value, "source_path": path, "line_start": line, "line_end": line,
            "context": "code", "quote": value}


def finding(entity, attribute, *sites, kind="contradiction"):
    values = {}
    for s in sites:
        values.setdefault(s["value"], []).append(s)
    return {"kind": kind, "severity": "high", "entity_id": entity,
            "attribute_id": attribute, "values": values}


def write_labels(tmp_path, *rows):
    path = tmp_path / "labels.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


INJECTED = finding("groq", "env_var", site(PY, 53, "GROQ_API_KEY"), site(PY, 54, "GROQ_TOKEN"))
NOISE = finding("groq", "model", site(PY, 87, "qwen"), site(PY, 140, "llama"))
ELSEWHERE = finding("xai", "model", site(OTHER, 79, "grok-beta"), site(OTHER, 181, "grok-4.3"))


def test_precision_recall_and_false_positives(tmp_path):
    labels = load_labels(write_labels(
        tmp_path,
        {"id": "a", "label": "real", "sites": [f"{PY}:54"], "note": "env var"},
        {"id": "b", "label": "real", "sites": [f"{JS}:62"], "note": "package"},
        {"id": "c", "label": "not-real", "sites": [f"{PY}:87"], "note": "example choice"},
    ))
    result = score([INJECTED, NOISE, ELSEWHERE], labels)
    assert result.precision == 0.5  # INJECTED of INJECTED + NOISE
    assert result.recall == 0.5  # a found, b missed
    assert result.out_of_scope == 1  # xai is not in a labeled file
    assert [fp["reason"] for fp in result.false_positives] == ["example choice"]
    assert [row["id"] for row in result.missed] == ["b"]


def test_a_finding_without_a_matching_label_is_a_false_positive(tmp_path):
    labels = load_labels(write_labels(tmp_path, {"label": "real", "sites": [f"{PY}:54"]}))
    result = score([NOISE], labels)
    assert result.false_positives[0]["reason"] == "matches no row labeled real"
    assert result.precision == 0.0 and result.recall == 0.0


def test_entity_and_attribute_must_agree_when_the_row_names_them(tmp_path):
    row = {"label": "real", "sites": [f"{PY}:54"], "attribute_id": "package"}
    result = score([INJECTED], load_labels(write_labels(tmp_path, row)))
    assert not result.true_positives


def test_one_real_label_found_by_two_findings_counts_once_for_recall(tmp_path):
    labels = load_labels(write_labels(tmp_path, {"label": "real", "sites": [f"{PY}:54"]}))
    twin = finding("groq", "env_var", site(PY, 54, "GROQ_TOKEN"), site(JS, 46, "GROQ_API_KEY"),
                   kind="divergence")
    result = score([INJECTED, twin], labels)
    assert len(result.true_positives) == 2 and result.found == 1 and result.recall == 1.0


def test_unlabeled_rows_are_skipped_and_bad_rows_rejected(tmp_path):
    rows = load_labels(write_labels(tmp_path, {"label": "", "sites": [f"{PY}:1"]}))
    assert rows == []
    for bad in ({"label": "maybe", "sites": [f"{PY}:1"]}, {"label": "real", "sites": []},
                {"label": "real", "sites": ["groq.mdx"]}):
        with pytest.raises(LabelError):
            load_labels(write_labels(tmp_path, bad))
    (tmp_path / "broken.jsonl").write_text("{not json\n")
    with pytest.raises(LabelError, match="broken.jsonl:1"):
        load_labels(tmp_path / "broken.jsonl")


def test_render_lists_every_false_positive_and_miss(tmp_path):
    labels = load_labels(write_labels(
        tmp_path, {"id": "b", "label": "real", "sites": [f"{PY}:69"], "note": "package"}))
    text = render(score([NOISE], labels), "control set")
    assert text.startswith("## Scoring against control set\n\nPrecision 0% (0 of 1")
    assert "groq/model" in text and "matches no row labeled real" in text
    assert f"- b at {PY}:69: package" in text


def test_replace_section_keeps_the_other_sections():
    report = "# Report\n\n## A\n\nold\n\n## B\n\nkeep\n"
    assert _replace_section(report, "A", "## A\n\nnew\n") == "# Report\n\n## A\n\nnew\n\n## B\n\nkeep\n"
    assert _replace_section(report, "C", "## C\n") == report + "\n## C\n"
    assert _replace_section("", "A", "## A\n") == "## A\n"


def test_score_command_writes_the_report_section(tmp_path):
    findings = tmp_path / "findings.json"
    findings.write_text(json.dumps([INJECTED]))
    labels = write_labels(tmp_path, {"label": "real", "sites": [f"{PY}:54"]})
    report = tmp_path / "report.md"
    report.write_text("# Report\n")
    assert main(["score", "--findings", str(findings), "--labels", str(labels),
                 "--name", "control set", "--report", str(report)]) == 0
    assert "Precision 100% (1 of 1" in report.read_text()
    assert main(["score", "--findings", str(findings), "--labels", str(tmp_path / "none")]) == 2
