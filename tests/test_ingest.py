from pathlib import Path

import pytest

from factdrift.ingest import Document, load_document, parse_spans

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def sample() -> Document:
    return load_document(FIXTURES / "sample.mdx", FIXTURES)


def span_at(document: Document, line_start: int):
    matches = [s for s in document.spans if s.line_start == line_start]
    assert matches, f"no span starting at line {line_start}"
    return matches[0]


def test_frontmatter_is_kept_as_prose(sample):
    span = span_at(sample, 2)
    assert span.kind == "prose"
    assert span.line_end == 4
    assert "langchain-example" in span.text


def test_component_syntax_is_stripped_but_placeholders_survive(sample):
    span = span_at(sample, 10)
    assert span.kind == "prose"
    assert span.line_end == 13
    assert "claude-sonnet-4-6" in span.text
    assert "<YOUR_API_KEY>" in span.text
    assert "<Tip>" not in span.text
    assert "snippets/oss" not in span.text


def test_code_block_keeps_content_language_and_line_numbers(sample):
    span = span_at(sample, 16)
    assert span.kind == "code"
    assert span.lang == "python"
    assert span.line_end == 18
    assert "import os" in span.text
    assert "claude-3-sonnet-20240229" in span.text


def test_line_numbers_address_the_original_file(sample):
    lines = sample.text.splitlines()
    for span in sample.spans:
        if span.kind != "code":
            continue
        assert span.text.split("\n") == lines[span.line_start - 1 : span.line_end]


def test_trailing_prose_after_a_code_block(sample):
    span = span_at(sample, 21)
    assert span.text == "Trailing prose."


def test_language_comes_from_the_path(tmp_path):
    path = tmp_path / "src" / "oss" / "python" / "integrations" / "chat" / "x.mdx"
    path.parent.mkdir(parents=True)
    path.write_text("Hello.\n")
    assert load_document(path, tmp_path).language == "python"


def test_empty_file_produces_no_spans():
    assert parse_spans("") == ()


def test_whitespace_only_file_produces_no_spans():
    assert parse_spans("\n\n   \n") == ()


def test_unterminated_frontmatter_is_treated_as_prose():
    spans = parse_spans("---\ntitle: x\n\nBody text.\n")
    assert len(spans) == 1
    assert spans[0].line_start == 1
    assert "title: x" in spans[0].text


def test_unterminated_code_fence_runs_to_end_of_file():
    spans = parse_spans("Intro.\n\n```python\nx = 1\ny = 2\n")
    code = [s for s in spans if s.kind == "code"]
    assert len(code) == 1
    assert code[0].line_start == 4
    assert code[0].line_end == 5


def test_import_inside_a_code_block_is_not_stripped():
    spans = parse_spans("```python\nimport os\n```\n")
    assert spans[0].kind == "code"
    assert spans[0].text == "import os"


def test_empty_code_block_produces_no_span():
    assert parse_spans("```python\n```\n") == ()
