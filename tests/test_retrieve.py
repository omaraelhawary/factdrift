import json

import chromadb
import pytest

from factdrift.extract import Usage
from factdrift.ingest import Document, parse_spans
from factdrift.questions import finding_id
from factdrift.retrieve import Chunk, Index, chunk_document, run, served_value, render_served

PAGE = """---
title: ChatExample
description: x
---

Intro line.

## Setup

Install it:

```bash
pip install langchain-example

pip install other
```

## Models

Use model-a.
"""


def doc(text=PAGE, path="src/oss/python/integrations/chat/example.mdx"):
    return Document(path, "python", text, parse_spans(text))


def test_chunks_follow_headings_and_skip_frontmatter():
    chunks = chunk_document(doc())
    assert [(c.line_start, c.line_end) for c in chunks] == [(5, 7), (8, 17), (18, 20)]
    assert chunks[1].text.startswith("ChatExample > Setup\n\n## Setup")
    assert "description" not in "".join(c.text for c in chunks)


def test_a_long_section_splits_between_blocks_not_inside_a_fence():
    chunks = chunk_document(doc(), max_chars=70)
    setup = [c for c in chunks if "pip" in c.text]
    # The fenced block, blank line and all, stays in one chunk, apart from
    # the prose above it.
    assert len(setup) == 1 and "pip install other" in setup[0].text
    assert (setup[0].line_start, setup[0].line_end) == (12, 17)
    assert "Install it:" not in setup[0].text


FEATURE_TABLE = """---
title: ChatHuggingFace integration
---

## Overview

### Model features

| [Tool calling](/oss/langchain/tools) | [Image input](/oss/langchain/messages#multimodal) | Native async |
| :---: | :---: | :---: |
| ✅ | ❌ | ✅ |
| ❌ | ✅ | ❌ |

See the API reference.
"""


def test_each_table_row_is_a_chunk_that_carries_its_header_and_heading():
    chunks = chunk_document(doc(FEATURE_TABLE))
    rows = {c.line_start: c for c in chunks if c.line_start == c.line_end}
    assert set(rows) == {11, 12}
    assert rows[11].text == (
        "ChatHuggingFace integration > Model features\n\n"
        "Tool calling: yes; Image input: no; Native async: yes"
    )
    assert "Image input: yes" in rows[12].text
    # The rows are not also indexed a second time inside the section chunk,
    # and no other chunk's line range claims them.
    assert not any("✅" in c.text for c in chunks)
    others = [c for c in chunks if c.line_start != c.line_end]
    assert not any(c.line_start <= 11 <= c.line_end for c in others)


def test_a_table_header_without_rows_stays_in_the_section_text():
    text = "## Models\n\n| Model | Context |\n| --- | --- |\n\nAfter.\n"
    chunks = chunk_document(doc(text))
    assert len(chunks) == 1 and "| Model | Context |" in chunks[0].text


class WordEmbedding:
    """Deterministic bag-of-letters embedding, so tests need no model."""

    def __call__(self, input):
        return [[float(t.lower().count(ch)) + 0.01 for ch in "abcdefghijklmnopqrstuvwxyz"] for t in input]

    def embed_query(self, input):
        return self(input)

    @staticmethod
    def name():
        return "test"

    def is_legacy(self):
        return False

    @staticmethod
    def supported_spaces():
        return ["cosine", "l2", "ip"]

    def get_config(self):
        return {}

    @staticmethod
    def build_from_config(config):
        return WordEmbedding()


def make_index(chunks, path):
    return Index(chunks, client=chromadb.PersistentClient(path=str(path / "chroma")),
                 embedding_function=WordEmbedding())


FINDING = {
    "kind": "contradiction",
    "severity": "high",
    "entity_id": "example",
    "attribute_id": "model",
    "values": {
        "model-a": [{"value": "model-a", "source_path": "src/oss/python/integrations/chat/example.mdx",
                     "line_start": 20, "line_end": 20, "context": "prose", "quote": "Use model-a."}],
        "model-b": [{"value": "model-b", "source_path": "elsewhere.mdx",
                     "line_start": 1, "line_end": 1, "context": "prose", "quote": "model-b"}],
    },
}


def test_served_value_reads_the_stated_value():
    assert served_value(FINDING, "Use model-a.", "model-a") == "model-a"
    assert served_value(FINDING, "It is model-b.", "") == "model-b"
    assert served_value(FINDING, "Either model-a or model-b.", "") == "several"
    assert served_value(FINDING, "Use model-c.", "model-c") == "other"
    assert served_value(FINDING, "The excerpts do not say.", "") is None


def test_yes_and_no_answers_map_to_polarity_values():
    finding = {**FINDING, "values": {"true": FINDING["values"]["model-a"], "false": FINDING["values"]["model-b"]}}
    assert served_value(finding, "Yes, it does.", "Yes") == "true"


class FakeBackend:
    def __init__(self, payload):
        self.payload = payload
        self.requests = []

    def complete_json(self, *, system, user, schema, model, max_tokens, schema_name="claims"):
        self.requests.append(user)
        return json.dumps(self.payload), Usage(input_tokens=500, output_tokens=30), "end_turn"


def test_run_records_winner_context_and_served_value_and_caches(tmp_path):
    index = make_index(chunk_document(doc()), tmp_path)
    questions = {finding_id(FINDING): "Which model should I use with ChatExample?"}
    backend = FakeBackend({"answer": "Use model-a.", "value": "model-a"})
    first = run([FINDING], questions, index, backend, model="m", top_k=3, cache_dir=tmp_path / "answers")
    record = first.served[0]
    assert record["served"] == "model-a"
    assert record["winner"] == "src/oss/python/integrations/chat/example.mdx"
    assert record["values_in_context"] == ["model-a"]
    assert "Excerpts:" in backend.requests[0]
    again = run([FINDING], questions, index, FakeBackend({}), model="m", top_k=3, cache_dir=tmp_path / "answers")
    assert (again.api_calls, again.cached, again.served[0]["served"]) == (0, 1, "model-a")
    assert "`model-a`" in render_served([FINDING], again.served)


def test_a_finding_without_a_question_or_with_a_bad_answer_is_recorded(tmp_path):
    index = make_index(chunk_document(doc()), tmp_path)
    assert run([FINDING], {}, index, FakeBackend({}), model="m", cache_dir=tmp_path / "answers").served[0]["error"] == "no question"
    bad = run([FINDING], {finding_id(FINDING): "q?"}, index, FakeBackend({"answer": 1}), model="m", cache_dir=tmp_path / "answers")
    assert bad.served[0]["served"] is None and "unusable" in bad.served[0]["error"]
    assert not (tmp_path / "answers").exists()


def test_the_index_is_reused_when_chunks_are_unchanged(tmp_path):
    client = chromadb.PersistentClient(path=str(tmp_path))
    chunks = chunk_document(doc())
    first = Index(chunks, client=client, embedding_function=WordEmbedding())
    second = Index(chunks, client=client, embedding_function=WordEmbedding())
    assert (first.built, second.built) == (len(chunks), 0)


def test_a_value_the_answer_does_not_state_is_ignored():
    # The model filled `value` from the excerpts, not from its own answer.
    assert served_value(FINDING, "See the provider docs for model names.", "model-b") is None


def test_a_leading_yes_or_no_is_the_served_polarity():
    finding = {**FINDING, "values": {"true": FINDING["values"]["model-a"],
                                     "false": FINDING["values"]["model-b"]}}
    assert served_value(finding, "Yes", "logprobs") == "true"
    assert served_value(finding, "No, it does not.", "") == "false"
