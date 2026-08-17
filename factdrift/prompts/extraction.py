"""The claim extraction prompt.

PROMPT_VERSION is part of the extraction cache key. Bump it whenever the
system prompt, the schema, or the document rendering changes, otherwise a
rerun will serve claims produced by the previous prompt.
"""

from __future__ import annotations

from factdrift.ingest import Document

PROMPT_VERSION = "extraction-v1"

SYSTEM = """\
You extract atomic factual claims from technical documentation.

A claim is one entity, one attribute of that entity, and the value the \
document gives for it. Extract a claim only when the document states a value.

Entities are the things the document is about: a package, a class, a model \
identifier, an environment variable, a provider, an API. Attributes are the \
properties they are given: package name, class name, default model, \
max_tokens, temperature, base URL, required environment variable, supported \
feature, install command.

Extract:
- Values in prose, including tables and frontmatter.
- Values in code blocks, including parameter values passed in examples and \
values in install commands.
- Values that look wrong or outdated. Report what the document says. Never \
correct it.

Do not extract:
- Values you infer, compute, or carry over from your own knowledge. Every \
claim must be visible in the lines you cite.
- Pure prose with no value attached to it.
- Section headings, link text, and navigation.

Field rules:
- value is the normalized form: strip surrounding quotes and backticks, keep \
the value itself byte for byte otherwise.
- raw_value is the value exactly as it appears in the source, including any \
quotes or backticks.
- quote is a verbatim substring of the source lines you cite. Copy it, do not \
retype it. A claim whose quote does not appear in the source is a defect.
- line_start and line_end are the line numbers shown in the left margin of the \
document below, and must contain the quote.
- context is "code" when the value comes from inside a code block and "prose" \
otherwise. The same value carries different weight in each.
- confidence is your confidence that this is a real, correctly-read claim.

Return every claim you find. Do not filter for importance.\
"""

CLAIM_PROPERTIES = {
    "entity": {
        "type": "string",
        "description": "The thing the claim is about, as the document names it.",
    },
    "attribute": {
        "type": "string",
        "description": "The property of the entity being given a value.",
    },
    "value": {"type": "string", "description": "The normalized value."},
    "raw_value": {"type": "string", "description": "The value as written."},
    "line_start": {"type": "integer", "description": "First source line, 1-indexed."},
    "line_end": {"type": "integer", "description": "Last source line, 1-indexed."},
    "quote": {
        "type": "string",
        "description": "Verbatim text from the cited lines containing the value.",
    },
    "context": {"type": "string", "enum": ["prose", "code"]},
    "confidence": {"type": "number", "description": "0.0 to 1.0."},
}

CLAIMS_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": CLAIM_PROPERTIES,
                "required": list(CLAIM_PROPERTIES),
                "additionalProperties": False,
            },
        }
    },
    "required": ["claims"],
    "additionalProperties": False,
}


def render_document(document: Document) -> str:
    """Render a document for extraction, with source line numbers in the margin.

    Only retained spans are shown, so stripped component syntax cannot be
    mistaken for content, and the gaps in the numbering are honest about it.
    """
    blocks = [f"# {document.path}"]
    for span in document.spans:
        if span.kind == "code":
            label = f"code block ({span.lang})" if span.lang else "code block"
        else:
            label = "prose"
        header = f"--- {label}, lines {span.line_start}-{span.line_end} ---"
        numbered = "\n".join(
            f"{lineno} | {line}"
            for lineno, line in enumerate(span.text.split("\n"), span.line_start)
        )
        blocks.append(f"{header}\n{numbered}")
    return "\n\n".join(blocks)


def user_message(document: Document) -> str:
    return (
        "Extract every factual claim from this document.\n\n"
        f"{render_document(document)}"
    )
