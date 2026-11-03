from factdrift.detect import detect, normalize_value, render_markdown


def norm(entity_id, attribute_id, value, *, language="python", context="prose", cardinality="single", line=1):
    return {
        "entity": entity_id,
        "attribute": attribute_id,
        "value": value,
        "raw_value": value,
        "source_path": f"src/oss/{language}/integrations/chat/x.mdx",
        "line_start": line,
        "line_end": line,
        "quote": value,
        "context": context,
        "confidence": 0.9,
        "entity_id": entity_id,
        "attribute_id": attribute_id,
        "language": language,
        "cardinality": cardinality,
    }


def test_normalize_value_ignores_quotes_case_and_spacing():
    assert normalize_value(' "Langchain-Anthropic". ') == "langchain-anthropic"
    assert normalize_value("`a  b`") == "a b"


def test_single_attribute_disagreement_is_a_contradiction():
    findings = detect([
        norm("anthropic", "package_name", "langchain-anthropic"),
        norm("anthropic", "package_name", "langchain_anthropic", context="code"),
    ])
    assert [(f.kind, f.severity) for f in findings] == [("contradiction", "medium")]


def test_agreeing_values_are_not_flagged():
    assert detect([
        norm("anthropic", "package_name", "langchain-anthropic"),
        norm("anthropic", "package_name", "'langchain-anthropic'", language="javascript"),
    ]) == []


def test_free_attribute_within_one_language_is_noise():
    claims = [
        norm("anthropic", "max_tokens", "1024", cardinality="free", context="code"),
        norm("anthropic", "max_tokens", "4096", cardinality="free", context="code"),
    ]
    assert detect(claims) == []


def test_free_attribute_disjoint_across_languages_is_a_divergence():
    claims = [
        norm("anthropic", "model", "claude-sonnet-4-6", cardinality="free", context="code"),
        norm("anthropic", "model", "claude-3-sonnet-20240229", language="javascript", cardinality="free", context="code"),
    ]
    findings = detect(claims)
    assert [(f.kind, f.severity) for f in findings] == [("divergence", "low")]
    assert detect(claims, divergence=False) == []


def test_shared_value_across_languages_is_not_a_divergence():
    claims = [
        norm("anthropic", "model", "a", cardinality="free"),
        norm("anthropic", "model", "b", cardinality="free"),
        norm("anthropic", "model", "a", language="javascript", cardinality="free"),
    ]
    assert detect(claims) == []


def test_findings_are_ordered_and_rendered():
    claims = [
        norm("x", "model", "a", cardinality="free"),
        norm("x", "model", "b", language="javascript", cardinality="free"),
        norm("y", "env_var", "A_KEY", context="code"),
        norm("y", "env_var", "B_KEY", context="code"),
        norm("z", "base_url", "https://a"),
        norm("z", "base_url", "https://b"),
    ]
    findings = detect(claims)
    assert [(f.kind, f.severity, f.entity_id) for f in findings] == [
        ("contradiction", "high", "z"),
        ("contradiction", "low", "y"),
        ("divergence", "high", "x"),
    ]
    report = render_markdown(findings, claim_count=len(claims))
    assert "6 claims, 3 findings." in report
    assert "| contradiction | 1 | 0 | 1 |" in report
