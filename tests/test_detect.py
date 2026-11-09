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


def test_yes_no_spellings_compare_equal():
    assert {normalize_value(v) for v in ["true", "Supported", "✅", "yes"]} == {"true"}
    assert {normalize_value(v) for v in ["false", "❌", "Not supported"]} == {"false"}


def test_install_commands_compare_by_package_set():
    same = [
        "npm install @langchain/anthropic @langchain/core",
        "yarn add @langchain/core @langchain/anthropic",
        "pnpm add @langchain/anthropic @langchain/core",
    ]
    assert len({normalize_value(v) for v in same}) == 1
    assert normalize_value("pip install -U langchain-anthropic") == normalize_value(
        "uv add langchain-anthropic"
    )
    assert normalize_value("!pip install -qU x") == normalize_value("%pip install x")
    assert normalize_value("pip install langchain-openai") != normalize_value(
        "pip install langchain-anthropic"
    )


def test_single_attribute_disagreement_is_a_contradiction():
    findings = detect([
        norm("anthropic", "package_name", "langchain-anthropic"),
        norm("anthropic", "package_name", "langchain_anthropic", context="code"),
    ])
    assert [(f.kind, f.severity) for f in findings] == [("contradiction", "medium")]


def test_single_attribute_differing_across_languages_is_not_flagged():
    assert detect([
        norm("anthropic", "package_name", "langchain-anthropic"),
        norm("anthropic", "package_name", "@langchain/anthropic", language="javascript"),
    ]) == []


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


def test_placeholders_are_not_values():
    for value in ["<SERVICE_URL>", "YOUR_ENDPOINT_NAME", "your-api-key", "langchain-(PACKAGE)",
                  "PASTE YOUR PROJECT_ID HERE", "ft:gpt-3.5-turbo-0613:{ORG_NAME}::{MODEL_ID}",
                  "https://your_custom_url.com"]:
        assert normalize_value(value) is None, value
    assert normalize_value("databricks-dbrx-instruct") == "databricks-dbrx-instruct"
    assert detect([
        norm("databricks", "endpoint", "databricks-dbrx-instruct", context="code"),
        norm("databricks", "endpoint", "YOUR_ENDPOINT_NAME", context="code"),
    ]) == []
    # A placeholder on one side leaves nothing to diverge from.
    assert detect([
        norm("cohere_api_key", "value", "your-api-key", language="javascript", cardinality="free"),
        norm("cohere_api_key", "value", "my-real-looking-key", cardinality="free"),
    ]) == []


def test_parenthesized_words_are_not_placeholders():
    for value in ["cannot be an assistant (ai) message", "ChatVertexAI (Legacy)", "✅ (Gemini)"]:
        assert normalize_value(value) is not None, value


def test_env_assignment_compares_as_the_variable_name():
    assert normalize_value("ANTHROPIC_API_KEY=your-api-key") == normalize_value("ANTHROPIC_API_KEY")
    assert detect([
        norm("anthropic", "required_env_var", "ANTHROPIC_API_KEY"),
        norm("anthropic", "required_env_var", "ANTHROPIC_API_KEY=your-api-key", context="code"),
    ]) == []


def test_null_spellings_compare_equal():
    assert {normalize_value(v) for v in ["None", "undefined", "null", "nil"]} == {"null"}
    assert detect([
        norm("groq", "max_tokens", "None", cardinality="free", context="code"),
        norm("groq", "max_tokens", "undefined", language="javascript", cardinality="free", context="code"),
    ]) == []


def test_negated_support_is_false():
    assert normalize_value("not currently supported") == "false"
    assert normalize_value("Not yet supported") == "false"


def test_install_command_with_extra_packages_agrees():
    claims = [
        norm("anthropic", "install_command", "pip install -U langchain-anthropic", context="code"),
        norm("anthropic", "install_command", "pip install -qU langchain-anthropic defusedxml", context="code"),
    ]
    assert detect(claims) == []
    # Disjoint package sets still disagree.
    assert len(detect(claims + [
        norm("anthropic", "install_command", "pip install anthropic-langchain", context="code"),
    ])) == 1
