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


def on_page(claim, page):
    return {**claim, "source_path": f"src/oss/{claim['language']}/integrations/chat/{page}.mdx"}


def test_divergence_compares_only_the_same_providers_pages():
    claims = [
        on_page(norm("x", "model", "a", cardinality="free"), "anthropic"),
        on_page(norm("x", "model", "a", language="javascript", cardinality="free"), "anthropic"),
        on_page(norm("x", "model", "b", language="javascript", cardinality="free"), "index"),
    ]
    # The python anthropic page shares a value with its javascript twin; the
    # javascript index page has no python twin, so it is not compared.
    assert detect(claims) == []
    paired = claims + [on_page(norm("x", "model", "c", cardinality="free"), "index")]
    findings = detect(paired)
    assert len(findings) == 1
    paths = {s["source_path"] for sites in findings[0].values.values() for s in sites}
    assert paths == {"src/oss/python/integrations/chat/index.mdx",
                     "src/oss/javascript/integrations/chat/index.mdx"}


def test_python_package_names_compare_by_pep_503():
    assert normalize_value("Langchain_OpenAI", package=True) == "langchain-openai"
    assert normalize_value("langchain.openai", package=True) == "langchain-openai"
    assert normalize_value("langchain_openai") == "langchain_openai"
    assert detect([
        norm("openai", "pypi_package", "langchain-openai"),
        norm("openai", "pypi_package", "langchain_openai", context="code"),
    ]) == []


def test_pep_503_applies_to_package_attributes_in_python_only():
    # Other identifiers keep - and _ distinct.
    assert len(detect([
        norm("x", "import_path", "langchain-openai"),
        norm("x", "import_path", "langchain_openai"),
    ])) == 1
    # npm names are not PEP 503 names.
    assert len(detect([
        norm("x", "npm_package", "@langchain/open-ai", language="javascript"),
        norm("x", "npm_package", "@langchain/open_ai", language="javascript"),
    ])) == 1
    assert detect([
        norm("x", "my_pkg", "a-b"), norm("x", "my_pkg", "a_b"),
    ], package_attributes={"my_pkg"}) == []


def test_a_response_that_reports_a_different_value_than_its_call_is_a_contradiction():
    call = norm("anthropic", "model", "claude-sonnet-4-6", language="javascript",
                cardinality="free", context="code", line=263)
    response = {**norm("anthropic", "model", "claude-3-sonnet-20240229", language="javascript",
                       cardinality="free", context="code", line=283), "reports_on_line": 263}
    findings = detect([call, response], divergence=False)
    assert [(f.kind, f.entity_id, f.attribute_id) for f in findings] == [
        ("contradiction", "anthropic", "model")
    ]
    assert set(findings[0].values) == {"claude-sonnet-4-6", "claude-3-sonnet-20240229"}
    # A response that matches its call is fine.
    assert detect([call, {**response, "value": "claude-sonnet-4-6"}], divergence=False) == []


def test_a_single_attribute_response_mismatch_is_reported_once():
    call = norm("x", "base_url", "https://a", context="code", line=1)
    response = {**norm("x", "base_url", "https://b", context="code", line=5), "reports_on_line": 1}
    assert len(detect([call, response])) == 1
