"""Command line entry point."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

import anthropic

from factdrift import detect as detect_mod
from factdrift import extract as extract_mod
from factdrift import normalize as normalize_mod
from factdrift import questions as questions_mod
from factdrift import retrieve as retrieve_mod
from factdrift import score as score_mod
from factdrift.ingest import (
    Document,
    find_sources,
    frontmatter_keys,
    load_corpus,
    load_document,
)
from factdrift.prompts.extraction import PROMPT_VERSION, render_document

DEFAULT_CORPUS = Path("corpus")


def _select(documents: list[Document], files: list[str] | None) -> list[Document]:
    """Resolve file arguments against corpus-relative paths.

    A full corpus-relative path is matched exactly; anything else is matched as
    a path suffix, and must identify exactly one file.
    """
    if not files:
        return documents
    by_path = {d.path: d for d in documents}
    selected = []
    for name in files:
        key = name.removeprefix("./").strip("/")
        if key not in by_path:
            matches = [p for p in by_path if p == key or p.endswith("/" + key)]
            if len(matches) != 1:
                raise SystemExit(
                    f"{'no' if not matches else len(matches)} corpus files match "
                    f"{name!r}; give a longer path"
                )
            key = matches[0]
        selected.append(by_path[key])
    return selected


def cmd_ingest(args: argparse.Namespace) -> int:
    documents = load_corpus(args.corpus, limit=args.limit)
    if not documents:
        print(f"no markdown or mdx files under {args.corpus}", file=sys.stderr)
        return 1

    if args.show:
        for document in _select(documents, args.show):
            print(render_document(document))
            print()
        return 0

    prose = sum(1 for d in documents for s in d.spans if s.kind == "prose")
    code = sum(1 for d in documents for s in d.spans if s.kind == "code")
    lines = sum(d.line_count for d in documents)
    print(f"files       {len(documents)}")
    print(f"lines       {lines}")
    print(f"prose spans {prose}")
    print(f"code spans  {code}")
    for document in documents:
        if not document.spans:
            print(f"warning: no spans in {document.path}", file=sys.stderr)
    return 0


def _fmt_cost(value: float | None) -> str:
    if value is None:
        return "unpriced"
    # Cents hide a subset run that costs a fraction of one.
    return f"${value:,.2f}" if value >= 1 else f"${value:.4f}"


def _make_backend(args: argparse.Namespace) -> Any:
    """Build the client for args.backend and fill in the default model.

    Returns None, after printing why, when the backend cannot be built."""
    if args.backend == "ollama":
        args.model = args.model or extract_mod.DEFAULT_OLLAMA_MODEL
        return extract_mod.OllamaBackend(
            args.ollama_url, num_ctx=args.num_ctx, timeout=args.timeout
        )
    if args.backend == "openrouter":
        args.model = args.model or extract_mod.DEFAULT_OPENROUTER_MODEL
        try:
            return extract_mod.OpenRouterBackend(
                os.environ.get("OPENROUTER_API_KEY", ""),
                args.openrouter_url,
                timeout=args.timeout,
            )
        except extract_mod.BackendUnavailable as exc:
            print(f"openrouter: {exc}", file=sys.stderr)
            return None
    args.model = args.model or extract_mod.DEFAULT_MODEL
    return anthropic.Anthropic()


def cmd_extract(args: argparse.Namespace) -> int:
    all_documents = load_corpus(args.corpus)
    if not all_documents:
        print(f"no markdown or mdx files under {args.corpus}", file=sys.stderr)
        return 1

    documents = _select(all_documents, args.files)
    if args.limit is not None:
        documents = documents[: args.limit]

    client = _make_backend(args)
    if client is None:
        return 2

    prices = dict(extract_mod.PRICES)
    if args.backend == "ollama":
        prices.setdefault(args.model, (0.0, 0.0))  # runs locally, no per-token charge
    if args.price_in is not None and args.price_out is not None:
        prices[args.model] = (args.price_in, args.price_out)

    def run(document: Document) -> extract_mod.ExtractResult:
        return extract_mod.extract_document(
            document,
            client,
            model=args.model,
            effort=args.effort,
            max_tokens=args.max_tokens,
            cache_dir=args.cache_dir,
        )

    started = time.monotonic()
    try:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            results = list(pool.map(run, documents))
    except anthropic.AuthenticationError:
        print(
            "the API rejected the key: set ANTHROPIC_API_KEY to a valid key",
            file=sys.stderr,
        )
        return 2
    except anthropic.PermissionDeniedError as exc:
        print(f"the key lacks access: {exc}", file=sys.stderr)
        return 2
    except anthropic.NotFoundError:
        print(f"unknown model {args.model!r}", file=sys.stderr)
        return 2
    except extract_mod.BackendUnavailable as exc:
        print(f"{args.backend}: {exc}", file=sys.stderr)
        return 2
    elapsed = time.monotonic() - started

    total = extract_mod.Usage()
    api_calls = 0
    for result in results:
        status = "cached" if result.cached else "api"
        if not result.cached:
            api_calls += 1
            total.add(result.usage)
        note = f" error={result.error}" if result.error else ""
        print(
            f"{status:6} {len(result.claims):4} claims "
            f"{len(result.rejected):3} rejected  {result.path}{note}"
        )

    claims = [c for r in results for c in r.claims]
    rejected = [x for r in results for x in r.rejected]

    print()
    print(f"documents      {len(results)}")
    print(f"api calls      {api_calls}")
    print(f"claims kept    {len(claims)}")
    print(f"claims dropped {len(rejected)}")
    if rejected:
        reasons: dict[str, int] = {}
        for item in rejected:
            reasons[item["reason"]] = reasons.get(item["reason"], 0) + 1
        for reason, count in sorted(reasons.items(), key=lambda kv: -kv[1]):
            print(f"  {count:4}  {reason}")
    print(f"input tokens   {total.input_tokens:,}")
    print(f"output tokens  {total.output_tokens:,}")
    subset_cost = extract_mod.cost(total, args.model, prices)
    print(f"cost           {_fmt_cost(subset_cost)}")
    print(f"wall clock     {elapsed:,.0f}s at concurrency {args.concurrency}")

    if api_calls and len(documents) < len(all_documents):
        _project(client, all_documents, documents, total, elapsed, args, prices)

    if args.show_claims:
        _print_claims(claims, args.show_claims)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                {
                    "model": args.model,
                    "prompt_version": PROMPT_VERSION,
                    "claims": [asdict(c) for c in claims],
                    "rejected": rejected,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nwrote {args.out}")

    return 0


def _project(
    client,
    all_documents: list[Document],
    ran: list[Document],
    usage: extract_mod.Usage,
    elapsed: float,
    args: argparse.Namespace,
    prices: dict[str, tuple[float, float]],
) -> None:
    """Project full-corpus cost from the subset that just ran.

    Input tokens are counted exactly for every file. Output tokens are scaled
    from the observed output-per-input ratio, so the number is an estimate.
    """
    method = "counting tokens" if args.backend == "anthropic" else "estimating tokens"
    print("\nprojecting full run over "
          f"{len(all_documents)} files ({method}, no charge)")
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        counts = list(
            pool.map(
                lambda d: extract_mod.count_input_tokens(client, d, args.model),
                all_documents,
            )
        )
    total_input = sum(counts)
    ran_paths = {d.path for d in ran}
    ran_input = sum(
        c for c, d in zip(counts, all_documents) if d.path in ran_paths
    )
    if not ran_input:
        return

    ratio = usage.output_tokens / max(usage.input_tokens, 1)
    projected = extract_mod.Usage(
        input_tokens=total_input, output_tokens=int(total_input * ratio)
    )
    scale = total_input / ran_input
    print(f"input tokens   {total_input:,}")
    print(f"output tokens  {projected.output_tokens:,} (estimated)")
    print(f"cost           {_fmt_cost(extract_mod.cost(projected, args.model, prices))}")
    print(f"wall clock     {elapsed * scale:,.0f}s at concurrency {args.concurrency}")


def _print_claims(claims: list[extract_mod.Claim], count: str) -> None:
    shown = claims if count == "all" else claims[: int(count)]
    print(f"\n{len(shown)} of {len(claims)} claims:\n")
    for claim in shown:
        print(json.dumps(asdict(claim), indent=2))
        print()


def _load_claims(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["claims"] if isinstance(payload, dict) else payload


def cmd_normalize(args: argparse.Namespace) -> int:
    if args.backend == "anthropic":
        print("normalize needs --backend ollama or openrouter", file=sys.stderr)
        return 2
    claims = _load_claims(args.claims)
    frontmatter = {d.path: frontmatter_keys(d.text) for d in load_corpus(args.corpus)}
    missing = {c["source_path"] for c in claims} - frontmatter.keys()
    if missing:
        print(
            f"warning: {len(missing)} claim sources are not under {args.corpus}; "
            "their frontmatter page metadata is not scoped to the page",
            file=sys.stderr,
        )
    client = _make_backend(args)
    if client is None:
        return 2
    try:
        annotated, result = normalize_mod.normalize_claims(
            claims,
            client,
            model=args.model,
            batch_size=args.batch_size,
            max_tokens=args.max_tokens,
            cache_dir=args.cache_dir,
            frontmatter=frontmatter,
            page_keys=args.page_keys,
            response_entities=args.response_entities,
            response_window=args.response_window,
        )
    except extract_mod.BackendUnavailable as exc:
        print(f"{args.backend}: {exc}", file=sys.stderr)
        return 2
    entities = {c["entity_id"] for c in annotated}
    attributes = {c["attribute_id"] for c in annotated}
    single = sorted(a for a in attributes if result.cardinality[a] == "single")
    print(
        f"{len(claims)} claims, {len(result.aliases)} distinct pairs -> "
        f"{len(entities)} entities, {len(attributes)} attributes "
        f"({len(single)} single-valued)"
    )
    print(
        f"{result.api_calls} requests, {result.input_tokens} in / "
        f"{result.output_tokens} out tokens, {result.fallbacks} fallbacks"
    )
    if args.price_in is not None and args.price_out is not None:
        dollars = (
            result.input_tokens * args.price_in + result.output_tokens * args.price_out
        ) / 1_000_000
        print(f"cost: ${dollars:.4f}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "model": args.model,
                "prompt_versions": [
                    normalize_mod.ALIAS_PROMPT_VERSION,
                    normalize_mod.CARDINALITY_PROMPT_VERSION,
                ],
                "cardinality": result.cardinality,
                "reasons": result.reasons,
                "claims": annotated,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"wrote {args.out}")
    return 0


def cmd_detect(args: argparse.Namespace) -> int:
    claims = _load_claims(args.claims)
    if claims and "cardinality" not in claims[0]:
        print(f"{args.claims} is not normalized; run normalize first", file=sys.stderr)
        return 2
    findings = detect_mod.detect(
        claims,
        divergence=not args.no_divergence,
        package_attributes=args.package_attributes,
    )
    report = detect_mod.render_markdown(findings, claim_count=len(claims))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps([f.as_dict() for f in findings], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report, encoding="utf-8")
    if not args.out and not args.report:
        print(report, end="")
    else:
        print(report.split("\n\n## ", 1)[0])
    return 0


def cmd_questions(args: argparse.Namespace) -> int:
    findings = json.loads(args.findings.read_text(encoding="utf-8"))
    selected = findings if args.limit is None else findings[: args.limit]
    if args.backend == "anthropic":
        args.model = args.model or questions_mod.DEFAULT_MODEL
    client = _make_backend(args)
    if client is None:
        return 2
    effort = None
    if args.backend == "anthropic":
        client = extract_mod.AnthropicBackend(client, effort=args.effort)
        effort = args.effort

    prices = dict(extract_mod.PRICES)
    if args.backend == "ollama":
        prices.setdefault(args.model, (0.0, 0.0))  # runs locally, no per-token charge
    if args.price_in is not None and args.price_out is not None:
        prices[args.model] = (args.price_in, args.price_out)

    started = time.monotonic()
    try:
        result = questions_mod.generate_questions(
            selected,
            client,
            model=args.model,
            max_tokens=args.max_tokens,
            cache_dir=args.cache_dir,
            effort=effort,
        )
    except (extract_mod.BackendUnavailable, anthropic.AuthenticationError) as exc:
        print(f"{args.backend}: {exc}", file=sys.stderr)
        return 2
    elapsed = time.monotonic() - started

    failed = [q for q in result.questions if q["error"]]
    print(
        f"{len(selected)} of {len(findings)} findings: "
        f"{len(selected) - len(failed)} questions, {len(failed)} failed, "
        f"{result.cached} from cache, {result.api_calls} requests"
    )
    for q in failed:
        print(f"  {q['entity_id']}/{q['attribute_id']}: {q['error']}")
    print(
        f"tokens: {result.usage.input_tokens:,} in / {result.usage.output_tokens:,} out, "
        f"cost {_fmt_cost(extract_mod.cost(result.usage, args.model, prices))}, "
        f"{elapsed:.1f}s"
    )
    if result.api_calls and len(selected) < len(findings):
        scale = len(findings) / result.api_calls
        projected = extract_mod.Usage(
            input_tokens=round(result.usage.input_tokens * scale),
            output_tokens=round(result.usage.output_tokens * scale),
        )
        print(
            f"projected for all {len(findings)} findings: "
            f"{projected.input_tokens:,} in / {projected.output_tokens:,} out, "
            f"cost {_fmt_cost(extract_mod.cost(projected, args.model, prices))}, "
            f"{elapsed * scale:,.0f}s"
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "model": args.model,
                "prompt_version": questions_mod.PROMPT_VERSION,
                "questions": result.questions,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"wrote {args.out}")
    return 0


def cmd_retrieve(args: argparse.Namespace) -> int:
    findings = json.loads(args.findings.read_text(encoding="utf-8"))
    payload = json.loads(args.questions.read_text(encoding="utf-8"))
    questions = {
        q["finding_id"]: q["question"] for q in payload["questions"] if q.get("question")
    }
    selected = findings if args.limit is None else findings[: args.limit]
    documents = load_corpus(args.corpus)
    if not documents:
        print(f"no markdown or mdx files under {args.corpus}", file=sys.stderr)
        return 1
    client = _make_backend(args)
    if client is None:
        return 2
    if args.backend == "anthropic":
        client = extract_mod.AnthropicBackend(client, effort=args.effort)

    started = time.monotonic()
    chunks = [c for d in documents for c in retrieve_mod.chunk_document(d, args.max_chars)]
    index = retrieve_mod.Index(
        chunks,
        index_dir=args.index_dir,
        embed_model=args.embed_model,
        ollama_url=args.ollama_url,
        batch_size=args.embed_batch_size,
    )
    print(
        f"{len(chunks)} chunks from {len(documents)} files, "
        f"{'embedded ' + str(index.built) if index.built else 'index reused'}"
    )
    try:
        result = retrieve_mod.run(
            selected,
            questions,
            index,
            client,
            model=args.model,
            top_k=args.top_k,
            max_tokens=args.max_tokens,
            cache_dir=args.cache_dir,
        )
    except (extract_mod.BackendUnavailable, anthropic.AuthenticationError) as exc:
        print(f"{args.backend}: {exc}", file=sys.stderr)
        return 2
    elapsed = time.monotonic() - started

    prices = dict(extract_mod.PRICES)
    if args.backend == "ollama":
        prices.setdefault(args.model, (0.0, 0.0))  # runs locally, no per-token charge
    if args.price_in is not None and args.price_out is not None:
        prices[args.model] = (args.price_in, args.price_out)
    print(
        f"{len(selected)} of {len(findings)} findings: {result.api_calls} requests, "
        f"{result.cached} from cache"
    )
    print(
        f"tokens: {result.usage.input_tokens:,} in / {result.usage.output_tokens:,} out, "
        f"cost {_fmt_cost(extract_mod.cost(result.usage, args.model, prices))}, "
        f"{elapsed:.1f}s"
    )
    if result.api_calls and len(selected) < len(findings):
        scale = len(findings) / result.api_calls
        projected = extract_mod.Usage(
            input_tokens=round(result.usage.input_tokens * scale),
            output_tokens=round(result.usage.output_tokens * scale),
        )
        print(
            f"projected for all {len(findings)} findings: "
            f"{projected.input_tokens:,} in / {projected.output_tokens:,} out, "
            f"cost {_fmt_cost(extract_mod.cost(projected, args.model, prices))}"
        )
    for finding, record in zip(selected, result.served):
        print(
            f"  {finding['entity_id']}/{finding['attribute_id']}: "
            f"{record.get('served') or record.get('error')}"
        )

    served = {r["finding_id"]: r for r in result.served}
    annotated = [
        {**f, "served": served.get(questions_mod.finding_id(f))} for f in findings
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(annotated, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {args.out}")
    if args.report:
        existing = args.report.read_text(encoding="utf-8") if args.report.exists() else ""
        section = retrieve_mod.render_served(selected, result.served)
        args.report.write_text(
            _replace_section(existing, "What retrieval serves", section), encoding="utf-8"
        )
        print(f"wrote {args.report}")
    return 0


def _replace_section(report: str, heading: str, section: str) -> str:
    """Put `section` in place of the `## heading` section of a markdown report,
    or append it, leaving every other section as it was."""
    marker = f"\n## {heading}"
    text = "\n" + report.strip("\n")
    if marker in text:
        before, rest = text.split(marker, 1)
        after = rest.split("\n## ", 1)
        tail = "\n## " + after[1].strip("\n") + "\n" if len(after) > 1 else ""
        before = before.strip("\n")
        return (f"{before}\n\n{section}" if before else section).rstrip("\n") + "\n" + tail
    text = text.strip("\n")
    return f"{text}\n\n{section}" if text else section


def cmd_score(args: argparse.Namespace) -> int:
    try:
        labels = score_mod.load_labels(args.labels)
    except (OSError, score_mod.LabelError) as error:
        print(f"cannot read labels: {error}", file=sys.stderr)
        return 2
    findings = json.loads(args.findings.read_text(encoding="utf-8"))
    result = score_mod.score(findings, labels)
    name = args.name or args.labels.name
    section = score_mod.render(result, name)
    print(section.split("\n### ", 1)[0].rstrip())
    for item in result.false_positives:
        f = item["finding"]
        print(f"  FP {f['entity_id']}/{f['attribute_id']}: {item['reason']}")
    for row in result.missed:
        print(f"  missed {row['id']}: {row.get('note', '')}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(score_mod.as_dict(result), indent=2, ensure_ascii=False),
                            encoding="utf-8")
        print(f"wrote {args.out}")
    if args.report:
        existing = args.report.read_text(encoding="utf-8") if args.report.exists() else ""
        heading = section.splitlines()[0].removeprefix("## ")
        args.report.write_text(_replace_section(existing, heading, section), encoding="utf-8")
        print(f"wrote {args.report}")
    return 0


def _backend_options(default_backend: str) -> argparse.ArgumentParser:
    """Backend flags, as a fresh parent parser per command so each command
    can have its own default backend (parents share action objects)."""
    backend_opts = argparse.ArgumentParser(add_help=False)
    backend_opts.add_argument(
        "--backend",
        choices=["anthropic", "ollama", "openrouter"],
        default=default_backend,
        help="where model requests go (default: %(default)s)",
    )
    backend_opts.add_argument(
        "--model",
        help=f"model id (default: {extract_mod.DEFAULT_MODEL} for anthropic, "
        f"{extract_mod.DEFAULT_OLLAMA_MODEL} for ollama, "
        f"{extract_mod.DEFAULT_OPENROUTER_MODEL} for openrouter)",
    )
    backend_opts.add_argument(
        "--ollama-url",
        default=extract_mod.DEFAULT_OLLAMA_URL,
        help="Ollama server (default: %(default)s)",
    )
    backend_opts.add_argument(
        "--openrouter-url",
        default=extract_mod.DEFAULT_OPENROUTER_URL,
        help="OpenRouter API base; the key comes from OPENROUTER_API_KEY "
        "(default: %(default)s)",
    )
    backend_opts.add_argument(
        "--num-ctx",
        type=int,
        default=extract_mod.DEFAULT_NUM_CTX,
        help="Ollama context window in tokens (default: %(default)s)",
    )
    backend_opts.add_argument(
        "--timeout",
        type=float,
        default=extract_mod.DEFAULT_TIMEOUT,
        help="seconds to wait for one Ollama or OpenRouter response "
        "(default: %(default)s)",
    )
    return backend_opts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="factdrift")
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--corpus",
        type=Path,
        default=DEFAULT_CORPUS,
        help="corpus root (default: %(default)s)",
    )
    common.add_argument("--limit", type=int, help="process at most this many files")

    ingest = sub.add_parser("ingest", parents=[common], help="load and inspect the corpus")
    ingest.add_argument(
        "--show",
        nargs="+",
        metavar="PATH",
        help="print these documents as the extraction pass sees them",
    )
    ingest.set_defaults(func=cmd_ingest)

    extract = sub.add_parser(
        "extract", parents=[common, _backend_options("anthropic")], help="extract claims"
    )
    extract.add_argument("--files", nargs="+", metavar="PATH", help="only these files")
    extract.add_argument(
        "--effort",
        default=extract_mod.DEFAULT_EFFORT,
        choices=["low", "medium", "high", "xhigh", "max"],
    )
    extract.add_argument("--max-tokens", type=int, default=extract_mod.DEFAULT_MAX_TOKENS)
    extract.add_argument("--cache-dir", type=Path, default=extract_mod.DEFAULT_CACHE_DIR)
    extract.add_argument("--concurrency", type=int, default=4)
    extract.add_argument("--out", type=Path, help="write all kept claims here as JSON")
    extract.add_argument(
        "--show-claims",
        metavar="N",
        help="print N claims in full, or 'all'",
    )
    extract.add_argument("--price-in", type=float, help="dollars per million input tokens")
    extract.add_argument(
        "--price-out", type=float, help="dollars per million output tokens"
    )
    extract.set_defaults(func=cmd_extract)

    normalize = sub.add_parser(
        "normalize",
        parents=[_backend_options("openrouter")],
        help="canonical ids and cardinality for extracted claims",
    )
    normalize.add_argument("--claims", type=Path, required=True, help="output of extract --out")
    normalize.add_argument(
        "--corpus",
        type=Path,
        default=DEFAULT_CORPUS,
        help="corpus the claims came from, read for frontmatter (default: %(default)s)",
    )
    normalize.add_argument(
        "--page-keys",
        nargs="+",
        default=list(normalize_mod.DEFAULT_PAGE_KEYS),
        metavar="KEY",
        help="frontmatter keys that describe the page, not the integration "
        "(default: %(default)s)",
    )
    normalize.add_argument(
        "--response-entities",
        nargs="+",
        default=list(normalize_mod.DEFAULT_RESPONSE_ENTITIES),
        metavar="ENTITY",
        help="entity ids of printed responses, whose fields belong to the call "
        "above them (default: %(default)s)",
    )
    normalize.add_argument(
        "--response-window",
        type=int,
        default=normalize_mod.DEFAULT_RESPONSE_WINDOW,
        help="most lines between a call and its printed response "
        "(default: %(default)s)",
    )
    normalize.add_argument("--out", type=Path, required=True)
    normalize.add_argument(
        "--batch-size", type=int, default=normalize_mod.DEFAULT_BATCH_SIZE
    )
    normalize.add_argument(
        "--max-tokens", type=int, default=normalize_mod.DEFAULT_MAX_TOKENS
    )
    normalize.add_argument(
        "--cache-dir", type=Path, default=normalize_mod.DEFAULT_CACHE_DIR
    )
    normalize.add_argument("--price-in", type=float, help="dollars per million input tokens")
    normalize.add_argument("--price-out", type=float, help="dollars per million output tokens")
    normalize.set_defaults(func=cmd_normalize)

    detect = sub.add_parser("detect", help="flag disagreeing claims")
    detect.add_argument("--claims", type=Path, required=True, help="output of normalize --out")
    detect.add_argument("--out", type=Path, help="write findings here as JSON")
    detect.add_argument("--report", type=Path, help="write the markdown report here")
    detect.add_argument(
        "--no-divergence",
        action="store_true",
        help="report contradictions only, not cross-language divergence",
    )
    detect.add_argument(
        "--package-attributes",
        nargs="+",
        default=sorted(detect_mod.DEFAULT_PACKAGE_ATTRIBUTES),
        metavar="ATTRIBUTE",
        help="attributes whose Python values are PyPI names, compared by PEP 503 "
        "(default: %(default)s)",
    )
    detect.set_defaults(func=cmd_detect)

    questions = sub.add_parser(
        "questions",
        parents=[_backend_options("anthropic")],
        help="write one user question per finding",
    )
    questions.add_argument("--findings", type=Path, required=True, help="output of detect --out")
    questions.add_argument("--out", type=Path, required=True)
    questions.add_argument(
        "--limit", type=int, help="only the first N findings, for a cost check"
    )
    questions.add_argument(
        "--effort",
        default="low",
        choices=["low", "medium", "high", "xhigh", "max"],
        help="anthropic backend only (default: %(default)s)",
    )
    questions.add_argument(
        "--max-tokens", type=int, default=questions_mod.DEFAULT_MAX_TOKENS
    )
    questions.add_argument(
        "--cache-dir", type=Path, default=questions_mod.DEFAULT_CACHE_DIR
    )
    questions.add_argument("--price-in", type=float, help="dollars per million input tokens")
    questions.add_argument("--price-out", type=float, help="dollars per million output tokens")
    questions.set_defaults(func=cmd_questions)

    retrieve = sub.add_parser(
        "retrieve",
        parents=[_backend_options("ollama")],
        help="answer each question from retrieved chunks and record the value served",
    )
    retrieve.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    retrieve.add_argument("--findings", type=Path, required=True, help="output of detect --out")
    retrieve.add_argument("--questions", type=Path, required=True, help="output of questions --out")
    retrieve.add_argument(
        "--out", type=Path, required=True, help="findings JSON with a served record each"
    )
    retrieve.add_argument(
        "--report", type=Path, help="markdown report to add the served-value section to"
    )
    retrieve.add_argument(
        "--limit", type=int, help="only the first N findings, for a cost check"
    )
    retrieve.add_argument("--top-k", type=int, default=retrieve_mod.DEFAULT_TOP_K)
    retrieve.add_argument(
        "--max-chars",
        type=int,
        default=retrieve_mod.DEFAULT_MAX_CHARS,
        help="longest chunk before a section is split (default: %(default)s)",
    )
    retrieve.add_argument("--index-dir", type=Path, default=retrieve_mod.DEFAULT_INDEX_DIR)
    retrieve.add_argument(
        "--embed-model",
        default=retrieve_mod.DEFAULT_EMBED_MODEL,
        help="'default' for Chroma's all-MiniLM-L6-v2, or an Ollama embedding "
        "model name (default: %(default)s)",
    )
    retrieve.add_argument(
        "--embed-batch-size",
        type=int,
        default=retrieve_mod.DEFAULT_EMBED_BATCH,
        help="chunks per embedding request while building the index (default: %(default)s)",
    )
    retrieve.add_argument(
        "--effort",
        default="low",
        choices=["low", "medium", "high", "xhigh", "max"],
        help="anthropic backend only (default: %(default)s)",
    )
    retrieve.add_argument("--max-tokens", type=int, default=retrieve_mod.DEFAULT_MAX_TOKENS)
    retrieve.add_argument("--cache-dir", type=Path, default=retrieve_mod.DEFAULT_CACHE_DIR)
    retrieve.add_argument("--price-in", type=float, help="dollars per million input tokens")
    retrieve.add_argument("--price-out", type=float, help="dollars per million output tokens")
    retrieve.set_defaults(func=cmd_retrieve)

    score = sub.add_parser("score", help="precision and recall against a labeled set")
    score.add_argument("--findings", type=Path, required=True, help="output of detect --out")
    score.add_argument("--labels", type=Path, required=True, help="labeled set, JSONL")
    score.add_argument("--name", help="name used in the report heading (default: labels file name)")
    score.add_argument("--out", type=Path, help="write the full result here as JSON")
    score.add_argument("--report", type=Path,
                       help="add or replace this labeled set's section in a markdown report")
    score.set_defaults(func=cmd_score)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
