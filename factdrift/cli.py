"""Command line entry point."""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import anthropic

from factdrift import extract as extract_mod
from factdrift.ingest import Document, load_corpus, load_document, find_sources
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
    return "unpriced" if value is None else f"${value:,.2f}"


def cmd_extract(args: argparse.Namespace) -> int:
    all_documents = load_corpus(args.corpus)
    if not all_documents:
        print(f"no markdown or mdx files under {args.corpus}", file=sys.stderr)
        return 1

    documents = _select(all_documents, args.files)
    if args.limit is not None:
        documents = documents[: args.limit]

    prices = dict(extract_mod.PRICES)
    if args.price_in is not None and args.price_out is not None:
        prices[args.model] = (args.price_in, args.price_out)

    client = anthropic.Anthropic()

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
    print("\nprojecting full run over "
          f"{len(all_documents)} files (counting tokens, no charge)")
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

    extract = sub.add_parser("extract", parents=[common], help="extract claims")
    extract.add_argument("--files", nargs="+", metavar="PATH", help="only these files")
    extract.add_argument("--model", default=extract_mod.DEFAULT_MODEL)
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

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
