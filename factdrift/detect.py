"""Detection: group normalized claims and flag the groups that disagree.

Two kinds of finding:

- contradiction: a "single" attribute of one entity has more than one value
  within the docs of one language. At least one document is wrong. Values
  are compared per language because package names, install commands and
  feature support legitimately differ between the Python and JavaScript
  libraries.
- divergence: a "free" attribute of one entity has values in both the
  Python and JavaScript docs, and the two sets share nothing. Neither is wrong on its
  own, but the two docs teach different things, which is how stale examples
  (an old model id in one language) show up.

Severity follows where the values were written. Prose is what readers trust,
so a disagreement between two prose statements ranks above one involving
code, and code-only disagreements rank last.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable

SEVERITY_RANK = {"high": 0, "medium": 1, "low": 2}
KIND_RANK = {"contradiction": 0, "divergence": 1}

_QUOTES = "\"'`"
_SPACE = re.compile(r"\s+")


def normalize_value(value: str) -> str:
    """Compare values without case, surrounding quotes, or spacing noise."""
    text = _SPACE.sub(" ", value.strip()).rstrip(".,;").strip(_QUOTES).strip()
    return text.casefold()


@dataclass
class Finding:
    kind: str
    severity: str
    entity_id: str
    attribute_id: str
    values: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "severity": self.severity,
            "entity_id": self.entity_id,
            "attribute_id": self.attribute_id,
            "values": self.values,
        }


def _site(claim: dict[str, Any]) -> dict[str, Any]:
    return {
        "value": claim["value"],
        "source_path": claim["source_path"],
        "line_start": claim["line_start"],
        "line_end": claim["line_end"],
        "context": claim["context"],
        "quote": claim["quote"],
    }


def _severity(by_value: dict[str, list[dict[str, Any]]]) -> str:
    """high when two different values each have a prose statement behind
    them, medium when at least one does, low when it is all code."""
    prose_values = sum(
        1 for sites in by_value.values() if any(s["context"] == "prose" for s in sites)
    )
    if prose_values >= 2:
        return "high"
    if prose_values == 1:
        return "medium"
    return "low"


def _group(claims: Iterable[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for claim in claims:
        groups[(claim["entity_id"], claim["attribute_id"])].append(claim)
    return groups


def _by_value(claims: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_value: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for claim in claims:
        by_value[normalize_value(claim["value"])].append(_site(claim))
    return dict(by_value)


def detect(claims: list[dict[str, Any]], *, divergence: bool = True) -> list[Finding]:
    """Return findings, most severe first. Claims must carry entity_id,
    attribute_id, language and cardinality (see normalize.normalize_claims)."""
    findings: list[Finding] = []
    for (entity_id, attribute_id), group in _group(claims).items():
        cardinality = group[0]["cardinality"]
        if cardinality == "single":
            for language in sorted({c["language"] for c in group}):
                by_value = _by_value([c for c in group if c["language"] == language])
                if len(by_value) > 1:
                    findings.append(
                        Finding(
                            "contradiction", _severity(by_value), entity_id, attribute_id, by_value
                        )
                    )
        elif divergence:
            python = {normalize_value(c["value"]) for c in group if c["language"] == "python"}
            javascript = {
                normalize_value(c["value"]) for c in group if c["language"] == "javascript"
            }
            if python and javascript and not python & javascript:
                by_value = _by_value(group)
                findings.append(
                    Finding("divergence", _severity(by_value), entity_id, attribute_id, by_value)
                )
    findings.sort(
        key=lambda f: (KIND_RANK[f.kind], SEVERITY_RANK[f.severity], f.entity_id, f.attribute_id)
    )
    return findings


def render_markdown(findings: list[Finding], *, claim_count: int) -> str:
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for finding in findings:
        counts[(finding.kind, finding.severity)] += 1
    lines = [
        "# factdrift report",
        "",
        f"{claim_count} claims, {len(findings)} findings.",
        "",
        "| kind | high | medium | low |",
        "| --- | --- | --- | --- |",
    ]
    for kind in KIND_RANK:
        row = " | ".join(str(counts[(kind, s)]) for s in SEVERITY_RANK)
        lines.append(f"| {kind} | {row} |")
    for finding in findings:
        lines += [
            "",
            f"## {finding.kind} ({finding.severity}): "
            f"`{finding.entity_id}` / `{finding.attribute_id}`",
            "",
        ]
        for sites in finding.values.values():
            for site in sites:
                lines.append(
                    f"- `{site['value']}` in {site['source_path']}:"
                    f"{site['line_start']} ({site['context']})"
                )
    return "\n".join(lines) + "\n"
