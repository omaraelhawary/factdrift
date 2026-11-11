"""Precision and recall of detector findings against a labeled set.

A labeled set is JSONL or CSV, one row per labeled group of claims:

    {"id": "control-01", "label": "real", "sites": ["src/.../deepseek.mdx:73"],
     "entity_id": "...", "attribute_id": "...", "note": "..."}

`label` is "real" or "not-real". `sites` are corpus-relative `path:line`
locations. `entity_id` and `attribute_id` are optional; when present they
must also agree with the finding. In CSV, `sites` is space-separated. Other
fields are carried through untouched.

A finding matches a row when one of the finding's value sites covers one of
the row's sites. Only findings with a site in a file the labeled set names
are scored, so a set that covers ten files is not charged for findings in
the other fifty-three.

Precision is the share of scored findings that match a real row. Recall is
the share of real rows matched by at least one finding.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

LABELS = ("real", "not-real")


class LabelError(ValueError):
    pass


def _site(text: str) -> tuple[str, int]:
    path, sep, line = str(text).rpartition(":")
    if not sep or not path or not line.isdigit():
        raise LabelError(f"site {text!r} is not path:line")
    return path, int(line)


def _raw_rows(path: Path) -> list[tuple[int, dict[str, Any]]]:
    if path.suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            return [(reader.line_num, {**row, "sites": (row.get("sites") or "").split()})
                    for row in reader]
    raw = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise LabelError(f"{path}:{number}: not JSON ({error.msg})") from None
        if not isinstance(row, dict):
            raise LabelError(f"{path}:{number}: expected an object")
        raw.append((number, row))
    return raw


def load_labels(
    path: Path, label_field: str = "label", note_field: str = "note"
) -> list[dict[str, Any]]:
    """Read a labeled set (JSONL, or CSV with space-separated sites),
    rejecting rows the scorer cannot use. `label_field` and `note_field` pick
    the columns, so a second labeling pass can be scored on its own. Rows with an empty label
    are skipped, so a half-labeled file can be scored."""
    rows = []
    for number, row in _raw_rows(path):
        label = (row.get(label_field) or "").strip()
        if not label:
            continue
        if label not in LABELS:
            raise LabelError(f"{path}:{number}: label {label!r} is not one of {LABELS}")
        sites = row.get("sites")
        if not isinstance(sites, list) or not sites:
            raise LabelError(f"{path}:{number}: sites must be a non-empty list")
        try:
            parsed = [_site(s) for s in sites]
        except LabelError as error:
            raise LabelError(f"{path}:{number}: {error}") from None
        rows.append({**row, "id": row.get("id") or f"line-{number}", "label": label,
                     "note": row.get(note_field),
                     "_sites": parsed})
    return rows


def _finding_sites(finding: dict[str, Any]) -> list[dict[str, Any]]:
    return [site for sites in finding["values"].values() for site in sites]


def _matches(finding: dict[str, Any], row: dict[str, Any]) -> bool:
    for key in ("entity_id", "attribute_id"):
        if row.get(key) and row[key] != finding[key]:
            return False
    return any(
        site["source_path"] == path and site["line_start"] <= line <= site["line_end"]
        for site in _finding_sites(finding)
        for path, line in row["_sites"]
    )


@dataclass
class Score:
    true_positives: list[dict[str, Any]] = field(default_factory=list)
    false_positives: list[dict[str, Any]] = field(default_factory=list)
    missed: list[dict[str, Any]] = field(default_factory=list)
    real: int = 0
    found: int = 0
    out_of_scope: int = 0

    @property
    def precision(self) -> float | None:
        scored = len(self.true_positives) + len(self.false_positives)
        return len(self.true_positives) / scored if scored else None

    @property
    def recall(self) -> float | None:
        return self.found / self.real if self.real else None


def score(findings: list[dict[str, Any]], labels: list[dict[str, Any]]) -> Score:
    files = {path for row in labels for path, _ in row["_sites"]}
    real = [row for row in labels if row["label"] == "real"]
    result = Score(real=len(real))
    matched_real: set[str] = set()
    for finding in findings:
        if not any(s["source_path"] in files for s in _finding_sites(finding)):
            result.out_of_scope += 1
            continue
        rows = [row for row in labels if _matches(finding, row)]
        hits = [row for row in rows if row["label"] == "real"]
        if hits:
            matched_real.update(row["id"] for row in hits)
            result.true_positives.append({"finding": finding, "labels": hits})
        else:
            notes = [row.get("note") for row in rows if row.get("note")]
            reason = "; ".join(notes) if notes else "matches no row labeled real"
            result.false_positives.append({"finding": finding, "reason": reason})
    result.found = len(matched_real)
    result.missed = [row for row in real if row["id"] not in matched_real]
    return result


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0%}"


def _values(finding: dict[str, Any]) -> str:
    parts = []
    for key, sites in finding["values"].items():
        where = ", ".join(f"{s['source_path'].rsplit('/', 2)[-2]}/"
                          f"{s['source_path'].rsplit('/', 1)[-1]}:{s['line_start']}"
                          for s in sites)
        parts.append(f"`{key}` ({where})")
    return "; ".join(parts)


def render(result: Score, name: str) -> str:
    tp, fp = len(result.true_positives), len(result.false_positives)
    lines = [
        f"## Scoring against {name}",
        "",
        f"Precision {_pct(result.precision)} ({tp} of {tp + fp} findings match a real "
        f"label). Recall {_pct(result.recall)} ({result.found} of {result.real} real "
        f"labels found). {result.out_of_scope} findings were outside the labeled files.",
        "",
        f"### False positives ({fp})",
        "",
    ]
    if fp:
        lines += ["| Finding | Values | Why it fired |", "| --- | --- | --- |"]
        for item in result.false_positives:
            f = item["finding"]
            lines.append(
                f"| {f['kind']}, {f['severity']}: {f['entity_id']}/{f['attribute_id']} "
                f"| {_values(f)} | {item['reason']} |"
            )
    else:
        lines.append("None.")
    lines += ["", f"### Missed ({len(result.missed)})", ""]
    if result.missed:
        for row in result.missed:
            sites = ", ".join(f"{p}:{n}" for p, n in row["_sites"])
            lines.append(f"- {row['id']} at {sites}: {row.get('note', '')}".rstrip(": "))
    else:
        lines.append("None.")
    return "\n".join(lines) + "\n"


def as_dict(result: Score) -> dict[str, Any]:
    strip = lambda row: {k: v for k, v in row.items() if k != "_sites"}  # noqa: E731
    return {
        "precision": result.precision,
        "recall": result.recall,
        "real": result.real,
        "found": result.found,
        "out_of_scope": result.out_of_scope,
        "true_positives": [
            {"finding": t["finding"], "labels": [strip(r) for r in t["labels"]]}
            for t in result.true_positives
        ],
        "false_positives": result.false_positives,
        "missed": [strip(r) for r in result.missed],
    }
