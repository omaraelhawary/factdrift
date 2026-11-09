"""The gold-set labeling file: one CSV row per detect.candidates() group.

Each row lists every value with its file, line, context and quote, says
whether `detect` reported it, and leaves `label`/`note` and
`label_2`/`note_2` empty for two labeling passes. score.load_labels reads
the file back.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Any

COLUMNS = ["id", "scope", "entity_id", "attribute_id", "cardinality", "detector",
           "values", "sites", "label", "note", "label_2", "note_2"]


def _short(path: str) -> str:
    parts = path.split("/")
    return "/".join(parts[-4:-3] + parts[-1:])  # python/groq.mdx


def _sites(values: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    return [s for sites in values.values() for s in sites]


def _detector(candidate: dict[str, Any], findings: list[dict[str, Any]]) -> str:
    here = {(s["source_path"], s["line_start"]) for s in _sites(candidate["values"])}
    hits = sorted({
        f"{f['kind']} {f['severity']}" for f in findings
        if (f["entity_id"], f["attribute_id"]) == (candidate["entity_id"], candidate["attribute_id"])
        and any((s["source_path"], s["line_start"]) in here for s in _sites(f["values"]))
    })
    return "; ".join(hits)


def rows(candidates: list[dict[str, Any]], findings: list[dict[str, Any]]) -> list[dict[str, str]]:
    out = []
    for c in candidates:
        key = "|".join([c["scope"], c["entity_id"], c["attribute_id"]])
        values = "\n".join(
            f"{value}\n" + "\n".join(
                f"  {_short(s['source_path'])}:{s['line_start']} [{s['context']}] {s['quote']}"
                for s in sites)
            for value, sites in c["values"].items()
        )
        out.append({
            "id": "g-" + hashlib.sha256(key.encode()).hexdigest()[:8],
            "scope": c["scope"], "entity_id": c["entity_id"], "attribute_id": c["attribute_id"],
            "cardinality": c["cardinality"], "detector": _detector(c, findings),
            "values": values,
            "sites": " ".join(f"{s['source_path']}:{s['line_start']}" for s in _sites(c["values"])),
            "label": "", "note": "", "label_2": "", "note_2": "",
        })
    return out


def write_csv(table: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(table)
