import json
from pathlib import Path

CONTROL = Path(__file__).parent.parent / "fixtures" / "control"


def test_every_injection_is_at_its_recorded_line():
    rows = [json.loads(line) for line in (CONTROL / "labels.jsonl").read_text().splitlines()]
    assert len(rows) == 12
    for row in rows:
        (site,) = row["sites"]
        path, line = site.rsplit(":", 1)
        text = (CONTROL / path).read_text().split("\n")[int(line) - 1]
        assert row["injected"] in text, site
        assert row["original"] not in text.replace(row["injected"], ""), site
