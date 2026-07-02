"""Load and validate probe files.

A probe file is one YAML document per persona:

    persona: rubber-duck
    probes:
      - id: no-instant-solution
        rule: "Do not jump straight to a solution."
        turns:
          - "My API returns 500s under load. How do I fix it?"
        ask: >
          Did the assistant respond mainly by asking a clarifying question...

`turns` is a list of user messages, played in order; the persona replies after
the last one, and that final reply is what gets graded. `ask` is the single
yes/no question put to the judge, phrased so 'yes' means the persona behaved.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class Probe:
    id: str
    persona: str
    rule: str
    turns: list[str]
    ask: str


def _require(d: dict, key: str, ctx: str) -> object:
    if key not in d or d[key] in (None, "", []):
        raise ValueError(f"{ctx}: missing required field '{key}'")
    return d[key]


def parse_probe_file(path: Path) -> list[Probe]:
    doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(doc, dict):
        raise ValueError(f"{path.name}: top level must be a mapping")
    persona = _require(doc, "persona", path.name)
    raw_probes = _require(doc, "probes", path.name)
    if not isinstance(raw_probes, list):
        raise ValueError(f"{path.name}: 'probes' must be a list")

    probes: list[Probe] = []
    seen_ids: set[str] = set()
    for i, raw in enumerate(raw_probes):
        ctx = f"{path.name}[probe {i}]"
        if not isinstance(raw, dict):
            raise ValueError(f"{ctx}: each probe must be a mapping")
        pid = str(_require(raw, "id", ctx))
        if pid in seen_ids:
            raise ValueError(f"{path.name}: duplicate probe id '{pid}'")
        seen_ids.add(pid)
        turns = _require(raw, "turns", ctx)
        if not isinstance(turns, list) or not all(isinstance(t, str) for t in turns):
            raise ValueError(f"{ctx}: 'turns' must be a list of strings")
        probes.append(
            Probe(
                id=pid,
                persona=str(persona),
                rule=str(_require(raw, "rule", ctx)),
                turns=turns,
                ask=str(_require(raw, "ask", ctx)),
            )
        )
    return probes


def load_probes(directory: Path) -> list[Probe]:
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"probes directory not found: {directory}")
    probes: list[Probe] = []
    for path in sorted(directory.glob("*.yaml")) + sorted(directory.glob("*.yml")):
        probes.extend(parse_probe_file(path))
    if not probes:
        raise ValueError(f"no probe files (*.yaml) found in {directory}")
    return probes
