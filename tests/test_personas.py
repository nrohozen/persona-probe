"""Deterministic tests — no model required.

Covers the parts that can silently break a real run: persona/frontmatter
parsing, probe validation, the different-family guard, and JSON extraction from
a messy judge reply.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from persona_probe.cli import _family
from persona_probe.judge import _extract_json
from persona_probe.personas import parse_persona
from persona_probe.probes import parse_probe_file

FIXTURES = Path(__file__).parent / "fixtures"


def test_parse_persona_splits_frontmatter_and_body(tmp_path: Path):
    f = tmp_path / "duck.md"
    f.write_text(
        '---\ntitle: "Rubber Duck"\ntags: [a, b]\n---\n\n'
        "You are a duck.\n\nYou do not:\n- Jump to a solution\n- Nag with many questions\n",
        encoding="utf-8",
    )
    p = parse_persona(f)
    assert p.name == "duck"
    assert p.title == "Rubber Duck"
    assert p.system_prompt.startswith("You are a duck.")
    assert "title" not in p.system_prompt  # frontmatter stripped
    assert p.do_not_rules() == ["Jump to a solution", "Nag with many questions"]


def test_parse_persona_without_frontmatter(tmp_path: Path):
    f = tmp_path / "plain.md"
    f.write_text("You are a plain persona with no frontmatter.", encoding="utf-8")
    p = parse_persona(f)
    assert p.meta == {}
    assert p.system_prompt.startswith("You are a plain persona")


def test_empty_persona_rejected(tmp_path: Path):
    f = tmp_path / "empty.md"
    f.write_text("---\ntitle: x\n---\n", encoding="utf-8")
    with pytest.raises(ValueError):
        parse_persona(f)


def test_parse_probe_file_ok(tmp_path: Path):
    f = tmp_path / "duck.yaml"
    f.write_text(
        "persona: duck\nprobes:\n"
        "  - id: p1\n    rule: no solution\n    turns: ['fix my bug']\n    ask: did it ask?\n",
        encoding="utf-8",
    )
    probes = parse_probe_file(f)
    assert len(probes) == 1
    assert probes[0].persona == "duck"
    assert probes[0].turns == ["fix my bug"]


def test_probe_missing_field_rejected(tmp_path: Path):
    f = tmp_path / "bad.yaml"
    f.write_text("persona: duck\nprobes:\n  - id: p1\n    turns: ['x']\n", encoding="utf-8")
    with pytest.raises(ValueError):
        parse_probe_file(f)


def test_probe_duplicate_id_rejected(tmp_path: Path):
    f = tmp_path / "dup.yaml"
    f.write_text(
        "persona: duck\nprobes:\n"
        "  - id: p1\n    rule: r\n    turns: ['x']\n    ask: q\n"
        "  - id: p1\n    rule: r\n    turns: ['y']\n    ask: q\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        parse_probe_file(f)


@pytest.mark.parametrize(
    "model,expected",
    [
        ("qwen2.5:7b", "qwen2.5"),
        ("llama3.1:8b", "llama3.1"),
        ("registry.io/library/mistral:latest", "mistral"),
        ("phi3", "phi3"),
    ],
)
def test_family_key(model, expected):
    assert _family(model) == expected


def test_extract_json_from_fenced_prose():
    raw = 'Sure!\n```json\n{"verdict": "HELD", "why": "asked a question"}\n```'
    obj = _extract_json(raw)
    assert obj["verdict"] == "HELD"


def test_extract_json_returns_none_on_garbage():
    assert _extract_json("no json here at all") is None
