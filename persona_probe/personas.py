"""Parse Jekyll persona files into (metadata, system_prompt) pairs.

A persona file on the site looks like:

    ---
    title: "Rubber Duck"
    tags: [debugging, thinking]
    description: "..."
    ---

    You are a rubber duck debugger: ...

    You do not:
    - Jump straight to a solution
    ...

The YAML frontmatter is metadata; everything after the closing `---` is the
system prompt we actually send to the model. We keep the body verbatim — the
persona is only being tested as written.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

_FRONTMATTER = re.compile(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", re.DOTALL)


@dataclass
class Persona:
    name: str  # slug, from the filename (matches a probe's `persona:` field)
    system_prompt: str  # the body after frontmatter, verbatim
    meta: dict = field(default_factory=dict)

    @property
    def title(self) -> str:
        return self.meta.get("title", self.name)

    def do_not_rules(self) -> list[str]:
        """Best-effort extraction of the 'You do not:' bullet list.

        Handy for scaffolding probes, and for reminding a human which rules
        currently have no probe covering them. Never used for grading.
        """
        lines = self.system_prompt.splitlines()
        rules: list[str] = []
        capturing = False
        for line in lines:
            stripped = line.strip()
            if re.match(r"(?i)you do not:?\s*$", stripped):
                capturing = True
                continue
            if capturing:
                if stripped.startswith(("-", "*")):
                    rules.append(stripped.lstrip("-* ").strip())
                elif stripped == "":
                    continue
                else:
                    break  # list ended
        return rules


def parse_persona(path: Path) -> Persona:
    text = path.read_text(encoding="utf-8")
    match = _FRONTMATTER.match(text)
    if match:
        meta = yaml.safe_load(match.group(1)) or {}
        body = match.group(2).strip()
    else:
        # No frontmatter — treat the whole file as the system prompt.
        meta = {}
        body = text.strip()
    if not isinstance(meta, dict):
        raise ValueError(f"{path.name}: frontmatter is not a mapping")
    if not body:
        raise ValueError(f"{path.name}: empty system prompt")
    return Persona(name=path.stem, system_prompt=body, meta=meta)


def load_personas(directory: Path) -> dict[str, Persona]:
    """Load every *.md persona in a directory, keyed by slug."""
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"personas directory not found: {directory}")
    personas: dict[str, Persona] = {}
    for path in sorted(directory.glob("*.md")):
        personas[path.stem] = parse_persona(path)
    if not personas:
        raise ValueError(f"no persona files (*.md) found in {directory}")
    return personas
