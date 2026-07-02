"""Drive a probe's conversation as the persona and capture the final reply."""

from __future__ import annotations

from .backends import Backend
from .personas import Persona
from .probes import Probe


def run_probe(client: Backend, model: str, persona: Persona, probe: Probe) -> str:
    """Play the probe's turns against the persona; return the last reply.

    Only the final assistant turn is graded, but earlier turns are included so
    multi-turn probes (e.g. "does it STILL ask on the second message?") work.
    """
    messages: list[dict] = [{"role": "system", "content": persona.system_prompt}]
    reply = ""
    for turn in probe.turns:
        messages.append({"role": "user", "content": turn})
        reply = client.chat(model, messages)
        messages.append({"role": "assistant", "content": reply})
    return reply
