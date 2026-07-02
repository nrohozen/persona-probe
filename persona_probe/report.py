"""Render results to the console and (optionally) a Markdown snapshot."""

from __future__ import annotations

from dataclasses import dataclass

from .judge import Verdict

_MARK = {"HELD": "PASS", "BROKEN": "FAIL", "UNCLEAR": "????"}


@dataclass
class Result:
    persona: str
    probe_id: str
    rule: str
    reply: str
    verdict: Verdict


def console_report(results: list[Result], verbose: bool = False) -> str:
    lines: list[str] = []
    by_persona: dict[str, list[Result]] = {}
    for r in results:
        by_persona.setdefault(r.persona, []).append(r)

    for persona, group in by_persona.items():
        lines.append(f"\n{persona}")
        for r in group:
            mark = _MARK[r.verdict.verdict]
            lines.append(f"  [{mark}] {r.probe_id}: {r.verdict.why}")
            if verbose:
                lines.append(f"        rule: {r.rule}")
                snippet = r.reply.replace("\n", " ")
                if len(snippet) > 200:
                    snippet = snippet[:200] + "..."
                lines.append(f"        reply: {snippet}")

    held = sum(r.verdict.verdict == "HELD" for r in results)
    broken = sum(r.verdict.verdict == "BROKEN" for r in results)
    unclear = sum(r.verdict.verdict == "UNCLEAR" for r in results)
    lines.append(
        f"\n{len(results)} probes: {held} held, {broken} broken, {unclear} unclear"
    )
    return "\n".join(lines)


def markdown_report(results: list[Result], responder: str, judge: str) -> str:
    held = sum(r.verdict.verdict == "HELD" for r in results)
    broken = sum(r.verdict.verdict == "BROKEN" for r in results)
    unclear = sum(r.verdict.verdict == "UNCLEAR" for r in results)

    out = ["# persona-probe report", ""]
    out.append(f"- responder: `{responder}`")
    out.append(f"- judge: `{judge}` (different family)")
    out.append(f"- result: **{held} held, {broken} broken, {unclear} unclear**")
    out.append("")
    out.append("| persona | probe | verdict | why |")
    out.append("|---|---|---|---|")
    for r in results:
        why = r.verdict.why.replace("|", "\\|")
        out.append(f"| {r.persona} | {r.probe_id} | {r.verdict.verdict} | {why} |")
    out.append("")
    return "\n".join(out)


def has_failures(results: list[Result]) -> bool:
    return any(r.verdict.verdict == "BROKEN" for r in results)
