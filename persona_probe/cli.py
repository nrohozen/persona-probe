"""persona-probe command line."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .backends import BackendError, make_backend
from .judge import judge_reply
from .personas import load_personas
from .probes import load_probes
from .report import Result, console_report, has_failures, markdown_report
from .runner import run_probe


def _family(model: str) -> str:
    """Crude model-family key: the part before the first ':' or '/'.

    'qwen2.5:7b' -> 'qwen2.5', 'llama3.1:8b' -> 'llama3.1'. Good enough to catch
    the 'judge is grading its own homework' mistake, which is the point.
    """
    return model.split(":")[0].split("/")[-1].strip().lower()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="persona-probe",
        description="Behavioral regression tests for persona system prompts.",
    )
    p.add_argument(
        "--personas",
        type=Path,
        default=Path("./personas"),
        help="directory of persona *.md files (defaults to the bundled samples)",
    )
    p.add_argument(
        "--probes",
        type=Path,
        default=Path("./probes"),
        help="directory of probe *.yaml files",
    )
    p.add_argument("--responder", default="qwen2.5:7b", help="model that plays the persona")
    p.add_argument("--judge", default="llama3.1:8b", help="model that grades (different family)")
    p.add_argument(
        "--backend",
        choices=["ollama", "openai"],
        default="ollama",
        help="ollama (native) or openai (LM Studio, llama.cpp, vLLM, hosted APIs)",
    )
    p.add_argument(
        "--host",
        default=None,
        help="server base URL (default: Ollama :11434, or OpenAI-compatible :1234/v1)",
    )
    p.add_argument(
        "--api-key",
        default=None,
        help="API key for the openai backend (or set OPENAI_API_KEY)",
    )
    p.add_argument("--only", action="append", help="restrict to persona slug(s); repeatable")
    p.add_argument("-o", "--output", type=Path, help="write a Markdown report here")
    p.add_argument("-v", "--verbose", action="store_true", help="print rules and replies")
    p.add_argument(
        "--allow-same-family",
        action="store_true",
        help="skip the different-family check (not recommended)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not args.allow_same_family and _family(args.responder) == _family(args.judge):
        print(
            f"error: responder ({args.responder}) and judge ({args.judge}) look like "
            f"the same family '{_family(args.responder)}'.\n"
            "The whole point is an independent judge. Pick a different judge model, "
            "or pass --allow-same-family if you really mean it.",
            file=sys.stderr,
        )
        return 2

    try:
        personas = load_personas(args.personas)
        probes = load_probes(args.probes)
    except (ValueError, NotADirectoryError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    if args.only:
        wanted = set(args.only)
        probes = [p for p in probes if p.persona in wanted]
        if not probes:
            print(f"error: no probes match --only {sorted(wanted)}", file=sys.stderr)
            return 2

    client = make_backend(
        args.backend,
        host=args.host,
        api_key=args.api_key or os.environ.get("OPENAI_API_KEY"),
    )
    results: list[Result] = []

    for probe in probes:
        persona = personas.get(probe.persona)
        if persona is None:
            print(
                f"warning: probe '{probe.id}' targets unknown persona "
                f"'{probe.persona}' (no {probe.persona}.md); skipping",
                file=sys.stderr,
            )
            continue
        print(f"probing {probe.persona}:{probe.id} ...", file=sys.stderr)
        try:
            reply = run_probe(client, args.responder, persona, probe)
            verdict = judge_reply(client, args.judge, probe, reply)
        except BackendError as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        results.append(Result(probe.persona, probe.id, probe.rule, reply, verdict))

    if not results:
        print("error: nothing was probed", file=sys.stderr)
        return 2

    print(console_report(results, verbose=args.verbose))

    if args.output:
        args.output.write_text(
            markdown_report(results, args.responder, args.judge), encoding="utf-8"
        )
        print(f"\nwrote {args.output}", file=sys.stderr)

    return 1 if has_failures(results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
