# persona-probe

`pytest` for persona system prompts.

A persona is a system prompt with a job. The interesting part of a good persona
isn't what it *does* — it's what it refuses to do. Every persona on
[Nick's Notes](https://nrohozen.github.io/personas/) has a **"You do not"**
section, and that section is usually doing most of the work: *don't* jump to a
solution, *don't* praise before critiquing, *don't* ask three questions at once.

The problem is that those constraints are the first thing to quietly rot. You
edit a persona, a phrase softens, a model updates underneath you — and the
persona starts answering when it was supposed to ask. You don't notice, because
you only read the prompt; you never *test the behavior*.

`persona-probe` tests the behavior. It reads your actual persona files, fires a
small battery of **probes** at each one through a local model, and grades
whether the persona's own rules held. A misbehaving persona becomes a failing
test — which is the whole point of
[iterating on prompts like code](https://nrohozen.github.io/approach/).

## The one design decision that matters

**The judge is a different model family than the responder.**

If you have `qwen` answer as the persona and then ask `qwen` whether it followed
the rules, you've let the model grade its own homework — and correlated mistakes
sail straight through. So the responder and the judge are pinned to different
families (config: `responder` vs `judge`). It's the
[Tilda council principle](https://github.com/nrohozen) — independent judges whose
errors don't correlate — shrunk down to a two-model panel. It is also the exact
caveat from [Evals That Mean Something](https://nrohozen.github.io/notes/evals-that-mean-something/):
an LLM judge is fine, right up until it's grading itself.

The verdict is deliberately blunt: each probe returns **HELD**, **BROKEN**, or
**UNCLEAR**, with a one-line rationale. `UNCLEAR` is a first-class outcome, not a
rounding error — a judge that isn't sure should say so rather than flip a coin.

## What it does *not* do

- It does not score persona "quality." Quality is taste; rule-adherence is
  testable. It only checks the second.
- It does not call a hosted API. Everything runs against local
  [Ollama](https://ollama.com), with deterministic params (`temperature: 0`, a
  fixed seed) so a run is reproducible and a regression is real, not sampling
  noise. Same discipline as [`model-bench`](https://github.com/nrohozen).
- It does not hardcode anything about you. It ships with a set of sample
  personas so it runs out of the box, but you can point it at any directory of
  persona files and any pair of models.

## Install

You need Python 3.10+ and a running [Ollama](https://ollama.com).

```bash
git clone https://github.com/nrohozen/persona-probe
cd persona-probe
python -m venv .venv
# Linux/macOS:  source .venv/bin/activate
# Windows:      .venv\Scripts\activate
pip install -e .
```

<details>
<summary>Windows PowerShell, without activating the venv</summary>

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -e .
```

Then prefix the commands below with `.venv\Scripts\python -m persona_probe` and
use backtick line-continuations instead of `\`.
</details>

Pull two models from **different families** — one answers, the other grades:

```bash
ollama pull qwen2.5:7b      # responder
ollama pull llama3.1:8b     # judge (different family on purpose)
```

## Run

The repo ships with sample personas in `personas/` and matching probes in
`probes/`, so a fresh clone runs with no arguments:

```bash
# probe every bundled persona against the default model pair
persona-probe

# one persona, verbose (prints each turn and the judge's reasoning)
persona-probe --only rubber-duck -v

# write a Markdown report you can commit as a behavioral snapshot
persona-probe -o report.md

# point it at your own personas and models
persona-probe --personas path/to/_personas --responder mistral:7b --judge gemma2:9b
```

(No console script? The equivalent is `python -m persona_probe`.)

Exit code is non-zero if any probe comes back `BROKEN`, so you can wire it into
CI or a pre-commit hook and treat a persona regression like a failing build.

## How a probe is written

A probe is a YAML file named after the persona it targets. Each entry pins one
rule — ideally lifted verbatim from that persona's "You do not" section — to a
short conversation and a single yes/no question the judge must answer.

```yaml
persona: rubber-duck            # matches _personas/rubber-duck.md
probes:
  - id: no-instant-solution
    rule: "Do not jump straight to a solution."
    turns:
      - "My API returns 500s under load. How do I fix it?"
    # The judge answers THIS, seeing only the rule, the turns, and the reply.
    # Phrase it so that 'yes' == the persona behaved.
    ask: >
      Did the assistant respond mainly by asking a clarifying question to
      understand the problem, rather than by offering fixes or a solution list?
```

The judge sees only the rule, the conversation, and `ask`. It never sees the
full persona prompt — so it grades the *observed behavior*, not its own guess at
what the author intended.

## Layout

```
persona_probe/
  personas.py   parse Jekyll persona files (frontmatter + system-prompt body)
  probes.py     load + validate probe YAML
  ollama.py     thin, dependency-light Ollama chat client (stdlib urllib)
  runner.py     drive the conversation as the persona, capture the reply
  judge.py      grade one reply against one rule with the OTHER model
  report.py     console + Markdown rendering
  cli.py        wiring
personas/       sample persona *.md files (bring your own to test real ones)
probes/         one YAML per persona
tests/          deterministic tests for the parts that don't need a model
```

## The seam this is really about

Persona files are prose, and prose drifts. This tool puts a **behavioral
contract** under the prose: the "You do not" lines stop being aspirational
comments and become assertions something actually checks. Knowledge (what the
persona should do) lives in editable YAML; judgment (did it?) is spent at the
boundary on a small local model; and no single model is trusted to both perform
and grade. That's the same shape as everything else in this stack — just aimed
at the prompts instead of the code.

## License

[MIT](LICENSE). Use it, fork it, point it at your own personas.
