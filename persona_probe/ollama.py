"""A thin Ollama chat client built on stdlib urllib.

No SDK, no extra dependency. We only need /api/chat with deterministic options,
and keeping it small means the tool has one real dependency (PyYAML) and a
failure here is easy to read.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass

DEFAULT_HOST = "http://localhost:11434"


class OllamaError(RuntimeError):
    pass


@dataclass
class Client:
    host: str = DEFAULT_HOST
    # Deterministic by default: same prompt -> same reply, so a regression is
    # real and not sampling noise. Same discipline as model-bench.
    temperature: float = 0.0
    seed: int = 0
    timeout: float = 120.0

    def chat(self, model: str, messages: list[dict]) -> str:
        """Send a chat and return the assistant's text.

        `messages` is a list of {"role": ..., "content": ...} dicts.
        """
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": self.temperature, "seed": self.seed},
        }
        req = urllib.request.Request(
            f"{self.host}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")
            if e.code == 404:
                raise OllamaError(
                    f"model '{model}' not found on {self.host}. "
                    f"Pull it first: `ollama pull {model}`"
                ) from e
            raise OllamaError(f"Ollama HTTP {e.code}: {detail}") from e
        except urllib.error.URLError as e:
            raise OllamaError(
                f"cannot reach Ollama at {self.host} ({e.reason}). Is it running?"
            ) from e

        content = (body.get("message") or {}).get("content")
        if not content:
            raise OllamaError(f"empty response from '{model}': {body}")
        return content.strip()
