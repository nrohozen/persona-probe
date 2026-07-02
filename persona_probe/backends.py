"""Chat backends, built on stdlib urllib. No SDK, no extra dependency.

persona-probe only ever needs one operation: send a list of chat messages to a
named model and get the assistant's text back, deterministically. Two backends
cover essentially every local or hosted runner:

  - `ollama`  -> Ollama's native POST /api/chat
  - `openai`  -> the OpenAI-compatible POST /v1/chat/completions, which is what
                 LM Studio, llama.cpp's server, vLLM, OpenRouter, Together, and
                 the OpenAI API all speak.

Both expose the same `chat(model, messages) -> str`, so runner.py and judge.py
don't care which one they're handed. Determinism (temperature 0, fixed seed) is
kept identical across backends so a regression is real, not sampling noise.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Protocol

OLLAMA_DEFAULT_HOST = "http://localhost:11434"
# LM Studio's default local server. llama.cpp uses :8080/v1, the OpenAI API uses
# https://api.openai.com/v1 — override with --host for those.
OPENAI_DEFAULT_HOST = "http://localhost:1234/v1"


class BackendError(RuntimeError):
    pass


# Back-compat alias: earlier versions raised OllamaError.
OllamaError = BackendError


class Backend(Protocol):
    def chat(self, model: str, messages: list[dict]) -> str:
        """Send chat `messages` to `model`; return the assistant's text."""
        ...


class _HTTPStatus(BackendError):
    """Internal: an HTTP error carrying its status code so a backend can craft
    a message (e.g. a model-not-found hint) before it reaches the CLI."""

    def __init__(self, code: int, detail: str):
        super().__init__(f"HTTP {code}: {detail}")
        self.code = code
        self.detail = detail


def _post(url: str, payload: dict, headers: dict, timeout: float) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise _HTTPStatus(e.code, e.read().decode("utf-8", "replace")) from e
    except urllib.error.URLError as e:
        raise BackendError(
            f"cannot reach server at {url} ({e.reason}). Is it running?"
        ) from e


@dataclass
class OllamaBackend:
    host: str = OLLAMA_DEFAULT_HOST
    # Deterministic by default: same prompt -> same reply.
    temperature: float = 0.0
    seed: int = 0
    timeout: float = 120.0

    def chat(self, model: str, messages: list[dict]) -> str:
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": self.temperature, "seed": self.seed},
        }
        try:
            body = _post(f"{self.host}/api/chat", payload, {}, self.timeout)
        except _HTTPStatus as e:
            if e.code == 404:
                raise BackendError(
                    f"model '{model}' not found on {self.host}. "
                    f"Pull it first: `ollama pull {model}`"
                ) from e
            raise
        content = (body.get("message") or {}).get("content")
        if not content:
            raise BackendError(f"empty response from '{model}': {body}")
        return content.strip()


@dataclass
class OpenAIBackend:
    # `host` is the API base, ending in /v1 (OpenAI SDK's base_url convention).
    host: str = OPENAI_DEFAULT_HOST
    api_key: str | None = None
    temperature: float = 0.0
    seed: int = 0
    timeout: float = 120.0

    def chat(self, model: str, messages: list[dict]) -> str:
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "temperature": self.temperature,
            "seed": self.seed,
        }
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        try:
            body = _post(
                f"{self.host}/chat/completions", payload, headers, self.timeout
            )
        except _HTTPStatus as e:
            if e.code == 404:
                raise BackendError(
                    f"model '{model}' not found at {self.host}. "
                    "Check the model name your server exposes."
                ) from e
            if e.code == 401:
                raise BackendError(
                    f"unauthorized at {self.host}. Pass --api-key or set "
                    "OPENAI_API_KEY."
                ) from e
            raise
        choices = body.get("choices") or []
        content = (choices[0].get("message") or {}).get("content") if choices else None
        if not content:
            raise BackendError(f"empty response from '{model}': {body}")
        return content.strip()


def make_backend(
    kind: str,
    *,
    host: str | None = None,
    api_key: str | None = None,
    temperature: float = 0.0,
    seed: int = 0,
    timeout: float = 120.0,
) -> Backend:
    """Build a backend by name, filling in a sensible default host."""
    if kind == "ollama":
        return OllamaBackend(
            host=(host or OLLAMA_DEFAULT_HOST).rstrip("/"),
            temperature=temperature,
            seed=seed,
            timeout=timeout,
        )
    if kind == "openai":
        return OpenAIBackend(
            host=(host or OPENAI_DEFAULT_HOST).rstrip("/"),
            api_key=api_key,
            temperature=temperature,
            seed=seed,
            timeout=timeout,
        )
    raise ValueError(f"unknown backend: {kind!r}")
