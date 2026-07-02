"""Backend construction and request-shaping tests — no network.

We monkeypatch the one HTTP primitive (`backends._post`) to capture what each
backend would send and to inject canned responses/errors, so the URL, headers,
payload, and error translation are all checked without a running server.
"""

from __future__ import annotations

import pytest

from persona_probe import backends
from persona_probe.backends import (
    BackendError,
    OllamaBackend,
    OpenAIBackend,
    _HTTPStatus,
    make_backend,
)

MESSAGES = [{"role": "user", "content": "hi"}]


def test_make_backend_selects_class_and_default_host():
    o = make_backend("ollama")
    assert isinstance(o, OllamaBackend)
    assert o.host == backends.OLLAMA_DEFAULT_HOST

    a = make_backend("openai")
    assert isinstance(a, OpenAIBackend)
    assert a.host == backends.OPENAI_DEFAULT_HOST


def test_make_backend_strips_trailing_slash_and_rejects_unknown():
    assert make_backend("openai", host="http://x/v1/").host == "http://x/v1"
    with pytest.raises(ValueError):
        make_backend("bogus")


def test_ollama_chat_hits_api_chat_with_deterministic_options(monkeypatch):
    seen = {}

    def fake_post(url, payload, headers, timeout):
        seen.update(url=url, payload=payload, headers=headers)
        return {"message": {"content": "  hello  "}}

    monkeypatch.setattr(backends, "_post", fake_post)
    out = OllamaBackend(host="http://h:11434").chat("qwen2.5:7b", MESSAGES)

    assert out == "hello"  # stripped
    assert seen["url"] == "http://h:11434/api/chat"
    assert seen["payload"]["options"] == {"temperature": 0.0, "seed": 0}
    assert seen["payload"]["stream"] is False
    assert "Authorization" not in seen["headers"]


def test_openai_chat_hits_chat_completions_and_sends_bearer(monkeypatch):
    seen = {}

    def fake_post(url, payload, headers, timeout):
        seen.update(url=url, payload=payload, headers=headers)
        return {"choices": [{"message": {"content": "hi there"}}]}

    monkeypatch.setattr(backends, "_post", fake_post)
    out = OpenAIBackend(host="http://h:1234/v1", api_key="sk-test").chat("gpt", MESSAGES)

    assert out == "hi there"
    assert seen["url"] == "http://h:1234/v1/chat/completions"
    assert seen["headers"]["Authorization"] == "Bearer sk-test"
    assert seen["payload"]["temperature"] == 0.0
    assert seen["payload"]["seed"] == 0


def test_openai_without_key_sends_no_auth_header(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        backends,
        "_post",
        lambda url, payload, headers, timeout: seen.update(headers=headers)
        or {"choices": [{"message": {"content": "x"}}]},
    )
    OpenAIBackend(host="http://h/v1").chat("m", MESSAGES)
    assert "Authorization" not in seen["headers"]


def test_ollama_404_gives_pull_hint(monkeypatch):
    def boom(*a, **k):
        raise _HTTPStatus(404, "not found")

    monkeypatch.setattr(backends, "_post", boom)
    with pytest.raises(BackendError, match="ollama pull"):
        OllamaBackend().chat("missing:7b", MESSAGES)


def test_openai_401_points_at_api_key(monkeypatch):
    def boom(*a, **k):
        raise _HTTPStatus(401, "nope")

    monkeypatch.setattr(backends, "_post", boom)
    with pytest.raises(BackendError, match="OPENAI_API_KEY"):
        OpenAIBackend().chat("gpt", MESSAGES)


def test_empty_response_is_an_error(monkeypatch):
    monkeypatch.setattr(
        backends, "_post", lambda *a, **k: {"choices": [{"message": {"content": ""}}]}
    )
    with pytest.raises(BackendError, match="empty response"):
        OpenAIBackend().chat("gpt", MESSAGES)
