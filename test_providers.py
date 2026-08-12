"""Unit tests for providers.py — all network is mocked."""
from __future__ import annotations

import json
import urllib.error
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest

import providers


def _fake_response(text: str) -> MagicMock:
    resp = MagicMock()
    resp.read.return_value = json.dumps({"choices": [{"message": {"content": text}}]}).encode()
    resp.__enter__.return_value = resp
    resp.__exit__.return_value = False
    return resp


def _fake_local_response(text: str) -> MagicMock:
    resp = MagicMock()
    resp.read.return_value = json.dumps({"response": text}).encode()
    resp.__enter__.return_value = resp
    resp.__exit__.return_value = False
    return resp


def test_load_chain_defaults_skip_missing_keys(monkeypatch):
    monkeypatch.delenv("YT_GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("YT_GROQ_API_KEY", raising=False)
    monkeypatch.setenv("YT_OPENROUTER_API_KEY", "or-key")
    chain = providers.load_chain()
    assert [p["name"] for p in chain] == ["openrouter", "local"]
    assert chain[0]["model"] == "google/gemma-4-26b-a4b-it:free"


def test_load_chain_custom_order_and_env_override(monkeypatch):
    monkeypatch.setenv("YT_GROQ_API_KEY", "g-key")
    monkeypatch.setenv("YT_GROQ_MODEL", "custom-groq-model")
    chain = providers.load_chain("groq,openrouter")
    assert [p["name"] for p in chain] == ["groq"]
    assert chain[0]["model"] == "custom-groq-model"


def test_request_provider_sends_authorization_for_hosted():
    provider = {"name": "openrouter", "base_url": "https://openrouter.ai/api/v1", "api_key": "secret-or", "model": "m"}
    with patch.object(providers.urllib.request, "urlopen") as mock_urlopen:
        mock_urlopen.return_value = _fake_response("hi")
        providers.request_provider("hello", provider=provider)
    call_args = mock_urlopen.call_args
    req = call_args[0][0]
    assert req.headers["Authorization"] == "Bearer secret-or"
    assert req.headers.get("Http-referer") == "https://devin.ai"
    body = json.loads(req.data)
    assert body["model"] == "m"
    assert body["messages"][0]["content"] == "hello"


def test_request_provider_no_auth_for_local():
    provider = {"name": "local", "host": "http://localhost:11434", "model": "llama"}
    with patch.object(providers.urllib.request, "urlopen") as mock_urlopen:
        mock_urlopen.return_value = _fake_local_response("local hi")
        providers.request_provider("hello", provider=provider)
    req = mock_urlopen.call_args[0][0]
    assert "Authorization" not in req.headers
    body = json.loads(req.data)
    assert body["model"] == "llama"


def test_request_provider_image_content(monkeypatch):
    monkeypatch.setenv("YT_GEMINI_API_KEY", "g-key")
    chain = providers.load_chain("gemini")
    provider = chain[0]
    with patch.object(providers.urllib.request, "urlopen") as mock_urlopen:
        mock_urlopen.return_value = _fake_response("image ok")
        providers.request_provider("describe", image_b64="abc", provider=provider)
    body = json.loads(mock_urlopen.call_args[0][0].data)
    content = body["messages"][0]["content"]
    assert len(content) == 2
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,abc")


def test_try_chain_failover_after_429(monkeypatch):
    monkeypatch.setenv("YT_GEMINI_API_KEY", "g-key")
    monkeypatch.setenv("YT_OPENROUTER_API_KEY", "o-key")
    chain = providers.load_chain("gemini,openrouter")
    calls = []
    def side_effect(req, **kwargs):
        calls.append(req)
        if "generativelanguage" in req.full_url:
            raise urllib.error.HTTPError(req.full_url, 429, "Too Many", {}, BytesIO(b"{}"))
        return _fake_response("from openrouter")
    with patch.object(providers.urllib.request, "urlopen", side_effect=side_effect):
        monkeypatch.setattr(providers.time, "sleep", lambda s: None)
        answer, used = providers.try_chain("hi", chain=chain)
    assert used["name"] == "openrouter"
    assert answer == "from openrouter"
    # 429 retries 3 times before moving on.
    assert sum("generativelanguage" in r.full_url for r in calls) == 3


def test_try_chain_auth_error_advances(monkeypatch):
    monkeypatch.setenv("YT_GEMINI_API_KEY", "bad")
    monkeypatch.setenv("YT_OPENROUTER_API_KEY", "o-key")
    chain = providers.load_chain("gemini,openrouter")
    def side_effect(req, **kwargs):
        if "generativelanguage" in req.full_url:
            raise urllib.error.HTTPError(req.full_url, 403, "Forbidden", {}, BytesIO(b"{}"))
        return _fake_response("or works")
    with patch.object(providers.urllib.request, "urlopen", side_effect=side_effect):
        answer, used = providers.try_chain("hi", chain=chain)
    assert used["name"] == "openrouter"
    assert answer == "or works"


def test_try_chain_exhausted_raises(monkeypatch):
    monkeypatch.setenv("YT_GEMINI_API_KEY", "bad")
    chain = providers.load_chain("gemini")
    with patch.object(providers.urllib.request, "urlopen") as mock_urlopen:
        mock_urlopen.side_effect = urllib.error.HTTPError("url", 500, "err", {}, BytesIO(b"{}"))
        with pytest.raises(RuntimeError, match="all providers failed"):
            providers.try_chain("hi", chain=chain)


def test_keys_not_printed_to_stderr(capfd, monkeypatch):
    monkeypatch.setenv("YT_OPENROUTER_API_KEY", "super-secret-key-123")
    with patch.object(providers.urllib.request, "urlopen") as mock_urlopen:
        mock_urlopen.return_value = _fake_response("ok")
        providers.request_provider("hi", provider=providers.load_chain("openrouter")[0])
    captured = capfd.readouterr()
    assert "super-secret-key-123" not in captured.err
    assert "super-secret-key-123" not in captured.out
