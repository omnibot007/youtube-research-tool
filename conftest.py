"""Make the test suite hermetic.

Added 2026-09-02 after a real failure: the suite passed on a machine with no
Gemini key and failed on the same commit once one was exported. Three tests
that assert the Ollama failure path were silently exercising the Gemini path
instead, because `vision_available()` dispatches on whether a key exists.

A suite whose result depends on the developer's shell is not a suite. Every
test now starts from a known-empty provider environment; the ones that WANT
Gemini set it themselves with monkeypatch, which runs after this fixture and
therefore still wins.
"""

import pytest

_PROVIDER_ENV = (
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "YT_GEMINI_API_KEY",
    "YT_VISION_PROVIDER",
    "YT_VISION_PROFILE",
    "YT_VISUAL_MODE",
    "YT_GEMINI_ENDPOINT",
    "YT_GEMINI_MODEL",
    "YT_GEMINI_AUTH",
    "YT_GEMINI_SEND_REVISION",
    "YT_GEMINI_MIN_INTERVAL",
    "YT_VISION_MODEL",
    "YT_OLLAMA_HOST",
)


@pytest.fixture(autouse=True)
def _isolated_provider_env(monkeypatch):
    """Strip provider config so tests never inherit a real environment."""
    for name in _PROVIDER_ENV:
        monkeypatch.delenv(name, raising=False)
    try:
        import visual
    except Exception:                      # a test module that never uses it
        return
    # Constants are read at import time, so clearing the environment is not
    # enough on its own. Default to the local path; Gemini tests opt in.
    monkeypatch.setattr(visual, "VISION_PROVIDER", "auto", raising=False)
    monkeypatch.setattr(visual, "GEMINI_AUTH_STYLE", "x-goog", raising=False)
    monkeypatch.setattr(visual, "GEMINI_SEND_REVISION", True, raising=False)
    monkeypatch.setattr(visual, "GEMINI_MIN_INTERVAL", 0.0, raising=False)
