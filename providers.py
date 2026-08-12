"""Shared provider chain for yt-scrape inference."""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from typing import Any

DEFAULT_CHAIN = "gemini,groq,openrouter,local"
_SPECS = {
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai", "YT_GEMINI_API_KEY", "YT_GEMINI_MODEL", "gemini-2.5-flash-lite"),
    "groq": ("https://api.groq.com/openai/v1", "YT_GROQ_API_KEY", "YT_GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct"),
    "openrouter": ("https://openrouter.ai/api/v1", "YT_OPENROUTER_API_KEY", "YT_OPENROUTER_MODEL", "google/gemma-4-26b-a4b-it:free"),
}


def load_chain(raw: str | None = None) -> list[dict[str, Any]]:
    raw = raw if raw is not None else os.environ.get("YT_PROVIDER_CHAIN", DEFAULT_CHAIN)
    chain: list[dict[str, Any]] = []
    for name in [n.strip() for n in raw.split(",") if n.strip()]:
        if name == "local":
            chain.append({"name": "local", "host": os.environ.get("YT_OLLAMA_HOST", "http://127.0.0.1:11434"), "model": ""})
        elif name in _SPECS:
            spec = _SPECS[name]
            key = os.environ.get(spec[1], "")
            if key:
                chain.append({"name": name, "base_url": spec[0], "api_key": key, "model": os.environ.get(spec[2], spec[3])})
            else:
                print(f"  [providers] skipping {name}: no API key", file=sys.stderr)
    return chain


def request_provider(prompt: str, image_b64: str | None = None, provider: dict[str, Any] | None = None) -> str:
    if not provider:
        raise ValueError("provider is required")
    if provider.get("name") == "local":
        payload: dict[str, Any] = {"model": provider.get("model", ""), "prompt": prompt, "stream": False, "options": provider.get("options", {"temperature": 0})}
        if image_b64:
            payload["images"] = [image_b64]
        if provider.get("format"):
            payload["format"] = provider["format"]
        if provider.get("keep_alive"):
            payload["keep_alive"] = provider["keep_alive"]
        data = json.dumps(payload).encode("utf-8")
        url = provider["host"].rstrip("/") + "/api/generate"
        headers = {"Content-Type": "application/json"}
    else:
        content: Any = prompt
        if image_b64:
            content = [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}]
        data = json.dumps({"model": provider["model"], "messages": [{"role": "user", "content": content}], "temperature": 0, "response_format": {"type": "json_object"}}).encode("utf-8")
        url = provider["base_url"].rstrip("/") + "/chat/completions"
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {provider['api_key']}"}
        if provider.get("name") == "openrouter":
            headers.update({"HTTP-Referer": "https://devin.ai", "X-Title": "yt-scrape"})
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=120) as r:
        body = json.loads(r.read().decode("utf-8"))
    if provider.get("name") == "local":
        return str(body.get("response", "")).strip()
    return str(body["choices"][0]["message"]["content"]).strip()


def try_chain(prompt: str, image_b64: str | None = None, chain: list[dict[str, Any]] | None = None, local_provider: dict[str, Any] | None = None) -> tuple[str, dict[str, Any]]:
    providers = chain if chain is not None else load_chain()
    last_error = ""
    for provider in providers:
        if provider.get("name") == "local" and local_provider:
            provider = {**provider, **local_provider}
        name = provider["name"]
        print(f"  [providers] trying {name}...", file=sys.stderr)
        for attempt in range(3):
            try:
                answer = request_provider(prompt, image_b64, provider)
                print(f"  [providers] {name} served request", file=sys.stderr)
                return answer, provider
            except urllib.error.HTTPError as e:
                last_error = f"{name}: HTTP {e.code}"
                if e.code in (401, 403):
                    break
                if e.code == 429 and attempt < 2:
                    print(f"  [providers] {name} 429, retry in {2 ** attempt}s", file=sys.stderr)
                    time.sleep(2 ** attempt)
                    continue
                if e.code >= 500:
                    break
                break
            except Exception as e:
                last_error = f"{name}: {type(e).__name__}: {e}"
                break
    raise RuntimeError(f"all providers failed; last error: {last_error}")
