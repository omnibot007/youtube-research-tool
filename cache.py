"""Disk-based result caching for yt_scrape.

Caches expensive operation results (metadata, transcripts) to disk as JSON.
yt-dlp owns its own HTTP layer, so HTTP-level caching wouldn't help —
we cache at the result level instead.

Extracted from yt_scrape.py for modularity (Phase 1b).
"""
from __future__ import annotations

import functools
import hashlib
import json
import time
from pathlib import Path
from typing import Callable

from models import VideoInfo


CACHE_DIR = Path.home() / ".yt_scrape_cache"
CACHE_TTL_METADATA = 86400       # 24h — metadata drifts (view counts)
CACHE_TTL_TRANSCRIPT = 604800    # 7d  — transcripts rarely change


def _cache_key(func_name: str, args: tuple, kwargs: dict) -> str:
    """Build a stable cache key from function name + args."""
    # Only hash the URL/ID positional arg + sorted kwargs that affect output
    key_str = f"{func_name}:{args}:{sorted(kwargs.items())}"
    return hashlib.sha256(key_str.encode("utf-8")).hexdigest()[:16]


def _cache_path(key: str) -> Path:
    return CACHE_DIR / f"{key}.json"


def cached_result(ttl_seconds: int):
    """Decorator: cache a function's return value to disk as JSON.

    Handles both VideoInfo results (uses to_dict/from_dict) and plain dicts.
    Only caches successful results (skips if error). Uses a simple file-based
    store keyed by function name + args. TTL controls freshness.

    Pass ``no_cache=True`` to bypass the cache on a per-call basis.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Pop no_cache before calling the wrapped function (it's a cache
            # control flag, not a real argument to the underlying function).
            no_cache = kwargs.pop("no_cache", False)
            if no_cache:
                return func(*args, **kwargs)
            try:
                key = _cache_key(func.__name__, args, kwargs)
                path = _cache_path(key)
                if path.exists():
                    age = time.time() - path.stat().st_mtime
                    if age < ttl_seconds:
                        data = json.loads(path.read_text(encoding="utf-8"))
                        # Don't return cached error results
                        if not data.get("error"):
                            # Reconstruct VideoInfo if it looks like one
                            if "id" in data and "has_transcript" in data:
                                return VideoInfo.from_dict(data)
                            return data
            except Exception:
                pass  # cache read failure → fall through to real call
            result = func(*args, **kwargs)
            # Cache successful results only
            try:
                if result is None:
                    return result
                is_error = (
                    getattr(result, "error", None) or
                    (isinstance(result, dict) and result.get("error"))
                )
                if not is_error:
                    key = _cache_key(func.__name__, args, kwargs)
                    path = _cache_path(key)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    serializable = (
                        result.to_dict() if hasattr(result, "to_dict") else result
                    )
                    path.write_text(
                        json.dumps(serializable, ensure_ascii=False, default=str),
                        encoding="utf-8",
                    )
            except Exception:
                pass  # cache write failure → non-fatal
            return result
        wrapper.__wrapped__ = func  # type: ignore[attr-defined]
        return wrapper
    return decorator
