"""Fact-check loop: claim -> free web search -> local-LLM verdict (backlog #3).

- Search: DuckDuckGo HTML endpoint (free tier, no API key), stdlib urllib.
- Verdict: local LLM via Ollama /api/generate. Uses YT_FACTCHECK_MODEL when
  set, otherwise auto-picks the first NON-vision model from /api/tags (the
  vision model would confabulate; text verdicts need a text model).
- Verdicts: verified / contradicted / unverifiable, always with sources when
  evidence exists.

Design rule carried over from the visual pipeline: a fact-check failure must
never take a scrape down. Every network/LLM error degrades to 'unverifiable'
with the reason recorded.
"""
from __future__ import annotations

import json
import os
import re
import urllib.parse
import urllib.request

OLLAMA_HOST = os.environ.get("YT_OLLAMA_HOST", "http://127.0.0.1:11434")
FACTCHECK_MODEL = os.environ.get("YT_FACTCHECK_MODEL", "")
SEARCH_TIMEOUT = int(os.environ.get("YT_FACTCHECK_SEARCH_TIMEOUT", "20"))
LLM_TIMEOUT = int(os.environ.get("YT_FACTCHECK_LLM_TIMEOUT", "120"))
CHECKABLE_TYPES = ("superlative", "statistical", "causal", "scientific",
                   "authority", "historical", "definition_fact")
_VERDICTS = ("verified", "contradicted", "unverifiable")
_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) yt-scrape-factcheck/1.0"

_RESULT_RE = re.compile(
    r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
    re.IGNORECASE | re.DOTALL,
)
_SNIPPET_RE = re.compile(
    r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>',
    re.IGNORECASE | re.DOTALL,
)
_TAG_RE = re.compile(r"<[^>]+>")


def _strip_tags(html: str) -> str:
    return _TAG_RE.sub("", html or "").strip()


def _clean_ddg_url(href: str) -> str:
    """DuckDuckGo wraps result links as /l/?uddg=<enc>; unwrap the real URL."""
    if "uddg=" in href:
        qs = urllib.parse.parse_qs(urllib.parse.urlparse(href).query)
        vals = qs.get("uddg")
        if vals:
            return vals[0]
    if href.startswith("//"):
        return "https:" + href
    return href


def search_web(query: str, max_results: int = 5) -> list[dict]:
    """Free-tier web search. Returns [] on ANY failure (never raises)."""
    try:
        url = ("https://html.duckduckgo.com/html/?q="
               + urllib.parse.quote_plus(query))
        req = urllib.request.Request(url, headers={"User-Agent": _UA})
        with urllib.request.urlopen(req, timeout=SEARCH_TIMEOUT) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except Exception:
        return []
    results: list[dict] = []
    snippets = _SNIPPET_RE.findall(html)
    for i, m in enumerate(_RESULT_RE.finditer(html)):
        if len(results) >= max_results:
            break
        results.append({
            "url": _clean_ddg_url(m.group(1)),
            "title": _strip_tags(m.group(2)),
            "snippet": _strip_tags(snippets[i]) if i < len(snippets) else "",
        })
    return results


def pick_text_model() -> str:
    """YT_FACTCHECK_MODEL, else first Ollama model WITHOUT vision capability."""
    if FACTCHECK_MODEL:
        return FACTCHECK_MODEL
    try:
        req = urllib.request.Request(f"{OLLAMA_HOST}/api/tags",
                                     headers={"User-Agent": _UA})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
    except Exception:
        return ""
    for m in data.get("models", []):
        name = m.get("name") or m.get("model") or ""
        if not name:
            continue
        caps = m.get("capabilities") or []
        if caps:
            if "vision" not in caps:
                return name
            continue
        # No capability info: fall back to a conservative name heuristic.
        if not re.search(r"(?i)vision|llava|\bvl\b|tars", name):
            return name
    return ""


def _ollama_generate(model: str, prompt: str) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0},
        "keep_alive": os.environ.get("YT_FACTCHECK_KEEP_ALIVE", "5m"),
    }
    req = urllib.request.Request(
        f"{OLLAMA_HOST}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "User-Agent": _UA},
    )
    with urllib.request.urlopen(req, timeout=LLM_TIMEOUT) as resp:
        data = json.loads(resp.read().decode("utf-8", errors="replace"))
    return data.get("response", "")


def llm_verdict(claim_sentence: str, evidence: list[dict], model: str) -> dict:
    """Ask the local text LLM for a verdict against the search evidence."""
    if not model:
        return {"verdict": "unverifiable",
                "reason": "no local text model available"}
    if not evidence:
        return {"verdict": "unverifiable", "reason": "no search results"}
    ev_lines = "\n".join(
        f"[{i + 1}] {e.get('title', '')} -- {e.get('snippet', '')} "
        f"({e.get('url', '')})"
        for i, e in enumerate(evidence)
    )
    prompt = (
        "You are a strict fact-checker. A YouTube trading video makes this "
        f"claim:\nCLAIM: {claim_sentence}\n\nWeb search evidence:\n{ev_lines}\n\n"
        'Reply with JSON only: {"verdict": "verified|contradicted|unverifiable", '
        '"reason": "one short sentence", "source_indexes": [1-based indexes of '
        "the evidence items you relied on]}. Use verified ONLY if the evidence "
        "clearly supports the claim; contradicted ONLY if it clearly refutes it; "
        "otherwise unverifiable."
    )
    try:
        raw = _ollama_generate(model, prompt)
        parsed = json.loads(raw) if raw and raw.strip().startswith("{") else {}
    except Exception as e:
        return {"verdict": "unverifiable", "reason": f"llm error: {e}"}
    verdict = str(parsed.get("verdict", "")).strip().lower()
    if verdict not in _VERDICTS:
        verdict = "unverifiable"
    out = {"verdict": verdict, "reason": str(parsed.get("reason", ""))[:300]}
    idxs = parsed.get("source_indexes") or []
    srcs: list[str] = []
    for i in idxs if isinstance(idxs, list) else []:
        try:
            srcs.append(evidence[int(i) - 1]["url"])
        except (ValueError, TypeError, IndexError, KeyError):
            continue
    out["sources"] = srcs or [e["url"] for e in evidence[:2] if e.get("url")]
    return out


def select_checkable_claims(claims: list, types=None, limit: int = 5) -> list[dict]:
    """Pick externally-checkable claims (superlative/statistical/causal/...)."""
    wanted = set(types or CHECKABLE_TYPES)
    picked: list[dict] = []
    for c in claims:
        if not isinstance(c, dict):
            continue
        if wanted.intersection(c.get("claim_types") or []):
            picked.append(c)
        if len(picked) >= limit:
            break
    return picked


def build_query(claim: dict) -> str:
    """Search query: the claim sentence (or span), prefixed with the subject."""
    sent = str(claim.get("sentence", "")).strip()
    subj = str(claim.get("subject", "")).strip()
    q = sent if 0 < len(sent) <= 120 else str(claim.get("matched_pattern", "")).strip()
    if subj and subj.lower() not in q.lower():
        q = f"{subj} {q}"
    return q


def factcheck_claims(claims: list, types=None, limit: int = 5, model=None,
                     search_fn=None, verdict_fn=None) -> dict:
    """Route checkable claims through search + LLM verdicts (injectable)."""
    search = search_fn or search_web
    verdict = verdict_fn or llm_verdict
    chosen_model = model if model is not None else pick_text_model()
    checked: list[dict] = []
    counts = {v: 0 for v in _VERDICTS}
    for claim in select_checkable_claims(claims, types=types, limit=limit):
        query = build_query(claim)
        evidence = search(query)
        v = verdict(str(claim.get("sentence", "")), evidence, chosen_model)
        entry = {
            "sentence": claim.get("sentence", ""),
            "claim_types": list(claim.get("claim_types") or []),
            "query": query,
            "verdict": v.get("verdict", "unverifiable"),
            "reason": v.get("reason", ""),
            "sources": list(v.get("sources", [])),
        }
        if claim.get("deep_link"):
            entry["deep_link"] = claim["deep_link"]
        counts[entry["verdict"]] = counts.get(entry["verdict"], 0) + 1
        checked.append(entry)
    return {
        "ok": True,
        "mode": "factcheck",
        "model": chosen_model,
        "checked_count": len(checked),
        "verdict_counts": counts,
        "checked": checked,
    }


def factcheck_package(package_path, types=None, limit: int = 5, model=None) -> dict:
    """Fact-check the extracted_claims of a saved research package."""
    with open(package_path, "r", encoding="utf-8") as f:
        pkg = json.load(f)
    claims = pkg.get("extracted_claims") or []
    result = factcheck_claims(claims, types=types, limit=limit, model=model)
    video = pkg.get("video") or {}
    result["video_id"] = video.get("id", "")
    result["video_title"] = video.get("title", "")
    return result
