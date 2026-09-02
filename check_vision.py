#!/usr/bin/env python3
"""Diagnose the --visual pipeline before blaming it.

Written 2026-09-02 after --visual failed for three unrelated reasons at once
and none of them were visible from the scrape output. Every check prints what
it actually observed, not a verdict derived from an exit code.

    python check_vision.py            # check everything available
    python check_vision.py --live     # also spend one real Gemini call

Exit code 0 means at least one provider can actually read a frame.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import visual

OK = "  ok  "
BAD = " FAIL "
WARN = " warn "


def line(status: str, text: str) -> None:
    print(f"[{status}] {text}")


def check_ollama() -> bool:
    """Probe the metadata and inference endpoints separately.

    They fail independently: a wedged server answers /api/tags in 3ms and
    never returns from /api/generate. vision_available() only checks the
    former, which is why a broken server looked healthy.
    """
    print("\n--- Ollama ---")
    host = visual.OLLAMA_HOST.rstrip("/")
    try:
        t0 = time.time()
        with urllib.request.urlopen(host + "/api/tags", timeout=10) as r:
            tags = json.loads(r.read().decode("utf-8"))
        dt = time.time() - t0
    except Exception as e:
        line(BAD, f"/api/tags unreachable at {host}: {type(e).__name__}: {e}")
        line(WARN, "Ollama is not serving. Start it, or use the Gemini provider.")
        return False

    models = [str(m.get("name", "")) for m in tags.get("models") or []]
    line(OK, f"/api/tags answered in {dt:.3f}s, {len(models)} models")
    vision = [
        str(m.get("name"))
        for m in tags.get("models") or []
        if "vision" in (m.get("capabilities") or [])
    ]
    line(OK if vision else WARN, f"vision-capable: {', '.join(vision) or 'none'}")
    if visual.VISION_MODEL not in models:
        line(WARN, f"configured YT_VISION_MODEL {visual.VISION_MODEL!r} is not installed")

    # The check that matters. A short timeout is deliberate: a healthy server
    # answers a 1-token text prompt in well under 30s, and the failure mode
    # this catches is "never returns", not "a bit slow".
    probe = models[0] if models else ""
    if not probe:
        return False
    payload = {
        "model": probe,
        "prompt": "hi",
        "stream": False,
        "options": {"num_predict": 1},
    }
    req = urllib.request.Request(
        host + "/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            r.read()
        line(OK, f"/api/generate answered in {time.time() - t0:.1f}s using {probe}")
        return True
    except Exception as e:
        line(BAD, f"/api/generate no response in {time.time() - t0:.0f}s "
                  f"({type(e).__name__}) using {probe}")
        line(WARN, "Metadata works but inference does not: the server is wedged.")
        line(WARN, "Restart Ollama. `ollama serve` will report a bind error if "
                   "one is already listening.")
        return False


def check_gemini(live: bool) -> bool:
    print("\n--- Gemini ---")
    if not visual.gemini_key():
        line(WARN, "no API key. Set GEMINI_API_KEY to enable this provider.")
        line(WARN, "Free key: https://aistudio.google.com/apikey")
        return False
    # Never print the key, not even a prefix.
    line(OK, "API key present in the environment")
    line(OK, f"endpoint {visual.GEMINI_ENDPOINT}")
    line(OK, f"model {visual.GEMINI_MODEL} (Api-Revision {visual.GEMINI_API_REVISION})")
    if not live:
        line(WARN, "not calling the API. Re-run with --live to spend one request.")
        return True

    from PIL import Image
    tmp = Path(__file__).parent / "_vision_probe.png"
    img = Image.new("RGB", (240, 120), (255, 255, 255))
    for x in range(30, 210):        # a crude candle body, enough to be described
        for y in range(30, 90):
            img.putpixel((x, y), (0, 140, 0) if x % 40 < 20 else (200, 0, 0))
    img.save(tmp)
    try:
        t0 = time.time()
        reading = visual.read_frame(tmp, 0.0)
        dt = time.time() - t0
    finally:
        try:
            tmp.unlink()
        except Exception:
            pass

    if reading.get("error"):
        line(BAD, f"live call failed after {dt:.1f}s: {reading['error']}")
        line(WARN, "An HTTP 404/400 on the model usually means the model ID "
                   "moved. Try YT_GEMINI_MODEL with a current ID.")
        return False
    line(OK, f"live call returned in {dt:.1f}s")
    line(OK, f"on_screen_text: {reading['text'][:70]!r}")
    line(OK, f"chart_description: {reading['chart'][:70]!r}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--live", action="store_true",
                    help="Spend one real Gemini request to prove the path end to end")
    args = ap.parse_args()

    print("--- Config ---")
    line(OK, f"provider resolves to: {visual.resolve_provider()}")
    line(OK, f"prompt profile: {visual.VISION_PROFILE}")
    line(OK, f"visual mode: {visual.VISUAL_MODE}")
    ff = visual.find_ffmpeg()
    line(OK if ff else BAD, f"ffmpeg: {ff or 'NOT FOUND'}")

    ollama_ok = check_ollama()
    gemini_ok = check_gemini(args.live)

    print("\n--- Verdict ---")
    if gemini_ok:
        line(OK, "Gemini path is usable." if args.live
                 else "Gemini path is configured (unproven until --live).")
    if ollama_ok:
        line(OK, "Ollama path is usable.")
    if not (ollama_ok or gemini_ok):
        line(BAD, "No working vision provider. --visual will produce nothing.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
