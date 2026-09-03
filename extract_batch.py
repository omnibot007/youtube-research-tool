#!/usr/bin/env python3
"""Extract several YouTube videos in one go, long ones included.

Uses the Gemini video path: no download, no ffmpeg, no local GPU, so it is
immune to the HTTP 403 that blocks yt-dlp here. Videos longer than
YT_GEMINI_VIDEO_CHUNK_SECONDS (default 30 min) are split into clipped windows
automatically and merged with absolute HH:MM:SS timestamps.

    export GEMINI_API_KEY=...
    python extract_batch.py URL URL URL
    python extract_batch.py --from urls.txt
    python extract_batch.py --from urls.txt --dry-run     # cost, no calls
    python extract_batch.py --from urls.txt --force       # ignore the cache

Already-extracted videos are SKIPPED by default. Re-running a 20-video batch
that died at 15 costs nothing for the first 14.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
DEFAULT_OUT = Path.home() / "yt_transcripts"

# Google list price for gemini-3.8-flash through 2026-12-31, per 1M tokens.
# A reseller will differ, so both are overridable. An ESTIMATE, never a bill.
PRICE_IN = float(os.environ.get("YT_PRICE_IN", "0.75"))
PRICE_OUT = float(os.environ.get("YT_PRICE_OUT", "3.75"))
# Video costs ~100 input tokens/second at the default low media resolution.
# Measured on this machine: 10,943 tokens for a 120s window, so ~91/s.
TOKENS_PER_SECOND = float(os.environ.get("YT_TOKENS_PER_SECOND", "95"))

_ID_RE = re.compile(r"(?:v=|/shorts/|youtu\.be/|/embed/)([A-Za-z0-9_-]{11})")


def video_id(url: str) -> str:
    """Pull the 11-char id out of any YouTube URL form, or accept a bare id."""
    text = url.strip()
    m = _ID_RE.search(text)
    if m:
        return m.group(1)
    return text if re.fullmatch(r"[A-Za-z0-9_-]{11}", text) else ""


def expand_playlist(url: str) -> list:
    """Return every video URL in a playlist, or [url] if it is not one.

    A `watch?v=X&list=Y` URL is ambiguous: it is one video AND a position in a
    playlist. Expanding is the useful default here, because that is the form
    YouTube hands you when you copy from inside a playlist. --no-playlist
    takes just the video.

    Metadata only (extract_flat), so this needs no API key and is not blocked
    by the HTTP 403 that stops media downloads on this machine.
    """
    if "list=" not in url:
        return [url]
    try:
        import yt_dlp

        opts = {"quiet": True, "no_warnings": True, "skip_download": True,
                "extract_flat": "in_playlist"}
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as e:
        print(f"  playlist expansion failed ({type(e).__name__}), treating as "
              f"a single video: {url}", file=sys.stderr)
        return [url]

    entries = [e for e in (info.get("entries") or []) if e and e.get("id")]
    if not entries:
        return [url]
    title = (info.get("title") or "playlist")[:48]
    total = sum(int(e.get("duration") or 0) for e in entries)
    print(f"  playlist '{title}': {len(entries)} videos, "
          f"{total // 60} min", file=sys.stderr)
    return [f"https://www.youtube.com/watch?v={e['id']}" for e in entries]


def package_path(vid: str, out_dir: str) -> Path:
    base = Path(out_dir) if out_dir else DEFAULT_OUT
    return base / f"{vid}_research_package.json"


def already_done(vid: str, out_dir: str) -> Path | None:
    """A cached package counts only if it parses AND holds visual claims.

    A truncated or vision-failed package must not block a retry, or one bad
    run poisons the corpus permanently.
    """
    p = package_path(vid, out_dir)
    if not vid or not p.exists():
        return None
    try:
        pkg = json.loads(p.read_text(encoding="utf-8-sig"))
    except Exception:
        return None
    vis = pkg.get("visual_extraction") or {}
    if vis.get("vision_error"):
        return None
    return p if (vis.get("visual_claims") or vis.get("segments")) else None


def probe_duration(url: str) -> int:
    """Duration in seconds via yt-dlp metadata. 0 when unavailable.

    Metadata still works here even though the media download 403s.
    """
    try:
        import yt_dlp

        opts = {"quiet": True, "no_warnings": True, "skip_download": True}
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
        return int((info or {}).get("duration") or 0)
    except Exception:
        return 0


def estimate(seconds: int) -> tuple:
    """(input_tokens, dollars) for a video of this length."""
    tin = seconds * TOKENS_PER_SECOND
    # Output scales with how much is on screen; ~250 tokens per minute is what
    # the live runs averaged. Small next to input, but not nothing.
    tout = seconds / 60.0 * 250
    return int(tin), tin / 1e6 * PRICE_IN + tout / 1e6 * PRICE_OUT


def load_urls(args) -> list:
    urls = list(args.urls)
    if args.from_file:
        p = Path(args.from_file)
        if not p.exists():
            print(f"url list not found: {p}", file=sys.stderr)
            sys.exit(2)
        for raw in p.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if line and not line.startswith("#"):
                urls.append(line)
    expanded = []
    for u in urls:
        expanded.extend([u] if args.no_playlist else expand_playlist(u))
    # De-dupe by VIDEO ID, not URL string: the same video reached through two
    # playlists, or with different ?si= tracking tails, is one video.
    seen, out = set(), []
    for u in expanded:
        key = video_id(u) or u
        if key not in seen:
            seen.add(key)
            out.append(u)
    return out


def extract_one(url: str, out_dir: str, timeout: int) -> dict:
    cmd = [sys.executable, str(HERE / "yt_scrape.py"), "deep-research", url,
           "--visual", "--json"]
    if out_dir:
        cmd += ["--output", out_dir]
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"timed out after {timeout}s",
                "seconds": time.time() - t0}
    elapsed = time.time() - t0

    raw = proc.stdout.decode("utf-8-sig", "replace").strip()
    if not raw:
        tail = proc.stderr.decode("utf-8", "replace").strip().splitlines()
        return {"ok": False, "seconds": elapsed,
                "error": tail[-1] if tail else f"no output (exit {proc.returncode})"}
    try:
        pkg = json.loads(raw)
    except json.JSONDecodeError as e:
        return {"ok": False, "seconds": elapsed, "error": f"unparseable JSON: {e}"}

    vis = pkg.get("visual_extraction") or {}
    usage = vis.get("usage") or {}
    return {
        "ok": True,
        "seconds": elapsed,
        "title": (pkg.get("video") or {}).get("title", "")[:52],
        "duration": (pkg.get("video") or {}).get("duration", 0),
        "windows": vis.get("windows", 0),
        "coverage_pct": vis.get("coverage_pct", 0.0),
        "segments": len(vis.get("segments") or []),
        "claims": len(vis.get("visual_claims") or []),
        "failed_windows": vis.get("frames_failed", 0),
        "in_tokens": usage.get("input_tokens", 0),
        "out_tokens": usage.get("output_tokens", 0),
        "vision_error": vis.get("vision_error", ""),
        "path": pkg.get("research_package_path", ""),
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("urls", nargs="*", help="YouTube URLs or 11-char ids")
    ap.add_argument("--from", dest="from_file", default="",
                    help="File with one URL per line (# comments allowed)")
    ap.add_argument("--out", default="", help="Output directory for packages")
    ap.add_argument("--timeout", type=int, default=0,
                    help="Per-video timeout. Default scales with duration.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Estimate tokens and cost without calling the API")
    ap.add_argument("--force", action="store_true",
                    help="Re-extract videos that already have a package")
    ap.add_argument("--no-playlist", action="store_true",
                    help="Treat a watch?v=X&list=Y URL as the single video X")
    args = ap.parse_args()

    urls = load_urls(args)
    if not urls:
        ap.print_help()
        return 2

    if args.dry_run:
        print(f"Dry run: {len(urls)} video(s), no API calls\n", file=sys.stderr)
        total_s, total_c, unknown = 0, 0.0, 0
        for url in urls:
            secs = probe_duration(url)
            vid = video_id(url)
            cached = "" if args.force else (
                " [cached, would skip]" if already_done(vid, args.out) else "")
            if not secs:
                unknown += 1
                print(f"  {vid or url}: duration unknown{cached}", file=sys.stderr)
                continue
            tin, cost = estimate(secs)
            if cached:
                print(f"  {vid}: {secs // 60}min{cached}", file=sys.stderr)
                continue
            total_s += secs
            total_c += cost
            wins = max(1, -(-secs // 1800))
            print(f"  {vid}: {secs // 60}min, ~{wins} window(s), "
                  f"~{tin:,} tokens, ~${cost:.3f}", file=sys.stderr)
        print(f"\ntotal: {total_s // 60} min of video, ~${total_c:.2f} "
              f"at ${PRICE_IN}/${PRICE_OUT} per 1M", file=sys.stderr)
        if unknown:
            print(f"({unknown} with unknown duration, not counted)", file=sys.stderr)
        return 0

    if not os.environ.get("GEMINI_API_KEY", "").strip():
        print("warning: GEMINI_API_KEY is not set. Without it this falls back "
              "to the local Ollama path, which cannot download video here.",
              file=sys.stderr)

    print(f"Extracting {len(urls)} video(s)\n", file=sys.stderr)
    results = []
    for i, url in enumerate(urls, start=1):
        vid = video_id(url)
        print(f"[{i}/{len(urls)}] {vid or url}", file=sys.stderr)

        if not args.force:
            cached = already_done(vid, args.out)
            if cached:
                print(f"        cached, skipping ({cached.name})", file=sys.stderr)
                results.append({"ok": True, "skipped": True, "url": url,
                                "seconds": 0.0, "claims": 0, "segments": 0,
                                "in_tokens": 0, "out_tokens": 0,
                                "title": vid, "path": str(cached),
                                "vision_error": ""})
                continue

        # A 3-hour video is 6 windows; a fixed timeout would strand it.
        timeout = args.timeout or max(900, (probe_duration(url) or 900) * 3)
        r = extract_one(url, args.out, timeout)
        r["url"] = url
        results.append(r)
        if r["ok"]:
            note = f" ERROR: {r['vision_error']}" if r["vision_error"] else ""
            if r.get("failed_windows"):
                note += f"  PARTIAL {r['coverage_pct']}% covered"
            print(f"        {(r['duration'] or 0) // 60}min, "
                  f"{r['windows']} window(s), {r['segments']} segments, "
                  f"{r['claims']} claims, {r['seconds']:.0f}s, "
                  f"{r['in_tokens'] + r['out_tokens']:,} tokens{note}",
                  file=sys.stderr)
        else:
            print(f"        FAILED: {r['error']}", file=sys.stderr)

    ok = [r for r in results if r["ok"] and not r.get("vision_error")]
    skipped = [r for r in results if r.get("skipped")]
    tin = sum(r.get("in_tokens", 0) for r in results)
    tout = sum(r.get("out_tokens", 0) for r in results)
    cost = tin / 1e6 * PRICE_IN + tout / 1e6 * PRICE_OUT

    print("\n" + "=" * 70, file=sys.stderr)
    print(f"{'video':<52}{'claims':>8}{'time':>8}", file=sys.stderr)
    print("-" * 70, file=sys.stderr)
    for r in results:
        label = (r.get("title") or r["url"])[:50]
        mark = "cached" if r.get("skipped") else (
            f"{r.get('claims', 0)}" if r["ok"] else "FAIL")
        print(f"{label:<52}{mark:>8}{r['seconds']:>7.0f}s", file=sys.stderr)
    print("-" * 70, file=sys.stderr)
    print(f"{len(ok)}/{len(results)} ok ({len(skipped)} cached), "
          f"{sum(r.get('claims', 0) for r in results)} claims, "
          f"{sum(r.get('segments', 0) for r in results)} segments",
          file=sys.stderr)
    print(f"tokens: {tin:,} in / {tout:,} out  ~= ${cost:.3f} "
          f"(override YT_PRICE_IN / YT_PRICE_OUT)", file=sys.stderr)
    for r in results:
        if r.get("path"):
            print(f"  {r['path']}", file=sys.stderr)
    print("\nBuild a queryable corpus:  python corpus.py", file=sys.stderr)

    print(json.dumps(results, indent=2, ensure_ascii=False))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
