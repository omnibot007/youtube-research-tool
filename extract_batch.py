#!/usr/bin/env python3
"""Extract several YouTube videos in one go, long ones included.

Uses the Gemini video path: no download, no ffmpeg, no local GPU, so it is
immune to the HTTP 403 that blocks yt-dlp on this machine. Videos longer than
YT_GEMINI_VIDEO_CHUNK_SECONDS (default 30 min) are split into clipped windows
automatically and merged back with absolute HH:MM:SS timestamps.

    set GEMINI_API_KEY=...           (or export, in bash)
    python extract_batch.py URL URL URL
    python extract_batch.py --from urls.txt
    python extract_batch.py --from urls.txt --out D:\\research

One line of progress per video, a summary table at the end, and a non-zero
exit only if EVERY video failed. A single bad URL never kills the run.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent


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
    # de-dupe, keep order
    seen, out = set(), []
    for u in urls:
        if u not in seen:
            seen.add(u)
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
    return {
        "ok": True,
        "seconds": elapsed,
        "title": (pkg.get("video") or {}).get("title", "")[:52],
        "duration": (pkg.get("video") or {}).get("duration", 0),
        "windows": vis.get("windows", 0),
        "analyzed": vis.get("frames_analyzed", 0),
        "failed": vis.get("frames_failed", 0),
        "claims": len(vis.get("visual_claims") or []),
        "charts": len(vis.get("chart_patterns") or []),
        "vision_error": vis.get("vision_error", ""),
        "path": pkg.get("research_package_path", ""),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("urls", nargs="*", help="YouTube URLs or 11-char video IDs")
    ap.add_argument("--from", dest="from_file", default="",
                    help="File with one URL per line (# comments allowed)")
    ap.add_argument("--out", default="", help="Output directory for packages")
    ap.add_argument("--timeout", type=int, default=3600,
                    help="Per-video timeout in seconds (default 3600)")
    args = ap.parse_args()

    urls = load_urls(args)
    if not urls:
        ap.print_help()
        return 2
    if not os.environ.get("GEMINI_API_KEY", "").strip():
        print("warning: GEMINI_API_KEY is not set. Without it this falls back "
              "to the local Ollama path, which cannot download video here.",
              file=sys.stderr)

    print(f"Extracting {len(urls)} video(s)\n", file=sys.stderr)
    results = []
    for i, url in enumerate(urls, start=1):
        print(f"[{i}/{len(urls)}] {url}", file=sys.stderr)
        r = extract_one(url, args.out, args.timeout)
        r["url"] = url
        results.append(r)
        if r["ok"]:
            mins = (r["duration"] or 0) // 60
            note = f" ERROR: {r['vision_error']}" if r["vision_error"] else ""
            print(f"        {mins}min, {r['windows']} window(s), "
                  f"{r['claims']} claims, {r['charts']} chart reads, "
                  f"{r['seconds']:.0f}s{note}", file=sys.stderr)
        else:
            print(f"        FAILED: {r['error']}", file=sys.stderr)

    ok = [r for r in results if r["ok"] and not r["vision_error"]]
    print("\n" + "=" * 68, file=sys.stderr)
    print(f"{'video':<54}{'claims':>7}{'time':>7}", file=sys.stderr)
    print("-" * 68, file=sys.stderr)
    for r in results:
        label = (r.get("title") or r["url"])[:52]
        if r["ok"]:
            print(f"{label:<54}{r['claims']:>7}{r['seconds']:>6.0f}s",
                  file=sys.stderr)
        else:
            print(f"{label:<54}{'FAIL':>7}{r['seconds']:>6.0f}s", file=sys.stderr)
    print("-" * 68, file=sys.stderr)
    print(f"{len(ok)}/{len(results)} succeeded, "
          f"{sum(r.get('claims', 0) for r in results)} claims total",
          file=sys.stderr)
    for r in results:
        if r.get("path"):
            print(f"  {r['path']}", file=sys.stderr)

    # stdout stays pure: machine-readable summary only.
    print(json.dumps(results, indent=2, ensure_ascii=False))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
