#!/usr/bin/env python3
"""Turn a folder of research packages into one queryable claim corpus.

A single video is worth very little. The value is cross-video: which indicator
periods recur, who contradicts whom, what every channel says about the same
instrument. Each package is its own JSON island until something joins them.

    python corpus.py                       # summary of everything extracted
    python corpus.py --csv claims.csv      # one row per claim
    python corpus.py --jsonl claims.jsonl  # same, for jq or pandas
    python corpus.py --type indicator_setting
    python corpus.py --instrument BTCUSD
    python corpus.py --grep divergence
    python corpus.py --conflicts           # same indicator, different periods

Reads only; it never calls an API and never modifies a package.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

DEFAULT_DIR = Path.home() / "yt_transcripts"

# Normalise at READ time as well as write time. Packages extracted before the
# normalisers existed still sit on disk, and re-extracting them costs real
# money, so the corpus cleans what it reads instead of demanding a re-run.
try:
    from visual import normalize_indicator, normalize_ticker, normalize_timeframe
except Exception:                                    # visual.py must never be
    def normalize_ticker(x):                          # a hard dependency here
        return str(x or "").strip().upper()

    def normalize_timeframe(x):
        return str(x or "").strip()

    def normalize_indicator(x):
        return str(x or "").strip().upper()


def load_packages(folder: Path) -> list:
    """Every *_research_package.json in the folder, newest last. Never raises."""
    out = []
    for p in sorted(folder.glob("*_research_package.json")):
        try:
            pkg = json.loads(p.read_text(encoding="utf-8-sig"))
        except Exception as e:
            print(f"  skipping unreadable {p.name}: {e}", file=sys.stderr)
            continue
        out.append((p, pkg))
    return out


def claim_rows(packages: list) -> list:
    """Flatten every visual claim into one table with its provenance.

    Provenance is the point. A claim without a video, a timestamp and a source
    is an assertion; with them it is evidence you can go back and check.
    """
    rows = []
    for path, pkg in packages:
        video = pkg.get("video") or {}
        vis = pkg.get("visual_extraction") or {}
        vid = video.get("id") or path.name.split("_")[0]
        title = video.get("title") or ""
        channel = video.get("channel") or ""
        model = vis.get("vision_model") or ""
        for c in vis.get("visual_claims") or []:
            rows.append({
                "video_id": vid,
                "title": title,
                "channel": channel,
                "timestamp": c.get("timestamp_str") or "",
                "seconds": c.get("timestamp") or 0,
                "type": c.get("type") or "",
                "claim": c.get("claim") or "",
                "source": c.get("source") or "",
                "model": model,
                "url": f"https://www.youtube.com/watch?v={vid}"
                       f"&t={int(c.get('timestamp') or 0)}s",
            })
    return rows


def value_of(claim: str) -> str:
    """'RSI = 14' -> '14'. 'Instrument: BTCUSD' -> 'BTCUSD'."""
    for sep in (" = ", ": "):
        if sep in claim:
            return claim.split(sep, 1)[1].strip()
    return claim.strip()


def subject_of(claim: str) -> str:
    """'RSI = 14' -> 'RSI'."""
    for sep in (" = ", ": "):
        if sep in claim:
            return claim.split(sep, 1)[0].strip()
    return claim.strip()


def find_conflicts(rows: list) -> list:
    """Same indicator, different periods, across different videos.

    Not necessarily an error. Two traders genuinely disagreeing about RSI
    length is exactly the kind of thing the corpus exists to surface.
    """
    by_subject: dict = defaultdict(lambda: defaultdict(set))
    for r in rows:
        if r["type"] != "indicator_setting":
            continue
        subject = normalize_indicator(subject_of(r["claim"]))
        if not subject:
            continue
        by_subject[subject][value_of(r["claim"])].add(r["video_id"])
    out = []
    for subject, values in sorted(by_subject.items()):
        if len(values) > 1:
            out.append((subject, {v: sorted(ids) for v, ids in values.items()}))
    return out


def summarise(rows: list, packages: list) -> None:
    if not rows:
        print("No visual claims found. Run extract_batch.py first.",
              file=sys.stderr)
        return
    videos = {r["video_id"] for r in rows}
    print(f"corpus: {len(packages)} package(s), {len(videos)} video(s), "
          f"{len(rows)} claims")
    print()

    def top(label: str, ctype: str, n: int = 12, transform=lambda c: c) -> None:
        counts = Counter(transform(r["claim"]) for r in rows
                         if r["type"] == ctype)
        if not counts:
            return
        print(f"--- {label} ---")
        width = max((len(k) for k, _ in counts.most_common(n)), default=0)
        for name, count in counts.most_common(n):
            print(f"  {name:<{width}}  {count:>4}")
        print()

    top("instruments", "instrument",
        transform=lambda c: normalize_ticker(value_of(c)) or value_of(c))
    top("timeframes", "timeframe",
        transform=lambda c: normalize_timeframe(value_of(c)) or value_of(c))
    top("indicator settings", "indicator_setting",
        transform=lambda c: f"{normalize_indicator(subject_of(c))} = {value_of(c)}")
    top("chart patterns", "chart_pattern", n=18)

    conflicts = find_conflicts(rows)
    if conflicts:
        print("--- indicators taught with different periods ---")
        for subject, values in conflicts[:12]:
            spread = ", ".join(f"{v} ({len(ids)} video(s))"
                               for v, ids in sorted(values.items()))
            print(f"  {subject}: {spread}")
        print()

    coverage = [(p.name.split('_')[0], (pkg.get('visual_extraction') or {}))
                for p, pkg in packages]
    partial = [(v, x.get("coverage_pct")) for v, x in coverage
               if x.get("frames_failed")]
    if partial:
        print("--- partially covered videos (re-run these) ---")
        for vid, pct in partial:
            print(f"  {vid}: {pct}% covered")
        print()


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=str(DEFAULT_DIR),
                    help=f"Folder of research packages (default {DEFAULT_DIR})")
    ap.add_argument("--csv", default="", help="Write one row per claim to CSV")
    ap.add_argument("--jsonl", default="", help="Write one JSON object per line")
    ap.add_argument("--type", default="", help="Filter by claim type")
    ap.add_argument("--instrument", default="",
                    help="Only videos mentioning this instrument")
    ap.add_argument("--grep", default="", help="Regex filter on the claim text")
    ap.add_argument("--conflicts", action="store_true",
                    help="Only show indicators taught with different periods")
    args = ap.parse_args()

    folder = Path(args.dir)
    if not folder.exists():
        print(f"no such folder: {folder}", file=sys.stderr)
        return 2
    packages = load_packages(folder)
    rows = claim_rows(packages)

    if args.instrument:
        want = args.instrument.strip().upper()
        keep = {r["video_id"] for r in rows
                if r["type"] == "instrument" and value_of(r["claim"]).upper() == want}
        rows = [r for r in rows if r["video_id"] in keep]
    if args.type:
        rows = [r for r in rows if r["type"] == args.type]
    if args.grep:
        try:
            pat = re.compile(args.grep, re.I)
        except re.error as e:
            print(f"bad --grep pattern: {e}", file=sys.stderr)
            return 2
        rows = [r for r in rows if pat.search(r["claim"])]

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else
                               ["video_id", "timestamp", "type", "claim"])
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {len(rows)} rows -> {args.csv}", file=sys.stderr)
    if args.jsonl:
        with open(args.jsonl, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"wrote {len(rows)} rows -> {args.jsonl}", file=sys.stderr)

    if args.conflicts:
        for subject, values in find_conflicts(rows):
            spread = ", ".join(f"{v} in {ids}" for v, ids in sorted(values.items()))
            print(f"{subject}: {spread}")
        return 0

    if args.type or args.grep or args.instrument:
        for r in rows[:200]:
            print(f"{r['timestamp']:>9}  {r['type']:<18} {r['claim'][:58]:<58} "
                  f"{r['video_id']}")
        print(f"\n{len(rows)} matching claim(s)", file=sys.stderr)
        return 0

    summarise(rows, packages)
    return 0


if __name__ == "__main__":
    sys.exit(main())
