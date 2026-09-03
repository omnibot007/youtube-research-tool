#!/usr/bin/env python3
"""Measure how faithful the extracted claims are to what was actually said.

    python audit_fidelity.py --only-from urls_zeussy_bigjohn.txt

The extraction model paraphrased ~30 hours of speech. That is useful and it is
not free of error, and a downstream analysis that does not know the error rate
inherits it silently.

This is a GROUNDING check, not a truth check. For each claim it asks whether
the distinctive vocabulary of that claim appears anywhere in that video's own
transcript. A claim built from terms the speaker never uttered is fabricated.
A claim whose terms are all present is *grounded* -- which is weaker than
correct, because the model could still have mangled the logic or attached the
right words to the wrong moment.

Limitations, stated plainly:
  - Transcripts here carry no timestamps, so TIMING cannot be audited. A
    grounded claim may still point at the wrong second.
  - Paraphrase is penalised: a faithful rule stated in the model's own words
    scores lower than a quoted one. Treat the score as a floor, not a verdict.
  - Visual-only claims are excluded, since nothing spoken supports them.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import corpus  # noqa: E402

TRANSCRIPTS = Path.home() / "yt_transcripts"

# Words that carry no evidentiary weight, so their presence proves nothing.
STOP = set("""the a an and or but if then than that this these those of to in
on at by for with from into over under is are was were be been being it its
as not no do does did done can could should would will shall may might must
you your we our they their he she his her him them i me my mine when while
where which who whom what how why all any both each few more most other some
such only own same so too very just also very上 price market trade trading
level levels move moves moving take takes taking use uses using look looks
looking see sees seeing want wants get gets got make makes made go goes going
come comes coming know knows knew think thinks thought like likes need needs
one two three first second next last before after up down out off""".split())


def content_words(text: str) -> set:
    """Distinctive words: 4+ chars, not stopwords. Numbers kept separately."""
    words = re.findall(r"[a-zA-Z]{4,}", text.lower())
    return {w for w in words if w not in STOP}


def load_transcript(video_id: str) -> str:
    for name in (f"{video_id}_clean.txt", f"{video_id}.txt"):
        p = TRANSCRIPTS / name
        if p.exists():
            try:
                return p.read_text(encoding="utf-8", errors="replace").lower()
            except Exception:
                continue
    return ""


def grounding(claim: str, transcript_words: set) -> float:
    """Fraction of a claim's distinctive words present in the transcript."""
    words = content_words(claim)
    if not words:
        return 1.0
    return len(words & transcript_words) / len(words)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only-from", default="", help="Restrict to a URL list")
    ap.add_argument("--sample", type=int, default=0,
                    help="Audit N random claims instead of all")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="Below this grounding fraction, flag for review")
    ap.add_argument("--out", default="", help="Write flagged claims to CSV")
    args = ap.parse_args()

    packages = corpus.load_packages(TRANSCRIPTS)
    if args.only_from:
        import build_findings
        keep = build_findings.requested_ids(args.only_from)
        packages = [(p, pkg) for p, pkg in packages
                    if ((pkg.get("video") or {}).get("id")
                        or p.name.split("_")[0]) in keep]
    rows = corpus.claim_rows(packages)

    # Spoken-content claim types only. A chart pattern is seen, not said.
    spoken = [r for r in rows
              if r["type"] in ("rule", "concept_definition", "concept")]
    if args.sample and args.sample < len(spoken):
        random.seed(7)
        spoken = random.sample(spoken, args.sample)

    cache: dict = {}
    missing_transcript = set()
    scored = []
    for r in spoken:
        vid = r["video_id"]
        if vid not in cache:
            text = load_transcript(vid)
            cache[vid] = content_words(text) if text else None
            if not text:
                missing_transcript.add(vid)
        tw = cache[vid]
        if tw is None:
            continue
        scored.append((grounding(r["claim"], tw), r))

    if not scored:
        print("no auditable claims (no transcripts found)", file=sys.stderr)
        return 1

    scores = [s for s, _ in scored]
    scores.sort()
    n = len(scores)

    def pct(p):
        return scores[min(n - 1, int(n * p))]

    flagged = [(s, r) for s, r in scored if s < args.threshold]
    by_type = collections.defaultdict(list)
    for s, r in scored:
        by_type[r["type"]].append(s)

    print(f"audited {n:,} spoken claims across "
          f"{len(cache) - len(missing_transcript)} videos")
    if missing_transcript:
        print(f"  ({len(missing_transcript)} video(s) had no transcript on "
              f"disk and were skipped)")
    print()
    print("grounding = fraction of a claim's distinctive words found in that")
    print("video's own transcript. 1.0 means every term was actually spoken.")
    print()
    print(f"  mean       {sum(scores)/n:.3f}")
    print(f"  median     {pct(0.50):.3f}")
    print(f"  p10        {pct(0.10):.3f}")
    print(f"  p25        {pct(0.25):.3f}")
    print(f"  fully grounded (1.0)   {sum(1 for s in scores if s >= 0.999):>5}"
          f"  ({100*sum(1 for s in scores if s >= 0.999)/n:.1f}%)")
    print(f"  >= 0.8                 {sum(1 for s in scores if s >= 0.8):>5}"
          f"  ({100*sum(1 for s in scores if s >= 0.8)/n:.1f}%)")
    print(f"  <  {args.threshold} (flagged)        {len(flagged):>5}"
          f"  ({100*len(flagged)/n:.1f}%)")
    print()
    print("by claim type:")
    for t, ss in sorted(by_type.items(), key=lambda kv: -len(kv[1])):
        print(f"  {t:<20} n={len(ss):>5}  mean={sum(ss)/len(ss):.3f}")

    if flagged:
        print()
        print(f"--- weakest 8 (verify these against the video first) ---")
        for s, r in sorted(flagged, key=lambda sr: sr[0])[:8]:
            print(f"  {s:.2f}  {r['claim'][:78]}")
            print(f"        {r['url']}")

    if args.out:
        import csv
        with open(args.out, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["grounding", "type", "claim", "video_id", "title",
                        "timestamp", "url"])
            for s, r in sorted(scored, key=lambda sr: sr[0]):
                w.writerow([f"{s:.3f}", r["type"], r["claim"], r["video_id"],
                            r["title"], r["timestamp"], r["url"]])
        print(f"\nwrote {len(scored):,} scored claims -> {args.out}",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
