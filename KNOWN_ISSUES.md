# Known Issues — yt-scrape → Open Notebook Integration

Tracked debt acknowledged by the multi-model council review (rounds 1-2).
None of these block shipping for local/single-user use. They may bite at scale
or in multi-user scenarios.

## Deduplication

### `GET /api/sources` pagination not handled
**File:** `open_notebook.py:_find_existing_titles()`
**Status: FIXED (2026-08-04).** `_find_existing_titles()` now pages
defensively: bare-list responses and dict envelopes (`items` / `sources` /
`data` / `results`) are both accepted, `total` / `next` hints are honored,
and paging stops when a page contributes no new titles — a server that
ignores the `page` param costs at most two requests, a small single page
costs exactly one. Hard stop at 50 pages. On top of that, `push` now dedups
single pushes too: pushing the same video twice is a no-op
(`skipped: already_exists`) unless `--force` is passed.
Tests: `TestFindExistingTitlesPagination`, `TestPushVideoSkipExisting`.

### Title collision skips different videos with same title
**File:** `open_notebook.py:_title_key()`
**Impact:** Two different videos with identical titles (e.g. numbered series
that got reposted) — the second is permanently skipped after the first push.
**Workaround:** `push --force` bypasses the title dedup for single pushes
(added 2026-08-04). `channel-push` still has no bypass flag.
**Fix:** Match on YouTube video ID embedded in the title `[vid_id]` suffix
instead of full title. The current `_title_key()` includes the ID, so this
should only collide if two videos share both title AND ID (impossible).

## Chunking

### Transcript-unaware split for oversized paragraphs
**File:** `open_notebook.py:_split_content()`
**Impact:** If a transcript has a single dense paragraph with no `\n\n` or
`\n` breaks (rare — most transcripts are line-segmented), the `\n`-fallback
may split mid-sentence. This degrades Open Notebook's embedding/search quality
at the chunk boundary.
**Workaround:** None. Transcripts are typically `[MM:SS] text` per line, so
this is uncommon.
**Fix:** Split at timestamp boundaries (`\[HH:MM`) rather than bare `\n` when
content matches a transcript pattern. Or split at sentence boundaries within
lines (`re.split(r'(?<=[.!?])\s+', paragraph)`).

### No hard byte-split fallback for monolithic lines
**File:** `open_notebook.py:_split_content()`
**Status: FIXED (2026-08-04).** A line larger than the whole chunk budget is
now hard-split on UTF-8-safe byte windows (never mid-codepoint, lossless
round-trip, each piece under the byte budget).
Tests: `TestSplitContentEdgeCases` — transcripts of length 0 / 1 / 1000 /
100000, a monolithic 5000-char no-newline line, and multibyte content.

## Testing

### No integration test against real Open Notebook instance
**File:** `test_open_notebook.py`
**Impact:** All tests mock urllib. Schema drift in Open Notebook's API won't
be caught until a real push fails.
**Workaround:** Run `python yt_scrape.py push <video> --notebook <id>` against
a local Open Notebook instance before relying on this tool.
**Fix:** Add `@pytest.mark.integration` test that spins up Open Notebook in
Docker, pushes one video, verifies the source appears. Marked so it doesn't
slow the unit suite.

## Other

### No Open Notebook schema version check at startup
**Impact:** `lfnovo/open-notebook` is active development. If the
`SourceCreate` schema changes, failures will be silent or confusing.
**Workaround:** Pin to a specific Open Notebook git commit in production.
**Fix:** Add a `GET /api/health` or version check at client init, fail loud
on mismatch.

### Claims cap (50) not configurable via CLI
**File:** `open_notebook.py:_build_metadata_footer()`
**Status: FIXED (2026-08-04).** `push` and `channel-push` now take
`--max-claims N` (default 50, unchanged). Truncation is no longer silent —
a `[notebook] claim list truncated to X of Y (raise with --max-claims)`
warning prints to stderr when the cap bites.
Tests: `TestMaxClaimsFooter` (default cap + warning, higher cap + quiet).

### Comment extraction has no empty-result warning
**File:** `comments.py:extract_comments()`
**Impact:** `--include-comments` on a video with disabled comments silently
succeeds with no comment source pushed. User has no signal.
**Workaround:** Check stderr output — currently prints "No comments found"
but this is easy to miss.
**Fix:** Log a warning, and consider pushing a "(comments unavailable)"
placeholder source so the notebook reflects the attempt.

### `MAX_CONTENT_BYTES` measured pre-JSON-encoding
**File:** `open_notebook.py:create_source()`
**Impact:** `json.dumps()` of a 128KB string produces ~130-140KB after
escaping. The actual POST body may slightly exceed the 128KB intent. Still
well under FastAPI's 1MB default, so not a live bug.
**Fix:** Recalculate against post-JSON-encoding payload size if precision
matters.

## Sentence integrity (punctuation restoration)

### `deepmultilingualpunctuation` breaks on transformers >= 5
**File:** `yt_sentences.py:_get_model()`
**Impact:** The library calls `pipeline("ner", ..., grouped_entities=False)`.
transformers 5.x removed that kwarg, so model load raises `TypeError`. The
caller swallows all exceptions by design (a bad model must never kill a
scrape), so restoration silently no-ops: transcripts stay unpunctuated
while the run still exits 0.
**Status:** Worked around in-repo. `_get_model()` catches `TypeError` and
rebuilds the pipeline with `aggregation_strategy="none"` (the modern
spelling), grafting it onto the model object.
**Detection:** `deep-research --json` reports `transcript.restoration_applied`.
If that is `false` on auto-caption input, restoration is failing silently.
**Verified:** transformers 5.14.1, torch 2.13.0, Python 3.12.10.

### Paragraph disk cache can mask a broken model
**File:** `yt_sentences.py:_restore_paragraph()`
**Impact:** The cache is checked *before* the model loads, so a warm cache
returns restored text without ever exercising the model. A model-load
regression can therefore pass a green test suite.
**Status:** `TestModelRestoration` isolates `CACHE_DIR` to a pytest
`tmp_path` and resets `_model`, forcing a cold load (~13s) every run.

### Metadata once described pre-restoration text
**File:** `yt_scrape.py:prepare_deep_research()`
**Impact:** `cleaned_chars`/`reduction_pct` were computed before restoration
ran, so the package described text that was never written to disk. This made
a working feature look broken during review.
**Status:** Fixed. Metrics are computed after restoration, and
`restoration_applied` + `sentence_mode` are now reported per run.

### Model splits inside trading jargon
**File:** `yt_sentences.py`
**Status: guarded (2026-08-04).** The punctuation model occasionally broke
sentences inside domain phrases ("overbought at 70. Level", "RSI. Is a
momentum indicator"). `_repair_jargon_splits()` now re-joins those specific
patterns post-restoration (applied to both fresh output and the disk cache;
idempotent). Legitimate boundaries like "RSI. Is that good?" are left alone
(the indicator rule requires an article after the verb). This is a targeted
guard, not a general boundary-quality fix — odd splits outside the protected
patterns can still occur.
Tests: `TestJargonSplitRepair` (repairs, idempotence, no-touch cases).

## Visual extraction (--visual)

**The feature was silently missing for months.** The original
OCR-first pipeline (frame extraction, dedup, Tesseract OCR, and
OpenAI/Anthropic/Ollama frame analysers) lived inside `yt_scrape.py`
and disappeared when an older copy of that file was committed over it.
`README.md` and `SKILL.md` kept documenting `--visual` the whole time,
so the feature looked healthy while no code path called a vision model
and the CLI flag did not exist. Restored 2026-08-04 as `visual.py`.

- **Tesseract is no longer used.** Frames are read by a local vision
  model through Ollama (`ui-tars-7b:latest` by default). This removes
  the Tesseract binary dependency, which was never installed here.
- **It is slow.** Roughly 35-45s per unique frame on CPU. Vision inference
  is ~95% of --visual wall time, which is why frame-count reduction below
  is the lever that matters.
- **Chapter-aware sampling restored (2026-08-04).** `prepare_deep_research`
  now runs `extract_all` before the visual pass and feeds real chapter
  start timestamps to the frame extractor (2+ distinct chapters required;
  fixed-interval sampling remains the fallback).
- **Round 4 frame-count reduction (2026-08-04):** global perceptual dedup
  (dHash, any-to-any — a slide that reappears minutes later no longer
  re-enters the queue; the old dedup only compared neighbors), a
  content-density ranking (JPEG-size proxy for on-screen text, no OCR
  dependency), and an adaptive cap (`YT_VISUAL_TARGET_FRAMES`, default 8)
  that keeps the densest frames in chronological order.
  Tests: `TestFrameSelection` (non-adjacent duplicate runs collapse 6->2,
  density cap evicts near-blank frames, unreadable frames never
  false-match).
- **Requires Ollama running** with a vision-capable model. A
  text-only model is explicitly refused, because it would return
  confident descriptions of images it cannot see.
- Tunables: `YT_VISION_MODEL`, `YT_OLLAMA_HOST`, `YT_VISION_TIMEOUT`,
  `YT_FRAME_INTERVAL`, `YT_MAX_FRAMES`, `YT_VISUAL_TARGET_FRAMES`,
  `YT_FRAME_HASH_DISTANCE`.

## Corpus tools (claim-graph / factcheck, added 2026-08-04)

### Claim-graph node identity is exact-text, not semantic
**File:** `claim_graph.py:claim_node_key()`
**Impact:** Nodes dedupe on normalized span text + claim types. "RSI is
overbought at 70" and "the RSI overbought level is 70" stay separate nodes;
paraphrase-level dedup would need embeddings (see `cluster` for the
BERTopic/sklearn machinery that could back it).
**Impact 2:** Contradiction edges use a metric-keyword + first-number
heuristic mirroring the intra-video detector. Metrics outside the keyword
list (overbought/oversold/support/resistance/stop loss/take profit/win
rate/RSI/MACD/EMA/SMA/ATR) do not produce edges.

### Fact-check depends on scrape-fragile free search + local-LLM judgment
**File:** `factcheck.py`
**Impact:** Search uses DuckDuckGo's HTML endpoint (free, no API key) —
layout changes or rate limits degrade it to zero results, which degrades
verdicts to `unverifiable` (never crashes the run). Verdicts come from a
local Ollama text model (`YT_FACTCHECK_MODEL`, else first non-vision model);
a box with only vision models loaded gets `unverifiable` with the reason
recorded. Verdict quality is bounded by the local model — treat `verified` /
`contradicted` as leads with sources, not ground truth.
