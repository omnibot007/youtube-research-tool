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


### 2026-09-02 — it had never actually run here, and three things blocked it

`--visual` was documented as working and restored in code, but no end-to-end
run had ever been verified on this machine. Three independent blockers, each
measured, each with a receipt:

**1. Ollama was not running.** `ollama --version` hung, then reported
`Warning: could not connect to a running Ollama instance`. `vision_available()`
correctly returned false, the scrape correctly did not crash, and the feature
correctly produced nothing. Starting the server fixed the gate, not the
pipeline.

**2. A single frame exceeds the timeout on this GPU.** With Ollama up and
`YT_VISION_MODEL=qwen2.5vl:3b`, a real run reached inference and died there:

    [visual] 1 unique frames; reading with qwen2.5vl:3b
    [visual] frame 1/1 at 00:00...
    [visual] frame 1 failed: TimeoutError: timed out

That is one 768px frame exceeding `YT_VISION_TIMEOUT=300`. **The Round 3
comment in `visual.py` is wrong about this card.** It says a 7B at ctx 2048
"fits the T1000 fully (5.0 GB, VRAM flat)" and justifies `num_gpu=99`.
`nvidia-smi` here reports **4096 MiB total**, so a 5.0 GB model cannot fully
offload, and forcing `num_gpu=99` is asking for exactly the thrash observed.
Either that measurement came from an 8 GB T1000 variant or it was never taken
on this machine. Treat the 43.5s/frame figure as unverified on this hardware.

**3. The video cannot be downloaded at all.** `yt-dlp` gets
`HTTP Error 403: Forbidden` from YouTube, and every cookie source fails:

| Source | Result |
|---|---|
| chrome | `Could not copy Chrome cookie database` (locked by a running Chrome) |
| edge | `Failed to decrypt with DPAPI` (app-bound encryption) |
| brave | not installed |
| firefox | not installed |

So on this machine the frame path is blocked before inference even matters.

### The fix: a provider seam, and a path that needs no download

`read_frame()` was a single-function seam. It now dispatches on
`YT_VISION_PROVIDER` (`auto` | `ollama` | `gemini`). `auto` selects Gemini only
when `GEMINI_API_KEY` is present, so a machine without a key is unaffected.

`analyze_youtube_video()` sends the YouTube URL straight to Gemini. Google
fetches the video server-side, so it bypasses blockers 2 and 3 entirely: no
download, no ffmpeg, no local GPU, and one request in place of
download + extract + dedup + N inferences.

**Gemini schema provenance (Rule 2).** Verified 2026-09-02 against
`ai.google.dev` `gemini-api/docs/quickstart`, `/image-understanding`,
`/video-understanding`, `api/interactions.md.txt` and
`gemini-api/docs/interactions/structured-output.md.txt`. Google replaced
`generateContent` with the **Interactions API**:

- `POST https://generativelanguage.googleapis.com/v1beta/interactions`
- headers `x-goog-api-key` and `Api-Revision: 2026-05-20`
- body `{"model": ..., "input": [{"type":"text",...},{"type":"image","data":<b64>,"mime_type":"image/jpeg"}]}`
- a video part is `{"type":"video","uri":"<youtube url>"}`, no mime_type needed,
  public videos only
- response text at `steps[] -> type "model_output" -> content[] -> type "text" -> text`;
  the pre-June-2026 `outputs[]` shape is also accepted so a revision change
  cannot silently blank a reading
- `response_format` with a JSON schema makes the two-key output contract a
  server-side guarantee instead of a polite request

**Not yet verified live.** Every Gemini test is offline: the request shape and
the parser are pinned against the vendor's own documented example, but no real
call has been made from this machine because no API key is present. Get a free
one at `aistudio.google.com/apikey`, set `GEMINI_API_KEY`, and the first real
run is the remaining acceptance test.

### Silent failure fixed

A run that failed every frame reported `vision_error: ""`, because per-frame
errors only went to stderr. Measured: `frames_extracted 1`,
`frames_after_dedup 1`, `frames_analyzed 0`, `frames_failed 1`,
`vision_error ""`. Per-frame errors now land in `result["frame_errors"]`, and
an all-frames-failed run sets `vision_error`.

### Claims now come from chart descriptions too

`extract_visual_claims()` ran only on `on_screen_text`, never on
`chart_description` — so the field that carries the candlestick and indicator
content produced no claims at all. It now runs on both (chart claims carry
`source: "visual_chart"`), and the parser learned prose, which is what a chart
description is:

| Input | Claim |
|---|---|
| `RSI(14) sub-panel` | `RSI = 14` (indicator_setting) |
| `a 200-period EMA` | `EMA = 200` (indicator_setting) |
| `RSI(14) ... oversold below 30` | `RSI below 30` (indicator_threshold) |
| `bullish divergence`, `double top` | chart_pattern |
| `stop loss at 2350.5` | `Stop Loss at 2350.5` (price_level) |

Unsigned levels needed their own rule: the original price matcher requires a
currency symbol, which forex and crypto never show.

### Trading profile is now the default

The old prompt asked for "one sentence describing any chart", which on a
TradingView screenshot yields something true and useless. `TRADING_PROMPT`
asks for chart type, instrument, timeframe, named indicators with settings,
drawn levels/trendlines/zones/Fibonacci, and named patterns, and forbids
guessing a number the model cannot read. `YT_VISION_PROFILE=general` or
`--visual-profile general` restores the old wording.

### New tunables

`YT_VISION_PROVIDER`, `YT_VISION_PROFILE`, `YT_VISUAL_MODE`,
`YT_GEMINI_API_KEY` / `GEMINI_API_KEY` / `GOOGLE_API_KEY`, `YT_GEMINI_MODEL`,
`YT_GEMINI_ENDPOINT`, `YT_GEMINI_API_REVISION`, `YT_GEMINI_TIMEOUT`,
`YT_GEMINI_VIDEO_TIMEOUT`, `YT_GEMINI_MAX_TOKENS`, `YT_GEMINI_TARGET_FRAMES`.
CLI equivalents: `--visual-provider`, `--visual-profile`, `--visual-mode`.

### Remaining debt

- **No live Gemini call has been made.** Offline contract tests only.
- **`GEMINI_MODEL` default is `gemini-3.8-flash`**, taken from the quickstart
  page on 2026-09-02. Model IDs vary across Google's own doc pages; if a run
  returns HTTP 404 or 400, set `YT_GEMINI_MODEL` to a current ID.
- **The local Ollama path stays unusable on this 4 GB card** for multi-frame
  videos. It is kept as an offline fallback, not a recommendation. Before
  trusting it, drop `YT_VISION_NUM_GPU` (let the scheduler split) and raise
  `YT_VISION_TIMEOUT`; the `num_gpu=99` default is tuned for a card this
  machine does not have.
- **Video mode returns one aggregated reading**, timestamped inside the text
  by the model rather than by the pipeline, so `frame_analyses` holds a single
  entry at 00:00. Per-timestamp structure would need the frame path or a
  follow-up parse of the model's own `MM:SS` prefixes.
- **`--visual-mode video` saves no audit PNGs**, because it never downloads the
  video. Use `--visual-mode frames` when you need frames on disk to check the
  model against, and note that the download blocker above applies.

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
