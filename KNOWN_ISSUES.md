# Known Issues — yt-scrape → Open Notebook Integration

Tracked debt acknowledged by the multi-model council review (rounds 1-2).
None of these block shipping for local/single-user use. They may bite at scale
or in multi-user scenarios.

## Deduplication

### `GET /api/sources` pagination not handled
**File:** `open_notebook.py:_find_existing_titles()`
**Impact:** If a notebook has more sources than the API returns in a single
page, `_find_existing_titles()` only sees page 1. Dedup will miss entries on
page 2+, causing duplicate pushes for videos whose titles match later sources.
**Workaround:** For notebooks with >100 existing sources, manually check for
duplicates after `channel-push`.
**Fix:** Check for a `next` cursor or `total` field in the API response and
paginate. Verify the actual Open Notebook API shape first.

### Title collision skips different videos with same title
**File:** `open_notebook.py:_title_key()`
**Impact:** Two different videos with identical titles (e.g. numbered series
that got reposted) — the second is permanently skipped after the first push.
**Workaround:** None currently. Use `--no-skip-existing` if this is a concern
(flag not yet implemented — would need to add).
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
**Impact:** If a single line exceeds `MAX_CONTENT_BYTES` (128KB), there's no
fallback to split it. `create_source_chunked()` would raise
`PayloadTooLargeError` on that chunk.
**Workaround:** None. Exceedingly rare — would require a single transcript
line >128KB.
**Fix:** Add a hard byte-split with a `# (split mid-line due to size limit)`
marker as a last resort.

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
**Impact:** Videos with >50 claims silently truncate at 50.
**Workaround:** None.
**Fix:** Add `--max-claims N` flag to `push` and `channel-push`.

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
