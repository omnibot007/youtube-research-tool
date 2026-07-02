---
name: yt-scrape
description: YouTube scraper — extract transcripts, search videos, scrape channels, get metadata, clean transcripts, and prepare Light Research reports. Uses yt-dlp, no API key needed for public content. Optional Whisper fallback for no-caption videos. Use when user says "scrape youtube", "get transcript", "download subtitles", "youtube search", "research video", "light research", or invokes /yt-scrape.
disable-model-invocation: false
allowed-tools: Bash(python *) Bash(python3 *) Bash(py *) Bash(yt-dlp *) Read Write
---

# YouTube Scraper

Extract transcripts, search videos, scrape channels, and get metadata from YouTube using yt-dlp. No API key required for public content. Optional Whisper fallback for videos without captions.

## Quick Start

### Single video transcript
```bash
python .devin/skills/yt-scrape/yt_scrape.py transcript <URL_OR_ID>
```

### Search and scrape top results
```bash
python .devin/skills/yt-scrape/yt_scrape.py search "algorithmic trading" --limit 5 --transcripts
```

### Scrape a channel's recent videos
```bash
python .devin/skills/yt-scrape/yt_scrape.py channel <CHANNEL_URL_OR_ID> --limit 10 --transcripts
```

### Batch scrape (file with one URL/ID per line)
```bash
python .devin/skills/yt-scrape/yt_scrape.py batch links.txt --transcripts
```

### Metadata only (no transcript)
```bash
python .devin/skills/yt-scrape/yt_scrape.py metadata <URL_OR_ID>
```

### List saved transcripts
```bash
python .devin/skills/yt-scrape/yt_scrape.py list
```

### Light Research — fetch + clean transcript for comprehension
```bash
python .devin/skills/yt-scrape/yt_scrape.py research <URL_OR_ID>
```
Downloads the transcript, cleans it (filler removal, HTML decode, speaker breaks, censor handling), and saves both raw and cleaned versions. Outputs the paths + schema for producing a Light Research report.

The cleaned transcript is saved as `<video_id>_clean.txt` with:
- Filler words removed (um, uh, you know, like, etc.)
- HTML entities decoded (&gt; → >, &nbsp; → space)
- Speaker markers (>>) converted to paragraph breaks
- Censored profanity ([__]) replaced with [expletive]
- Repeated phrases collapsed
- Basic punctuation heuristics applied

The Light Research report (produced by Devin, not an external API) includes:
- TL;DR, detailed summary, key points
- Topics, action items, notable quotes
- Mentioned resources, target audience, difficulty
- Controversial claims, questions raised, sentiment, energy

Output as JSON:
```bash
python yt_scrape.py research <URL> --json
```

### Get a cleaned transcript only (no research)
```bash
python yt_scrape.py transcript <URL> --clean
```
Saves both the raw transcript and a cleaned version (`<video_id>_clean.txt`).

### Deep Research — comprehensive analysis with claim verification
```bash
python yt_scrape.py deep-research <URL_OR_ID>
```
Fetches transcript, cleans it, **extracts factual claims** (percentages, win rates, causal statements, authority appeals, superlatives, scientific claims, comparatives, historical claims), **extracts sources** (URLs, books, papers, people), and saves a research package JSON.

The research package includes:
- Cleaned transcript (ready for reading)
- Pre-extracted claims list (regex-identified, for Devin to augment)
- Pre-extracted sources (URLs, books, papers, people mentioned)
- Video metadata + description
- Full Deep Research schema
- Step-by-step instructions for Phase 2

**Phase 2 (Devin executes):**
1. Read the cleaned transcript thoroughly
2. Review + augment the extracted claims list
3. Verify EVERY significant claim via web search (quality over speed — no caps)
4. Map argument structure (thesis → premises → conclusions)
5. Identify logical fallacies
6. Assess bias, credibility, conflicts of interest
7. Cross-reference with authoritative sources
8. Identify omissions (what's NOT said that should be)
9. Produce deep research brief JSON

Output saved as `<video_id>_deep_research.json`.

**Claim types extracted:**
- `win_rate` — win/success rate claims
- `statistical` — percentages, proportions
- `financial` — dollar amounts, financial figures
- `causal` — X causes/leads to/prevents Y
- `authority` — "studies show", "research proves"
- `superlative` — "the best", "always works", "guaranteed"
- `scientific` — brain, neurotransmitters, subconscious, quantum, DNA
- `comparative` — "better than", "superior to", "outperforms"
- `historical` — dates, historical references

Output as JSON:
```bash
python yt_scrape.py deep-research <URL> --json
```

## Advanced Features

### Whisper fallback (no-caption videos)
When a video has no subtitles, fall back to audio transcription with faster-whisper:
```bash
python yt_scrape.py transcript <URL> --whisper
python yt_scrape.py transcript <URL> --whisper --whisper-model small --whisper-device cpu
```
Requires: `pip install faster-whisper` and ffmpeg on PATH.
Models: tiny, base (default), small, medium, large (larger = slower but more accurate).

### Timestamp-preserving transcripts
Save a `.tsv` file with `[HH:MM:SS --> HH:MM:SS] text` per segment:
```bash
python yt_scrape.py transcript <URL> --timestamps
```
Produces both `.txt` (plain text) and `.tsv` (timestamped) files.

### Multi-language support
Specify preferred subtitle languages:
```bash
python yt_scrape.py transcript <URL> --lang en,es,fr
```
Download ALL available subtitle languages:
```bash
python yt_scrape.py transcript <URL> --all-langs
```
Each language is saved as `<video_id>_<lang>.txt`.

### Cookie support (age-restricted / members-only)
Use a cookies file:
```bash
python yt_scrape.py transcript <URL> --cookies cookies.txt
```
Use cookies from your browser:
```bash
python yt_scrape.py transcript <URL> --cookies-from-browser chrome
```
Supported browsers: chrome, firefox, edge, brave.

### Proxy support (region unlocking)
```bash
python yt_scrape.py transcript <URL> --proxy http://proxy:8080
```

### Retry logic
Automatic retry with exponential backoff on network errors (429, 5xx, timeouts):
```bash
python yt_scrape.py transcript <URL> --retries 5
```

## Output

- Transcripts saved to `~/yt_transcripts/` as `.txt` files
- Timestamped transcripts saved as `.tsv` files (with `--timestamps`)
- Multi-language transcripts saved as `<video_id>_<lang>.txt` (with `--all-langs`)
- Metadata printed as JSON to stdout (with `--json`)
- Raw VTT files cleaned up automatically
- Each transcript file includes: TITLE, URL, CHANNEL, VIDEO_ID, DURATION, UPLOAD_DATE, TRANSCRIPT_SOURCE, TRANSCRIPT_LANG, then clean text

## All Options

- `--lang en,es,fr` — Subtitle languages in preference order (default: en,en-US,en-GB)
- `--all-langs` — Download all available subtitle languages
- `--timestamps` — Also save timestamped transcript (.tsv)
- `--whisper` — Fall back to Whisper if no captions
- `--whisper-model base` — Whisper model (tiny/base/small/medium/large)
- `--whisper-device cpu` — Whisper device (cpu/cuda)
- `--whisper-compute-type int8` — Compute type (int8/float16/float32)
- `--cookies FILE` — Cookies file for age-restricted/members content
- `--cookies-from-browser BROWSER` — Use cookies from browser (chrome/firefox/edge/brave)
- `--proxy URL` — Proxy URL for region unlocking
- `--retries N` — Max retries on network errors (default: 3)
- `--rate-limit N` — Seconds between requests (default: 2)
- `--output DIR` — Custom output directory (default: ~/yt_transcripts)
- `--json` — Output results as JSON (for piping to other tools)

## How It Works

1. Uses `yt-dlp` as a Python library (no subprocess overhead)
2. Downloads auto-generated captions + manual subtitles when available
3. Parses VTT → clean text (strips timestamps, HTML tags, duplicate lines)
4. Optionally preserves timestamps in `.tsv` format (with `--timestamps`)
5. Falls back to any available language if preferred languages missing
6. If no captions AND `--whisper` enabled: downloads audio, transcribes with faster-whisper
7. Rate-limits between requests to avoid getting blocked
8. Retries on network errors with exponential backoff
9. Maps yt-dlp errors to clear user messages (private, age-restricted, region-locked, etc.)

## Error Handling

Common errors are mapped to clear messages with `error_type`:
- `private` — Video is private, deleted, or region-locked
- `age_restricted` — Use --cookies-from-browser
- `members_only` — Use --cookies FILE
- `region_locked` — Use --proxy URL
- `rate_limited` — YouTube rate limited the request
- `no_subtitles` — No captions found (use --whisper for fallback)
- `no_ffmpeg` — ffmpeg not on PATH (required for Whisper)

## Dependencies

- `yt-dlp` (pip install yt-dlp) — already installed on this machine
- `faster-whisper` (pip install faster-whisper) — optional, only for --whisper
- `ffmpeg` — already installed, required for Whisper fallback
- Python 3.10+ (uses `|` type unions)

## Light Research Workflow

The `research` command is Step 1 of a two-step process:

1. **Step 1 (CLI):** `python yt_scrape.py research <URL>` — fetches + cleans the transcript, saves both versions, outputs paths and schema
2. **Step 2 (Devin):** Devin reads the cleaned transcript and produces a structured Light Research report (JSON) with: TL;DR, summary, key points, topics, action items, quotes, resources, controversial claims, questions raised, sentiment, energy

The report is saved as `<video_id>_research.json`.

## Deep Research (built)

Deep Research goes beyond Light Research — it verifies claims, maps arguments, assesses bias, and cross-references with authoritative sources. Quality over speed — no caps on claims verified.

**CLI (Phase 1):** `python yt_scrape.py deep-research <URL>` — fetch + clean + extract claims + extract sources → research package JSON

**Devin (Phase 2):** Reads the package, performs web research, verifies every significant claim, maps argument structure, identifies fallacies, assesses bias, cross-references, identifies omissions, produces deep research brief.

Deep Research brief includes:
- Executive summary (goes beyond what the video says)
- Argument structure (thesis, premises, conclusions, reasoning quality, fallacies)
- Claim verification (each claim with verdict, evidence, sources, confidence)
- Bias assessment (credibility, biases, conflicts of interest, financial ecosystem)
- Cross-references (video claims vs authoritative sources)
- Omission analysis (what's NOT said that should be)
- Source bibliography (with reliability tiers)
- Research gaps + open questions
- Methodology + overall confidence

## When to Use

- "Get me the transcript of this video: <URL>"
- "Research this video: <URL>" — Light Research mode
- "Deep research this video: <URL>" — Deep Research mode (comprehensive analysis)
- "Summarize this YouTube video for me: <URL>"
- "What are the key points of this video: <URL>"
- "Clean up this transcript" (with --clean)
- "Scrape all videos from this channel"
- "Search YouTube for X and get the top 5 transcripts"
- "Download subtitles for these videos" (with a list)
- "This video has no captions — transcribe it with Whisper"
- "I need timestamps in the transcript"
- "Scrape this age-restricted video" (with cookies)
- "This video is region-locked — use a proxy"
