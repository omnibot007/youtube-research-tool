# YouTube Research Tool

A Python CLI tool for fetching, cleaning, and analyzing YouTube video transcripts. Supports two research modes: **Light Research** (fast summarization) and **Deep Research** (comprehensive claim verification with web sources).

No API key needed for public videos. No external LLM required — the AI agent does the comprehension directly.

## Features

### Transcript Fetching
- Download subtitles/captions from any YouTube video
- Auto-fallback to [Whisper](https://github.com/openai/whisper) for videos without captions
- Multi-language support (en, es, fr, etc.)
- Timestamp-preserving transcripts (.tsv)
- Channel scraping and batch downloading
- YouTube search with transcript download

### Transcript Cleanup
Auto-generated YouTube captions are messy — no punctuation, filler words, HTML entities, censored profanity, speaker markers. The cleanup pipeline fixes all of this:

- **HTML decode** — `&gt;` → `>`, `&nbsp;` → space, `&amp;` → `&`
- **Censor handling** — `[__]` → `[expletive]`
- **Speaker markers** — `>>` converted to paragraph breaks
- **Filler word removal** — um, uh, you know, like, basically, etc.
- **Repeat collapse** — "the the the truth" → "the truth"
- **Punctuation heuristics** — capitalize "i" → "I", sentence starts, trailing periods

### Light Research Mode
```bash
python yt_scrape.py research <URL>
```
Fetches + cleans the transcript and prepares it for AI comprehension. The AI agent reads the cleaned transcript and produces a structured report:

- TL;DR, detailed summary, key points
- Topics, action items, notable quotes
- Mentioned resources, target audience, difficulty
- Controversial claims, questions raised
- Sentiment, energy, format

### Deep Research Mode
```bash
python yt_scrape.py deep-research <URL>
```
Goes beyond summarization — verifies claims, maps arguments, assesses bias, and cross-references with authoritative sources. **Quality over speed — no caps on claims verified.**

**Phase 1 (CLI):** Fetches + cleans transcript, extracts claims (9 types), extracts sources (URLs, books, papers, people), saves research package.

**Phase 2 (AI Agent):** Reads the package, performs web research, and produces a comprehensive research brief:

- **Executive summary** — goes beyond what the video says
- **Argument structure** — thesis, premises, conclusions, reasoning quality, logical fallacies
- **Claim verification** — each claim with verdict (verified / contradicted / unverified / opinion), evidence, sources, confidence
- **Bias assessment** — speaker credibility, financial incentives, conflicts of interest
- **Cross-references** — video claims vs. authoritative sources
- **Omission analysis** — what's NOT said that should be
- **Source bibliography** — with reliability tiers
- **Research gaps + open questions**

#### Claim Types Extracted
| Type | Example |
|------|---------|
| `win_rate` | "86% win rate" |
| `statistical` | "90% of traders fail" |
| `financial` | "I made $50,000" |
| `causal` | "Meditation causes brain changes" |
| `authority` | "Studies show..." |
| `superlative` | "The best strategy ever" |
| `scientific` | "Subconscious mind controls everything" |
| `comparative` | "Better than all others" |
| `historical` | "Invented in 1995" |

## Installation

```bash
git clone https://github.com/omnibot007/youtube-research-tool.git
cd youtube-research-tool
pip install -r requirements.txt
```

### Optional: Whisper fallback (for videos without captions)
```bash
pip install faster-whisper
# Also requires ffmpeg on PATH
```

## Usage

### Get a transcript
```bash
python yt_scrape.py transcript "https://youtube.com/watch?v=..."
```

### Get a cleaned transcript
```bash
python yt_scrape.py transcript "https://youtube.com/watch?v=..." --clean
```

### Light Research (fast summarization)
```bash
python yt_scrape.py research "https://youtube.com/watch?v=..."
python yt_scrape.py research "https://youtube.com/watch?v=..." --json
```

### Deep Research (comprehensive analysis)
```bash
python yt_scrape.py deep-research "https://youtube.com/watch?v=..."
python yt_scrape.py deep-research "https://youtube.com/watch?v=..." --json
```

### Search YouTube
```bash
python yt_scrape.py search "RSI trading strategy" --limit 5 --transcripts
```

### Scrape a channel
```bash
python yt_scrape.py channel "https://youtube.com/@channelname" --limit 20 --transcripts
```

### List saved transcripts
```bash
python yt_scrape.py list
```

### Get metadata only
```bash
python yt_scrape.py metadata "https://youtube.com/watch?v=..."
```

## All Commands

| Command | Description |
|---------|-------------|
| `transcript <URL>` | Get transcript for a single video |
| `transcript <URL> --clean` | Get transcript + cleaned version |
| `transcript <URL> --whisper` | Fall back to Whisper if no captions |
| `transcript <URL> --timestamps` | Save timestamped .tsv file |
| `transcript <URL> --all-langs` | Download all available languages |
| `research <URL>` | Light Research — fetch + clean + prepare for comprehension |
| `deep-research <URL>` | Deep Research — fetch + clean + extract claims + sources |
| `search <query>` | Search YouTube (with `--transcripts` to also download) |
| `channel <URL>` | Scrape a channel's videos (with `--transcripts`) |
| `batch <file>` | Batch scrape from a file of URLs |
| `metadata <URL>` | Get video metadata only |
| `list` | List saved transcripts |

## Common Options

| Flag | Description |
|------|-------------|
| `--lang en,es,fr` | Preferred subtitle languages |
| `--cookies cookies.txt` | Cookies file for age-restricted/members content |
| `--cookies-from-browser chrome` | Use cookies from browser |
| `--proxy http://proxy:8080` | Proxy for region unlocking |
| `--output ~/transcripts` | Custom output directory |
| `--json` | Output as JSON |
| `--retries 5` | Max retries on network errors |

## Output Files

All files are saved to `~/yt_transcripts/` (or custom `--output` directory):

| File | Description |
|------|-------------|
| `<id>.txt` | Raw transcript |
| `<id>_clean.txt` | Cleaned transcript (filler removed, HTML decoded, etc.) |
| `<id>.tsv` | Timestamped transcript (with `--timestamps`) |
| `<id>_research_package.json` | Deep Research package (claims + sources + schema) |
| `<id>_research.json` | Light Research report |
| `<id>_deep_research.json` | Deep Research brief |

## Tests

```bash
python -m pytest test_yt_scrape.py -v
```

73 tests covering: transcript cleanup pipeline, claim extraction (9 types), source extraction, Light Research workflow, Deep Research workflow, and performance.

## Requirements

- Python 3.10+ (uses `|` type unions)
- `yt-dlp` (pip install yt-dlp)
- Optional: `faster-whisper` + `ffmpeg` for Whisper fallback

## How It Works

### Light Research Workflow
1. **CLI** fetches the transcript and cleans it (7-step pipeline)
2. **AI agent** reads the cleaned transcript and produces a structured report
3. No external LLM API needed — the agent does the comprehension directly

### Deep Research Workflow
1. **CLI** fetches + cleans the transcript, extracts claims (9 regex patterns), extracts sources
2. **AI agent** reads the research package, performs web research:
   - Verifies every significant claim via web search
   - Maps the argument structure and identifies logical fallacies
   - Assesses the speaker's bias, credibility, and conflicts of interest
   - Cross-references with authoritative sources
   - Identifies what's NOT said (omissions)
3. **Output** is a comprehensive research brief JSON

### Source Reliability Rubric (Deep Research)
| Tier | Sources |
|------|---------|
| High | Peer-reviewed papers, government statistics, established news |
| Moderate | Industry publications, textbooks, recognized experts, established blogs |
| Low | YouTube videos, Reddit, personal blogs, marketing content |

## License

MIT — see [LICENSE](LICENSE)

## Acknowledgments

- Built with [yt-dlp](https://github.com/yt-dlp/yt-dlp)
- Optional Whisper fallback uses [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
