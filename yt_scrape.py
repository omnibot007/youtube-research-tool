"""YouTube scraper — transcripts, search, channel scraping, metadata.

Uses yt-dlp as a Python library. No API key needed for public content.
Optional Whisper fallback for videos without captions (requires faster-whisper).

Usage:
    python yt_scrape.py transcript <URL_OR_ID> [options]
    python yt_scrape.py search "query" --limit 5 --transcripts
    python yt_scrape.py channel <CHANNEL_URL_OR_ID> --limit 10 --transcripts
    python yt_scrape.py batch links.txt --transcripts
    python yt_scrape.py metadata <URL_OR_ID>
    python yt_scrape.py list

Options:
    --lang en,es,fr          Subtitle languages in preference order
    --all-langs              Download all available subtitle languages
    --timestamps             Also save timestamped transcript (.tsv)
    --whisper                Fall back to Whisper if no captions
    --whisper-model base     Whisper model (tiny/base/small/medium/large)
    --whisper-device cpu     Whisper device (cpu/cuda)
    --cookies FILE           Cookies file for age-restricted/members content
    --cookies-from-browser B Use cookies from browser (chrome/firefox/edge/brave)
    --proxy URL              Proxy URL for region unlocking
    --retries 3              Max retries on network errors
    --rate-limit 2           Seconds between requests
    --output DIR             Custom output directory
    --json                   Output as JSON
"""
from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import functools
import glob
import json
import os
import re
import shutil
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

try:
    import yt_dlp
except ImportError:
    print("ERROR: yt-dlp not installed. Run: pip install yt-dlp", file=sys.stderr)
    sys.exit(1)

try:
    import edge_tts
except ImportError:
    edge_tts = None  # TTS features disabled — run: pip install edge-tts

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUTPUT_DIR = Path.home() / "yt_transcripts"
DEFAULT_LANGS = ["en", "en-US", "en-GB"]
DEFAULT_RATE_LIMIT = 2  # seconds between requests
DEFAULT_RETRIES = 3


# ---------------------------------------------------------------- DATA MODEL

@dataclass
class VideoInfo:
    id: str = ""
    title: str = ""
    url: str = ""
    channel: str = ""
    duration: int = 0
    upload_date: str = ""
    view_count: int = 0
    like_count: int = 0
    transcript_path: str = ""
    transcript_chars: int = 0
    has_transcript: bool = False
    transcript_source: str = ""  # "caption" | "whisper" | ""
    transcript_lang: str = ""
    timestamped_path: str = ""   # .tsv file path if --timestamps
    segments: list = field(default_factory=list)  # TranscriptSegment list (not serialized by asdict if we exclude)
    error: str = ""
    error_type: str = ""         # "no_subtitles" | "private" | "age_restricted" | "region_locked" | "rate_limited" | "network" | "unknown"

    def to_dict(self) -> dict:
        d = asdict(self)
        # Don't serialize segments in the dict (they're internal)
        d.pop("segments", None)
        return d


@dataclass
class TranscriptSegment:
    """One segment of a transcript with timing info."""
    text: str
    start: float
    end: float

    def format_timestamped(self) -> str:
        """Format as [HH:MM:SS --> HH:MM:SS] text."""
        def fmt(sec: float) -> str:
            h = int(sec // 3600)
            m = int((sec % 3600) // 60)
            s = int(sec % 60)
            return f"{h:02d}:{m:02d}:{s:02d}"
        return f"[{fmt(self.start)} --> {fmt(self.end)}] {self.text}"


# ---------------------------------------------------------------- URL PARSING

_VIDEO_ID_RE = re.compile(
    r"(?:v=|/|be/|shorts/|embed/|live/)([A-Za-z0-9_-]{11})(?:[?&].*)?$"
)


def parse_video_id(url_or_id: str) -> str:
    """Extract the 11-char YouTube video ID from a URL or accept a bare ID.

    Handles: watch?v=, youtu.be/, embed/, shorts/, live/, bare ID.
    """
    s = url_or_id.strip()
    if not s:
        return ""
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", s):
        return s
    m = _VIDEO_ID_RE.search(s)
    if not m:
        return ""
    return m.group(1)


def canonical_url(video_id: str) -> str:
    """Stable canonical URL for a video_id."""
    return f"https://www.youtube.com/watch?v={video_id}"


def _normalize_url(url_or_id: str) -> str:
    """Normalize input to a full YouTube URL."""
    s = url_or_id.strip()
    if not s:
        return ""
    if not s.startswith("http"):
        vid = parse_video_id(s)
        return canonical_url(vid) if vid else s
    return s


def _normalize_upload_date(raw: str | None) -> str:
    """yt-dlp returns upload_date as YYYYMMDD; normalize to ISO YYYY-MM-DD."""
    if not raw or not re.fullmatch(r"\d{8}", raw):
        return ""
    return f"{raw[0:4]}-{raw[4:6]}-{raw[6:8]}"


# ---------------------------------------------------------------- ERROR MAPPING

_ERROR_MAP: list[tuple[str, str, str]] = [
    # (regex pattern, error_type, user message)
    (r"Video unavailable", "private", "Video is private, deleted, or region-locked"),
    (r"Sign in to confirm your age", "age_restricted", "Age-restricted video — use --cookies-from-browser chrome"),
    (r"Members-only|join this channel", "members_only", "Members-only content — use --cookies FILE"),
    (r"HTTP Error 429|Too Many Requests", "rate_limited", "Rate limited by YouTube — wait and retry, or use --proxy"),
    (r"This video is not available in your country", "region_locked", "Region-locked video — use --proxy URL"),
    (r"Private video", "private", "Video is private"),
    (r"has been removed|Deleted video", "deleted", "Video has been deleted"),
    (r"LIVE event|premieres", "live", "Video is a live stream or premiere — may not have captions yet"),
    (r"[Uu]se --cookies|confirm you.re not a bot|Sign in to (?:confirm|access)", "cookies_needed", "Cookies required — use --cookies FILE or --cookies-from-browser BROWSER"),
]


def _map_error(error_str: str) -> tuple[str, str]:
    """Map a yt-dlp error string to (error_type, user_friendly_message)."""
    for pattern, error_type, message in _ERROR_MAP:
        if re.search(pattern, error_str, re.I):
            return error_type, message
    return "unknown", error_str[:200]


# ---------------------------------------------------------------- RETRY LOGIC

def with_retry(
    max_retries: int = DEFAULT_RETRIES,
    base_delay: float = 1.0,
    max_delay: float = 16.0,
) -> Callable:
    """Decorator: retry on network errors with exponential backoff.

    Retries on: ConnectionError, TimeoutError, HTTP 429, HTTP 5xx.
    Respects Retry-After header on 429 when available.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    last_exc = exc
                    err_str = str(exc)
                    # Don't retry on permanent errors (private, deleted, age-restricted)
                    if any(p in err_str for p in ("Video unavailable", "Private video", "removed", "Sign in to confirm")):
                        raise
                    # Don't retry on the last attempt
                    if attempt >= max_retries:
                        raise
                    # Calculate delay with exponential backoff
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    # Check for Retry-After hint in error
                    retry_match = re.search(r"Retry-After:\s*(\d+)", err_str, re.I)
                    if retry_match:
                        delay = max(delay, float(retry_match.group(1)))
                    print(f"  [retry {attempt + 1}/{max_retries}] {type(exc).__name__}, waiting {delay:.1f}s...", file=sys.stderr)
                    time.sleep(delay)
            raise last_exc  # type: ignore[misc]
        return wrapper
    return decorator


# ---------------------------------------------------------------- VTT PARSER

def parse_vtt(vtt_path: str) -> tuple[str | None, list[TranscriptSegment]]:
    """Parse a VTT subtitle file into (clean_text, segments).

    Returns:
        Tuple of (plain text string, list of TranscriptSegment with timestamps).
    """
    try:
        with open(vtt_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    except Exception:
        return None, []

    lines = content.split("\n")
    text_lines: list[str] = []
    segments: list[TranscriptSegment] = []
    prev_line = ""
    current_start: float | None = None
    current_end: float | None = None
    current_texts: list[str] = []

    for line in lines:
        line = line.strip()
        if not line or line.startswith("WEBVTT") or line.startswith("Kind:") or line.startswith("Language:"):
            continue
        if line.startswith("NOTE"):
            continue

        # Timestamp line: 00:00:01.234 --> 00:00:05.678
        if "-->" in line:
            # Save previous segment if any
            if current_start is not None and current_texts:
                seg_text = " ".join(current_texts)
                segments.append(TranscriptSegment(
                    text=seg_text, start=current_start,
                    end=current_end if current_end is not None else current_start,
                ))
                current_texts = []

            # Parse new timestamps — strip extra metadata (align:start, position:0%, etc.)
            times = line.split("-->")
            current_start = _parse_vtt_time(times[0].strip())
            if len(times) > 1:
                # End time may have "align:start position:0%" after it
                end_str = times[1].strip().split()[0]  # take first whitespace-delimited token
                current_end = _parse_vtt_time(end_str)
            else:
                current_end = None
            continue

        # Text line
        clean = re.sub(r"<[^>]+>", "", line)
        if clean and clean != prev_line:
            text_lines.append(clean)
            prev_line = clean
            if current_start is not None:
                current_texts.append(clean)

    # Save last segment
    if current_start is not None and current_texts:
        seg_text = " ".join(current_texts)
        segments.append(TranscriptSegment(
            text=seg_text, start=current_start,
            end=current_end if current_end is not None else current_start,
        ))

    plain_text = " ".join(text_lines) if text_lines else None
    return plain_text, segments


def _parse_vtt_time(time_str: str) -> float:
    """Parse VTT timestamp (HH:MM:SS.mmm or MM:SS.mmm) to seconds."""
    time_str = time_str.strip()
    parts = time_str.split(":")
    try:
        if len(parts) == 3:
            h, m, s = parts
            return float(h) * 3600 + float(m) * 60 + float(s)
        elif len(parts) == 2:
            m, s = parts
            return float(m) * 60 + float(s)
        else:
            return float(time_str)
    except ValueError:
        return 0.0


# ---------------------------------------------------------------- WHISPER FALLBACK

_WHISPER_MODEL_CACHE: dict[str, Any] = {}


def _check_ffmpeg() -> bool:
    """Check if ffmpeg is available on PATH."""
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def _get_whisper_model(model_name: str, device: str, compute_type: str) -> Any:
    """Lazy-load and cache a faster-whisper model."""
    key = f"{model_name}|{device}|{compute_type}"
    if key not in _WHISPER_MODEL_CACHE:
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise RuntimeError("faster-whisper not installed. Run: pip install faster-whisper")
        _WHISPER_MODEL_CACHE[key] = WhisperModel(model_name, device=device, compute_type=compute_type)
    return _WHISPER_MODEL_CACHE[key]


def _extract_audio(video_id: str, dest_dir: Path, opts: dict | None = None) -> Path:
    """Extract best-audio m4a via yt-dlp + ffmpeg. Returns the audio path."""
    url = canonical_url(video_id)
    out_template = str(dest_dir / f"{video_id}.%(ext)s")
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "format": "bestaudio/best",
        "outtmpl": out_template,
        "noplaylist": True,
        "postprocessors": [
            {"key": "FFmpegExtractAudio", "preferredcodec": "m4a", "preferredquality": "32"},
        ],
    }
    if opts:
        ydl_opts.update(opts)
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except Exception as exc:
        raise RuntimeError(f"Audio extraction failed for {video_id}: {exc}") from exc
    audio_path = dest_dir / f"{video_id}.m4a"
    if not audio_path.exists():
        candidates = list(dest_dir.glob(f"{video_id}.*"))
        if not candidates:
            raise RuntimeError(f"Audio file not found after extraction for {video_id}")
        audio_path = candidates[0]
    return audio_path


def transcribe_with_whisper(
    video_id: str,
    model_name: str = "base",
    device: str = "cpu",
    compute_type: str = "int8",
    ydl_opts: dict | None = None,
) -> tuple[str, list[TranscriptSegment]]:
    """Download audio and transcribe with faster-whisper.

    Returns (transcript_text, segments).
    Raises RuntimeError on any failure.
    """
    if not _check_ffmpeg():
        raise RuntimeError("ffmpeg/ffprobe not found on PATH. Required for Whisper fallback.")

    model = _get_whisper_model(model_name, device, compute_type)

    with tempfile.TemporaryDirectory(prefix="yt_whisper_") as tmp:
        audio_path = _extract_audio(video_id, Path(tmp), ydl_opts)
        try:
            segments_iter, _info = model.transcribe(str(audio_path), beam_size=5)
            segments: list[TranscriptSegment] = []
            parts: list[str] = []
            for seg in segments_iter:
                text = (seg.text or "").strip()
                if not text:
                    continue
                segments.append(TranscriptSegment(
                    text=text,
                    start=float(seg.start),
                    end=float(seg.end),
                ))
                parts.append(text)
        except Exception as exc:
            raise RuntimeError(f"Whisper transcription failed for {video_id}: {exc}") from exc

    transcript = " ".join(parts).strip()
    if not transcript:
        raise RuntimeError(f"Whisper produced empty transcript for {video_id}")
    return transcript, segments


# ---------------------------------------------------------------- PUNCTUATION RESTORATION

# Speech-pause thresholds (seconds) for timing-based sentence detection.
_PAUSE_SENTENCE = 0.8   # gap >= this → sentence boundary (period)
_PAUSE_COMMA = 0.3      # gap >= this → clause boundary (comma)
# Group segments into chunks of ~this many words before sending to LLM.
_LLM_CHUNK_WORDS = 400
# Fallback: target word count per sentence when timing gaps are absent.
_SENTENCE_WORD_TARGET = 20
_SENTENCE_WORD_MIN = 10


def _group_segments_by_pauses(
    segments: list[TranscriptSegment],
) -> list[tuple[str, float, float]]:
    """Group VTT segments into sentence-like units using speech pauses.

    Returns list of (text, start, end) tuples. Each tuple is one probable
    sentence. Uses two signals:
      1. Timing gaps (>= _PAUSE_SENTENCE → new sentence, >= _PAUSE_COMMA → comma)
      2. Word count heuristic (if no timing gaps, split at ~_SENTENCE_WORD_TARGET words)
    """
    if not segments:
        return []
    sentences: list[tuple[str, float, float]] = []
    current_words: list[str] = []
    current_start = segments[0].start
    current_end = segments[0].end
    prev_end = segments[0].end

    # Check if we have meaningful timing gaps
    has_gaps = False
    for i in range(1, len(segments)):
        gap = segments[i].start - segments[i - 1].end
        if gap >= _PAUSE_COMMA:
            has_gaps = True
            break

    for i, seg in enumerate(segments):
        text = seg.text.strip()
        if not text:
            continue
        gap = seg.start - prev_end if i > 0 else 0.0
        word_count = len(current_words)

        # Start new sentence on: long pause OR word count target (when no gaps)
        should_break = False
        if gap >= _PAUSE_SENTENCE and current_words:
            should_break = True
        elif not has_gaps and word_count >= _SENTENCE_WORD_TARGET and current_words:
            should_break = True
        elif gap >= _PAUSE_COMMA and current_words and has_gaps:
            current_words.append(",")

        if should_break:
            sentences.append((" ".join(current_words), current_start, current_end))
            current_words = []
            current_start = seg.start

        current_words.extend(text.split())
        current_end = seg.end
        prev_end = seg.end

    if current_words:
        sentences.append((" ".join(current_words), current_start, current_end))
    return sentences


def _restore_punctuation_llm(text: str, provider: str | None = None) -> str | None:
    """Use an LLM to restore punctuation and capitalization.

    Optional enhancement — the word-count heuristic in restore_punctuation()
    already produces good results without any API call. This function is only
    called when use_llm=True is passed to restore_punctuation().

    Supports multiple providers (auto-detected):
      - OpenAI / 9router (OPENAI_API_KEY, optionally OPENAI_BASE_URL)
      - Anthropic (ANTHROPIC_API_KEY)
      - Ollama (local, no API key, needs ollama running)

    Returns punctuated text, or None if no provider available or call fails.
    """
    provider = provider or _detect_llm_provider()
    if not provider:
        return None

    # Chunk the text for API limits
    words = text.split()
    chunks: list[str] = []
    for i in range(0, len(words), _LLM_CHUNK_WORDS):
        chunk = " ".join(words[i : i + _LLM_CHUNK_WORDS])
        chunks.append(chunk)

    system_msg = (
        "You are a punctuation restoration tool. Add periods, "
        "commas, question marks, and capitalization to the "
        "following auto-generated transcript text. Preserve "
        "the original words exactly — only add punctuation and "
        "fix capitalization. Do not add or remove words."
    )

    try:
        results: list[str] = []

        if provider == "openai":
            from openai import OpenAI
            api_key = os.environ.get("OPENAI_API_KEY", "")
            client_kwargs: dict[str, Any] = {"api_key": api_key, "timeout": 60}
            base_url = os.environ.get("OPENAI_BASE_URL")
            if base_url:
                client_kwargs["base_url"] = base_url
            client = OpenAI(**client_kwargs)
            for chunk in chunks:
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": chunk},
                    ],
                    temperature=0,
                    max_tokens=2000,
                )
                results.append(resp.choices[0].message.content.strip())

        elif provider == "anthropic":
            import anthropic
            client = anthropic.Anthropic(
                api_key=os.environ.get("ANTHROPIC_API_KEY", ""), timeout=60,
            )
            for chunk in chunks:
                resp = client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=2000,
                    system=system_msg,
                    messages=[{"role": "user", "content": chunk}],
                )
                text_out = resp.content[0].text if resp.content else chunk
                results.append(text_out.strip())

        elif provider == "ollama":
            import ollama
            for chunk in chunks:
                resp = ollama.chat(
                    model="llama3.2",
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": chunk},
                    ],
                )
                text_out = resp.get("message", {}).get("content", chunk)
                results.append(text_out.strip())

        else:
            return None

        return " ".join(results)

    except Exception:
        return None


def restore_punctuation(
    text: str,
    segments: list[TranscriptSegment],
    use_llm: bool = False,
) -> tuple[str, bool]:
    """Restore punctuation to unpunctuated auto-caption text.

    Primary method: word-count heuristic + timing-based sentence boundaries
    from VTT segments. This requires NO API key and works offline.

    Optional enhancement: if use_llm=True, tries an LLM (OpenAI/Anthropic/
    Ollama, auto-detected) for better quality. This costs ~$0.01/video with
    cloud providers, or is free with local Ollama.

    Returns (punctuated_text, used_llm) tuple.
    """
    # Already punctuated? Don't touch it.
    period_count = text.count(".") + text.count("?") + text.count("!")
    if period_count > 5 or len(text) < 200:
        return text, False

    # Primary: timing-based / word-count heuristic (always works, no API)
    grouped = _group_segments_by_pauses(segments)
    if grouped:
        sentences: list[str] = []
        for raw, _start, _end in grouped:
            s = raw.strip().rstrip(",").strip()
            if not s:
                continue
            # Capitalize first letter
            s = s[0].upper() + s[1:] if s else s
            # Ensure terminal punctuation
            if s[-1] not in ".!?":
                s += "."
            sentences.append(s)
        heuristic_result = " ".join(sentences)
    else:
        heuristic_result = text

    # Optional: LLM enhancement (only if explicitly requested)
    if use_llm:
        llm_result = _restore_punctuation_llm(text)
        if llm_result and len(llm_result) > len(text) * 0.5:
            return llm_result, True

    return heuristic_result, False


# ---------------------------------------------------------------- YT-DLP OPTIONS BUILDER

def _make_ydl_opts(
    output_template: str,
    langs: list[str] | None = None,
    cookies_file: str | None = None,
    cookies_from_browser: str | None = None,
    proxy: str | None = None,
    all_langs: bool = False,
) -> dict:
    """Build yt-dlp options for subtitle extraction with optional auth/proxy."""
    opts: dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitlesformat": "vtt",
        "skip_download": True,
        "outtmpl": output_template,
    }

    # Language selection
    if all_langs:
        opts["subtitleslangs"] = ["all"]
    else:
        opts["subtitleslangs"] = langs or DEFAULT_LANGS

    # Cookie support
    if cookies_file:
        opts["cookiefile"] = cookies_file
    if cookies_from_browser:
        # yt-dlp expects a tuple: (browser, profile, keyring, container)
        opts["cookiesfrombrowser"] = (cookies_from_browser,)

    # Proxy support
    if proxy:
        opts["proxy"] = proxy

    return opts


def _make_ydl_opts_no_sub(
    cookies_file: str | None = None,
    cookies_from_browser: str | None = None,
    proxy: str | None = None,
) -> dict:
    """Build yt-dlp options for metadata-only extraction."""
    opts: dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "extract_flat": False,
    }
    if cookies_file:
        opts["cookiefile"] = cookies_file
    if cookies_from_browser:
        opts["cookiesfrombrowser"] = (cookies_from_browser,)
    if proxy:
        opts["proxy"] = proxy
    return opts


# ---------------------------------------------------------------- TRANSCRIPT WRITERS

def _write_transcript(
    info: VideoInfo,
    text: str,
    segments: list[TranscriptSegment],
    out_dir: Path,
    timestamps: bool = False,
) -> None:
    """Write transcript to .txt and optionally .tsv."""
    # Plain text version (always)
    txt_path = out_dir / f"{info.id}.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"TITLE: {info.title}\n")
        f.write(f"URL: {info.url}\n")
        f.write(f"CHANNEL: {info.channel}\n")
        f.write(f"VIDEO_ID: {info.id}\n")
        f.write(f"DURATION: {info.duration}s\n")
        f.write(f"UPLOAD_DATE: {_normalize_upload_date(info.upload_date)}\n")
        f.write(f"TRANSCRIPT_SOURCE: {info.transcript_source}\n")
        if info.transcript_lang:
            f.write(f"TRANSCRIPT_LANG: {info.transcript_lang}\n")
        f.write(f"{'=' * 60}\n\n")
        f.write(text)
    info.transcript_path = str(txt_path)

    # Timestamped version (optional)
    if timestamps and segments:
        tsv_path = out_dir / f"{info.id}.tsv"
        with open(tsv_path, "w", encoding="utf-8") as f:
            f.write(f"TITLE: {info.title}\n")
            f.write(f"URL: {info.url}\n")
            f.write(f"CHANNEL: {info.channel}\n")
            f.write(f"VIDEO_ID: {info.id}\n")
            f.write(f"DURATION: {info.duration}s\n")
            f.write(f"UPLOAD_DATE: {_normalize_upload_date(info.upload_date)}\n")
            f.write(f"TRANSCRIPT_SOURCE: {info.transcript_source}\n")
            if info.transcript_lang:
                f.write(f"TRANSCRIPT_LANG: {info.transcript_lang}\n")
            f.write(f"{'=' * 60}\n\n")
            for seg in segments:
                f.write(f"{seg.format_timestamped()}\n")
        info.timestamped_path = str(tsv_path)


def _write_multi_lang_transcript(
    info: VideoInfo,
    text: str,
    lang: str,
    out_dir: Path,
) -> str:
    """Write a language-specific transcript. Returns the path."""
    path = out_dir / f"{info.id}_{lang}.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"TITLE: {info.title}\n")
        f.write(f"URL: {info.url}\n")
        f.write(f"CHANNEL: {info.channel}\n")
        f.write(f"VIDEO_ID: {info.id}\n")
        f.write(f"DURATION: {info.duration}s\n")
        f.write(f"UPLOAD_DATE: {_normalize_upload_date(info.upload_date)}\n")
        f.write(f"TRANSCRIPT_SOURCE: {info.transcript_source}\n")
        f.write(f"TRANSCRIPT_LANG: {lang}\n")
        f.write(f"{'=' * 60}\n\n")
        f.write(text)
    return str(path)


# ---------------------------------------------------------------- CORE FUNCTIONS

def extract_metadata(
    url: str,
    cookies_file: str | None = None,
    cookies_from_browser: str | None = None,
    proxy: str | None = None,
    retries: int = DEFAULT_RETRIES,
) -> dict:
    """Extract video metadata without downloading subtitles.

    Retries transient network errors; returns an error dict on final failure.
    """
    opts = _make_ydl_opts_no_sub(cookies_file, cookies_from_browser, proxy)

    @with_retry(max_retries=retries)
    def _fetch() -> dict:
        with yt_dlp.YoutubeDL(opts) as ydl:
            return ydl.extract_info(url, download=False)

    try:
        info = _fetch()
        return {
            "id": info.get("id", ""),
            "title": info.get("title", ""),
            "url": info.get("webpage_url", url),
            "channel": info.get("uploader", info.get("channel", "")),
            "duration": info.get("duration", 0),
            "upload_date": info.get("upload_date", ""),
            "view_count": info.get("view_count", 0),
            "like_count": info.get("like_count", 0),
            "description": (info.get("description") or "")[:500],
        }
    except Exception as exc:
        err_type, err_msg = _map_error(str(exc))
        return {"error": err_msg, "error_type": err_type, "url": url}


def extract_transcript(
    url_or_id: str,
    langs: list[str] | None = None,
    output_dir: Path | None = None,
    timestamps: bool = False,
    whisper: bool = False,
    whisper_model: str = "base",
    whisper_device: str = "cpu",
    whisper_compute_type: str = "int8",
    cookies_file: str | None = None,
    cookies_from_browser: str | None = None,
    proxy: str | None = None,
    all_langs: bool = False,
    retries: int = DEFAULT_RETRIES,
) -> VideoInfo:
    """Download and parse a transcript for a single video.

    Tries captions first, then Whisper fallback if enabled.
    Handles cookies, proxy, multi-language, timestamps, and retries.
    """
    out_dir = output_dir or OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "raw_subs").mkdir(exist_ok=True)

    # Normalize to URL + extract video ID
    url = _normalize_url(url_or_id)
    vid_id = parse_video_id(url_or_id)
    if not vid_id:
        return VideoInfo(url=url, error="Could not extract video ID", error_type="invalid_url")

    # Build yt-dlp options
    template = str(out_dir / "raw_subs" / vid_id)
    ydl_opts = _make_ydl_opts(template, langs, cookies_file, cookies_from_browser, proxy, all_langs)

    info = VideoInfo(id=vid_id, url=url)

    # --- ATTEMPT 1: YouTube captions ---
    try:
        retry_decorator = with_retry(max_retries=retries)
        def _do_extract():
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                return ydl.extract_info(url, download=True)
        data = retry_decorator(_do_extract)()

        info.title = data.get("title", "")
        info.channel = data.get("uploader", data.get("channel", ""))
        info.duration = data.get("duration", 0)
        info.upload_date = data.get("upload_date", "")
        info.view_count = data.get("view_count", 0) or 0
        info.like_count = data.get("like_count", 0) or 0

        subs = data.get("subtitles", {})
        auto = data.get("automatic_captions", {})

        # Determine which language we got
        lang_found = ""
        if all_langs:
            # Download all available languages
            available_langs = list(subs.keys()) + list(auto.keys())
            if not available_langs:
                if whisper:
                    pass  # Fall through to Whisper below
                else:
                    info.error = "No subtitles available"
                    info.error_type = "no_subtitles"
                    return info

            # Download each language
            all_paths: list[str] = []
            for lang_code in available_langs:
                vtt_files = glob.glob(str(out_dir / "raw_subs" / f"{vid_id}*{lang_code}*.vtt"))
                vtt_files = [f for f in vtt_files if not f.endswith(".part")]
                if vtt_files:
                    text, segs = parse_vtt(vtt_files[0])
                    if text and len(text) >= 50:
                        path = _write_multi_lang_transcript(info, text, lang_code, out_dir)
                        all_paths.append(path)
                        if not info.has_transcript:
                            # First successful language becomes the primary:
                            # write {id}.txt (+ optional .tsv) which sets
                            # info.transcript_path / timestamped_path.
                            info.transcript_chars = len(text)
                            info.has_transcript = True
                            info.transcript_source = "caption"
                            info.transcript_lang = lang_code
                            _write_transcript(info, text, segs, out_dir, timestamps)
                        # Clean up VTT
                        for f in vtt_files:
                            try:
                                os.remove(f)
                            except OSError:
                                pass

            if all_paths:
                return info
            # No captions worked, fall through to Whisper

        else:
            # Single/preferred language mode
            has_preferred = False
            for pref_lang in (langs or DEFAULT_LANGS):
                if subs.get(pref_lang) or auto.get(pref_lang):
                    has_preferred = True
                    lang_found = pref_lang
                    break

            if not has_preferred:
                # Try any available language
                any_sub = list(subs.keys())[:1] or list(auto.keys())[:1]
                if any_sub:
                    lang_found = any_sub[0]
                else:
                    if not whisper:
                        available = list(subs.keys())[:5] or list(auto.keys())[:5]
                        info.error = f"No subtitles available" + (f". Available langs: {', '.join(available)}" if available else "")
                        info.error_type = "no_subtitles"
                        return info

    except Exception as exc:
        err_type, err_msg = _map_error(str(exc))
        info.error = err_msg
        info.error_type = err_type
        if not whisper or err_type in ("private", "deleted", "age_restricted"):
            return info
        # For network/rate-limit errors, don't try Whisper
        if err_type in ("rate_limited", "network", "region_locked"):
            return info
        # For other errors, we might still be able to get metadata for Whisper
        # Try to get metadata without subtitles
        try:
            meta = extract_metadata(url, cookies_file, cookies_from_browser, proxy)
            if "error" not in meta:
                info.title = meta.get("title", "")
                info.channel = meta.get("channel", "")
                info.duration = meta.get("duration", 0)
                info.upload_date = meta.get("upload_date", "")
        except Exception:
            pass

    # --- Parse VTT if we got captions ---
    if not all_langs:
        vtt_files = glob.glob(str(out_dir / "raw_subs" / f"{vid_id}*.vtt"))
        vtt_files = [f for f in vtt_files if not f.endswith(".part")]

        if vtt_files:
            # Prefer the VTT that matches the language we found (glob order
            # is arbitrary when multiple languages were downloaded).
            chosen = vtt_files[0]
            if lang_found:
                for f in vtt_files:
                    if f".{lang_found}." in os.path.basename(f):
                        chosen = f
                        break
            text, segments = parse_vtt(chosen)
            if text and len(text) >= 50:
                info.transcript_source = "caption"
                info.transcript_lang = lang_found or "en"
                info.transcript_chars = len(text)
                info.has_transcript = True
                _write_transcript(info, text, segments, out_dir, timestamps)
                # Store segments for downstream use (deep-research)
                info.segments = segments
                # Clean up VTT
                for f in vtt_files:
                    try:
                        os.remove(f)
                    except OSError:
                        pass
                return info
            else:
                # VTT too short, clean up
                for f in vtt_files:
                    try:
                        os.remove(f)
                    except OSError:
                        pass

    # --- ATTEMPT 2: Whisper fallback ---
    if whisper:
        if not _check_ffmpeg():
            info.error = "Whisper fallback requires ffmpeg on PATH"
            info.error_type = "no_ffmpeg"
            return info
        try:
            whisper_opts = {}
            if cookies_file:
                whisper_opts["cookiefile"] = cookies_file
            if cookies_from_browser:
                whisper_opts["cookiesfrombrowser"] = (cookies_from_browser,)
            if proxy:
                whisper_opts["proxy"] = proxy

            text, segments = transcribe_with_whisper(
                vid_id, whisper_model, whisper_device, whisper_compute_type, whisper_opts,
            )
            info.transcript_source = "whisper"
            info.transcript_lang = "en"
            info.transcript_chars = len(text)
            info.has_transcript = True
            _write_transcript(info, text, segments, out_dir, timestamps)
            return info
        except Exception as exc:
            err_type, err_msg = _map_error(str(exc))
            if info.error:
                info.error += f" | Whisper also failed: {err_msg}"
            else:
                info.error = f"Whisper failed: {err_msg}"
                info.error_type = err_type
            return info

    # No transcript available
    if not info.error:
        info.error = "No subtitles available"
        info.error_type = "no_subtitles"
    return info


def search_videos(
    query: str,
    limit: int = 10,
    transcripts: bool = False,
    langs: list[str] | None = None,
    rate_limit_sec: float = DEFAULT_RATE_LIMIT,
    output_dir: Path | None = None,
    timestamps: bool = False,
    whisper: bool = False,
    cookies_file: str | None = None,
    cookies_from_browser: str | None = None,
    proxy: str | None = None,
    retries: int = DEFAULT_RETRIES,
) -> list[VideoInfo]:
    """Search YouTube and optionally download transcripts for results."""
    opts = {
        "quiet": True, "no_warnings": True,
        "extract_flat": True, "skip_download": True,
        "playlistend": limit,
    }
    if proxy:
        opts["proxy"] = proxy
    if cookies_file:
        opts["cookiefile"] = cookies_file
    if cookies_from_browser:
        opts["cookiesfrombrowser"] = (cookies_from_browser,)

    url = f"ytsearch{limit}:{query}"
    results: list[VideoInfo] = []

    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            data = ydl.extract_info(url, download=False)
    except Exception as exc:
        err_type, err_msg = _map_error(str(exc))
        return [VideoInfo(error=err_msg, error_type=err_type)]

    entries = data.get("entries", []) if isinstance(data, dict) else []
    for entry in entries[:limit]:
        if not entry:
            continue
        vid = VideoInfo(
            id=entry.get("id", ""),
            title=entry.get("title", ""),
            url=entry.get("url", entry.get("webpage_url", "")),
            channel=entry.get("uploader", entry.get("channel", "")),
            duration=entry.get("duration", 0),
        )
        if transcripts and vid.id:
            vid = extract_transcript(
                vid.id, langs=langs, output_dir=output_dir,
                timestamps=timestamps, whisper=whisper,
                cookies_file=cookies_file, cookies_from_browser=cookies_from_browser,
                proxy=proxy, retries=retries,
            )
            time.sleep(rate_limit_sec)
        results.append(vid)

    return results


def scrape_channel(
    channel_url: str,
    limit: int = 10,
    transcripts: bool = False,
    langs: list[str] | None = None,
    rate_limit_sec: float = DEFAULT_RATE_LIMIT,
    output_dir: Path | None = None,
    timestamps: bool = False,
    whisper: bool = False,
    cookies_file: str | None = None,
    cookies_from_browser: str | None = None,
    proxy: str | None = None,
    retries: int = DEFAULT_RETRIES,
) -> list[VideoInfo]:
    """Scrape a channel's recent videos and optionally get transcripts."""
    opts = {
        "quiet": True, "no_warnings": True,
        "extract_flat": True, "skip_download": True,
        "playlistend": limit,
    }
    if proxy:
        opts["proxy"] = proxy
    if cookies_file:
        opts["cookiefile"] = cookies_file
    if cookies_from_browser:
        opts["cookiesfrombrowser"] = (cookies_from_browser,)

    results: list[VideoInfo] = []

    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            data = ydl.extract_info(channel_url, download=False)
    except Exception as exc:
        err_type, err_msg = _map_error(str(exc))
        return [VideoInfo(error=err_msg, error_type=err_type)]

    entries = data.get("entries", []) if isinstance(data, dict) else []
    for entry in entries[:limit]:
        if not entry:
            continue
        vid_id = entry.get("id", "")
        vid_url = entry.get("url", entry.get("webpage_url", canonical_url(vid_id)))
        vid = VideoInfo(
            id=vid_id,
            title=entry.get("title", ""),
            url=vid_url,
            channel=entry.get("uploader", entry.get("channel", "")),
            duration=entry.get("duration", 0),
        )
        if transcripts and vid_id:
            vid = extract_transcript(
                vid_id, langs=langs, output_dir=output_dir,
                timestamps=timestamps, whisper=whisper,
                cookies_file=cookies_file, cookies_from_browser=cookies_from_browser,
                proxy=proxy, retries=retries,
            )
            time.sleep(rate_limit_sec)
        results.append(vid)

    return results


def batch_scrape(
    file_path: str,
    transcripts: bool = True,
    langs: list[str] | None = None,
    rate_limit_sec: float = DEFAULT_RATE_LIMIT,
    output_dir: Path | None = None,
    timestamps: bool = False,
    whisper: bool = False,
    cookies_file: str | None = None,
    cookies_from_browser: str | None = None,
    proxy: str | None = None,
    retries: int = DEFAULT_RETRIES,
) -> list[VideoInfo]:
    """Batch scrape URLs from a file (one URL/ID per line)."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            urls = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    except Exception as exc:
        return [VideoInfo(error=f"Cannot read file: {exc}", error_type="file_error")]

    results: list[VideoInfo] = []
    for url in urls:
        vid = extract_transcript(
            url, langs=langs, output_dir=output_dir,
            timestamps=timestamps, whisper=whisper,
            cookies_file=cookies_file, cookies_from_browser=cookies_from_browser,
            proxy=proxy, retries=retries,
        )
        results.append(vid)
        time.sleep(rate_limit_sec)
    return results


def list_saved_transcripts(output_dir: Path | None = None) -> list[dict]:
    """List all saved transcript files."""
    out_dir = output_dir or OUTPUT_DIR
    if not out_dir.exists():
        return []

    results: list[dict] = []
    for txt_file in sorted(out_dir.glob("*.txt")):
        stat = txt_file.stat()
        # Read first few lines for metadata
        meta: dict[str, str] = {}
        try:
            with open(txt_file, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    if line.startswith("TITLE: "):
                        meta["title"] = line[7:].strip()
                    elif line.startswith("URL: "):
                        meta["url"] = line[5:].strip()
                    elif line.startswith("CHANNEL: "):
                        meta["channel"] = line[9:].strip()
                    elif line.startswith("VIDEO_ID: "):
                        meta["video_id"] = line[10:].strip()
                    elif line.startswith("TRANSCRIPT_SOURCE: "):
                        meta["source"] = line[19:].strip()
                    elif line.startswith("TRANSCRIPT_LANG: "):
                        meta["lang"] = line[17:].strip()
                    elif line.startswith("=" * 10):
                        break
        except Exception:
            pass

        results.append({
            "file": str(txt_file),
            "title": meta.get("title", "?"),
            "url": meta.get("url", ""),
            "channel": meta.get("channel", ""),
            "video_id": meta.get("video_id", txt_file.stem),
            "source": meta.get("source", "caption"),
            "lang": meta.get("lang", "en"),
            "size_bytes": stat.st_size,
            "modified": dt.datetime.fromtimestamp(stat.st_mtime).isoformat(),
        })

    return results


# ---------------------------------------------------------------- TRANSCRIPT CLEANUP

# Filler words and verbal tics to remove (case-insensitive, whole word)
_FILLER_WORDS = {
    "um", "uh", "umm", "uhh", "err", "ah", "eh",
    "you know", "i mean", "you know what i mean", "sort of", "kind of",
    "like i said", "like", "basically", "literally", "actually",
    "right", "so yeah", "and yeah", "but yeah", "or whatever",
}

# HTML entities to decode
_HTML_ENTITIES = {
    "&gt;": ">",
    "&lt;": "<",
    "&amp;": "&",
    "&quot;": '"',
    "&#39;": "'",
    "&nbsp;": " ",
}


def _decode_html(text: str) -> str:
    """Decode common HTML entities in transcript text."""
    for entity, char in _HTML_ENTITIES.items():
        text = text.replace(entity, char)
    return text


def _remove_filler_words(text: str) -> str:
    """Remove verbal filler words and tics from transcript text."""
    # Build a regex that matches any filler as a whole phrase
    fillers = sorted(_FILLER_WORDS, key=len, reverse=True)
    pattern = r"\b(?:" + "|".join(re.escape(f) for f in fillers) + r")\b"
    # Remove filler + surrounding extra spaces (but preserve newlines)
    text = re.sub(pattern, "", text, flags=re.I)
    # Collapse multiple spaces left behind (only horizontal whitespace)
    text = re.sub(r"[ \t]{2,}", " ", text)
    # Strip leading/trailing horizontal whitespace per line
    text = "\n".join(line.strip(" \t") for line in text.split("\n"))
    return text.strip(" \t")


def _handle_censors(text: str) -> str:
    """Replace YouTube's censored profanity markers with readable tags."""
    # [&nbsp;__&nbsp;] or [__] → [expletive] (before HTML decode)
    text = re.sub(r"\[?&nbsp;?__&nbsp;?\]?", "[expletive]", text)
    # [ __ ] or [__] → [expletive] (after HTML decode, with spaces)
    text = re.sub(r"\[\s*__\s*\]", "[expletive]", text)
    # [bleep] → [expletive]
    text = re.sub(r"\[bleep\]", "[expletive]", text, flags=re.I)
    return text


def _handle_speaker_markers(text: str) -> str:
    """Convert >> speaker markers to paragraph breaks."""
    # >> at start of a phrase = speaker change
    text = re.sub(r"\s*>>\s*", "\n\n", text)
    # &gt;&gt; (HTML-encoded >>)
    text = re.sub(r"\s*&gt;&gt;\s*", "\n\n", text)
    return text


def _add_punctuation_heuristics(text: str) -> str:
    """Apply basic punctuation heuristics to unpunctuated auto-captions.

    This is NOT a full punctuation model — it handles the most obvious cases:
    - Capitalize first letter of sentences (after . ! ? or start of text)
    - Capitalize standalone "i" → "I"
    - Add period at end if missing
    """
    # Capitalize "i" when standalone
    text = re.sub(r"\bi\b", "I", text)

    # Capitalize first letter of the text
    if text:
        text = text[0].upper() + text[1:]

    # Capitalize after sentence endings (if there are any)
    text = re.sub(r"([.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), text)

    # Add period at end if missing
    text = text.rstrip()
    if text and text[-1] not in ".!?\"'":
        text += "."

    return text


def _collapse_repeats(text: str) -> str:
    """Collapse immediately repeated phrases (common in auto-captions).

    e.g. "the the the truth" → "the truth"
    """
    # Remove immediate word-level repeats (3+ repeats → 1)
    text = re.sub(r"\b(\w+)\s+\1\s+\1\b", r"\1", text, flags=re.I)
    # Remove immediate phrase repeats (2+ repeats of 3-5 word phrases)
    text = re.sub(r"\b((?:\w+\s+){2,4}\w+)\s+\1\b", r"\1", text, flags=re.I)
    return text


def clean_transcript_text(raw: str) -> str:
    """Full cleanup pipeline for auto-generated transcript text.

    Steps:
    1. Decode HTML entities
    2. Handle censored profanity markers
    3. Convert speaker markers to paragraph breaks
    4. Remove filler words
    5. Collapse repeated phrases
    6. Clean up whitespace
    7. Apply basic punctuation heuristics

    Returns cleaned text. Does NOT add full punctuation (that requires
    an LLM or punctuation model) — but makes the text significantly
    more readable for comprehension.
    """
    if not raw:
        return ""

    # 1. HTML entities
    text = _decode_html(raw)

    # 2. Censored profanity
    text = _handle_censors(text)

    # 3. Speaker markers → paragraph breaks
    text = _handle_speaker_markers(text)

    # 4. Filler words
    text = _remove_filler_words(text)

    # 5. Collapse repeats
    text = _collapse_repeats(text)

    # 6. Whitespace cleanup
    # Collapse multiple spaces
    text = re.sub(r"[ \t]{2,}", " ", text)
    # Remove spaces before punctuation
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    # Collapse multiple newlines to max 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip leading/trailing whitespace per paragraph
    text = "\n\n".join(p.strip() for p in text.split("\n\n") if p.strip())

    # 7. Punctuation heuristics (per paragraph)
    paragraphs = text.split("\n\n")
    cleaned_paragraphs = [_add_punctuation_heuristics(p) for p in paragraphs]
    text = "\n\n".join(cleaned_paragraphs)

    return text.strip()


def _write_cleaned_transcript(info: VideoInfo, text: str, out_dir: Path) -> str:
    """Write a cleaned transcript file. Returns the path."""
    path = out_dir / f"{info.id}_clean.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"TITLE: {info.title}\n")
        f.write(f"URL: {info.url}\n")
        f.write(f"CHANNEL: {info.channel}\n")
        f.write(f"VIDEO_ID: {info.id}\n")
        f.write(f"DURATION: {info.duration}s\n")
        f.write(f"UPLOAD_DATE: {_normalize_upload_date(info.upload_date)}\n")
        f.write(f"TRANSCRIPT_SOURCE: {info.transcript_source}\n")
        if info.transcript_lang:
            f.write(f"TRANSCRIPT_LANG: {info.transcript_lang}\n")
        f.write(f"CLEANED: true\n")
        f.write(f"{'=' * 60}\n\n")
        f.write(text)
    return str(path)


# ---------------------------------------------------------------- LIGHT RESEARCH

# The Light Research report schema — produced by Devin (the agent) after
# reading the transcript, NOT by an external LLM API.
# This schema defines the structured output format for the "research" command.

LIGHT_RESEARCH_SCHEMA = {
    "title": "str — video title",
    "channel": "str — channel name",
    "url": "str — video URL",
    "duration_seconds": "int",
    "category": "education|tutorial|interview|podcast|review|commentary|motivation|entertainment|tech|finance|health|other",
    "tldr": "str — 2-3 sentence summary",
    "summary": "str — 5-10 sentence detailed summary",
    "key_points": ["str — 3-8 key takeaways"],
    "topics": [{"name": "str", "description": "str"}],
    "action_items": ["str — concrete actionable steps"],
    "quotes": ["str — verbatim quotes from the transcript"],
    "mentioned_resources": ["str — tools, books, people, websites mentioned"],
    "target_audience": "str",
    "difficulty": "beginner|intermediate|advanced",
    "controversial_claims": ["str — debatable or unevidenced claims"],
    "questions_raised": ["str — questions the video raises but doesn't answer"],
    "sentiment": "positive|neutral|negative|mixed",
    "energy": "calm|informative|energetic|inspirational|aggressive|comedic",
    "format": "str — lecture|conversation|interview|tutorial|review|other",
    "comprehended_by": "str — who/what produced this report",
    "comprehended_at": "str — ISO date",
}


def prepare_light_research(
    url_or_id: str,
    output_dir: Path | None = None,
    cookies_file: str | None = None,
    cookies_from_browser: str | None = None,
    proxy: str | None = None,
    retries: int = DEFAULT_RETRIES,
) -> dict:
    """Fetch transcript, clean it, and prepare it for Light Research comprehension.

    This is Step 1 of the Light Research workflow:
    1. This function: fetch + clean transcript, save both raw and cleaned versions
    2. Devin (the agent) reads the cleaned transcript and produces the report

    Returns a dict with:
    - ok: bool
    - video: VideoInfo dict
    - raw_transcript_path: str
    - cleaned_transcript_path: str
    - cleaned_chars: int
    - raw_chars: int
    - reduction_pct: float (how much the cleanup reduced the text)
    - schema: the Light Research report schema (for reference)
    - instructions: str — what to do next (read the cleaned transcript, produce report)
    """
    # Step 1: Fetch transcript
    print(f"[1/2] Fetching transcript for: {url_or_id}", file=sys.stderr)
    result = extract_transcript(
        url_or_id, output_dir=output_dir,
        cookies_file=cookies_file, cookies_from_browser=cookies_from_browser,
        proxy=proxy, retries=retries,
    )
    if result.error or not result.has_transcript:
        return {
            "ok": False,
            "error": result.error or "No transcript available",
            "error_type": result.error_type or "no_transcript",
            "video": result.to_dict(),
        }

    # Step 2: Read raw transcript and clean it
    print(f"[2/2] Cleaning transcript...", file=sys.stderr)
    raw_path = Path(result.transcript_path)
    raw_text = raw_path.read_text(encoding="utf-8", errors="replace")
    # Strip header
    header_end = raw_text.find("=" * 20)
    raw_body = raw_text[header_end:].lstrip("=\n").strip() if header_end != -1 else raw_text

    cleaned_body = clean_transcript_text(raw_body)
    raw_chars = len(raw_body)
    cleaned_chars = len(cleaned_body)
    reduction = ((raw_chars - cleaned_chars) / raw_chars * 100) if raw_chars > 0 else 0

    # Save cleaned transcript
    out = output_dir or OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    cleaned_path = _write_cleaned_transcript(result, cleaned_body, out)

    print(f"  Raw:      {raw_chars:,} chars", file=sys.stderr)
    print(f"  Cleaned:  {cleaned_chars:,} chars ({reduction:.1f}% reduction)", file=sys.stderr)
    print(f"  Saved to: {cleaned_path}", file=sys.stderr)

    return {
        "ok": True,
        "mode": "light_research",
        "video": {
            "id": result.id,
            "title": result.title,
            "channel": result.channel,
            "url": result.url,
            "duration": result.duration,
            "upload_date": _normalize_upload_date(result.upload_date),
        },
        "raw_transcript_path": str(raw_path),
        "cleaned_transcript_path": cleaned_path,
        "raw_chars": raw_chars,
        "cleaned_chars": cleaned_chars,
        "reduction_pct": round(reduction, 1),
        "schema": LIGHT_RESEARCH_SCHEMA,
        "instructions": (
            "Read the cleaned transcript at the provided path, then produce a "
            "Light Research report following the schema. The report should be "
            "saved as <video_id>_research.json in the same directory. "
            "Be specific, factual, and quote the transcript verbatim where possible. "
            "Flag controversial claims honestly. Include action items only if the "
            "video actually provides them."
        ),
    }


# ---------------------------------------------------------------- DEEP RESEARCH

# The Deep Research report schema — produced by Devin after extensive
# web research, source verification, argument mapping, and bias analysis.
# This is NOT a summary — it's a comprehensive research brief.

DEEP_RESEARCH_SCHEMA = {
    "research_mode": "deep",
    "title": "str — video title",
    "channel": "str — channel name",
    "url": "str — video URL",
    "executive_summary": "str — 3-5 paragraph nuanced summary that goes beyond what the video says",
    "argument_structure": {
        "main_thesis": "str — the central argument in one sentence",
        "premises": [{"premise": "str", "evidence_type": "empirical|anecdotal|logical|authority|intuitive|none"}],
        "conclusions": ["str"],
        "reasoning_quality": "str — assessment of logical validity",
        "fallacies_identified": [{"fallacy": "str", "example": "str", "explanation": "str"}],
    },
    "claim_verification": [
        {
            "claim": "str",
            "verbatim_quote": "str",
            "verdict": "verified|partially-verified|unverified|contradicted|unverifiable|opinion",
            "evidence": "str",
            "sources": [{"url": "str", "title": "str", "reliability": "high|moderate|low", "agreement": "supports|contradicts|mixed"}],
            "confidence": "HIGH|MODERATE|LOW",
        }
    ],
    "bias_assessment": {
        "speaker_credibility": "str",
        "potential_biases": ["str"],
        "conflicts_of_interest": ["str"],
        "financial_ecosystem": "str — what they sell, who funds them",
        "overall_reliability": "high|moderate|low|mixed",
    },
    "cross_references": [
        {
            "topic": "str",
            "this_video_claims": "str",
            "authoritative_sources_say": "str",
            "agreement_level": "consistent|partially-consistent|contradictory|insufficient-data",
            "sources": ["url"],
        }
    ],
    "omission_analysis": ["str — what's NOT said that should be"],
    "source_bibliography": [
        {"url": "str", "title": "str", "type": "academic|news|blog|video|book|government|other", "reliability": "high|moderate|low"}
    ],
    "research_gaps": ["str — what needs more research"],
    "open_questions": ["str"],
    "methodology": {
        "approach": "str",
        "sources_consulted": "int",
        "web_searches_performed": "int",
        "limitations": ["str"],
    },
    "overall_confidence": {"level": "HIGH|MODERATE|LOW", "rationale": "str"},
}


# ---------------------------------------------------------------- TIMESTAMPED SEGMENT PIPELINE

@dataclass
class TimestampedSentence:
    """A sentence extracted from a transcript segment, preserving timing."""
    text: str
    start: float       # seconds from video start
    end: float         # seconds from video start
    segment_index: int # which segment this came from

    @property
    def timestamp_str(self) -> str:
        """Human-readable timestamp like '3:42'."""
        s = int(self.start)
        if s >= 3600:
            return f"{s // 3600}:{(s % 3600) // 60:02d}:{s % 60:02d}"
        return f"{s // 60}:{s % 60:02d}"

    @property
    def youtube_url(self) -> str:
        """YouTube URL that jumps to this timestamp."""
        return f"&t={int(self.start)}s"

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "start": round(self.start, 2),
            "end": round(self.end, 2),
            "timestamp": self.timestamp_str,
        }


# ---------------------------------------------------------------- SENTENCE SEGMENTATION

# Common abbreviations that should NOT trigger sentence breaks
_ABBREVIATIONS = frozenset({
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "rev", "hon",
    "vs", "etc", "inc", "ltd", "co", "corp", "dept", "est", "approx",
    "min", "max", "avg", "fig", "no", "vol", "pp", "ch", "sec", "al",
    "u.s", "u.k", "e.g", "i.e", "a.m", "p.m", "b.c", "a.d",
})

# Title abbreviations — never split even when followed by uppercase (Dr. Smith)
_TITLE_ABBREVIATIONS = frozenset({
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "rev", "hon",
})

# Single-letter initials like "A." or "J." — don't break after these
_RE_INITIAL = re.compile(r"^\b[A-Z]\.$")


def split_sentences(text: str) -> list[str]:
    """Split text into sentences, handling abbreviations and decimals.

    Handles:
    - Abbreviations (Dr., Mr., U.S., etc.)
    - Decimal numbers (3.14, 0.5)
    - Initials (A. J. Smith)
    - Multiple punctuation (.!? combined)
    - Newlines as sentence boundaries (paragraph breaks)
    """
    if not text:
        return []

    sentences: list[str] = []
    current: list[str] = []
    i = 0
    n = len(text)

    while i < n:
        ch = text[i]
        current.append(ch)

        if ch in ".!?":
            # Look ahead to decide if this is a real sentence boundary
            # Check if next char is a space or end (real boundary candidate)
            j = i + 1
            while j < n and text[j] in " \t":
                j += 1

            if j >= n:
                # End of text — real boundary
                sentences.append("".join(current).strip())
                current = []
                i = j
                continue

            next_ch = text[j]

            # Newline = always a boundary
            if text[i + 1:i + 2] == "\n":
                sentences.append("".join(current).strip())
                current = []
                i = i + 1
                continue

            # Check for abbreviation: word before the period
            if ch == ".":
                # Get the word before this period
                word_start = i - 1
                while word_start >= 0 and text[word_start].isalpha():
                    word_start -= 1
                word = text[word_start + 1:i].lower()

                if word in _ABBREVIATIONS:
                    # Title abbreviations (Dr., Mr.) — never split
                    if word in _TITLE_ABBREVIATIONS:
                        i += 1
                        continue
                    # Other abbreviations — split if next word starts uppercase
                    if j < n and text[j].isupper():
                        sentences.append("".join(current).strip())
                        current = []
                        i = j
                        continue
                    i += 1
                    continue

                # Single capital letter (initial): "A." "J."
                if len(word) == 1 and word.isalpha():
                    i += 1
                    continue

                # Decimal: digit before and digit after the period
                if i > 0 and text[i - 1].isdigit():
                    if j < n and text[j].isdigit():
                        i += 1
                        continue

                # Ellipsis: "..."
                if text[i + 1:i + 3] == "..":
                    i += 1
                    continue

            # Multiple punctuation: "!!!" "?!?" — consume all
            while j < n and text[j] in ".!?":
                current.append(text[j])
                i = j
                j = i + 1

            # Next word starts with lowercase or is a common continuation
            if next_ch.islower() and ch != "!":
                # Likely continuation, not a new sentence
                # But "!" almost always ends a sentence
                i += 1
                continue

            # Real sentence boundary
            sentences.append("".join(current).strip())
            current = []
            i = j
            continue

        if ch == "\n":
            # Newline = sentence boundary
            sentences.append("".join(current).strip())
            current = []
            i += 1
            continue

        i += 1

    # Don't forget trailing text
    if current:
        sentences.append("".join(current).strip())

    # Filter empty and very short fragments
    return [s for s in sentences if len(s) > 2]


def segments_to_sentences(segments: list[TranscriptSegment]) -> list[TimestampedSentence]:
    """Convert transcript segments into timestamped sentences.

    Each sentence inherits the timestamp of the segment it came from.
    If a segment contains multiple sentences, they all share that segment's start time.
    """
    result: list[TimestampedSentence] = []
    for idx, seg in enumerate(segments):
        if not seg.text.strip():
            continue
        sents = split_sentences(seg.text)
        for sent in sents:
            if len(sent) < 3:
                continue
            result.append(TimestampedSentence(
                text=sent,
                start=seg.start,
                end=seg.end,
                segment_index=idx,
            ))
    return result


# ---------------------------------------------------------------- ENTITY EXTRACTION

# --- People ---
# Pattern 1: Title + Name (Dr. Jordan Peterson, Professor X)
_RE_PERSON_TITLE = re.compile(
    r"\b(?:Dr\.?|Mr\.?|Mrs\.?|Ms\.?|Professor|Prof\.?|Sir|Lord|Lady|President|"
    r"CEO|CTO|Author|Researcher|Scientist|Doctor)\s+"
    r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})",
)
# Pattern 2: "according to X", "says X", "X claims"
_RE_PERSON_ATTR = re.compile(
    r"(?i:according\s+to|said\s+by|stated\s+by|claims?\s+by|"
    r"as\s+\w+\s+said|\bper\b)\s+"
    r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})",
)
# Pattern 3: First Last (two capitalized words, not common phrases)
_RE_PERSON_NAME = re.compile(
    r"\b([A-Z][a-z]{2,}\s+[A-Z][a-z]{2,})\b"
)
# Common false positives to filter
_PERSON_BLACKLIST = frozenset({
    "The Best", "The Most", "The Only", "The First", "The Last",
    "New York", "San Francisco", "Los Angeles", "United States",
    "Wall Street", "Main Street", "High Street", "Central Park",
    "Bollinger Bands", "Moving Average", "Standard Deviation",
    "Relative Strength", "Fibonacci Retracement", "Support Resistance",
    "Bull Market", "Bear Market", "Bull Trap", "Bear Trap",
    "Candle Stick", "Candlestick Pattern", "Day Trading",
    "Swing Trading", "Position Trading", "Trend Line",
    "Stop Loss", "Take Profit", "Risk Reward",
    "Real Time", "Long Term", "Short Term", "Mid Term",
    "YouTube Channel", "Social Media", "Web Site",
    "Thank You", "Good Morning", "Good Evening", "Good Night",
    "White House", "Red Sea", "Black Sea",
    # Book titles commonly mistaken for people
    "Dopamine Nation", "Growth Mindset", "Rich Dad", "Poor Dad",
    "Market Wizards", "Trading View", "Trading Course",
    # Concept/jargon commonly mistaken for people
    "Macuna Pruriens", "Mucuna Pruriens",  # plant, not person
    "Cell Press", "Nature Reviews", "National Academy",
    "Applied Physiology", "Clinical Endocrinology",
    "European Journal", "Stanford School", "Huberman Lab",
    "Cold Exposure", "Cold Shower", "Ice Bath",
    "Growth Hormone", "Blood Flow",
})

# --- Organizations ---
_RE_ORG_SUFFIX = re.compile(
    r"\b([A-Z][A-Za-z]*(?:\s+[A-Z][A-Za-z]*){0,4})\s+"
    r"(?:Inc\.?|LLC|Ltd\.?|Corp\.?|Corporation|Company|Co\.?|"
    r"University|Institute|Foundation|Association|"
    r"Bank|Group|Holdings|Partners|Capital|Management)\b",
)
_RE_ORG_KNOWN = re.compile(
    r"\b(?:Federal\s+Reserve|Wall\s+Street|NASDAQ|NYSE|S&P\s*500|"
    r"SEC|FTC|FDA|CDC|WHO|NASA|MIT|Harvard|Stanford|Oxford|Cambridge|"
    r"Google|Apple|Microsoft|Amazon|Meta|Facebook|Netflix|Tesla|"
    r"OpenAI|Anthropic|DeepMind|Goldman\s+Sachs|JPMorgan|BlackRock|"
    r"Vanguard|Fidelity|Berkshire\s+Hathaway)\b",
)

# --- Tools / Software / Platforms ---
_RE_TOOLS = re.compile(
    r"\b(?:TradingView|MetaTrader|MT4|MT5|ThinkOrSwim|NinjaTrader|"
    r"eSignal|StockCharts|Finviz|Yahoo\s+Finance|Bloomberg|Reuters|"
    r"Coinbase|Binance|Kraken|Robinhood|Wealthfront|Betterment|"
    r"Excel|Google\s+Sheets|Python|R\s+Studio|Jupyter|"
    r"ChatGPT|Claude|Gemini|Copilot|Midjourney|Stable\s+Diffusion|"
    r"Notion|Obsidian|Roam|Linear|GitHub|GitLab|Bitbucket|"
    r"Photoshop|Illustrator|Figma|Sketch|After\s+Effects|Premiere|"
    # Health/wellness tools
    r"Headspace|Calm|Whoop|Oura\s+Ring|Eight\s+Sleep|"
    r"Oura|Levels|Dexcom|Veri|Nutrisense|"
    r"ROKA|Roka|Athletic\s+Greens|AG1|"
    r"Theragun|Hyperice|Sauna\s+Space|"
    r"InsideTracker|Function\s+Health)\b",
)

# --- Financial / Technical Metrics ---
_RE_METRICS_FINANCIAL = re.compile(
    r"\b(?:P\/E\s+ratio|price[- ]to[- ]earnings|market\s+cap|"
    r"earnings\s+per\s+share|EPS|dividend\s+yield|"
    r"alpha|beta|Sharpe\s+ratio|Sortino\s+ratio|"
    r"ROI|ROE|ROA|ROIC|EBITDA|gross\s+margin|net\s+margin|"
    r"debt[- ]to[- ]equity|current\s+ratio|quick\s+ratio)\b",
    re.I,
)
_RE_METRICS_TRADING = re.compile(
    r"\b(?:RSI|MACD|Bollinger\s+Bands?|moving\s+average|MA|EMA|SMA|WMA|"
    r"stochastic|Stoch|ATR|ADX|CCI|Williams\s+%R|OBV|"
    r"Fibonacci|Fib|pivot\s+points?|support\s+level|resistance\s+level|"
    r"stop\s+loss|take\s+profit|position\s+sizing|risk[- ]reward|"
    r"drawdown|max\s+drawdown|backtest|forward\s+test|"
    r"win\s+rate|loss\s+rate|profit\s+factor|expectancy|"
    r"Kelly\s+criterion|Martingale|DCA|dollar[- ]cost\s+averaging)\b",
    re.I,
)
_RE_METRICS_HEALTH = re.compile(
    r"\b(?:BMI|blood\s+pressure|heart\s+rate|cholesterol|LDL|HDL|triglycerides|"
    r"A1C|glucose|insulin|cortisol|testosterone|estrogen|"
    r"VO2\s+max|resting\s+heart\s+rate|sleep\s+quality)\b",
    re.I,
)

# --- Concepts / Topics (signal words that indicate topic areas) ---
_RE_CONCEPTS = re.compile(
    r"\b(?:inflation|deflation|stagflation|recession|depression|"
    r"supply\s+chain|monetary\s+policy|fiscal\s+policy|quantitative\s+easing|"
    r"interest\s+rate|yield\s+curve|bond\s+market|stock\s+market|"
    r"crypto(?:currency)?|blockchain|DeFi|NFT|Web3|"
    r"machine\s+learning|deep\s+learning|neural\s+network|"
    r"artificial\s+intelligence|AI|AGI|transformer|LLM|"
    r"intermittent\s+fasting|keto|paleo|vegan|Mediterranean\s+diet|"
    r"high[- ]intensity|HIIT|strength\s+training|cardio|"
    r"meditation|mindfulness|gratitude|stoicism)\b",
    re.I,
)


def extract_entities(text: str, trust_punctuation: bool = True) -> dict:
    """Extract named entities from text.

    Returns a dict with:
    - people: list of {name, context} — people mentioned
    - organizations: list of {name, context}
    - tools: list of {name, context} — software/platforms
    - metrics: list of {name, type, context} — financial/trading/health metrics
    - concepts: list of {name, context} — topic concepts

    Each "context" is the sentence containing the entity.
    """
    sentences = split_sentences(text)
    sentence_map: list[tuple[str, int, int]] = []  # (sentence, start, end)
    pos = 0
    for s in sentences:
        idx = text.find(s, pos)
        if idx == -1:
            pos += len(s)
            continue
        sentence_map.append((s, idx, idx + len(s)))
        pos = idx + len(s)

    def _find_context(entity: str, entity_pos: int) -> str:
        """Find the sentence containing this entity position."""
        for sent, start, end in sentence_map:
            if start <= entity_pos < end:
                return sent
        # Fallback: find by text search
        for sent, start, end in sentence_map:
            if entity.lower() in sent.lower():
                return sent
        return ""

    def _dedup(items: list[dict], key: str = "name") -> list[dict]:
        """Deduplicate by name, preserving first occurrence."""
        seen: set[str] = set()
        result: list[dict] = []
        for item in items:
            name_lower = item[key].lower().strip()
            if name_lower not in seen:
                seen.add(name_lower)
                result.append(item)
        return result

    people: list[dict] = []
    organizations: list[dict] = []
    tools: list[dict] = []
    metrics: list[dict] = []
    concepts: list[dict] = []

    # --- People ---
    for pattern in [_RE_PERSON_TITLE, _RE_PERSON_ATTR]:
        for m in pattern.finditer(text):
            name = m.group(1).strip()
            if name.split()[0].lower() in {"the", "a", "an"}:
                continue
            if name in _PERSON_BLACKLIST:
                continue
            ctx = _find_context(name, m.start(1))
            people.append({"name": name, "context": ctx})

    # Pattern 3: Two-capitalized-words (filter aggressively)
    # Only use this pattern when we have proper sentence punctuation
    # (auto-captions with no punctuation produce too many false positives)
    # Count only real sentence-ending punctuation, NOT newlines
    punct_count = text.count(".") + text.count("?") + text.count("!")
    # When trust_punctuation=False (auto-captions with fake periods),
    # require a higher threshold and check period-to-word ratio
    if trust_punctuation:
        has_sentence_punctuation = punct_count > 10
    else:
        # Real punctuation has ~1 period per 15-20 words
        word_count = len(text.split())
        period_ratio = punct_count / max(word_count, 1)
        has_sentence_punctuation = punct_count > 10 and period_ratio < 0.1

    if has_sentence_punctuation:
        for m in _RE_PERSON_NAME.finditer(text):
            name = m.group(1).strip()
            if name in _PERSON_BLACKLIST:
                continue
            # Skip if both words are common nouns
            words = name.split()
            if any(w.lower() in {"the", "best", "most", "only", "first", "last",
                                 "new", "old", "good", "bad", "big", "small",
                                 "high", "low", "long", "short", "bull", "bear",
                                 "candle", "candlestick", "moving", "average",
                                 "standard", "deviation", "relative", "strength",
                                 "support", "resistance", "stop", "loss", "take",
                                 "profit", "risk", "reward", "day", "swing",
                                 "position", "trend", "line", "white", "black",
                                 "red", "green", "blue", "thank", "good",
                                 # Additional common false positives from trading videos
                                 "financial", "market", "advanced", "traders",
                                 "divergence", "prints", "lower", "one",
                                 "direction", "styles", "tab", "line", "this",
                                 "finding", "going", "using", "unshaking",
                                 "based", "back", "here", "way", "special",
                                 "setting", "highs", "unshaking",
                                 # Common false positives from science/educational videos
                                 "with", "and", "for", "but", "or", "nor", "yet",
                                 "so", "as", "at", "by", "in", "on", "to", "of",
                                 "from", "into", "onto", "upon", "over", "under",
                                 "addiction", "learning", "rethink", "reimagining",
                                 "european", "nature", "science", "journal",
                                 "central", "valley", "giardia", "roka",
                                 "stanford", "harvard", "mit", "oxford",
                                 "huberman", "lab", "school", "university",
                                 "clinic", "hospital", "institute"} for w in words):
                continue
            # Skip if first word is a conjunction/preposition (false positive pattern)
            first_word = words[0].lower()
            if first_word in {"with", "and", "for", "but", "or", "so", "as", "at",
                              "by", "in", "on", "to", "of", "from", "into", "onto",
                              "upon", "over", "under", "yet", "nor"}:
                continue
            ctx = _find_context(name, m.start(1))
            # Only add if not already captured by title/attr patterns
            if not any(p["name"].lower() == name.lower() for p in people):
                people.append({"name": name, "context": ctx})

    # --- Organizations ---
    for m in _RE_ORG_SUFFIX.finditer(text):
        name = (m.group(1) + " " + text[m.end(1):m.end()].strip()).strip()
        ctx = _find_context(name, m.start())
        organizations.append({"name": name, "context": ctx})

    for m in _RE_ORG_KNOWN.finditer(text):
        name = m.group()
        ctx = _find_context(name, m.start())
        organizations.append({"name": name, "context": ctx})

    # --- Tools ---
    for m in _RE_TOOLS.finditer(text):
        name = m.group()
        ctx = _find_context(name, m.start())
        tools.append({"name": name, "context": ctx})

    # --- Metrics ---
    for m in _RE_METRICS_FINANCIAL.finditer(text):
        name = m.group()
        ctx = _find_context(name, m.start())
        metrics.append({"name": name, "type": "financial", "context": ctx})

    for m in _RE_METRICS_TRADING.finditer(text):
        name = m.group()
        ctx = _find_context(name, m.start())
        metrics.append({"name": name, "type": "trading", "context": ctx})

    for m in _RE_METRICS_HEALTH.finditer(text):
        name = m.group()
        ctx = _find_context(name, m.start())
        metrics.append({"name": name, "type": "health", "context": ctx})

    # --- Concepts ---
    for m in _RE_CONCEPTS.finditer(text):
        name = m.group()
        ctx = _find_context(name, m.start())
        concepts.append({"name": name, "context": ctx})

    return {
        "people": _dedup(people),
        "organizations": _dedup(organizations),
        "tools": _dedup(tools),
        "metrics": _dedup(metrics),
        "concepts": _dedup(concepts),
    }


# ---------------------------------------------------------------- CLAIM ENRICHMENT

# Negation words that flip a claim's polarity
_NEGATION_WORDS = frozenset({
    "not", "no", "never", "none", "nobody", "nothing", "nowhere",
    "neither", "nor", "cannot", "cant", "wont", "doesnt", "dont",
    "didnt", "isnt", "arent", "wasnt", "werent", "hasnt", "havent",
    "hadnt", "shouldnt", "wouldnt", "couldnt",
})

# Claim strength modifiers
_STRENGTH_HIGH = frozenset({
    "proven", "demonstrated", "established", "confirmed", "verified",
    "definitively", "conclusively", "undoubtedly", "certainly",
    "always", "never", "100%", "guaranteed", "fact", "truth",
})
_STRENGTH_MODERATE = frozenset({
    "studies", "research", "evidence", "data", "shown", "suggests",
    "indicates", "likely", "probably", "usually", "typically",
    "often", "generally", "tends", "appears",
})
_STRENGTH_LOW = frozenset({
    "might", "could", "may", "possibly", "perhaps", "maybe",
    "seems", "feels", "believe", "think", "guess", "suppose",
    "arguably", "debatably", "allegedly", "supposedly",
})


def _detect_negation(sentence: str) -> bool:
    """Check if a sentence contains negation within 3 words before a claim keyword."""
    words = sentence.lower().split()
    for i, word in enumerate(words):
        # Strip punctuation for comparison
        clean = word.strip(".,!?;:\"'()[]{}")
        if clean in _NEGATION_WORDS:
            return True
    return False


def _assess_strength(sentence: str) -> str:
    """Assess the strength of a claim: high, moderate, low, or opinion."""
    lower = sentence.lower()
    words = set(re.findall(r"\b\w+\b", lower))

    # Opinion: first-person + belief verb (checked first — trumps strength)
    opinion_markers = {"i", "me", "my", "we", "our", "feel", "believe", "think", "guess"}
    opinion_verbs = {"feel", "believe", "think", "guess", "suppose"}
    if words & opinion_verbs and words & {"i", "we"}:
        return "opinion"

    high_hits = words & _STRENGTH_HIGH
    if high_hits:
        return "high"

    low_hits = words & _STRENGTH_LOW
    moderate_hits = words & _STRENGTH_MODERATE

    if moderate_hits and not low_hits:
        return "moderate"
    if low_hits and not moderate_hits:
        return "low"
    if low_hits and moderate_hits:
        return "low"  # hedging language weakens

    return "moderate"  # default: bare assertion


def _extract_subject(sentence: str, claim_type: str, matched: str) -> str:
    """Try to extract the subject of a claim (what the claim is about).

    For "RSI causes false signals" → subject is "RSI"
    For "studies show that..." → subject is the topic after "that"
    """
    lower = sentence.lower()

    # For causal claims, look for "X causes/leads to Y" pattern
    if claim_type == "causal":
        # Try to find the subject before the causal keyword
        for kw in ["causes", "leads to", "results in", "prevents", "cures",
                    "fixes", "solves", "creates", "produces", "generates"]:
            idx = lower.find(kw)
            if idx > 0:
                subject = sentence[:idx].strip().rstrip(",.")
                # Take last 1-3 words of subject (the noun phrase)
                words = subject.split()
                if len(words) > 4:
                    return " ".join(words[-3:])
                return subject

    # For authority claims, look for the topic after "that" or "show"
    if claim_type == "authority":
        for kw in ["that", "show", "shows", "prove", "proves", "indicate", "indicates"]:
            idx = lower.find(kw)
            if idx > 0:
                after = sentence[idx + len(kw):].strip().lstrip(",.")
                words = after.split()
                if words:
                    return " ".join(words[:4])

    # For comparative claims, look for what's being compared
    if claim_type == "comparative":
        for kw in ["better than", "worse than", "more effective than",
                    "superior to", "inferior to", "outperforms"]:
            idx = lower.find(kw)
            if idx > 0:
                subject = sentence[:idx].strip().rstrip(",.")
                words = subject.split()
                if len(words) > 3:
                    return " ".join(words[-2:])
                return subject

    # For win_rate/statistical, look for the metric name
    if claim_type in ("win_rate", "statistical"):
        # Find words near the percentage/rate
        idx = lower.find(matched.lower())
        if idx > 0:
            before = sentence[:idx].strip().rstrip(",.:;")
            words = before.split()
            if words:
                return " ".join(words[-3:])

    # For financial claims, the subject is usually right before the dollar amount
    if claim_type == "financial":
        idx = lower.find(matched.lower())
        if idx > 0:
            before = sentence[:idx].strip().rstrip(",.:;")
            words = before.split()
            if words:
                return " ".join(words[-3:])

    # For superlative claims, look for what's "the best/worst"
    if claim_type == "superlative":
        for kw in ["the best", "the worst", "the only", "the most", "the fastest"]:
            idx = lower.find(kw)
            if idx > 0:
                before = sentence[:idx].strip().rstrip(",.:;")
                words = before.split()
                if words:
                    return " ".join(words[-2:])

    return ""


def enrich_claim(
    claim: dict,
    sentence: str,
    timestamp: TimestampedSentence | None = None,
) -> dict:
    """Enrich a raw claim dict with additional metadata.

    Adds:
    - subject: what the claim is about
    - negated: whether the claim is negated
    - strength: high | moderate | low | opinion
    - timestamp: {start, end, timestamp_str} if available
    - youtube_url: clickable URL to the moment in the video
    """
    claim_type = claim.get("claim_types", ["unknown"])[0] if isinstance(claim.get("claim_types"), list) else "unknown"
    matched = claim.get("matched_pattern", "")

    enriched = dict(claim)  # immutable: copy, don't mutate original
    enriched["subject"] = _extract_subject(sentence, claim_type, matched)
    enriched["negated"] = _detect_negation(sentence)
    enriched["strength"] = _assess_strength(sentence)

    if timestamp:
        enriched["timestamp"] = timestamp.to_dict()
        enriched["youtube_url"] = f"https://youtube.com/watch?v={{VIDEO_ID}}{timestamp.youtube_url}"

    return enriched


def extract_claims_enriched(
    text: str,
    sentences: list[TimestampedSentence] | None = None,
) -> list[dict]:
    """Extract and enrich claims from text, with optional timestamp mapping.

    This is the enhanced version of extract_claims() that adds:
    - Subject extraction (what the claim is about)
    - Negation detection
    - Strength assessment (high/moderate/low/opinion)
    - Timestamp mapping (if sentences provided)

    Args:
        text: The transcript text
        sentences: Pre-split timestamped sentences. If None, uses plain text.

    Returns:
        List of enriched claim dicts.
    """
    # Get raw claims using existing extraction
    raw_claims = extract_claims(text)

    if not sentences:
        # No timestamps — just enrich without timing
        return [enrich_claim(c, c.get("sentence", "")) for c in raw_claims]

    # Build a position → timestamp mapping
    # For each raw claim, find which timestamped sentence it belongs to
    enriched: list[dict] = []
    for claim in raw_claims:
        claim_sentence = claim.get("sentence", "")
        claim_offset = claim.get("char_offset", 0)

        # Find the matching timestamped sentence
        best_ts = None
        best_score = -1

        for ts_sent in sentences:
            # Score by text overlap
            if claim_sentence.strip() == ts_sent.text.strip():
                best_ts = ts_sent
                break
            # Partial overlap: check if claim sentence contains the ts sentence
            # or vice versa
            if ts_sent.text in claim_sentence or claim_sentence in ts_sent.text:
                overlap = min(len(ts_sent.text), len(claim_sentence))
                if overlap > best_score:
                    best_ts = ts_sent
                    best_score = overlap

        enriched.append(enrich_claim(claim, claim_sentence, best_ts))

    return enriched


# ---------------------------------------------------------------- ADVANCED EXTRACTION (10 FEATURES)

# === 1. CONTRADICTION DETECTION ============================================

# Pairs of words that are semantic opposites
_ANTONYM_PAIRS = [
    ({"always", "every", "all"}, {"never", "none", "no"}),
    ({"works", "effective", "profitable", "good", "great", "best"}, {"fails", "ineffective", "unprofitable", "bad", "worst", "useless"}),
    ({"buy", "long", "bullish"}, {"sell", "short", "bearish"}),
    ({"safe", "risk-free", "guaranteed"}, {"risky", "dangerous", "speculative"}),
    ({"easy", "simple", "straightforward"}, {"hard", "difficult", "complex", "complicated"}),
    ({"true", "correct", "accurate", "right"}, {"false", "wrong", "incorrect", "inaccurate"}),
    ({"increase", "grow", "rise", "gain"}, {"decrease", "shrink", "fall", "lose"}),
    ({"proven", "confirmed", "verified"}, {"debunked", "disproven", "refuted"}),
    ({"legal", "allowed", "permitted"}, {"illegal", "banned", "prohibited"}),
    ({"necessary", "required", "essential"}, {"unnecessary", "optional", "inessential"}),
    ({"recommend", "endorse", "support"}, {"avoid", "warn against", "oppose"}),
]

# Negation flip: if one claim is negated and the other isn't, that's a contradiction
def _subjects_match(subj_a: str, subj_b: str) -> bool:
    """Check if two subjects are the same (fuzzy matching)."""
    if not subj_a or not subj_b:
        return False
    # Exact match
    if subj_a == subj_b:
        return True
    # One is substring of the other
    if subj_a in subj_b or subj_b in subj_a:
        return True
    # Shared key word (both contain a significant word)
    words_a = set(subj_a.split()) - {"the", "a", "an", "is", "are", "was", "were", "this", "that", "it"}
    words_b = set(subj_b.split()) - {"the", "a", "an", "is", "are", "was", "were", "this", "that", "it"}
    shared = words_a & words_b
    # If they share a significant word (3+ chars), they're related
    if any(len(w) >= 3 for w in shared):
        return True
    return False


def _claims_contradict(claim_a: dict, claim_b: dict) -> tuple[bool, str]:
    """Check if two claims contradict each other.

    Returns (is_contradiction, reason).
    """
    sent_a = claim_a.get("sentence", "").lower()
    sent_b = claim_b.get("sentence", "").lower()
    neg_a = claim_a.get("negated", False)
    neg_b = claim_b.get("negated", False)

    # Same subject, different polarity → contradiction
    subj_a = claim_a.get("subject", "").lower().strip()
    subj_b = claim_b.get("subject", "").lower().strip()

    # Check antonym pairs
    for positive_set, negative_set in _ANTONYM_PAIRS:
        a_pos = any(w in sent_a for w in positive_set)
        a_neg = any(w in sent_a for w in negative_set)
        b_pos = any(w in sent_b for w in positive_set)
        b_neg = any(w in sent_b for w in negative_set)

        # A says positive, B says negative (or vice versa)
        if (a_pos and b_neg) or (a_neg and b_pos):
            # Use fuzzy subject matching
            if _subjects_match(subj_a, subj_b):
                return True, f"Related subject with opposite polarity"

    # Negation flip: A says X, B says NOT X (about same subject)
    if _subjects_match(subj_a, subj_b):
        if neg_a != neg_b:
            return True, f"Same subject with negation flip"

    # Also check: same sentence text with negation flip (no subject needed)
    if sent_a and sent_b and neg_a != neg_b:
        # If sentences are very similar (same claim, different polarity)
        if subj_a and subj_b and _subjects_match(subj_a, subj_b):
            return True, f"Negation flip on same subject"

    return False, ""


def detect_contradictions(claims: list[dict]) -> list[dict]:
    """Detect contradictions between claims in the same video.

    Compares every pair of claims and flags contradictions based on:
    - Antonym pairs (works/fails, buy/sell, safe/risky)
    - Negation flips (same subject, opposite polarity)
    - Same subject, opposite strength (high vs contradicted)

    Returns list of contradiction dicts:
    - claim_a: first claim sentence
    - claim_b: second claim sentence
    - timestamp_a, timestamp_b: when each was said
    - reason: why they contradict
    """
    contradictions: list[dict] = []
    n = len(claims)

    for i in range(n):
        for j in range(i + 1, n):
            a, b = claims[i], claims[j]
            is_contra, reason = _claims_contradict(a, b)
            if is_contra:
                contradictions.append({
                    "claim_a": a.get("sentence", ""),
                    "claim_b": b.get("sentence", ""),
                    "timestamp_a": a.get("timestamp", {}).get("timestamp", "") if isinstance(a.get("timestamp"), dict) else "",
                    "timestamp_b": b.get("timestamp", {}).get("timestamp", "") if isinstance(b.get("timestamp"), dict) else "",
                    "subject": a.get("subject", ""),
                    "reason": reason,
                })

    return contradictions


# === 2. MARKETING PRESSURE DETECTION ======================================

_RE_URGENCY = re.compile(
    r"\b(?:act\s+now|hurry|limited\s+time|last\s+chance|"
    r"only\s+\d+\s+(?:spots?|seats?|copies?|left|remaining|available)|"
    r"selling\s+out|going\s+fast|"
    r"deadline|expires?|expiring|"
    r"today\s+only|this\s+week\s+only|"
    r"before\s+it.s\s+(?:too\s+late|gone)|"
    r"don.t\s+(?:miss|wait)|"
    r"once\s+in\s+a\s+lifetime|"
    r"now\s+is\s+the\s+time)\b",
    re.I,
)
_RE_SCARCITY = re.compile(
    r"\b(?:only\s+\d+\s*(?:%|percent)?\s*(?:off|discount)|"
    r"exclusive\s+(?:offer|deal|access)|"
    r"members?\s+only|vip\s+only|"
    r"invite[- ]only|"
    r"not\s+(?:everyone|everyone\s+can|available\s+to\s+everyone)|"
    r"secret\s+(?:link|method|strategy|access)|"
    r"special\s+(?:link|offer|price|deal)|"
    r"bonus\s+(?:if|when)\s+you\s+(?:join|sign\s+up|buy|purchase)|"
    r"free\s+(?:bonus|gift|trial)|"
    r"money[- ]back\s+guarantee|"
    r"risk[- ]free|"
    r"act\s+fast|"
    r"(?:price|cost).{0,20}(?:going\s+up|increasing|rising))\b",
    re.I,
)
_RE_SOCIAL_PROOF = re.compile(
    r"\b(?:join\s+\d{1,3}(?:,\d{3})*\s+(?:traders?|members?|students?|users?|people)|"
    r"\d{1,3}(?:,\d{3})*\s+(?:traders?|members?|students?|users?|people)\s+(?:have|already|use)|"
    r"thousands\s+of\s+(?:traders?|students?|people)|"
    r"what\s+(?:our|the)\s+(?:members?|students?|traders?)\s+say|"
    r"testimonials?|"
    r"verified\s+(?:results?|reviews?)|"
    r"\d+\s+star\s+(?:rating|review))\b",
    re.I,
)
_RE_AFFILIATE = re.compile(
    r"\b(?:affiliate\s+link|referral\s+link|"
    r"use\s+(?:my|the)\s+link|"
    r"link\s+in\s+(?:the\s+)?description|"
    r"link\s+below|"
    r"promo\s+code|coupon\s+code|discount\s+code|"
    r"sponsored\s+by|brought\s+to\s+you\s+by|"
    r"partner|partnership|"
    # Sponsor read patterns (common in podcasts/YouTube)
    r"today.s\s+(?:episode|video)\s+is\s+(?:sponsored|brought\s+to\s+you)|"
    r"this\s+(?:episode|video)\s+is\s+(?:sponsored|brought\s+to\s+you)|"
    r"our\s+sponsor|"
    r"check\s+out\s+(?:our\s+)?sponsor|"
    r"special\s+thanks\s+to\s+(?:our\s+)?sponsor|"
    r"use\s+code\s+\w+|"
    r"visit\s+\w+\.com|"
    r"go\s+to\s+\w+\.com|"
    r"head\s+over\s+to\s+\w+\.com|"
    r"if\s+you.d\s+like\s+to\s+try\s+\w+|"
    r"if\s+you.re\s+(?:interested|looking)\s+in\s+\w+.{0,30}(?:link|code|below)|"
    r"discount\s+at\s+\w+|"
    r"special\s+(?:offer|deal|price)\s+for\s+(?:you|our|the)\s+(?:viewers?|listeners?|audience))\b",
    re.I,
)


def detect_marketing_pressure(text: str) -> dict:
    """Detect marketing manipulation tactics in the transcript.

    Returns a dict with:
    - urgency: list of {phrase, sentence, timestamp?}
    - scarcity: list of {phrase, sentence}
    - social_proof: list of {phrase, sentence}
    - affiliate: list of {phrase, sentence}
    - pressure_score: 0-100 (higher = more manipulative)
    - summary: human-readable assessment
    """
    sentences = split_sentences(text)
    sentence_map = []
    pos = 0
    for s in sentences:
        idx = text.find(s, pos)
        if idx == -1:
            pos += len(s)
            continue
        sentence_map.append((s, idx, idx + len(s)))
        pos = idx + len(s)

    def _find_context(phrase: str, pos: int) -> str:
        for sent, start, end in sentence_map:
            if start <= pos < end:
                return sent
        return ""

    def _extract(pattern, label):
        results = []
        for m in pattern.finditer(text):
            ctx = _find_context(m.group(), m.start())
            results.append({"phrase": m.group(), "sentence": ctx})
        return results

    urgency = _extract(_RE_URGENCY, "urgency")
    scarcity = _extract(_RE_SCARCITY, "scarcity")
    social_proof = _extract(_RE_SOCIAL_PROOF, "social_proof")
    affiliate = _extract(_RE_AFFILIATE, "affiliate")

    total_hits = len(urgency) + len(scarcity) + len(social_proof) + len(affiliate)
    # Score: 0 hits = 0, 1-2 = 25, 3-5 = 50, 6-10 = 75, 10+ = 100
    if total_hits == 0:
        score = 0
    elif total_hits <= 2:
        score = 25
    elif total_hits <= 5:
        score = 50
    elif total_hits <= 10:
        score = 75
    else:
        score = 100

    if score == 0:
        summary = "No marketing pressure tactics detected."
    elif score <= 25:
        summary = "Low marketing pressure — minor promotional language."
    elif score <= 50:
        summary = "Moderate marketing pressure — some urgency/scarcity tactics."
    elif score <= 75:
        summary = "High marketing pressure — multiple manipulation tactics detected."
    else:
        summary = "Extreme marketing pressure — heavy use of urgency, scarcity, and social proof."

    return {
        "urgency": urgency,
        "scarcity": scarcity,
        "social_proof": social_proof,
        "affiliate": affiliate,
        "pressure_score": score,
        "total_tactics": total_hits,
        "summary": summary,
    }


# === 3. NUMERIC PARAMETER EXTRACTION ======================================

_RE_PARAM_INDICATOR = re.compile(
    r"\b(?:RSI|MACD|Bollinger|moving\s+average|MA|EMA|SMA|stochastic|ATR|ADX|"
    r"stop\s+loss|take\s+profit|risk\s+reward|position\s+size|"
    r"timeframe|time\s+frame|period|length|setting|parameter)\b",
    re.I,
)
_RE_NUMBER = re.compile(r"\b(\d+(?:\.\d+)?)\b")
_RE_PARAM_PATTERN = re.compile(
    r"\b(RSI|MACD|Bollinger|moving\s+average|MA|EMA|SMA|stochastic|ATR|ADX|CCI|"
    r"stop\s+loss|take\s+profit|risk\s+reward|position\s+size|"
    r"timeframe|time\s+frame|period|length|setting|parameter|"
    r"overbought|oversold|threshold|crossover|"
    r"leverage|margin|risk\s+per\s+trade|"
    r"win\s+rate|profit\s+factor|drawdown|expectancy)"
    r"\s*(?:of|is|at|set\s+to|=|:|to)?\s*"
    r"(\d+(?:\.\d+)?)\b",
    re.I,
)
_RE_RANGE = re.compile(
    r"\b((?:overbought|oversold|RSI|level|threshold)"
    r"\s*(?:of|is|at|set\s+to|=|:)?\s*"
    r"(\d+(?:\.\d+)?)\s*(?:to|-|–|until)\s*(\d+(?:\.\d+)?))\b",
    re.I,
)


def extract_parameters(text: str) -> list[dict]:
    """Extract numeric strategy parameters from text.

    Catches things like:
    - "RSI length 14"
    - "overbought at 70"
    - "stop loss of 2%"
    - "risk reward 1:3"
    - "leverage 10x"
    - "win rate 86%"

    Returns list of {parameter, value, unit, sentence} dicts.
    """
    results: list[dict] = []
    seen: set[str] = set()

    # Pattern-based extraction
    for m in _RE_PARAM_PATTERN.finditer(text):
        param = m.group(1).strip().lower()
        value = m.group(2)
        key = f"{param}:{value}"
        if key in seen:
            continue
        seen.add(key)

        # Find context sentence
        sent_start = text.rfind(".", 0, m.start())
        sent_start = sent_start + 1 if sent_start != -1 else 0
        sent_end = text.find(".", m.end())
        sent_end = sent_end + 1 if sent_end != -1 else len(text)
        sentence = text[sent_start:sent_end].strip()

        results.append({
            "parameter": param,
            "value": float(value) if "." in value else int(value),
            "unit": "",
            "sentence": sentence,
        })

    # Range extraction (overbought 70-30, RSI 30-70)
    for m in _RE_RANGE.finditer(text):
        param = m.group(1).strip().lower()
        val_low = m.group(2)
        val_high = m.group(3)
        key = f"{param}:{val_low}-{val_high}"
        if key in seen:
            continue
        seen.add(key)

        sent_start = text.rfind(".", 0, m.start())
        sent_start = sent_start + 1 if sent_start != -1 else 0
        sent_end = text.find(".", m.end())
        sent_end = sent_end + 1 if sent_end != -1 else len(text)
        sentence = text[sent_start:sent_end].strip()

        results.append({
            "parameter": param,
            "value": f"{val_low}-{val_high}",
            "unit": "range",
            "sentence": sentence,
        })

    # Risk reward ratio (1:3, 1:2)
    for m in re.finditer(r"\brisk\s+reward\s+(?:ratio\s+)?(?:of\s+)?(\d+):(\d+)\b", text, re.I):
        key = f"risk_reward:{m.group(1)}:{m.group(2)}"
        if key in seen:
            continue
        seen.add(key)
        results.append({
            "parameter": "risk_reward_ratio",
            "value": f"{m.group(1)}:{m.group(2)}",
            "unit": "ratio",
            "sentence": text[max(0, m.start()-30):min(len(text), m.end()+30)].strip(),
        })

    # Leverage (10x, 50x)
    for m in re.finditer(r"\b(\d+)x\s+leverage\b|\bleverage\s+(?:of\s+)?(\d+)x\b", text, re.I):
        val = m.group(1) or m.group(2)
        key = f"leverage:{val}"
        if key in seen:
            continue
        seen.add(key)
        results.append({
            "parameter": "leverage",
            "value": int(val),
            "unit": "x",
            "sentence": text[max(0, m.start()-30):min(len(text), m.end()+30)].strip(),
        })

    # Percentage parameters (2% risk, 1% per trade)
    for m in re.finditer(r"\b(\d+(?:\.\d+)?)\s*%\s*(?:risk|per\s+trade|per\s+position|stop\s+loss|drawdown)", text, re.I):
        key = f"pct:{m.group(1)}"
        if key in seen:
            continue
        seen.add(key)
        results.append({
            "parameter": "percentage",
            "value": float(m.group(1)),
            "unit": "%",
            "sentence": text[max(0, m.start()-30):min(len(text), m.end()+30)].strip(),
        })

    return results


# === 4. CONDITION/ACTION RULE EXTRACTION ==================================

_RE_IF_THEN = re.compile(
    r"\b(?:if|when|whenever|once|after)\s+(.{5,120}?)"
    r"(?:,\s*|\s+,\s*|\s+(?:then?|you\s+(?:should|can|must|need\s+to)|just|simply)\s+)"
    r"(.{3,120}?)(?:[.!?]|\n|$)",
    re.I,
)
_RE_WHEN_DO = re.compile(
    r"\b(?:when|whenever)\s+(.{5,100}?)\s*,?\s+"
    r"(?:you\s+)?(do|use|apply|set|place|enter|exit|buy|sell|go|switch|move|add|remove|check|look\s+for)\s+(.{5,100}?)(?:[.!?]|\n|$)",
    re.I,
)


def extract_rules(text: str) -> list[dict]:
    """Extract if/then condition-action rules from text.

    Catches trading rules like:
    - "If RSI is below 30, buy"
    - "When the MACD crosses, sell"
    - "Once price hits support, enter"

    Returns list of {condition, action, sentence} dicts.
    """
    rules: list[dict] = []
    seen: set[str] = set()

    for m in _RE_IF_THEN.finditer(text):
        condition = m.group(1).strip().rstrip(",.")
        action = m.group(2).strip().rstrip(",.")

        # Filter out non-rules (too short, no action verb)
        if len(condition) < 5 or len(action) < 3:
            continue
        # Must have a trading-relevant action verb
        action_verbs = {"buy", "sell", "enter", "exit", "set", "place", "use", "apply",
                       "go", "switch", "move", "add", "remove", "check", "look",
                       "close", "open", "increase", "decrease", "raise", "lower",
                       "stop", "start", "avoid", "wait", "confirm", "verify",
                       "take", "put", "cut", "dump", "load", "unload"}
        # Common non-trading phrases to filter out
        non_rule_phrases = {"want to", "pause", "watch", "learn", "understand",
                           "take a look", "look at", "see", "find", "notice",
                           "know", "think", "remember", "forget",
                           "love", "like", "enjoy", "prefer", "feel",
                           "hope", "wish", "believe", "guess",
                           "talk", "discuss", "explain", "describe",
                           "show", "demonstrate", "illustrate",
                           "give", "offer", "provide", "share",
                           "try", "attempt", "consider"}
        # Trading-relevant keywords — at least one must appear in condition or action
        # Use specific compound terms to avoid false positives from ambiguous words
        trading_keywords = {"rsi", "macd", "bollinger", "moving average", "stochastic",
                           "atr", "adx", "ema", "sma", "indicator", "signal",
                           "buy", "sell", "bullish", "bearish",
                           "stop loss", "take profit", "risk reward", "leverage",
                           "position size", "trade", "trading", "entry", "exit",
                           "support level", "resistance level", "breakout", "reversal",
                           "candlestick", "chart pattern", "price action", "volume",
                           "overbought", "oversold", "divergence",
                           "portfolio", "invest", "asset", "stock", "crypto",
                           "forex", "futures", "options", "margin call",
                           # Health/science rules — specific terms only
                           "dose", "supplement", "sleep", "exercise", "diet",
                           "intermittent fasting", "cold exposure", "cold shower",
                           "ice bath", "sauna", "protocol", "prescription"}
        action_lower = action.lower()
        condition_lower = condition.lower()
        combined_lower = condition_lower + " " + action_lower

        # Skip if the action is just "look at" or "take a look" — not a rule
        if any(nrp in action_lower for nrp in non_rule_phrases):
            continue
        # Skip if condition is too generic ("you guys", "you have", "we take")
        if condition_lower in {"you guys", "you have", "we take", "you want",
                               "we have", "you can", "we can", "you need",
                               "we need", "you see", "we see", "i work",
                               "i drive", "there are", "there is", "it is",
                               "this is", "that is", "here is"}:
            continue
        # Require at least one trading/health keyword in the rule
        if not any(kw in combined_lower for kw in trading_keywords):
            continue
        if not any(av in action_lower for av in action_verbs):
            continue

        key = f"{condition[:50]}:{action[:50]}".lower()
        if key in seen:
            continue
        seen.add(key)

        full_sentence = text[max(0, m.start()-10):min(len(text), m.end()+10)].strip()
        rules.append({
            "condition": condition,
            "action": action,
            "sentence": full_sentence,
        })

    return rules


# === 5. CITATION DECOMPOSITION ============================================

_RE_CITATION_FULL = re.compile(
    r"\b((?:19|20)\d{2})\s+"
    r"((?:Harvard|Stanford|MIT|Oxford|Cambridge|Yale|Princeton|"
    r"Johns\s+Hopkins|Columbia|UCLA|UC\s+Berkeley|"
    r"Federal\s+Reserve|NIH|NASA|WHO|CDC|FDA|"
    r"Goldman\s+Sachs|JPMorgan|BlackRock)?\s*"
    r"(?:University\s+of\s+\w+|Institute\s+of\s+\w+)?)\s*"
    r"(?:study|research|paper|report|trial|survey|analysis)\s+"
    r"(?:of\s+)?(\d+(?:,\d{3})*)?\s*"
    r"(?:patients?|participants?|subjects?|people|traders?|students?|cases?)?",
    re.I,
)
_RE_CITATION_JOURNAL = re.compile(
    r"\b(?:published\s+in\s+|appearing\s+in\s+|from\s+)"
    r"(?:the\s+)?(Journal\s+of\s+\w+|Nature|Science|Lancet|JAMA|BMJ|"
    r"New\s+England\s+Journal|Wall\s+Street\s+Journal|Financial\s+Times)\b",
    re.I,
)
_RE_CITATION_AUTHOR = re.compile(
    r"\b(?:by|from|led\s+by)\s+"
    r"(?:Dr\.?\s+|Professor\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\s*"
    r"(?:et\s+al\.?)?",
)


def extract_citations(text: str) -> list[dict]:
    """Decompose authority citations into structured data.

    Turns "A 2023 Harvard study of 10,000 patients showed..."
    into {year: 2023, institution: "Harvard", sample_size: 10000, type: "study"}

    Returns list of citation dicts.
    """
    citations: list[dict] = []
    seen: set[str] = set()

    for m in _RE_CITATION_FULL.finditer(text):
        year = m.group(1)
        institution = (m.group(2) or "").strip()
        sample_size = m.group(3) if m.group(3) else ""

        key = f"{year}:{institution}:{sample_size}"
        if key in seen:
            continue
        seen.add(key)

        # Find context
        sent_start = text.rfind(".", 0, m.start())
        sent_start = sent_start + 1 if sent_start != -1 else 0
        sent_end = text.find(".", m.end())
        sent_end = sent_end + 1 if sent_end != -1 else len(text)
        sentence = text[sent_start:sent_end].strip()

        citation = {
            "year": int(year) if year else None,
            "institution": institution if institution else None,
            "sample_size": int(sample_size.replace(",", "")) if sample_size else None,
            "type": "study",
            "sentence": sentence,
            "verifiable": bool(year and institution),
        }
        citations.append(citation)

    # Journal references
    for m in _RE_CITATION_JOURNAL.finditer(text):
        journal = m.group(1).strip()
        key = f"journal:{journal.lower()}"
        if key in seen:
            continue
        seen.add(key)

        sent_start = text.rfind(".", 0, m.start())
        sent_start = sent_start + 1 if sent_start != -1 else 0
        sent_end = text.find(".", m.end())
        sent_end = sent_end + 1 if sent_end != -1 else len(text)

        citations.append({
            "year": None,
            "institution": journal,
            "sample_size": None,
            "type": "journal",
            "sentence": text[sent_start:sent_end].strip(),
            "verifiable": True,
        })

    # Author references
    for m in _RE_CITATION_AUTHOR.finditer(text):
        author = m.group(1).strip()
        if author in _PERSON_BLACKLIST:
            continue
        key = f"author:{author.lower()}"
        if key in seen:
            continue
        seen.add(key)

        citations.append({
            "year": None,
            "institution": None,
            "sample_size": None,
            "author": author,
            "type": "author_reference",
            "sentence": text[max(0, m.start()-20):min(len(text), m.end()+40)].strip(),
            "verifiable": True,
        })

    return citations


# === 6. QUESTION EXTRACTION ===============================================

_RE_QUESTION = re.compile(
    r"([A-Z][^.!?]{5,200}?\?)",
)
_RE_RHETORICAL = re.compile(
    r"\b(?:who\s+(?:knows|cares)|what\s+if|why\s+(?:not|bother)|"
    r"does\s+it\s+(?:really|actually)\s+(?:matter|work)|"
    r"is\s+it\s+(?:really|actually)\s+(?:worth|necessary)|"
    r"can\s+you\s+(?:really|actually)|"
    r"haven.t\s+you\s+(?:heard|seen|noticed)|"
    r"don.t\s+you\s+(?:think|want|agree))\b",
    re.I,
)


def extract_questions(text: str) -> list[dict]:
    """Extract questions from the transcript.

    Every question is a research lead. Categorizes as:
    - research: a genuine question that should be answered
    - rhetorical: a rhetorical question (still useful for bias assessment)
    - sales: a leading question used in marketing ("Want to make money?")

    Returns list of {question, type, sentence} dicts.
    """
    questions: list[dict] = []
    seen: set[str] = set()

    for m in _RE_QUESTION.finditer(text):
        q = m.group(1).strip()
        key = q.lower()[:80]
        if key in seen:
            continue
        seen.add(key)

        # Classify
        q_lower = q.lower()
        if _RE_RHETORICAL.search(q_lower):
            q_type = "rhetorical"
        elif any(w in q_lower for w in ["want to", "would you", "are you ready", "do you want"]):
            q_type = "sales"
        else:
            q_type = "research"

        questions.append({
            "question": q,
            "type": q_type,
        })

    return questions


# === 7. HEDGE WORD DENSITY ================================================

_HEDGE_WORDS = frozenset({
    "sort", "kind", "basically", "essentially", "arguably", "debatably",
    "allegedly", "supposedly", "presumably", "ostensibly", "apparently",
    "seemingly", "purportedly", "reportedly", "presumably",
    "maybe", "perhaps", "possibly", "probably", "likely", "unlikely",
    "somewhat", "relatively", "fairly", "quite", "rather",
    "tend", "tends", "generally", "usually", "typically", "often",
    "mostly", "largely", "mainly", "primarily",
    "i", "think", "guess", "suppose", "assume", "believe", "feel",
    "seems", "appears", "looks", "sounds",
})


def calculate_hedge_density(text: str) -> dict:
    """Calculate hedge word density — how uncertain the speaker is.

    High hedge density = low confidence = less trustworthy for factual claims.

    Returns:
    - hedge_count: total hedge words
    - word_count: total words
    - density: hedge words per 100 words
    - level: low | moderate | high | extreme
    - hedges_found: list of which hedge words were found
    """
    words = re.findall(r"\b\w+\b", text.lower())
    word_count = len(words)
    if word_count == 0:
        return {"hedge_count": 0, "word_count": 0, "density": 0.0, "level": "low", "hedges_found": []}

    hedge_found: list[str] = []
    for w in words:
        if w in _HEDGE_WORDS:
            hedge_found.append(w)

    hedge_count = len(hedge_found)
    density = (hedge_count / word_count) * 100

    if density < 1:
        level = "low"
    elif density < 3:
        level = "moderate"
    elif density < 5:
        level = "high"
    else:
        level = "extreme"

    # Count unique hedges
    from collections import Counter
    hedge_counter = dict(Counter(hedge_found))

    return {
        "hedge_count": hedge_count,
        "word_count": word_count,
        "density": round(density, 2),
        "level": level,
        "hedges_found": hedge_counter,
    }


# === 8. TOPIC SEGMENTATION (AUTO-CHAPTERING) ==============================

_TOPIC_TRANSITION_WORDS = re.compile(
    r"\b(?:so\s+(?:now|let.s|today|first|next)|"
    r"let.s\s+(?:talk|move|go|start|dive|look|get)|"
    r"now\s+(?:let.s|we|I|the|this|moving)|"
    r"moving\s+on|next\s+(?:up|thing|topic|section|let.s)|"
    r"first\s+(?:thing|of\s+all|let.s)|"
    r"to\s+start|to\s+begin|"
    r"by\s+the\s+way|"
    r"anyway|"
    r"with\s+that\s+(?:said|out\s+of\s+the\s+way)|"
    r"that\s+said|"
    r"wrapping\s+up|in\s+conclusion|to\s+summarize|"
    r"finally|lastly|in\s+the\s+end)\b",
    re.I,
)

_TOPIC_INTRO_PATTERNS = re.compile(
    r"\b(?:what\s+is|how\s+to|why\s+\w+|let.s\s+talk\s+about|"
    r"today\s+we.re\s+(?:going\s+to|talking\s+about)|"
    r"this\s+is\s+(?:called|known\s+as)|"
    r"the\s+(?:first|second|third|next|last)\s+(?:thing|step|part|concept|rule))\b",
    re.I,
)


def segment_topics(
    text: str,
    sentences: list[TimestampedSentence] | None = None,
    min_segment_words: int = 50,
) -> list[dict]:
    """Auto-segment transcript into topic chapters.

    Detects topic transitions based on:
    - Transition phrases ("let's move on", "next up", "now let's")
    - Topic introductions ("what is X", "how to Y")
    - Long pauses (if timestamps available)

    Returns list of chapter dicts:
    - title: auto-generated title
    - start_timestamp: when chapter starts
    - end_timestamp: when chapter ends
    - word_count: words in chapter
    - first_sentence: opening sentence
    """
    if sentences is None:
        # Create pseudo-sentences from text
        sents = split_sentences(text)
        sentences = [
            TimestampedSentence(text=s, start=0.0, end=0.0, segment_index=i)
            for i, s in enumerate(sents)
        ]

    if not sentences:
        return []

    # Find transition points
    transitions: list[int] = [0]  # always start with first sentence
    for i, ts in enumerate(sentences[1:], 1):
        if _TOPIC_TRANSITION_WORDS.search(ts.text) or _TOPIC_INTRO_PATTERNS.search(ts.text):
            # Don't create too-short segments
            if i - transitions[-1] >= 3:
                transitions.append(i)

    # Build chapters
    chapters: list[dict] = []
    for ci, start_idx in enumerate(transitions):
        end_idx = transitions[ci + 1] if ci + 1 < len(transitions) else len(sentences)
        chapter_sents = sentences[start_idx:end_idx]
        if not chapter_sents:
            continue

        chapter_text = " ".join(s.text for s in chapter_sents)
        word_count = len(chapter_text.split())

        if word_count < min_segment_words and ci > 0:
            # Too short — merge into previous chapter
            if chapters:
                chapters[-1]["end_timestamp"] = chapter_sents[-1].timestamp_str if chapter_sents else chapters[-1]["end_timestamp"]
                chapters[-1]["word_count"] += word_count
                continue

        # Auto-generate title from first sentence
        first = chapter_sents[0].text
        # Try to extract a topic from the first sentence
        title = first[:60] + "..." if len(first) > 60 else first

        chapters.append({
            "chapter": ci + 1,
            "title": title,
            "start_timestamp": chapter_sents[0].timestamp_str,
            "end_timestamp": chapter_sents[-1].timestamp_str if chapter_sents else chapter_sents[0].timestamp_str,
            "start_seconds": chapter_sents[0].start,
            "word_count": word_count,
            "sentence_count": len(chapter_sents),
            "first_sentence": first,
        })

    return chapters


# === 9. EMOTIONAL LANGUAGE TRACKING =======================================

_EMOTIONAL_WORDS = {
    "amazing": 3, "incredible": 3, "unbelievable": 3, "mind-blowing": 3, "revolutionary": 3,
    "game-changer": 3, "life-changing": 3, "extraordinary": 3, "phenomenal": 3, "stunning": 3,
    "terrible": -3, "horrible": -3, "awful": -3, "disaster": -3, "catastrophic": -3,
    "devastating": -3, "nightmare": -3, "scam": -3, "fraud": -3, "ripoff": -3,
    "great": 2, "awesome": 2, "fantastic": 2, "excellent": 2, "perfect": 2,
    "brilliant": 2, "outstanding": 2, "remarkable": 2, "superb": 2, "wonderful": 2,
    "bad": -2, "poor": -2, "wrong": -2, "stupid": -2, "dumb": -2,
    "dangerous": -2, "concerning": -2, "worrying": -2, "troubling": -2,
    "good": 1, "nice": 1, "cool": 1, "solid": 1, "decent": 1, "fine": 1,
    "exciting": 2, "thrilling": 2, "powerful": 2, "massive": 2, "huge": 2,
    "fail": -2, "failure": -2, "crash": -2, "collapse": -2, "bubble": -2,
    "fear": -2, "panic": -2, "greed": -1, "hype": -1,
    "love": 2, "hate": -2, "best": 2, "worst": -2,
    "boom": 2, "bust": -2, "soar": 2, "plunge": -2, "surge": 2, "crater": -2,
}


def track_emotional_language(
    text: str,
    sentences: list[TimestampedSentence] | None = None,
) -> dict:
    """Track emotional language throughout the transcript.

    Detects spikes in emotional language — manipulation points where
    the speaker uses strong words to sway the audience.

    Returns:
    - overall_sentiment: -3 to +3
    - emotional_moments: list of {word, score, timestamp, sentence}
    - positive_count, negative_count
    - emotional_density: emotional words per 100 words
    - spikes: list of timestamps where emotional intensity peaks
    """
    if sentences is None:
        sents = split_sentences(text)
        sentences = [TimestampedSentence(text=s, start=0.0, end=0.0, segment_index=i) for i, s in enumerate(sents)]

    emotional_moments: list[dict] = []
    positive_count = 0
    negative_count = 0
    total_score = 0
    word_count = len(text.split())

    for ts in sentences:
        words = re.findall(r"\b[\w-]+\b", ts.text.lower())
        for word in words:
            if word in _EMOTIONAL_WORDS:
                score = _EMOTIONAL_WORDS[word]
                total_score += score
                if score > 0:
                    positive_count += 1
                else:
                    negative_count += 1
                emotional_moments.append({
                    "word": word,
                    "score": score,
                    "timestamp": ts.timestamp_str,
                    "start_seconds": ts.start,
                    "sentence": ts.text,
                })

    # Detect spikes: timestamps with 2+ emotional words within 10 seconds
    spikes: list[dict] = []
    if emotional_moments:
        # Group by approximate time (10-second windows)
        time_groups: dict[int, list[dict]] = {}
        for m in emotional_moments:
            window = int(m["start_seconds"] // 10)
            time_groups.setdefault(window, []).append(m)

        for window, moments in time_groups.items():
            if len(moments) >= 2:
                total_intensity = sum(abs(m["score"]) for m in moments)
                spikes.append({
                    "timestamp": moments[0]["timestamp"],
                    "start_seconds": moments[0]["start_seconds"],
                    "intensity": total_intensity,
                    "words": [m["word"] for m in moments],
                    "sentence": moments[0]["sentence"],
                })

    # Sort spikes by intensity
    spikes.sort(key=lambda s: s["intensity"], reverse=True)

    overall = total_score / max(len(emotional_moments), 1)
    density = (len(emotional_moments) / max(word_count, 1)) * 100

    return {
        "overall_sentiment": round(overall, 2),
        "positive_count": positive_count,
        "negative_count": negative_count,
        "emotional_density": round(density, 2),
        "emotional_moments": emotional_moments[:50],  # cap for size
        "spikes": spikes[:10],  # top 10 spikes
        "spike_count": len(spikes),
    }


# === 10. DEFINITION EXTRACTION ============================================

_RE_DEFINITION_IS = re.compile(
    r"\b([A-Z][a-zA-Z]{1,}(?:\s+[a-z]+){0,4})\s+is\s+(?:a|an|the)\s+(.{10,150}?)(?:[.!?]|\n)",
)
_RE_DEFINITION_CALLED = re.compile(
    r"\b(?:is\s+called|known\s+as|referred\s+to\s+as)\s+(.{3,80}?)(?:[.!?]|\n)",
    re.I,
)
_RE_DEFINITION_MEANS = re.compile(
    r"\b([A-Z][a-zA-Z]{1,}(?:\s+[a-z]+){0,3})\s+(?:means|refers\s+to|is\s+defined\s+as)\s+(.{10,150}?)(?:[.!?]|\n)",
)
_RE_DEFINITION_TERM = re.compile(
    r"\b(?:what\s+(?:is|are)\s+|what.s\s+)"
    r"([A-Z][a-zA-Z]{1,}(?:\s+[a-z]+){0,4})\?"
    r"\s*(.{10,200}?)(?:[.!?]|\n)",
)


def extract_definitions(text: str) -> list[dict]:
    """Extract term definitions from the transcript.

    Catches:
    - "RSI is a momentum oscillator..."
    - "Bollinger Bands are volatility bands..."
    - "This is called mean reversion."
    - "MACD means Moving Average Convergence Divergence."

    Returns list of {term, definition, sentence} dicts.
    """
    definitions: list[dict] = []
    seen_terms: set[str] = set()

    # Definitions require real sentence punctuation to work properly
    # (auto-captions with fake periods produce too many false positives)
    punct_count = text.count(".") + text.count("?") + text.count("!")
    word_count = len(text.split())
    if word_count > 0:
        period_ratio = punct_count / word_count
    else:
        period_ratio = 0
    # Real punctuation: ~1 period per 15-20 words. Fake: ~1 per 5-8.
    # Only apply ratio check for longer texts (transcripts, not test sentences)
    # Note: auto-captioned videos rebuilt from VTT segments have ~1 period
    # per 8-12 words (shorter sentences), so use a more lenient threshold
    if word_count > 100:
        has_real_punctuation = punct_count > 10 and period_ratio < 0.15
    else:
        has_real_punctuation = punct_count > 0
    if not has_real_punctuation:
        return []

    # Pattern 1: "X is a/an/the Y"
    for m in _RE_DEFINITION_IS.finditer(text):
        term = m.group(1).strip()
        definition = m.group(2).strip().rstrip(".")
        term_lower = term.lower()

        # Filter out non-terms (pronouns, common words, phrases)
        _NON_TERMS = {"this", "that", "it", "here", "there", "what", "when", "why", "how",
                      "look at what", "words what", "works so this", "let's", "now",
                      "so", "but", "and", "or", "the", "a", "an",
                      "question what", "fast", "mement right here", "right here",
                      "trends", "time and this", "the first one", "the second one",
                      "the last one", "the next one"}
        if term_lower in _NON_TERMS:
            continue
        # Skip if term starts with a common non-term word
        if any(term_lower.startswith(w + " ") for w in {"look", "let", "now", "so", "but", "and", "or", "what", "words", "works", "this", "that", "here", "there", "when", "why", "how", "good", "okay", "well",
                                                          "question", "fast", "mement", "right", "trends", "time", "the"}):
            continue
        # Term must start with an uppercase letter that's a real term
        # (not a fragment from auto-caption segment boundary)
        if len(term) < 3 or len(definition) < 10:
            continue
        # Definition must contain actual content words (not just fragments)
        def_words = definition.split()
        if len(def_words) < 3:
            continue

        if term_lower in seen_terms:
            continue
        seen_terms.add(term_lower)

        definitions.append({
            "term": term,
            "definition": definition,
            "sentence": _extract_sentence(text, m.start(), m.end()),
        })

    # Pattern 2: "is called X" / "known as X"
    for m in _RE_DEFINITION_CALLED.finditer(text):
        term = m.group(1).strip().rstrip(".")
        if len(term) < 3:
            continue

        # Find what's being defined (look backwards)
        before = text[max(0, m.start()-80):m.start()].strip()
        # Find the subject before "is called"
        sent_start = text.rfind(".", 0, m.start())
        subject = text[sent_start+1:m.start()].strip() if sent_start != -1 else before

        term_lower = term.lower()
        if term_lower in seen_terms:
            continue
        seen_terms.add(term_lower)

        definitions.append({
            "term": term,
            "definition": subject,
            "sentence": _extract_sentence(text, m.start(), m.end()),
        })

    # Pattern 3: "X means Y" / "X refers to Y"
    for m in _RE_DEFINITION_MEANS.finditer(text):
        term = m.group(1).strip()
        definition = m.group(2).strip().rstrip(".")
        term_lower = term.lower()

        if term_lower in {"this", "that", "it", "here", "there"}:
            continue
        if len(term) < 3 or len(definition) < 10:
            continue

        if term_lower in seen_terms:
            continue
        seen_terms.add(term_lower)

        definitions.append({
            "term": term,
            "definition": definition,
            "sentence": text[max(0, m.start()-5):min(len(text), m.end()+5)].strip(),
        })

    # Pattern 4: "What is X? X is..."
    for m in _RE_DEFINITION_TERM.finditer(text):
        term = m.group(1).strip()
        definition = m.group(2).strip().rstrip(".")
        term_lower = term.lower()

        if term_lower in seen_terms:
            continue
        seen_terms.add(term_lower)

        definitions.append({
            "term": term,
            "definition": definition,
            "sentence": text[m.start():min(len(text), m.end()+5)].strip(),
        })

    return definitions


# === MASTER EXTRACTION FUNCTION ===========================================

def extract_all(text: str, sentences: list[TimestampedSentence] | None = None) -> dict:
    """Run all 10 advanced extraction features on the text.

    This is the comprehensive extraction that feeds into the Deep Research package.

    Returns a dict with all extraction results:
    - contradictions
    - marketing_pressure
    - parameters
    - rules
    - citations
    - questions
    - hedge_density
    - chapters
    - emotional_language
    - definitions
    """
    return {
        "contradictions": detect_contradictions(
            extract_claims_enriched(text, sentences)
        ),
        "marketing_pressure": detect_marketing_pressure(text),
        "parameters": extract_parameters(text),
        "rules": extract_rules(text),
        "citations": extract_citations(text),
        "questions": extract_questions(text),
        "hedge_density": calculate_hedge_density(text),
        "chapters": segment_topics(text, sentences),
        "emotional_language": track_emotional_language(text, sentences),
        "definitions": extract_definitions(text),
    }


# ---------------------------------------------------------------- CLAIM EXTRACTION (ORIGINAL)

# Regex patterns for pre-identifying factual claims in transcript text.
# These give Devin a starting list — Devin will add more during reading.

# Statistical claims: percentages, ratios, proportions
_RE_PERCENT = re.compile(r"\b\d+(?:\.\d+)?\s*%\b", re.I)
# Win rate / success rate claims
_RE_WIN_RATE = re.compile(r"\b(?:win\s*rate|success\s*rate|accuracy|hit\s*rate)\b", re.I)
# Dollar amounts / financial figures
_RE_DOLLAR = re.compile(r"\$[\d,]+(?:\.\d+)?(?:\s*(?:million|billion|trillion|k|thousand))?|\b\d+\s*(?:million|billion|trillion)\s+dollars?\b", re.I)
# Causal claims: X causes/leads to/results in/prevents/cures Y
_RE_CAUSAL = re.compile(
    r"\b(?:causes?|leads?\s+to|results?\s+in|prevents?|cures?|fixes?|solves?|"
    r"creates?|produces?|generates?|results?\s+from|stems?\s+from|"
    r"is\s+(?:the\s+)?(?:cause|reason|result)\s+of|"
    r"because\s+of|due\s+to|thanks\s+to|owing\s+to)\b",
    re.I,
)
# Authority appeals: "studies show", "research proves", "scientists say"
_RE_AUTHORITY = re.compile(
    r"\b(?:studies?\s+show|research\s+(?:shows?|proves?|indicates?)|"
    r"scientists?\s+say|experts?\s+say|doctors?\s+say|"
    r"according\s+to\s+(?:research|studies?|science)|"
    r"it\s+has\s+been\s+proven|proven\s+to|"
    r"backed\s+by\s+(?:science|research|studies)|"
    r"peer[- ]reviewed|clinical\s+trials?)\b",
    re.I,
)
# Superlatives / absolute claims: "the best", "always works", "never fails"
_RE_SUPERLATIVE = re.compile(
    r"\b(?:the\s+(?:best|worst|only|most|least|biggest|smallest|greatest|"
    r"fastest|easiest|hardest|most\s+powerful|most\s+effective)|"
    r"always\s+works?|never\s+fails?|guaranteed|"
    r"100\s*%\s*(?:effective|successful|accurate|certain|sure)|"
    r"everyone\s+(?:should|must|needs?\s+to)|"
    r"nobody\s+(?:can|should|does)|"
    r"impossible\s+to\s+(?:fail|lose)|"
    r"foolproof|infallible|perfect)\b",
    re.I,
)
# Scientific / medical claims
_RE_SCIENTIFIC = re.compile(
    r"\b(?:brain\s+(?:waves?|chemistry|plasticity|activity)|"
    r"neurotransmitters?|dopamine|serotonin|cortisol|"
    r"subconscious\s+mind|cognitive\s+(?:bias|dissonance)|"
    r"placebo\s+effect|epigenetics|quantum\s+(?:physics|mechanics|field)|"
    r"frequency|vibration|energy\s+field|"
    r"DNA|genes?|cells?|immune\s+system|"
    r"inflammation|metabolism|hormones?)\b",
    re.I,
)
# Comparative claims: "better than", "more effective than", "superior to"
_RE_COMPARATIVE = re.compile(
    r"\b(?:better\s+than|worse\s+than|more\s+effective\s+than|"
    r"superior\s+to|inferior\s+to|outperforms?|"
    r"unlike\s+(?:other|most|all)|"
    r"the\s+only\s+(?:one|method|approach|way))\b",
    re.I,
)
# Historical / factual claims with dates
_RE_HISTORICAL = re.compile(
    r"\b(?:in\s+(?:the\s+)?(?:18|19|20|21)\d{2}s?|"
    r"since\s+(?:the\s+)?(?:18|19|20|21)\d{2}|"
    r"(?:discovered|invented|founded|created|established)\s+in\s+\d{4}|"
    r"(?:ancient|medieval|classical)\s+(?:greeks?|romans?|egyptians?|chinese?|indians?|civilization))\b",
    re.I,
)
# Prediction claims: "will go up", "going to crash", "expected to"
_RE_PREDICTION = re.compile(
    r"\b(?:will\s+(?:go\s+(?:up|down|higher|lower)|rise|fall|crash|drop|surge|climb|plunge|collapse)|"
    r"going\s+to\s+(?:go\s+(?:up|down)|rise|fall|crash|drop|surge|climb|plunge|collapse)|"
    r"expected\s+to\s+(?:rise|fall|increase|decrease|grow|shrink|crash|drop)|"
    r"predicted\s+to\s+(?:rise|fall|increase|decrease|grow|shrink)|"
    r"(?:bullish|bearish)\s+(?:on|outlook|forecast|prediction)|"
    r"(?:price|market|stock)\s+(?:will|is\s+going\s+to|should)\s+(?:rise|fall|go\s+(?:up|down))|"
    r"(?:next|coming)\s+(?:year|month|week|decade).{0,30}(?:rise|fall|crash|drop|surge|boom|bust))\b",
    re.I,
)
# Numerical assertions: "studies show X%", "research indicates", "data suggests"
_RE_NUMERICAL_ASSERT = re.compile(
    r"\b(?:studies\s+show|research\s+(?:shows|indicates|suggests|demonstrates)|"
    r"data\s+(?:shows|suggests|indicates)|"
    r"evidence\s+(?:shows|suggests)|"
    r"results\s+show|findings\s+(?:show|indicate|suggest)|"
    r"experiments?\s+show|tests?\s+show|"
    r"it\s+has\s+been\s+(?:shown|proven|demonstrated)|"
    r"proven\s+to\s+(?:work|increase|decrease|improve|reduce|boost|enhance)|"
    r"demonstrated\s+to\s+(?:work|increase|decrease|improve))\b",
    re.I,
)
# Definition-as-fact: "X is the best Y", "X is the most Y"
_RE_DEFINITION_FACT = re.compile(
    r"\b(?:is\s+the\s+(?:best|worst|most|least|fastest|slowest|biggest|smallest|"
    r"most\s+(?:important|effective|powerful|reliable|accurate|dangerous)|"
    r"key\s+to|secret\s+to|reason\s+(?:for|why)))\b",
    re.I,
)


def _extract_sentence(text: str, match_start: int, match_end: int) -> str:
    """Extract the full sentence containing a regex match."""
    # Find sentence start (previous . ! ? or start of text/paragraph)
    sent_start = match_start
    for i in range(match_start - 1, -1, -1):
        if text[i] in ".!?\n":
            sent_start = i + 1
            break
    else:
        sent_start = 0
    # Find sentence end (next . ! ? or end of text/paragraph)
    sent_end = match_end
    for i in range(match_end, len(text)):
        if text[i] in ".!?\n":
            sent_end = i + 1
            break
    else:
        sent_end = len(text)
    return text[sent_start:sent_end].strip()


def extract_claims(text: str) -> list[dict]:
    """Pre-identify factual claims in transcript text using regex patterns.

    Returns a list of claim dicts:
    - claim_type: statistical|win_rate|financial|causal|authority|superlative|scientific|comparative|historical
    - matched_pattern: the regex match
    - sentence: the full sentence containing the match
    - char_offset: position in the text

    NOTE: This is a starting list for Devin. Devin will add claims it
    identifies during reading that regex cannot catch (implicit claims,
    contextual claims, nuanced assertions).
    """
    claims: list[dict] = []
    patterns = [
        ("win_rate", _RE_WIN_RATE),
        ("statistical", _RE_PERCENT),
        ("financial", _RE_DOLLAR),
        ("authority", _RE_AUTHORITY),
        ("superlative", _RE_SUPERLATIVE),
        ("causal", _RE_CAUSAL),
        ("scientific", _RE_SCIENTIFIC),
        ("comparative", _RE_COMPARATIVE),
        ("historical", _RE_HISTORICAL),
        ("prediction", _RE_PREDICTION),
        ("numerical_assertion", _RE_NUMERICAL_ASSERT),
        ("definition_fact", _RE_DEFINITION_FACT),
    ]

    seen_sentences: set[str] = set()

    for claim_type, pattern in patterns:
        for match in pattern.finditer(text):
            sentence = _extract_sentence(text, match.start(), match.end())
            if not sentence or len(sentence) < 10:
                continue
            # Deduplicate by sentence (a sentence might match multiple patterns)
            key = sentence.lower()[:100]
            if key in seen_sentences:
                # Add the claim type to the existing entry
                for c in claims:
                    if c["sentence"] == sentence:
                        if claim_type not in c["claim_types"]:
                            c["claim_types"].append(claim_type)
                        break
                continue
            seen_sentences.add(key)
            claims.append({
                "claim_types": [claim_type],
                "matched_pattern": match.group(),
                "sentence": sentence,
                "char_offset": match.start(),
            })

    # Sort by position in text
    claims.sort(key=lambda c: c["char_offset"])
    return claims


# ---------------------------------------------------------------- SOURCE EXTRACTION

# URL pattern (http/https)
_RE_URL = re.compile(r"https?://[^\s<>\"]+")
# Book title patterns: "Title" by Author, the book "Title"
_RE_BOOK = re.compile(
    r'(?:the\s+book\s+)?"([A-Z][^"]{3,80})"\s+by\s+([A-Z][a-zA-Z\s]+)'
    r'|"([A-Z][^"]{3,80})"\s+(?:is\s+)?(?:a\s+)?(?:book|bestseller|novel)',
    re.I,
)
# Paper/article references: "et al", "Journal of", "DOI"
_RE_PAPER = re.compile(
    r'\b(?:et\s+al\.?|Journal\s+of\s+[A-Z][a-z]+|doi:?\s*10\.\d{4}|'
    r'arXiv:?\d{4}|Nature|Science|Lancet|JAMA|BMJ)\b',
    re.I,
)
# Person names after "according to", "says", "by"
_RE_PERSON = re.compile(
    r'(?i:according\s+to|said\s+by|stated\s+by|\bby\b)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})',
)


def extract_sources(text: str, description: str = "") -> dict:
    """Extract mentioned sources from transcript text and video description.

    Returns a dict with:
    - urls: list of URL strings
    - books: list of {title, author} dicts
    - papers: list of reference strings
    - people: list of person name strings
    """
    combined = text + "\n" + description

    urls = list(set(_RE_URL.findall(combined)))

    books: list[dict] = []
    for m in _RE_BOOK.finditer(combined):
        title = m.group(1) or m.group(3)
        author = m.group(2) or ""
        if title:
            books.append({"title": title.strip(), "author": author.strip()})

    papers = list(set(_RE_PAPER.findall(combined)))

    people = list(set(_RE_PERSON.findall(combined)))

    return {
        "urls": urls,
        "books": books,
        "papers": papers,
        "people": people,
    }


# ---------------------------------------------------------------- DEEP RESEARCH PREP

def prepare_deep_research(
    url_or_id: str,
    output_dir: Path | None = None,
    cookies_file: str | None = None,
    cookies_from_browser: str | None = None,
    proxy: str | None = None,
    retries: int = DEFAULT_RETRIES,
    enable_visual: bool = False,
) -> dict:
    """Fetch transcript, clean it, extract claims + sources, and prepare
    a comprehensive research package for Deep Research analysis.

    This is Step 1 of the Deep Research workflow:
    1. This function: fetch + clean + extract claims + extract sources → research package
    2. Devin: reads the package, performs web research, verifies claims,
       maps arguments, assesses bias, cross-references, produces deep research brief

    Quality over speed — no caps on claims. Every significant claim gets verified.

    If enable_visual is True, also downloads the video and extracts visual
    content (on-screen text, chart patterns) using OCR (Tesseract, free, local)
    as the primary method, with optional LLM analysis for chart patterns.
    """
    # Step 1: Fetch transcript
    print(f"[1/3] Fetching transcript for: {url_or_id}", file=sys.stderr)
    result = extract_transcript(
        url_or_id, output_dir=output_dir,
        cookies_file=cookies_file, cookies_from_browser=cookies_from_browser,
        proxy=proxy, retries=retries,
    )
    if result.error or not result.has_transcript:
        return {
            "ok": False,
            "error": result.error or "No transcript available",
            "error_type": result.error_type or "no_transcript",
            "video": result.to_dict(),
        }

    # Step 2: Clean transcript
    print(f"[2/4] Cleaning transcript + extracting claims...", file=sys.stderr)
    raw_path = Path(result.transcript_path)
    raw_text = raw_path.read_text(encoding="utf-8", errors="replace")
    header_end = raw_text.find("=" * 20)
    raw_body = raw_text[header_end:].lstrip("=\n").strip() if header_end != -1 else raw_text

    cleaned_body = clean_transcript_text(raw_body)
    raw_chars = len(raw_body)
    cleaned_chars = len(cleaned_body)
    reduction = ((raw_chars - cleaned_chars) / raw_chars * 100) if raw_chars > 0 else 0

    # Save cleaned transcript
    out = output_dir or OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    # Get segments from VideoInfo (stored during extract_transcript)
    print(f"[3/4] Extracting entities and enriched claims...", file=sys.stderr)
    segments: list[TranscriptSegment] = result.segments if hasattr(result, 'segments') else []

    # If text has no punctuation (common in auto-captions), restore it.
    period_count = raw_body.count(".") + raw_body.count("?") + raw_body.count("!")
    auto_captioned = False  # track whether we restored punctuation
    used_llm_punctuation = False
    if segments and period_count < 5 and len(raw_body) > 1000:
        # Restore punctuation using word-count heuristic (no API needed)
        # LLM enhancement is optional and only used if use_llm=True
        restored, used_llm_punctuation = restore_punctuation(raw_body, segments)
        if restored != raw_body:
            raw_body = restored
            cleaned_body = clean_transcript_text(raw_body)
            auto_captioned = True

    # extractor_body is the properly-punctuated text for sentence-dependent
    # extractors (definitions, rules, questions). For auto-captioned videos
    # we already restored punctuation above, so it matches cleaned_body.
    # For human-captioned videos, cleaned_body already has real punctuation.
    extractor_body = cleaned_body

    cleaned_path = _write_cleaned_transcript(result, cleaned_body, out)

    # Convert segments to timestamped sentences
    ts_sentences: list[TimestampedSentence] = []
    if segments:
        ts_sentences = segments_to_sentences(segments)

    # Fetch metadata for description (VideoInfo doesn't carry it)
    description = ""
    try:
        meta = extract_metadata(url_or_id, cookies_file=cookies_file,
                                cookies_from_browser=cookies_from_browser,
                                proxy=proxy, retries=retries)
        description = meta.get("description", "") if isinstance(meta, dict) else ""
    except Exception:
        pass

    # Extract enriched claims (with subject, negation, strength, timestamps)
    claims = extract_claims_enriched(cleaned_body, ts_sentences)

    # Extract entities (people, orgs, tools, metrics, concepts)
    # Use extractor_body (with real sentence boundaries) when available
    entities = extract_entities(extractor_body, trust_punctuation=not auto_captioned)

    # Extract sources from cleaned transcript + description
    sources = extract_sources(extractor_body, description)

    # Extract advanced content (contradictions, marketing, rules, etc.)
    advanced = extract_all(extractor_body, ts_sentences or None)

    # Extract visual content (frames + OCR + optional LLM analysis)
    visual_content: dict[str, Any] = {
        "enabled": enable_visual,
        "ocr_available": False,
        "llm_provider": "",
        "video_downloaded": False,
        "frames_extracted": 0,
        "frames_after_dedup": 0,
        "frames_analyzed_ocr": 0,
        "frames_analyzed_llm": 0,
        "visual_claims": [],
        "on_screen_texts": [],
        "chart_patterns": [],
        "frame_analyses": [],
    }
    if enable_visual:
        print(f"[4/4] Extracting visual content from video frames...", file=sys.stderr)
        video_id = parse_video_id(url_or_id)
        # Use chapter boundaries for smart frame sampling
        chapter_ts: list[float] | None = None
        chapters = advanced.get("chapters", [])
        if chapters:
            chapter_ts = [c.get("start", 0) for c in chapters if c.get("start", 0) > 0]
        visual_content = extract_visual_content(
            video_id, out, duration=result.duration,
            cookies_file=cookies_file, cookies_from_browser=cookies_from_browser,
            proxy=proxy, enable_visual=enable_visual,
            chapter_timestamps=chapter_ts,
        )

    # Build the research package
    package = {
        "ok": True,
        "mode": "deep_research",
        "video": {
            "id": result.id,
            "title": result.title,
            "channel": result.channel,
            "url": result.url,
            "duration": result.duration,
            "upload_date": _normalize_upload_date(result.upload_date),
            "view_count": result.view_count,
            "like_count": result.like_count,
            "description": description,
        },
        "transcript": {
            "raw_path": str(raw_path),
            "cleaned_path": cleaned_path,
            "raw_chars": raw_chars,
            "cleaned_chars": cleaned_chars,
            "reduction_pct": round(reduction, 1),
            "paragraph_count": len(cleaned_body.split("\n\n")),
            "has_timestamps": len(segments) > 0,
            "segment_count": len(segments),
            "auto_captioned": auto_captioned,
            "punctuation_restored": auto_captioned,
            "punctuation_method": "llm" if used_llm_punctuation else ("timing" if auto_captioned else "original"),
        },
        "extracted_claims": claims,
        "extracted_entities": entities,
        "extracted_sources": sources,
        "claim_count": len(claims),
        "entity_count": sum(len(v) if isinstance(v, list) else 0 for v in entities.values()),
        "source_count": sum(len(v) if isinstance(v, list) else 0 for v in sources.values()),
        "timestamped_sentences": [s.to_dict() for s in ts_sentences] if ts_sentences else [],
        "advanced_extraction": advanced,
        "visual_extraction": visual_content,
        "schema": DEEP_RESEARCH_SCHEMA,
        "instructions": (
            "DEEP RESEARCH WORKFLOW — QUALITY OVER SPEED\n\n"
            "This is a comprehensive research brief, not a summary. Take as long as needed.\n\n"
            "Phase 2 — Devin executes the following:\n\n"
            "1. READ the cleaned transcript thoroughly.\n"
            "2. REVIEW the extracted_claims list. Each claim now includes:\n"
            "   - subject: what the claim is about (when extractable)\n"
            "   - negated: whether the claim is negated (NOT X causes Y)\n"
            "   - strength: high | moderate | low | opinion\n"
            "   - timestamp: when in the video this claim was made (if available)\n"
            "   ADD any claims you identify that regex missed (implicit claims, contextual claims).\n"
            "3. REVIEW the extracted_entities: people, organizations, tools, metrics, and concepts.\n"
            "   Use these to guide your research — verify the people's credentials, the tools' claims, etc.\n"
            "3. PRIORITIZE claims for verification. Verify EVERY significant claim — do not cap the number. "
            "A claim is significant if it is: central to the argument, potentially harmful if false, "
            "verifiable, or controversial.\n"
            "4. VERIFY each priority claim using web_search and webfetch. Find at least 2 independent sources "
            "per claim when possible. Assess each source's reliability.\n"
            "5. MAP the argument structure: main thesis, premises, conclusions, reasoning quality.\n"
            "6. IDENTIFY logical fallacies (ad hominem, strawman, appeal to authority, cherry-picking, "
            "survivorship bias, confirmation bias, false causation, etc.).\n"
            "7. ASSESS bias: investigate the speaker/channel's credibility, financial incentives, "
            "conflicts of interest, and ideological position.\n"
            "8. CROSS-REFERENCE: compare the video's claims with authoritative sources on the same topic.\n"
            "9. IDENTIFY OMISSIONS: what's NOT said that should be? What counterarguments are ignored?\n"
            "10. PRODUCE the deep research brief as JSON following the schema.\n"
            "11. SAVE as <video_id>_deep_research.json in the same directory.\n\n"
            "Source reliability rubric:\n"
            "  Tier 1 (high): Peer-reviewed papers, government statistics, established news (Reuters, AP, BBC)\n"
            "  Tier 2 (moderate): Industry publications, textbooks, recognized experts, established blogs\n"
            "  Tier 3 (low): YouTube videos, Reddit, personal blogs, marketing content\n\n"
            "Verdict definitions:\n"
            "  verified: Multiple reliable sources confirm the claim\n"
            "  partially-verified: Sources confirm part of the claim but not all\n"
            "  unverified: No reliable sources found to confirm or deny\n"
            "  contradicted: Reliable sources contradict the claim\n"
            "  unverifiable: The claim cannot be tested or is subjective\n"
            "  opinion: The claim is a stated opinion, not a factual assertion\n"
        ),
    }

    # Save the research package
    package_path = out / f"{result.id}_research_package.json"
    with open(package_path, "w", encoding="utf-8") as f:
        json.dump(package, f, ensure_ascii=False, indent=2)
    package["research_package_path"] = str(package_path)

    print(f"  Claims extracted:  {len(claims)}", file=sys.stderr)
    print(f"  Sources extracted: {package['source_count']}", file=sys.stderr)
    print(f"  Package saved:     {package_path}", file=sys.stderr)

    return package


# ---------------------------------------------------------------- VISUAL EXTRACTION

# Frame extraction interval (seconds). One frame every N seconds.
_FRAME_INTERVAL = 60
# Max frames to extract (cap for very long videos).
_MAX_FRAMES = 60
# Video download format — low quality, no audio, smallest possible.
_VIDEO_FORMAT = "worst[ext=mp4]/worst"
# Frames directory name within output dir.
_FRAMES_SUBDIR = "_frames"
# Tesseract binary path (auto-detected, fallback to PATH).
_TESSERACT_CMD = ""
# Minimum OCR confidence to keep a word (0-100).
_OCR_MIN_CONFIDENCE = 50
# Minimum OCR text length to consider a frame "has content".
_OCR_MIN_TEXT_LEN = 5
# Frame deduplication: skip if image hash matches previous (perceptual diff threshold).
_FRAME_DEDUP_THRESHOLD = 0.95


def _find_tesseract() -> str:
    """Find the Tesseract binary path. Caches result."""
    global _TESSERACT_CMD
    if _TESSERACT_CMD:
        return _TESSERACT_CMD
    # Check common Windows install locations
    candidates = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ]
    for c in candidates:
        if Path(c).exists():
            _TESSERACT_CMD = c
            return c
    # Check PATH
    found = shutil.which("tesseract")
    if found:
        _TESSERACT_CMD = found
        return found
    return ""


def _download_video_low_quality(
    video_id: str,
    output_dir: Path,
    cookies_file: str | None = None,
    cookies_from_browser: str | None = None,
    proxy: str | None = None,
) -> Path | None:
    """Download video in lowest quality for frame extraction.

    Returns path to downloaded video file, or None on failure.
    """
    video_path = output_dir / f"{video_id}_video.mp4"
    if video_path.exists() and video_path.stat().st_size > 1000:
        return video_path  # already downloaded

    opts: dict[str, Any] = {
        "format": _VIDEO_FORMAT,
        "outtmpl": str(video_path),
        "quiet": True,
        "no_warnings": True,
        "skip_download": False,
    }
    if cookies_file:
        opts["cookiefile"] = cookies_file
    if cookies_from_browser:
        opts["cookiesfrombrowser"] = (cookies_from_browser,)
    if proxy:
        opts["proxy"] = proxy

    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([video_id])
        if video_path.exists() and video_path.stat().st_size > 1000:
            return video_path
    except Exception as e:
        print(f"  [visual] Video download failed: {e}", file=sys.stderr)
    return None


def _extract_frames(
    video_path: Path,
    output_dir: Path,
    interval: int = _FRAME_INTERVAL,
    max_frames: int = _MAX_FRAMES,
    timestamps: list[float] | None = None,
) -> list[Path]:
    """Extract frames from video at regular intervals or specific timestamps.

    Args:
        video_path: Path to the video file.
        output_dir: Directory for the _frames subdirectory.
        interval: Seconds between frames (ignored if timestamps given).
        max_frames: Maximum number of frames to extract.
        timestamps: Optional list of specific timestamps (seconds) to extract.
                     If provided, overrides interval-based extraction.

    Returns list of frame file paths (PNG).
    """
    frames_dir = output_dir / _FRAMES_SUBDIR
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Clear old frames
    for old in frames_dir.glob("*.png"):
        old.unlink()

    frames: list[Path] = []
    try:
        import subprocess

        if timestamps:
            # Extract at specific timestamps (e.g., chapter boundaries)
            ts_list = timestamps[:max_frames]
        else:
            # Get video duration for interval-based extraction
            probe = subprocess.run(
                ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
                capture_output=True, text=True, timeout=30,
            )
            duration = float(probe.stdout.strip()) if probe.stdout.strip() else 0
            if duration <= 0:
                return []
            num_frames = min(max_frames, int(duration / interval))
            if num_frames < 1:
                num_frames = 1
            ts_list = [i * interval for i in range(num_frames)]

        for i, timestamp in enumerate(ts_list):
            frame_path = frames_dir / f"frame_{i:04d}_{int(timestamp//60):02d}m{int(timestamp%60):02d}s.png"
            subprocess.run(
                ["ffmpeg", "-ss", str(timestamp), "-i", str(video_path),
                 "-frames:v", "1", "-q:v", "2", "-y", str(frame_path)],
                capture_output=True, timeout=30,
            )
            if frame_path.exists() and frame_path.stat().st_size > 1000:
                frames.append(frame_path)

    except Exception as e:
        print(f"  [visual] Frame extraction failed: {e}", file=sys.stderr)

    return frames


def _extract_ocr_from_frame(frame_path: Path) -> dict:
    """Extract text from a video frame using Tesseract OCR.

    This is the primary visual extraction method — free, local, no API needed.
    Returns dict with:
        - text: Full OCR text
        - words: List of (word, confidence) tuples
        - lines: List of text lines
        - has_content: bool, whether meaningful text was found
    """
    tess_path = _find_tesseract()
    if not tess_path:
        return {"text": "", "words": [], "lines": [], "has_content": False,
                "error": "Tesseract not found"}

    try:
        import pytesseract
        from PIL import Image
        pytesseract.pytesseract.tesseract_cmd = tess_path

        img = Image.open(frame_path)

        # Full text extraction
        full_text = pytesseract.image_to_string(img)

        # Word-level data with confidence scores
        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)

        words: list[tuple[str, int]] = []
        for w, conf in zip(data["text"], data["conf"]):
            w = w.strip()
            if w and conf != "-1":
                try:
                    c = int(conf)
                    if c >= _OCR_MIN_CONFIDENCE:
                        words.append((w, c))
                except (ValueError, TypeError):
                    pass

        # Lines (grouped by line number)
        lines_dict: dict[int, list[str]] = {}
        for i, line_num in enumerate(data["line_num"]):
            if line_num == 0:
                continue
            w = data["text"][i].strip()
            if w:
                lines_dict.setdefault(line_num, []).append(w)
        lines = [" ".join(ws) for ws in lines_dict.values()]

        # Clean text
        clean_text = full_text.strip()
        # Remove excessive whitespace
        clean_text = re.sub(r"\n{3,}", "\n\n", clean_text)

        has_content = len(clean_text) >= _OCR_MIN_TEXT_LEN

        return {
            "text": clean_text,
            "words": words,
            "lines": lines,
            "has_content": has_content,
        }

    except Exception as e:
        return {"text": "", "words": [], "lines": [], "has_content": False,
                "error": str(e)}


def _frames_are_similar(frame1: Path, frame2: Path) -> bool:
    """Check if two frames are visually similar (for deduplication).

    Uses a simple histogram comparison. Returns True if frames are
    nearly identical (e.g., same slide with minor changes).
    """
    try:
        from PIL import Image
        import struct

        img1 = Image.open(frame1).convert("L").resize((64, 64))
        img2 = Image.open(frame2).convert("L").resize((64, 64))

        # Simple pixel difference
        pixels1 = list(img1.getpixel((x, y)) for y in range(64) for x in range(64))
        pixels2 = list(img2.getpixel((x, y)) for y in range(64) for x in range(64))

        if len(pixels1) != len(pixels2):
            return False

        diff = sum(abs(a - b) for a, b in zip(pixels1, pixels2))
        max_diff = 255 * len(pixels1)
        similarity = 1 - (diff / max_diff)
        return similarity >= _FRAME_DEDUP_THRESHOLD

    except Exception:
        return False


def _deduplicate_frames(frames: list[Path]) -> list[Path]:
    """Remove consecutive duplicate frames. Keeps first occurrence.

    Returns filtered list of frame paths.
    """
    if len(frames) <= 1:
        return frames

    result: list[Path] = [frames[0]]
    for i in range(1, len(frames)):
        if not _frames_are_similar(frames[i], frames[i - 1]):
            result.append(frames[i])
        else:
            # Remove the duplicate frame file
            try:
                frames[i].unlink()
            except Exception:
                pass
    return result


# --- Pluggable LLM backend for chart pattern analysis ---

def _llm_analyze_frame_openai(
    frame_path: Path,
    timestamp: float,
    api_key: str,
    base_url: str | None = None,
) -> dict | None:
    """Analyze frame using OpenAI-compatible vision API (OpenAI, 9router, etc.)."""
    try:
        import base64
        from openai import OpenAI

        client_kwargs: dict[str, Any] = {"api_key": api_key, "timeout": 60}
        if base_url:
            client_kwargs["base_url"] = base_url
        client = OpenAI(**client_kwargs)

        with open(frame_path, "rb") as f:
            img_data = base64.b64encode(f.read()).decode("utf-8")

        ext = frame_path.suffix.lower()
        mime = "image/png" if ext == ".png" else "image/jpeg"
        timestamp_str = f"{int(timestamp // 60):02d}:{int(timestamp % 60):02d}"

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a visual content analyzer for educational/trading videos. "
                        "Analyze the provided video frame and extract:\n"
                        "1. chart_patterns: If this shows a financial chart, identify any "
                        "visible patterns (head and shoulders, candlesticks, trendlines, "
                        "support/resistance levels, indicators shown)\n"
                        "2. visual_content: What's being shown (diagram, chart, slide, "
                        "person talking, screen recording, demo, etc.)\n"
                        "3. description: One-sentence summary of what's visible\n\n"
                        "Respond as JSON with these exact keys."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Frame from timestamp {timestamp_str}. Analyze it.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{img_data}"},
                        },
                    ],
                },
            ],
            temperature=0,
            max_tokens=400,
            response_format={"type": "json_object"},
        )

        result = json.loads(resp.choices[0].message.content)
        result["timestamp"] = timestamp
        result["timestamp_str"] = timestamp_str
        result["frame_file"] = frame_path.name
        result["llm_provider"] = "openai"
        return result

    except Exception as e:
        print(f"  [visual] OpenAI analysis failed: {e}", file=sys.stderr)
        return None


def _llm_analyze_frame_anthropic(
    frame_path: Path,
    timestamp: float,
    api_key: str,
) -> dict | None:
    """Analyze frame using Anthropic Claude vision API."""
    try:
        import base64
        import anthropic

        client = anthropic.Anthropic(api_key=api_key, timeout=60)

        with open(frame_path, "rb") as f:
            img_data = base64.b64encode(f.read()).decode("utf-8")

        ext = frame_path.suffix.lower()
        media_type = "image/png" if ext == ".png" else "image/jpeg"
        timestamp_str = f"{int(timestamp // 60):02d}:{int(timestamp % 60):02d}"

        resp = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=400,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": img_data,
                            },
                        },
                        {
                            "type": "text",
                            "text": (
                                f"Frame from timestamp {timestamp_str} in an educational video. "
                                "Analyze this frame and respond as JSON with these keys: "
                                "chart_patterns (visible chart patterns, indicators, price levels), "
                                "visual_content (what's being shown), "
                                "description (one-sentence summary)."
                            ),
                        },
                    ],
                },
            ],
        )

        text = resp.content[0].text if resp.content else ""
        # Try to parse JSON from response
        try:
            result = json.loads(text)
        except json.JSONDecodeError:
            # Extract JSON from markdown code blocks
            match = re.search(r"\{[^}]+\}", text, re.DOTALL)
            if match:
                result = json.loads(match.group())
            else:
                result = {"description": text, "chart_patterns": "", "visual_content": ""}

        result["timestamp"] = timestamp
        result["timestamp_str"] = timestamp_str
        result["frame_file"] = frame_path.name
        result["llm_provider"] = "anthropic"
        return result

    except Exception as e:
        print(f"  [visual] Anthropic analysis failed: {e}", file=sys.stderr)
        return None


def _llm_analyze_frame_ollama(
    frame_path: Path,
    timestamp: float,
    model: str = "llama3.2-vision",
) -> dict | None:
    """Analyze frame using local Ollama vision model. No API key needed."""
    try:
        import ollama

        timestamp_str = f"{int(timestamp // 60):02d}:{int(timestamp % 60):02d}"

        resp = ollama.chat(
            model=model,
            messages=[{
                "role": "user",
                "content": (
                    f"Frame from timestamp {timestamp_str} in an educational video. "
                    "Analyze this frame and respond as JSON with these keys: "
                    "chart_patterns, visual_content, description."
                ),
                "images": [str(frame_path)],
            }],
        )

        text = resp.get("message", {}).get("content", "")
        try:
            result = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{[^}]+\}", text, re.DOTALL)
            if match:
                result = json.loads(match.group())
            else:
                result = {"description": text, "chart_patterns": "", "visual_content": ""}

        result["timestamp"] = timestamp
        result["timestamp_str"] = timestamp_str
        result["frame_file"] = frame_path.name
        result["llm_provider"] = "ollama"
        return result

    except Exception as e:
        print(f"  [visual] Ollama analysis failed: {e}", file=sys.stderr)
        return None


def _detect_llm_provider() -> str:
    """Auto-detect which LLM provider is available.

    Priority: OPENAI_API_KEY (or 9router) > ANTHROPIC_API_KEY > Ollama running.
    Returns: "openai" | "anthropic" | "ollama" | ""
    """
    # Check for OpenAI (includes 9router which uses OpenAI-compatible API)
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    # Check for Anthropic
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    # Check if Ollama is running locally
    try:
        import urllib.request
        req = urllib.request.Request("http://localhost:11434/api/tags", method="GET")
        urllib.request.urlopen(req, timeout=2)
        return "ollama"
    except Exception:
        pass
    return ""


def _analyze_frame_with_llm(
    frame_path: Path,
    timestamp: float,
    provider: str | None = None,
) -> dict | None:
    """Analyze a frame using the best available LLM provider.

    This is OPTIONAL — OCR runs first and always. LLM adds chart pattern
    analysis and visual context that OCR can't provide.

    Args:
        frame_path: Path to the frame image.
        timestamp: Timestamp in seconds.
        provider: Force a specific provider ("openai" | "anthropic" | "ollama").
                  If None, auto-detects.

    Returns dict with chart_patterns, visual_content, description, or None.
    """
    provider = provider or _detect_llm_provider()

    if provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY", "")
        base_url = os.environ.get("OPENAI_BASE_URL")  # 9router uses this
        return _llm_analyze_frame_openai(frame_path, timestamp, api_key, base_url)

    elif provider == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        return _llm_analyze_frame_anthropic(frame_path, timestamp, api_key)

    elif provider == "ollama":
        return _llm_analyze_frame_ollama(frame_path, timestamp)

    return None


def _extract_visual_claims_from_ocr(
    ocr_text: str,
    timestamp: float,
) -> list[dict]:
    """Parse OCR text into structured visual claims.

    Detects:
    - Bullet points / numbered lists (slide content)
    - Indicator settings (RSI: 14, MA: 200, etc.)
    - Price levels ($100, €50, etc.)
    - Percentages (70%, 30%)
    - Definitions (X = Y, X is Y)
    """
    claims: list[dict] = []
    timestamp_str = f"{int(timestamp // 60):02d}:{int(timestamp % 60):02d}"
    lines = [l.strip() for l in ocr_text.split("\n") if l.strip()]

    # Detect bullet points / list items
    for line in lines:
        # Bullet point: ". Item" or "• Item" or "1. Item"
        bullet_match = re.match(r"^[.•\-\*]\s*(.+)", line) or re.match(r"^\d+[.)]\s*(.+)", line)
        if bullet_match:
            content = bullet_match.group(1).strip()
            if len(content) > 3:
                claims.append({
                    "timestamp": timestamp,
                    "timestamp_str": timestamp_str,
                    "claim": content,
                    "source": "visual_ocr",
                    "type": "bullet_point",
                })

    # Detect indicator settings: "RSI: 14", "MA = 200", "Length: 9"
    for line in lines:
        setting_match = re.search(
            r"(?i)\b(RSI|RSI length|MA|SMA|EMA|MACD|Stochastic|ATR|Bollinger|"
            r"Length|Period|Standard Deviation|Risk Reward|Stop Loss|Take Profit)"
            r"\s*[:=]\s*(\d+(?:\.\d+)?)",
            line,
        )
        if setting_match:
            claims.append({
                "timestamp": timestamp,
                "timestamp_str": timestamp_str,
                "claim": f"{setting_match.group(1)} = {setting_match.group(2)}",
                "source": "visual_ocr",
                "type": "indicator_setting",
            })

    # Detect threshold patterns: "RSI is above 70", "RSI is below 30"
    for line in lines:
        threshold_match = re.search(
            r"(?i)\b(RSI|MACD|Stochastic|ATR|MA|SMA|EMA|OBV|ADX)"
            r"\s+(?:is\s+)?(above|below|over|under|>|<)\s*(\d+(?:\.\d+)?)",
            line,
        )
        if threshold_match:
            indicator = threshold_match.group(1)
            direction = threshold_match.group(2)
            value = threshold_match.group(3)
            claims.append({
                "timestamp": timestamp,
                "timestamp_str": timestamp_str,
                "claim": f"{indicator} {direction} {value}",
                "source": "visual_ocr",
                "type": "indicator_threshold",
            })

    # Detect definitions on slides: "X = Y" or "X is Y"
    for line in lines:
        def_match = re.search(
            r"(?i)\b(Momentum|Divergence|Overbought|Oversold|Bullish|Bearish|"
            r"Support|Resistance|Trendline|Breakout|Reversal)"
            r"\s*[=:]\s*(.{5,80})",
            line,
        )
        if def_match:
            term = def_match.group(1)
            definition = def_match.group(2).strip().rstrip(".,;|")
            if len(definition) > 5:
                claims.append({
                    "timestamp": timestamp,
                    "timestamp_str": timestamp_str,
                    "claim": f"{term} = {definition}",
                    "source": "visual_ocr",
                    "type": "visual_definition",
                })

    # Detect price levels: $100, €50, £75
    for line in lines:
        for m in re.finditer(r"[$€£]\s*(\d+(?:,\d{3})*(?:\.\d+)?)", line):
            claims.append({
                "timestamp": timestamp,
                "timestamp_str": timestamp_str,
                "claim": f"Price level: {m.group(0)}",
                "source": "visual_ocr",
                "type": "price_level",
            })

    # Detect percentages
    for line in lines:
        for m in re.finditer(r"(\d+(?:\.\d+)?)\s*%", line):
            pct = m.group(0)
            # Avoid double-counting with indicator settings
            if not any(c["claim"].endswith(pct) for c in claims if c["type"] == "indicator_setting"):
                claims.append({
                    "timestamp": timestamp,
                    "timestamp_str": timestamp_str,
                    "claim": f"Percentage shown: {pct}",
                    "source": "visual_ocr",
                    "type": "percentage",
                })

    return claims


def extract_visual_content(
    video_id: str,
    output_dir: Path,
    duration: int = 0,
    cookies_file: str | None = None,
    cookies_from_browser: str | None = None,
    proxy: str | None = None,
    enable_visual: bool = True,
    llm_provider: str | None = None,
    chapter_timestamps: list[float] | None = None,
) -> dict:
    """Extract visual content from a YouTube video.

    Pipeline:
      1. Download video in lowest quality
      2. Extract frames at intervals (or chapter boundaries if provided)
      3. Deduplicate similar frames
      4. Run OCR (Tesseract) on each frame — ALWAYS, free, local
      5. Optionally run LLM vision analysis for chart patterns — pluggable
      6. Parse OCR text into structured claims

    OCR is the primary method and requires no API key.
    LLM analysis is optional and auto-detects the best available provider
    (OpenAI/9router, Anthropic, or local Ollama).

    Returns:
        {
            "enabled": bool,
            "ocr_available": bool,
            "llm_provider": str,  # "" if none
            "video_downloaded": bool,
            "frames_extracted": int,
            "frames_after_dedup": int,
            "frames_analyzed_ocr": int,
            "frames_analyzed_llm": int,
            "visual_claims": [...],
            "on_screen_texts": [...],
            "chart_patterns": [...],
            "frame_analyses": [...],
        }
    """
    result: dict[str, Any] = {
        "enabled": enable_visual,
        "ocr_available": bool(_find_tesseract()),
        "llm_provider": "",
        "video_downloaded": False,
        "frames_extracted": 0,
        "frames_after_dedup": 0,
        "frames_analyzed_ocr": 0,
        "frames_analyzed_llm": 0,
        "visual_claims": [],
        "on_screen_texts": [],
        "chart_patterns": [],
        "frame_analyses": [],
    }

    if not enable_visual:
        return result

    # Detect LLM provider (for optional chart pattern analysis)
    provider = llm_provider or _detect_llm_provider()
    result["llm_provider"] = provider

    # Check OCR availability
    if not result["ocr_available"]:
        print("  [visual] Tesseract OCR not found — install: winget install UB-Mannheim.TesseractOCR",
              file=sys.stderr)
        # Still try LLM-only if available
        if not provider:
            print("  [visual] No OCR and no LLM — cannot do visual extraction", file=sys.stderr)
            return result

    # Step 1: Download video
    print("  [visual] Downloading video (low quality)...", file=sys.stderr)
    video_path = _download_video_low_quality(
        video_id, output_dir, cookies_file, cookies_from_browser, proxy,
    )
    if not video_path:
        return result
    result["video_downloaded"] = True

    # Step 2: Extract frames
    print("  [visual] Extracting frames...", file=sys.stderr)
    frames = _extract_frames(
        video_path, output_dir,
        timestamps=chapter_timestamps if chapter_timestamps else None,
    )
    result["frames_extracted"] = len(frames)
    if not frames:
        return result

    # Step 3: Deduplicate similar frames
    frames = _deduplicate_frames(frames)
    result["frames_after_dedup"] = len(frames)
    print(f"  [visual] {len(frames)} unique frames after dedup", file=sys.stderr)

    # Step 4: OCR on each frame (always runs if Tesseract available)
    on_screen_texts: list[dict] = []
    visual_claims: list[dict] = []
    frame_analyses: list[dict] = []

    if result["ocr_available"]:
        print(f"  [visual] Running OCR on {len(frames)} frames...", file=sys.stderr)
        for frame in frames:
            # Parse timestamp from filename
            match = re.search(r"(\d+)m(\d+)s", frame.name)
            timestamp = int(match.group(1)) * 60 + int(match.group(2)) if match else 0
            timestamp_str = f"{int(timestamp // 60):02d}:{int(timestamp % 60):02d}"

            ocr_result = _extract_ocr_from_frame(frame)
            if ocr_result.get("has_content"):
                result["frames_analyzed_ocr"] += 1

                # Store on-screen text
                on_screen_texts.append({
                    "timestamp": timestamp,
                    "timestamp_str": timestamp_str,
                    "text": ocr_result["text"],
                    "word_count": len(ocr_result["words"]),
                    "avg_confidence": (
                        sum(c for _, c in ocr_result["words"]) / len(ocr_result["words"])
                        if ocr_result["words"] else 0
                    ),
                })

                # Extract structured claims from OCR text
                ocr_claims = _extract_visual_claims_from_ocr(ocr_result["text"], timestamp)
                visual_claims.extend(ocr_claims)

                # Store frame analysis
                frame_analyses.append({
                    "timestamp": timestamp,
                    "timestamp_str": timestamp_str,
                    "frame_file": frame.name,
                    "ocr_text": ocr_result["text"],
                    "ocr_lines": ocr_result["lines"],
                    "ocr_word_count": len(ocr_result["words"]),
                    "ocr_avg_confidence": (
                        sum(c for _, c in ocr_result["words"]) / len(ocr_result["words"])
                        if ocr_result["words"] else 0
                    ),
                })

    # Step 5: Optional LLM analysis for chart patterns
    if provider:
        print(f"  [visual] Running LLM analysis ({provider}) on frames with content...",
              file=sys.stderr)
        for fa in frame_analyses:
            frame_path = output_dir / _FRAMES_SUBDIR / fa["frame_file"]
            if not frame_path.exists():
                continue
            llm_result = _analyze_frame_with_llm(frame_path, fa["timestamp"], provider)
            if llm_result:
                result["frames_analyzed_llm"] += 1
                fa["llm_analysis"] = llm_result

                # Collect chart patterns
                cp = llm_result.get("chart_patterns", "")
                if cp and len(cp) > 3:
                    result["chart_patterns"].append({
                        "timestamp": fa["timestamp"],
                        "timestamp_str": fa["timestamp_str"],
                        "patterns": cp,
                        "provider": provider,
                    })

                # Add visual description as a claim
                desc = llm_result.get("description", "")
                if desc:
                    visual_claims.append({
                        "timestamp": fa["timestamp"],
                        "timestamp_str": fa["timestamp_str"],
                        "claim": desc,
                        "source": f"visual_llm_{provider}",
                        "type": "visual_description",
                    })

    result["on_screen_texts"] = on_screen_texts
    result["visual_claims"] = visual_claims
    result["frame_analyses"] = frame_analyses

    # Clean up video file to save space (keep frames)
    try:
        video_path.unlink()
    except Exception:
        pass

    return result


# ---------------------------------------------------------------- TEXT-TO-SPEECH (TTS)

# Default voice — Microsoft Ava neural (natural, female, US English)
DEFAULT_VOICE = "en-US-AvaNeural"

# Voices that sound good for research reports
VOICE_OPTIONS = {
    "ava": "en-US-AvaNeural",          # Female, natural
    "aria": "en-US-AriaNeural",        # Female, conversational
    "emma": "en-US-EmmaNeural",        # Female, warm
    "andrew": "en-US-AndrewNeural",    # Male, natural
    "brian": "en-US-BrianNeural",      # Male, conversational
    "christopher": "en-US-ChristopherNeural",  # Male, authoritative
}


def _format_deep_research_for_tts(report: dict) -> str:
    """Convert a deep research JSON report into a natural spoken script.

    Produces narration-ready text — not raw JSON. Handles all sections:
    executive summary, argument structure, claim verdicts, bias, omissions.
    """
    parts: list[str] = []

    title = report.get("title", "")
    channel = report.get("channel", "")
    if title:
        parts.append(f"Deep Research Report. {title}.")
    if channel:
        parts.append(f"Channel: {channel}.")
    parts.append("")  # pause

    # Executive summary
    summary = report.get("executive_summary", "")
    if summary:
        parts.append("Executive Summary.")
        parts.append(summary)
        parts.append("")

    # Argument structure
    arg = report.get("argument_structure", {})
    if arg:
        thesis = arg.get("main_thesis", "")
        if thesis:
            parts.append("Main Thesis.")
            parts.append(thesis)
            parts.append("")

        premises = arg.get("premises", [])
        if premises:
            parts.append("Premises.")
            for i, p in enumerate(premises, 1):
                premise_text = p.get("premise", "") if isinstance(p, dict) else str(p)
                etype = p.get("evidence_type", "") if isinstance(p, dict) else ""
                evidence_note = f" (Evidence type: {etype})" if etype else ""
                parts.append(f"Premise {i}. {premise_text}{evidence_note}.")
            parts.append("")

        conclusions = arg.get("conclusions", [])
        if conclusions:
            parts.append("Conclusions.")
            for c in conclusions:
                parts.append(f"{c}.")
            parts.append("")

        reasoning = arg.get("reasoning_quality", "")
        if reasoning:
            parts.append("Reasoning Quality.")
            parts.append(reasoning)
            parts.append("")

        fallacies = arg.get("fallacies_identified", [])
        if fallacies:
            parts.append("Logical Fallacies Identified.")
            for f in fallacies:
                if isinstance(f, dict):
                    fname = f.get("fallacy", "")
                    fexample = f.get("example", "")
                    fexplanation = f.get("explanation", "")
                    parts.append(f"{fname}.")
                    if fexample:
                        parts.append(f"Example: {fexample}.")
                    if fexplanation:
                        parts.append(f"Explanation: {fexplanation}.")
                    parts.append("")
            parts.append("")

    # Claim verification
    claims = report.get("claim_verification", [])
    if claims:
        parts.append("Claim Verification.")
        parts.append(f"Total claims verified: {len(claims)}.")
        parts.append("")
        for i, c in enumerate(claims, 1):
            claim_text = c.get("claim", "")
            verdict = c.get("verdict", "")
            evidence = c.get("evidence", "")
            confidence = c.get("confidence", "")
            quote = c.get("verbatim_quote", "")

            parts.append(f"Claim {i}.")
            if quote:
                parts.append(f"Quote: {quote}.")
            parts.append(f"Claim: {claim_text}.")
            parts.append(f"Verdict: {verdict}.")
            if confidence:
                parts.append(f"Confidence: {confidence}.")
            if evidence:
                parts.append(f"Evidence: {evidence}.")
            sources = c.get("sources", [])
            if sources:
                src_count = len(sources)
                parts.append(f"Sources consulted: {src_count}.")
            parts.append("")

    # Bias assessment
    bias = report.get("bias_assessment", {})
    if bias:
        parts.append("Bias Assessment.")
        credibility = bias.get("speaker_credibility", "")
        if credibility:
            parts.append(f"Speaker credibility: {credibility}.")
            parts.append("")

        biases = bias.get("potential_biases", [])
        if biases:
            parts.append("Potential biases.")
            for b in biases:
                parts.append(f"{b}.")
            parts.append("")

        conflicts = bias.get("conflicts_of_interest", [])
        if conflicts:
            parts.append("Conflicts of interest.")
            for c in conflicts:
                parts.append(f"{c}.")
            parts.append("")

        ecosystem = bias.get("financial_ecosystem", "")
        if ecosystem:
            parts.append(f"Financial ecosystem: {ecosystem}.")
            parts.append("")

        reliability = bias.get("overall_reliability", "")
        if reliability:
            parts.append(f"Overall reliability: {reliability}.")
            parts.append("")

    # Cross-references
    xrefs = report.get("cross_references", [])
    if xrefs:
        parts.append("Cross References.")
        for xr in xrefs:
            topic = xr.get("topic", "")
            video_claims = xr.get("this_video_claims", "")
            auth_says = xr.get("authoritative_sources_say", "")
            agreement = xr.get("agreement_level", "")

            parts.append(f"Topic: {topic}.")
            if video_claims:
                parts.append(f"This video claims: {video_claims}.")
            if auth_says:
                parts.append(f"Authoritative sources say: {auth_says}.")
            if agreement:
                parts.append(f"Agreement level: {agreement}.")
            parts.append("")

    # Omission analysis
    omissions = report.get("omission_analysis", [])
    if omissions:
        parts.append("Omission Analysis. What was NOT said that should have been.")
        for o in omissions:
            parts.append(f"{o}.")
        parts.append("")

    # Research gaps
    gaps = report.get("research_gaps", [])
    if gaps:
        parts.append("Research Gaps.")
        for g in gaps:
            parts.append(f"{g}.")
        parts.append("")

    # Open questions
    questions = report.get("open_questions", [])
    if questions:
        parts.append("Open Questions.")
        for q in questions:
            parts.append(f"{q}.")
        parts.append("")

    # Overall confidence
    conf = report.get("overall_confidence", {})
    if conf:
        level = conf.get("level", "")
        rationale = conf.get("rationale", "")
        parts.append("Overall Confidence.")
        if level:
            parts.append(f"Level: {level}.")
        if rationale:
            parts.append(f"Rationale: {rationale}.")
        parts.append("")

    # Methodology
    methodology = report.get("methodology", {})
    if methodology:
        approach = methodology.get("approach", "")
        sources_consulted = methodology.get("sources_consulted", 0)
        searches = methodology.get("web_searches_performed", 0)
        limitations = methodology.get("limitations", [])

        parts.append("Methodology.")
        if approach:
            parts.append(f"Approach: {approach}.")
        if sources_consulted:
            parts.append(f"Sources consulted: {sources_consulted}.")
        if searches:
            parts.append(f"Web searches performed: {searches}.")
        if limitations:
            parts.append("Limitations.")
            for lim in limitations:
                parts.append(f"{lim}.")
        parts.append("")

    parts.append("End of report.")

    return "\n".join(parts)


def _format_light_research_for_tts(report: dict) -> str:
    """Convert a light research JSON report into a natural spoken script."""
    parts: list[str] = []

    title = report.get("title", "")
    channel = report.get("channel", "")
    if title:
        parts.append(f"Light Research Report. {title}.")
    if channel:
        parts.append(f"Channel: {channel}.")
    parts.append("")

    tldr = report.get("tldr", "")
    if tldr:
        parts.append("TL;DR.")
        parts.append(tldr)
        parts.append("")

    summary = report.get("summary", "")
    if summary:
        parts.append("Summary.")
        parts.append(summary)
        parts.append("")

    key_points = report.get("key_points", [])
    if key_points:
        parts.append("Key Points.")
        for i, kp in enumerate(key_points, 1):
            parts.append(f"Point {i}. {kp}.")
        parts.append("")

    topics = report.get("topics", [])
    if topics:
        parts.append("Topics Covered.")
        for t in topics:
            if isinstance(t, dict):
                name = t.get("name", "")
                desc = t.get("description", "")
                parts.append(f"{name}. {desc}.")
            else:
                parts.append(f"{t}.")
        parts.append("")

    action_items = report.get("action_items", [])
    if action_items:
        parts.append("Action Items.")
        for a in action_items:
            parts.append(f"{a}.")
        parts.append("")

    quotes = report.get("quotes", [])
    if quotes:
        parts.append("Notable Quotes.")
        for q in quotes:
            parts.append(f'"{q}".')
        parts.append("")

    controversial = report.get("controversial_claims", [])
    if controversial:
        parts.append("Controversial Claims.")
        for c in controversial:
            parts.append(f"{c}.")
        parts.append("")

    sentiment = report.get("sentiment", "")
    if sentiment:
        parts.append(f"Sentiment: {sentiment}.")
    energy = report.get("energy", "")
    if energy:
        parts.append(f"Energy: {energy}.")
    parts.append("")

    parts.append("End of report.")
    return "\n".join(parts)


def _format_report_for_tts(report: dict) -> str:
    """Auto-detect report type and format for TTS."""
    mode = report.get("research_mode", "")
    if mode == "deep":
        return _format_deep_research_for_tts(report)
    # Light research or untyped
    return _format_light_research_for_tts(report)


async def _generate_speech(
    text: str,
    output_path: Path,
    voice: str = DEFAULT_VOICE,
    rate: str = "+0%",
    volume: str = "+0%",
) -> None:
    """Generate speech audio file using edge-tts.

    Args:
        text: The text to speak
        output_path: Where to save the MP3
        voice: edge-tts voice name (e.g. en-US-AvaNeural)
        rate: Speech rate adjustment (e.g. "+10%", "-5%", "+0%")
        volume: Volume adjustment (e.g. "+10%", "-5%", "+0%")
    """
    communicate = edge_tts.Communicate(
        text,
        voice=voice,
        rate=rate,
        volume=volume,
    )
    await communicate.save(str(output_path))


def read_report(
    report_path: str | Path,
    output_dir: Path | None = None,
    voice: str = DEFAULT_VOICE,
    rate: str = "+0%",
    volume: str = "+0%",
) -> dict:
    """Read a research report aloud as an MP3 file.

    Args:
        report_path: Path to the _research.json or _deep_research.json file
        output_dir: Where to save the MP3 (defaults to same dir as report)
        voice: edge-tts voice name
        rate: Speech rate (e.g. "+10%" faster, "-10%" slower)
        volume: Volume adjustment

    Returns:
        Dict with ok, audio_path, text_length, char_count, voice, duration_estimate
    """
    report_path = Path(report_path)
    if not report_path.exists():
        return {"ok": False, "error": f"Report file not found: {report_path}"}

    # Load report JSON
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return {"ok": False, "error": f"Invalid JSON: {e}"}

    # Format for TTS
    script = _format_report_for_tts(report)
    if not script.strip():
        return {"ok": False, "error": "Report is empty or has no readable content"}

    # Determine output path
    out = output_dir or report_path.parent
    out.mkdir(parents=True, exist_ok=True)
    stem = report_path.stem  # e.g. "pCmJ8wsAS_w_deep_research"
    audio_path = out / f"{stem}_audio.mp3"

    # Generate speech
    try:
        asyncio.run(_generate_speech(script, audio_path, voice, rate, volume))
    except Exception as e:
        return {"ok": False, "error": f"TTS generation failed: {e}"}

    if not audio_path.exists():
        return {"ok": False, "error": "Audio file was not created"}

    audio_size = audio_path.stat().st_size
    char_count = len(script)
    # Rough duration estimate: edge-tts ~150 chars/sec at +0%
    duration_estimate = char_count / 150

    return {
        "ok": True,
        "audio_path": str(audio_path),
        "report_path": str(report_path),
        "text_length": char_count,
        "audio_size_bytes": audio_size,
        "audio_size_mb": round(audio_size / (1024 * 1024), 2),
        "voice": voice,
        "duration_estimate_seconds": round(duration_estimate),
        "duration_estimate_minutes": round(duration_estimate / 60, 1),
        "rate": rate,
        "volume": volume,
    }


# ---------------------------------------------------------------- INTERACTIVE WEB REPORTS

import html as _html


def _verdict_color(verdict: str) -> str:
    """Map a verdict string to a color for the HTML badge."""
    v = verdict.lower().strip()
    if "verified" == v or v == "verified":
        return ("#22c55e", "#16a34a", "✓ Verified")
    if "partially" in v:
        return ("#eab308", "#ca8a04", "◐ Partially Verified")
    if "contradicted" in v:
        return ("#ef4444", "#dc2626", "✗ Contradicted")
    if "unverifiable" in v:
        return ("#94a3b8", "#64748b", "? Unverifiable")
    if "opinion" in v:
        return ("#94a3b8", "#64748b", "💬 Opinion")
    # unverified or default
    return ("#f59e0b", "#d97706", "? Unverified")


def _confidence_bar(confidence: str) -> str:
    """Generate an HTML confidence bar."""
    c = confidence.upper().strip()
    if c == "HIGH":
        pct, color = 90, "#22c55e"
    elif c == "MODERATE":
        pct, color = 60, "#eab308"
    elif c == "LOW":
        pct, color = 30, "#ef4444"
    else:
        pct, color = 50, "#94a3b8"
    return f'<div class="confidence-bar"><div class="confidence-fill" style="width:{pct}%;background:{color}"></div><span>{c}</span></div>'


def _reliability_score(reliability: str) -> int:
    """Convert reliability string to 0-100 score for the bias meter."""
    r = reliability.lower().strip()
    if "high" in r and "mixed" not in r:
        return 85
    if "moderate" in r:
        return 55
    if "mixed" in r:
        return 45
    if "low" in r:
        return 20
    return 50


def _escape(text: str) -> str:
    """HTML-escape text for safe insertion."""
    return _html.escape(str(text)) if text else ""


def _format_source_list(sources: list) -> str:
    """Generate HTML for a list of sources."""
    if not sources:
        return ""
    items = []
    for s in sources:
        if isinstance(s, dict):
            url = s.get("url", "")
            title = s.get("title", url)
            reliability = s.get("reliability", "")
            rel_badge = f'<span class="rel-badge rel-{reliability}">{reliability}</span>' if reliability else ""
            if url:
                items.append(f'<li><a href="{_escape(url)}" target="_blank" rel="noopener">{_escape(title)}</a> {rel_badge}</li>')
            else:
                items.append(f'<li>{_escape(title)} {rel_badge}</li>')
        elif isinstance(s, str):
            if s.startswith("http"):
                items.append(f'<li><a href="{_escape(s)}" target="_blank" rel="noopener">{_escape(s)}</a></li>')
            else:
                items.append(f'<li>{_escape(s)}</li>')
    return f'<ul class="source-list">{"".join(items)}</ul>'


def report_to_html(report: dict, output_path: str | Path | None = None) -> str:
    """Convert a research report (Light or Deep) into a self-contained interactive HTML file.

    The HTML is fully self-contained — no external CSS, JS, or CDN dependencies.
    Works offline, can be emailed, shared in Discord, etc.

    Args:
        report: The report dict (from JSON)
        output_path: Where to save the HTML file. If None, returns HTML string only.

    Returns:
        The HTML string (also saves to file if output_path given)
    """
    is_deep = report.get("research_mode") == "deep"
    title = report.get("title", "Unknown Video")
    channel = report.get("channel", "")
    url = report.get("url", "")
    video_id = ""
    if url:
        m = re.search(r"(?:v=|youtu\.be/)([\w-]{11})", url)
        if m:
            video_id = m.group(1)

    # --- Build sections ---
    sections: list[str] = []

    # --- Executive Summary ---
    summary = report.get("executive_summary", "") or report.get("summary", "")
    if summary:
        sections.append(f'''
        <section class="card" id="summary">
            <h2 onclick="toggleSection(this)">▾ Executive Summary</h2>
            <div class="card-body">
                <p>{_escape(summary).replace(chr(10), '<br>')}</p>
            </div>
        </section>''')

    # --- TL;DR (Light Research) ---
    tldr = report.get("tldr", "")
    if tldr:
        sections.append(f'''
        <section class="card" id="tldr">
            <h2 onclick="toggleSection(this)">▾ TL;DR</h2>
            <div class="card-body">
                <p class="tldr">{_escape(tldr)}</p>
            </div>
        </section>''')

    # --- Key Points (Light Research) ---
    key_points = report.get("key_points", [])
    if key_points:
        kp_items = "".join(f"<li>{_escape(kp)}</li>" for kp in key_points)
        sections.append(f'''
        <section class="card" id="key-points">
            <h2 onclick="toggleSection(this)">▾ Key Points</h2>
            <div class="card-body">
                <ol>{kp_items}</ol>
            </div>
        </section>''')

    # --- Bias Meter + Assessment ---
    bias = report.get("bias_assessment", {})
    if bias:
        reliability = bias.get("overall_reliability", "")
        score = _reliability_score(reliability)
        credibility = bias.get("speaker_credibility", "")
        biases = bias.get("potential_biases", [])
        conflicts = bias.get("conflicts_of_interest", [])
        ecosystem = bias.get("financial_ecosystem", "")

        bias_items = "".join(f"<li>{_escape(b)}</li>" for b in biases)
        conflict_items = "".join(f"<li>{_escape(c)}</li>" for c in conflicts)

        # Gauge color based on score
        if score >= 70:
            gauge_color = "#22c55e"
        elif score >= 50:
            gauge_color = "#eab308"
        else:
            gauge_color = "#ef4444"

        sections.append(f'''
        <section class="card" id="bias">
            <h2 onclick="toggleSection(this)">▾ Bias Assessment</h2>
            <div class="card-body">
                <div class="bias-meter">
                    <div class="bias-gauge">
                        <div class="gauge-track">
                            <div class="gauge-fill" style="width:{score}%;background:{gauge_color}"></div>
                        </div>
                        <div class="gauge-label">
                            <span class="gauge-score">{score}/100</span>
                            <span class="gauge-reliability">Reliability: {_escape(reliability)}</span>
                        </div>
                    </div>
                </div>
                {f'<p><strong>Speaker Credibility:</strong> {_escape(credibility)}</p>' if credibility else ''}
                {f'<h3>Potential Biases</h3><ul>{bias_items}</ul>' if bias_items else ''}
                {f'<h3>Conflicts of Interest</h3><ul class="conflicts">{conflict_items}</ul>' if conflict_items else ''}
                {f'<p><strong>Financial Ecosystem:</strong> {_escape(ecosystem)}</p>' if ecosystem else ''}
            </div>
        </section>''')

    # --- Claim Verification ---
    claims = report.get("claim_verification", [])
    if claims:
        claim_cards = []
        for i, c in enumerate(claims, 1):
            claim_text = c.get("claim", "")
            quote = c.get("verbatim_quote", "")
            verdict = c.get("verdict", "")
            evidence = c.get("evidence", "")
            confidence = c.get("confidence", "")
            sources = c.get("sources", [])

            bg, border, label = _verdict_color(verdict)
            conf_bar = _confidence_bar(confidence)
            source_html = _format_source_list(sources)

            # Timestamp link — jump to exact moment if we have timestamp, else just video
            timestamp_html = ""
            if video_id:
                claim_ts = c.get("timestamp", {})
                ts_seconds = claim_ts.get("start") if isinstance(claim_ts, dict) else None
                ts_label = claim_ts.get("timestamp", "") if isinstance(claim_ts, dict) else ""
                if ts_seconds is not None:
                    timestamp_html = f'<a href="https://youtube.com/watch?v={video_id}&t={int(ts_seconds)}s" target="_blank" class="timestamp-link">▶ Jump to {ts_label}</a>'
                else:
                    timestamp_html = f'<a href="https://youtube.com/watch?v={video_id}" target="_blank" class="timestamp-link">▶ Watch on YouTube</a>'

            # Strength badge
            strength = c.get("strength", "")
            strength_badge = ""
            if strength:
                s_colors = {"high": "#22c55e", "moderate": "#eab308", "low": "#f59e0b", "opinion": "#94a3b8"}
                s_color = s_colors.get(strength, "#94a3b8")
                strength_badge = f'<span class="strength-badge" style="background:{s_color};color:#fff">{strength}</span>'

            # Negation indicator
            negated = c.get("negated", False)
            negated_badge = '<span class="negated-badge" style="background:#ef4444;color:#fff">NEGATED</span>' if negated else ""

            # Subject
            subject = c.get("subject", "")
            subject_html = f'<div class="claim-subject"><strong>Subject:</strong> {_escape(subject)}</div>' if subject else ""

            claim_cards.append(f'''
            <div class="claim-card" style="border-left:4px solid {border}">
                <div class="claim-header">
                    <span class="verdict-badge" style="background:{bg};color:#fff">{label}</span>
                    {strength_badge}
                    {negated_badge}
                    <span class="claim-number">Claim {i}</span>
                </div>
                {f'<div class="claim-quote">"{_escape(quote)}"</div>' if quote else ''}
                <div class="claim-text">{_escape(claim_text)}</div>
                {subject_html}
                {conf_bar}
                {f'<div class="claim-evidence"><strong>Evidence:</strong> {_escape(evidence)}</div>' if evidence else ''}
                {timestamp_html}
                {source_html}
            </div>''')

        sections.append(f'''
        <section class="card" id="claims">
            <h2 onclick="toggleSection(this)">▾ Claim Verification ({len(claims)} claims)</h2>
            <div class="card-body">
                {''.join(claim_cards)}
            </div>
        </section>''')

    # --- Controversial Claims (Light Research) ---
    controversial = report.get("controversial_claims", [])
    if controversial:
        items = "".join(f"<li>{_escape(c)}</li>" for c in controversial)
        sections.append(f'''
        <section class="card" id="controversial">
            <h2 onclick="toggleSection(this)">▾ Controversial Claims</h2>
            <div class="card-body">
                <ul>{items}</ul>
            </div>
        </section>''')

    # --- Argument Structure ---
    arg = report.get("argument_structure", {})
    if arg:
        thesis = arg.get("main_thesis", "")
        premises = arg.get("premises", [])
        conclusions = arg.get("conclusions", [])
        reasoning = arg.get("reasoning_quality", "")
        fallacies = arg.get("fallacies_identified", [])

        premise_items = ""
        for i, p in enumerate(premises, 1):
            if isinstance(p, dict):
                ptext = p.get("premise", "")
                etype = p.get("evidence_type", "")
                premise_items += f'<li><span class="evidence-type evidence-{_escape(etype)}">{_escape(etype)}</span> {_escape(ptext)}</li>'
            else:
                premise_items += f"<li>{_escape(p)}</li>"

        conclusion_items = "".join(f"<li>{_escape(c)}</li>" for c in conclusions)

        fallacy_cards = ""
        for f in fallacies:
            if isinstance(f, dict):
                fname = f.get("fallacy", "")
                fexample = f.get("example", "")
                fexplanation = f.get("explanation", "")
                fallacy_cards += f'''
                <div class="fallacy-card">
                    <h4>{_escape(fname)}</h4>
                    {f'<p class="fallacy-example">"{_escape(fexample)}"</p>' if fexample else ''}
                    {f'<p class="fallacy-explanation">{_escape(fexplanation)}</p>' if fexplanation else ''}
                </div>'''

        sections.append(f'''
        <section class="card" id="argument">
            <h2 onclick="toggleSection(this)">▾ Argument Structure</h2>
            <div class="card-body">
                {f'<div class="thesis"><strong>Main Thesis:</strong> {_escape(thesis)}</div>' if thesis else ''}
                {f'<h3>Premises</h3><ol>{premise_items}</ol>' if premise_items else ''}
                {f'<h3>Conclusions</h3><ul>{conclusion_items}</ul>' if conclusion_items else ''}
                {f'<h3>Reasoning Quality</h3><p>{_escape(reasoning)}</p>' if reasoning else ''}
                {f'<h3>Logical Fallacies ({len(fallacies)})</h3>{fallacy_cards}' if fallacy_cards else ''}
            </div>
        </section>''')

    # --- Cross-References ---
    xrefs = report.get("cross_references", [])
    if xrefs:
        xref_cards = ""
        for xr in xrefs:
            topic = xr.get("topic", "")
            video_claims = xr.get("this_video_claims", "")
            auth_says = xr.get("authoritative_sources_say", "")
            agreement = xr.get("agreement_level", "")
            xr_sources = xr.get("sources", [])

            # Agreement badge
            ag_color = "#94a3b8"
            if "consistent" in agreement.lower() and "partial" not in agreement.lower():
                ag_color = "#22c55e"
            elif "partial" in agreement.lower():
                ag_color = "#eab308"
            elif "contradict" in agreement.lower():
                ag_color = "#ef4444"

            xref_cards += f'''
            <div class="xref-card">
                <h4>{_escape(topic)}</h4>
                <div class="xref-comparison">
                    <div class="xref-side">
                        <strong>Video Claims:</strong>
                        <p>{_escape(video_claims)}</p>
                    </div>
                    <div class="xref-side">
                        <strong>Authoritative Sources Say:</strong>
                        <p>{_escape(auth_says)}</p>
                    </div>
                </div>
                <span class="agreement-badge" style="background:{ag_color};color:#fff">{_escape(agreement)}</span>
                {_format_source_list(xr_sources)}
            </div>'''

        sections.append(f'''
        <section class="card" id="cross-refs">
            <h2 onclick="toggleSection(this)">▾ Cross-References</h2>
            <div class="card-body">
                {xref_cards}
            </div>
        </section>''')

    # --- Omission Analysis ---
    omissions = report.get("omission_analysis", [])
    if omissions:
        items = "".join(f"<li>{_escape(o)}</li>" for o in omissions)
        sections.append(f'''
        <section class="card" id="omissions">
            <h2 onclick="toggleSection(this)">▾ Omission Analysis — What Was NOT Said</h2>
            <div class="card-body">
                <ul class="omissions">{items}</ul>
            </div>
        </section>''')

    # --- Topics (Light Research) ---
    topics = report.get("topics", [])
    if topics:
        topic_items = ""
        for t in topics:
            if isinstance(t, dict):
                topic_items += f'<div class="topic-card"><h4>{_escape(t.get("name",""))}</h4><p>{_escape(t.get("description",""))}</p></div>'
            else:
                topic_items += f'<div class="topic-card"><h4>{_escape(t)}</h4></div>'
        sections.append(f'''
        <section class="card" id="topics">
            <h2 onclick="toggleSection(this)">▾ Topics</h2>
            <div class="card-body">
                <div class="topic-grid">{topic_items}</div>
            </div>
        </section>''')

    # --- Action Items (Light Research) ---
    action_items = report.get("action_items", [])
    if action_items:
        items = "".join(f"<li>{_escape(a)}</li>" for a in action_items)
        sections.append(f'''
        <section class="card" id="actions">
            <h2 onclick="toggleSection(this)">▾ Action Items</h2>
            <div class="card-body">
                <ul>{items}</ul>
            </div>
        </section>''')

    # --- Quotes (Light Research) ---
    quotes = report.get("quotes", [])
    if quotes:
        items = "".join(f'<blockquote>"{_escape(q)}"</blockquote>' for q in quotes)
        sections.append(f'''
        <section class="card" id="quotes">
            <h2 onclick="toggleSection(this)">▾ Notable Quotes</h2>
            <div class="card-body">
                {items}
            </div>
        </section>''')

    # --- Source Bibliography ---
    bibliography = report.get("source_bibliography", [])
    if bibliography:
        bib_items = ""
        for b in bibliography:
            if isinstance(b, dict):
                burl = b.get("url", "")
                btitle = b.get("title", burl)
                btype = b.get("type", "")
                brel = b.get("reliability", "")
                rel_badge = f'<span class="rel-badge rel-{brel}">{brel}</span>' if brel else ""
                type_badge = f'<span class="type-badge">{btype}</span>' if btype else ""
                if burl:
                    bib_items += f'<li><a href="{_escape(burl)}" target="_blank" rel="noopener">{_escape(btitle)}</a> {type_badge} {rel_badge}</li>'
                else:
                    bib_items += f'<li>{_escape(btitle)} {type_badge} {rel_badge}</li>'
        sections.append(f'''
        <section class="card" id="bibliography">
            <h2 onclick="toggleSection(this)">▾ Source Bibliography ({len(bibliography)} sources)</h2>
            <div class="card-body">
                <ul class="bibliography">{bib_items}</ul>
            </div>
        </section>''')

    # --- Research Gaps ---
    gaps = report.get("research_gaps", [])
    if gaps:
        items = "".join(f"<li>{_escape(g)}</li>" for g in gaps)
        sections.append(f'''
        <section class="card" id="gaps">
            <h2 onclick="toggleSection(this)">▾ Research Gaps</h2>
            <div class="card-body">
                <ul>{items}</ul>
            </div>
        </section>''')

    # --- Open Questions ---
    questions = report.get("open_questions", [])
    if questions:
        items = "".join(f"<li>{_escape(q)}</li>" for q in questions)
        sections.append(f'''
        <section class="card" id="questions">
            <h2 onclick="toggleSection(this)">▾ Open Questions</h2>
            <div class="card-body">
                <ul>{items}</ul>
            </div>
        </section>''')

    # --- Overall Confidence ---
    conf = report.get("overall_confidence", {})
    if conf:
        level = conf.get("level", "")
        rationale = conf.get("rationale", "")
        conf_bar = _confidence_bar(level)
        sections.append(f'''
        <section class="card" id="confidence">
            <h2 onclick="toggleSection(this)">▾ Overall Confidence</h2>
            <div class="card-body">
                {conf_bar}
                {f'<p>{_escape(rationale)}</p>' if rationale else ''}
            </div>
        </section>''')

    # --- Methodology ---
    methodology = report.get("methodology", {})
    if methodology:
        approach = methodology.get("approach", "")
        sources_count = methodology.get("sources_consulted", 0)
        searches = methodology.get("web_searches_performed", 0)
        limitations = methodology.get("limitations", [])

        lim_items = "".join(f"<li>{_escape(l)}</li>" for l in limitations)
        sections.append(f'''
        <section class="card" id="methodology">
            <h2 onclick="toggleSection(this)">▾ Methodology</h2>
            <div class="card-body">
                {f'<p><strong>Approach:</strong> {_escape(approach)}</p>' if approach else ''}
                <div class="methodology-stats">
                    <div class="stat"><span class="stat-num">{sources_count}</span><span class="stat-label">Sources</span></div>
                    <div class="stat"><span class="stat-num">{searches}</span><span class="stat-label">Web Searches</span></div>
                </div>
                {f'<h3>Limitations</h3><ul>{lim_items}</ul>' if lim_items else ''}
            </div>
        </section>''')

    # --- Sentiment / Energy (Light Research) ---
    sentiment = report.get("sentiment", "")
    energy = report.get("energy", "")
    if sentiment or energy:
        sections.append(f'''
        <section class="card" id="meta">
            <h2 onclick="toggleSection(this)">▾ Video Meta</h2>
            <div class="card-body">
                {f'<p><strong>Sentiment:</strong> {_escape(sentiment)}</p>' if sentiment else ''}
                {f'<p><strong>Energy:</strong> {_escape(energy)}</p>' if energy else ''}
            </div>
        </section>''')

    # --- Assemble full HTML ---
    mode_label = "Deep Research" if is_deep else "Light Research"
    mode_class = "deep" if is_deep else "light"
    thumbnail = f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg" if video_id else ""

    html_doc = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{_escape(title)} — {mode_label} Report</title>
<style>
:root {{
  --bg: #0f172a;
  --card-bg: #1e293b;
  --card-border: #334155;
  --text: #e2e8f0;
  --text-dim: #94a3b8;
  --accent: #3b82f6;
  --accent-hover: #2563eb;
  --green: #22c55e;
  --red: #ef4444;
  --yellow: #eab308;
  --orange: #f59e0b;
  --gray: #94a3b8;
  --radius: 12px;
}}
[data-theme="light"] {{
  --bg: #f8fafc;
  --card-bg: #ffffff;
  --card-border: #e2e8f0;
  --text: #1e293b;
  --text-dim: #64748b;
}}
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
  background: var(--bg);
  color: var(--text);
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  line-height: 1.6;
  padding: 20px;
  max-width: 900px;
  margin: 0 auto;
}}
.theme-toggle {{
  position: fixed;
  top: 20px;
  right: 20px;
  background: var(--card-bg);
  border: 1px solid var(--card-border);
  color: var(--text);
  padding: 8px 16px;
  border-radius: 8px;
  cursor: pointer;
  font-size: 14px;
  z-index: 100;
}}
.theme-toggle:hover {{ border-color: var(--accent); }}

/* Header */
.header {{
  margin-bottom: 30px;
  padding-bottom: 20px;
  border-bottom: 1px solid var(--card-border);
}}
.mode-badge {{
  display: inline-block;
  padding: 4px 12px;
  border-radius: 20px;
  font-size: 12px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 1px;
  margin-bottom: 12px;
}}
.mode-badge.deep {{ background: var(--accent); color: #fff; }}
.mode-badge.light {{ background: var(--yellow); color: #000; }}
.header h1 {{
  font-size: 28px;
  margin-bottom: 8px;
  line-height: 1.3;
}}
.header .channel {{ color: var(--text-dim); font-size: 16px; }}
.header .channel a {{ color: var(--accent); text-decoration: none; }}
.header .video-link {{
  display: inline-block;
  margin-top: 10px;
  padding: 6px 14px;
  background: var(--card-bg);
  border: 1px solid var(--card-border);
  border-radius: 8px;
  color: var(--text);
  text-decoration: none;
  font-size: 14px;
}}
.header .video-link:hover {{ border-color: var(--accent); }}
{f'.header img.thumbnail {{ width:100%; max-height:300px; object-fit:cover; border-radius:var(--radius); margin:15px 0; }}' if thumbnail else ''}

/* Cards */
.card {{
  background: var(--card-bg);
  border: 1px solid var(--card-border);
  border-radius: var(--radius);
  margin-bottom: 16px;
  overflow: hidden;
}}
.card h2 {{
  padding: 16px 20px;
  cursor: pointer;
  font-size: 18px;
  font-weight: 600;
  user-select: none;
  display: flex;
  align-items: center;
  gap: 8px;
  transition: background 0.2s;
}}
.card h2:hover {{ background: rgba(59,130,246,0.1); }}
.card-body {{ padding: 0 20px 20px; }}
.card-body h3 {{ font-size: 15px; margin: 16px 0 8px; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.5px; }}
.card-body h4 {{ font-size: 16px; margin: 12px 0 6px; }}
.card-body p {{ margin-bottom: 10px; }}
.card-body ul, .card-body ol {{ padding-left: 20px; margin-bottom: 12px; }}
.card-body li {{ margin-bottom: 6px; }}
.card.collapsed .card-body {{ display: none; }}
.card.collapsed h2::before {{ content: '▸ '; }}
.card h2::before {{ content: '▾ '; }}

/* TL;DR */
.tldr {{ font-size: 18px; font-weight: 500; padding: 12px; background: rgba(59,130,246,0.1); border-radius: 8px; border-left: 4px solid var(--accent); }}

/* Bias Meter */
.bias-meter {{ margin-bottom: 20px; }}
.gauge-track {{ width: 100%; height: 30px; background: var(--bg); border-radius: 15px; overflow: hidden; border: 1px solid var(--card-border); }}
.gauge-fill {{ height: 100%; border-radius: 15px; transition: width 0.5s ease; }}
.gauge-label {{ display: flex; justify-content: space-between; margin-top: 8px; }}
.gauge-score {{ font-size: 24px; font-weight: 700; }}
.gauge-reliability {{ color: var(--text-dim); font-size: 14px; align-self: center; }}

/* Claim Cards */
.claim-card {{
  background: var(--bg);
  border-radius: 8px;
  padding: 16px;
  margin-bottom: 12px;
}}
.claim-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }}
.verdict-badge {{
  padding: 4px 12px;
  border-radius: 20px;
  font-size: 13px;
  font-weight: 600;
}}
.claim-number {{ color: var(--text-dim); font-size: 13px; margin-left: auto; }}
.strength-badge {{ padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: 600; }}
.negated-badge {{ padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: 600; }}
.claim-subject {{ font-size: 14px; color: var(--text-dim); margin-bottom: 8px; padding: 6px 10px; background: var(--bg); border-radius: 6px; }}
.claim-quote {{ font-style: italic; color: var(--text-dim); margin-bottom: 8px; padding-left: 12px; border-left: 3px solid var(--card-border); }}
.claim-text {{ font-size: 16px; margin-bottom: 10px; }}
.claim-evidence {{ font-size: 14px; color: var(--text-dim); margin-bottom: 10px; }}
.timestamp-link {{
  display: inline-block;
  padding: 4px 10px;
  background: var(--accent);
  color: #fff;
  border-radius: 6px;
  text-decoration: none;
  font-size: 13px;
  margin-bottom: 10px;
}}
.timestamp-link:hover {{ background: var(--accent-hover); }}

/* Confidence Bar */
.confidence-bar {{
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 10px;
}}
.confidence-bar div {{
  height: 8px;
  border-radius: 4px;
  flex: 1;
  background: var(--card-border);
  position: relative;
  overflow: hidden;
}}
.confidence-fill {{
  height: 100%;
  border-radius: 4px;
  position: absolute;
  left: 0;
  top: 0;
}}
.confidence-bar span {{ font-size: 12px; color: var(--text-dim); white-space: nowrap; }}

/* Source List */
.source-list {{ list-style: none; padding-left: 0; }}
.source-list li {{ padding: 4px 0; font-size: 14px; }}
.source-list a {{ color: var(--accent); text-decoration: none; }}
.source-list a:hover {{ text-decoration: underline; }}
.rel-badge {{
  padding: 2px 8px;
  border-radius: 10px;
  font-size: 11px;
  font-weight: 600;
  margin-left: 6px;
}}
.rel-high {{ background: rgba(34,197,94,0.2); color: var(--green); }}
.rel-moderate {{ background: rgba(234,179,8,0.2); color: var(--yellow); }}
.rel-low {{ background: rgba(239,68,68,0.2); color: var(--red); }}
.type-badge {{
  padding: 2px 8px;
  border-radius: 10px;
  font-size: 11px;
  background: var(--card-border);
  color: var(--text-dim);
  margin-left: 6px;
}}

/* Fallacy Cards */
.fallacy-card {{
  background: var(--bg);
  border-radius: 8px;
  padding: 14px;
  margin-bottom: 10px;
  border-left: 3px solid var(--red);
}}
.fallacy-example {{ font-style: italic; color: var(--text-dim); margin: 6px 0; }}
.fallacy-explanation {{ font-size: 14px; }}

/* Evidence Type Badges */
.evidence-type {{
  padding: 2px 8px;
  border-radius: 10px;
  font-size: 11px;
  font-weight: 600;
  margin-right: 6px;
}}
.evidence-empirical {{ background: rgba(34,197,94,0.2); color: var(--green); }}
.evidence-anecdotal {{ background: rgba(234,179,8,0.2); color: var(--yellow); }}
.evidence-logical {{ background: rgba(59,130,246,0.2); color: var(--accent); }}
.evidence-authority {{ background: rgba(168,85,247,0.2); color: #a855f7; }}
.evidence-none {{ background: rgba(239,68,68,0.2); color: var(--red); }}

/* Cross-Reference Cards */
.xref-card {{
  background: var(--bg);
  border-radius: 8px;
  padding: 14px;
  margin-bottom: 12px;
}}
.xref-comparison {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin: 10px 0; }}
.xref-side {{ padding: 10px; border-radius: 6px; }}
.xref-side:first-child {{ background: rgba(239,68,68,0.1); }}
.xref-side:last-child {{ background: rgba(34,197,94,0.1); }}
.xref-side strong {{ font-size: 13px; color: var(--text-dim); }}
.agreement-badge {{
  display: inline-block;
  padding: 4px 12px;
  border-radius: 20px;
  font-size: 12px;
  font-weight: 600;
}}

/* Omissions */
.omissions li {{
  list-style: none;
  padding: 10px 14px;
  background: rgba(239,68,68,0.1);
  border-radius: 6px;
  margin-bottom: 6px;
  border-left: 3px solid var(--red);
}}
.omissions li::before {{ content: '⚠ '; }}

/* Conflicts */
.conflicts li {{
  list-style: none;
  padding: 10px 14px;
  background: rgba(239,68,68,0.1);
  border-radius: 6px;
  margin-bottom: 6px;
  border-left: 3px solid var(--orange);
}}

/* Topic Grid */
.topic-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 12px; }}
.topic-card {{ background: var(--bg); border-radius: 8px; padding: 14px; }}

/* Methodology Stats */
.methodology-stats {{ display: flex; gap: 20px; margin: 16px 0; }}
.stat {{ text-align: center; }}
.stat-num {{ display: block; font-size: 32px; font-weight: 700; color: var(--accent); }}
.stat-label {{ font-size: 12px; color: var(--text-dim); text-transform: uppercase; }}

/* Thesis */
.thesis {{ font-size: 18px; padding: 14px; background: rgba(59,130,246,0.1); border-radius: 8px; border-left: 4px solid var(--accent); margin-bottom: 16px; }}

/* Bibliography */
.bibliography {{ list-style: none; padding-left: 0; }}
.bibliography li {{ padding: 6px 0; font-size: 14px; border-bottom: 1px solid var(--card-border); }}
.bibliography a {{ color: var(--accent); text-decoration: none; }}

/* Blockquotes */
blockquote {{
  padding: 12px 16px;
  background: var(--bg);
  border-left: 4px solid var(--accent);
  border-radius: 0 8px 8px 0;
  margin-bottom: 10px;
  font-style: italic;
}}

/* Footer */
.footer {{
  text-align: center;
  padding: 30px 0;
  color: var(--text-dim);
  font-size: 13px;
  border-top: 1px solid var(--card-border);
  margin-top: 30px;
}}

/* Responsive */
@media (max-width: 600px) {{
  body {{ padding: 12px; }}
  .header h1 {{ font-size: 22px; }}
  .xref-comparison {{ grid-template-columns: 1fr; }}
  .methodology-stats {{ flex-direction: column; gap: 10px; }}
  .topic-grid {{ grid-template-columns: 1fr; }}
}}

/* Print */
@media print {{
  .card {{ break-inside: avoid; }}
  .card-body {{ display: block !important; }}
  .theme-toggle {{ display: none; }}
  body {{ background: #fff; color: #000; }}
  .card {{ background: #fff; border: 1px solid #ddd; }}
}}
</style>
</head>
<body data-theme="dark">
<button class="theme-toggle" onclick="toggleTheme()">🌙 Dark</button>

<div class="header">
  <span class="mode-badge {mode_class}">{mode_label}</span>
  <h1>{_escape(title)}</h1>
  <p class="channel">Channel: <a href="https://youtube.com/@{_escape(channel)}" target="_blank">{_escape(channel)}</a></p>
  {f'<a href="{_escape(url)}" target="_blank" class="video-link">▶ Watch on YouTube</a>' if url else ''}
  {f'<img class="thumbnail" src="{thumbnail}" alt="Thumbnail">' if thumbnail else ''}
</div>

{''.join(sections)}

<div class="footer">
  Generated by YouTube Research Tool · {mode_label} Mode<br>
  <a href="https://github.com/omnibot007/youtube-research-tool" target="_blank">github.com/omnibot007/youtube-research-tool</a>
</div>

<script>
function toggleSection(header) {{
  const card = header.parentElement;
  card.classList.toggle('collapsed');
}}
function toggleTheme() {{
  const body = document.body;
  const btn = document.querySelector('.theme-toggle');
  if (body.getAttribute('data-theme') === 'dark') {{
    body.setAttribute('data-theme', 'light');
    btn.textContent = '☀ Light';
  }} else {{
    body.setAttribute('data-theme', 'dark');
    btn.textContent = '🌙 Dark';
  }}
}}
</script>
</body>
</html>'''

    if output_path:
        output_path = Path(output_path)
        output_path.write_text(html_doc, encoding="utf-8")

    return html_doc


# ---------------------------------------------------------------- CLI

def _parse_langs(lang_str: str) -> list[str]:
    """Parse comma-separated language string to list."""
    if not lang_str:
        return DEFAULT_LANGS
    return [l.strip() for l in lang_str.split(",") if l.strip()]


def main() -> int:
    p = argparse.ArgumentParser(
        prog="yt_scrape.py",
        description="YouTube scraper — transcripts, search, channels, metadata",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # Common options function
    def add_common_opts(sp):
        sp.add_argument("--lang", default="", help="Subtitle languages (comma-separated, e.g. en,es,fr)")
        sp.add_argument("--all-langs", action="store_true", help="Download all available subtitle languages")
        sp.add_argument("--timestamps", action="store_true", help="Also save timestamped transcript (.tsv)")
        sp.add_argument("--clean", action="store_true", help="Also save a cleaned transcript (filler removal, punctuation, speaker breaks)")
        sp.add_argument("--whisper", action="store_true", help="Fall back to Whisper if no captions")
        sp.add_argument("--whisper-model", default="base", help="Whisper model (tiny/base/small/medium/large)")
        sp.add_argument("--whisper-device", default="cpu", help="Whisper device (cpu/cuda)")
        sp.add_argument("--whisper-compute-type", default="int8", help="Whisper compute type (int8/float16/float32)")
        sp.add_argument("--cookies", default="", help="Cookies file path for age-restricted/members content")
        sp.add_argument("--cookies-from-browser", default="", help="Use cookies from browser (chrome/firefox/edge/brave)")
        sp.add_argument("--proxy", default="", help="Proxy URL for region unlocking")
        sp.add_argument("--retries", type=int, default=DEFAULT_RETRIES, help="Max retries on network errors")
        sp.add_argument("--rate-limit", type=float, default=DEFAULT_RATE_LIMIT, help="Seconds between requests")
        sp.add_argument("--output", default="", help="Custom output directory")
        sp.add_argument("--json", action="store_true", help="Output as JSON")

    sp_t = sub.add_parser("transcript", help="Get transcript for a single video")
    sp_t.add_argument("url", help="YouTube URL or 11-char video ID")
    add_common_opts(sp_t)

    sp_s = sub.add_parser("search", help="Search YouTube")
    sp_s.add_argument("query", help="Search query")
    sp_s.add_argument("--limit", type=int, default=10, help="Max results")
    sp_s.add_argument("--transcripts", action="store_true", help="Also download transcripts")
    add_common_opts(sp_s)

    sp_c = sub.add_parser("channel", help="Scrape a channel's videos")
    sp_c.add_argument("url", help="Channel URL or ID")
    sp_c.add_argument("--limit", type=int, default=10, help="Max videos")
    sp_c.add_argument("--transcripts", action="store_true", help="Also download transcripts")
    add_common_opts(sp_c)

    sp_b = sub.add_parser("batch", help="Batch scrape from a file")
    sp_b.add_argument("file", help="File with one URL/ID per line")
    sp_b.add_argument("--transcripts", action="store_true", help="Download transcripts (default: on)")
    sp_b.set_defaults(transcripts=True)
    add_common_opts(sp_b)

    sp_m = sub.add_parser("metadata", help="Get metadata only")
    sp_m.add_argument("url", help="YouTube URL or 11-char video ID")
    sp_m.add_argument("--cookies", default="", help="Cookies file path")
    sp_m.add_argument("--cookies-from-browser", default="", help="Browser for cookies")
    sp_m.add_argument("--proxy", default="", help="Proxy URL")
    sp_m.add_argument("--json", action="store_true", help="Output as JSON")

    sp_l = sub.add_parser("list", help="List saved transcripts")
    sp_l.add_argument("--output", default="", help="Custom transcripts directory to list")
    sp_l.add_argument("--json", action="store_true", help="Output as JSON")

    sp_r = sub.add_parser("research", help="Light Research — fetch + clean transcript, prepare for AI comprehension")
    sp_r.add_argument("url", help="YouTube URL or 11-char video ID")
    sp_r.add_argument("--cookies", default="", help="Cookies file path")
    sp_r.add_argument("--cookies-from-browser", default="", help="Browser for cookies")
    sp_r.add_argument("--proxy", default="", help="Proxy URL")
    sp_r.add_argument("--retries", type=int, default=DEFAULT_RETRIES, help="Max retries on network errors")
    sp_r.add_argument("--output", default="", help="Custom output directory")
    sp_r.add_argument("--json", action="store_true", help="Output as JSON (default: human-readable)")

    sp_dr = sub.add_parser("deep-research", help="Deep Research — fetch + clean + extract claims + sources, prepare for comprehensive analysis")
    sp_dr.add_argument("url", help="YouTube URL or 11-char video ID")
    sp_dr.add_argument("--cookies", default="", help="Cookies file path")
    sp_dr.add_argument("--cookies-from-browser", default="", help="Browser for cookies")
    sp_dr.add_argument("--proxy", default="", help="Proxy URL")
    sp_dr.add_argument("--retries", type=int, default=DEFAULT_RETRIES, help="Max retries on network errors")
    sp_dr.add_argument("--output", default="", help="Custom output directory")
    sp_dr.add_argument("--json", action="store_true", help="Output as JSON (default: human-readable)")
    sp_dr.add_argument("--visual", action="store_true", help="Extract visual content from video frames (OCR-first, no API key needed. Optional LLM for chart patterns.)")

    sp_read = sub.add_parser("read", help="Read a research report aloud as MP3 (text-to-speech)")
    sp_read.add_argument("report", help="Path to report JSON (_research.json or _deep_research.json)")
    sp_read.add_argument("--voice", default="ava", help="Voice name: ava, aria, emma, andrew, brian, christopher (default: ava)")
    sp_read.add_argument("--rate", default="+0%", help="Speech rate (e.g. +10%% faster, -10%% slower)")
    sp_read.add_argument("--volume", default="+0%", help="Volume adjustment (e.g. +10%% louder)")
    sp_read.add_argument("--output", default="", help="Custom output directory for MP3")
    sp_read.add_argument("--json", action="store_true", help="Output as JSON (default: human-readable)")

    sp_voices = sub.add_parser("voices", help="List available TTS voices")
    sp_voices.add_argument("--json", action="store_true", help="Output as JSON")

    sp_web = sub.add_parser("web", help="Generate interactive HTML report from a research JSON")
    sp_web.add_argument("report", help="Path to report JSON (_research.json or _deep_research.json)")
    sp_web.add_argument("--output", default="", help="Custom output path for HTML file")
    sp_web.add_argument("--open", action="store_true", help="Open in browser after generating")

    args = p.parse_args()

    # Resolve common options
    out_dir = Path(args.output) if hasattr(args, "output") and args.output else None
    langs = _parse_langs(getattr(args, "lang", ""))
    cookies_file = getattr(args, "cookies", "") or None
    cookies_browser = getattr(args, "cookies_from_browser", "") or None
    proxy = getattr(args, "proxy", "") or None
    retries = getattr(args, "retries", DEFAULT_RETRIES)
    rate_limit = getattr(args, "rate_limit", DEFAULT_RATE_LIMIT)
    timestamps = getattr(args, "timestamps", False)
    whisper = getattr(args, "whisper", False)
    whisper_model = getattr(args, "whisper_model", "base")
    whisper_device = getattr(args, "whisper_device", "cpu")
    whisper_ct = getattr(args, "whisper_compute_type", "int8")
    all_langs = getattr(args, "all_langs", False)

    if args.cmd == "transcript":
        result = extract_transcript(
            args.url, langs=langs, output_dir=out_dir, timestamps=timestamps,
            whisper=whisper, whisper_model=whisper_model, whisper_device=whisper_device,
            whisper_compute_type=whisper_ct,
            cookies_file=cookies_file, cookies_from_browser=cookies_browser,
            proxy=proxy, all_langs=all_langs, retries=retries,
        )
        # If --clean flag, also produce a cleaned transcript
        clean_path = ""
        if getattr(args, "clean", False) and result.has_transcript:
            raw = Path(result.transcript_path).read_text(encoding="utf-8", errors="replace")
            header_end = raw.find("=" * 20)
            raw_body = raw[header_end:].lstrip("=\n").strip() if header_end != -1 else raw
            cleaned = clean_transcript_text(raw_body)
            out = out_dir or OUTPUT_DIR
            clean_path = _write_cleaned_transcript(result, cleaned, out)
        if args.json:
            d = result.to_dict()
            if clean_path:
                d["cleaned_transcript_path"] = clean_path
            print(json.dumps(d, ensure_ascii=False, indent=2))
        else:
            if result.error:
                print(f"FAILED: {result.error}", file=sys.stderr)
                if result.error_type:
                    print(f"  Error type: {result.error_type}", file=sys.stderr)
            else:
                print(f"Title:    {result.title}")
                print(f"Channel:  {result.channel}")
                print(f"Duration: {result.duration}s")
                print(f"URL:      {result.url}")
                print(f"Source:   {result.transcript_source}")
                if result.transcript_lang:
                    print(f"Lang:     {result.transcript_lang}")
                print(f"Transcript: {result.transcript_chars} chars -> {result.transcript_path}")
                if result.timestamped_path:
                    print(f"Timestamped: {result.timestamped_path}")
                if clean_path:
                    print(f"Cleaned:   {clean_path}")
        return 1 if result.error else 0

    if args.cmd == "search":
        results = search_videos(
            args.query, limit=args.limit, transcripts=args.transcripts,
            langs=langs, rate_limit_sec=rate_limit, output_dir=out_dir,
            timestamps=timestamps, whisper=whisper,
            cookies_file=cookies_file, cookies_from_browser=cookies_browser,
            proxy=proxy, retries=retries,
        )
        if args.json:
            print(json.dumps([r.to_dict() for r in results], ensure_ascii=False, indent=2))
        else:
            for r in results:
                status = "OK" if r.has_transcript else ("FAIL" if r.error else "meta")
                print(f"  [{status}] {r.title}  ({r.id})")
                if r.transcript_path:
                    print(f"    transcript: {r.transcript_chars} chars")
                if r.error:
                    print(f"    error: {r.error}")
        return 0

    if args.cmd == "channel":
        results = scrape_channel(
            args.url, limit=args.limit, transcripts=args.transcripts,
            langs=langs, rate_limit_sec=rate_limit, output_dir=out_dir,
            timestamps=timestamps, whisper=whisper,
            cookies_file=cookies_file, cookies_from_browser=cookies_browser,
            proxy=proxy, retries=retries,
        )
        if args.json:
            print(json.dumps([r.to_dict() for r in results], ensure_ascii=False, indent=2))
        else:
            for r in results:
                status = "OK" if r.has_transcript else ("FAIL" if r.error else "meta")
                print(f"  [{status}] {r.title}  ({r.id})")
        return 0

    if args.cmd == "batch":
        results = batch_scrape(
            args.file, transcripts=args.transcripts, langs=langs,
            rate_limit_sec=rate_limit, output_dir=out_dir,
            timestamps=timestamps, whisper=whisper,
            cookies_file=cookies_file, cookies_from_browser=cookies_browser,
            proxy=proxy, retries=retries,
        )
        if args.json:
            print(json.dumps([r.to_dict() for r in results], ensure_ascii=False, indent=2))
        else:
            ok = sum(1 for r in results if r.has_transcript)
            fail = sum(1 for r in results if r.error)
            print(f"Done: {ok} OK, {fail} failed, {len(results)} total")
            for r in results:
                status = "OK" if r.has_transcript else "FAIL"
                print(f"  [{status}] {r.title or r.id}")
        return 0

    if args.cmd == "metadata":
        result = extract_metadata(args.url, cookies_file, cookies_browser, proxy)
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            for k, v in result.items():
                print(f"  {k}: {v}")
        return 0 if "error" not in result else 1

    if args.cmd == "list":
        items = list_saved_transcripts(out_dir)
        if args.json:
            print(json.dumps(items, ensure_ascii=False, indent=2))
        else:
            for item in items:
                print(f"  [{item['source']}] {item['title']}  ({item['video_id']})")
                print(f"    {item['file']}")
                print(f"    lang={item['lang']}  size={item['size_bytes']}  modified={item['modified']}")
        return 0

    if args.cmd == "research":
        result = prepare_light_research(
            args.url,
            output_dir=out_dir,
            cookies_file=cookies_file,
            cookies_from_browser=cookies_browser,
            proxy=proxy,
            retries=retries,
        )
        if not result.get("ok"):
            print(f"FAILED: {result.get('error', 'unknown error')}", file=sys.stderr)
            if args.json:
                print(json.dumps(result, ensure_ascii=False, indent=2))
            return 1
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            v = result["video"]
            print(f"{'=' * 70}")
            print(f"  LIGHT RESEARCH — TRANSCRIPT READY FOR COMPREHENSION")
            print(f"{'=' * 70}")
            print(f"  Title:    {v['title']}")
            print(f"  Channel:  {v['channel']}")
            print(f"  URL:      {v['url']}")
            print(f"  Duration: {v['duration']}s")
            print(f"{'─' * 70}")
            print(f"  Raw transcript:      {result['raw_transcript_path']}")
            print(f"  Cleaned transcript:  {result['cleaned_transcript_path']}")
            print(f"  Raw chars:           {result['raw_chars']:,}")
            print(f"  Cleaned chars:       {result['cleaned_chars']:,}")
            print(f"  Reduction:           {result['reduction_pct']}%")
            print(f"{'─' * 70}")
            print(f"  NEXT STEP: Read the cleaned transcript and produce a")
            print(f"  Light Research report (JSON) following the schema.")
            print(f"  Save as: {v['id']}_research.json")
            print(f"{'=' * 70}")
        return 0

    if args.cmd == "deep-research":
        result = prepare_deep_research(
            args.url,
            output_dir=out_dir,
            cookies_file=cookies_file,
            cookies_from_browser=cookies_browser,
            proxy=proxy,
            retries=retries,
            enable_visual=getattr(args, "visual", False),
        )
        if not result.get("ok"):
            print(f"FAILED: {result.get('error', 'unknown error')}", file=sys.stderr)
            if args.json:
                print(json.dumps(result, ensure_ascii=False, indent=2))
            return 1
        if args.json:
            # Don't print the full instructions + schema in JSON mode (too verbose)
            output = {k: v for k, v in result.items() if k not in ("schema", "instructions")}
            print(json.dumps(output, ensure_ascii=False, indent=2))
        else:
            v = result["video"]
            t = result["transcript"]
            print(f"{'=' * 70}")
            print(f"  DEEP RESEARCH — RESEARCH PACKAGE READY")
            print(f"{'=' * 70}")
            print(f"  Title:    {v['title']}")
            print(f"  Channel:  {v['channel']}")
            print(f"  URL:      {v['url']}")
            print(f"  Duration: {v['duration']}s")
            if v.get("view_count"):
                print(f"  Views:    {v['view_count']:,}")
            print(f"{'─' * 70}")
            print(f"  TRANSCRIPT")
            print(f"  Raw:      {t['raw_chars']:,} chars → {t['raw_path']}")
            print(f"  Cleaned:  {t['cleaned_chars']:,} chars → {t['cleaned_path']}")
            print(f"  Reduction: {t['reduction_pct']}%  ({t['paragraph_count']} paragraphs)")
            print(f"{'─' * 70}")
            print(f"  EXTRACTED CLAIMS: {result['claim_count']}")
            if result["claim_count"] > 0:
                print(f"  Claim types found:")
                type_counts: dict[str, int] = {}
                for c in result["extracted_claims"]:
                    for ct in c["claim_types"]:
                        type_counts[ct] = type_counts.get(ct, 0) + 1
                for ct, count in sorted(type_counts.items(), key=lambda x: -x[1]):
                    print(f"    {ct}: {count}")
            print(f"{'─' * 70}")
            print(f"  EXTRACTED SOURCES: {result['source_count']}")
            src = result["extracted_sources"]
            if src["urls"]:
                print(f"  URLs: {len(src['urls'])}")
            if src["books"]:
                print(f"  Books: {len(src['books'])}")
                for b in src["books"][:5]:
                    print(f"    - {b['title']}" + (f" by {b['author']}" if b["author"] else ""))
            if src["papers"]:
                print(f"  Papers: {len(src['papers'])}")
            if src["people"]:
                print(f"  People: {len(src['people'])}")
                for p in src["people"][:5]:
                    print(f"    - {p}")
            print(f"{'─' * 70}")
            # Visual extraction summary
            ve = result.get("visual_extraction", {})
            if ve.get("enabled"):
                print(f"  VISUAL EXTRACTION")
                print(f"  OCR available:    {ve.get('ocr_available', False)}")
                llm_prov = ve.get("llm_provider", "")
                print(f"  LLM provider:     {llm_prov or 'none (OCR only)'}")
                print(f"  Video downloaded: {ve.get('video_downloaded', False)}")
                print(f"  Frames extracted: {ve.get('frames_extracted', 0)}")
                print(f"  After dedup:      {ve.get('frames_after_dedup', 0)}")
                print(f"  OCR analyzed:     {ve.get('frames_analyzed_ocr', 0)}")
                if llm_prov:
                    print(f"  LLM analyzed:     {ve.get('frames_analyzed_llm', 0)}")
                if ve.get("on_screen_texts"):
                    print(f"  On-screen texts:  {len(ve['on_screen_texts'])}")
                if ve.get("chart_patterns"):
                    print(f"  Chart patterns:   {len(ve['chart_patterns'])}")
                if ve.get("visual_claims"):
                    print(f"  Visual claims:    {len(ve['visual_claims'])}")
                print(f"{'─' * 70}")
            print(f"  Research package: {result['research_package_path']}")
            print(f"{'─' * 70}")
            print(f"  NEXT STEP: Deep Research Phase 2 — Devin executes:")
            print(f"  1. Read the cleaned transcript thoroughly")
            print(f"  2. Review + augment extracted claims")
            print(f"  3. Verify EVERY significant claim via web search")
            print(f"  4. Map argument structure + identify fallacies")
            print(f"  5. Assess bias, credibility, conflicts of interest")
            print(f"  6. Cross-reference with authoritative sources")
            print(f"  7. Identify omissions (what's NOT said)")
            print(f"  8. Produce deep research brief JSON")
            print(f"  Save as: {v['id']}_deep_research.json")
            print(f"{'=' * 70}")
        return 0

    if args.cmd == "voices":
        if edge_tts is None:
            print("ERROR: edge-tts not installed. Run: pip install edge-tts", file=sys.stderr)
            return 1
        voices_data = asyncio.run(edge_tts.list_voices())
        en_voices = [v for v in voices_data if v["Locale"].startswith("en-")]
        if args.json:
            print(json.dumps([
                {"short_name": v["ShortName"], "gender": v["Gender"],
                 "locale": v["Locale"], "friendly_name": v["FriendlyName"]}
                for v in en_voices
            ], ensure_ascii=False, indent=2))
        else:
            print(f"{'=' * 70}")
            print(f"  AVAILABLE TTS VOICES (English)")
            print(f"{'=' * 70}")
            # Show preset voices first
            print(f"  Preset shortcuts (use with --voice):")
            for name, voice_id in VOICE_OPTIONS.items():
                print(f"    --voice {name:<14} → {voice_id}")
            print(f"{'─' * 70}")
            print(f"  All English voices:")
            for v in en_voices:
                print(f"    {v['ShortName']:<40} {v['Gender']:<8} {v['FriendlyName']}")
            print(f"{'=' * 70}")
        return 0

    if args.cmd == "read":
        if edge_tts is None:
            print("ERROR: edge-tts not installed. Run: pip install edge-tts", file=sys.stderr)
            return 1

        # Resolve voice
        voice_input = args.voice
        if voice_input in VOICE_OPTIONS:
            voice = VOICE_OPTIONS[voice_input]
        elif voice_input.startswith("en-"):
            voice = voice_input  # Full voice name passed directly
        else:
            print(f"ERROR: Unknown voice '{voice_input}'. Use one of: {', '.join(VOICE_OPTIONS.keys())}", file=sys.stderr)
            return 1

        read_out = Path(args.output) if args.output else None

        result = read_report(
            args.report,
            output_dir=read_out,
            voice=voice,
            rate=args.rate,
            volume=args.volume,
        )

        if not result.get("ok"):
            print(f"FAILED: {result.get('error', 'unknown error')}", file=sys.stderr)
            if args.json:
                print(json.dumps(result, ensure_ascii=False, indent=2))
            return 1

        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print(f"{'=' * 70}")
            print(f"  REPORT READ ALOUD — MP3 GENERATED")
            print(f"{'=' * 70}")
            print(f"  Report:   {result['report_path']}")
            print(f"  Audio:    {result['audio_path']}")
            print(f"{'─' * 70}")
            print(f"  Voice:    {result['voice']}")
            print(f"  Rate:     {result['rate']}")
            print(f"  Volume:   {result['volume']}")
            print(f"{'─' * 70}")
            print(f"  Text length:       {result['text_length']:,} chars")
            print(f"  Audio size:        {result['audio_size_mb']} MB")
            print(f"  Est. duration:     {result['duration_estimate_minutes']} min ({result['duration_estimate_seconds']}s)")
            print(f"{'=' * 70}")
        return 0

    if args.cmd == "web":
        report_path = Path(args.report)
        if not report_path.exists():
            print(f"ERROR: Report file not found: {report_path}", file=sys.stderr)
            return 1

        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            print(f"ERROR: Invalid JSON: {e}", file=sys.stderr)
            return 1

        # Determine output path
        if args.output:
            html_path = Path(args.output)
        else:
            html_path = report_path.with_suffix(".html")

        html_content = report_to_html(report, html_path)

        html_size = html_path.stat().st_size
        print(f"{'=' * 70}")
        print(f"  INTERACTIVE WEB REPORT GENERATED")
        print(f"{'=' * 70}")
        print(f"  Report:  {report_path}")
        print(f"  HTML:    {html_path}")
        print(f"  Size:    {html_size / 1024:.1f} KB")
        print(f"  Mode:    {'Deep Research' if report.get('research_mode') == 'deep' else 'Light Research'}")
        print(f"{'=' * 70}")
        if args.open:
            import webbrowser
            webbrowser.open(str(html_path))
            print(f"  Opened in browser.")
        else:
            print(f"  Open with: --open flag or double-click the HTML file")
        print(f"{'=' * 70}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
