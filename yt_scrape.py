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
from dataclasses import asdict, dataclass
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
    error: str = ""
    error_type: str = ""         # "no_subtitles" | "private" | "age_restricted" | "region_locked" | "rate_limited" | "network" | "unknown"

    def to_dict(self) -> dict:
        return asdict(self)


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

            # Parse new timestamps
            times = line.split("-->")
            current_start = _parse_vtt_time(times[0].strip())
            current_end = _parse_vtt_time(times[1].strip()) if len(times) > 1 else None
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


# ---------------------------------------------------------------- CLAIM EXTRACTION

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
    r'(?:according\s+to|said\s+by|stated\s+by|by)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})',
    re.I,
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
) -> dict:
    """Fetch transcript, clean it, extract claims + sources, and prepare
    a comprehensive research package for Deep Research analysis.

    This is Step 1 of the Deep Research workflow:
    1. This function: fetch + clean + extract claims + extract sources → research package
    2. Devin: reads the package, performs web research, verifies claims,
       maps arguments, assesses bias, cross-references, produces deep research brief

    Quality over speed — no caps on claims. Every significant claim gets verified.
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
    print(f"[2/3] Cleaning transcript + extracting claims...", file=sys.stderr)
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
    cleaned_path = _write_cleaned_transcript(result, cleaned_body, out)

    # Fetch metadata for description (VideoInfo doesn't carry it)
    print(f"[3/3] Extracting claims and sources...", file=sys.stderr)
    description = ""
    try:
        meta = extract_metadata(url_or_id, cookies_file=cookies_file,
                                cookies_from_browser=cookies_from_browser,
                                proxy=proxy, retries=retries)
        description = meta.get("description", "") if isinstance(meta, dict) else ""
    except Exception:
        pass

    # Extract claims and sources from cleaned transcript + description
    claims = extract_claims(cleaned_body)
    sources = extract_sources(cleaned_body, description)

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
        },
        "extracted_claims": claims,
        "extracted_sources": sources,
        "claim_count": len(claims),
        "source_count": sum(len(v) if isinstance(v, list) else 0 for v in sources.values()),
        "schema": DEEP_RESEARCH_SCHEMA,
        "instructions": (
            "DEEP RESEARCH WORKFLOW — QUALITY OVER SPEED\n\n"
            "This is a comprehensive research brief, not a summary. Take as long as needed.\n\n"
            "Phase 2 — Devin executes the following:\n\n"
            "1. READ the cleaned transcript thoroughly.\n"
            "2. REVIEW the extracted_claims list. ADD any claims you identify that regex missed "
            "(implicit claims, contextual claims, nuanced assertions).\n"
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

    sp_read = sub.add_parser("read", help="Read a research report aloud as MP3 (text-to-speech)")
    sp_read.add_argument("report", help="Path to report JSON (_research.json or _deep_research.json)")
    sp_read.add_argument("--voice", default="ava", help="Voice name: ava, aria, emma, andrew, brian, christopher (default: ava)")
    sp_read.add_argument("--rate", default="+0%", help="Speech rate (e.g. +10%% faster, -10%% slower)")
    sp_read.add_argument("--volume", default="+0%", help="Volume adjustment (e.g. +10%% louder)")
    sp_read.add_argument("--output", default="", help="Custom output directory for MP3")
    sp_read.add_argument("--json", action="store_true", help="Output as JSON (default: human-readable)")

    sp_voices = sub.add_parser("voices", help="List available TTS voices")
    sp_voices.add_argument("--json", action="store_true", help="Output as JSON")

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

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
