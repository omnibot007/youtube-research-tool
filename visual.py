"""Visual content extraction for yt-scrape.

Restored 2026-08-04. The original OCR-first pipeline lived in yt_scrape.py and
was lost when an older copy of that file was committed over it (the docs kept
advertising --visual for months afterwards). This version keeps the original
structure and claim types but reads frames with a LOCAL VISION MODEL through
Ollama instead of Tesseract, so no OCR binary is required.

Pipeline:
  1. Download the video at the lowest available quality
  2. Extract frames at chapter boundaries (or fixed intervals)
  3. Drop consecutive near-identical frames (same slide held on screen)
  4. Read each unique frame with the vision model
  5. Parse the recovered text into structured visual claims

Design rules carried over from the rest of the codebase:
  - Nothing here may ever crash a scrape. Every stage is guarded and the
    function always returns a result dict.
  - Progress goes to stderr ONLY. stdout must stay pure so --json parses.
"""

from __future__ import annotations

import base64
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

# Frame extraction interval (seconds) when the video has no chapters.
FRAME_INTERVAL = int(os.environ.get("YT_FRAME_INTERVAL", "60"))
# Hard cap on frames per video. Vision inference is ~45s/frame measured on
# this machine (2026-08-04), so this is a wall-clock guard, not a memory one.
MAX_FRAMES = int(os.environ.get("YT_MAX_FRAMES", "20"))
# Lowest-quality video is plenty for reading slides.
VIDEO_FORMAT = "worst[ext=mp4]/worst"
FRAMES_SUBDIR = "_frames"
# Two frames are "the same slide" above this grayscale similarity.
FRAME_DEDUP_THRESHOLD = 0.95
# Local vision model served by Ollama.
VISION_MODEL = os.environ.get("YT_VISION_MODEL", "ui-tars-7b:latest")
OLLAMA_HOST = os.environ.get("YT_OLLAMA_HOST", "http://127.0.0.1:11434")
VISION_TIMEOUT = int(os.environ.get("YT_VISION_TIMEOUT", "300"))
# Minimum recovered text length for a frame to count as having content.
MIN_TEXT_LEN = 5

# --- Benchmark adoptions (Rounds 1-2, 2026-08-04) ---------------------------
# 768px JPEG input: same accuracy as native PNG on every scored slide, faster
# end to end, and it read punctuation the PNG path missed. prompt_eval_count
# is identical either way (the encoder normalizes internally), so the win is
# decode/transfer, not fewer tokens.
FRAME_MAX_WIDTH = int(os.environ.get("YT_FRAME_MAX_WIDTH", "768"))
FRAME_JPEG_QUALITY = int(os.environ.get("YT_FRAME_JPEG_QUALITY", "85"))
# Real answers run 60-90 tokens; the cap is runaway insurance, nearly free.
VISION_NUM_PREDICT = int(os.environ.get("YT_VISION_NUM_PREDICT", "256"))
# Measured: ctx 2048 loads faster than the shipped default, same accuracy.
VISION_NUM_CTX = int(os.environ.get("YT_VISION_NUM_CTX", "2048"))
# Keep the model resident between frames. Without this a multi-frame video
# can evict and reload the model between every frame -- pure waste.
VISION_KEEP_ALIVE = os.environ.get("YT_VISION_KEEP_ALIVE", "10m")
# Force full GPU offload (Ollama num_gpu). Round 3 verdict (2026-08-04): with
# ctx 2048 the 7B fits the T1000 fully (5.0 GB, VRAM flat across calls) and
# ran 43.5s/frame with sd 1.6s regardless of background CPU load, while the
# scheduler-decides split averaged 80.8s (sd 34.8s) because other processes
# fighting for cores wreck CPU prompt eval. Full offload trades a slightly
# slower quiet-machine floor (~43s vs ~40s) for total immunity to CPU
# contention. Set YT_VISION_NUM_GPU= (empty) to let the scheduler decide.
VISION_NUM_GPU = os.environ.get("YT_VISION_NUM_GPU", "99").strip()

# --- Vision provider selection (added 2026-09-02) ----------------------------
# The local Ollama path is offline and free. On this machine 2026-09-02 it was
# also unusable: the Ollama server answers /api/tags in 3ms but never returns
# from /api/generate, even for a 1.3GB text-only model, so every frame read
# times out. That is a wedged server, not a slow model -- restart Ollama before
# concluding anything about local speed. See KNOWN_ISSUES.md.
# The Gemini backend trades offline-ness for accuracy and speed. "auto" selects
# Gemini only when a key is present, so a machine without one is unaffected.
VISION_PROVIDER = os.environ.get("YT_VISION_PROVIDER", "auto").strip().lower()

# Google Gemini "Interactions API". Schema verified 2026-09-02 against
# ai.google.dev/gemini-api/docs/quickstart, /image-understanding,
# /video-understanding, /api/interactions.md.txt and
# /gemini-api/docs/interactions/structured-output.md.txt.
#   Request : POST {endpoint}, headers x-goog-api-key + Api-Revision,
#             body {"model":..., "input":[{"type":"text",...},{"type":"image",...}]}
#   Response: steps[] -> type "model_output" -> content[] -> type "text" -> text
GEMINI_ENDPOINT = os.environ.get(
    "YT_GEMINI_ENDPOINT",
    "https://generativelanguage.googleapis.com/v1beta/interactions",
)
GEMINI_MODEL = os.environ.get("YT_GEMINI_MODEL", "gemini-3.8-flash")
GEMINI_API_REVISION = os.environ.get("YT_GEMINI_API_REVISION", "2026-05-20")
# Auth style, for third-party gateways that re-expose the native Gemini API.
# "x-goog" (Google, default) | "bearer" | "both".
GEMINI_AUTH_STYLE = os.environ.get("YT_GEMINI_AUTH", "x-goog").strip().lower()
# Some proxies reject an Api-Revision header they do not recognise.
GEMINI_SEND_REVISION = os.environ.get(
    "YT_GEMINI_SEND_REVISION", "1").strip() not in ("0", "false", "no")
GEMINI_TIMEOUT = int(os.environ.get("YT_GEMINI_TIMEOUT", "180"))
GEMINI_VIDEO_TIMEOUT = int(os.environ.get("YT_GEMINI_VIDEO_TIMEOUT", "600"))
GEMINI_MAX_TOKENS = int(os.environ.get("YT_GEMINI_MAX_TOKENS", "1024"))
# Video mode answers for a whole window, so it needs far more room than a
# single frame. At 1024*4 a 30-minute window truncated mid-answer.
GEMINI_VIDEO_MAX_TOKENS = int(
    os.environ.get("YT_GEMINI_VIDEO_MAX_TOKENS", "8192"))
# Window size for clipped analysis. Video costs ~100 tokens/second at the
# default low media resolution, so ~3h is the 1M-context ceiling for ONE
# request; 30min windows stay far clear of it and keep each answer dense.
GEMINI_VIDEO_CHUNK_SECONDS = int(
    os.environ.get("YT_GEMINI_VIDEO_CHUNK_SECONDS", "1800"))
# Frames are cheap to Gemini, so the local frame cap is not the binding
# constraint it is for Ollama. Kept separate so raising one cannot slow the other.
GEMINI_TARGET_FRAMES = int(os.environ.get("YT_GEMINI_TARGET_FRAMES", "16"))

# Enforced output contract. structured-output.md.txt documents response_format
# as {"type":"text","mime_type":"application/json","schema":<JSON Schema>}, which
# makes the two-key contract a server-side guarantee instead of a polite request.
GEMINI_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "on_screen_text": {"type": "string"},
        "chart_description": {"type": "string"},
    },
    "required": ["on_screen_text", "chart_description"],
}


_FFMPEG_CMD = ""

VISION_PROMPT = (
    "You are reading a single frame from an educational video. "
    "Respond with JSON only, no commentary, using exactly these two keys:\n"
    '{"on_screen_text": "every line of text visible in the image, verbatim, '
    'separated by \\n. Empty string if there is no text.", '
    '"chart_description": "one sentence describing any chart, graph, diagram '
    'or drawing. Empty string if there is none."}'
)


def find_ffmpeg() -> str:
    """Locate an ffmpeg binary. Falls back to the imageio-ffmpeg wheel.

    The original code assumed ffmpeg and ffprobe were both on PATH. Neither is
    on this machine, so we resolve a real path and avoid ffprobe entirely
    (duration comes from the video metadata we already fetched).
    """
    global _FFMPEG_CMD
    if _FFMPEG_CMD:
        return _FFMPEG_CMD
    found = shutil.which("ffmpeg")
    if found:
        _FFMPEG_CMD = found
        return found
    try:
        import imageio_ffmpeg

        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and Path(exe).exists():
            _FFMPEG_CMD = str(exe)
            return _FFMPEG_CMD
    except Exception:
        pass
    return ""


VISUAL_MODE = os.environ.get("YT_VISUAL_MODE", "auto").strip().lower()


def vision_available() -> tuple[bool, str]:
    """Check the active provider is usable. Returns (available, detail)."""
    if resolve_provider() == "gemini":
        if not gemini_key():
            return False, (
                "no Gemini API key -- set GEMINI_API_KEY "
                "(free key at aistudio.google.com/apikey)"
            )
        return True, GEMINI_MODEL
    return _vision_available_ollama()


def _vision_available_ollama() -> tuple[bool, str]:
    """Check that Ollama is up and the configured vision model is installed.

    Returns (available, detail). `detail` is the model name on success or a
    human-readable reason on failure.
    """
    url = OLLAMA_HOST.rstrip("/") + "/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            tags = json.loads(r.read().decode("utf-8"))
    except Exception as e:
        return False, f"Ollama not reachable at {OLLAMA_HOST} ({e})"

    models = tags.get("models") or []
    names = [str(m.get("name", "")) for m in models]
    if VISION_MODEL not in names:
        return False, (
            f"vision model {VISION_MODEL!r} is not installed "
            f"(available: {', '.join(names[:6]) or 'none'})"
        )

    # Confirm the model actually declares vision support; a text-only model
    # would return confident nonsense about images it cannot see.
    for m in models:
        if str(m.get("name", "")) == VISION_MODEL:
            caps = m.get("capabilities") or []
            if caps and "vision" not in caps:
                return False, (
                    f"model {VISION_MODEL!r} does not declare vision support "
                    f"(capabilities: {', '.join(str(c) for c in caps)})"
                )
            break
    return True, VISION_MODEL


def download_video_low_quality(
    video_id: str,
    output_dir: Path,
    cookies_file: str | None = None,
    cookies_from_browser: str | None = None,
    proxy: str | None = None,
) -> Path | None:
    """Download the video at lowest quality for frame extraction."""
    video_path = output_dir / f"{video_id}_video.mp4"
    if video_path.exists() and video_path.stat().st_size > 1000:
        return video_path

    try:
        import yt_dlp
    except Exception as e:
        print(f"  [visual] yt_dlp unavailable: {e}", file=sys.stderr)
        return None

    opts: dict[str, Any] = {
        "format": VIDEO_FORMAT,
        "outtmpl": str(video_path),
        "quiet": True,
        "no_warnings": True,
        "skip_download": False,
        # Progress bars must never touch stdout (--json stays parseable).
        "noprogress": True,
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


def extract_frames(
    video_path: Path,
    output_dir: Path,
    duration: int = 0,
    interval: int = FRAME_INTERVAL,
    max_frames: int = MAX_FRAMES,
    timestamps: list[float] | None = None,
) -> list[Path]:
    """Extract frames as PNGs. Returns the frame paths that were written."""
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        print(
            "  [visual] ffmpeg not found — install it or `pip install imageio-ffmpeg`",
            file=sys.stderr,
        )
        return []

    frames_dir = output_dir / FRAMES_SUBDIR
    frames_dir.mkdir(parents=True, exist_ok=True)
    for old in frames_dir.glob("*.png"):
        try:
            old.unlink()
        except Exception:
            pass

    if timestamps:
        ts_list = [float(t) for t in timestamps][:max_frames]
    else:
        if duration <= 0:
            # Without a duration we cannot space frames; take the first one.
            ts_list = [0.0]
        else:
            count = max(1, min(max_frames, int(duration // interval) or 1))
            ts_list = [float(i * interval) for i in range(count)]
            ts_list = [t for t in ts_list if t < duration] or [0.0]

    frames: list[Path] = []
    for i, ts in enumerate(ts_list):
        name = f"frame_{i:04d}_{int(ts // 60):02d}m{int(ts % 60):02d}s.png"
        frame_path = frames_dir / name
        try:
            subprocess.run(
                [
                    ffmpeg, "-nostdin", "-loglevel", "error",
                    "-ss", str(ts), "-i", str(video_path),
                    "-frames:v", "1", "-q:v", "2", "-y", str(frame_path),
                ],
                capture_output=True,
                timeout=60,
            )
        except Exception as e:
            print(f"  [visual] Frame at {ts}s failed: {e}", file=sys.stderr)
            continue
        if frame_path.exists() and frame_path.stat().st_size > 1000:
            frames.append(frame_path)

    return frames


def frames_are_similar(frame1: Path, frame2: Path) -> bool:
    """True when two frames are near-identical (same slide held on screen)."""
    try:
        from PIL import Image

        a = list(Image.open(frame1).convert("L").resize((64, 64)).getdata())
        b = list(Image.open(frame2).convert("L").resize((64, 64)).getdata())
        if len(a) != len(b) or not a:
            return False
        diff = sum(abs(x - y) for x, y in zip(a, b))
        similarity = 1 - (diff / (255 * len(a)))
        return similarity >= FRAME_DEDUP_THRESHOLD
    except Exception:
        return False


# --- Round 4: fewer frames beat faster frames --------------------------------
# Vision inference is ~95% of --visual wall time, so the pipeline now spends
# pennies on pixels to avoid dollars on the model:
#   1. GLOBAL perceptual dedup (dHash): a slide that reappears minutes later
#      no longer re-enters the queue (the old dedup only compared neighbors).
#   2. Content-density ranking: JPEG-compressed size of a downscaled grayscale
#      frame is a cheap proxy for on-screen text/structure. Blank transitions
#      and plain talking-head shots compress tiny and rank last. No OCR dep.
#   3. Adaptive cap: at most YT_VISUAL_TARGET_FRAMES (default 8) frames reach
#      the model, densest first, then re-sorted chronologically.
TARGET_FRAMES = int(os.environ.get("YT_VISUAL_TARGET_FRAMES", "8"))
FRAME_HASH_DISTANCE = int(os.environ.get("YT_FRAME_HASH_DISTANCE", "6"))


def frame_dhash(frame_path: Path) -> int:
    """64-bit difference hash of a frame (0 when unreadable)."""
    try:
        from PIL import Image

        im = Image.open(frame_path).convert("L").resize((9, 8))
        px = list(im.getdata())
        bits = 0
        for row in range(8):
            for col in range(8):
                bits = (bits << 1) | (1 if px[row * 9 + col] > px[row * 9 + col + 1] else 0)
        return bits
    except Exception:
        return 0


def hash_distance(a: int, b: int) -> int:
    """Hamming distance between two 64-bit frame hashes."""
    return bin(a ^ b).count("1")


def frame_density(frame_path: Path) -> int:
    """Content-density proxy: JPEG byte size of a 256px grayscale render.

    Text-heavy slides keep hard edges after downscaling and compress LARGE;
    blank transitions and soft webcam shots compress SMALL.
    """
    try:
        import io

        from PIL import Image

        im = Image.open(frame_path).convert("L")
        if im.width > 256:
            im = im.resize((256, max(1, int(im.height * 256 / im.width))))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=60)
        return buf.tell()
    except Exception:
        return 0


def select_frames(frames: list[Path], target: int = 0) -> list[Path]:
    """Global dedup + density-ranked adaptive cap. Chronological output.

    Unlike deduplicate_frames (neighbors only), a frame is dropped when it is
    perceptually close to ANY kept frame, then only the `target` densest
    frames survive (default TARGET_FRAMES; negative target disables the cap).
    Dropped frames are unlinked so the audit dir matches the vision queue.
    """
    if target == 0:
        target = TARGET_FRAMES
    unique: list = []
    for frame in frames:
        h = frame_dhash(frame)
        dup = any(
            h and kept_hash and hash_distance(h, kept_hash) <= FRAME_HASH_DISTANCE
            for _, kept_hash in unique
        )
        if not dup and unique and frames_are_similar(frame, unique[-1][0]):
            dup = True  # pixel-diff backstop for hash near-misses
        if dup:
            try:
                frame.unlink()
            except Exception:
                pass
            continue
        unique.append((frame, h))

    kept = [f for f, _ in unique]
    if target > 0 and len(kept) > target:
        ranked = sorted(kept, key=frame_density, reverse=True)
        chosen = set(ranked[:target])
        dropped = [f for f in kept if f not in chosen]
        for frame in dropped:
            try:
                frame.unlink()
            except Exception:
                pass
        kept = [f for f in kept if f in chosen]
    return kept


def deduplicate_frames(frames: list[Path]) -> list[Path]:
    """Drop consecutive duplicate frames, keeping the first of each run."""
    if len(frames) <= 1:
        return list(frames)
    kept: list[Path] = [frames[0]]
    for i in range(1, len(frames)):
        if frames_are_similar(frames[i], kept[-1]):
            try:
                frames[i].unlink()
            except Exception:
                pass
        else:
            kept.append(frames[i])
    return kept


def _encode_frame(frame_path: Path) -> str:
    """Base64-encode a frame for the vision model.

    Downscales to FRAME_MAX_WIDTH and re-encodes as JPEG (measured: same
    accuracy as native PNG, faster end to end). Falls back to the raw file
    bytes if Pillow is unavailable -- encoding must never crash a scrape.
    """
    raw = frame_path.read_bytes()
    try:
        import io

        from PIL import Image

        im = Image.open(io.BytesIO(raw)).convert("RGB")
        if FRAME_MAX_WIDTH and im.width > FRAME_MAX_WIDTH:
            h = max(1, int(im.height * (FRAME_MAX_WIDTH / im.width)))
            im = im.resize((FRAME_MAX_WIDTH, h), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=FRAME_JPEG_QUALITY)
        raw = buf.getvalue()
    except Exception:
        pass
    return base64.b64encode(raw).decode("ascii")


# --- Trading profile (added 2026-09-02) -------------------------------------
# The general prompt asks for "one sentence describing any chart", which on a
# TradingView screenshot yields "a chart with lines on it" -- true, useless.
# This one names the fields a trader needs and forbids guessing numbers, which
# is the characteristic failure of small vision models on dense chart frames.
TRADING_PROMPT = (
    "You are reading a single frame from a TRADING education video. "
    "Report ONLY what is literally visible. Never guess a number you cannot "
    "actually read; omit it instead. "
    "Respond with JSON only, no commentary, using exactly these two keys:\n"
    '{"on_screen_text": "every line of text visible in the image, verbatim, '
    'separated by \\n. Include ticker symbols, the selected timeframe, '
    'indicator names with their settings (for example RSI 14, EMA 200), '
    'price-axis values, and any text labels drawn on the chart. '
    'Empty string if there is no text.", '
    '"chart_description": "if a price chart is visible, describe it in this '
    'order: chart type (candlestick, line, bar, Heikin Ashi); instrument and '
    'timeframe if shown; trend direction; each indicator overlay or sub-panel '
    'you can name, with its settings if displayed; horizontal levels, '
    'trendlines, zones, Fibonacci retracements, arrows or annotations drawn '
    'on it; and any named candlestick or price pattern. '
    'Empty string if there is no chart."}'
)

# Whole-video prompt for the Gemini YouTube-URL path. One request replaces
# download + ffmpeg + dedup + N frame calls.
VIDEO_PROMPT = (
    "You are analysing a TRADING education video. Watch the whole video and "
    "report ONLY what is literally shown on screen. Never guess a number you "
    "cannot read. Respond with JSON only, no commentary, using exactly these "
    "two keys:\n"
    '{"on_screen_text": "the on-screen text that matters, grouped by moment '
    'and prefixed with its MM:SS timestamp, separated by \\n. Include ticker '
    'symbols, timeframes, indicator names with settings, price levels and '
    'labels drawn on charts.", '
    '"chart_description": "for each distinct chart shown, one line prefixed '
    'with its MM:SS timestamp giving: chart type, instrument and timeframe, '
    'trend direction, named indicators with settings, drawn levels or '
    'trendlines or zones, and any named pattern."}'
)

VISION_PROFILE = os.environ.get("YT_VISION_PROFILE", "trading").strip().lower()


def active_prompt(profile: str = "") -> str:
    """Return the frame prompt for the active profile.

    Defaults to the trading profile: this tool is pointed at trading videos and
    the general prompt measurably under-describes chart frames. Set
    YT_VISION_PROFILE=general for the pre-2026-09-02 wording.
    """
    name = (profile or VISION_PROFILE or "trading").strip().lower()
    return VISION_PROMPT if name == "general" else TRADING_PROMPT


# --- Gemini backend ---------------------------------------------------------

def gemini_key() -> str:
    """Return the Gemini API key from the environment, or "".

    Never logged, never returned in an error string. Checked in priority order
    so a yt-scrape-specific key can override a machine-wide one.
    """
    for name in ("YT_GEMINI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return ""


def resolve_provider() -> str:
    """Decide which backend read_frame should use.

    "auto" (the default) selects Gemini only when a key is present, so a
    machine with no key behaves exactly as it did before this backend existed.
    """
    choice = (VISION_PROVIDER or "auto").strip().lower()
    if choice == "gemini":
        return "gemini"
    if choice == "ollama":
        return "ollama"
    return "gemini" if gemini_key() else "ollama"


def _gemini_extract_text(body: dict) -> str:
    """Pull the model's text out of an Interactions API response.

    Current schema (verified 2026-09-02, /api/interactions.md.txt):
        steps[] -> {"type": "model_output", "content": [{"type":"text","text":...}]}
    The May-2026 migration guide documents a legacy shape with an `outputs`
    array; both are accepted so a revision pin change cannot silently blank the
    reading.
    """
    for step in body.get("steps") or []:
        if not isinstance(step, dict) or step.get("type") != "model_output":
            continue
        for chunk in step.get("content") or []:
            if isinstance(chunk, dict) and chunk.get("type") == "text":
                text = chunk.get("text")
                if text:
                    return str(text)
    for out in reversed(body.get("outputs") or []):
        if isinstance(out, dict) and out.get("text"):
            return str(out["text"])
    if body.get("output_text"):
        return str(body["output_text"])
    return ""


def _gemini_headers(key: str) -> dict:
    """Build request headers for Google or for a compatible proxy.

    Google's own endpoint authenticates with `x-goog-api-key`. Third-party
    gateways that re-expose the native Gemini API (AiHubMix and similar) mostly
    accept the same header, but some want `Authorization: Bearer`. Set
    YT_GEMINI_AUTH=bearer for those, or YT_GEMINI_AUTH=both to send both, which
    is harmless when only one is read.

    Api-Revision pins Google's schema revision. A proxy that does not recognise
    it can reject the whole request, so YT_GEMINI_SEND_REVISION=0 drops it.
    The key is never logged, and never appears in any error string.
    """
    headers = {"Content-Type": "application/json"}
    style = GEMINI_AUTH_STYLE
    if style in ("bearer", "both"):
        headers["Authorization"] = f"Bearer {key}"
    if style in ("x-goog", "both"):
        headers["x-goog-api-key"] = key
    if GEMINI_SEND_REVISION:
        headers["Api-Revision"] = GEMINI_API_REVISION
    return headers


def _gemini_post(parts: list, timeout: int, max_tokens: int = 0) -> tuple:
    """POST one Interactions request. Returns (text, error). Never raises.

    Retries once with the optional blocks stripped if the server rejects the
    request, so an unrecognised generation_config or response_format key
    degrades to a plain call instead of failing the frame outright.
    """
    key = gemini_key()
    if not key:
        return "", (
            "no Gemini API key -- set GEMINI_API_KEY "
            "(get one free at aistudio.google.com/apikey)"
        )

    full = {
        "model": GEMINI_MODEL,
        "input": parts,
        "generation_config": {
            "temperature": 0,
            "max_output_tokens": max_tokens or GEMINI_MAX_TOKENS,
        },
        "response_format": {
            "type": "text",
            "mime_type": "application/json",
            "schema": GEMINI_JSON_SCHEMA,
        },
    }
    minimal = {"model": GEMINI_MODEL, "input": parts}

    last_error = ""
    for attempt, payload in enumerate((full, minimal)):
        try:
            req = urllib.request.Request(
                GEMINI_ENDPOINT,
                data=json.dumps(payload).encode("utf-8"),
                headers=_gemini_headers(key),
            )
            with urllib.request.urlopen(req, timeout=timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = e.read().decode("utf-8", "replace")[:300]
            except Exception:
                pass
            last_error = f"HTTP {e.code}: {detail or e.reason}"
            # 4xx on the rich payload is worth one plain retry; 401/403/429 are
            # not about the payload at all, so stop immediately.
            if e.code in (401, 403, 404, 429) or attempt == 1:
                return "", last_error
            continue
        except Exception as e:
            return "", f"{type(e).__name__}: {e}"

        text = _gemini_extract_text(body)
        if text:
            return text, ""
        status = body.get("status") or "no text in response"
        last_error = f"empty model_output (status: {status})"
        if attempt == 1:
            return "", last_error
    return "", last_error


def _parse_vision_json(answer: str) -> tuple:
    """Split a model reply into (on_screen_text, chart_description).

    Falls back to treating the whole reply as on-screen text when the model
    ignores the JSON instruction, rather than discarding a good reading.
    """
    parsed = None
    try:
        parsed = json.loads(answer)
    except Exception:
        match = re.search(r"\{.*\}", answer, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group())
            except Exception:
                parsed = None
    if isinstance(parsed, dict):
        return (
            str(parsed.get("on_screen_text") or ""),
            str(parsed.get("chart_description") or ""),
        )
    return answer, ""


def _finish_reading(out: dict, text: str, chart: str) -> dict:
    """Normalise text/chart into the reading dict every caller expects."""
    text = re.sub(r"\n{3,}", "\n\n", text.replace("\\n", "\n")).strip()
    chart = re.sub(r"\n{3,}", "\n\n", chart.replace("\\n", "\n")).strip()
    out["text"] = text
    out["lines"] = [l.strip() for l in text.split("\n") if l.strip()]
    out["chart"] = chart
    out["has_content"] = len(text) >= MIN_TEXT_LEN or len(chart) >= MIN_TEXT_LEN
    return out


def _read_frame_gemini(frame_path: Path, timestamp: float,
                       prompt: str = "") -> dict:
    """Read one frame with Gemini. Never raises."""
    out = {
        "text": "",
        "lines": [],
        "chart": "",
        "has_content": False,
        "timestamp": timestamp,
        "frame_file": frame_path.name,
        "provider": "gemini",
        "model": GEMINI_MODEL,
    }
    try:
        encoded = _encode_frame(frame_path)
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
        return out

    parts = [
        {"type": "text", "text": prompt or active_prompt()},
        {"type": "image", "data": encoded, "mime_type": "image/jpeg"},
    ]
    answer, error = _gemini_post(parts, GEMINI_TIMEOUT)
    if error:
        out["error"] = error
        return out
    text, chart = _parse_vision_json(answer.strip())
    return _finish_reading(out, text, chart)


def _hhmmss(seconds: float) -> str:
    """Absolute position label. Long videos need hours, not MM:SS."""
    s = max(0, int(seconds))
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def analyze_youtube_video(url: str, prompt: str = "", start_s: float = 0.0,
                          end_s: float = 0.0) -> dict:
    """Analyse a YouTube video, or one time window of it, in one request.

    Schema verified 2026-09-02 against /gemini-api/docs/video-understanding:
    a video part is {"type": "video", "uri": "<youtube url>"}; mime_type is not
    required and only public videos are supported.

    Clipping (measured 2026-09-02, NOT documented for YouTube URLs -- the docs
    show it only for uploaded files, so this was probed directly):
        {"processing": {"type": "static",
                        "start_offset": "7200s", "end_offset": "7320s"}}
    works on a YouTube URL, but ONLY with duration STRINGS. Integer
    milliseconds -- the form the docs use for uploaded files -- returns
    HTTP 400 "Invalid input at 'input[1].processing'".

    Video costs roughly 100 tokens per second at the default low media
    resolution, so a 1M-context model reaches about 3 hours in one request.
    Windowing is what makes a 9-hour course tractable.
    """
    out = {
        "text": "",
        "lines": [],
        "chart": "",
        "has_content": False,
        "timestamp": float(start_s or 0.0),
        "frame_file": "",
        "provider": "gemini",
        "model": GEMINI_MODEL,
        "mode": "video",
        "window_start": float(start_s or 0.0),
        "window_end": float(end_s or 0.0),
    }

    video_part: dict[str, Any] = {"type": "video", "uri": url}
    instruction = prompt or VIDEO_PROMPT
    if end_s and end_s > start_s:
        video_part["processing"] = {
            "type": "static",
            "start_offset": f"{int(start_s)}s",
            "end_offset": f"{int(end_s)}s",
        }
        # The model sees only the clip, so its own clock starts at zero. Give
        # it the window's absolute position and demand absolute stamps back,
        # otherwise every window collides at 00:00 when merged.
        instruction = (
            f"This clip starts at {_hhmmss(start_s)} of a longer video and "
            f"ends at {_hhmmss(end_s)}. Prefix every line with its ABSOLUTE "
            f"HH:MM:SS position in the FULL video, not a position within this "
            f"clip.\n\n" + instruction
        )

    parts = [{"type": "text", "text": instruction}, video_part]
    answer, error = _gemini_post(
        parts, GEMINI_VIDEO_TIMEOUT, max_tokens=GEMINI_VIDEO_MAX_TOKENS,
    )
    if error:
        out["error"] = error
        return out
    text, chart = _parse_vision_json(answer.strip())
    return _finish_reading(out, text, chart)


def video_windows(duration: float, chunk: int = 0) -> list:
    """Split a duration into [(start, end)] windows for clipped analysis.

    Returns [] when the video fits in one request, so the caller sends no
    processing block at all and behaviour is unchanged for short videos.
    A trailing sliver is folded into the previous window rather than becoming
    a near-empty request.
    """
    size = int(chunk or GEMINI_VIDEO_CHUNK_SECONDS)
    if size <= 0 or duration <= 0 or duration <= size:
        return []
    edges = []
    start = 0
    while start < duration:
        end = min(start + size, int(duration))
        if duration - end < size * 0.2:      # fold a short tail into this one
            end = int(duration)
        edges.append((float(start), float(end)))
        if end >= duration:
            break
        start = end
    return edges


def read_frame(frame_path: Path, timestamp: float, prompt: str = "") -> dict:
    """Read one frame with the active vision provider. Never raises."""
    if resolve_provider() == "gemini":
        return _read_frame_gemini(frame_path, timestamp, prompt)
    return _read_frame_ollama(frame_path, timestamp, prompt)


def _read_frame_ollama(frame_path: Path, timestamp: float,
                       prompt: str = "") -> dict:
    """Read one frame with the local vision model.

    Never raises. Returns a dict with text / lines / chart / has_content, and
    an `error` key when the model could not be reached or parsed.
    """
    out: dict[str, Any] = {
        "text": "",
        "lines": [],
        "chart": "",
        "has_content": False,
        "timestamp": timestamp,
        "frame_file": frame_path.name,
        "provider": "ollama",
        "model": VISION_MODEL,
    }
    try:
        options: dict[str, Any] = {
            "temperature": 0,
            "num_predict": VISION_NUM_PREDICT,
            "num_ctx": VISION_NUM_CTX,
        }
        if VISION_NUM_GPU:
            try:
                options["num_gpu"] = int(VISION_NUM_GPU)
            except ValueError:
                pass
        payload = {
            "model": VISION_MODEL,
            "prompt": prompt or active_prompt(),
            "images": [_encode_frame(frame_path)],
            "stream": False,
            "options": options,
            "keep_alive": VISION_KEEP_ALIVE,
        }
        req = urllib.request.Request(
            OLLAMA_HOST.rstrip("/") + "/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=VISION_TIMEOUT) as r:
            body = json.loads(r.read().decode("utf-8"))
        answer = (body.get("response") or "").strip()
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
        return out

    text = ""
    chart = ""
    parsed: dict[str, Any] | None = None
    try:
        parsed = json.loads(answer)
    except Exception:
        match = re.search(r"\{.*\}", answer, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group())
            except Exception:
                parsed = None

    if isinstance(parsed, dict):
        text = str(parsed.get("on_screen_text") or "")
        chart = str(parsed.get("chart_description") or "")
    else:
        # Model ignored the JSON instruction; treat the whole reply as text
        # rather than throwing away a good reading.
        text = answer

    text = re.sub(r"\n{3,}", "\n\n", text.replace("\\n", "\n")).strip()
    out["text"] = text
    out["lines"] = [l.strip() for l in text.split("\n") if l.strip()]
    out["chart"] = chart.strip()
    out["has_content"] = len(text) >= MIN_TEXT_LEN or len(out["chart"]) >= MIN_TEXT_LEN
    return out


def extract_visual_claims(text: str, timestamp: float,
                         source: str = "visual_vision") -> list[dict]:
    """Parse recovered on-screen text into structured visual claims.

    Claim types match the originals: bullet_point, indicator_setting,
    indicator_threshold, visual_definition, price_level, percentage.
    Added 2026-08-04: timeframe, plus a wider indicator-setting vocabulary
    (Multiplier, Threshold, Target, Win rate...) -- Round 2 showed the model
    reading these values correctly while the parser silently dropped them.
    """
    claims: list[dict] = []
    timestamp_str = f"{int(timestamp // 60):02d}:{int(timestamp % 60):02d}"
    lines = [l.strip() for l in text.split("\n") if l.strip()]

    def add(claim: str, ctype: str) -> None:
        claims.append({
            "timestamp": timestamp,
            "timestamp_str": timestamp_str,
            "claim": claim,
            "source": source,
            "type": ctype,
        })

    for line in lines:
        m = re.match(r"^[.\u2022\-\*]\s*(.+)", line) or re.match(r"^\d+[.)]\s*(.+)", line)
        if m and len(m.group(1).strip()) > 3:
            add(m.group(1).strip(), "bullet_point")

    for line in lines:
        m = re.search(
            r"(?i)\b(RSI|RSI length|MA|SMA|EMA|MACD|Stochastic|ATR Multiplier|"
            r"ATR|Bollinger|Length|Period|Standard Deviation|Risk Reward|"
            r"Stop Loss|Take Profit|Multiplier|Threshold|Factor|Smoothing|"
            r"Lookback|Target|Win rate needed|Win rate|Risk per trade|"
            r"Max daily loss)"
            r"\s*[:=]\s*(\d+(?:\.\d+)?)",
            line,
        )
        if m:
            add(f"{m.group(1)} = {m.group(2)}", "indicator_setting")

    for line in lines:
        m = re.search(
            r"(?i)\b(RSI|MACD|Stochastic|ATR|MA|SMA|EMA|OBV|ADX)"
            r"\s+(?:is\s+)?(above|below|over|under|>|<)\s*(\d+(?:\.\d+)?)",
            line,
        )
        if m:
            add(f"{m.group(1)} {m.group(2)} {m.group(3)}", "indicator_threshold")

    for line in lines:
        m = re.search(
            r"(?i)\b(Momentum|Divergence|Overbought|Oversold|Bullish|Bearish|"
            r"Support|Resistance|Trendline|Breakout|Reversal)"
            r"\s*[=:]\s*(.{5,80})",
            line,
        )
        if m:
            definition = m.group(2).strip().rstrip(".,;|")
            if len(definition) > 5:
                add(f"{m.group(1)} = {definition}", "visual_definition")

    seen_tf: set[str] = set()
    for line in lines:
        for m in re.finditer(
            r"(?i)\b(?:timeframe|time frame|tf|chart)\s*[:=]\s*"
            r"(\d{1,3}\s*(?:m|min|minutes?|h|hr|hours?|d|days?|w|weeks?)\b)",
            line,
        ):
            tf = m.group(1).replace(" ", "").upper()
            if tf not in seen_tf:
                seen_tf.add(tf)
                add(f"Timeframe: {tf}", "timeframe")
        # Standalone tokens like "4H" / "15M" (uppercase only, and never right
        # after a currency sign or digit -- "$100M" is money, not a timeframe).
        for m in re.finditer(r"(?<![$\u20ac\u00a3\d.,])\b(\d{1,3}[MHDW])\b", line):
            tf = m.group(1)
            if tf not in seen_tf:
                seen_tf.add(tf)
                add(f"Timeframe: {tf}", "timeframe")

    for line in lines:
        for m in re.finditer(r"[$\u20ac\u00a3]\s*(\d+(?:,\d{3})*(?:\.\d+)?)", line):
            add(f"Price level: {m.group(0)}", "price_level")

    for line in lines:
        for m in re.finditer(r"(\d+(?:\.\d+)?)\s*%", line):
            pct = m.group(0)
            if not any(
                c["claim"].endswith(pct)
                for c in claims
                if c["type"] == "indicator_setting"
            ):
                add(f"Percentage shown: {pct}", "percentage")

    # --- Prose-chart vocabulary (added 2026-09-02) --------------------------
    # The blocks above were written for slide text ("RSI Length: 14"). A
    # chart_description is prose ("RSI(14) sub-panel", "a 200-period EMA"), and
    # measured 2026-09-02 the parser dropped every setting written that way.
    _IND = (r"RSI|MACD|Stochastic|Stoch|ATR|EMA|SMA|WMA|MA|Bollinger|"
            r"ADX|OBV|VWAP|CCI|Ichimoku|SuperTrend")

    for line in lines:
        # "RSI(14)", "EMA (200)"
        for m in re.finditer(rf"(?i)\b({_IND})\s*\(\s*(\d+(?:\.\d+)?)\s*\)", line):
            add(f"{m.group(1).upper()} = {m.group(2)}", "indicator_setting")
        # "200-period EMA", "14 period RSI", "50-day MA".
        # The (?<![:\d]) guard is load-bearing: video mode prefixes every line
        # with MM:SS, and without it "00:29 RSI Settings" parsed as RSI = 29.
        # Measured on a real run 2026-09-02 -- 4 of 11 indicator_settings were
        # clock times. Timestamps are data about the line, not content in it.
        for m in re.finditer(
            rf"(?i)(?<![:\d])\b(\d+)[-\s]*(?:period|day|bar|length)?[-\s]+"
            rf"({_IND})\b", line
        ):
            add(f"{m.group(2).upper()} = {m.group(1)}", "indicator_setting")
        # "RSI 14", "RSI Length 14", "MA Length 14" -- the on-screen settings
        # panel writes these WITHOUT a colon, so the [:=]-anchored block above
        # missed every one of them. Requires the number to follow the
        # indicator directly (optionally via Length/Period), so "RSI Upper
        # Band 70" stays a threshold rather than becoming a setting.
        for m in re.finditer(
            rf"(?i)\b({_IND})\s+(?:length|period|len)?\s*[:=]?\s*(\d{{1,4}})\b",
            line,
        ):
            add(f"{m.group(1).upper()} = {m.group(2)}", "indicator_setting")

    for line in lines:
        # "RSI ... oversold below 30" -- allow a few words between the
        # indicator and its comparison, which prose always has.
        for m in re.finditer(
            rf"(?i)\b({_IND})\b(?:\s*\(\d+\))?(?:\W+\w+){{0,4}}?\W+"
            r"(above|below|over|under|crosses above|crosses below)\s+"
            r"(\d+(?:\.\d+)?)",
            line,
        ):
            claim = f"{m.group(1).upper()} {m.group(2).lower()} {m.group(3)}"
            if not any(c["claim"] == claim for c in claims):
                add(claim, "indicator_threshold")

    # Named chart types, structures and price patterns, so a good
    # chart_description becomes queryable claims instead of a wall of prose.
    _CHART_TERMS = (
        r"candlestick chart|heikin ashi|line chart|bar chart|renko|"
        r"double top|double bottom|head and shoulders|inverse head and shoulders|"
        r"bullish engulfing|bearish engulfing|engulfing|doji|hammer|shooting star|"
        r"morning star|evening star|pin bar|inside bar|"
        r"ascending triangle|descending triangle|symmetrical triangle|triangle|"
        r"bull flag|bear flag|flag|pennant|wedge|channel|cup and handle|"
        r"support level|resistance level|support|resistance|trendline|trend line|"
        r"supply zone|demand zone|order block|fair value gap|liquidity sweep|"
        r"fibonacci retracement|fibonacci|golden pocket|"
        r"bullish divergence|bearish divergence|divergence|"
        r"breakout|breakdown|retest|pullback|consolidation|"
        r"moving average crossover|golden cross|death cross"
    )
    # Forex and crypto levels carry no currency sign, so the $-anchored block
    # above never sees them. Anchor on the words instead of the number, which
    # keeps bare figures from becoming price claims.
    for line in lines:
        for m in re.finditer(
            r"(?i)\b(support|resistance|level|target|entry|stop loss|stop|"
            r"take profit|tp|sl)\b[^.\n]{0,24}?\b(?:at|near|around|@)\s*"
            # Comma-grouped form first, else a plain run of digits. A bare
            # 1-3 digit cap silently truncated "2350.5" to "235".
            r"(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)",
            line,
        ):
            add(f"{m.group(1).title()} at {m.group(2)}", "price_level")

    seen_terms: set[str] = set()
    for line in lines:
        for m in re.finditer(rf"(?i)\b({_CHART_TERMS})\b", line):
            term = m.group(1).lower()
            if term not in seen_terms:
                seen_terms.add(term)
                add(term, "chart_pattern")

    return claims


def empty_result(enable_visual: bool = False) -> dict:
    """The shape callers can rely on even when visual extraction is skipped."""
    return {
        "enabled": enable_visual,
        "vision_available": False,
        # Report the model that will actually run, not the Ollama constant.
        # A live Gemini run reported vision_model "ui-tars-7b:latest".
        "vision_model": (
            GEMINI_MODEL if resolve_provider() == "gemini" else VISION_MODEL
        ),
        "vision_error": "",
        "ffmpeg": "",
        "video_downloaded": False,
        "frames_extracted": 0,
        "frames_after_dedup": 0,
        "frames_analyzed": 0,
        "frames_failed": 0,
        "provider": resolve_provider(),
        "mode": "frames",
        "windows": 0,
        "frame_errors": [],
        "visual_claims": [],
        "on_screen_texts": [],
        "chart_patterns": [],
        "frame_analyses": [],
    }


def extract_visual_content(
    video_id: str,
    output_dir: Path,
    duration: int = 0,
    cookies_file: str | None = None,
    cookies_from_browser: str | None = None,
    proxy: str | None = None,
    enable_visual: bool = True,
    chapter_timestamps: list[float] | None = None,
    max_frames: int = MAX_FRAMES,
    target_frames: int = 0,
) -> dict:
    """Extract on-screen content from a video's frames using a vision model.

    Always returns a result dict. Any failure is recorded in the dict rather
    than raised, so enabling --visual can never break a working scrape.
    """
    result = empty_result(enable_visual)
    if not enable_visual:
        return result

    try:
        ok, detail = vision_available()
        result["vision_available"] = ok
        if not ok:
            result["vision_error"] = detail
            print(f"  [visual] {detail}", file=sys.stderr)
            return result

        provider = resolve_provider()
        result["provider"] = provider
        mode = (VISUAL_MODE or "auto").strip().lower()

        # --- Gemini whole-video path (added 2026-09-02) ------------------
        # Measured on this machine 2026-09-02: yt-dlp gets HTTP 403 from
        # YouTube and every --cookies-from-browser source fails (Chrome DB
        # locked, Edge DPAPI, Brave/Firefox absent). Gemini fetches the URL
        # server-side, so this path needs no download, no ffmpeg and no local
        # GPU. Long videos are split into clipped windows -- video costs
        # ~100 tokens/second, so ~3 hours is the single-request ceiling.
        if provider == "gemini" and mode in ("auto", "video"):
            url = f"https://www.youtube.com/watch?v={video_id}"
            windows = video_windows(duration)
            result["mode"] = "video"
            result["windows"] = len(windows) or 1

            if windows:
                print(
                    f"  [visual] {_hhmmss(duration)} video -> {len(windows)} "
                    f"windows of {GEMINI_VIDEO_CHUNK_SECONDS // 60}min, "
                    f"analysing with {GEMINI_MODEL} (no download)...",
                    file=sys.stderr,
                )
            else:
                print(
                    f"  [visual] Analysing whole video with {GEMINI_MODEL} "
                    f"(no download)...",
                    file=sys.stderr,
                )

            claims: list[dict] = []
            texts: list[dict] = []
            charts: list[dict] = []
            analyses: list[dict] = []
            first_error = ""

            for idx, (w_start, w_end) in enumerate(
                    windows or [(0.0, 0.0)], start=1):
                if windows:
                    print(
                        f"  [visual] window {idx}/{len(windows)} "
                        f"{_hhmmss(w_start)}-{_hhmmss(w_end)}...",
                        file=sys.stderr,
                    )
                reading = analyze_youtube_video(
                    url, start_s=w_start, end_s=w_end)

                if reading.get("error"):
                    result["frames_failed"] += 1
                    result["frame_errors"].append({
                        "timestamp": w_start,
                        "timestamp_str": _hhmmss(w_start),
                        "frame_file": "",
                        "error": reading["error"],
                    })
                    first_error = first_error or reading["error"]
                    print(f"  [visual] window {idx} failed: "
                          f"{reading['error']}", file=sys.stderr)
                    continue
                if not reading.get("has_content"):
                    continue

                result["frames_analyzed"] += 1
                ts = float(w_start)
                ts_str = _hhmmss(ts)

                if reading["text"]:
                    texts.append({
                        "timestamp": ts,
                        "timestamp_str": ts_str,
                        "text": reading["text"],
                        "line_count": len(reading["lines"]),
                    })
                    claims.extend(extract_visual_claims(reading["text"], ts))
                if reading["chart"]:
                    charts.append({
                        "timestamp": ts,
                        "timestamp_str": ts_str,
                        "patterns": reading["chart"],
                        "model": GEMINI_MODEL,
                    })
                    claims.extend(extract_visual_claims(
                        reading["chart"], ts, source="visual_chart"))
                analyses.append({
                    "timestamp": ts,
                    "timestamp_str": ts_str,
                    "frame_file": "",
                    "window_start": ts,
                    "window_end": float(w_end),
                    "text": reading["text"],
                    "lines": reading["lines"],
                    "chart": reading["chart"],
                })

            result["on_screen_texts"] = texts
            result["chart_patterns"] = charts
            result["frame_analyses"] = analyses
            result["visual_claims"] = claims

            if result["frames_analyzed"] == 0:
                result["vision_error"] = (
                    first_error or "model returned no usable content")
                # "auto" may still fall through to frames; video mode stops.
                if mode == "video" or first_error:
                    return result
            else:
                return result

        ffmpeg = find_ffmpeg()
        result["ffmpeg"] = ffmpeg
        if not ffmpeg:
            result["vision_error"] = "ffmpeg not found"
            print(
                "  [visual] ffmpeg not found — `pip install imageio-ffmpeg`",
                file=sys.stderr,
            )
            return result

        print("  [visual] Downloading video (lowest quality)...", file=sys.stderr)
        video_path = download_video_low_quality(
            video_id, output_dir, cookies_file, cookies_from_browser, proxy,
        )
        if not video_path:
            result["vision_error"] = "video download failed"
            return result
        result["video_downloaded"] = True

        print("  [visual] Extracting frames...", file=sys.stderr)
        frames = extract_frames(
            video_path, output_dir, duration=duration,
            max_frames=max_frames,
            timestamps=chapter_timestamps or None,
        )
        result["frames_extracted"] = len(frames)
        if not frames:
            result["vision_error"] = "no frames extracted"
            return result

        frames = select_frames(frames, target=target_frames)
        result["frames_after_dedup"] = len(frames)
        print(
            f"  [visual] {len(frames)} unique frames; reading with {VISION_MODEL}"
            f" (~45s each, model stays resident between frames)...",
            file=sys.stderr,
        )

        on_screen_texts: list[dict] = []
        visual_claims: list[dict] = []
        chart_patterns: list[dict] = []
        frame_analyses: list[dict] = []

        for idx, frame in enumerate(frames, start=1):
            m = re.search(r"(\d+)m(\d+)s", frame.name)
            ts = float(int(m.group(1)) * 60 + int(m.group(2))) if m else 0.0
            ts_str = f"{int(ts // 60):02d}:{int(ts % 60):02d}"

            print(
                f"  [visual] frame {idx}/{len(frames)} at {ts_str}...",
                file=sys.stderr,
            )
            reading = read_frame(frame, ts)
            if reading.get("error"):
                result["frames_failed"] += 1
                result["frame_errors"].append({
                    "timestamp": ts,
                    "timestamp_str": ts_str,
                    "frame_file": frame.name,
                    "error": reading["error"],
                })
                print(
                    f"  [visual] frame {idx} failed: {reading['error']}",
                    file=sys.stderr,
                )
                continue
            if not reading.get("has_content"):
                continue

            result["frames_analyzed"] += 1

            if reading["text"]:
                on_screen_texts.append({
                    "timestamp": ts,
                    "timestamp_str": ts_str,
                    "text": reading["text"],
                    "line_count": len(reading["lines"]),
                })
                visual_claims.extend(extract_visual_claims(reading["text"], ts))

            if reading["chart"]:
                visual_claims.extend(extract_visual_claims(
                    reading["chart"], ts, source="visual_chart"))
                chart_patterns.append({
                    "timestamp": ts,
                    "timestamp_str": ts_str,
                    "patterns": reading["chart"],
                    "model": VISION_MODEL,
                })

            frame_analyses.append({
                "timestamp": ts,
                "timestamp_str": ts_str,
                "frame_file": frame.name,
                "text": reading["text"],
                "lines": reading["lines"],
                "chart": reading["chart"],
            })

        if result["frames_analyzed"] == 0 and result["frames_failed"]:
            first = result["frame_errors"][0]["error"]
            result["vision_error"] = (
                f'all {result["frames_failed"]} frame(s) failed; '
                f"first error: {first}"
            )

        result["on_screen_texts"] = on_screen_texts
        result["visual_claims"] = visual_claims
        result["chart_patterns"] = chart_patterns
        result["frame_analyses"] = frame_analyses

        # Frames are small and useful for auditing; the video is not.
        try:
            video_path.unlink()
        except Exception:
            pass

    except Exception as e:
        # A visual failure must never take down the scrape.
        result["vision_error"] = f"{type(e).__name__}: {e}"
        print(f"  [visual] extraction failed: {e}", file=sys.stderr)

    return result
