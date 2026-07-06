"""Data models for yt_scrape.

Pure dataclasses with no external dependencies — safe to import standalone.
Extracted from yt_scrape.py for modularity (Phase 1b).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


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

    @classmethod
    def from_dict(cls, d: dict) -> "VideoInfo":
        """Reconstruct a VideoInfo from a dict (inverse of to_dict)."""
        # Filter to only known fields (ignore unknown keys for forward-compat)
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in known}
        return cls(**filtered)


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
