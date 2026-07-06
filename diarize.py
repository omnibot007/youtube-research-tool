"""Speaker diarization for transcript segments.

Labels each transcript segment with a speaker ID (SPEAKER_00, SPEAKER_01, ...)
using pyannote.audio if installed, or a lightweight fallback heuristic based
on segment gaps and text cues.

pyannote is a heavy dependency (requires model download + HuggingFace token),
so it's optional. The fallback heuristic uses inter-segment silence gaps and
speaker-turn cues ("and then I said", "yeah exactly") to estimate speaker
boundaries — not as accurate as pyannote but always available.

Extracted as a separate module (Phase 3.4).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from models import TranscriptSegment


@dataclass(frozen=True)
class DiarizedSegment:
    """A transcript segment with an assigned speaker label."""
    text: str
    start: float
    end: float
    speaker: str  # "SPEAKER_00", "SPEAKER_01", etc.

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "start": round(self.start, 2),
            "end": round(self.end, 2),
            "speaker": self.speaker,
        }


@dataclass(frozen=True)
class DiarizationResult:
    """Result of diarizing a list of segments."""
    segments: tuple[DiarizedSegment, ...]
    speaker_count: int
    method: str  # "pyannote" | "heuristic"
    speakers: tuple[str, ...]


# Cues that often indicate a speaker change in conversational transcripts
_SPEAKER_CHANGE_CUES = (
    "yeah exactly", "that's right", "i agree", "i think", "i believe",
    "you said", "you mentioned", "and then i", "so i", "but i",
    "what do you think", "let me ask", "can i ask",
)


def diarize_segments(
    segments: list[TranscriptSegment],
    audio_path: Optional[str] = None,
    method: str = "auto",
    hf_token: Optional[str] = None,
    min_speakers: int = 1,
    max_speakers: int = 6,
) -> DiarizationResult:
    """Assign speaker labels to transcript segments.

    Args:
        segments: List of TranscriptSegment to diarize.
        audio_path: Path to audio file (required for pyannote). If None, uses heuristic.
        method: "auto" (pyannote if audio + installed, else heuristic),
                "pyannote", or "heuristic".
        hf_token: HuggingFace token for pyannote model download.
        min_speakers: Minimum speakers for pyannote.
        max_speakers: Maximum speakers for pyannote.

    Returns:
        DiarizationResult with speaker-labeled segments.
    """
    if not segments:
        return DiarizationResult(
            segments=(), speaker_count=0, method="empty", speakers=(),
        )

    if method in ("auto", "pyannote") and audio_path:
        result = _try_pyannote(segments, audio_path, hf_token, min_speakers, max_speakers)
        if result is not None:
            return result

    return _heuristic_diarize(segments)


def _try_pyannote(
    segments: list[TranscriptSegment],
    audio_path: str,
    hf_token: Optional[str],
    min_speakers: int,
    max_speakers: int,
) -> Optional[DiarizationResult]:
    """Attempt pyannote.audio diarization. Returns None if not installed."""
    try:
        import os
        from pyannote.audio import Pipeline
    except ImportError:
        return None

    token = hf_token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    if not token:
        return None

    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", use_auth_token=token,
        )
        diarization = pipeline(
            audio_path, min_speakers=min_speakers, max_speakers=max_speakers,
        )

        # Map each transcript segment to the speaker speaking at its midpoint
        labeled: list[DiarizedSegment] = []
        speakers_seen: set[str] = set()
        for seg in segments:
            midpoint = (seg.start + seg.end) / 2
            speaker = "SPEAKER_00"
            # Find the pyannote turn active at the midpoint
            for turn, _, label in diarization.itertracks(yield_label=True):
                if turn.start <= midpoint <= turn.end:
                    speaker = label
                    break
            speakers_seen.add(speaker)
            labeled.append(DiarizedSegment(
                text=seg.text, start=seg.start, end=seg.end, speaker=speaker,
            ))

        return DiarizationResult(
            segments=tuple(labeled),
            speaker_count=len(speakers_seen),
            method="pyannote",
            speakers=tuple(sorted(speakers_seen)),
        )
    except Exception:
        return None


def _heuristic_diarize(segments: list[TranscriptSegment]) -> DiarizationResult:
    """Fallback diarization using silence gaps and speaker-turn cues.

    Heuristic:
    1. A silence gap > 2.5s between segments suggests a speaker change.
    2. A segment starting with a speaker-change cue suggests a speaker change.
    3. Otherwise, keep the same speaker as the previous segment.
    """
    if not segments:
        return DiarizationResult(
            segments=(), speaker_count=0, method="heuristic", speakers=(),
        )

    GAP_THRESHOLD = 2.5  # seconds of silence → likely speaker change
    labeled: list[DiarizedSegment] = []
    current_speaker_idx = 0
    speakers_seen: set[str] = set()

    for i, seg in enumerate(segments):
        if i == 0:
            current_speaker_idx = 0
        else:
            prev = segments[i - 1]
            gap = seg.start - prev.end
            text_lower = seg.text.lower().strip()
            has_cue = any(text_lower.startswith(cue) for cue in _SPEAKER_CHANGE_CUES)
            if gap > GAP_THRESHOLD or has_cue:
                current_speaker_idx += 1

        speaker = f"SPEAKER_{current_speaker_idx:02d}"
        speakers_seen.add(speaker)
        labeled.append(DiarizedSegment(
            text=seg.text, start=seg.start, end=seg.end, speaker=speaker,
        ))

    return DiarizationResult(
        segments=tuple(labeled),
        speaker_count=len(speakers_seen),
        method="heuristic",
        speakers=tuple(sorted(speakers_seen)),
    )


def format_diarized_transcript(result: DiarizationResult) -> str:
    """Format a diarization result as a readable transcript with speaker labels.

    Interleaves timestamps with speaker labels in a single block:
        [00:00] SPEAKER_00: Hello and welcome to the show.
        [00:05] SPEAKER_01: Thanks for having me.

    Includes a warning header when the heuristic fallback was used, since
    heuristic speaker labels are unreliable (based on silence gaps + cues,
    not voice identity).
    """
    if not result.segments:
        return ""

    lines: list[str] = []
    if result.method == "heuristic":
        lines.append(
            "# WARNING: Speaker attribution is heuristic (no pyannote model available). "
            "Labels are based on silence gaps and conversational cues — NOT voice identity. "
            "Treat SPEAKER_NN labels as approximate turn boundaries, not real speakers."
        )
        lines.append("")

    for seg in result.segments:
        ts = _format_ts(seg.start)
        lines.append(f"[{ts}] {seg.speaker}: {seg.text.strip()}")
    return "\n".join(lines)


def _format_ts(seconds: float) -> str:
    """Format seconds as MM:SS or HH:MM:SS."""
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"
