"""Tests for speaker diarization (Phase 3.4).

Tests cover:
  - Heuristic diarization (always available)
  - Speaker change detection via silence gaps
  - Speaker change detection via conversational cues
  - Empty/edge cases
  - DiarizedSegment and DiarizationResult dataclasses
  - Formatting
"""
import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from models import TranscriptSegment
from diarize import (
    DiarizedSegment, DiarizationResult,
    diarize_segments, format_diarized_transcript,
    _heuristic_diarize, _format_ts,
)


def _make_segments():
    """Build segments with a clear speaker change (3s gap)."""
    return [
        TranscriptSegment(text="Hello and welcome.", start=0.0, end=5.0),
        TranscriptSegment(text="Thanks for having me.", start=8.0, end=11.0),  # 3s gap
        TranscriptSegment(text="Let's dive in.", start=11.5, end=13.0),       # 0.5s gap (same speaker)
    ]


class TestHeuristicDiarize:
    def test_assigns_speakers(self):
        segs = _make_segments()
        result = _heuristic_diarize(segs)
        assert result.method == "heuristic"
        assert len(result.segments) == 3
        # Each segment should have a speaker label
        for seg in result.segments:
            assert seg.speaker.startswith("SPEAKER_")

    def test_speaker_change_on_silence_gap(self):
        """A gap > 2.5s should trigger a speaker change."""
        segs = _make_segments()
        result = _heuristic_diarize(segs)
        # First two segments have a 3s gap → different speakers
        assert result.segments[0].speaker != result.segments[1].speaker

    def test_same_speaker_on_short_gap(self):
        """A gap < 2.5s should keep the same speaker."""
        segs = _make_segments()
        result = _heuristic_diarize(segs)
        # Segments 2 and 3 have a 0.5s gap → same speaker
        assert result.segments[1].speaker == result.segments[2].speaker

    def test_speaker_change_on_cue(self):
        """A segment starting with a speaker-change cue should change speaker."""
        segs = [
            TranscriptSegment(text="I love Python.", start=0.0, end=2.0),
            TranscriptSegment(text="Yeah exactly, me too.", start=2.5, end=4.0),  # 0.5s gap but cue
        ]
        result = _heuristic_diarize(segs)
        assert result.segments[0].speaker != result.segments[1].speaker

    def test_single_segment(self):
        segs = [TranscriptSegment(text="Solo speaker.", start=0.0, end=5.0)]
        result = _heuristic_diarize(segs)
        assert len(result.segments) == 1
        assert result.speaker_count == 1
        assert result.segments[0].speaker == "SPEAKER_00"

    def test_empty_segments(self):
        result = _heuristic_diarize([])
        assert result.segments == ()
        assert result.speaker_count == 0

    def test_speaker_count_reflects_unique_speakers(self):
        segs = _make_segments()
        result = _heuristic_diarize(segs)
        # 2 unique speakers (SPEAKER_00 and SPEAKER_01)
        assert result.speaker_count == 2
        assert len(result.speakers) == 2


class TestDiarizeSegments:
    def test_empty_input(self):
        result = diarize_segments([])
        assert result.segments == ()
        assert result.method == "empty"

    def test_auto_without_audio_uses_heuristic(self):
        """Without audio_path, auto should fall back to heuristic."""
        segs = _make_segments()
        result = diarize_segments(segs, audio_path=None, method="auto")
        assert result.method == "heuristic"

    def test_force_heuristic(self):
        segs = _make_segments()
        result = diarize_segments(segs, method="heuristic")
        assert result.method == "heuristic"

    def test_pyannote_without_audio_falls_back(self):
        """Forcing pyannote without audio should fall back to heuristic."""
        segs = _make_segments()
        result = diarize_segments(segs, audio_path=None, method="pyannote")
        assert result.method == "heuristic"

    def test_pyannote_token_error_falls_back(self):
        """If pyannote raises during model load (e.g. bad HF token), fallback fires."""
        import sys as _sys
        segs = _make_segments()
        # Simulate pyannote being installed but failing at Pipeline.from_pretrained
        fake_pyannote = MagicMock()
        fake_pyannote.Pipeline.from_pretrained.side_effect = Exception("401 Unauthorized")
        with patch.dict(_sys.modules, {"pyannote.audio": fake_pyannote}), \
             patch.dict(os.environ, {"HF_TOKEN": "bad-token"}):
            result = diarize_segments(
                segs, audio_path="/fake/audio.wav", method="pyannote",
            )
        assert result.method == "heuristic", \
            "Fallback should fire when pyannote raises during model load"

    def test_pyannote_no_token_falls_back(self):
        """Without HF_TOKEN env var, pyannote path should be skipped."""
        segs = _make_segments()
        with patch.dict(os.environ, {}, clear=True):
            result = diarize_segments(
                segs, audio_path="/fake/audio.wav", method="pyannote",
            )
        assert result.method == "heuristic"


class TestDiarizedSegment:
    def test_immutable(self):
        seg = DiarizedSegment(text="hi", start=0.0, end=1.0, speaker="SPEAKER_00")
        with pytest.raises(Exception):
            seg.speaker = "SPEAKER_01"

    def test_to_dict(self):
        seg = DiarizedSegment(text="hi", start=1.5, end=2.5, speaker="SPEAKER_00")
        d = seg.to_dict()
        assert d["text"] == "hi"
        assert d["start"] == 1.5
        assert d["end"] == 2.5
        assert d["speaker"] == "SPEAKER_00"


class TestFormatDiarizedTranscript:
    def test_formats_with_speaker_labels(self):
        result = DiarizationResult(
            segments=(
                DiarizedSegment(text="Hello.", start=0.0, end=5.0, speaker="SPEAKER_00"),
                DiarizedSegment(text="Hi there.", start=8.0, end=11.0, speaker="SPEAKER_01"),
            ),
            speaker_count=2,
            method="heuristic",
            speakers=("SPEAKER_00", "SPEAKER_01"),
        )
        text = format_diarized_transcript(result)
        assert "[00:00] SPEAKER_00: Hello." in text
        assert "[00:08] SPEAKER_01: Hi there." in text

    def test_empty_result(self):
        result = DiarizationResult(
            segments=(), speaker_count=0, method="empty", speakers=(),
        )
        assert format_diarized_transcript(result) == ""


class TestFormatTs:
    def test_minutes_seconds(self):
        assert _format_ts(65.0) == "01:05"

    def test_hours(self):
        assert _format_ts(3725.0) == "01:02:05"

    def test_zero(self):
        assert _format_ts(0.0) == "00:00"
