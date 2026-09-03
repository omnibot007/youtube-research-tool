"""Tests for the corpus aggregator and the batch runner.

Both shipped untested on 2026-09-02, which is exactly the gap that let a
regex bug reach a live run earlier the same day. No network here: every test
builds real package files on disk and reads them back.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import corpus
import extract_batch


def write_package(folder, vid, claims, *, title="A Course", channel="Chan",
                  vision_error="", failed=0, coverage=100.0, segments=1):
    """Write a package shaped like a real one, so the reader is exercised."""
    pkg = {
        "ok": True,
        "video": {"id": vid, "title": title, "channel": channel,
                  "duration": 600},
        "visual_extraction": {
            "vision_model": "gemini-3.8-flash",
            "vision_error": vision_error,
            "frames_failed": failed,
            "coverage_pct": coverage,
            "segments": [{"timestamp": "00:00:10"}] * segments,
            "visual_claims": claims,
        },
    }
    p = Path(folder) / f"{vid}_research_package.json"
    p.write_text(json.dumps(pkg), encoding="utf-8")
    return p


def claim(ctype, text, ts="00:01:00", secs=60.0, source="visual_segment"):
    return {"type": ctype, "claim": text, "timestamp_str": ts,
            "timestamp": secs, "source": source}


class TestVideoId:
    @pytest.mark.parametrize("url", [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://youtu.be/dQw4w9WgXcQ",
        "https://www.youtube.com/embed/dQw4w9WgXcQ",
        "https://www.youtube.com/shorts/dQw4w9WgXcQ",
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=42s",
        "dQw4w9WgXcQ",
    ])
    def test_every_url_form_yields_the_id(self, url):
        assert extract_batch.video_id(url) == "dQw4w9WgXcQ"

    def test_nonsense_yields_empty_not_a_guess(self):
        assert extract_batch.video_id("https://example.com/nope") == ""
        assert extract_batch.video_id("") == ""


class TestResumeCache:
    def test_a_good_package_is_treated_as_done(self, tmp_path):
        write_package(tmp_path, "aaaaaaaaaaa", [claim("timeframe", "Timeframe: 1d")])
        assert extract_batch.already_done("aaaaaaaaaaa", str(tmp_path))

    def test_a_missing_package_is_not_done(self, tmp_path):
        assert extract_batch.already_done("bbbbbbbbbbb", str(tmp_path)) is None

    def test_a_vision_failed_package_does_not_block_a_retry(self, tmp_path):
        write_package(tmp_path, "ccccccccccc", [claim("timeframe", "x")],
                      vision_error="all 3 frame(s) failed")
        assert extract_batch.already_done("ccccccccccc", str(tmp_path)) is None

    def test_an_empty_package_does_not_block_a_retry(self, tmp_path):
        write_package(tmp_path, "ddddddddddd", [], segments=0)
        assert extract_batch.already_done("ddddddddddd", str(tmp_path)) is None

    def test_corrupt_json_does_not_block_a_retry(self, tmp_path):
        (tmp_path / "eeeeeeeeeee_research_package.json").write_text(
            "{ truncated", encoding="utf-8")
        assert extract_batch.already_done("eeeeeeeeeee", str(tmp_path)) is None


class TestCostEstimate:
    def test_longer_videos_cost_more(self):
        _, cheap = extract_batch.estimate(600)
        _, dear = extract_batch.estimate(7200)
        assert dear > cheap

    def test_token_estimate_tracks_the_measured_rate(self):
        tokens, _ = extract_batch.estimate(3600)
        # ~95 tokens/second measured live; allow generous slack.
        assert 250_000 < tokens < 450_000

    def test_zero_length_is_free(self):
        tokens, cost = extract_batch.estimate(0)
        assert tokens == 0 and cost == 0


class TestCorpusRows:
    def test_claims_from_many_packages_are_joined(self, tmp_path):
        write_package(tmp_path, "vid00000001", [claim("timeframe", "Timeframe: 1d")])
        write_package(tmp_path, "vid00000002", [claim("timeframe", "Timeframe: 4h")])
        rows = corpus.claim_rows(corpus.load_packages(tmp_path))
        assert len(rows) == 2
        assert {r["video_id"] for r in rows} == {"vid00000001", "vid00000002"}

    def test_every_row_carries_provenance(self, tmp_path):
        write_package(tmp_path, "vid00000003",
                      [claim("indicator_setting", "RSI = 14", secs=95.0)])
        row = corpus.claim_rows(corpus.load_packages(tmp_path))[0]
        for field in ("video_id", "title", "channel", "timestamp", "type",
                      "claim", "source", "model", "url"):
            assert row[field] != "", field
        assert row["url"].endswith("&t=95s"), "deep link to the exact second"

    def test_an_unreadable_package_is_skipped_not_fatal(self, tmp_path):
        write_package(tmp_path, "vid00000004", [claim("timeframe", "Timeframe: 1d")])
        (tmp_path / "broken_research_package.json").write_text(
            "not json", encoding="utf-8")
        assert len(corpus.claim_rows(corpus.load_packages(tmp_path))) == 1

    def test_empty_folder_is_not_an_error(self, tmp_path):
        assert corpus.claim_rows(corpus.load_packages(tmp_path)) == []


class TestValueParsing:
    def test_splits_both_separators(self):
        assert corpus.value_of("RSI = 14") == "14"
        assert corpus.subject_of("RSI = 14") == "RSI"
        assert corpus.value_of("Instrument: BTCUSD") == "BTCUSD"
        assert corpus.subject_of("Instrument: BTCUSD") == "Instrument"

    def test_a_bare_claim_is_its_own_value(self):
        assert corpus.value_of("bearish divergence") == "bearish divergence"


class TestConflictDetection:
    """The corpus earning its keep: disagreement no single video can show."""

    def test_same_indicator_different_periods_is_reported(self, tmp_path):
        write_package(tmp_path, "vid00000005",
                      [claim("indicator_setting", "RSI = 14")])
        write_package(tmp_path, "vid00000006",
                      [claim("indicator_setting", "RSI = 21")])
        rows = corpus.claim_rows(corpus.load_packages(tmp_path))
        conflicts = dict(corpus.find_conflicts(rows))
        assert "RSI" in conflicts
        assert set(conflicts["RSI"]) == {"14", "21"}

    def test_agreement_is_not_a_conflict(self, tmp_path):
        write_package(tmp_path, "vid00000007",
                      [claim("indicator_setting", "RSI = 14")])
        write_package(tmp_path, "vid00000008",
                      [claim("indicator_setting", "RSI = 14")])
        rows = corpus.claim_rows(corpus.load_packages(tmp_path))
        assert corpus.find_conflicts(rows) == []

    def test_spelling_variants_are_one_indicator(self, tmp_path):
        write_package(tmp_path, "vid00000009",
                      [claim("indicator_setting", "Relative Strength Index = 14")])
        write_package(tmp_path, "vid00000010",
                      [claim("indicator_setting", "rsi = 21")])
        rows = corpus.claim_rows(corpus.load_packages(tmp_path))
        conflicts = dict(corpus.find_conflicts(rows))
        assert "RSI" in conflicts, "normalisation must join these"
        assert set(conflicts["RSI"]) == {"14", "21"}

    def test_non_indicator_claims_are_ignored(self, tmp_path):
        write_package(tmp_path, "vid00000011",
                      [claim("chart_pattern", "uptrend"),
                       claim("chart_pattern", "downtrend")])
        rows = corpus.claim_rows(corpus.load_packages(tmp_path))
        assert corpus.find_conflicts(rows) == []


class TestSummaryIsSafe:
    def test_summary_of_an_empty_corpus_does_not_raise(self, capsys):
        corpus.summarise([], [])

    def test_summary_reports_partial_coverage(self, tmp_path, capsys):
        write_package(tmp_path, "vid00000012", [claim("timeframe", "Timeframe: 1d")],
                      failed=2, coverage=66.6)
        pkgs = corpus.load_packages(tmp_path)
        corpus.summarise(corpus.claim_rows(pkgs), pkgs)
        out = capsys.readouterr().out
        assert "partially covered" in out
        assert "66.6" in out


class TestPlaylistAndDedup:
    """A watch?v=X&list=Y URL is one video AND a playlist position."""

    class Args:
        def __init__(self, urls, no_playlist=False):
            self.urls = urls
            self.from_file = ""
            self.no_playlist = no_playlist

    def test_plain_urls_pass_through(self):
        got = extract_batch.load_urls(self.Args(
            ["https://www.youtube.com/watch?v=aaaaaaaaaaa"]))
        assert len(got) == 1

    def test_same_video_twice_is_one_entry(self):
        got = extract_batch.load_urls(self.Args([
            "https://www.youtube.com/watch?v=aaaaaaaaaaa",
            "https://youtu.be/aaaaaaaaaaa",
        ]))
        assert len(got) == 1, "de-dupe must key on video id, not URL text"

    def test_tracking_tails_do_not_create_duplicates(self):
        got = extract_batch.load_urls(self.Args([
            "https://youtu.be/aaaaaaaaaaa?si=ABC",
            "https://youtu.be/aaaaaaaaaaa?si=XYZ",
        ]))
        assert len(got) == 1

    def test_no_playlist_flag_keeps_the_single_video(self):
        url = "https://www.youtube.com/watch?v=aaaaaaaaaaa&list=PLxxxx"
        got = extract_batch.load_urls(self.Args([url], no_playlist=True))
        assert got == [url], "must not hit the network with --no-playlist"

    def test_a_non_playlist_url_is_never_expanded(self, monkeypatch):
        def boom(*a, **k):
            raise AssertionError("expansion attempted on a plain URL")
        monkeypatch.setattr(extract_batch, "expand_playlist",
                            lambda u: [u] if "list=" not in u else boom())
        got = extract_batch.load_urls(self.Args(
            ["https://www.youtube.com/watch?v=bbbbbbbbbbb"]))
        assert len(got) == 1
