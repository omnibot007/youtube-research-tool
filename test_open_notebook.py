"""Tests for the Open Notebook integration client.

Mocks urllib.request so no live server is needed. Tests cover:
  - Payload mapping (VideoInfo + segments → SourceCreate dict)
  - Timestamped transcript formatting
  - Metadata footer (channel, claims, URL)
  - Client request/response handling
  - Error handling (HTTP errors, connection failures)
  - Notebook lookup by name
"""
import json
import os
import sys
import urllib.error
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from yt_scrape import VideoInfo, TranscriptSegment, extract_claims_for_video
from open_notebook import (
    OpenNotebookClient, OpenNotebookConfig, OpenNotebookError,
    build_timestamped_transcript, video_info_to_source_payload,
    _format_timestamp, _build_metadata_footer,
)


# ---------------------------------------------------------- helpers

def _make_video(**overrides):
    """Build a VideoInfo with sensible defaults for tests."""
    defaults = dict(
        id="abc123XYZ",
        title="Test Video Title",
        url="https://www.youtube.com/watch?v=abc123XYZ",
        channel="TestChannel",
        duration=300,
        upload_date="20240115",
        transcript_lang="en",
        transcript_source="caption",
        has_transcript=True,
    )
    defaults.update(overrides)
    return VideoInfo(**defaults)


def _make_segments():
    """Build a small list of TranscriptSegments."""
    return [
        TranscriptSegment(text="Hello world.", start=0.0, end=2.0),
        TranscriptSegment(text="This is a test.", start=2.0, end=5.0),
        TranscriptSegment(text="Goodbye.", start=65.0, end=66.0),
    ]


def _mock_urlopen(response_bytes=b'{"id": "source:123", "title": "ok"}', status=200):
    """Create a mock urlopen that returns the given response."""
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=cm)
    cm.__exit__ = MagicMock(return_value=False)
    cm.read = MagicMock(return_value=response_bytes)
    cm.status = status
    return MagicMock(return_value=cm)


# ---------------------------------------------------------- timestamp formatting

class TestFormatTimestamp:
    def test_seconds_only(self):
        assert _format_timestamp(5.0) == "00:05"

    def test_minutes(self):
        assert _format_timestamp(65.0) == "01:05"

    def test_hours(self):
        assert _format_timestamp(3725.0) == "01:02:05"

    def test_zero(self):
        assert _format_timestamp(0.0) == "00:00"

    def test_none_safe(self):
        assert _format_timestamp(None) == "00:00"


# ---------------------------------------------------------- transcript builder

class TestBuildTimestampedTranscript:
    def test_renders_inline_timestamps(self):
        segs = _make_segments()
        result = build_timestamped_transcript(segs)
        assert "[00:00] Hello world." in result
        assert "[00:02] This is a test." in result
        assert "[01:05] Goodbye." in result

    def test_skips_empty_segments(self):
        segs = [
            TranscriptSegment(text="", start=0.0, end=1.0),
            TranscriptSegment(text="Real content.", start=1.0, end=3.0),
        ]
        result = build_timestamped_transcript(segs)
        assert "[00:00]" not in result
        assert "[00:01] Real content." in result

    def test_empty_list(self):
        assert build_timestamped_transcript([]) == ""


# ---------------------------------------------------------- payload mapping

class TestVideoInfoToSourcePayload:
    def test_basic_payload_shape(self):
        video = _make_video()
        segs = _make_segments()
        payload = video_info_to_source_payload(video, segs, notebook_id="notebook:abc")

        assert payload["type"] == "text"
        assert "notebook:abc" in payload["notebooks"]
        assert payload["embed"] is True
        assert payload["async_processing"] is False  # sync is the safe default
        assert "Test Video Title" in payload["title"]
        assert "abc123XYZ" in payload["title"]  # video ID appended

    def test_transcript_in_content(self):
        video = _make_video()
        segs = _make_segments()
        payload = video_info_to_source_payload(video, segs, notebook_id="nb:1")
        assert "[00:00] Hello world." in payload["content"]
        assert "[01:05] Goodbye." in payload["content"]

    def test_metadata_footer_in_content(self):
        video = _make_video()
        segs = _make_segments()
        payload = video_info_to_source_payload(video, segs, notebook_id="nb:1")
        assert "--- METADATA ---" in payload["content"]
        assert "Source URL: https://www.youtube.com/watch?v=abc123XYZ" in payload["content"]
        assert "Channel: TestChannel" in payload["content"]
        assert "Duration: 300s" in payload["content"]

    def test_claims_in_footer(self):
        video = _make_video()
        segs = _make_segments()
        claims = [
            {"claim": "Python is fast", "timestamp": "00:30", "confidence": "high"},
            {"claim": "Tests catch bugs", "timestamp": "01:00"},
        ]
        payload = video_info_to_source_payload(video, segs, claims=claims, notebook_id="nb:1")
        assert "--- CLAIMS (2) ---" in payload["content"]
        assert "Python is fast" in payload["content"]
        assert "(at 00:30)" in payload["content"]
        assert "[high]" in payload["content"]

    def test_claims_capped_at_50(self):
        video = _make_video()
        segs = _make_segments()
        claims = [{"claim": f"Claim {i}"} for i in range(100)]
        payload = video_info_to_source_payload(video, segs, claims=claims, notebook_id="nb:1")
        # Only first 50 claims should appear
        assert "Claim 0" in payload["content"]
        assert "Claim 49" in payload["content"]
        assert "Claim 50" not in payload["content"]

    def test_multi_notebook(self):
        video = _make_video()
        payload = video_info_to_source_payload(
            video, [], notebook_id="nb:1", extra_notebooks=["nb:2", "nb:3"],
        )
        assert payload["notebooks"] == ["nb:1", "nb:2", "nb:3"]

    def test_empty_segments_falls_back_gracefully(self):
        video = _make_video()
        payload = video_info_to_source_payload(video, [], notebook_id="nb:1")
        assert "(no transcript available)" in payload["content"]
        # Metadata footer should still be present
        assert "--- METADATA ---" in payload["content"]

    def test_no_embed_option(self):
        video = _make_video()
        payload = video_info_to_source_payload(video, [], notebook_id="nb:1", embed=False)
        assert payload["embed"] is False

    def test_sync_processing_is_default(self):
        video = _make_video()
        payload = video_info_to_source_payload(video, [], notebook_id="nb:1")
        assert payload["async_processing"] is False

    def test_async_processing_opt_in(self):
        video = _make_video()
        payload = video_info_to_source_payload(
            video, [], notebook_id="nb:1", async_processing=True,
        )
        assert payload["async_processing"] is True


# ---------------------------------------------------------- client

class TestOpenNotebookClient:
    def _make_client(self, base_url="http://localhost:5055", api_key="secret"):
        return OpenNotebookClient(config=OpenNotebookConfig(
            base_url=base_url, api_key=api_key,
        ))

    def test_headers_include_api_key(self):
        client = self._make_client(api_key="mykey")
        h = client._headers()
        assert h["Authorization"] == "Bearer mykey"
        assert h["Content-Type"] == "application/json"

    def test_headers_without_api_key(self):
        client = self._make_client(api_key="")
        h = client._headers()
        assert "Authorization" not in h

    def test_create_source_posts_to_sources_json(self):
        client = self._make_client()
        captured_req = {}

        def fake_urlopen(req, timeout=None):
            captured_req["url"] = req.full_url
            captured_req["method"] = req.method
            captured_req["data"] = req.data
            cm = MagicMock()
            cm.__enter__ = MagicMock(return_value=cm)
            cm.__exit__ = MagicMock(return_value=False)
            cm.read = MagicMock(return_value=b'{"id": "source:new"}')
            return cm

        with patch("open_notebook.urllib.request.urlopen", side_effect=fake_urlopen):
            result = client.create_source({"type": "text", "content": "hi"})

        assert result == {"id": "source:new"}
        assert captured_req["url"] == "http://localhost:5055/api/sources/json"
        assert captured_req["method"] == "POST"
        assert json.loads(captured_req["data"]) == {"type": "text", "content": "hi"}

    def test_list_notebooks_returns_list(self):
        client = self._make_client()
        with patch("open_notebook.urllib.request.urlopen",
                   side_effect=_mock_urlopen(b'[{"id":"nb:1","name":"Research"}]')):
            notebooks = client.list_notebooks()
        assert len(notebooks) == 1
        assert notebooks[0]["name"] == "Research"

    def test_list_notebooks_single_dict_wrapped(self):
        client = self._make_client()
        with patch("open_notebook.urllib.request.urlopen",
                   side_effect=_mock_urlopen(b'{"id":"nb:1","name":"Solo"}')):
            notebooks = client.list_notebooks()
        assert len(notebooks) == 1
        assert notebooks[0]["name"] == "Solo"

    def test_find_notebook_by_name_case_insensitive(self):
        client = self._make_client()
        with patch("open_notebook.urllib.request.urlopen",
                   side_effect=_mock_urlopen(
                       b'[{"id":"nb:1","name":"Research"},{"id":"nb:2","name":"Trading"}]')):
            nb = client.find_notebook_by_name("RESEARCH")
        assert nb is not None
        assert nb["id"] == "nb:1"

    def test_find_notebook_by_name_not_found(self):
        client = self._make_client()
        with patch("open_notebook.urllib.request.urlopen",
                   side_effect=_mock_urlopen(b'[{"id":"nb:1","name":"Research"}]')):
            nb = client.find_notebook_by_name("Nonexistent")
        assert nb is None

    def test_push_video_end_to_end(self):
        client = self._make_client()
        video = _make_video()
        segs = _make_segments()

        with patch("open_notebook.urllib.request.urlopen",
                   side_effect=_mock_urlopen(b'{"id":"source:abc","title":"ok"}')):
            result = client.push_video(video, segs, notebook_id="nb:1")

        assert result["id"] == "source:abc"

    def test_http_error_raises_open_notebook_error(self):
        """5xx errors retry 3x then raise. Mock time.sleep to skip waits."""
        client = self._make_client()
        call_count = {"n": 0}
        def fake_urlopen(req, timeout=None):
            call_count["n"] += 1
            raise urllib.error.HTTPError(
                req.full_url, 500, "Server Error", {},
                MagicMock(read=MagicMock(return_value=b'{"detail":"boom"}')),
            )
        with patch("open_notebook.urllib.request.urlopen", side_effect=fake_urlopen), \
             patch("open_notebook.time.sleep"):
            with pytest.raises(OpenNotebookError) as exc:
                client.create_source({"type": "text"})
        assert "HTTP 500" in str(exc.value)
        assert "boom" in str(exc.value)
        assert call_count["n"] == 3  # retried 3 times

    def test_429_retries_with_retry_after_header(self):
        """429 Too Many Requests should retry, respecting Retry-After header."""
        client = self._make_client()
        responses = []

        def fake_urlopen(req, timeout=None):
            if len(responses) == 0:
                responses.append(1)
                raise urllib.error.HTTPError(
                    req.full_url, 429, "Too Many Requests",
                    {"Retry-After": "2"},
                    MagicMock(read=MagicMock(return_value=b'{"detail":"rate limited"}')),
                )
            cm = MagicMock()
            cm.__enter__ = MagicMock(return_value=MagicMock(
                read=MagicMock(return_value=b'{"id":"src:1"}'),
            ))
            cm.__exit__ = MagicMock(return_value=False)
            return cm

        with patch("open_notebook.urllib.request.urlopen", side_effect=fake_urlopen), \
             patch("open_notebook.time.sleep") as mock_sleep:
            result = client.create_source({"type": "text", "content": "x"})
        assert result["id"] == "src:1"
        # Should have slept for Retry-After value (2) + jitter
        mock_sleep.assert_called_once()
        sleep_arg = mock_sleep.call_args[0][0]
        assert sleep_arg >= 2  # at least the Retry-After value

    def test_429_respects_retry_after_then_fails_after_max_retries(self):
        """429 should retry up to 3 times, then raise."""
        client = self._make_client()
        call_count = {"n": 0}

        def fake_urlopen(req, timeout=None):
            call_count["n"] += 1
            raise urllib.error.HTTPError(
                req.full_url, 429, "Too Many Requests",
                {"Retry-After": "0"},
                MagicMock(read=MagicMock(return_value=b'{"detail":"rate limited"}')),
            )

        with patch("open_notebook.urllib.request.urlopen", side_effect=fake_urlopen), \
             patch("open_notebook.time.sleep"):
            with pytest.raises(OpenNotebookError) as exc:
                client.create_source({"type": "text"})
        assert "HTTP 429" in str(exc.value)
        assert call_count["n"] == 3

    def test_4xx_does_not_retry(self):
        """4xx errors (except 429) should fail fast without retrying."""
        client = self._make_client()
        call_count = {"n": 0}

        def fake_urlopen(req, timeout=None):
            call_count["n"] += 1
            raise urllib.error.HTTPError(
                req.full_url, 404, "Not Found", {},
                MagicMock(read=MagicMock(return_value=b'{"detail":"nope"}')),
            )

        with patch("open_notebook.urllib.request.urlopen", side_effect=fake_urlopen), \
             patch("open_notebook.time.sleep") as mock_sleep:
            with pytest.raises(OpenNotebookError) as exc:
                client.create_source({"type": "text"})
        assert "HTTP 404" in str(exc.value)
        assert call_count["n"] == 1  # no retries
        mock_sleep.assert_not_called()

    def test_5xx_retries_then_succeeds(self):
        """5xx should retry and succeed if a later attempt works."""
        client = self._make_client()
        responses = [None]  # first call fails, second succeeds

        def fake_urlopen(req, timeout=None):
            if len(responses) > 0 and responses[0] is None:
                responses.pop()
                raise urllib.error.HTTPError(
                    req.full_url, 503, "Service Unavailable", {},
                    MagicMock(read=MagicMock(return_value=b'{"detail":"down"}')),
                )
            cm = MagicMock()
            cm.__enter__ = MagicMock(return_value=MagicMock(
                read=MagicMock(return_value=b'{"id":"src:1"}'),
            ))
            cm.__exit__ = MagicMock(return_value=False)
            return cm

        with patch("open_notebook.urllib.request.urlopen", side_effect=fake_urlopen), \
             patch("open_notebook.time.sleep"):
            result = client.create_source({"type": "text", "content": "x"})
        assert result["id"] == "src:1"

    def test_connection_error_raises_open_notebook_error(self):
        client = self._make_client()
        def fake_urlopen(req, timeout=None):
            raise urllib.error.URLError("Connection refused")
        with patch("open_notebook.urllib.request.urlopen", side_effect=fake_urlopen):
            with pytest.raises(OpenNotebookError) as exc:
                client.list_notebooks()
        assert "Cannot reach Open Notebook" in str(exc.value)

    def test_get_source_status(self):
        client = self._make_client()
        with patch("open_notebook.urllib.request.urlopen",
                   side_effect=_mock_urlopen(b'{"status":"completed"}')):
            status = client.get_source_status("source:123")
        assert status["status"] == "completed"


class TestWaitForSource:
    def test_returns_when_completed(self):
        client = OpenNotebookClient(config=OpenNotebookConfig())
        statuses = [
            {"status": "processing"},
            {"status": "processing"},
            {"status": "completed"},
        ]
        def fake_urlopen(req, timeout=None):
            cm = MagicMock()
            cm.__enter__ = MagicMock(return_value=cm)
            cm.__exit__ = MagicMock(return_value=False)
            cm.read = MagicMock(return_value=json.dumps(statuses.pop(0)).encode())
            return cm
        with patch("open_notebook.urllib.request.urlopen", side_effect=fake_urlopen), \
             patch("open_notebook.time.sleep"):
            result = client.wait_for_source("source:1", timeout=10, poll_interval=0)
        assert result["status"] == "completed"

    def test_raises_on_failed(self):
        client = OpenNotebookClient(config=OpenNotebookConfig())
        def fake_urlopen(req, timeout=None):
            cm = MagicMock()
            cm.__enter__ = MagicMock(return_value=cm)
            cm.__exit__ = MagicMock(return_value=False)
            cm.read = MagicMock(return_value=b'{"status":"failed","error":"oom"}')
            return cm
        with patch("open_notebook.urllib.request.urlopen", side_effect=fake_urlopen), \
             patch("open_notebook.time.sleep"):
            with pytest.raises(OpenNotebookError) as exc:
                client.wait_for_source("source:1", timeout=10, poll_interval=0)
        assert "failed" in str(exc.value)


# ---------------------------------------------------------- config

class TestOpenNotebookConfig:
    def test_defaults_from_env(self, monkeypatch):
        monkeypatch.setenv("OPEN_NOTEBOOK_URL", "http://myhost:9999")
        monkeypatch.setenv("OPEN_NOTEBOOK_API_KEY", "envkey")
        cfg = OpenNotebookConfig()
        assert cfg.base_url == "http://myhost:9999"
        assert cfg.api_key == "envkey"

    def test_immutable(self):
        cfg = OpenNotebookConfig(base_url="http://x", api_key="k")
        with pytest.raises(Exception):
            cfg.base_url = "http://y"  # frozen dataclass


# ---------------------------------------------------------- channel ingest

class TestPushChannel:
    """Tests for channel-wide notebook ingest (push_channel)."""

    def _make_client(self):
        return OpenNotebookClient(config=OpenNotebookConfig(
            base_url="http://localhost:5055", api_key="test",
        ))

    def test_pushes_all_videos(self):
        """All scraped videos should be pushed to the notebook."""
        client = self._make_client()
        videos = [
            VideoInfo(id="vid1", title="Video 1", has_transcript=True,
                      segments=[TranscriptSegment(text="content 1", start=0, end=1)]),
            VideoInfo(id="vid2", title="Video 2", has_transcript=True,
                      segments=[TranscriptSegment(text="content 2", start=0, end=1)]),
        ]
        push_results = iter([
            {"id": "source:1"},
            {"id": "source:2"},
        ])

        with patch("yt_scrape.scrape_channel", return_value=videos), \
             patch.object(client, "_find_existing_titles", return_value=set()), \
             patch.object(client, "push_video", side_effect=lambda *a, **k: next(push_results)):
            summary = client.push_channel("channel_url", "notebook:abc", limit=2)

        assert summary["pushed"] == 2
        assert summary["failed"] == 0
        assert summary["total"] == 2
        assert summary["source_ids"] == ["source:1", "source:2"]

    def test_skips_videos_with_errors(self):
        """Videos with scrape errors should be skipped, not pushed."""
        client = self._make_client()
        videos = [
            VideoInfo(id="vid1", title="OK", has_transcript=True,
                      segments=[TranscriptSegment(text="content", start=0, end=1)]),
            VideoInfo(id="vid2", error="private video", error_type="private"),
            VideoInfo(id="vid3", title="OK 3", has_transcript=True,
                      segments=[TranscriptSegment(text="content 3", start=0, end=1)]),
        ]
        push_results = iter([{"id": "source:1"}, {"id": "source:3"}])

        with patch("yt_scrape.scrape_channel", return_value=videos), \
             patch.object(client, "_find_existing_titles", return_value=set()), \
             patch.object(client, "push_video", side_effect=lambda *a, **k: next(push_results)):
            summary = client.push_channel("channel_url", "notebook:abc", limit=3)

        assert summary["pushed"] == 2
        assert summary["skipped"] == 1  # the error video is skipped, not failed
        assert summary["total"] == 3
        assert summary["failed"] == 0  # no push failures

    def test_skips_already_existing_videos(self):
        """Videos already in the notebook (by title) should be skipped for dedup."""
        client = self._make_client()
        videos = [
            VideoInfo(id="vid1", title="Existing Video", has_transcript=True,
                      segments=[TranscriptSegment(text="content", start=0, end=1)]),
            VideoInfo(id="vid2", title="New Video", has_transcript=True,
                      segments=[TranscriptSegment(text="content 2", start=0, end=1)]),
        ]
        # "existing video [vid1]" already in notebook
        existing = {"existing video [vid1]"}

        with patch("yt_scrape.scrape_channel", return_value=videos), \
             patch.object(client, "_find_existing_titles", return_value=existing), \
             patch.object(client, "push_video", return_value={"id": "source:new"}):
            summary = client.push_channel("channel_url", "notebook:abc", limit=2)

        assert summary["pushed"] == 1  # only the new video
        assert summary["skipped"] == 1  # the existing one

    def test_push_failure_recorded_not_raised(self):
        """A push failure should be recorded in errors, not raise."""
        client = self._make_client()
        videos = [
            VideoInfo(id="vid1", title="V1", has_transcript=True,
                      segments=[TranscriptSegment(text="c", start=0, end=1)]),
        ]

        def failing_push(*a, **k):
            raise OpenNotebookError("API down")

        with patch("yt_scrape.scrape_channel", return_value=videos), \
             patch.object(client, "_find_existing_titles", return_value=set()), \
             patch.object(client, "push_video", side_effect=failing_push):
            summary = client.push_channel("channel_url", "notebook:abc", limit=1)

        assert summary["pushed"] == 0
        assert summary["failed"] == 1
        assert "API down" in summary["errors"][0]["error"]

    def test_progress_callback_called(self):
        """on_progress should be called for each video."""
        client = self._make_client()
        videos = [
            VideoInfo(id="vid1", title="V1", has_transcript=True,
                      segments=[TranscriptSegment(text="c", start=0, end=1)]),
            VideoInfo(id="vid2", title="V2", has_transcript=True,
                      segments=[TranscriptSegment(text="c2", start=0, end=1)]),
        ]
        progress_calls: list[dict] = []

        with patch("yt_scrape.scrape_channel", return_value=videos), \
             patch.object(client, "_find_existing_titles", return_value=set()), \
             patch.object(client, "push_video", side_effect=[{"id": "s:1"}, {"id": "s:2"}]):
            client.push_channel(
                "channel_url", "notebook:abc", limit=2,
                on_progress=lambda p: progress_calls.append(p),
            )

        assert len(progress_calls) == 2
        assert progress_calls[0]["done"] == 1
        assert progress_calls[0]["total"] == 2
        assert progress_calls[0]["status"] == "pushed"
        assert progress_calls[1]["done"] == 2

    def test_with_claims_extracts(self):
        """with_claims=True should call extract_claims_enriched."""
        client = self._make_client()
        videos = [
            VideoInfo(id="vid1", title="V1", has_transcript=True,
                      segments=[TranscriptSegment(text="claim here", start=0, end=1)]),
        ]

        with patch("yt_scrape.scrape_channel", return_value=videos), \
             patch.object(client, "_find_existing_titles", return_value=set()), \
             patch("yt_scrape.extract_claims_enriched", return_value=[{"claim": "test"}]) as mock_claims, \
             patch.object(client, "push_video", return_value={"id": "s:1"}):
            client.push_channel("channel_url", "notebook:abc", limit=1, with_claims=True)

        mock_claims.assert_called_once()

    def test_empty_channel(self):
        """A channel with no videos should return zero counts."""
        client = self._make_client()
        with patch("yt_scrape.scrape_channel", return_value=[]):
            summary = client.push_channel("channel_url", "notebook:abc", limit=10)
        assert summary["pushed"] == 0
        assert summary["failed"] == 0
        assert summary["total"] == 0


# ---------------------------------------------------------- citation URLs

class TestExtractClaimsForVideo:
    """Tests for extract_claims_for_video — timestamped claims with citation URLs."""

    def _make_video(self, segments=None):
        segs = segments or [
            TranscriptSegment(text="Studies show that 85% of traders lose money.", start=30.0, end=35.0),
            TranscriptSegment(text="Bitcoin will reach $100,000 by December.", start=35.0, end=38.0),
        ]
        return VideoInfo(
            id="abc123XYZ", title="Test", segments=segs, has_transcript=True,
        )

    def test_returns_claims_with_video_id(self):
        video = self._make_video()
        claims = extract_claims_for_video(video)
        assert len(claims) > 0
        # Each claim should have a youtube_url with the video ID
        for claim in claims:
            if "youtube_url" in claim:
                assert "abc123XYZ" in claim["youtube_url"]

    def test_citation_url_includes_timestamp(self):
        """The youtube_url should include &t=N to jump to the claim's timestamp."""
        video = self._make_video()
        claims = extract_claims_for_video(video)
        # At least one claim should have a timestamp-based URL
        ts_claims = [c for c in claims if c.get("youtube_url") and c.get("timestamp")]
        if ts_claims:
            url = ts_claims[0]["youtube_url"]
            assert "t=" in url
            # The timestamp should be in seconds
            ts = ts_claims[0]["timestamp"]
            assert "start" in ts

    def test_no_citation_urls_when_disabled(self):
        """include_citation_urls=False should skip youtube_url."""
        video = self._make_video()
        claims = extract_claims_for_video(video, include_citation_urls=False)
        # No youtube_url should be present (video_id is empty)
        for claim in claims:
            assert "youtube_url" not in claim

    def test_empty_segments_returns_empty(self):
        video = VideoInfo(id="x", segments=[], has_transcript=False)
        assert extract_claims_for_video(video) == []

    def test_claims_appear_in_metadata_footer(self):
        """When pushed, claims with citation URLs should appear in the content footer."""
        video = self._make_video()
        claims = extract_claims_for_video(video)
        payload = video_info_to_source_payload(video, video.segments, claims=claims, notebook_id="nb:1")
        assert "--- CLAIMS" in payload["content"]
        # The citation URL should be in the footer
        if any(c.get("youtube_url") for c in claims):
            assert "youtube.com/watch?v=abc123XYZ" in payload["content"]
