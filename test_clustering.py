"""Tests for topic clustering across videos (Phase 3.3).

Tests cover:
  - Keyword fallback clustering (always available)
  - sklearn clustering (if installed)
  - BERTopic clustering (if installed)
  - Empty/edge cases
  - Summary formatting
  - Topic dataclass immutability
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from yt_scrape import VideoInfo, TranscriptSegment
from clustering import (
    Topic, ClusteringResult, cluster_videos, format_clustering_summary,
    _tokenize, _extract_video_text, _keyword_cluster,
)


def _make_video(vid_id: str, text: str, title: str = "") -> VideoInfo:
    """Build a VideoInfo with a single segment containing the given text."""
    return VideoInfo(
        id=vid_id,
        title=title or f"Video {vid_id}",
        segments=[TranscriptSegment(text=text, start=0.0, end=10.0)],
        has_transcript=True,
    )


class TestTokenize:
    def test_lowercase_split(self):
        tokens = _tokenize("Hello World Test")
        assert "hello" in tokens
        assert "world" in tokens

    def test_drops_short_tokens(self):
        tokens = _tokenize("a an the cat dog")
        assert "cat" in tokens
        assert "dog" in tokens
        assert "a" not in tokens

    def test_drops_stopwords(self):
        tokens = _tokenize("the quick brown fox jumps")
        assert "the" not in tokens
        assert "quick" in tokens

    def test_drops_youtube_noise(self):
        tokens = _tokenize("hey guys subscribe like comment share")
        assert "hey" not in tokens
        assert "subscribe" not in tokens


class TestExtractVideoText:
    def test_uses_segments(self):
        v = _make_video("v1", "content from segments")
        assert _extract_video_text(v) == "content from segments"

    def test_falls_back_to_title(self):
        v = VideoInfo(id="v1", title="Title only", segments=[])
        assert _extract_video_text(v) == "Title only"

    def test_empty_video(self):
        v = VideoInfo(id="v1", title="", segments=[])
        assert _extract_video_text(v) == ""


class TestKeywordCluster:
    def test_groups_by_shared_keywords(self):
        videos = [
            _make_video("v1", "bitcoin cryptocurrency trading market analysis"),
            _make_video("v2", "bitcoin price prediction crypto wallet"),
            _make_video("v3", "cooking recipe pasta italian food kitchen"),
            _make_video("v4", "pasta sauce recipe garlic tomato cooking"),
        ]
        result = _keyword_cluster(videos, n_topics=3)
        assert result.method == "keyword"
        assert len(result.topics) >= 1
        # Bitcoin videos should cluster together (share "bitcoin" keyword)
        all_topic_video_sets = [set(t.video_ids) for t in result.topics]
        bitcoin_topic = next(
            (s for s in all_topic_video_sets if "v1" in s and "v2" in s), None
        )
        assert bitcoin_topic is not None, "Bitcoin videos should be in the same topic"

    def test_empty_videos_returns_empty(self):
        result = _keyword_cluster([], n_topics=3)
        assert result.topics == ()
        assert result.total_videos == 0

    def test_single_video(self):
        videos = [_make_video("v1", "unique content about programming python")]
        result = _keyword_cluster(videos, n_topics=3)
        assert result.total_videos == 1
        assert len(result.topics) >= 1


class TestClusterVideos:
    def test_empty_input(self):
        result = cluster_videos([])
        assert result.topics == ()
        assert result.total_videos == 0
        assert result.method == "empty"

    def test_keyword_fallback_sets_low_confidence(self):
        """Keyword fallback must signal confidence='low' so callers can refuse to push."""
        videos = [
            _make_video("v1", "trading stocks market analysis finance"),
            _make_video("v2", "trading options futures investment"),
        ]
        result = cluster_videos(videos, method="keyword")
        assert result.method == "keyword"
        assert result.confidence == "low"

    def test_sklearn_sets_high_confidence(self):
        """sklearn clustering (if available) sets confidence='high'."""
        videos = [
            _make_video("v1", "trading stocks market analysis finance"),
            _make_video("v2", "trading options futures investment"),
            _make_video("v3", "gardening plants flowers soil water"),
        ]
        result = cluster_videos(videos, method="sklearn")
        if result.method == "sklearn":
            assert result.confidence == "high"

    def test_format_summary_includes_warning_for_low_confidence(self):
        """format_clustering_summary must include a WARNING for low confidence."""
        from clustering import ClusteringResult, Topic
        topics = (Topic(topic_id=0, label="Test", keywords=("kw",),
                        video_ids=("v1",), video_count=1),)
        result = ClusteringResult(
            topics=topics, method="keyword", total_videos=1, confidence="low",
        )
        summary = format_clustering_summary(result)
        assert "WARNING" in summary
        assert "keyword fallback" in summary.lower()

    def test_format_summary_no_warning_for_high_confidence(self):
        """format_clustering_summary must NOT warn for high confidence."""
        from clustering import ClusteringResult, Topic
        topics = (Topic(topic_id=0, label="Test", keywords=("kw",),
                        video_ids=("v1",), video_count=1),)
        result = ClusteringResult(
            topics=topics, method="bertopic", total_videos=1, confidence="high",
        )
        summary = format_clustering_summary(result)
        assert "WARNING" not in summary

    def test_filters_error_videos(self):
        videos = [
            _make_video("v1", "content about python programming"),
            VideoInfo(id="v2", error="private", error_type="private"),
        ]
        result = cluster_videos(videos, method="keyword")
        assert result.total_videos == 1  # only the valid video

    def test_auto_falls_back_to_keyword(self):
        """Without bertopic/sklearn, auto should use keyword method."""
        videos = [
            _make_video("v1", "machine learning neural networks deep"),
            _make_video("v2", "cooking recipes food kitchen"),
        ]
        result = cluster_videos(videos, method="auto")
        # Should produce at least one topic
        assert result.method in ("bertopic", "sklearn", "keyword")
        assert len(result.topics) >= 1

    def test_force_keyword_method(self):
        videos = [
            _make_video("v1", "trading stocks market analysis finance"),
            _make_video("v2", "trading options futures investment"),
            _make_video("v3", "gardening plants flowers soil water"),
        ]
        result = cluster_videos(videos, method="keyword")
        assert result.method == "keyword"

    def test_force_sklearn_method(self):
        """If sklearn is installed, should use it; otherwise graceful fallback."""
        videos = [
            _make_video("v1", "trading stocks market analysis finance"),
            _make_video("v2", "trading options futures investment"),
            _make_video("v3", "gardening plants flowers soil water"),
        ]
        result = cluster_videos(videos, method="sklearn")
        # Either sklearn worked, or it fell back to keyword
        assert result.method in ("sklearn", "keyword")

    def test_topics_have_video_ids(self):
        videos = [
            _make_video("v1", "python programming code software"),
            _make_video("v2", "python django flask web framework"),
        ]
        result = cluster_videos(videos, method="keyword")
        all_ids = set()
        for t in result.topics:
            all_ids.update(t.video_ids)
        assert "v1" in all_ids
        assert "v2" in all_ids


class TestTopicDataclass:
    def test_immutable(self):
        t = Topic(topic_id=0, label="Test", keywords=("a", "b"),
                  video_ids=("v1",), video_count=1)
        with pytest.raises(Exception):
            t.label = "Other"  # frozen

    def test_fields(self):
        t = Topic(topic_id=0, label="Test", keywords=("a", "b"),
                  video_ids=("v1", "v2"), video_count=2)
        assert t.topic_id == 0
        assert t.video_count == 2
        assert t.keywords == ("a", "b")


class TestFormatClusteringSummary:
    def test_empty_result(self):
        result = ClusteringResult(topics=(), method="empty", total_videos=0)
        summary = format_clustering_summary(result)
        assert "No topics" in summary

    def test_includes_topic_labels(self):
        topics = (
            Topic(topic_id=0, label="Bitcoin", keywords=("bitcoin", "crypto"),
                  video_ids=("v1", "v2"), video_count=2),
            Topic(topic_id=1, label="Cooking", keywords=("recipe", "pasta"),
                  video_ids=("v3",), video_count=1),
        )
        result = ClusteringResult(topics=topics, method="keyword", total_videos=3)
        summary = format_clustering_summary(result)
        assert "Bitcoin" in summary
        assert "Cooking" in summary
        assert "2 videos" in summary
        assert "keyword method" in summary

    def test_truncates_long_video_lists(self):
        video_ids = tuple(f"v{i}" for i in range(15))
        topics = (
            Topic(topic_id=0, label="Big", keywords=("kw",),
                  video_ids=video_ids, video_count=15),
        )
        result = ClusteringResult(topics=topics, method="keyword", total_videos=15)
        summary = format_clustering_summary(result)
        assert "and 5 more" in summary
