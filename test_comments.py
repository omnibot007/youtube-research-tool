"""Tests for comment thread ingestion (Phase 3.5).

Tests cover:
  - Comment parsing from yt-dlp dicts
  - Thread tree building (replies nested)
  - Formatting for Open Notebook
  - Source payload mapping
  - Edge cases (empty, pinned, hearted)
"""
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from models import VideoInfo
from comments import (
    Comment, CommentThread, extract_comments, format_comments_for_notebook,
    comments_to_source_payload, _parse_comment, _count_replies, _format_thread,
)


class TestParseComment:
    def test_basic_comment(self):
        raw = {
            "author": "Alice",
            "text": "Great video!",
            "like_count": 42,
            "timestamp": "2 weeks ago",
            "id": "c1",
        }
        c = _parse_comment(raw)
        assert c.author == "Alice"
        assert c.text == "Great video!"
        assert c.likes == 42
        assert c.timestamp == "2 weeks ago"
        assert c.comment_id == "c1"
        assert c.replies == ()

    def test_pinned_and_hearted(self):
        raw = {"author": "Bob", "text": "Pinned!", "is_pinned": True, "is_hearted": True}
        c = _parse_comment(raw)
        assert c.is_pinned is True
        assert c.is_hearted is True

    def test_with_replies(self):
        raw = {
            "author": "Alice",
            "text": "Top comment",
            "replies": [
                {"author": "Bob", "text": "Reply 1"},
                {"author": "Carol", "text": "Reply 2"},
            ],
        }
        c = _parse_comment(raw)
        assert len(c.replies) == 2
        assert c.replies[0].author == "Bob"
        assert c.replies[1].author == "Carol"

    def test_missing_fields_default(self):
        c = _parse_comment({})
        assert c.author == ""
        assert c.text == ""
        assert c.likes == 0
        assert c.replies == ()


class TestCountReplies:
    def test_no_replies(self):
        c = Comment(author="A", text="t", likes=0, timestamp="")
        assert _count_replies(c) == 0

    def test_flat_replies(self):
        c = Comment(author="A", text="t", likes=0, timestamp="",
                    replies=(Comment(author="B", text="r", likes=0, timestamp=""),
                             Comment(author="C", text="r2", likes=0, timestamp="")))
        assert _count_replies(c) == 2

    def test_nested_replies(self):
        c = Comment(author="A", text="t", likes=0, timestamp="",
                    replies=(Comment(author="B", text="r", likes=0, timestamp="",
                                     replies=(Comment(author="D", text="rr", likes=0, timestamp=""),)),))
        assert _count_replies(c) == 2  # B + D


class TestCommentThread:
    def test_total_messages_no_replies(self):
        root = Comment(author="A", text="t", likes=0, timestamp="")
        thread = CommentThread(root=root, reply_count=0)
        assert thread.total_messages == 1

    def test_total_messages_with_replies(self):
        root = Comment(author="A", text="t", likes=0, timestamp="",
                       replies=(Comment(author="B", text="r", likes=0, timestamp=""),))
        thread = CommentThread(root=root, reply_count=1)
        assert thread.total_messages == 2


class TestFormatCommentsForNotebook:
    def test_empty_comments(self):
        result = format_comments_for_notebook([])
        assert "no comments" in result.lower()

    def test_includes_video_title(self):
        video = VideoInfo(id="abc", title="My Video")
        comments = [Comment(author="Alice", text="Great!", likes=5, timestamp="1d ago")]
        result = format_comments_for_notebook(comments, video)
        assert "My Video" in result
        assert "Alice" in result
        assert "Great!" in result
        assert "5 likes" in result

    def test_indents_replies(self):
        comments = [
            Comment(author="Alice", text="Top", likes=10, timestamp="",
                    replies=(Comment(author="Bob", text="Reply", likes=2, timestamp=""),)),
        ]
        result = format_comments_for_notebook(comments)
        assert "@Alice" in result
        assert "@Bob" in result
        # Reply should be indented (2 spaces)
        lines = result.split("\n")
        reply_line = next(l for l in lines if "@Bob" in l)
        assert reply_line.startswith("  @Bob")

    def test_marks_pinned_and_hearted(self):
        comments = [
            Comment(author="Alice", text="Pinned!", likes=100, timestamp="",
                    is_pinned=True, is_hearted=True),
        ]
        result = format_comments_for_notebook(comments)
        assert "PINNED" in result
        assert "HEARTED" in result

    def test_truncates_long_lists(self):
        comments = [
            Comment(author=f"User{i}", text=f"Comment {i}", likes=0, timestamp="")
            for i in range(60)
        ]
        result = format_comments_for_notebook(comments, max_threads=50)
        assert "and 10 more" in result


class TestCommentsToSourcePayload:
    def test_basic_payload(self):
        video = VideoInfo(id="abc", title="Test Video", url="https://youtube.com/watch?v=abc")
        comments = [Comment(author="Alice", text="Great!", likes=5, timestamp="1d ago")]
        payload = comments_to_source_payload(comments, video, notebook_id="nb:1")

        assert payload["type"] == "text"
        assert "Comments: Test Video" in payload["title"]
        assert "nb:1" in payload["notebooks"]
        assert payload["embed"] is True
        assert "Alice" in payload["content"]

    def test_no_notebook(self):
        video = VideoInfo(id="abc", title="Test")
        comments = [Comment(author="A", text="t", likes=0, timestamp="")]
        payload = comments_to_source_payload(comments, video)
        assert payload["notebooks"] == []


class TestExtractComments:
    def test_returns_empty_on_import_error(self):
        """If yt-dlp isn't installed, should return empty list (not raise)."""
        with patch.dict(sys.modules, {"yt_dlp": None}):
            result = extract_comments("abc123")
            assert result == []

    def test_parses_yt_dlp_comments(self):
        """Should parse comments from yt-dlp's extract_info output."""
        mock_ydl = MagicMock()
        mock_ydl.extract_info.return_value = {
            "comments": [
                {"author": "Alice", "text": "Great!", "like_count": 5, "id": "c1"},
                {"author": "Bob", "text": "Nice.", "like_count": 2, "id": "c2"},
            ],
        }
        mock_ydl.__enter__ = MagicMock(return_value=mock_ydl)
        mock_ydl.__exit__ = MagicMock(return_value=False)

        with patch("yt_dlp.YoutubeDL", return_value=mock_ydl):
            comments = extract_comments("abc123", max_comments=10)

        assert len(comments) == 2
        assert comments[0].author == "Alice"
        assert comments[1].author == "Bob"

    def test_returns_empty_on_error(self):
        """yt-dlp errors should return empty list, not raise."""
        mock_ydl = MagicMock()
        mock_ydl.extract_info.side_effect = Exception("Network error")
        mock_ydl.__enter__ = MagicMock(return_value=mock_ydl)
        mock_ydl.__exit__ = MagicMock(return_value=False)

        with patch("yt_dlp.YoutubeDL", return_value=mock_ydl):
            comments = extract_comments("abc123")
        assert comments == []

    def test_handles_no_comments_field(self):
        """Videos with comments disabled return empty."""
        mock_ydl = MagicMock()
        mock_ydl.extract_info.return_value = {"id": "abc", "title": "Test"}
        mock_ydl.__enter__ = MagicMock(return_value=mock_ydl)
        mock_ydl.__exit__ = MagicMock(return_value=False)

        with patch("yt_dlp.YoutubeDL", return_value=mock_ydl):
            comments = extract_comments("abc123")
        assert comments == []


class TestCommentImmutability:
    def test_comment_is_frozen(self):
        c = Comment(author="A", text="t", likes=0, timestamp="")
        with pytest.raises(Exception):
            c.author = "B"

    def test_to_dict_serializes_replies(self):
        c = Comment(author="A", text="top", likes=5, timestamp="1d",
                    replies=(Comment(author="B", text="reply", likes=1, timestamp="2d"),))
        d = c.to_dict()
        assert d["author"] == "A"
        assert len(d["replies"]) == 1
        assert d["replies"][0]["author"] == "B"
