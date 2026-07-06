"""YouTube comment thread ingestion.

Extracts comments (with replies and threading) from a YouTube video using
yt-dlp's built-in comment support, then formats them for ingestion into
Open Notebook as a searchable comment source.

yt-dlp options for comments:
  - getcomments=True: fetch comments
  - maxcomments=N: limit total comments (including replies)
  - maxcommentdepth=N: limit reply nesting depth

Extracted as a separate module (Phase 3.5).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from models import VideoInfo


@dataclass(frozen=True)
class Comment:
    """A single YouTube comment with optional replies."""
    author: str
    text: str
    likes: int
    timestamp: str  # ISO-ish relative timestamp from yt-dlp ("2 weeks ago")
    comment_id: str = ""
    is_pinned: bool = False
    is_hearted: bool = False
    replies: tuple["Comment", ...] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        return {
            "author": self.author,
            "text": self.text,
            "likes": self.likes,
            "timestamp": self.timestamp,
            "comment_id": self.comment_id,
            "is_pinned": self.is_pinned,
            "is_hearted": self.is_hearted,
            "replies": [r.to_dict() for r in self.replies],
        }


@dataclass(frozen=True)
class CommentThread:
    """A top-level comment with its reply tree."""
    root: Comment
    reply_count: int

    @property
    def total_messages(self) -> int:
        """Total messages in this thread (root + all replies)."""
        return 1 + _count_replies(self.root)


def _count_replies(comment: Comment) -> int:
    """Recursively count all replies in a comment tree."""
    count = len(comment.replies)
    for reply in comment.replies:
        count += _count_replies(reply)
    return count


def extract_comments(
    video_id_or_url: str,
    max_comments: int = 100,
    max_depth: int = 2,
    cookies_file: Optional[str] = None,
    cookies_from_browser: Optional[str] = None,
    proxy: Optional[str] = None,
) -> list[Comment]:
    """Extract comments from a YouTube video using yt-dlp.

    Args:
        video_id_or_url: YouTube video ID or URL.
        max_comments: Max total comments to fetch (including replies).
        max_depth: Max reply nesting depth.
        cookies_file: Path to cookies file (for age-restricted/private videos).
        cookies_from_browser: Browser name to extract cookies from.
        proxy: Proxy URL.

    Returns:
        List of top-level Comment objects (each may have replies).
    """
    try:
        import yt_dlp
    except ImportError:
        return []

    opts = {
        "getcomments": True,
        "maxcomments": max_comments,
        "maxcommentdepth": max_depth,
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "extract_flat": True,
    }
    if cookies_file:
        opts["cookiefile"] = cookies_file
    if cookies_from_browser:
        opts["cookiesfrombrowser"] = (cookies_from_browser,)
    if proxy:
        opts["proxy"] = proxy

    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            data = ydl.extract_info(video_id_or_url, download=False)
    except Exception:
        return []

    if not isinstance(data, dict):
        return []

    raw_comments = data.get("comments") or []
    return [_parse_comment(c) for c in raw_comments if c]


def _parse_comment(raw: dict) -> Comment:
    """Parse a yt-dlp comment dict into a Comment dataclass.

    yt-dlp returns flat comments with parent_id fields; we rebuild the tree.
    For simplicity, this returns top-level comments with their replies nested.
    """
    author = raw.get("author", "") or raw.get("author_id", "")
    text = raw.get("text", "") or raw.get("content", "")
    likes = raw.get("like_count", 0) or 0
    timestamp = raw.get("timestamp", "") or raw.get("time_text", "") or ""
    comment_id = raw.get("id", "") or raw.get("comment_id", "")
    is_pinned = bool(raw.get("is_pinned", False))
    is_hearted = bool(raw.get("is_hearted", False))

    # yt-dlp may nest replies under "replies" or return them flat
    raw_replies = raw.get("replies") or []
    replies = tuple(_parse_comment(r) for r in raw_replies if r)

    return Comment(
        author=author, text=text, likes=likes, timestamp=timestamp,
        comment_id=comment_id, is_pinned=is_pinned, is_hearted=is_hearted,
        replies=replies,
    )


def format_comments_for_notebook(
    comments: list[Comment],
    video: Optional[VideoInfo] = None,
    max_threads: int = 50,
) -> str:
    """Format comments as a text block suitable for Open Notebook ingestion.

    Threads are rendered with indentation to preserve reply structure.
    Pinned and hearted comments are marked. Like counts are included so the
    AI can gauge community sentiment.
    """
    if not comments:
        return "(no comments available)"

    lines: list[str] = []
    if video:
        lines.append(f"--- COMMENTS on: {video.title or video.id} ---")
        if video.url:
            lines.append(f"Video URL: {video.url}")
        lines.append("")

    for i, comment in enumerate(comments[:max_threads]):
        thread_lines = _format_thread(comment, indent=0)
        lines.append(thread_lines)
        lines.append("")

    total = len(comments)
    if total > max_threads:
        lines.append(f"... and {total - max_threads} more top-level threads")

    return "\n".join(lines)


def _format_thread(comment: Comment, indent: int) -> str:
    """Recursively format a comment and its replies with indentation."""
    prefix = "  " * indent
    markers = []
    if comment.is_pinned:
        markers.append("PINNED")
    if comment.is_hearted:
        markers.append("HEARTED")
    marker_str = f" [{', '.join(markers)}]" if markers else ""

    lines: list[str] = [
        f"{prefix}@{comment.author} ({comment.likes} likes){marker_str}: {comment.text}",
    ]
    for reply in comment.replies:
        lines.append(_format_thread(reply, indent + 1))
    return "\n".join(lines)


def comments_to_source_payload(
    comments: list[Comment],
    video: VideoInfo,
    notebook_id: str = "",
    embed: bool = True,
) -> dict:
    """Build an Open Notebook SourceCreate payload for a comment source.

    The comments become a separate text source (distinct from the transcript)
    so the AI can distinguish "what was said in the video" from "what the
    audience said about it."
    """
    content = format_comments_for_notebook(comments, video)
    title = f"Comments: {video.title or video.id} [{video.id}]"

    notebooks: list[str] = [notebook_id] if notebook_id else []

    return {
        "type": "text",
        "title": title,
        "content": content,
        "embed": embed,
        "async_processing": True,
        "notebooks": notebooks,
    }
