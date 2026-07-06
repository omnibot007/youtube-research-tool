"""Open Notebook integration client.

Pushes scraped YouTube content (transcripts, metadata, claims) into a running
Open Notebook instance via its REST API (port 5055 by default).

Verified against lfnovo/open-notebook main branch:
  - POST /api/sources/json  — create source (JSON payload, SourceCreate schema)
  - GET  /api/notebooks     — list notebooks
  - GET  /api/sources/{id}/status — processing status

SourceCreate schema (api/models.py:297):
  notebook_id: str | None        # deprecated, use `notebooks`
  notebooks: list[str] | None    # multi-notebook support
  type: str                      # "link" | "upload" | "text"
  url: str | None                # for link type
  file_path: str | None          # for upload type
  content: str | None            # for text type
  title: str | None
  transformations: list[str]     # transformation IDs
  embed: bool                    # embed for vector search
  delete_source: bool            # delete uploaded file after processing
  async_processing: bool         # process asynchronously

Usage:
    from open_notebook import OpenNotebookClient
    client = OpenNotebookClient()  # reads OPEN_NOTEBOOK_URL + OPEN_NOTEBOOK_API_KEY env
    result = client.push_video(video, segments, claims, notebook_id="notebook:abc123")
"""
from __future__ import annotations

import os
import random
import threading
import time
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Optional

# urllib is in the stdlib — avoids adding httpx as a hard dependency.
# yt_scrape.py uses yt-dlp for HTTP; we keep this module dependency-light.
import urllib.request
import urllib.error
import json as _json


def _env_base_url() -> str:
    return os.getenv("OPEN_NOTEBOOK_URL", "http://localhost:5055")


def _env_api_key() -> str:
    return os.getenv("OPEN_NOTEBOOK_API_KEY", "")


def _env_password() -> str:
    return os.getenv("OPEN_NOTEBOOK_PASSWORD", "")


DEFAULT_TIMEOUT = 60  # seconds — source creation may trigger content processing
MAX_CONTENT_BYTES = 128 * 1024  # 128KB — safe under FastAPI's default 1MB body limit
                                   # leaves room for metadata + JSON overhead


@dataclass(frozen=True)
class OpenNotebookConfig:
    """Immutable configuration for the Open Notebook client."""
    base_url: str = field(default_factory=_env_base_url)
    api_key: str = field(default_factory=_env_api_key)
    timeout: int = DEFAULT_TIMEOUT
    password: str = field(default_factory=_env_password)


def build_timestamped_transcript(segments: list) -> str:
    """Render TranscriptSegments as text with inline [HH:MM:SS] timestamps.

    Inline timestamps (vs a structured array) mean every chunk Open Notebook
    splits out carries its own citation anchor — critical for chat-with-context.
    """
    lines: list[str] = []
    for seg in segments:
        # TranscriptSegment has start (float seconds) and text
        start = getattr(seg, "start", 0.0) or 0.0
        text = getattr(seg, "text", "") or ""
        if not text.strip():
            continue
        ts = _format_timestamp(start)
        lines.append(f"[{ts}] {text.strip()}")
    return "\n".join(lines)


def _format_timestamp(seconds: Optional[float]) -> str:
    """Format seconds as HH:MM:SS (or MM:SS if under an hour). None-safe."""
    if seconds is None:
        return "00:00"
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def video_info_to_source_payload(
    video,
    segments: list,
    claims: Optional[list[dict]] = None,
    notebook_id: str = "",
    extra_notebooks: Optional[list[str]] = None,
    embed: bool = True,
    async_processing: bool = False,
    extra_content: str = "",
) -> dict:
    """Map a scraped VideoInfo + TranscriptSegments to Open Notebook's SourceCreate.

    Returns a dict ready to POST to /api/sources/json. Uses type="text" with the
    transcript as content (gives us full control vs relying on Open Notebook's
    built-in YouTube URL handler). The original YouTube URL is preserved in
    the title metadata for traceability.

    Args:
        extra_content: Optional extra text appended to the content (e.g. diarized
            transcript). Useful for adding derived content without mutating the video.
    """
    transcript_text = build_timestamped_transcript(segments) if segments else ""
    title = video.title or f"YouTube: {video.id}"
    # Append video ID to title for traceability in the Open Notebook UI
    if video.id and video.id not in title:
        title = f"{title} [{video.id}]"

    notebooks: list[str] = list(extra_notebooks or [])
    if notebook_id:
        notebooks.insert(0, notebook_id)

    payload: dict[str, Any] = {
        "type": "text",
        "title": title,
        "content": transcript_text or "(no transcript available)",
        "embed": embed,
        "async_processing": async_processing,
        "notebooks": notebooks,
    }

    # Stash rich metadata in the title isn't possible — Open Notebook's
    # SourceCreate schema doesn't have a metadata field. We encode channel +
    # claims as a footer block in the content so they're searchable.
    metadata_footer = _build_metadata_footer(video, claims or [])
    if metadata_footer:
        payload["content"] = payload["content"] + "\n\n" + metadata_footer

    # Append any extra derived content (e.g. diarized transcript)
    if extra_content:
        payload["content"] = payload["content"] + "\n\n" + extra_content

    return payload


def _build_metadata_footer(video, claims: list[dict]) -> str:
    """Build a searchable metadata footer appended to the transcript content."""
    lines: list[str] = ["--- METADATA ---"]
    if video.url:
        lines.append(f"Source URL: {video.url}")
    elif video.id:
        lines.append(f"Source URL: https://www.youtube.com/watch?v={video.id}")
    if video.channel:
        lines.append(f"Channel: {video.channel}")
    if video.duration:
        lines.append(f"Duration: {video.duration}s")
    if video.upload_date:
        lines.append(f"Upload date: {video.upload_date}")
    if video.transcript_lang:
        lines.append(f"Transcript language: {video.transcript_lang}")
    if video.transcript_source:
        lines.append(f"Transcript source: {video.transcript_source}")

    if claims:
        lines.append("")
        lines.append(f"--- CLAIMS ({len(claims)}) ---")
        for c in claims[:50]:  # cap at 50 to avoid huge payloads
            claim_text = c.get("claim") or c.get("text") or ""
            ts_raw = c.get("timestamp") or c.get("source_timestamp") or ""
            # timestamp can be a dict {start, end, timestamp_str} or a plain string
            if isinstance(ts_raw, dict):
                ts_str = ts_raw.get("timestamp") or ts_raw.get("timestamp_str") or ""
            else:
                ts_str = str(ts_raw) if ts_raw else ""
            conf = c.get("confidence") or c.get("strength", "")
            yt_url = c.get("youtube_url", "")
            line = f"- {claim_text}"
            if ts_str:
                line += f" (at {ts_str})"
            if conf:
                line += f" [{conf}]"
            if yt_url:
                line += f" → {yt_url}"
            lines.append(line)

    return "\n".join(lines)


class OpenNotebookError(Exception):
    """Raised when the Open Notebook API returns an error."""


class PayloadTooLargeError(OpenNotebookError):
    """Raised when a source payload exceeds MAX_CONTENT_BYTES and cannot be chunked."""


def _thread_lock() -> threading.Lock:
    """Factory for a fresh Lock (kept as function so tests can patch if needed)."""
    return threading.Lock()


def _split_content(content: str, chunk_size: int) -> list[str]:
    """Split content into chunks at paragraph boundaries, each under chunk_size bytes.

    Splits on double-newlines (paragraph breaks) first, then on single newlines
    if a paragraph itself exceeds the chunk size. Preserves readability.
    """
    if len(content.encode("utf-8")) <= chunk_size:
        return [content]

    chunks: list[str] = []
    current: list[str] = []
    current_bytes = 0

    paragraphs = content.split("\n\n")
    for para in paragraphs:
        para_bytes = len(para.encode("utf-8")) + 2  # +2 for the \n\n
        # If a single paragraph exceeds chunk_size, split it on single newlines
        if para_bytes > chunk_size:
            if current:
                chunks.append("\n\n".join(current))
                current = []
                current_bytes = 0
            for line in para.split("\n"):
                line_bytes = len(line.encode("utf-8")) + 1
                if current_bytes + line_bytes > chunk_size and current:
                    chunks.append("\n".join(current))
                    current = []
                    current_bytes = 0
                current.append(line)
                current_bytes += line_bytes
            continue

        if current_bytes + para_bytes > chunk_size and current:
            chunks.append("\n\n".join(current))
            current = []
            current_bytes = 0
        current.append(para)
        current_bytes += para_bytes

    if current:
        chunks.append("\n\n".join(current) if "\n\n" not in "".join(current) else "\n".join(current))

    return chunks


@dataclass
class OpenNotebookClient:
    """REST client for pushing scraped content into Open Notebook.

    Uses urllib (stdlib) to avoid adding httpx as a hard dependency. For async
    use, see push_video_async (requires httpx).
    """
    config: OpenNotebookConfig = field(default_factory=OpenNotebookConfig)

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.config.api_key:
            h["Authorization"] = f"Bearer {self.config.api_key}"
        if self.config.password:
            h["X-Notebook-Password"] = self.config.password
        return h

    def _request(
        self,
        method: str,
        path: str,
        body: Optional[dict] = None,
        params: Optional[dict] = None,
        retries: int = 3,
    ) -> Any:
        """Execute an HTTP request with exponential backoff retry.

        Retries on transient errors: 5xx HTTP responses and URLError (network/
        connection issues). 4xx errors are not retried (client errors are
        deterministic — retrying won't help). Backoff: 1s, 2s, 4s.
        """
        url = f"{self.config.base_url.rstrip('/')}{path}"
        if params:
            from urllib.parse import urlencode
            url = f"{url}?{urlencode(params)}"
        data = _json.dumps(body).encode("utf-8") if body is not None else None

        last_exc: Optional[Exception] = None
        for attempt in range(retries):
            req = urllib.request.Request(
                url, data=data, method=method, headers=self._headers(),
            )
            try:
                with urllib.request.urlopen(req, timeout=self.config.timeout) as resp:
                    raw = resp.read().decode("utf-8")
                    if not raw:
                        return None
                    return _json.loads(raw)
            except urllib.error.HTTPError as exc:
                detail = ""
                try:
                    detail = exc.read().decode("utf-8", errors="replace")
                except Exception:
                    pass
                # 429 Too Many Requests: respect Retry-After header, then retry
                if exc.code == 429 and attempt < retries - 1:
                    retry_after = exc.headers.get("Retry-After") if exc.headers else None
                    try:
                        wait = int(retry_after) if retry_after else 60
                    except (ValueError, TypeError):
                        wait = 60
                    time.sleep(wait + random.uniform(0, 1))  # jitter
                    last_exc = exc
                    continue
                # 5xx (server errors): retry with exponential backoff + jitter
                if 500 <= exc.code < 600 and attempt < retries - 1:
                    last_exc = exc
                    time.sleep(2 ** attempt + random.uniform(0, 1))  # 1s+j, 2s+j, 4s+j
                    continue
                raise OpenNotebookError(
                    f"Open Notebook API {method} {path} failed: HTTP {exc.code} — {detail}"
                ) from exc
            except urllib.error.URLError as exc:
                # Network/connection errors — retry with exponential backoff + jitter
                if attempt < retries - 1:
                    last_exc = exc
                    time.sleep(2 ** attempt + random.uniform(0, 1))
                    continue
                raise OpenNotebookError(
                    f"Cannot reach Open Notebook at {self.config.base_url} ({exc.reason}). "
                    "Is the server running? Set OPEN_NOTEBOOK_URL if it's on a different host/port."
                ) from exc
        # Should not reach here, but be explicit
        raise OpenNotebookError(
            f"Open Notebook API {method} {path} failed after {retries} retries: {last_exc}"
        )

    # ---------------------------------------------------------- notebooks

    def list_notebooks(self) -> list[dict]:
        """List all notebooks. Returns raw API response (list of dicts)."""
        result = self._request("GET", "/api/notebooks")
        if isinstance(result, list):
            return result
        return [result] if result else []

    def find_notebook_by_name(self, name: str) -> Optional[dict]:
        """Find a notebook by exact name (case-insensitive). Returns first match."""
        target = name.lower().strip()
        for nb in self.list_notebooks():
            nb_name = (nb.get("name") or nb.get("title") or "").lower()
            if nb_name == target:
                return nb
        return None

    # ------------------------------------------------------------- sources

    def create_source(self, payload: dict) -> dict:
        """Create a source from a pre-built SourceCreate payload.

        Raises PayloadTooLargeError if the content exceeds MAX_CONTENT_BYTES.
        Use create_source_chunked() for large transcripts.
        """
        content_bytes = (payload.get("content") or "").encode("utf-8")
        if len(content_bytes) > MAX_CONTENT_BYTES:
            raise PayloadTooLargeError(
                f"Content is {len(content_bytes)} bytes (limit: {MAX_CONTENT_BYTES}). "
                "Use create_source_chunked() or reduce content size."
            )
        result = self._request("POST", "/api/sources/json", body=payload)
        if isinstance(result, list):
            return result[0] if result else {}
        return result or {}

    def create_source_chunked(self, payload: dict, chunk_size: int = 100_000) -> list[dict]:
        """Push a large source in multiple chunks (part 1/N, part 2/N, ...).

        Each chunk becomes a separate source in the notebook. Titles get a
        "(part N/M)" suffix so they group in the UI.
        """
        content = payload.get("content") or ""
        if len(content.encode("utf-8")) <= MAX_CONTENT_BYTES:
            return [self.create_source(payload)]

        # Split content at paragraph boundaries near the chunk size
        chunks = _split_content(content, chunk_size)
        total = len(chunks)
        base_title = payload.get("title", "Source")
        created: list[dict] = []
        for i, chunk in enumerate(chunks, 1):
            chunk_payload = dict(payload)
            chunk_payload["content"] = chunk
            chunk_payload["title"] = f"{base_title} (part {i}/{total})"
            result = self._request("POST", "/api/sources/json", body=chunk_payload)
            if isinstance(result, list):
                created.append(result[0] if result else {})
            else:
                created.append(result or {})
        return created

    def push_video(
        self,
        video,
        segments: list,
        claims: Optional[list[dict]] = None,
        notebook_id: str = "",
        extra_notebooks: Optional[list[str]] = None,
        embed: bool = True,
        async_processing: bool = False,
        extra_content: str = "",
    ) -> dict:
        """Scrape → map → push a single video to Open Notebook.

        Returns the created source dict (with `id`). For large transcripts that
        exceed MAX_CONTENT_BYTES, automatically chunks and returns the first chunk.
        """
        payload = video_info_to_source_payload(
            video, segments, claims,
            notebook_id=notebook_id,
            extra_notebooks=extra_notebooks,
            embed=embed,
            async_processing=async_processing,
            extra_content=extra_content,
        )
        content_bytes = len((payload.get("content") or "").encode("utf-8"))
        if content_bytes > MAX_CONTENT_BYTES:
            chunks = self.create_source_chunked(payload)
            return chunks[0] if chunks else {}
        return self.create_source(payload)

    def get_source_status(self, source_id: str) -> dict:
        """Check processing status of a source (async_processing=True)."""
        return self._request("GET", f"/api/sources/{source_id}/status")

    def wait_for_source(
        self,
        source_id: str,
        timeout: int = 300,
        poll_interval: int = 3,
    ) -> dict:
        """Poll source status until processing completes or timeout.

        Returns the final status dict. Raises OpenNotebookError on failure
        or timeout.
        """
        deadline = time.time() + timeout
        last: dict = {}
        while time.time() < deadline:
            last = self.get_source_status(source_id)
            status = (last.get("status") or "").lower()
            if status in ("completed", "done", "success", "ready"):
                return last
            if status in ("failed", "error"):
                raise OpenNotebookError(
                    f"Source {source_id} processing failed: {last.get('error', 'unknown')}"
                )
            time.sleep(poll_interval)
        raise OpenNotebookError(
            f"Source {source_id} did not complete within {timeout}s (last status: {last})"
        )

    # --------------------------------------------------- channel-wide ingest

    def push_channel(
        self,
        channel_url: str,
        notebook_id: str,
        *,
        limit: int = 10,
        concurrency: int = 3,
        with_claims: bool = False,
        embed: bool = True,
        async_processing: bool = False,
        langs: Optional[list[str]] = None,
        cookies_file: Optional[str] = None,
        cookies_from_browser: Optional[str] = None,
        proxy: Optional[str] = None,
        retries: int = 3,
        rate_limit_sec: float = 2.0,
        on_progress: Optional[Any] = None,
        skip_existing: bool = True,
    ) -> dict:
        """Scrape all videos from a channel and push each to Open Notebook.

        This is the killer use case: one command, one notebook, all videos.
        Open Notebook's chat and search then work across the entire channel.

        Args:
            channel_url: YouTube channel URL or ID.
            notebook_id: Target notebook ID.
            limit: Max videos to scrape (default 10).
            concurrency: Parallel workers for the PUSH phase (default 3).
                Scrape phase is sequential (scrape_channel handles rate limiting).
            with_claims: Extract enriched claims for each video.
            on_progress: Optional callback(progress_dict) called after each video.
            skip_existing: If True, skip videos already in the notebook (dedup by title).

        Returns:
            Summary dict: {pushed, failed, skipped, total, source_ids, errors}
        """
        # Lazy import to avoid circular dependency at module load time
        from yt_scrape import scrape_channel, extract_claims_for_video
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # 1. Scrape channel metadata + transcripts (sequential — scrape_channel
        #    handles its own rate limiting; parallelizing channel listing is risky)
        videos = scrape_channel(
            channel_url, limit=limit, transcripts=True, langs=langs,
            cookies_file=cookies_file, cookies_from_browser=cookies_from_browser,
            proxy=proxy, retries=retries, rate_limit_sec=rate_limit_sec,
        )

        total = len(videos)

        # 2. Dedup: skip videos already in the notebook (by title match)
        existing_titles: set[str] = set()
        if skip_existing:
            try:
                existing_titles = self._find_existing_titles(notebook_id)
            except OpenNotebookError:
                pass  # if we can't list sources, just push (may create dups)

        # 3. Build push tasks (skip errored + already-existing videos)
        tasks: list[tuple[int, VideoInfo, list[dict]]] = []
        skipped = 0
        for i, video in enumerate(videos, 1):
            if video.error:
                if on_progress:
                    on_progress({"done": i, "total": total, "status": "skipped",
                                 "video_id": video.id, "reason": "scrape_error"})
                skipped += 1
                continue
            title_key = self._title_key(video)
            if title_key in existing_titles:
                if on_progress:
                    on_progress({"done": i, "total": total, "status": "skipped",
                                 "video_id": video.id, "reason": "already_exists"})
                skipped += 1
                continue

            claims: list[dict] = []
            if with_claims and video.segments:
                try:
                    claims = extract_claims_for_video(video) or []
                except Exception:
                    pass  # claim extraction is best-effort

            tasks.append((i, video, claims))

        # 4. Push in parallel with ThreadPoolExecutor (urllib is synchronous,
        #    so threads are the right concurrency model here)
        pushed: list[str] = []
        errors: list[dict] = []
        lock = _thread_lock()

        def do_push(task: tuple) -> tuple[int, str, dict, Optional[OpenNotebookError]]:
            idx, vid, vid_claims = task
            try:
                result = self.push_video(
                    vid, vid.segments, claims=vid_claims,
                    notebook_id=notebook_id,
                    embed=embed,
                    async_processing=async_processing,
                )
                return idx, vid.id, result, None
            except OpenNotebookError as exc:
                return idx, vid.id, {}, exc

        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
            futures = {pool.submit(do_push, t): t for t in tasks}
            for fut in as_completed(futures):
                idx, vid_id, result, exc = fut.result()
                with lock:
                    if exc:
                        errors.append({"video_id": vid_id, "error": str(exc)})
                        status = "failed"
                    else:
                        source_id = result.get("id", "?")
                        pushed.append(source_id)
                        status = "pushed"
                if on_progress:
                    on_progress({"done": idx, "total": total, "status": status,
                                 "video_id": vid_id, "source_id": result.get("id")})

        return {
            "pushed": len(pushed),
            "failed": len(errors),
            "skipped": skipped,
            "total": total,
            "source_ids": pushed,
            "errors": errors,
        }

    @staticmethod
    def _title_key(video) -> str:
        """Build a normalized title key for dedup matching.

        Uses unicodedata.normalize('NFC') + casefold() for robust matching
        across Unicode forms and case variants (e.g. "Héllo" vs "héllo").
        """
        base = video.title or f"YouTube: {video.id}"
        full = f"{base} [{video.id}]" if video.id and video.id not in base else base
        return unicodedata.normalize("NFC", full).casefold().strip()

    def _find_existing_titles(self, notebook_id: str) -> set[str]:
        """List existing source titles in a notebook for dedup matching.

        Note: GET /api/sources may be paginated. This currently reads only the
        first page. For notebooks with >100 existing sources, dedup may miss
        entries. See KNOWN_ISSUES.md.
        """
        result = self._request("GET", "/api/sources", params={"notebook_id": notebook_id})
        if isinstance(result, list):
            return {
                unicodedata.normalize("NFC", s.get("title") or "").casefold().strip()
                for s in result if s.get("title")
            }
        return set()
