"""Topic clustering across videos.

Groups scraped videos by latent topic using either BERTopic (if installed) or
a lightweight TF-IDF + k-means fallback (stdlib + scikit-learn if available,
pure-stdlib fallback otherwise). The resulting topic summaries can be pushed
to Open Notebook as a synthetic "summary source" so the AI can answer
cross-video questions like "what does this channel think about X?"

Extracted as a separate module (Phase 3.3) — optional dependency design:
  - BERTopic: best quality, heavy install (pip install bertopic)
  - scikit-learn: good quality, medium install (pip install scikit-learn)
  - pure stdlib: always available, basic keyword-based clustering
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class Topic:
    """A cluster of videos sharing a latent topic."""
    topic_id: int
    label: str
    keywords: tuple[str, ...]
    video_ids: tuple[str, ...]
    video_count: int


@dataclass(frozen=True)
class ClusteringResult:
    """Result of clustering a set of videos by topic."""
    topics: tuple[Topic, ...]
    method: str  # "bertopic" | "sklearn" | "keyword"
    total_videos: int
    confidence: str = "high"  # "high" (bertopic/sklearn) | "low" (keyword fallback)


# Stopwords for the keyword fallback (small, focused on English)
_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "must", "can", "this", "that", "these",
    "those", "i", "you", "he", "she", "it", "we", "they", "what", "which",
    "who", "when", "where", "why", "how", "all", "each", "every", "both",
    "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only",
    "own", "same", "so", "than", "too", "very", "just", "also", "now", "then",
    "here", "there", "out", "up", "down", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "into", "through", "during", "before",
    "after", "above", "below", "between", "my", "your", "his", "her", "its",
    "our", "their", "me", "him", "us", "them", "am", "if", "about", "against",
    "because", "while", "of", "off", "over", "under", "again", "further",
    "once", "very", "s", "t", "don", "ve", "re", "ll", "d", "m", "y",
    "video", "watch", "channel", "subscribe", "like", "comment", "share",
    "today", "guys", "hey", "okay", "yeah", "uh", "um", "ah", "so",
})


def _tokenize(text: str) -> list[str]:
    """Split text into lowercase word tokens, dropping stopwords and short tokens."""
    tokens = re.findall(r"[a-z]{3,}", text.lower())
    return [t for t in tokens if t not in _STOPWORDS]


def _extract_video_text(video) -> str:
    """Get the text content from a VideoInfo (segments or title)."""
    if video.segments:
        return " ".join(seg.text for seg in video.segments if seg.text.strip())
    return video.title or ""


def cluster_videos(
    videos: list,
    n_topics: int = 5,
    method: str = "auto",
) -> ClusteringResult:
    """Cluster videos by latent topic.

    Args:
        videos: List of VideoInfo objects with .segments and .id.
        n_topics: Target number of topics (actual may be fewer for small inputs).
        method: "auto" (try bertopic → sklearn → keyword), or force a specific one.

    Returns:
        ClusteringResult with topics and the method used.
    """
    # Filter to videos with content
    valid = [v for v in videos if v and not v.error and _extract_video_text(v)]
    if not valid:
        return ClusteringResult(topics=(), method="empty", total_videos=0)

    if method in ("auto", "bertopic"):
        result = _try_bertopic(valid, n_topics)
        if result is not None:
            return result

    if method in ("auto", "sklearn"):
        result = _try_sklearn(valid, n_topics)
        if result is not None:
            return result

    # Always-available fallback: keyword-based clustering
    return _keyword_cluster(valid, n_topics)


def _try_bertopic(videos: list, n_topics: int) -> Optional[ClusteringResult]:
    """Attempt BERTopic clustering. Returns None if BERTopic not installed."""
    try:
        from bertopic import BERTopic
    except ImportError:
        return None

    docs = [_extract_video_text(v) for v in videos]
    try:
        model = BERTopic(nr_topics=min(n_topics, len(docs)))
        topics, _ = model.fit_transform(docs)
        return _build_result_from_assignments(videos, topics, model, "bertopic")
    except Exception:
        return None


def _try_sklearn(videos: list, n_topics: int) -> Optional[ClusteringResult]:
    """Attempt TF-IDF + k-means clustering. Returns None if sklearn not installed."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.cluster import KMeans
    except ImportError:
        return None

    docs = [_extract_video_text(v) for v in videos]
    if len(docs) < 2:
        return None

    k = min(n_topics, len(docs))
    try:
        vectorizer = TfidfVectorizer(max_features=1000, stop_words="english")
        X = vectorizer.fit_transform(docs)
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X)

        # Extract keywords per cluster from cluster centers
        feature_names = vectorizer.get_feature_names_out()
        topics: list[Topic] = []
        for cluster_id in range(k):
            mask = labels == cluster_id
            video_ids = tuple(videos[i].id for i in range(len(videos)) if mask[i])
            if not video_ids:
                continue
            # Top keywords from cluster center
            center = km.cluster_centers_[cluster_id]
            top_indices = center.argsort()[-5:][::-1]
            keywords = tuple(feature_names[i] for i in top_indices if center[i] > 0)
            label = " / ".join(keywords[:3]) if keywords else f"Topic {cluster_id}"
            topics.append(Topic(
                topic_id=cluster_id,
                label=label,
                keywords=keywords,
                video_ids=video_ids,
                video_count=len(video_ids),
            ))
        return ClusteringResult(
            topics=tuple(topics), method="sklearn", total_videos=len(videos),
        )
    except Exception:
        return None


def _keyword_cluster(videos: list, n_topics: int) -> ClusteringResult:
    """Pure-stdlib fallback: group videos by shared top keywords."""
    # Build keyword sets per video
    video_keywords: list[Counter] = []
    for v in videos:
        tokens = _tokenize(_extract_video_text(v))
        video_keywords.append(Counter(tokens))

    # Find top keywords across all videos
    all_keywords = Counter()
    for kc in video_keywords:
        all_keywords.update(kc.keys())

    # Pick top N keywords as topic seeds
    top_seeds = [kw for kw, _ in all_keywords.most_common(n_topics * 3)]

    # Assign each video to the seed keyword it contains most
    assignments: dict[int, list[str]] = defaultdict(list)
    for i, v in enumerate(videos):
        best_seed = -1
        best_score = 0
        for seed_idx, seed in enumerate(top_seeds):
            score = video_keywords[i].get(seed, 0)
            if score > best_score:
                best_score = score
                best_seed = seed_idx
        if best_seed < 0:
            # No keyword match — put in a catch-all topic
            assignments[999].append(v.id)
        else:
            assignments[best_seed].append(v.id)

    # Build Topic objects, merging the catch-all
    topics: list[Topic] = []
    for seed_idx, video_ids in sorted(assignments.items()):
        if not video_ids:
            continue
        if seed_idx == 999:
            label = "Other"
            keywords = ()
        else:
            keywords = (top_seeds[seed_idx],)
            label = top_seeds[seed_idx].capitalize()
        topics.append(Topic(
            topic_id=len(topics),
            label=label,
            keywords=keywords,
            video_ids=tuple(video_ids),
            video_count=len(video_ids),
        ))

    return ClusteringResult(
        topics=tuple(topics), method="keyword", total_videos=len(videos),
        confidence="low",
    )


def _build_result_from_assignments(
    videos: list,
    topic_ids: list[int],
    model: object,
    method: str,
) -> ClusteringResult:
    """Build ClusteringResult from BERTopic assignments."""
    topics: list[Topic] = []
    for tid in sorted(set(topic_ids)):
        if tid < 0:
            continue  # -1 is BERTopic's outlier/noise topic
        video_ids = tuple(videos[i].id for i in range(len(videos)) if topic_ids[i] == tid)
        if not video_ids:
            continue
        # Try to get topic label and keywords from BERTopic
        try:
            info = model.get_topic_info()
            row = info[info["Topic"] == tid]
            label = row["Name"].iloc[0] if not row.empty else f"Topic {tid}"
            keywords_tuple = tuple(str(k) for k, _ in (model.get_topic(tid) or [])[:5])
        except Exception:
            label = f"Topic {tid}"
            keywords_tuple = ()
        topics.append(Topic(
            topic_id=tid, label=label, keywords=keywords_tuple,
            video_ids=video_ids, video_count=len(video_ids),
        ))
    return ClusteringResult(
        topics=tuple(topics), method=method, total_videos=len(videos),
    )


def format_clustering_summary(result: ClusteringResult) -> str:
    """Format a ClusteringResult as a human-readable summary string.

    Suitable for pushing to Open Notebook as a synthetic "topic map" source.
    Includes a confidence warning when the keyword fallback was used.
    """
    if not result.topics:
        return "No topics identified (insufficient content)."

    lines = [f"Topic Map ({result.method} method, {result.total_videos} videos):"]
    if result.confidence == "low":
        lines.append(
            "WARNING: Clustering used the keyword fallback (no BERTopic/sklearn installed). "
            "Groupings are based on shared high-frequency words only — treat as approximate."
        )
    lines.append("")
    for topic in result.topics:
        lines.append(f"## {topic.label} ({topic.video_count} videos)")
        if topic.keywords:
            lines.append(f"Keywords: {', '.join(topic.keywords)}")
        lines.append(f"Videos: {', '.join(topic.video_ids[:10])}")
        if topic.video_count > 10:
            lines.append(f"  ... and {topic.video_count - 10} more")
        lines.append("")
    return "\n".join(lines)
