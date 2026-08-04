"""Sentence-integrity layer for yt-scrape.

Restores real punctuation and sentence boundaries to auto-caption text using
a local punctuation model (deepmultilingualpunctuation, free, CPU-friendly),
with pysbd for sentence segmentation.

Tiers:
  0. Idempotence gate -- text that already has healthy punctuation density is
     returned untouched (keeps punctuated fixtures byte-identical).
  1. Punctuation model -- oliverguhr/fullstop-punctuation-multilang (local, free).
  2. ASR -- already available upstream via `transcript --whisper` (faster-whisper).

All heavy imports are lazy. Restored paragraphs are cached on disk keyed by
(model version, paragraph hash) so restoration runs once per video, ever.
"""
from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

MODEL_VERSION = "fullstop-multilang-v1"
CACHE_DIR = Path(os.environ.get("YT_SENTENCES_CACHE",
                                str(Path.home() / ".yt_sentences_cache")))

# One sentence terminator per this many words = "already punctuated".
PUNCTUATED_WORDS_PER_TERMINATOR = 40

_TERMINATORS = re.compile(r"[.!?]")
_model = None


def punctuation_density(text: str) -> float:
    """Sentence terminators per word. 0.0 for empty text."""
    words = len(text.split())
    if not words:
        return 0.0
    return len(_TERMINATORS.findall(text)) / words


def is_punctuated(text: str) -> bool:
    """Tier-0 gate: True when text already has healthy punctuation density."""
    return punctuation_density(text) >= 1.0 / PUNCTUATED_WORDS_PER_TERMINATOR


def split_sentences(text: str) -> list[str]:
    """Split punctuated text into sentences (pysbd, regex fallback)."""
    try:
        import pysbd
        seg = pysbd.Segmenter(language="en", clean=False)
        return [s.strip() for s in seg.segment(text) if s.strip()]
    except Exception:
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _get_model():
    global _model
    if _model is None:
        from deepmultilingualpunctuation import PunctuationModel
        try:
            _model = PunctuationModel()
        except TypeError:
            # transformers >= 5 removed the legacy `grouped_entities` kwarg
            # that deepmultilingualpunctuation still passes at load time.
            # Build the pipeline ourselves (aggregation_strategy="none" is
            # the modern spelling of grouped_entities=False) and graft it
            # onto the model object; restore_punctuation() only uses self.pipe.
            from transformers import pipeline
            grafted = PunctuationModel.__new__(PunctuationModel)
            grafted.pipe = pipeline(
                "token-classification",
                model="oliverguhr/fullstop-punctuation-multilang-large",
                aggregation_strategy="none",
            )
            _model = grafted
    return _model


def _cache_path(text: str) -> Path:
    key = hashlib.sha1((MODEL_VERSION + "\x00" + text).encode("utf-8")).hexdigest()
    return CACHE_DIR / f"{key}.txt"


def _restore_paragraph(paragraph: str) -> str:
    cache = _cache_path(paragraph)
    if cache.exists():
        return cache.read_text(encoding="utf-8")
    restored = _get_model().restore_punctuation(paragraph)
    sentences = split_sentences(restored)
    # Capitalize sentence starts; the model does not always do it.
    fixed = []
    for s in sentences:
        if s and s[0].islower():
            s = s[0].upper() + s[1:]
        fixed.append(s)
    result = " ".join(fixed)
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache.write_text(result, encoding="utf-8")
    except OSError:
        pass  # cache is best-effort
    return result


def restore_text(text: str, mode: str = "model") -> str:
    """Restore punctuation/sentence boundaries paragraph by paragraph.

    mode="off" or empty input returns the input object unchanged.
    Already-punctuated paragraphs pass through untouched (Tier 0), so
    punctuated input is effectively idempotent.
    """
    if mode == "off" or not text:
        return text
    out = []
    for paragraph in text.split("\n\n"):
        p = paragraph.strip()
        if not p:
            continue
        out.append(p if is_punctuated(p) else _restore_paragraph(p))
    return "\n\n".join(out)


def restore_sentences(text: str, mode: str = "model") -> list[str]:
    """Restored text as a flat list of sentences."""
    restored = restore_text(text, mode=mode)
    sentences: list[str] = []
    for paragraph in restored.split("\n\n"):
        sentences.extend(split_sentences(paragraph))
    return sentences
