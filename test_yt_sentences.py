"""Tests for the sentence-integrity layer (yt_sentences.py).

Additive only -- the original 383 tests are untouched. Gate/wiring tests do
not load the punctuation model; TestModelRestoration deliberately DOES load
it, so a silent model-load failure (like the transformers-5 grouped_entities
regression) can never hide behind a green suite again.
"""
import yt_scrape
import yt_sentences


class TestTier0Gate:
    def test_off_mode_returns_input_object_unchanged(self):
        text = "some unpunctuated caption text with no periods at all"
        assert yt_sentences.restore_text(text, mode="off") is text

    def test_punctuated_text_passes_gate_untouched(self):
        text = "This is a sentence. Here is another one! And a third?"
        assert yt_sentences.restore_text(text, mode="model") == text

    def test_density_gate_accepts_punctuated(self):
        assert yt_sentences.is_punctuated("Short. Sentences. Everywhere.")

    def test_density_gate_rejects_runon(self):
        assert not yt_sentences.is_punctuated(" ".join(["word"] * 200))

    def test_empty_text(self):
        assert yt_sentences.restore_text("", mode="model") == ""
        assert yt_sentences.punctuation_density("") == 0.0


class TestSplitSentences:
    def test_basic_split(self):
        parts = yt_sentences.split_sentences("First one. Second one! Third?")
        assert len(parts) == 3

    def test_abbreviation_not_split(self):
        # pysbd should not split on the decimal point
        parts = yt_sentences.split_sentences("RSI hit 70.5 today. Then it fell.")
        assert len(parts) == 2


class TestYtScrapeWiring:
    def test_default_mode_is_off_for_library_callers(self):
        # The 383-test guarantee: importing yt_scrape never enables restoration.
        assert yt_scrape.SENTENCE_MODE == "off"

    def test_maybe_restore_off_is_identity(self):
        text = "unpunctuated words " * 30
        assert yt_scrape._maybe_restore_sentences(text) is text


class TestModelRestoration:
    """Loads the real punctuation model (HF-cached after first run).

    This is the regression test the first build was missing: it asserts the
    model actually transforms unpunctuated input instead of silently falling
    back to pass-through.
    """

    def test_model_actually_restores_punctuation(self, tmp_path, monkeypatch):
        # Isolate the on-disk paragraph cache and force a fresh model load.
        # A warm cache would otherwise let this test pass without ever
        # loading the model -- the exact failure mode it exists to catch.
        monkeypatch.setattr(yt_sentences, "CACHE_DIR", tmp_path / "cache")
        monkeypatch.setattr(yt_sentences, "_model", None)
        text = (
            "the RSI indicator measures momentum it moves between zero and "
            "one hundred traders consider a reading above seventy overbought "
            "and a reading below thirty oversold this is one of the most "
            "widely used oscillators in technical analysis"
        )
        restored = yt_sentences.restore_text(text, mode="model")
        assert restored != text, "model returned input unchanged -- load failure?"
        assert restored.count(".") >= 2, f"expected sentence breaks, got: {restored!r}"


class TestDependencyDeclaration:
    """A fresh install missing these deps degrades silently, not loudly."""

    def test_requirements_declares_sentence_deps(self):
        from pathlib import Path
        req = (Path(__file__).with_name("requirements.txt")
               .read_text(encoding="utf-8").lower())
        for dep in ("deepmultilingualpunctuation", "pysbd",
                    "transformers", "torch"):
            assert dep in req, f"{dep} missing from requirements.txt"


class TestFailureIsolation:
    """A broken punctuation model must never take a scrape down with it."""

    def test_restoration_failure_never_breaks_a_scrape(self, monkeypatch):
        def boom(*args, **kwargs):
            raise RuntimeError("model exploded")

        monkeypatch.setattr(yt_sentences, "restore_text", boom)
        monkeypatch.setattr(yt_scrape, "SENTENCE_MODE", "model")
        text = "unpunctuated caption words " * 20
        assert yt_scrape._maybe_restore_sentences(text) == text

    def test_odd_input_does_not_raise(self):
        for sample in ("", "   ", "\n\n\n", "cafe naive resume"):
            yt_sentences.restore_text(sample, mode="off")
            yt_sentences.split_sentences(sample)
            yt_sentences.punctuation_density(sample)
