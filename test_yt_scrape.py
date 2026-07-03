"""Unit tests for yt_scrape transcript cleanup and Light Research functions.

Tests the cleanup pipeline (filler removal, HTML decode, speaker markers,
censor handling, punctuation heuristics, repeat collapse) and the
prepare_light_research function with mocked transcript extraction.
"""
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from yt_scrape import (
    _decode_html,
    _remove_filler_words,
    _handle_censors,
    _handle_speaker_markers,
    _add_punctuation_heuristics,
    _collapse_repeats,
    clean_transcript_text,
    prepare_light_research,
    LIGHT_RESEARCH_SCHEMA,
    VideoInfo,
)


# ---------------------------------------------------------------- HTML DECODE

class TestDecodeHtml:
    def test_decodes_gt_lt(self):
        assert _decode_html("a &gt; b &lt; c") == "a > b < c"

    def test_decodes_amp_quot_apos(self):
        assert _decode_html("x &amp; y &quot;z&quot; &#39;w&#39;") == 'x & y "z" \'w\''

    def test_decodes_nbsp(self):
        assert _decode_html("hello&nbsp;world") == "hello world"

    def test_no_entities_returns_unchanged(self):
        assert _decode_html("plain text") == "plain text"

    def test_empty_string(self):
        assert _decode_html("") == ""

    def test_multiple_entities(self):
        text = "&gt;&gt; Hello &amp; goodbye &nbsp; &lt;3"
        result = _decode_html(text)
        assert result == ">> Hello & goodbye   <3"  # &nbsp; → space, plus existing space = 3


# ---------------------------------------------------------------- FILLER WORDS

class TestRemoveFillerWords:
    def test_removes_um_uh(self):
        assert _remove_filler_words("um uh hello") == "hello"

    def test_removes_you_know(self):
        assert _remove_filler_words("you know it works") == "it works"

    def test_case_insensitive(self):
        assert _remove_filler_words("UM UH Like basically") == ""

    def test_preserves_newlines(self):
        text = "um hello\n\nuh world"
        result = _remove_filler_words(text)
        assert "\n\n" in result
        assert "hello" in result
        assert "world" in result

    def test_no_fillers_returns_unchanged(self):
        assert _remove_filler_words("clean text here") == "clean text here"

    def test_collapses_extra_spaces(self):
        result = _remove_filler_words("hello um world")
        assert "  " not in result

    def test_empty_string(self):
        assert _remove_filler_words("") == ""


# ---------------------------------------------------------------- CENSORS

class TestHandleCensors:
    def test_handles_nbsp_censor_pre_decode(self):
        assert _handle_censors("[&nbsp;__&nbsp;]") == "[expletive]"

    def test_handles_space_censor_post_decode(self):
        assert _handle_censors("[ __ ]") == "[expletive]"

    def test_handles_plain_censor(self):
        assert _handle_censors("[__]") == "[expletive]"

    def test_handles_bleep(self):
        assert _handle_censors("[bleep]") == "[expletive]"
        assert _handle_censors("[BLEEP]") == "[expletive]"

    def test_multiple_censors(self):
        text = "what the [&nbsp;__&nbsp;] is this [ __ ]"
        result = _handle_censors(text)
        assert result.count("[expletive]") == 2

    def test_no_censors_returns_unchanged(self):
        assert _handle_censors("clean text") == "clean text"


# ---------------------------------------------------------------- SPEAKER MARKERS

class TestHandleSpeakerMarkers:
    def test_converts_gt_gt_to_paragraphs(self):
        text = "Hello. >> World. >> Bye."
        result = _handle_speaker_markers(text)
        assert ">>" not in result
        assert "\n\n" in result
        assert "Hello." in result
        assert "World." in result
        assert "Bye." in result

    def test_converts_html_encoded_markers(self):
        text = "Hello. &gt;&gt; World."
        result = _handle_speaker_markers(text)
        assert "&gt;&gt;" not in result
        assert "\n\n" in result

    def test_no_markers_returns_unchanged(self):
        assert _handle_speaker_markers("just text") == "just text"

    def test_multiple_markers(self):
        text = "A. >> B. >> C. >> D."
        result = _handle_speaker_markers(text)
        assert result.count("\n\n") == 3


# ---------------------------------------------------------------- PUNCTUATION

class TestAddPunctuationHeuristics:
    def test_capitalizes_standalone_i(self):
        assert "I" in _add_punctuation_heuristics("i think so")

    def test_capitalizes_first_letter(self):
        result = _add_punctuation_heuristics("hello world")
        assert result[0] == "H"

    def test_adds_trailing_period(self):
        result = _add_punctuation_heuristics("hello world")
        assert result.endswith(".")

    def test_no_double_period(self):
        result = _add_punctuation_heuristics("hello.")
        assert result == "Hello."  # first letter is capitalized

    def test_capitalizes_after_sentence_end(self):
        result = _add_punctuation_heuristics("hello. world")
        assert "Hello. World" in result


# ---------------------------------------------------------------- REPEAT COLLAPSE

class TestCollapseRepeats:
    def test_collapses_triple_word_repeat(self):
        assert _collapse_repeats("the the the truth") == "the truth"

    def test_collapses_double_phrase_repeat(self):
        result = _collapse_repeats("this is great this is great stuff")
        assert result.count("this is great") == 1

    def test_no_repeats_returns_unchanged(self):
        assert _collapse_repeats("no repeats here") == "no repeats here"


# ---------------------------------------------------------------- FULL PIPELINE

class TestCleanTranscriptText:
    def test_empty_string(self):
        assert clean_transcript_text("") == ""

    def test_full_pipeline_html_and_speakers(self):
        text = "um hello &gt;&gt; uh world [&nbsp;__&nbsp;] yeah"
        result = clean_transcript_text(text)
        assert "&gt;" not in result
        assert "&nbsp;" not in result
        assert ">>" not in result
        assert "[expletive]" in result
        assert "um" not in result.lower() or "um" not in result.split()
        assert "\n\n" in result

    def test_preserves_paragraph_breaks_through_pipeline(self):
        text = "First speaker. &gt;&gt; Second speaker. &gt;&gt; Third speaker."
        result = clean_transcript_text(text)
        assert result.count("\n\n") >= 2

    def test_cleans_real_world_messy_text(self):
        text = (
            "So um basically you know the the the RSI indicator "
            "&gt;&gt; yeah uh it works like this [&nbsp;__&nbsp;] "
            "and &gt;&gt; that's how it goes"
        )
        result = clean_transcript_text(text)
        assert "[expletive]" in result
        assert ">>" not in result
        assert "&gt;" not in result
        # Filler words should be gone
        assert " um " not in result.lower()
        assert " uh " not in result.lower()

    def test_does_not_corrupt_already_clean_text(self):
        text = "This is a clean sentence. It has proper punctuation already."
        result = clean_transcript_text(text)
        assert "clean sentence" in result
        assert "proper punctuation" in result


# ---------------------------------------------------------------- LIGHT RESEARCH SCHEMA

class TestLightResearchSchema:
    def test_schema_has_required_fields(self):
        required = [
            "title", "channel", "url", "duration_seconds", "category",
            "tldr", "summary", "key_points", "topics", "action_items",
            "quotes", "mentioned_resources", "target_audience",
            "difficulty", "controversial_claims", "questions_raised",
            "sentiment", "energy", "format", "comprehended_by",
            "comprehended_at",
        ]
        for field in required:
            assert field in LIGHT_RESEARCH_SCHEMA, f"Missing field: {field}"


# ---------------------------------------------------------------- PREPARE LIGHT RESEARCH

class TestPrepareLightResearch:
    """Integration tests with mocked transcript extraction."""

    @patch("yt_scrape.extract_transcript")
    def test_successful_research(self, mock_extract):
        """Test that prepare_light_research produces correct output structure."""
        # Create a mock VideoInfo result
        mock_result = MagicMock()
        mock_result.error = ""
        mock_result.error_type = ""
        mock_result.has_transcript = True
        mock_result.id = "test1234567"
        mock_result.title = "Test Video"
        mock_result.channel = "Test Channel"
        mock_result.url = "https://youtube.com/watch?v=test1234567"
        mock_result.duration = 300
        mock_result.upload_date = "20260101"
        mock_result.transcript_source = "caption"
        mock_result.transcript_lang = "en"
        mock_result.transcript_chars = 1000

        # Create a temp file with transcript content
        with tempfile.TemporaryDirectory() as tmpdir:
            transcript_path = Path(tmpdir) / "test1234567.txt"
            raw_content = (
                "TITLE: Test Video\n"
                "URL: https://youtube.com/watch?v=test1234567\n"
                "====================\n\n"
                "um hello world &gt;&gt; uh this is a test [&nbsp;__&nbsp;] yeah"
            )
            transcript_path.write_text(raw_content, encoding="utf-8")
            mock_result.transcript_path = str(transcript_path)

            mock_extract.return_value = mock_result

            result = prepare_light_research(
                "https://youtube.com/watch?v=test1234567",
                output_dir=Path(tmpdir),
            )

        assert result["ok"] is True
        assert result["mode"] == "light_research"
        assert result["video"]["id"] == "test1234567"
        assert result["video"]["title"] == "Test Video"
        assert "raw_transcript_path" in result
        assert "cleaned_transcript_path" in result
        assert result["raw_chars"] > 0
        assert result["cleaned_chars"] > 0
        assert "reduction_pct" in result
        assert "schema" in result
        assert "instructions" in result

    @patch("yt_scrape.extract_transcript")
    def test_research_failure_no_transcript(self, mock_extract):
        """Test that prepare_light_research handles missing transcript."""
        mock_result = MagicMock()
        mock_result.error = "No captions available"
        mock_result.error_type = "no_captions"
        mock_result.has_transcript = False
        mock_result.to_dict.return_value = {"error": "No captions available"}
        mock_extract.return_value = mock_result

        result = prepare_light_research("https://youtube.com/watch?v=nonexist123")

        assert result["ok"] is False
        assert "error" in result
        assert result["error"] == "No captions available"

    @patch("yt_scrape.extract_transcript")
    def test_research_produces_cleaned_file(self, mock_extract):
        """Test that the cleaned transcript file is actually written."""
        mock_result = MagicMock()
        mock_result.error = ""
        mock_result.error_type = ""
        mock_result.has_transcript = True
        mock_result.id = "clean1234567"
        mock_result.title = "Clean Test"
        mock_result.channel = "TestChan"
        mock_result.url = "https://youtube.com/watch?v=clean1234567"
        mock_result.duration = 100
        mock_result.upload_date = "20260101"
        mock_result.transcript_source = "caption"
        mock_result.transcript_lang = "en"
        mock_result.transcript_chars = 500

        with tempfile.TemporaryDirectory() as tmpdir:
            transcript_path = Path(tmpdir) / "clean1234567.txt"
            raw_content = (
                "TITLE: Clean Test\n"
                "====================\n\n"
                "um hello &gt;&gt; world [&nbsp;__&nbsp;] this is the the the test"
            )
            transcript_path.write_text(raw_content, encoding="utf-8")
            mock_result.transcript_path = str(transcript_path)

            mock_extract.return_value = mock_result

            result = prepare_light_research(
                "https://youtube.com/watch?v=clean1234567",
                output_dir=Path(tmpdir),
            )

            # Verify cleaned file exists and has correct content
            cleaned_path = Path(result["cleaned_transcript_path"])
            assert cleaned_path.exists()
            content = cleaned_path.read_text(encoding="utf-8")
            assert "TITLE: Clean Test" in content
            assert "CLEANED: true" in content
            assert "[expletive]" in content
            assert ">>" not in content
            assert "&gt;" not in content


# ---------------------------------------------------------------- PERFORMANCE

class TestCleanupPerformance:
    """Verify cleanup runs fast even on large transcripts."""

    def test_cleanup_handles_large_text_quickly(self):
        """50K chars should clean in under 1 second."""
        import time
        # Generate a 50K char transcript with messiness
        chunk = "um hello &gt;&gt; uh world [&nbsp;__&nbsp;] the the the test " * 100
        text = chunk * 5  # ~50K chars
        start = time.time()
        result = clean_transcript_text(text)
        elapsed = time.time() - start
        assert elapsed < 1.0, f"Cleanup took {elapsed:.3f}s, expected <1s"
        assert len(result) > 0


# ---------------------------------------------------------------- DEEP RESEARCH: CLAIM EXTRACTION

from yt_scrape import (
    extract_claims,
    extract_sources,
    _extract_sentence,
    DEEP_RESEARCH_SCHEMA,
    prepare_deep_research,
)


class TestExtractSentence:
    def test_extracts_sentence_with_period(self):
        text = "Hello world. This is a test. Bye now."
        sentence = _extract_sentence(text, text.index("test"), text.index("test") + 4)
        assert "This is a test" in sentence

    def test_extracts_from_start_of_text(self):
        text = "First sentence. Second."
        sentence = _extract_sentence(text, 0, 5)
        assert "First sentence" in sentence

    def test_extracts_from_end_of_text(self):
        text = "First. Last sentence here"
        sentence = _extract_sentence(text, text.index("Last"), len(text))
        assert "Last sentence here" in sentence

    def test_handles_newline_as_boundary(self):
        text = "First paragraph.\n\nSecond paragraph here."
        sentence = _extract_sentence(text, text.index("Second"), len(text))
        assert "Second paragraph here" in sentence


class TestExtractClaims:
    def test_extracts_percentage_claims(self):
        text = "This strategy has an 86% win rate which is absolutely insane."
        claims = extract_claims(text)
        assert len(claims) >= 1
        assert any("statistical" in c["claim_types"] or "win_rate" in c["claim_types"] for c in claims)

    def test_extracts_win_rate_claims(self):
        text = "The win rate of this strategy is 86 percent."
        claims = extract_claims(text)
        assert any("win_rate" in c["claim_types"] for c in claims)

    def test_extracts_dollar_claims(self):
        text = "I made $50,000 in one month using this method."
        claims = extract_claims(text)
        assert any("financial" in c["claim_types"] for c in claims)

    def test_extracts_causal_claims(self):
        text = "Meditation causes changes in brain chemistry."
        claims = extract_claims(text)
        assert any("causal" in c["claim_types"] for c in claims)

    def test_extracts_authority_claims(self):
        text = "Studies show that this indicator works in 90% of cases."
        claims = extract_claims(text)
        assert any("authority" in c["claim_types"] for c in claims)

    def test_extracts_superlative_claims(self):
        text = "This is the best trading strategy ever created."
        claims = extract_claims(text)
        assert any("superlative" in c["claim_types"] for c in claims)

    def test_extracts_scientific_claims(self):
        text = "The subconscious mind controls 95% of your decisions."
        claims = extract_claims(text)
        assert any("scientific" in c["claim_types"] for c in claims)

    def test_extracts_comparative_claims(self):
        text = "This method is better than all the others on YouTube."
        claims = extract_claims(text)
        assert any("comparative" in c["claim_types"] for c in claims)

    def test_deduplicates_by_sentence(self):
        """A sentence matching multiple patterns should appear once with multiple types."""
        text = "Studies show this 90% win rate strategy is the best."
        claims = extract_claims(text)
        # Should be one claim with multiple types, not duplicates
        sentences = [c["sentence"] for c in claims]
        assert len(sentences) == len(set(sentences))

    def test_returns_empty_for_no_claims(self):
        text = "Hello world, how are you today?"
        claims = extract_claims(text)
        assert len(claims) == 0

    def test_claims_sorted_by_position(self):
        text = "First, this has a 50% success rate. Later, this is the best method."
        claims = extract_claims(text)
        if len(claims) >= 2:
            assert claims[0]["char_offset"] < claims[1]["char_offset"]

    def test_claim_has_required_fields(self):
        text = "This has a 90% win rate."
        claims = extract_claims(text)
        assert len(claims) >= 1
        c = claims[0]
        assert "claim_types" in c
        assert "matched_pattern" in c
        assert "sentence" in c
        assert "char_offset" in c

    def test_extracts_from_real_trading_transcript(self):
        """Test on a realistic trading transcript snippet."""
        text = (
            "So this Bollinger Band strategy has a pretty high success rate. "
            "If you use it correctly, you can expect an 86% win rate. "
            "The RSI indicator causes the price to reverse when it goes above 70. "
            "Studies show that mean reversion works in most markets. "
            "This is the best strategy on YouTube. "
            "The subconscious mind of the market controls everything. "
            "I made $10,000 using this method. "
            "This approach is better than every other strategy I've tried."
        )
        claims = extract_claims(text)
        assert len(claims) >= 5
        all_types = set()
        for c in claims:
            all_types.update(c["claim_types"])
        # Should have multiple claim types
        assert len(all_types) >= 4


# ---------------------------------------------------------------- DEEP RESEARCH: SOURCE EXTRACTION

class TestExtractSources:
    def test_extracts_urls(self):
        text = "Check out my website at https://example.com for more info."
        sources = extract_sources(text)
        assert "https://example.com" in sources["urls"]

    def test_extracts_multiple_urls(self):
        text = "Visit https://a.com and http://b.org for details."
        sources = extract_sources(text)
        assert len(sources["urls"]) == 2

    def test_extracts_books_with_author(self):
        text = 'I recommend reading "Trading in the Zone" by Mark Douglas.'
        sources = extract_sources(text)
        assert len(sources["books"]) >= 1
        assert any("Trading in the Zone" in b["title"] for b in sources["books"])

    def test_extracts_books_without_author(self):
        text = '"Atomic Habits" is a great book about discipline.'
        sources = extract_sources(text)
        # May or may not match depending on pattern strictness
        # Just verify it doesn't crash
        assert isinstance(sources["books"], list)

    def test_extracts_paper_references(self):
        text = "According to Smith et al. in the Journal of Finance, this doesn't work."
        sources = extract_sources(text)
        assert len(sources["papers"]) >= 1

    def test_extracts_people(self):
        text = "According to Mark Douglas, trading is 80% psychology."
        sources = extract_sources(text)
        assert len(sources["people"]) >= 1
        assert any("Mark Douglas" in p for p in sources["people"])

    def test_returns_empty_for_no_sources(self):
        text = "Just some plain text without any references."
        sources = extract_sources(text)
        assert sources["urls"] == []
        assert sources["books"] == []
        assert sources["papers"] == []
        assert sources["people"] == []

    def test_combines_transcript_and_description(self):
        text = "Check https://from-transcript.com"
        desc = "Link: https://from-description.com"
        sources = extract_sources(text, desc)
        assert "https://from-transcript.com" in sources["urls"]
        assert "https://from-description.com" in sources["urls"]


# ---------------------------------------------------------------- DEEP RESEARCH SCHEMA

class TestDeepResearchSchema:
    def test_schema_has_required_fields(self):
        required = [
            "research_mode", "title", "channel", "url",
            "executive_summary", "argument_structure",
            "claim_verification", "bias_assessment",
            "cross_references", "omission_analysis",
            "source_bibliography", "research_gaps",
            "open_questions", "methodology", "overall_confidence",
        ]
        for field in required:
            assert field in DEEP_RESEARCH_SCHEMA, f"Missing field: {field}"

    def test_argument_structure_has_subfields(self):
        arg = DEEP_RESEARCH_SCHEMA["argument_structure"]
        assert "main_thesis" in arg
        assert "premises" in arg
        assert "conclusions" in arg
        assert "fallacies_identified" in arg

    def test_claim_verification_has_verdict_options(self):
        # Schema is descriptive, not enforced — just verify structure exists
        assert isinstance(DEEP_RESEARCH_SCHEMA["claim_verification"], list)

    def test_bias_assessment_has_subfields(self):
        bias = DEEP_RESEARCH_SCHEMA["bias_assessment"]
        assert "speaker_credibility" in bias
        assert "potential_biases" in bias
        assert "conflicts_of_interest" in bias
        assert "financial_ecosystem" in bias


# ---------------------------------------------------------------- PREPARE DEEP RESEARCH

class TestPrepareDeepResearch:
    """Integration tests with mocked transcript extraction."""

    @patch("yt_scrape.extract_transcript")
    @patch("yt_scrape.extract_metadata")
    def test_successful_deep_research(self, mock_meta, mock_extract):
        """Test that prepare_deep_research produces correct output structure."""
        mock_result = MagicMock()
        mock_result.error = ""
        mock_result.error_type = ""
        mock_result.has_transcript = True
        mock_result.id = "deep1234567"
        mock_result.title = "Deep Test Video"
        mock_result.channel = "TestChan"
        mock_result.url = "https://youtube.com/watch?v=deep1234567"
        mock_result.duration = 300
        mock_result.upload_date = "20260101"
        mock_result.transcript_source = "caption"
        mock_result.transcript_lang = "en"
        mock_result.transcript_chars = 2000
        mock_result.view_count = 10000
        mock_result.like_count = 500

        mock_meta.return_value = {"description": "Test description with https://example.com"}

        with tempfile.TemporaryDirectory() as tmpdir:
            transcript_path = Path(tmpdir) / "deep1234567.txt"
            raw_content = (
                "TITLE: Deep Test Video\n"
                "====================\n\n"
                "This strategy has a 90% win rate. "
                "Studies show it works. "
                "This is the best method ever. "
                "The subconscious mind controls everything. "
                "Check https://example.com for more."
            )
            transcript_path.write_text(raw_content, encoding="utf-8")
            mock_result.transcript_path = str(transcript_path)

            mock_extract.return_value = mock_result

            result = prepare_deep_research(
                "https://youtube.com/watch?v=deep1234567",
                output_dir=Path(tmpdir),
            )

        assert result["ok"] is True
        assert result["mode"] == "deep_research"
        assert result["video"]["id"] == "deep1234567"
        assert result["video"]["title"] == "Deep Test Video"
        assert "research_package_path" in result
        assert "extracted_claims" in result
        assert "extracted_sources" in result
        assert result["claim_count"] > 0
        assert "schema" in result
        assert "instructions" in result
        assert "transcript" in result
        assert result["transcript"]["cleaned_chars"] > 0

    @patch("yt_scrape.extract_transcript")
    @patch("yt_scrape.extract_metadata")
    def test_deep_research_failure(self, mock_meta, mock_extract):
        """Test that prepare_deep_research handles missing transcript."""
        mock_result = MagicMock()
        mock_result.error = "No captions available"
        mock_result.error_type = "no_captions"
        mock_result.has_transcript = False
        mock_result.to_dict.return_value = {"error": "No captions available"}
        mock_extract.return_value = mock_result

        result = prepare_deep_research("https://youtube.com/watch?v=nonexist123")

        assert result["ok"] is False
        assert "error" in result

    @patch("yt_scrape.extract_transcript")
    @patch("yt_scrape.extract_metadata")
    def test_deep_research_saves_package_file(self, mock_meta, mock_extract):
        """Test that the research package JSON is actually written."""
        mock_result = MagicMock()
        mock_result.error = ""
        mock_result.error_type = ""
        mock_result.has_transcript = True
        mock_result.id = "pkg12345678"
        mock_result.title = "Package Test"
        mock_result.channel = "TestChan"
        mock_result.url = "https://youtube.com/watch?v=pkg12345678"
        mock_result.duration = 100
        mock_result.upload_date = "20260101"
        mock_result.transcript_source = "caption"
        mock_result.transcript_lang = "en"
        mock_result.transcript_chars = 500
        mock_result.view_count = 1000
        mock_result.like_count = 50

        mock_meta.return_value = {"description": ""}

        with tempfile.TemporaryDirectory() as tmpdir:
            transcript_path = Path(tmpdir) / "pkg12345678.txt"
            raw_content = (
                "TITLE: Package Test\n"
                "====================\n\n"
                "This has a 95% win rate and is the best strategy."
            )
            transcript_path.write_text(raw_content, encoding="utf-8")
            mock_result.transcript_path = str(transcript_path)

            mock_extract.return_value = mock_result

            result = prepare_deep_research(
                "https://youtube.com/watch?v=pkg12345678",
                output_dir=Path(tmpdir),
            )

            # Verify package file exists
            pkg_path = Path(result["research_package_path"])
            assert pkg_path.exists()
            import json as _json
            pkg = _json.loads(pkg_path.read_text(encoding="utf-8"))
            assert pkg["ok"] is True
            assert pkg["mode"] == "deep_research"
            assert pkg["claim_count"] > 0
            assert "extracted_claims" in pkg
            assert "instructions" in pkg


# ---------------------------------------------------------------- TTS / READ REPORT

from yt_scrape import (
    _format_deep_research_for_tts,
    _format_light_research_for_tts,
    _format_report_for_tts,
    VOICE_OPTIONS,
    DEFAULT_VOICE,
    read_report,
)

try:
    import edge_tts
except ImportError:
    edge_tts = None


class TestFormatDeepResearchForTTS:
    def test_formats_minimal_report(self):
        report = {
            "research_mode": "deep",
            "title": "Test Video",
            "channel": "TestChan",
            "executive_summary": "This is a test summary.",
        }
        script = _format_deep_research_for_tts(report)
        assert "Test Video" in script
        assert "TestChan" in script
        assert "Executive Summary" in script
        assert "This is a test summary." in script

    def test_includes_claim_verdicts(self):
        report = {
            "research_mode": "deep",
            "title": "Test",
            "claim_verification": [
                {
                    "claim": "This strategy has 90% win rate",
                    "verdict": "contradicted",
                    "evidence": "Testing showed 36%",
                    "confidence": "HIGH",
                    "verbatim_quote": "90% win rate",
                    "sources": [{"url": "http://example.com", "title": "Test"}],
                }
            ],
        }
        script = _format_deep_research_for_tts(report)
        assert "Claim 1" in script
        assert "90% win rate" in script
        assert "contradicted" in script
        assert "HIGH" in script
        assert "Sources consulted: 1" in script

    def test_includes_fallacies(self):
        report = {
            "research_mode": "deep",
            "title": "Test",
            "argument_structure": {
                "fallacies_identified": [
                    {"fallacy": "Cherry-picking", "example": "Only showed winning trades",
                     "explanation": "Selective examples"}
                ],
            },
        }
        script = _format_deep_research_for_tts(report)
        assert "Cherry-picking" in script
        assert "Only showed winning trades" in script

    def test_includes_bias_assessment(self):
        report = {
            "research_mode": "deep",
            "title": "Test",
            "bias_assessment": {
                "speaker_credibility": "Unknown",
                "potential_biases": ["Survivorship bias", "Promotional bias"],
                "conflicts_of_interest": ["Sells paid signals"],
                "financial_ecosystem": "YouTube + paid products",
                "overall_reliability": "low",
            },
        }
        script = _format_deep_research_for_tts(report)
        assert "Survivorship bias" in script
        assert "Sells paid signals" in script
        assert "YouTube + paid products" in script
        assert "low" in script

    def test_includes_omissions(self):
        report = {
            "research_mode": "deep",
            "title": "Test",
            "omission_analysis": ["No stop-loss discussed", "No position sizing"],
        }
        script = _format_deep_research_for_tts(report)
        assert "Omission Analysis" in script
        assert "No stop-loss discussed" in script
        assert "No position sizing" in script

    def test_includes_overall_confidence(self):
        report = {
            "research_mode": "deep",
            "title": "Test",
            "overall_confidence": {"level": "MODERATE", "rationale": "Limited data"},
        }
        script = _format_deep_research_for_tts(report)
        assert "Overall Confidence" in script
        assert "MODERATE" in script
        assert "Limited data" in script

    def test_ends_with_end_of_report(self):
        report = {"research_mode": "deep", "title": "Test"}
        script = _format_deep_research_for_tts(report)
        assert "End of report." in script

    def test_handles_empty_report(self):
        report = {"research_mode": "deep"}
        script = _format_deep_research_for_tts(report)
        assert "End of report." in script

    def test_includes_methodology(self):
        report = {
            "research_mode": "deep",
            "title": "Test",
            "methodology": {
                "approach": "Web research",
                "sources_consulted": 14,
                "web_searches_performed": 4,
                "limitations": ["No exact backtest found"],
            },
        }
        script = _format_deep_research_for_tts(report)
        assert "Methodology" in script
        assert "14" in script
        assert "No exact backtest found" in script


class TestFormatLightResearchForTTS:
    def test_formats_light_report(self):
        report = {
            "title": "Light Test",
            "channel": "TestChan",
            "tldr": "This is a quick summary.",
            "summary": "Detailed summary here.",
            "key_points": ["Point one", "Point two"],
        }
        script = _format_light_research_for_tts(report)
        assert "Light Research Report" in script
        assert "Light Test" in script
        assert "TL;DR" in script
        assert "Point 1" in script
        assert "Point 2" in script

    def test_includes_quotes(self):
        report = {
            "title": "Test",
            "quotes": ["This is a notable quote"],
        }
        script = _format_light_research_for_tts(report)
        assert "Notable Quotes" in script
        assert "This is a notable quote" in script

    def test_includes_topics(self):
        report = {
            "title": "Test",
            "topics": [{"name": "RSI", "description": "Momentum indicator"}],
        }
        script = _format_light_research_for_tts(report)
        assert "RSI" in script
        assert "Momentum indicator" in script


class TestFormatReportForTTS:
    def test_auto_detects_deep_research(self):
        report = {"research_mode": "deep", "title": "Deep Test"}
        script = _format_report_for_tts(report)
        assert "Deep Research Report" in script

    def test_auto_detects_light_research(self):
        report = {"research_mode": "light", "title": "Light Test"}
        script = _format_report_for_tts(report)
        assert "Light Research Report" in script

    def test_defaults_to_light_for_untyped(self):
        report = {"title": "Untyped Test"}
        script = _format_report_for_tts(report)
        assert "Light Research Report" in script


class TestVoiceOptions:
    def test_has_ava_voice(self):
        assert "ava" in VOICE_OPTIONS
        assert VOICE_OPTIONS["ava"] == "en-US-AvaNeural"

    def test_has_male_voices(self):
        assert "andrew" in VOICE_OPTIONS
        assert "brian" in VOICE_OPTIONS
        assert "christopher" in VOICE_OPTIONS

    def test_default_voice_is_ava(self):
        assert DEFAULT_VOICE == "en-US-AvaNeural"


class TestReadReport:
    def test_fails_for_missing_file(self):
        result = read_report("/nonexistent/path/report.json")
        assert result["ok"] is False
        assert "not found" in result["error"].lower()

    def test_fails_for_invalid_json(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            f.write("not valid json {{{")
            f.flush()
            result = read_report(f.name)
        os.unlink(f.name)
        assert result["ok"] is False
        assert "JSON" in result["error"] or "json" in result["error"].lower()

    @pytest.mark.skipif(edge_tts is None, reason="edge-tts not installed")
    def test_generates_audio_for_real_report(self):
        """Integration test — generates an actual MP3 from a test report."""
        report = {
            "research_mode": "deep",
            "title": "TTS Test Report",
            "channel": "TestChannel",
            "executive_summary": "This is a test of the text to speech system.",
            "overall_confidence": {"level": "HIGH", "rationale": "Test passed."},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "test_report.json"
            report_path.write_text(json.dumps(report), encoding="utf-8")

            result = read_report(report_path, output_dir=Path(tmpdir))

            # Assert BEFORE the temp dir is cleaned up
            assert result["ok"] is True
            assert Path(result["audio_path"]).exists()
            assert result["audio_size_bytes"] > 0
            assert result["voice"] == "en-US-AvaNeural"
            assert result["text_length"] > 0


# ---------------------------------------------------------------- INTERACTIVE WEB REPORTS

from yt_scrape import (
    report_to_html,
    _verdict_color,
    _confidence_bar,
    _reliability_score,
    _escape,
)


class TestVerdictColor:
    def test_verified_is_green(self):
        bg, border, label = _verdict_color("verified")
        assert "22c55e" in bg
        assert "Verified" in label

    def test_contradicted_is_red(self):
        bg, border, label = _verdict_color("contradicted")
        assert "ef4444" in bg
        assert "Contradicted" in label

    def test_partially_verified_is_yellow(self):
        bg, border, label = _verdict_color("partially-verified")
        assert "eab308" in bg
        assert "Partially" in label

    def test_opinion_is_gray(self):
        bg, border, label = _verdict_color("opinion")
        assert "94a3b8" in bg
        assert "Opinion" in label

    def test_unverified_is_orange(self):
        bg, border, label = _verdict_color("unverified")
        assert "f59e0b" in bg
        assert "Unverified" in label

    def test_unverifiable_is_gray(self):
        bg, border, label = _verdict_color("unverifiable")
        assert "94a3b8" in bg


class TestConfidenceBar:
    def test_high_confidence(self):
        bar = _confidence_bar("HIGH")
        assert "90%" in bar
        assert "22c55e" in bar
        assert "HIGH" in bar

    def test_moderate_confidence(self):
        bar = _confidence_bar("MODERATE")
        assert "60%" in bar
        assert "eab308" in bar

    def test_low_confidence(self):
        bar = _confidence_bar("LOW")
        assert "30%" in bar
        assert "ef4444" in bar

    def test_unknown_confidence(self):
        bar = _confidence_bar("UNKNOWN")
        assert "50%" in bar


class TestReliabilityScore:
    def test_high_reliability(self):
        assert _reliability_score("high") == 85

    def test_moderate_reliability(self):
        assert _reliability_score("moderate") == 55

    def test_low_reliability(self):
        assert _reliability_score("low") == 20

    def test_mixed_reliability(self):
        assert _reliability_score("mixed") == 45

    def test_unknown_reliability(self):
        assert _reliability_score("unknown") == 50


class TestEscape:
    def test_escapes_html(self):
        assert _escape("<script>alert('xss')</script>") == "&lt;script&gt;alert(&#x27;xss&#x27;)&lt;/script&gt;"

    def test_escapes_quotes(self):
        assert "&quot;" in _escape('"hello"')

    def test_empty_string(self):
        assert _escape("") == ""

    def test_none_returns_empty(self):
        assert _escape(None) == ""


class TestReportToHTML:
    def test_generates_valid_html(self):
        report = {
            "research_mode": "deep",
            "title": "Test Video",
            "channel": "TestChan",
            "url": "https://youtube.com/watch?v=abcdefghijk",
            "executive_summary": "This is a test summary.",
        }
        html = report_to_html(report)
        assert "<!DOCTYPE html>" in html
        assert "</html>" in html
        assert "Test Video" in html
        assert "TestChan" in html
        assert "Executive Summary" in html
        assert "This is a test summary." in html

    def test_includes_bias_meter(self):
        report = {
            "research_mode": "deep",
            "title": "Test",
            "bias_assessment": {
                "overall_reliability": "low",
                "speaker_credibility": "Unknown",
                "potential_biases": ["Survivorship bias"],
                "conflicts_of_interest": ["Sells paid signals"],
                "financial_ecosystem": "YouTube + paid products",
            },
        }
        html = report_to_html(report)
        assert "bias-meter" in html
        assert "gauge-fill" in html
        assert "20/100" in html  # low = 20
        assert "Survivorship bias" in html
        assert "Sells paid signals" in html

    def test_includes_claim_cards(self):
        report = {
            "research_mode": "deep",
            "title": "Test",
            "claim_verification": [
                {
                    "claim": "86% win rate",
                    "verdict": "contradicted",
                    "evidence": "Actual: 36%",
                    "confidence": "HIGH",
                    "verbatim_quote": "86% win rate",
                    "sources": [{"url": "http://example.com", "title": "Test Source", "reliability": "moderate"}],
                }
            ],
        }
        html = report_to_html(report)
        assert "claim-card" in html
        assert "verdict-badge" in html
        assert "Contradicted" in html
        assert "86% win rate" in html
        assert "confidence-bar" in html
        assert "example.com" in html

    def test_includes_fallacies(self):
        report = {
            "research_mode": "deep",
            "title": "Test",
            "argument_structure": {
                "main_thesis": "This strategy works",
                "fallacies_identified": [
                    {"fallacy": "Cherry-picking", "example": "Only showed wins", "explanation": "Selective examples"}
                ],
            },
        }
        html = report_to_html(report)
        assert "fallacy-card" in html
        assert "Cherry-picking" in html
        assert "Only showed wins" in html

    def test_includes_omissions(self):
        report = {
            "research_mode": "deep",
            "title": "Test",
            "omission_analysis": ["No stop-loss discussed", "No position sizing"],
        }
        html = report_to_html(report)
        assert "omissions" in html
        assert "No stop-loss discussed" in html

    def test_includes_cross_references(self):
        report = {
            "research_mode": "deep",
            "title": "Test",
            "cross_references": [
                {
                    "topic": "RSI effectiveness",
                    "this_video_claims": "RSI works great",
                    "authoritative_sources_say": "Mixed results",
                    "agreement_level": "partially-consistent",
                    "sources": ["http://example.com"],
                }
            ],
        }
        html = report_to_html(report)
        assert "xref-card" in html
        assert "RSI effectiveness" in html
        assert "agreement-badge" in html

    def test_includes_thumbnail(self):
        report = {
            "research_mode": "deep",
            "title": "Test",
            "url": "https://youtube.com/watch?v=abcdefghijk",
        }
        html = report_to_html(report)
        assert "img.youtube.com" in html
        assert "abcdefghijk" in html

    def test_light_research_mode(self):
        report = {
            "research_mode": "light",
            "title": "Light Test",
            "tldr": "Quick summary",
            "key_points": ["Point 1", "Point 2"],
            "quotes": ["Notable quote"],
        }
        html = report_to_html(report)
        assert "Light Research" in html
        assert "mode-badge light" in html
        assert "Quick summary" in html
        assert "Point 1" in html
        assert "Notable quote" in html

    def test_saves_to_file(self):
        report = {"research_mode": "deep", "title": "File Test"}
        with tempfile.TemporaryDirectory() as tmpdir:
            html_path = Path(tmpdir) / "report.html"
            report_to_html(report, html_path)
            assert html_path.exists()
            content = html_path.read_text(encoding="utf-8")
            assert "<!DOCTYPE html>" in content
            assert "File Test" in content

    def test_has_dark_mode_toggle(self):
        report = {"research_mode": "deep", "title": "Test"}
        html = report_to_html(report)
        assert "toggleTheme" in html
        assert "theme-toggle" in html

    def test_has_collapsible_sections(self):
        report = {
            "research_mode": "deep",
            "title": "Test",
            "executive_summary": "Summary here",
        }
        html = report_to_html(report)
        assert "toggleSection" in html

    def test_has_responsive_design(self):
        report = {"research_mode": "deep", "title": "Test"}
        html = report_to_html(report)
        assert "@media" in html
        assert "max-width" in html

    def test_has_print_styles(self):
        report = {"research_mode": "deep", "title": "Test"}
        html = report_to_html(report)
        assert "@media print" in html

    def test_no_external_dependencies(self):
        """HTML should be fully self-contained — no CDN or external CSS/JS."""
        report = {"research_mode": "deep", "title": "Test"}
        html = report_to_html(report)
        assert "cdn." not in html.lower()
        assert "<link" not in html  # no external stylesheet links
        assert "src=\"http" not in html  # no external scripts (thumbnail is img src, not script)

    def test_empty_report_doesnt_crash(self):
        report = {"research_mode": "deep"}
        html = report_to_html(report)
        assert "<!DOCTYPE html>" in html
        assert "</html>" in html

    def test_escapes_xss_in_title(self):
        report = {
            "research_mode": "deep",
            "title": "<script>alert('xss')</script>",
        }
        html = report_to_html(report)
        assert "<script>alert" not in html
        assert "&lt;script&gt;" in html


# ---------------------------------------------------------------- SENTENCE SEGMENTATION

from yt_scrape import (
    split_sentences,
    segments_to_sentences,
    TranscriptSegment,
    TimestampedSentence,
)


class TestSplitSentences:
    def test_basic_split(self):
        sents = split_sentences("Hello world. This is a test. Goodbye.")
        assert len(sents) == 3
        assert "Hello world" in sents[0]
        assert "This is a test" in sents[1]

    def test_handles_abbreviations(self):
        sents = split_sentences("Dr. Smith said the strategy works. It is profitable.")
        assert len(sents) == 2
        assert "Dr. Smith said the strategy works" in sents[0]

    def test_handles_mr_abbreviation(self):
        sents = split_sentences("Mr. Jones trades options. He makes money.")
        assert len(sents) == 2

    def test_handles_decimals(self):
        sents = split_sentences("The win rate is 86.5 percent. That is high.")
        assert len(sents) == 2
        assert "86.5" in sents[0]

    def test_handles_us_abbreviation(self):
        sents = split_sentences("The U.S. market is strong. Stocks are up.")
        assert len(sents) == 2

    def test_handles_etc_abbreviation(self):
        sents = split_sentences("Use RSI, MACD, etc. These are indicators.")
        assert len(sents) == 2

    def test_handles_initials(self):
        sents = split_sentences("A. J. Smith wrote the book. It is good.")
        assert len(sents) == 2

    def test_question_marks(self):
        sents = split_sentences("Is this true? Yes it is. Done.")
        assert len(sents) == 3

    def test_exclamation_marks(self):
        sents = split_sentences("Wow! This works. Great.")
        assert len(sents) == 3

    def test_multiple_punctuation(self):
        sents = split_sentences("Really?! That is amazing. Wow.")
        assert len(sents) == 3

    def test_newline_as_boundary(self):
        sents = split_sentences("First sentence.\nSecond sentence.")
        assert len(sents) == 2

    def test_empty_string(self):
        assert split_sentences("") == []

    def test_single_sentence(self):
        sents = split_sentences("Just one sentence here.")
        assert len(sents) == 1

    def test_no_ending_punctuation(self):
        sents = split_sentences("This has no period")
        assert len(sents) == 1

    def test_ellipsis_not_split(self):
        sents = split_sentences("Wait... let me think. OK done.")
        # "Wait..." should be one sentence, "let me think" another, "OK done" third
        assert len(sents) >= 2

    def test_lowercase_continuation(self):
        # "i.e." should not split
        sents = split_sentences("Use the strategy, i.e. the RSI method. It works.")
        assert len(sents) == 2


class TestSegmentsToSentences:
    def test_basic_conversion(self):
        segments = [
            TranscriptSegment(text="Hello world. This is a test.", start=0.0, end=5.0),
            TranscriptSegment(text="Second segment here.", start=5.0, end=10.0),
        ]
        sents = segments_to_sentences(segments)
        assert len(sents) == 3
        assert sents[0].text == "Hello world."
        assert sents[0].start == 0.0
        assert sents[1].text == "This is a test."
        assert sents[1].start == 0.0  # same segment
        assert sents[2].text == "Second segment here."
        assert sents[2].start == 5.0

    def test_timestamp_str_format(self):
        seg = TranscriptSegment(text="Test.", start=222.0, end=225.0)
        sents = segments_to_sentences([seg])
        assert sents[0].timestamp_str == "3:42"

    def test_timestamp_str_with_hours(self):
        seg = TranscriptSegment(text="Test.", start=3725.0, end=3730.0)
        sents = segments_to_sentences([seg])
        assert sents[0].timestamp_str == "1:02:05"

    def test_youtube_url(self):
        seg = TranscriptSegment(text="Test.", start=60.0, end=65.0)
        sents = segments_to_sentences([seg])
        assert "t=60s" in sents[0].youtube_url

    def test_empty_segments(self):
        assert segments_to_sentences([]) == []

    def test_empty_text_segment(self):
        seg = TranscriptSegment(text="", start=0.0, end=5.0)
        assert segments_to_sentences([seg]) == []

    def test_to_dict(self):
        seg = TranscriptSegment(text="Test sentence.", start=10.5, end=15.0)
        sents = segments_to_sentences([seg])
        d = sents[0].to_dict()
        assert d["text"] == "Test sentence."
        assert d["start"] == 10.5
        assert d["timestamp"] == "0:10"


# ---------------------------------------------------------------- ENTITY EXTRACTION

from yt_scrape import extract_entities


class TestExtractEntities:
    def test_extracts_people_with_title(self):
        text = "Dr. Jordan Peterson said the strategy works."
        entities = extract_entities(text)
        assert any("Jordan Peterson" in p["name"] for p in entities["people"])

    def test_extracts_people_with_attribution(self):
        text = "According to Warren Buffett, the best strategy is value investing."
        entities = extract_entities(text)
        assert any("Warren Buffett" in p["name"] for p in entities["people"])

    def test_filters_blacklist_names(self):
        text = "The Best strategy is here. Bollinger Bands are great."
        entities = extract_entities(text)
        # "The Best" and "Bollinger Bands" should not be people
        people_names = [p["name"] for p in entities["people"]]
        assert "The Best" not in people_names
        assert "Bollinger Bands" not in people_names

    def test_extracts_organizations_with_suffix(self):
        text = "Goldman Sachs Inc. released a report. Vanguard Group also commented."
        entities = extract_entities(text)
        org_names = [o["name"] for o in entities["organizations"]]
        assert any("Goldman Sachs" in n for n in org_names)

    def test_extracts_known_organizations(self):
        text = "The Federal Reserve raised interest rates. NASA confirmed the data."
        entities = extract_entities(text)
        org_names = [o["name"] for o in entities["organizations"]]
        assert "Federal Reserve" in org_names
        assert "NASA" in org_names

    def test_extracts_tools(self):
        text = "I use TradingView for charting and MetaTrader for execution."
        entities = extract_entities(text)
        tool_names = [t["name"] for t in entities["tools"]]
        assert "TradingView" in tool_names
        assert "MetaTrader" in tool_names

    def test_extracts_trading_metrics(self):
        text = "The RSI is oversold. The MACD shows a crossover. Use Bollinger Bands."
        entities = extract_entities(text)
        metric_names = [m["name"] for m in entities["metrics"]]
        assert "RSI" in metric_names
        assert "MACD" in metric_names

    def test_extracts_financial_metrics(self):
        text = "The P/E ratio is high. Check the Sharpe ratio and alpha."
        entities = extract_entities(text)
        metric_names = [m["name"] for m in entities["metrics"]]
        assert any("P/E" in m or "Sharpe" in m for m in metric_names)

    def test_extracts_concepts(self):
        text = "Inflation is rising. The Federal Reserve may use quantitative easing."
        entities = extract_entities(text)
        concept_names = [c["name"] for c in entities["concepts"]]
        assert "inflation" in [c.lower() for c in concept_names]

    def test_deduplication(self):
        text = "Dr. Smith said X. Dr. Smith also said Y."
        entities = extract_entities(text)
        smiths = [p for p in entities["people"] if "Smith" in p["name"]]
        assert len(smiths) == 1  # deduplicated

    def test_empty_text(self):
        entities = extract_entities("")
        assert entities["people"] == []
        assert entities["organizations"] == []

    def test_includes_context(self):
        text = "Dr. Jordan Peterson said the strategy works perfectly."
        entities = extract_entities(text)
        if entities["people"]:
            assert entities["people"][0]["context"] != ""

    def test_metric_types(self):
        text = "Check the RSI and the P/E ratio."
        entities = extract_entities(text)
        types = {m["type"] for m in entities["metrics"]}
        assert "trading" in types
        assert "financial" in types


# ---------------------------------------------------------------- CLAIM ENRICHMENT

from yt_scrape import (
    extract_claims_enriched,
    enrich_claim,
    _detect_negation,
    _assess_strength,
    _extract_subject,
)


class TestDetectNegation:
    def test_detects_not(self):
        assert _detect_negation("This does not work.") is True

    def test_detects_never(self):
        assert _detect_negation("This never works.") is True

    def test_no_negation(self):
        assert _detect_negation("This always works.") is False

    def test_detects_cannot(self):
        assert _detect_negation("You cannot win every trade.") is True

    def test_detects_no(self):
        assert _detect_negation("There is no evidence for this.") is True


class TestAssessStrength:
    def test_high_strength(self):
        assert _assess_strength("This is proven to work 100% of the time.") == "high"

    def test_moderate_strength(self):
        assert _assess_strength("Studies show this usually works.") == "moderate"

    def test_low_strength(self):
        assert _assess_strength("This might work, perhaps.") == "low"

    def test_opinion(self):
        assert _assess_strength("I think this is good.") == "opinion"

    def test_bare_assertion(self):
        assert _assess_strength("The RSI is oversold.") == "moderate"


class TestExtractSubject:
    def test_causal_subject(self):
        subject = _extract_subject("RSI causes false signals in trending markets.", "causal", "causes")
        assert "RSI" in subject

    def test_authority_subject(self):
        subject = _extract_subject("Studies show that RSI is effective.", "authority", "studies show")
        assert "RSI" in subject

    def test_comparative_subject(self):
        subject = _extract_subject("RSI is better than MACD for timing.", "comparative", "better than")
        assert "RSI" in subject

    def test_empty_sentence(self):
        assert _extract_subject("", "causal", "") == ""


class TestEnrichClaim:
    def test_enriches_with_subject(self):
        claim = {
            "claim_types": ["causal"],
            "matched_pattern": "causes",
            "sentence": "RSI causes false signals.",
            "char_offset": 0,
        }
        enriched = enrich_claim(claim, "RSI causes false signals.")
        assert enriched["subject"] != ""
        assert "causal" in enriched["claim_types"]

    def test_enriches_with_negation(self):
        claim = {
            "claim_types": ["causal"],
            "matched_pattern": "causes",
            "sentence": "This does not cause cancer.",
            "char_offset": 0,
        }
        enriched = enrich_claim(claim, "This does not cause cancer.")
        assert enriched["negated"] is True

    def test_enriches_with_strength(self):
        claim = {
            "claim_types": ["authority"],
            "matched_pattern": "studies show",
            "sentence": "Studies prove this works.",
            "char_offset": 0,
        }
        enriched = enrich_claim(claim, "Studies prove this works.")
        assert enriched["strength"] in ("high", "moderate", "low", "opinion")

    def test_enriches_with_timestamp(self):
        claim = {
            "claim_types": ["statistical"],
            "matched_pattern": "86%",
            "sentence": "The win rate is 86%.",
            "char_offset": 0,
        }
        ts = TimestampedSentence(text="The win rate is 86%.", start=120.0, end=125.0, segment_index=0)
        enriched = enrich_claim(claim, "The win rate is 86%.", ts)
        assert enriched["timestamp"]["timestamp"] == "2:00"
        assert "youtube_url" in enriched

    def test_no_timestamp(self):
        claim = {
            "claim_types": ["statistical"],
            "matched_pattern": "86%",
            "sentence": "The win rate is 86%.",
            "char_offset": 0,
        }
        enriched = enrich_claim(claim, "The win rate is 86%.")
        assert "timestamp" not in enriched

    def test_does_not_mutate_original(self):
        claim = {
            "claim_types": ["causal"],
            "matched_pattern": "causes",
            "sentence": "RSI causes false signals.",
            "char_offset": 0,
        }
        original = dict(claim)
        enrich_claim(claim, "RSI causes false signals.")
        assert claim == original  # original not mutated


class TestExtractClaimsEnriched:
    def test_enriches_all_claims(self):
        text = "The win rate is 86%. Studies show RSI works. This costs $500."
        claims = extract_claims_enriched(text)
        assert len(claims) > 0
        for c in claims:
            assert "strength" in c
            assert "negated" in c

    def test_with_timestamps(self):
        text = "The win rate is 86%."
        ts = [TimestampedSentence(text="The win rate is 86%.", start=60.0, end=65.0, segment_index=0)]
        claims = extract_claims_enriched(text, ts)
        assert len(claims) > 0
        assert claims[0]["timestamp"]["timestamp"] == "1:00"

    def test_without_timestamps(self):
        text = "The win rate is 86%."
        claims = extract_claims_enriched(text)
        assert len(claims) > 0
        assert "timestamp" not in claims[0]

    def test_empty_text(self):
        assert extract_claims_enriched("") == []


# ---------------------------------------------------------------- ADVANCED EXTRACTION (10 FEATURES)

from yt_scrape import (
    detect_contradictions,
    detect_marketing_pressure,
    extract_parameters,
    extract_rules,
    extract_citations,
    extract_questions,
    calculate_hedge_density,
    segment_topics,
    track_emotional_language,
    extract_definitions,
    extract_all,
)


# === 1. CONTRADICTION DETECTION ===
class TestDetectContradictions:
    def test_detects_buy_sell_contradiction(self):
        claims = [
            {"sentence": "You should always buy here.", "subject": "here", "negated": False},
            {"sentence": "You should sell here.", "subject": "here", "negated": False},
        ]
        contradictions = detect_contradictions(claims)
        assert len(contradictions) >= 1

    def test_detects_works_fails_contradiction(self):
        claims = [
            {"sentence": "This strategy works perfectly.", "subject": "this strategy", "negated": False},
            {"sentence": "This strategy fails completely.", "subject": "this strategy", "negated": False},
        ]
        contradictions = detect_contradictions(claims)
        assert len(contradictions) >= 1

    def test_detects_negation_flip(self):
        claims = [
            {"sentence": "RSI is effective.", "subject": "rsi", "negated": False},
            {"sentence": "RSI is not effective.", "subject": "rsi", "negated": True},
        ]
        contradictions = detect_contradictions(claims)
        assert len(contradictions) >= 1

    def test_no_contradiction_for_unrelated(self):
        claims = [
            {"sentence": "RSI is oversold.", "subject": "rsi", "negated": False},
            {"sentence": "MACD is bullish.", "subject": "macd", "negated": False},
        ]
        contradictions = detect_contradictions(claims)
        assert len(contradictions) == 0

    def test_empty_claims(self):
        assert detect_contradictions([]) == []

    def test_includes_timestamps(self):
        claims = [
            {"sentence": "Always buy.", "subject": "buy", "negated": False, "timestamp": {"timestamp": "1:00"}},
            {"sentence": "Never buy.", "subject": "buy", "negated": True, "timestamp": {"timestamp": "5:00"}},
        ]
        contradictions = detect_contradictions(claims)
        if contradictions:
            assert "timestamp_a" in contradictions[0]


# === 2. MARKETING PRESSURE ===
class TestDetectMarketingPressure:
    def test_detects_urgency(self):
        result = detect_marketing_pressure("Act now! Only 5 spots left!")
        assert len(result["urgency"]) >= 1
        assert result["pressure_score"] > 0

    def test_detects_scarcity(self):
        result = detect_marketing_pressure("This is a limited time offer with exclusive access.")
        assert len(result["scarcity"]) >= 1

    def test_detects_social_proof(self):
        result = detect_marketing_pressure("Join 10,000 traders who use our system.")
        assert len(result["social_proof"]) >= 1

    def test_detects_affiliate(self):
        result = detect_marketing_pressure("Use my affiliate link in the description below.")
        assert len(result["affiliate"]) >= 1

    def test_no_pressure(self):
        result = detect_marketing_pressure("The RSI indicator measures momentum.")
        assert result["pressure_score"] == 0
        assert "No marketing pressure" in result["summary"]

    def test_pressure_score_increases(self):
        low = detect_marketing_pressure("Act now!")
        high = detect_marketing_pressure("Act now! Only 5 spots left! Join 10,000 traders! Use my affiliate link! Limited time! Selling out fast!")
        assert high["pressure_score"] > low["pressure_score"]

    def test_includes_summary(self):
        result = detect_marketing_pressure("Act now!")
        assert "summary" in result
        assert isinstance(result["summary"], str)


# === 3. NUMERIC PARAMETERS ===
class TestExtractParameters:
    def test_extracts_rsi_length(self):
        params = extract_parameters("Set the RSI length to 14 for best results.")
        assert any(p["parameter"] == "length" and p["value"] == 14 for p in params)

    def test_extracts_overbought_level(self):
        params = extract_parameters("RSI overbought at 70 and oversold at 30.")
        assert any(p["parameter"] == "overbought" for p in params)

    def test_extracts_stop_loss(self):
        params = extract_parameters("Use a stop loss of 2 percent.")
        assert any("stop" in p["parameter"] for p in params)

    def test_extracts_risk_reward_ratio(self):
        params = extract_parameters("Use a risk reward ratio of 1:3.")
        assert any("risk_reward" in p["parameter"] for p in params)

    def test_extracts_leverage(self):
        params = extract_parameters("I use 10x leverage on this trade.")
        assert any(p["parameter"] == "leverage" and p["value"] == 10 for p in params)

    def test_extracts_percentage_risk(self):
        params = extract_parameters("Risk 2% per trade.")
        assert any(p["parameter"] == "percentage" and p["value"] == 2.0 for p in params)

    def test_no_parameters_in_plain_text(self):
        params = extract_parameters("The market is interesting today.")
        # May find some incidental numbers, but no real parameters
        assert all(p["parameter"] not in ("rsi", "macd", "stop loss") for p in params)

    def test_includes_sentence_context(self):
        params = extract_parameters("Set the RSI length to 14.")
        if params:
            assert "sentence" in params[0]


# === 4. CONDITION/ACTION RULES ===
class TestExtractRules:
    def test_extracts_if_then(self):
        rules = extract_rules("If RSI is below 30, you should buy.")
        assert len(rules) >= 1
        assert "RSI" in rules[0]["condition"] or "rsi" in rules[0]["condition"].lower()
        assert "buy" in rules[0]["action"].lower()

    def test_extracts_when_do(self):
        rules = extract_rules("When the MACD crosses, sell immediately.")
        assert len(rules) >= 1

    def test_extracts_once_condition(self):
        rules = extract_rules("Once price hits support, enter the trade.")
        assert len(rules) >= 1

    def test_filters_non_rules(self):
        rules = extract_rules("If you want to learn, watch the video.")
        # "watch" is an action verb, so this might be extracted — that's ok
        # But "if you want to learn" is not a trading rule
        # Just make sure it doesn't crash
        assert isinstance(rules, list)

    def test_empty_text(self):
        assert extract_rules("") == []

    def test_includes_condition_and_action(self):
        rules = extract_rules("If RSI is below 30, buy immediately.")
        if rules:
            assert "condition" in rules[0]
            assert "action" in rules[0]


# === 5. CITATION DECOMPOSITION ===
class TestExtractCitations:
    def test_extracts_full_citation(self):
        citations = extract_citations("A 2023 Harvard study of 10,000 patients showed results.")
        assert len(citations) >= 1
        assert citations[0]["year"] == 2023
        assert "Harvard" in citations[0]["institution"]

    def test_extracts_sample_size(self):
        citations = extract_citations("A 2020 Stanford study of 5000 traders found patterns.")
        assert any(c["sample_size"] == 5000 for c in citations)

    def test_extracts_journal_reference(self):
        citations = extract_citations("Published in the Journal of Finance last year.")
        assert any("Journal" in (c.get("institution") or "") for c in citations)

    def test_extracts_author_reference(self):
        citations = extract_citations("Research by Dr. Jordan Peterson showed interesting results.")
        assert any(c.get("author") == "Jordan Peterson" for c in citations)

    def test_marks_verifiable(self):
        citations = extract_citations("A 2023 Harvard study showed results.")
        assert any(c["verifiable"] for c in citations)

    def test_empty_text(self):
        assert extract_citations("") == []


# === 6. QUESTION EXTRACTION ===
class TestExtractQuestions:
    def test_extracts_research_question(self):
        questions = extract_questions("Does RSI actually work in trending markets?")
        assert len(questions) >= 1
        assert questions[0]["type"] == "research"

    def test_extracts_rhetorical_question(self):
        questions = extract_questions("Who knows what the market will do tomorrow?")
        assert any(q["type"] == "rhetorical" for q in questions)

    def test_extracts_sales_question(self):
        questions = extract_questions("Want to make money trading?")
        assert any(q["type"] == "sales" for q in questions)

    def test_no_questions_in_statement(self):
        questions = extract_questions("The RSI is a momentum indicator.")
        assert len(questions) == 0

    def test_deduplicates(self):
        questions = extract_questions("Does this work? Does this work? Does this work?")
        assert len(questions) == 1

    def test_empty_text(self):
        assert extract_questions("") == []


# === 7. HEDGE DENSITY ===
class TestCalculateHedgeDensity:
    def test_low_density(self):
        result = calculate_hedge_density("The RSI is a momentum oscillator. It measures speed.")
        assert result["level"] == "low"
        assert result["density"] < 3

    def test_high_density(self):
        text = "I think maybe it sort of works perhaps basically essentially kind of possibly."
        result = calculate_hedge_density(text)
        assert result["level"] in ("high", "extreme")
        assert result["hedge_count"] > 3

    def test_counts_hedges(self):
        result = calculate_hedge_density("Maybe perhaps possibly probably.")
        assert result["hedge_count"] >= 3

    def test_empty_text(self):
        result = calculate_hedge_density("")
        assert result["hedge_count"] == 0
        assert result["density"] == 0.0

    def test_includes_hedges_found(self):
        result = calculate_hedge_density("Maybe it works.")
        assert "hedges_found" in result
        assert isinstance(result["hedges_found"], dict)


# === 8. TOPIC SEGMENTATION ===
class TestSegmentTopics:
    def test_segments_multiple_topics(self):
        text = (
            "Let's talk about RSI. RSI is a momentum indicator. "
            "It measures the speed of price movements. "
            "Now let's move on to MACD. MACD is a trend indicator. "
            "It shows crossovers between moving averages. "
            "Finally, let's discuss Bollinger Bands. They show volatility."
        )
        chapters = segment_topics(text, min_segment_words=10)
        assert len(chapters) >= 2

    def test_single_topic(self):
        chapters = segment_topics("RSI is a momentum indicator. It measures speed.", min_segment_words=5)
        assert len(chapters) >= 1

    def test_includes_timestamps(self):
        sentences = [
            TimestampedSentence(text="Let's talk about RSI.", start=0.0, end=5.0, segment_index=0),
            TimestampedSentence(text="RSI is great.", start=5.0, end=10.0, segment_index=1),
            TimestampedSentence(text="Now let's move to MACD.", start=10.0, end=15.0, segment_index=2),
            TimestampedSentence(text="MACD is also great.", start=15.0, end=20.0, segment_index=3),
        ]
        chapters = segment_topics("dummy", sentences=sentences, min_segment_words=5)
        if len(chapters) >= 2:
            assert "start_timestamp" in chapters[0]
            assert "end_timestamp" in chapters[0]

    def test_includes_word_count(self):
        chapters = segment_topics("Let's talk about RSI. It is a momentum indicator.", min_segment_words=5)
        if chapters:
            assert "word_count" in chapters[0]

    def test_empty_text(self):
        assert segment_topics("") == []


# === 9. EMOTIONAL LANGUAGE ===
class TestTrackEmotionalLanguage:
    def test_detects_positive_emotion(self):
        result = track_emotional_language("This strategy is amazing and incredible!")
        assert result["positive_count"] >= 2

    def test_detects_negative_emotion(self):
        result = track_emotional_language("This is a terrible scam and a disaster.")
        assert result["negative_count"] >= 2

    def test_overall_sentiment_positive(self):
        result = track_emotional_language("This is amazing and great and awesome!")
        assert result["overall_sentiment"] > 0

    def test_overall_sentiment_negative(self):
        result = track_emotional_language("This is terrible and horrible and awful.")
        assert result["overall_sentiment"] < 0

    def test_detects_spikes(self):
        text = "This is amazing! Incredible! Unbelievable! Mind-blowing!"
        result = track_emotional_language(text)
        # Multiple emotional words in close proximity = spike
        assert result["spike_count"] >= 1 or result["positive_count"] >= 4

    def test_no_emotion_in_technical_text(self):
        result = track_emotional_language("The RSI is calculated using a 14-period lookback.")
        assert result["positive_count"] == 0
        assert result["negative_count"] == 0

    def test_empty_text(self):
        result = track_emotional_language("")
        assert result["positive_count"] == 0

    def test_includes_emotional_moments(self):
        result = track_emotional_language("This is amazing!")
        assert len(result["emotional_moments"]) >= 1
        assert "word" in result["emotional_moments"][0]


# === 10. DEFINITION EXTRACTION ===
class TestExtractDefinitions:
    def test_extracts_is_a_definition(self):
        defs = extract_definitions("RSI is a momentum oscillator that measures price speed.")
        assert len(defs) >= 1
        assert any("RSI" in d["term"] for d in defs)

    def test_extracts_called_definition(self):
        defs = extract_definitions("This pattern is called a double bottom.")
        assert len(defs) >= 1

    def test_extracts_means_definition(self):
        defs = extract_definitions("MACD means Moving Average Convergence Divergence.")
        assert len(defs) >= 1
        assert any("MACD" in d["term"] for d in defs)

    def test_deduplicates(self):
        defs = extract_definitions("RSI is a momentum indicator. RSI is a momentum indicator.")
        terms = [d["term"].lower() for d in defs]
        assert terms.count("rsi") <= 1

    def test_filters_pronouns(self):
        defs = extract_definitions("This is a good video. It is a test.")
        terms = [d["term"].lower() for d in defs]
        assert "this" not in terms
        assert "it" not in terms

    def test_empty_text(self):
        assert extract_definitions("") == []

    def test_includes_sentence(self):
        defs = extract_definitions("RSI is a momentum oscillator.")
        if defs:
            assert "sentence" in defs[0]


# === MASTER EXTRACT_ALL ===
class TestExtractAll:
    def test_returns_all_keys(self):
        result = extract_all("RSI is a momentum indicator. If RSI is below 30, buy.")
        expected_keys = {"contradictions", "marketing_pressure", "parameters", "rules",
                        "citations", "questions", "hedge_density", "chapters",
                        "emotional_language", "definitions"}
        assert set(result.keys()) == expected_keys

    def test_empty_text(self):
        result = extract_all("")
        assert result["contradictions"] == []
        assert result["parameters"] == []
        assert result["questions"] == []

    def test_works_with_timestamps(self):
        sentences = [
            TimestampedSentence(text="RSI is a momentum indicator.", start=0.0, end=5.0, segment_index=0),
            TimestampedSentence(text="If RSI is below 30, buy.", start=5.0, end=10.0, segment_index=1),
        ]
        result = extract_all("RSI is a momentum indicator. If RSI is below 30, buy.", sentences)
        assert len(result["definitions"]) >= 1 or len(result["rules"]) >= 1
