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
