"""Tests for the trading-jargon split guard in yt_sentences (backlog #6).

Pure-function tests: no punctuation model load, no network, no cache.
"""
import unittest

from yt_sentences import _repair_jargon_splits


class TestJargonSplitRepair(unittest.TestCase):
    def test_threshold_level_split_repaired(self):
        broken = "The RSI is overbought at 70. Level and rising fast."
        out = _repair_jargon_splits(broken)
        self.assertNotIn(". level", out.lower())
        self.assertIn("at 70 level", out.lower())

    def test_threshold_level_split_with_article_repaired(self):
        broken = "It is oversold at the 30. Level on the daily chart."
        out = _repair_jargon_splits(broken)
        self.assertNotIn(". level", out.lower())
        self.assertIn("at the 30 level", out.lower())

    def test_indicator_definition_split_repaired(self):
        broken = "RSI. Is a momentum indicator used by traders."
        out = _repair_jargon_splits(broken)
        self.assertNotIn("rsi. is a", out.lower())
        self.assertIn("rsi is a momentum indicator", out.lower())

    def test_idempotent(self):
        broken = (
            "The RSI is overbought at 70. Level and oversold at the 30. Level. "
            "RSI. Is a momentum indicator."
        )
        once = _repair_jargon_splits(broken)
        twice = _repair_jargon_splits(once)
        self.assertEqual(once, twice)

    def test_legitimate_question_untouched(self):
        text = "I use RSI. Is that good?"
        self.assertEqual(_repair_jargon_splits(text), text)

    def test_plain_sentences_untouched(self):
        text = "This is a normal sentence. Another normal sentence follows here."
        self.assertEqual(_repair_jargon_splits(text), text)


if __name__ == "__main__":
    unittest.main()
