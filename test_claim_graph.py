"""Tests for the cross-video claim graph (backlog #2).

Spec gate: 3 mock packages with 2 shared claims + 1 contradiction ->
assert graph shape (1 node with 3 citations, not 3 nodes; contradiction edge
with both values; frequency weights).
"""
import claim_graph


def _pkg(video_id, title, claims):
    return {"ok": True, "video": {"id": video_id, "title": title},
            "extracted_claims": claims}


def _claim(pattern, types, sentence, subject="", offset=0):
    return {"claim_types": list(types), "matched_pattern": pattern,
            "sentence": sentence, "char_offset": offset, "subject": subject}


def _shared_a():
    return _claim("overbought at 70 level", ["threshold"],
                  "The RSI is overbought at 70 level.", subject="overbought")


def _shared_b():
    return _claim("is a momentum indicator", ["definition_fact"],
                  "The RSI is a momentum indicator.")


class TestGraphShape:
    def _graph(self):
        pkgs = [
            _pkg("vid1", "One", [_shared_a(), _shared_b()]),
            _pkg("vid2", "Two", [_shared_a(), _shared_b()]),
            _pkg("vid3", "Three", [
                _shared_a(),
                _claim("overbought at 80", ["threshold"],
                       "Actually the RSI is overbought at 80.",
                       subject="overbought"),
            ]),
        ]
        return claim_graph.build_claim_graph(pkgs)

    def test_same_claim_in_three_videos_is_one_node_three_citations(self):
        g = self._graph()
        nodes = [n for n in g["nodes"] if n["claim"] == "overbought at 70 level"]
        assert len(nodes) == 1, "same claim must dedupe into ONE node"
        assert len(nodes[0]["citations"]) == 3
        assert nodes[0]["frequency"] == 3
        assert set(nodes[0]["video_ids"]) == {"vid1", "vid2", "vid3"}

    def test_two_shared_claims_plus_outlier_is_three_nodes(self):
        g = self._graph()
        assert g["node_count"] == 3
        assert g["video_count"] == 3

    def test_cross_video_contradiction_edge_with_values(self):
        g = self._graph()
        assert g["edge_count"] == 1
        e = g["edges"][0]
        assert e["type"] == "contradiction"
        assert set(e["values"]) == {"70", "80"}
        assert e["cross_video"] is True
        assert "overbought" in e["reason"]

    def test_frequency_map_ranks_most_repeated_claim_first(self):
        g = self._graph()
        assert g["top_claims"][0]["claim"] == "overbought at 70 level"
        assert g["top_claims"][0]["frequency"] == 3

    def test_citations_carry_video_and_sentence(self):
        g = self._graph()
        node = [n for n in g["nodes"] if n["claim"] == "overbought at 70 level"][0]
        cite = node["citations"][0]
        assert cite["video_id"] == "vid1"
        assert "overbought at 70" in cite["sentence"]


class TestNormalization:
    def test_articles_case_and_whitespace_fold(self):
        assert (claim_graph.normalize_claim_text("The  Biggest")
                == claim_graph.normalize_claim_text("the biggest"))

    def test_trailing_punctuation_folds(self):
        assert (claim_graph.normalize_claim_text("overbought at 70.")
                == "overbought at 70")


class TestConflictLogic:
    def test_same_metric_same_value_is_not_a_conflict(self):
        ok, _, _ = claim_graph.texts_conflict(
            "overbought at 70", "overbought", "overbought at 70 level", "overbought")
        assert ok is False

    def test_different_metrics_do_not_conflict(self):
        ok, _, _ = claim_graph.texts_conflict(
            "overbought at 70", "", "oversold at the 30 level", "")
        assert ok is False

    def test_no_metric_no_conflict(self):
        ok, _, _ = claim_graph.texts_conflict(
            "is a momentum indicator", "", "overbought at 70", "")
        assert ok is False


class TestEdgeCases:
    def test_empty_packages_build_empty_graph(self):
        g = claim_graph.build_claim_graph([])
        assert g["node_count"] == 0
        assert g["edge_count"] == 0
        assert g["nodes"] == [] and g["edges"] == []

    def test_malformed_claims_are_skipped(self):
        pkgs = [_pkg("vid1", "One", [
            {"matched_pattern": "", "claim_types": ["threshold"]},
            "not-a-dict",
        ])]
        g = claim_graph.build_claim_graph(pkgs)
        assert g["node_count"] == 0
