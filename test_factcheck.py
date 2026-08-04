"""Tests for the fact-check loop (backlog #3).

Spec gate: mock the search + LLM, assert the verdict ROUTING logic.
No network, no model loads -- everything injectable is injected.
"""
import factcheck


def _claim(sentence, types, pattern="", subject=""):
    return {"claim_types": list(types), "sentence": sentence,
            "matched_pattern": pattern or sentence, "subject": subject}


class TestSelection:
    def test_only_checkable_types_are_selected(self):
        claims = [
            _claim("The RSI was developed by J. Welles Wilder.", ["authority"]),
            _claim("overbought at 70", ["threshold"]),
            _claim("90% of traders lose money.", ["statistical"]),
        ]
        picked = factcheck.select_checkable_claims(claims)
        assert len(picked) == 2
        assert all(set(c["claim_types"]) & set(factcheck.CHECKABLE_TYPES)
                   for c in picked)

    def test_limit_respected(self):
        claims = [_claim(f"claim {i}", ["statistical"]) for i in range(10)]
        assert len(factcheck.select_checkable_claims(claims, limit=3)) == 3

    def test_type_filter_narrows_selection(self):
        claims = [
            _claim("best indicator ever", ["superlative"]),
            _claim("90% of traders lose", ["statistical"]),
        ]
        picked = factcheck.select_checkable_claims(claims, types=["statistical"])
        assert len(picked) == 1
        assert picked[0]["claim_types"] == ["statistical"]


class TestVerdictRouting:
    def test_verified_route_keeps_source_url(self):
        evidence = [{"url": "https://en.wikipedia.org/wiki/Relative_strength_index",
                     "title": "RSI", "snippet": "Developed by J. Welles Wilder"}]
        seen = {}

        def fake_search(q, max_results=5):
            seen["query"] = q
            return evidence

        def fake_verdict(sentence, ev, model):
            assert ev == evidence
            assert model == "fake-model"
            return {"verdict": "verified", "reason": "matches evidence",
                    "sources": [ev[0]["url"]]}

        out = factcheck.factcheck_claims(
            [_claim("The RSI was developed by J. Welles Wilder.", ["authority"])],
            model="fake-model", search_fn=fake_search, verdict_fn=fake_verdict)
        assert out["checked_count"] == 1
        entry = out["checked"][0]
        assert entry["verdict"] == "verified"
        assert entry["sources"] == [evidence[0]["url"]]
        assert out["verdict_counts"]["verified"] == 1
        assert seen["query"], "search must receive a non-empty query"

    def test_no_search_results_routes_to_unverifiable(self):
        out = factcheck.factcheck_claims(
            [_claim("This strategy never loses.", ["superlative"])],
            model="",  # no model available -> guard path
            search_fn=lambda q, max_results=5: [])
        assert out["checked"][0]["verdict"] == "unverifiable"
        assert out["verdict_counts"]["unverifiable"] == 1

    def test_contradicted_route_counts(self):
        def fake_verdict(sentence, ev, model):
            return {"verdict": "contradicted", "reason": "refuted",
                    "sources": ["https://example.com/a"]}

        out = factcheck.factcheck_claims(
            [_claim("This indicator wins 100% of the time.", ["superlative"])],
            model="fake",
            search_fn=lambda q, max_results=5: [
                {"url": "https://example.com/a", "title": "t", "snippet": "s"}],
            verdict_fn=fake_verdict)
        assert out["verdict_counts"]["contradicted"] == 1

    def test_uncheckable_claims_are_skipped_entirely(self):
        calls = {"n": 0}

        def fake_search(q, max_results=5):
            calls["n"] += 1
            return []

        out = factcheck.factcheck_claims(
            [_claim("overbought at 70", ["threshold"])],
            model="", search_fn=fake_search)
        assert out["checked_count"] == 0
        assert calls["n"] == 0


class TestLlmVerdictGuards:
    def test_no_model_is_unverifiable(self):
        v = factcheck.llm_verdict(
            "claim", [{"url": "u", "title": "t", "snippet": "s"}], "")
        assert v["verdict"] == "unverifiable"

    def test_no_evidence_is_unverifiable(self):
        v = factcheck.llm_verdict("claim", [], "some-model")
        assert v["verdict"] == "unverifiable"

    def test_bad_llm_json_degrades_to_unverifiable(self, monkeypatch):
        monkeypatch.setattr(factcheck, "_ollama_generate",
                            lambda model, prompt: "NOT JSON AT ALL")
        v = factcheck.llm_verdict(
            "claim", [{"url": "https://x", "title": "t", "snippet": "s"}], "m")
        assert v["verdict"] == "unverifiable"
        assert v["sources"] == ["https://x"]

    def test_unknown_verdict_string_is_normalized(self, monkeypatch):
        monkeypatch.setattr(
            factcheck, "_ollama_generate",
            lambda model, prompt: '{"verdict": "MAYBE", "reason": "?"}')
        v = factcheck.llm_verdict(
            "claim", [{"url": "https://x", "title": "t", "snippet": "s"}], "m")
        assert v["verdict"] == "unverifiable"


class TestQueryBuilding:
    def test_subject_prefixes_query_when_missing(self):
        q = factcheck.build_query(_claim("It is overbought at 70.",
                                         ["threshold"], subject="rsi"))
        assert q.lower().startswith("rsi ")

    def test_long_sentences_fall_back_to_span(self):
        q = factcheck.build_query(_claim("x" * 200, ["statistical"],
                                         pattern="90% of traders"))
        assert q == "90% of traders"


class TestDdgUrlCleaning:
    def test_uddg_unwrap(self):
        wrapped = ("//duckduckgo.com/l/?uddg="
                   "https%3A%2F%2Fen.wikipedia.org%2Fwiki%2FRSI&rut=abc")
        assert (factcheck._clean_ddg_url(wrapped)
                == "https://en.wikipedia.org/wiki/RSI")

    def test_plain_url_passthrough(self):
        assert (factcheck._clean_ddg_url("https://example.com/x")
                == "https://example.com/x")
