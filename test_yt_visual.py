"""Tests for the restored visual extraction pipeline.

These are wiring/contract tests: none of them load the vision model or hit the
network, so they stay fast. The lesson from the punctuation-restoration bug is
baked in here: a feature that silently no-ops must be detectable, so several of
these tests assert on the FAILURE paths rather than the happy path.
"""

import base64
import json
import subprocess
import sys
from pathlib import Path

import pytest

import visual

REPO = Path(__file__).parent


class TestClaimParsing:
    """The claim parser is pure text -> claims, so it is fully testable."""

    def test_indicator_settings_and_thresholds(self):
        text = "RSI INDICATOR SETTINGS\nLength: 14\nRSI above 70\nOversold: 30"
        claims = visual.extract_visual_claims(text, 60.0)
        types = {c["type"] for c in claims}
        assert "indicator_setting" in types
        assert "indicator_threshold" in types
        assert any("14" in c["claim"] for c in claims)
        assert all(c["timestamp"] == 60.0 for c in claims)
        assert all(c["timestamp_str"] == "01:00" for c in claims)

    def test_bullets_prices_and_percentages(self):
        text = "1. What is the RSI?\n- Momentum matters\n$1,250 target\n70% win rate"
        claims = visual.extract_visual_claims(text, 0.0)
        types = {c["type"] for c in claims}
        assert "bullet_point" in types
        assert "price_level" in types
        assert "percentage" in types

    def test_empty_text_yields_no_claims(self):
        assert visual.extract_visual_claims("", 0.0) == []
        assert visual.extract_visual_claims("   \n  \n ", 0.0) == []

    def test_source_label_is_recorded(self):
        claims = visual.extract_visual_claims("Length: 14", 0.0)
        assert claims and all(c["source"] == "visual_vision" for c in claims)


class TestResultContract:
    """Callers depend on the result shape even when everything fails."""

    def test_disabled_returns_full_shape_without_work(self):
        r = visual.extract_visual_content("abc", REPO, enable_visual=False)
        assert r["enabled"] is False
        assert r["frames_extracted"] == 0
        assert r["visual_claims"] == []
        for key in visual.empty_result().keys():
            assert key in r, f"missing contract key: {key}"

    def test_result_is_json_serialisable(self):
        # The dict is embedded in the research package, which is dumped to JSON.
        json.dumps(visual.empty_result(True))

    def test_unreachable_ollama_is_reported_not_raised(self, monkeypatch):
        monkeypatch.setattr(visual, "OLLAMA_HOST", "http://127.0.0.1:9")
        ok, detail = visual.vision_available()
        assert ok is False
        assert "not reachable" in detail

    def test_extraction_never_raises_when_backend_is_down(self, monkeypatch, tmp_path):
        monkeypatch.setattr(visual, "OLLAMA_HOST", "http://127.0.0.1:9")
        r = visual.extract_visual_content("abc", tmp_path, enable_visual=True)
        assert r["vision_available"] is False
        assert r["vision_error"]
        assert r["video_downloaded"] is False

    def test_read_frame_returns_error_dict_for_missing_file(self, tmp_path):
        r = visual.read_frame(tmp_path / "nope.png", 0.0)
        assert r["has_content"] is False
        assert "error" in r


class TestTextOnlyModelIsRejected:
    """A text-only model would confabulate about images it cannot see."""

    def test_model_without_vision_capability_is_refused(self, monkeypatch):
        fake = {"models": [{"name": "textonly:latest",
                            "capabilities": ["completion"]}]}

        class FakeResp:
            def read(self):
                return json.dumps(fake).encode()

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        monkeypatch.setattr(visual, "VISION_MODEL", "textonly:latest")
        monkeypatch.setattr(visual.urllib.request, "urlopen",
                            lambda *a, **k: FakeResp())
        ok, detail = visual.vision_available()
        assert ok is False
        assert "vision" in detail


class TestFfmpegResolution:
    def test_ffmpeg_is_resolvable(self):
        # Either on PATH or via the imageio-ffmpeg wheel declared in requirements.
        assert visual.find_ffmpeg(), "no ffmpeg available; frame extraction cannot work"

    def test_no_frames_without_ffmpeg(self, monkeypatch, tmp_path):
        monkeypatch.setattr(visual, "_FFMPEG_CMD", "")
        monkeypatch.setattr(visual.shutil, "which", lambda _n: None)
        monkeypatch.setitem(sys.modules, "imageio_ffmpeg", None)
        assert visual.extract_frames(tmp_path / "v.mp4", tmp_path, duration=120) == []


class TestCliWiring:
    """The docs promised --visual for months while the flag did not exist."""

    def test_deep_research_exposes_visual_flag(self):
        proc = subprocess.run(
            [sys.executable, "yt_scrape.py", "deep-research", "--help"],
            cwd=REPO, capture_output=True, text=True, timeout=120,
        )
        assert proc.returncode == 0, proc.stderr
        assert "--visual" in proc.stdout


class TestDependencyDeclaration:
    def test_requirements_declare_visual_deps(self):
        reqs = (REPO / "requirements.txt").read_text(encoding="utf-8").lower()
        for dep in ("pillow", "imageio-ffmpeg"):
            assert dep in reqs, f"{dep} missing from requirements.txt"


class TestRound2GroundTruth:
    """Regression suite from the 2026-08-04 Round 2 hard slides.

    These are the EXACT texts the 7B recovered (12/12 needles, verbatim).
    The parser must promote every number that costs money if misread, and
    must promote nothing from hype text. Pre-registered kill criteria: if a
    vocab change makes the hype test fail, the vocab change gets reverted.
    """

    BULLETS = ("POSITION SIZING RULES\n- Risk per trade: 1.5%\n"
               "- Max daily loss: 4.5%\n- Stop loss: 1.8 ATR\n"
               "- Target: 2.7 R\n- Win rate needed: 38%")
    PANEL = ("INDICATOR PANEL\nEMA Length: 21\nRSI Period: 9\n"
             "Timeframe: 15M\nATR Multiplier: 2.5")
    CHART = "Resistance $1,285.50\nSupport $1,240.00\nXAUUSD 4H"
    HYPE = "F*CK THAT! WE ARE GOING TO MAKE IT HAPPEN!"

    def test_dense_bullets_promote_all_five_numbers(self):
        claims = visual.extract_visual_claims(self.BULLETS, 0.0)
        joined = " | ".join(c["claim"] for c in claims)
        for needle in ("1.5", "4.5", "1.8", "2.7", "38"):
            assert needle in joined, f"{needle} not promoted: {joined}"

    def test_panel_promotes_multiplier_and_timeframe(self):
        claims = visual.extract_visual_claims(self.PANEL, 0.0)
        joined = " | ".join(c["claim"] for c in claims)
        assert "Multiplier = 2.5" in joined, joined
        assert "Length = 21" in joined, joined
        assert "Period = 9" in joined, joined
        assert any(c["type"] == "timeframe" and "15M" in c["claim"]
                   for c in claims), joined

    def test_chart_promotes_prices_and_timeframe(self):
        claims = visual.extract_visual_claims(self.CHART, 0.0)
        joined = " | ".join(c["claim"] for c in claims)
        assert "$1,285.50" in joined, joined
        assert "$1,240.00" in joined, joined
        assert any(c["type"] == "timeframe" and "4H" in c["claim"]
                   for c in claims), joined

    def test_hype_text_promotes_nothing(self):
        assert visual.extract_visual_claims(self.HYPE, 0.0) == []

    def test_dollar_millions_is_not_a_timeframe(self):
        claims = visual.extract_visual_claims("$100M raised", 0.0)
        assert not [c for c in claims if c["type"] == "timeframe"]


class TestFrameSelection:
    """Round 4 (#1): global dedup + density cap must gut the vision queue.

    Spec gate: a video with known duplicate-slide runs -> dedup catches them
    even when the duplicates are NOT adjacent (the old consecutive-only dedup
    kept every re-appearance).
    """

    @staticmethod
    def _dense_grid(tmp_path, name):
        from PIL import Image, ImageDraw
        im = Image.new("RGB", (320, 180), "white")
        d = ImageDraw.Draw(im)
        for row in range(0, 170, 12):
            for col in range(0, 300, 34):
                d.rectangle([col, row, col + 20, row + 6], fill="black")
        p = tmp_path / name
        im.save(p)
        return p

    @staticmethod
    def _dense_bars(tmp_path, name):
        from PIL import Image, ImageDraw
        im = Image.new("RGB", (320, 180), "white")
        d = ImageDraw.Draw(im)
        for col in range(0, 300, 24):
            d.rectangle([col, 10, col + 8, 170], fill="black")
        p = tmp_path / name
        im.save(p)
        return p

    @staticmethod
    def _sparse(tmp_path, name):
        from PIL import Image, ImageDraw
        im = Image.new("RGB", (320, 180), "white")
        ImageDraw.Draw(im).rectangle([150, 80, 170, 100], fill="gray")
        p = tmp_path / name
        im.save(p)
        return p

    def test_duplicate_slide_runs_are_collapsed_globally(self, tmp_path):
        import shutil as _sh
        # A B A B A B — the same two slides alternating (speaker cuts back
        # and forth). Consecutive-only dedup keeps all six.
        a = self._dense_grid(tmp_path, "f0.png")
        b = self._dense_bars(tmp_path, "f1.png")
        frames = [a, b]
        for i, src in ((2, a), (3, b), (4, a), (5, b)):
            dst = tmp_path / f"f{i}.png"
            _sh.copyfile(src, dst)
            frames.append(dst)
        kept = visual.select_frames(list(frames), target=8)
        assert [k.name for k in kept] == ["f0.png", "f1.png"], (
            "non-adjacent duplicates must collapse: 6 frames -> 2 (67% cut)")

    def test_density_cap_keeps_densest_in_chronological_order(self, tmp_path):
        f0 = self._dense_grid(tmp_path, "f0.png")
        f1 = self._sparse(tmp_path, "f1.png")
        f2 = self._dense_bars(tmp_path, "f2.png")
        kept = visual.select_frames([f0, f1, f2], target=2)
        assert [k.name for k in kept] == ["f0.png", "f2.png"], (
            "the near-blank frame must lose its seat to text-dense slides")

    def test_density_ranks_text_over_blank(self, tmp_path):
        dense = self._dense_grid(tmp_path, "dense.png")
        sparse = self._sparse(tmp_path, "sparse.png")
        assert visual.frame_density(dense) > visual.frame_density(sparse)

    def test_default_cap_comes_from_module_constant(self, monkeypatch, tmp_path):
        monkeypatch.setattr(visual, "TARGET_FRAMES", 1)
        f0 = self._dense_grid(tmp_path, "f0.png")
        f1 = self._dense_bars(tmp_path, "f1.png")
        kept = visual.select_frames([f0, f1])  # target=0 -> use TARGET_FRAMES
        assert len(kept) == 1

    def test_identical_hashes_have_zero_distance(self, tmp_path):
        f0 = self._dense_grid(tmp_path, "f0.png")
        h = visual.frame_dhash(f0)
        assert h != 0
        assert visual.hash_distance(h, h) == 0

    def test_unreadable_frame_hashes_to_zero_and_survives(self, tmp_path):
        bogus = tmp_path / "bogus.png"
        bogus.write_bytes(b"not an image")
        assert visual.frame_dhash(bogus) == 0
        # A zero hash must never match other frames as a duplicate.
        real = self._dense_grid(tmp_path, "real.png")
        kept = visual.select_frames([bogus, real], target=8)
        assert len(kept) == 2


class TestVisionCallConfig:
    """The Round 1+2 adoptions must actually be in the request payload.

    Round 2 caught the shipped code still sending native PNG with no cap,
    no ctx, and no keep_alive while the reports said adopted. Never again:
    these tests read the real payload read_frame builds.
    """

    def _capture(self, monkeypatch, tmp_path):
        captured = {}

        class FakeResp:
            def read(self):
                return json.dumps({"response": json.dumps(
                    {"on_screen_text": "Length: 14",
                     "chart_description": ""})}).encode()

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        def fake_urlopen(req, timeout=0):
            captured["payload"] = json.loads(req.data.decode("utf-8"))
            return FakeResp()

        # Pin the provider: this class asserts on the OLLAMA payload, and
        # resolve_provider() would pick Gemini on a machine with a key set.
        monkeypatch.setattr(visual, "VISION_PROVIDER", "ollama")
        monkeypatch.setattr(visual.urllib.request, "urlopen", fake_urlopen)
        from PIL import Image
        frame = tmp_path / "f.png"
        Image.new("RGB", (1920, 1080), (30, 30, 30)).save(frame)
        reading = visual.read_frame(frame, 0.0)
        return captured["payload"], reading

    def test_payload_carries_cap_ctx_and_keep_alive(self, monkeypatch, tmp_path):
        payload, reading = self._capture(monkeypatch, tmp_path)
        assert payload["keep_alive"] == visual.VISION_KEEP_ALIVE
        assert payload["options"]["num_predict"] == visual.VISION_NUM_PREDICT
        assert payload["options"]["num_ctx"] == visual.VISION_NUM_CTX
        if visual.VISION_NUM_GPU:
            assert payload["options"]["num_gpu"] == int(visual.VISION_NUM_GPU)
        else:
            assert "num_gpu" not in payload["options"]
        assert reading["text"] == "Length: 14"

    def test_frame_is_downscaled_jpeg(self, monkeypatch, tmp_path):
        import io
        from PIL import Image
        payload, _ = self._capture(monkeypatch, tmp_path)
        img_bytes = base64.b64decode(payload["images"][0])
        assert img_bytes[:2] == b"\xff\xd8", "frame is not JPEG-encoded"
        assert Image.open(io.BytesIO(img_bytes)).width <= visual.FRAME_MAX_WIDTH


class TestGeminiBackend:
    """Contract tests for the Gemini Interactions backend (added 2026-09-02).

    No network. These pin the request against the schema published at
    ai.google.dev on 2026-09-02 and the response parser against the vendor's
    own verbatim example, so a doc drift shows up here rather than as an empty
    reading in production.
    """

    # Verbatim from ai.google.dev/api/interactions.md.txt, 2026-09-02.
    DOC_RESPONSE = {
        "created": "2025-11-26T12:25:15Z",
        "id": "v1_ChdPU0F4YWFtNkFwS2kxZThQZ05lbXdROBIX",
        "model": "gemini-3.6-flash",
        "object": "interaction",
        "status": "completed",
        "steps": [
            {
                "type": "model_output",
                "content": [
                    {"type": "text", "text": "Hello! I'm functioning perfectly."}
                ],
            }
        ],
        "updated": "2025-11-26T12:25:15Z",
        "usage": {"total_tokens": 49},
    }

    def _capture_gemini(self, monkeypatch, tmp_path, answer=None):
        captured = {}

        class FakeResp:
            def read(self):
                body = dict(TestGeminiBackend.DOC_RESPONSE)
                if answer is not None:
                    body["steps"] = [{
                        "type": "model_output",
                        "content": [{"type": "text", "text": answer}],
                    }]
                return json.dumps(body).encode()

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        def fake_urlopen(req, timeout=0):
            captured["url"] = req.full_url
            captured["headers"] = dict(req.headers)
            captured["payload"] = json.loads(req.data.decode("utf-8"))
            return FakeResp()

        monkeypatch.setattr(visual, "VISION_PROVIDER", "gemini")
        monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-a-real-secret")
        monkeypatch.setattr(visual.urllib.request, "urlopen", fake_urlopen)
        from PIL import Image
        frame = tmp_path / "f.png"
        Image.new("RGB", (1920, 1080), (20, 20, 20)).save(frame)
        reading = visual.read_frame(frame, 0.0)
        return captured, reading

    def test_request_matches_documented_interactions_schema(
            self, monkeypatch, tmp_path):
        cap, _ = self._capture_gemini(monkeypatch, tmp_path)
        assert cap["url"] == visual.GEMINI_ENDPOINT
        assert cap["url"].endswith("/v1beta/interactions")
        # Header names are title-cased by urllib's internal store.
        headers = {k.lower(): v for k, v in cap["headers"].items()}
        assert headers["x-goog-api-key"] == "test-key-not-a-real-secret"
        assert headers["api-revision"] == visual.GEMINI_API_REVISION
        body = cap["payload"]
        assert body["model"] == visual.GEMINI_MODEL
        # input[] is a flat list of type-tagged parts, not contents/parts.
        assert isinstance(body["input"], list)
        assert body["input"][0]["type"] == "text"
        img = body["input"][1]
        assert img["type"] == "image"
        assert img["mime_type"] == "image/jpeg"
        assert base64.b64decode(img["data"])[:2] == b"\xff\xd8"

    def test_json_contract_is_enforced_server_side(self, monkeypatch, tmp_path):
        cap, _ = self._capture_gemini(monkeypatch, tmp_path)
        fmt = cap["payload"]["response_format"]
        assert fmt["mime_type"] == "application/json"
        # segments joined the contract 2026-09-02; the two prose keys stay so
        # every existing consumer keeps working.
        assert set(fmt["schema"]["required"]) == {
            "on_screen_text", "chart_description", "segments"}
        seg = fmt["schema"]["properties"]["segments"]["items"]
        for field in ("instrument", "timeframe", "indicators", "patterns"):
            assert field in seg["properties"], field

    def test_parses_vendors_verbatim_response_example(self):
        text = visual._gemini_extract_text(self.DOC_RESPONSE)
        assert text == "Hello! I'm functioning perfectly."

    def test_parses_legacy_outputs_shape(self):
        legacy = {"id": "int_123", "role": "model",
                  "outputs": [{"type": "text", "text": "legacy body"}]}
        assert visual._gemini_extract_text(legacy) == "legacy body"

    def test_unknown_response_shape_yields_empty_not_crash(self):
        assert visual._gemini_extract_text({"nonsense": True}) == ""

    def test_two_key_answer_splits_into_text_and_chart(
            self, monkeypatch, tmp_path):
        answer = json.dumps({
            "on_screen_text": "RSI Length: 14\nEURUSD 4H",
            "chart_description": "Candlestick chart with an RSI sub-panel.",
        })
        _, reading = self._capture_gemini(monkeypatch, tmp_path, answer=answer)
        assert reading["text"].startswith("RSI Length: 14")
        assert reading["chart"] == "Candlestick chart with an RSI sub-panel."
        assert reading["has_content"] is True
        assert reading["provider"] == "gemini"

    def test_missing_key_is_a_clean_error(self, monkeypatch, tmp_path):
        for name in ("YT_GEMINI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setattr(visual, "VISION_PROVIDER", "gemini")
        from PIL import Image
        frame = tmp_path / "f.png"
        Image.new("RGB", (64, 64), (0, 0, 0)).save(frame)
        reading = visual.read_frame(frame, 0.0)
        assert "GEMINI_API_KEY" in reading["error"]
        assert reading["has_content"] is False

    def test_api_key_never_appears_in_an_error_string(
            self, monkeypatch, tmp_path):
        secret = "AIza-this-must-never-be-echoed-anywhere"
        monkeypatch.setattr(visual, "VISION_PROVIDER", "gemini")
        monkeypatch.setenv("GEMINI_API_KEY", secret)

        def boom(req, timeout=0):
            raise OSError("connection reset")

        monkeypatch.setattr(visual.urllib.request, "urlopen", boom)
        from PIL import Image
        frame = tmp_path / "f.png"
        Image.new("RGB", (64, 64), (0, 0, 0)).save(frame)
        reading = visual.read_frame(frame, 0.0)
        assert reading["error"]
        assert secret not in json.dumps(reading)

    def test_youtube_video_part_shape(self, monkeypatch):
        captured = {}

        class FakeResp:
            def read(self):
                return json.dumps({"steps": [{"type": "model_output",
                    "content": [{"type": "text", "text": "{}"}]}]}).encode()

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        def fake_urlopen(req, timeout=0):
            captured["payload"] = json.loads(req.data.decode("utf-8"))
            return FakeResp()

        monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-a-real-secret")
        monkeypatch.setattr(visual.urllib.request, "urlopen", fake_urlopen)
        url = "https://www.youtube.com/watch?v=abc123"
        visual.analyze_youtube_video(url)
        parts = captured["payload"]["input"]
        assert parts[0]["type"] == "text"
        # video-understanding docs: {"type":"video","uri":...}; no mime_type
        # is required for a YouTube URL.
        assert parts[1] == {"type": "video", "uri": url}


class TestProviderSelection:
    def test_auto_stays_on_ollama_without_a_key(self, monkeypatch):
        for name in ("YT_GEMINI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setattr(visual, "VISION_PROVIDER", "auto")
        assert visual.resolve_provider() == "ollama"

    def test_auto_switches_to_gemini_when_a_key_exists(self, monkeypatch):
        monkeypatch.setattr(visual, "VISION_PROVIDER", "auto")
        monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-a-real-secret")
        assert visual.resolve_provider() == "gemini"

    def test_explicit_ollama_ignores_a_present_key(self, monkeypatch):
        monkeypatch.setattr(visual, "VISION_PROVIDER", "ollama")
        monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-a-real-secret")
        assert visual.resolve_provider() == "ollama"


class TestTradingProfile:
    def test_trading_is_the_default_profile(self, monkeypatch):
        monkeypatch.setattr(visual, "VISION_PROFILE", "trading")
        assert visual.active_prompt() == visual.TRADING_PROMPT

    def test_general_profile_restores_the_old_prompt(self):
        assert visual.active_prompt("general") == visual.VISION_PROMPT

    def test_trading_prompt_asks_for_the_fields_a_trader_needs(self):
        p = visual.TRADING_PROMPT.lower()
        for token in ("candlestick", "timeframe", "trendlines", "indicator"):
            assert token in p, token

    def test_trading_prompt_keeps_the_two_key_json_contract(self):
        assert '"on_screen_text"' in visual.TRADING_PROMPT
        assert '"chart_description"' in visual.TRADING_PROMPT


class TestFailuresAreVisible:
    """A run where every frame failed must not report vision_error "".

    Measured 2026-09-02: a real run returned frames_failed=1,
    frames_analyzed=0 and vision_error="" -- the per-frame TimeoutError went
    to stderr only, so the JSON looked clean while carrying nothing.
    """

    def test_chart_description_produces_claims(self):
        chart = "Candlestick chart on the 4H timeframe with RSI Length: 14"
        claims = visual.extract_visual_claims(chart, 0.0, source="visual_chart")
        kinds = {c["type"] for c in claims}
        assert "indicator_setting" in kinds
        assert "timeframe" in kinds
        assert all(c["source"] == "visual_chart" for c in claims)

    def test_result_shape_carries_provider_mode_and_frame_errors(self):
        r = visual.empty_result(True)
        assert r["mode"] == "frames"
        assert r["frame_errors"] == []
        assert r["provider"] in ("ollama", "gemini")


class TestProseChartClaims:
    """chart_description is prose, not slide text (added 2026-09-02).

    Measured 2026-09-02: the parser read "RSI Length: 14" from a slide but
    dropped "RSI(14)", "200-period EMA" and "oversold below 30" from a chart
    description, so a good reading produced zero claims.
    """

    def test_parenthesised_indicator_settings(self):
        claims = visual.extract_visual_claims(
            "RSI(14) sub-panel and an EMA(200) overlay", 0.0)
        settings = {c["claim"] for c in claims
                    if c["type"] == "indicator_setting"}
        assert "RSI = 14" in settings
        assert "EMA = 200" in settings

    def test_period_first_indicator_settings(self):
        claims = visual.extract_visual_claims(
            "a 200-period EMA and a 14 period RSI", 0.0)
        settings = {c["claim"] for c in claims
                    if c["type"] == "indicator_setting"}
        assert "EMA = 200" in settings
        assert "RSI = 14" in settings

    def test_threshold_survives_words_in_between(self):
        claims = visual.extract_visual_claims(
            "the RSI(14) sub-panel is oversold below 30", 0.0)
        assert any(c["claim"] == "RSI below 30" for c in claims)

    def test_named_chart_patterns_become_claims(self):
        claims = visual.extract_visual_claims(
            "Candlestick chart showing a bullish divergence, a double top "
            "and a trendline break", 0.0)
        terms = {c["claim"] for c in claims if c["type"] == "chart_pattern"}
        assert "candlestick chart" in terms
        assert "bullish divergence" in terms
        assert "double top" in terms

    def test_unsigned_price_levels_are_read(self):
        claims = visual.extract_visual_claims(
            "stop loss at 2350.5 and resistance at 1.0950", 0.0)
        levels = {c["claim"] for c in claims if c["type"] == "price_level"}
        assert "Stop Loss at 2350.5" in levels, levels
        assert "Resistance at 1.0950" in levels, levels

    def test_bare_numbers_do_not_become_price_claims(self):
        claims = visual.extract_visual_claims(
            "the price went up a lot today, roughly 2350 points of it", 0.0)
        assert not [c for c in claims if c["type"] == "price_level"]


class TestTimestampsAreNotContent:
    """Video mode prefixes every line with MM:SS (added 2026-09-02).

    Caught on the first live Gemini run, not by any offline test: 4 of 11
    extracted indicator_settings were clock times. "00:29 RSI Settings" became
    "RSI = 29". A timestamp is data ABOUT the line, not content IN it.
    """

    TIMESTAMPED = (
        "00:01 BTCUSD 1W CRYPTO | RSI 14 close, SMA 14 close\n"
        "00:29 RSI Settings: RSI Length 14, Source Close, MA Length 14\n"
        "00:31 RSI Style: RSI Upper Band 70, RSI Middle Band 50\n"
        "02:44 RSI Style: RSI color white, RSI-based MA yellow"
    )

    def test_clock_times_do_not_become_indicator_settings(self):
        claims = visual.extract_visual_claims(self.TIMESTAMPED, 0.0)
        settings = {c["claim"] for c in claims
                    if c["type"] == "indicator_setting"}
        for bogus in ("RSI = 29", "RSI = 31", "RSI = 44", "RSI = 1"):
            assert bogus not in settings, f"{bogus} came from a timestamp"

    def test_real_settings_are_still_read_without_a_colon(self):
        """The settings panel writes "RSI Length 14" with no colon, which the
        original [:=]-anchored parser missed entirely."""
        claims = visual.extract_visual_claims(self.TIMESTAMPED, 0.0)
        settings = {c["claim"] for c in claims
                    if c["type"] == "indicator_setting"}
        assert "RSI = 14" in settings, settings
        assert "SMA = 14" in settings, settings

    def test_band_values_do_not_become_settings(self):
        claims = visual.extract_visual_claims("RSI Upper Band 70", 0.0)
        assert "RSI = 70" not in {c["claim"] for c in claims}


class TestReportedModelMatchesProvider:
    def test_gemini_run_reports_the_gemini_model(self, monkeypatch):
        monkeypatch.setattr(visual, "VISION_PROVIDER", "gemini")
        monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-a-real-secret")
        assert visual.empty_result(True)["vision_model"] == visual.GEMINI_MODEL

    def test_ollama_run_reports_the_ollama_model(self, monkeypatch):
        monkeypatch.setattr(visual, "VISION_PROVIDER", "ollama")
        assert visual.empty_result(True)["vision_model"] == visual.VISION_MODEL


class TestLongVideoWindows:
    """Video costs ~100 tokens/sec at low media resolution, so a 1M-context
    model reaches ~3h in one request. Longer videos must be windowed."""

    def test_short_video_is_not_windowed(self):
        assert visual.video_windows(491) == []
        assert visual.video_windows(1800) == []

    def test_long_video_splits_into_contiguous_windows(self):
        w = visual.video_windows(4355, chunk=1800)
        assert w[0][0] == 0.0
        assert w[-1][1] == 4355
        for a, b in zip(w, w[1:]):
            assert a[1] == b[0], "windows must not overlap or gap"

    def test_short_tail_is_folded_not_left_as_a_sliver(self):
        w = visual.video_windows(1900, chunk=1800)
        assert len(w) == 1 and w[0] == (0.0, 1900.0)

    def test_nine_hour_course_is_tractable(self):
        w = visual.video_windows(32400, chunk=1800)
        assert len(w) == 18
        assert w[-1][1] == 32400

    def test_hhmmss_uses_hours_for_long_videos(self):
        assert visual._hhmmss(0) == "00:00:00"
        assert visual._hhmmss(7325) == "02:02:05"


class TestVideoClipRequest:
    """Clipping a YOUTUBE URL is undocumented -- the docs show it only for
    uploaded files. Probed live 2026-09-02: duration STRINGS work, integer
    milliseconds return HTTP 400 Invalid input at 'input[1].processing'."""

    def _capture(self, monkeypatch, **kw):
        captured = {}

        class FakeResp:
            def read(self):
                return json.dumps({"steps": [{"type": "model_output",
                    "content": [{"type": "text", "text": "{}"}]}]}).encode()

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        def fake_urlopen(req, timeout=0):
            captured["payload"] = json.loads(req.data.decode("utf-8"))
            return FakeResp()

        monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-a-real-secret")
        monkeypatch.setattr(visual.urllib.request, "urlopen", fake_urlopen)
        visual.analyze_youtube_video("https://www.youtube.com/watch?v=abc", **kw)
        return captured["payload"]

    def test_unclipped_request_sends_no_processing_block(self, monkeypatch):
        payload = self._capture(monkeypatch)
        assert payload["input"][1] == {
            "type": "video", "uri": "https://www.youtube.com/watch?v=abc"}

    def test_offsets_are_duration_strings_not_milliseconds(self, monkeypatch):
        payload = self._capture(monkeypatch, start_s=7200, end_s=7320)
        proc = payload["input"][1]["processing"]
        assert proc["type"] == "static"
        assert proc["start_offset"] == "7200s", "ints return HTTP 400"
        assert proc["end_offset"] == "7320s"

    def test_clip_prompt_demands_absolute_timestamps(self, monkeypatch):
        payload = self._capture(monkeypatch, start_s=1800, end_s=3600)
        text = payload["input"][0]["text"]
        assert "00:30:00" in text and "01:00:00" in text
        assert "ABSOLUTE" in text


class TestProxyCompatibleAuth:
    """Third-party gateways re-expose the native Gemini API but do not all
    accept Google's header set (added 2026-09-02)."""

    def test_google_default_uses_x_goog_and_pins_the_revision(self, monkeypatch):
        monkeypatch.setattr(visual, "GEMINI_AUTH_STYLE", "x-goog")
        monkeypatch.setattr(visual, "GEMINI_SEND_REVISION", True)
        h = visual._gemini_headers("K")
        assert h["x-goog-api-key"] == "K"
        assert h["Api-Revision"] == visual.GEMINI_API_REVISION
        assert "Authorization" not in h

    def test_bearer_style_for_proxies(self, monkeypatch):
        monkeypatch.setattr(visual, "GEMINI_AUTH_STYLE", "bearer")
        h = visual._gemini_headers("K")
        assert h["Authorization"] == "Bearer K"
        assert "x-goog-api-key" not in h

    def test_both_sends_either_header(self, monkeypatch):
        monkeypatch.setattr(visual, "GEMINI_AUTH_STYLE", "both")
        h = visual._gemini_headers("K")
        assert h["Authorization"] == "Bearer K"
        assert h["x-goog-api-key"] == "K"

    def test_revision_can_be_dropped_for_strict_proxies(self, monkeypatch):
        monkeypatch.setattr(visual, "GEMINI_SEND_REVISION", False)
        assert "Api-Revision" not in visual._gemini_headers("K")

    def test_endpoint_is_overridable_without_code_changes(self):
        import os
        assert os.environ.get("YT_GEMINI_ENDPOINT") or visual.GEMINI_ENDPOINT
        # the constant is env-driven, which is what lets a proxy be swapped in
        assert "YT_GEMINI_ENDPOINT" in open("visual.py", encoding="utf-8").read()


class TestTypedSegments:
    """Model-filled fields instead of regex over prose (added 2026-09-02).

    The regex layer shipped two real bugs in one day. These claims come from
    fields the model populated, so no pattern can mistake a clock for a period.
    """

    SEG = {
        "timestamp": "01:02:03", "instrument": "BTCUSD", "timeframe": "15m",
        "chart_type": "candlestick", "trend": "downtrend",
        "indicators": [{"name": "RSI", "period": "14"},
                       {"name": "MACD", "period": ""}],
        "drawn": ["descending trendline"], "patterns": ["bearish divergence"],
        "levels": ["42150"],
    }

    def _claims(self, seg=None):
        return visual.claims_from_segments([seg or dict(self.SEG)])

    def test_instrument_and_timeframe_become_claims(self):
        got = {(c["type"], c["claim"]) for c in self._claims()}
        assert ("instrument", "Instrument: BTCUSD") in got
        assert ("timeframe", "Timeframe: 15m") in got

    def test_indicator_period_needs_no_parsing(self):
        got = {c["claim"] for c in self._claims()}
        assert "RSI = 14" in got
        assert "MACD present" in got, "an indicator with no shown period"

    def test_timestamp_parses_to_absolute_seconds(self):
        c = self._claims()[0]
        assert c["timestamp"] == 3723.0
        assert c["timestamp_str"] == "01:02:03"

    def test_every_claim_is_sourced_to_the_segment(self):
        assert all(c["source"] == "visual_segment" for c in self._claims())

    def test_filler_is_treated_as_absence(self):
        seg = dict(self.SEG)
        seg["instrument"] = "unidentified asset"
        seg["timeframe"] = "unknown"
        got = {c["type"] for c in self._claims(seg)}
        assert "instrument" not in got
        assert "timeframe" not in got

    def test_empty_fields_produce_no_claims(self):
        seg = {"timestamp": "00:05", "instrument": "", "timeframe": "",
               "chart_type": "", "trend": "", "indicators": [], "drawn": [],
               "patterns": [], "levels": []}
        assert visual.claims_from_segments([seg]) == []

    def test_duplicate_claims_at_one_timestamp_collapse(self):
        seg = dict(self.SEG)
        seg["drawn"] = ["trendline", "Trendline", "trendline"]
        drawn = [c for c in self._claims(seg) if c["claim"] == "trendline"]
        assert len(drawn) == 1

    def test_malformed_segments_never_raise(self):
        assert visual.claims_from_segments(None) == []
        assert visual.claims_from_segments(["not a dict", 7]) == []
        assert visual.claims_from_segments([{"timestamp": "x"}]) == []
