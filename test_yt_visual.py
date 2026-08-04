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
