"""_test_ocr_engine_unit.py — Settings -> OCR: user_settings.ocr_engine, the tools/_ocr.py dispatcher, the OpenThai
wrapper, the vendored runtime pin and the installer.  LLM-free.  Tests that need the real model and a Thai font are
skipped when either is missing (run scripts/install_openthai_ocr.py first)."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Never read or write the owner's real settings file from a test.
_TMP = tempfile.TemporaryDirectory(prefix="ocr-engine-unit-")
SETTINGS = Path(_TMP.name) / "user_settings.json"
os.environ["V2_USER_SETTINGS_PATH"] = str(SETTINGS)

import user_settings as us  # noqa: E402
from tools import _ocr, _ocr_openthai  # noqa: E402

FONT = Path("/System/Library/Fonts/Supplemental/Ayuthaya.ttf")
REAL = _ocr_openthai.availability()[0]

PINNED_RUNTIME = {
    "line_engine.py": "6afceb5572aae32cb6192358f82ec94a7d0626c893659aa518dbf4a2b473858a",
    "thai_norm.py": "53a03ac139fb0de3e7aec0f5b8ab69b0b519fb5e636f7c1de09cb357b023344a",
    "dewarp.py": "6c278ce935c54ff66302153babe6892e9e2c9af42bc531120fe1e1d9206c98af",
}
AUDITED_WEIGHTS = {
    "det.onnx": "193bab7a04fca699a6c82e6abb5b81bdb28177f0abd4062552b04908dafb19f8",
    "rec.onnx": "82ec89533346faadcb30df96703dad2dd7521de65709e60d83fd9745b8968124",
    "rec_dict.txt": "259ef2636cec0eeae225ae9fd9a63fa9305285650c22d7ec34fe8a82b24c2f1a",
    "config.json": "f22dcf63fe64d88a8f8f066b1168e0e373ec008ea9d67568c29f0b8221fa1d31",
    "rec2.onnx": "6ec0f96f5ade665fc7eb33a9800ed055bc220870cbf963cf97c4750e64cd27ce",
    "det.yml": "3ac018be6f97499a08faa3bbdeb33640968d9307f6736d152902747a9f259593",
}


def _reset_settings(content: str | None = None) -> None:
    if content is None:
        SETTINGS.unlink(missing_ok=True)
    else:
        SETTINGS.write_text(content, encoding="utf-8")


class UserSettingsOcr(unittest.TestCase):
    def setUp(self) -> None:
        _reset_settings()

    def test_default_is_apple_and_only_known_engines_are_accepted(self) -> None:
        self.assertEqual(us.ocr_engine(), "apple")
        for text in ("{not json", "[1]", "", '{"ocr_engine": "tesseract"}', '{"ocr_engine": null}', '{"ocr_engine": ["openthai"]}'):
            _reset_settings(text)
            self.assertEqual(us.ocr_engine(), "apple", text)

    def test_set_and_read_back_and_other_keys_survive(self) -> None:
        _reset_settings(json.dumps({"schema_version": 1, "internet": "offline", "future_key": [1, 2]}))
        us.set_ocr_engine("openthai")
        self.assertEqual(us.ocr_engine(), "openthai")
        data = json.loads(SETTINGS.read_text())
        self.assertEqual(data, {"schema_version": 1, "internet": "offline", "future_key": [1, 2], "ocr_engine": "openthai"})
        us.set_internet(True)
        self.assertEqual(us.ocr_engine(), "openthai", "writing Internet must not drop the OCR choice")
        us.set_ocr_engine("apple")
        self.assertEqual(us.ocr_engine(), "apple")
        self.assertEqual(json.loads(SETTINGS.read_text())["internet"], "online")

    def test_invalid_engine_is_rejected_and_nothing_is_written(self) -> None:
        for bad in ("", "Apple", "tesseract", None, 1, True, ["apple"]):
            with self.assertRaises(ValueError, msg=repr(bad)):
                us.set_ocr_engine(bad)
        self.assertFalse(SETTINGS.exists())


class Dispatcher(unittest.TestCase):
    def setUp(self) -> None:
        _reset_settings()
        _ocr._fallback_warned = False

    def _boxes(self, text: str) -> list[dict]:
        return [{"text": text, "x": 0.1, "y": 0.2, "w": 0.3, "h": 0.05}]

    def test_apple_selected_never_touches_openthai(self) -> None:
        with patch.object(_ocr._openthai, "read_layout", side_effect=AssertionError("must not run")), \
             patch.object(_ocr, "_read_layout_apple", return_value=self._boxes("apple")) as apple:
            self.assertEqual(_ocr.read_layout("a.png")[0]["text"], "apple")
            apple.assert_called_once_with("a.png")

    def test_openthai_selected_uses_it_and_skips_apple(self) -> None:
        us.set_ocr_engine("openthai")
        with patch.object(_ocr._openthai, "read_layout", return_value=self._boxes("thai")) as ot, \
             patch.object(_ocr, "_read_layout_apple", side_effect=AssertionError("must not run")):
            self.assertEqual(_ocr.read_layout("a.png")[0]["text"], "thai")
            ot.assert_called_once_with("a.png")

    def test_an_empty_page_is_an_answer_not_a_failure(self) -> None:
        us.set_ocr_engine("openthai")
        with patch.object(_ocr._openthai, "read_layout", return_value=[]), \
             patch.object(_ocr, "_read_layout_apple", side_effect=AssertionError("blank page must not fall back")):
            self.assertEqual(_ocr.read_layout("blank.png"), [])

    def test_a_call_openthai_cannot_process_falls_back_to_apple_and_logs_once(self) -> None:
        us.set_ocr_engine("openthai")
        with patch.object(_ocr._openthai, "read_layout", return_value=None), \
             patch.object(_ocr._openthai, "availability", return_value=(False, "packages missing")), \
             patch.object(_ocr, "_read_layout_apple", return_value=self._boxes("apple")) as apple:
            with self.assertLogs("tools._ocr", level="WARNING") as logs:
                self.assertEqual(_ocr.read_layout("a.png")[0]["text"], "apple")
            self.assertIn("using Apple Vision", logs.output[0])
            self.assertEqual(_ocr.read_layout("b.png")[0]["text"], "apple")  # second fallback: still works, no second warning
            self.assertEqual(apple.call_count, 2)

    def test_read_text_is_the_plain_view_of_the_selected_engine(self) -> None:
        us.set_ocr_engine("openthai")
        with patch.object(_ocr._openthai, "read_layout", return_value=self._boxes("one") + self._boxes("two")):
            self.assertEqual(_ocr.read_text("a.png"), ["one", "two"])

    def test_empty_path_is_safe(self) -> None:
        us.set_ocr_engine("openthai")
        with patch.object(_ocr._openthai, "read_layout", side_effect=AssertionError("no path, no work")):
            self.assertEqual(_ocr.read_layout(""), [])

    def test_effective_engine_label_and_readiness_follow_availability(self) -> None:
        with patch.object(_ocr, "_ensure_binary", return_value=True):
            self.assertEqual((_ocr._effective_engine(), _ocr.engine_label(), _ocr.ocr_ready()), ("apple", "Apple Vision", True))
        with patch.object(_ocr, "_ensure_binary", return_value=False):
            self.assertFalse(_ocr.ocr_ready(), "Apple selected and its helper missing: not ready")
        us.set_ocr_engine("openthai")
        with patch.object(_ocr._openthai, "availability", return_value=(True, "")), \
             patch.object(_ocr, "_ensure_binary", side_effect=AssertionError("Apple helper is irrelevant while OpenThai is ready")):
            self.assertEqual((_ocr._effective_engine(), _ocr.engine_label(), _ocr.ocr_ready()), ("openthai", "OpenThai", True))
        with patch.object(_ocr._openthai, "availability", return_value=(False, "x")), patch.object(_ocr, "_ensure_binary", return_value=True):
            self.assertEqual((_ocr._effective_engine(), _ocr.engine_label(), _ocr.ocr_ready()), ("apple", "Apple Vision", True),
                             "selected but unusable: everything reports the engine really in use")

    def test_engine_status_shape(self) -> None:
        us.set_ocr_engine("openthai")
        with patch.object(_ocr._openthai, "availability", return_value=(False, "model files missing")):
            status = _ocr.engine_status()
        self.assertEqual(status["selected"], "openthai")
        self.assertEqual(status["effective"], "apple")
        self.assertEqual(set(status["engines"]), {"apple", "openthai"})
        self.assertEqual(status["engines"]["openthai"], {"available": False, "detail": "model files missing"})
        self.assertIn("available", status["engines"]["apple"])

    def test_unreadable_settings_module_means_apple(self) -> None:
        with patch.dict(sys.modules, {"user_settings": None}):  # ImportError on import
            self.assertEqual(_ocr._selected_engine(), "apple")

    def test_apple_internals_the_old_tests_rely_on_still_exist(self) -> None:
        for name in ("_ensure_binary", "_BIN", "_SRC", "read_text", "read_layout", "subprocess"):
            self.assertTrue(hasattr(_ocr, name), name)


class HelperBinaryIdentity(unittest.TestCase):
    """Regression (2026-11-15): the Swift helper was compiled to ONE fixed path.  After its source dropped --fast, the
    still-running older agent called the new binary with --fast, got 'ERR: cannot load --fast' and silently read
    nothing.  Code versions must never share, and so overwrite, one binary: its name carries a hash of its source."""

    def test_binary_name_follows_the_swift_source(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            a, b, a2 = (Path(d) / n for n in ("a.swift", "b.swift", "a2.swift"))
            a.write_text("let x = 1\n"); b.write_text("let x = 2\n"); a2.write_text("let x = 1\n")
            self.assertEqual(_ocr._binary_path(a).name, _ocr._binary_path(a2).name, "same source, same binary")
            self.assertNotEqual(_ocr._binary_path(a).name, _ocr._binary_path(b).name, "changed source, different binary")
            self.assertNotEqual(_ocr._binary_path(Path(d) / "missing.swift"), _ocr._binary_path(a))

    def test_production_binary_is_versioned_not_the_legacy_shared_name(self) -> None:
        self.assertEqual(_ocr._BIN, _ocr._binary_path())
        self.assertRegex(_ocr._BIN.name, r"^endeavor_v2_vision_ocr-[0-9a-f]{10}$")
        self.assertNotEqual(_ocr._BIN.name, "endeavor_v2_vision_ocr", "the unversioned name is what older running agents share")


class BoxConversion(unittest.TestCase):
    def test_pixel_polygon_becomes_a_normalized_bottom_left_box(self) -> None:
        # a 1000x500 image; a line whose top-left is (100,50) and bottom-right (400,150), y down in pixels
        line = {"text": " hello ", "box": [[100, 50], [400, 50], [400, 150], [100, 150]]}
        box = _ocr_openthai._boxes_from_lines([line], 1000, 500)[0]
        self.assertEqual(box["text"], "hello")
        self.assertAlmostEqual(box["x"], 0.1)
        self.assertAlmostEqual(box["w"], 0.3)
        self.assertAlmostEqual(box["h"], 0.2)
        self.assertAlmostEqual(box["y"], 1.0 - 150 / 500, msg="y is the BOTTOM edge measured upward, like Vision")

    def test_higher_lines_get_larger_y(self) -> None:
        top = {"text": "top", "box": [[0, 10], [50, 10], [50, 40], [0, 40]]}
        low = {"text": "low", "box": [[0, 400], [50, 400], [50, 430], [0, 430]]}
        boxes = _ocr_openthai._boxes_from_lines([low, top], 100, 500)
        self.assertGreater(boxes[1]["y"], boxes[0]["y"])

    def test_clamped_and_degenerate_lines_are_handled(self) -> None:
        over = {"text": "edge", "box": [[-5, -5], [120, -5], [120, 60], [-5, 60]]}
        box = _ocr_openthai._boxes_from_lines([over], 100, 50)[0]
        self.assertTrue(0 <= box["x"] and box["x"] + box["w"] <= 1.0 and 0 <= box["y"] and box["y"] + box["h"] <= 1.0, box)
        junk = [{"text": "", "box": [[0, 0], [1, 0], [1, 1], [0, 1]]}, {"text": "no box"}, {"text": "flat", "box": [[5, 5], [9, 5], [9, 5], [5, 5]]}]
        self.assertEqual(_ocr_openthai._boxes_from_lines(junk, 10, 10), [])


class Availability(unittest.TestCase):
    def test_missing_model_files_say_how_to_install(self) -> None:
        with tempfile.TemporaryDirectory() as empty, patch.object(_ocr_openthai, "model_dir", return_value=empty):
            ok, reason = _ocr_openthai.availability()
        self.assertFalse(ok)
        self.assertIn("det.onnx", reason)
        self.assertIn("install_openthai_ocr.py", reason)

    def test_missing_packages_are_named(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            for name in ("det.onnx", "rec.onnx", "rec_dict.txt"):
                (Path(d) / name).write_bytes(b"x")
            real = importlib.util.find_spec
            with patch.object(_ocr_openthai, "model_dir", return_value=d), \
                 patch("importlib.util.find_spec", side_effect=lambda n, *a: None if n == "pyclipper" else real(n, *a)):
                ok, reason = _ocr_openthai.availability()
        self.assertFalse(ok)
        self.assertIn("pyclipper", reason)

    def test_unreadable_or_missing_image_returns_none_not_an_exception(self) -> None:
        _ocr_openthai.reset_for_tests()
        with tempfile.TemporaryDirectory() as empty, patch.object(_ocr_openthai, "model_dir", return_value=empty):
            self.assertIsNone(_ocr_openthai.read_layout("/definitely/not/here.png"))

    def test_loading_the_engine_puts_opencv_thread_count_back(self) -> None:
        """LineOCR pins OpenCV's pool to its own budget for the whole process; the wrapper must undo that.  cv2's real
        getNumThreads() does not read back reliably on every build (macOS GCD), so a plain counter stands in for it."""
        import cv2

        state = {"n": 7}

        class FakeLineOCR:
            def __init__(self, directory, threads=1):
                cv2.setNumThreads(threads)  # what the real LineOCR.__init__ does

        _ocr_openthai.reset_for_tests()
        with patch.object(cv2, "getNumThreads", side_effect=lambda: state["n"]), \
             patch.object(cv2, "setNumThreads", side_effect=lambda n: state.update(n=n)), \
             patch("tools._openthai_ocr.line_engine.LineOCR", FakeLineOCR), \
             patch.object(_ocr_openthai, "availability", return_value=(True, "")):
            with _ocr_openthai._lock:
                engine = _ocr_openthai._get_engine()
        self.assertIsInstance(engine, FakeLineOCR)
        self.assertEqual(state["n"], 7, "the process-wide OpenCV thread count must be back to what it was")
        _ocr_openthai.reset_for_tests()

    def test_a_model_that_fails_to_load_is_not_retried_on_every_call(self) -> None:
        _ocr_openthai.reset_for_tests()
        with tempfile.TemporaryDirectory() as d:
            for name in ("det.onnx", "rec.onnx", "rec_dict.txt"):
                (Path(d) / name).write_bytes(b"not a model")
            with patch.object(_ocr_openthai, "model_dir", return_value=d):
                self.assertIsNone(_ocr_openthai.read_layout(__file__))  # corrupt weights -> load error remembered
                self.assertFalse(_ocr_openthai.availability()[0])
                self.assertIn("failed to load", _ocr_openthai.availability()[1])
                with patch("tools._openthai_ocr.line_engine.LineOCR", side_effect=AssertionError("no second load attempt")):
                    self.assertIsNone(_ocr_openthai.read_layout(__file__))
        _ocr_openthai.reset_for_tests()


class VendoredRuntimeAndInstaller(unittest.TestCase):
    def test_vendored_runtime_is_byte_identical_to_the_audited_upstream(self) -> None:
        for name, want in PINNED_RUNTIME.items():
            got = hashlib.sha256((ROOT / "tools" / "_openthai_ocr" / name).read_bytes()).hexdigest()
            self.assertEqual(got, want, f"{name} changed: re-vendor from upstream, audit it, and update the pin")
        self.assertTrue((ROOT / "tools" / "_openthai_ocr" / "NOTICE").is_file())

    def _installer(self):
        spec = importlib.util.spec_from_file_location("install_openthai_ocr", ROOT / "scripts" / "install_openthai_ocr.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_installer_pins_the_audited_weights(self) -> None:
        self.assertEqual(self._installer().FILES, AUDITED_WEIGHTS)

    def test_installer_verifies_before_placing_and_leaves_nothing_behind_on_mismatch(self) -> None:
        installer = self._installer()
        good = b"good bytes"
        evil = b"tampered bytes"

        class Resp:
            def __init__(self, body): self.body = body
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def read(self, n=-1):
                chunk, self.body = self.body[:n], self.body[n:]
                return chunk

        with tempfile.TemporaryDirectory() as target, patch.object(installer, "FILES", {"w.bin": hashlib.sha256(good).hexdigest()}):
            with patch.object(installer.urllib.request, "urlopen", side_effect=lambda *a, **k: Resp(evil)):
                with self.assertRaises(RuntimeError) as ctx:
                    installer.install(target)
            self.assertIn("mismatch", str(ctx.exception))
            self.assertEqual(os.listdir(target), [], "a rejected download must leave no file and no temp file")
            with patch.object(installer.urllib.request, "urlopen", side_effect=lambda *a, **k: Resp(good)):
                self.assertEqual(installer.install(target), ["w.bin"])
            self.assertEqual((Path(target) / "w.bin").read_bytes(), good)
            with patch.object(installer.urllib.request, "urlopen", side_effect=AssertionError("already valid: no download")):
                self.assertEqual(installer.install(target), [])
            (Path(target) / "w.bin").write_bytes(evil)  # a file edited on disk is replaced, not trusted
            with patch.object(installer.urllib.request, "urlopen", side_effect=lambda *a, **k: Resp(good)):
                self.assertEqual(installer.install(target), ["w.bin"])


@unittest.skipUnless(REAL and FONT.is_file(), "OpenThai weights/packages or the Thai font are not available on this machine")
class RealEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from PIL import Image, ImageDraw, ImageFont

        cls.tmp = tempfile.TemporaryDirectory(prefix="ocr-real-")
        cls.image = str(Path(cls.tmp.name) / "page.png")
        font = ImageFont.truetype(str(FONT), 40)
        page = Image.new("RGB", (1200, 420), "white")
        draw = ImageDraw.Draw(page)
        cls.lines = ["ประกาศคณะกรรมการควบคุมการขนส่ง", "ฉบับที่ ๑๒๗ (พ.ศ. ๒๕๐๙)", "เรื่อง ยกเลิกและกำหนดเส้นทาง"]
        for i, text in enumerate(cls.lines):
            draw.text((60, 40 + i * 110), text, font=font, fill="black")
        page.save(cls.image)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def setUp(self) -> None:
        _reset_settings()
        us.set_ocr_engine("openthai")

    def test_reads_thai_with_vision_style_geometry_through_the_dispatcher(self) -> None:
        boxes = _ocr.read_layout(self.image)
        self.assertEqual(len(boxes), 3, boxes)
        self.assertEqual([b["text"] for b in boxes], self.lines)
        for box in boxes:
            self.assertTrue(0 <= box["x"] < 1 and 0 < box["y"] <= 1 and box["w"] > 0 and box["h"] > 0, box)
        ys = [b["y"] for b in boxes]
        self.assertEqual(ys, sorted(ys, reverse=True), "reading order is top to bottom, so y (measured upward) falls")
        self.assertEqual(_ocr.engine_label(), "OpenThai")

    def test_a_blank_page_returns_an_empty_list_not_a_fallback(self) -> None:
        from PIL import Image

        blank = str(Path(self.tmp.name) / "blank.png")
        Image.new("RGB", (800, 400), "white").save(blank)
        with patch.object(_ocr, "_read_layout_apple", side_effect=AssertionError("a blank page is an answer")):
            self.assertEqual(_ocr.read_layout(blank), [])

    def test_concurrent_callers_get_identical_results(self) -> None:
        results: list = []
        errors: list = []

        def worker() -> None:
            try:
                results.append(json.dumps(_ocr.read_layout(self.image), ensure_ascii=False))
            except Exception as exc:  # noqa: BLE001 — the test wants every failure surfaced
                errors.append(repr(exc))

        _ocr_openthai.reset_for_tests()  # six callers race to the first (lazy) load
        threads = [threading.Thread(target=worker) for _ in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=120)
        self.assertEqual(errors, [])
        self.assertEqual(len(results), 6)
        self.assertEqual(len(set(results)), 1, "six threads on one engine must agree")

    def test_a_file_opencv_cannot_decode_falls_back_to_apple_for_that_call(self) -> None:
        bad = str(Path(self.tmp.name) / "broken.png")
        Path(bad).write_bytes(b"not an image")
        with patch.object(_ocr, "_read_layout_apple", return_value=[{"text": "apple", "x": 0, "y": 0, "w": 1, "h": 1}]) as apple, \
             self.assertLogs("tools", level="WARNING"):
            _ocr._fallback_warned = False
            self.assertEqual(_ocr.read_layout(bad)[0]["text"], "apple")
        apple.assert_called_once()


if __name__ == "__main__":
    unittest.main(verbosity=2)
