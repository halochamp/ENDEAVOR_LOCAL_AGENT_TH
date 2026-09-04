"""CPU-only regressions for text-only image compatibility.

Run from the target repository with:
    /opt/homebrew/anaconda3/envs/mlx/bin/python tests/test_text_only_vision.py

All backend/model calls and computer paths are mocked.  The tests do not contact
an MLX server or mutate the desktop.
"""
from __future__ import annotations

import importlib
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessageChunk, HumanMessage
from langchain_core.outputs import ChatGenerationChunk
from langchain_openai import ChatOpenAI
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class UnsupportedImageError(RuntimeError):
    status_code = 400
    body = {"error": {"message": "model does not support image input"}}


class VisionCompatibilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ri = importlib.import_module("tools.read_image")
        self.computer = importlib.import_module("tools.computer_use")
        self.capability = importlib.import_module("tools._vision_capability")
        self.llm = importlib.import_module("llm")
        self.capability.reset_for_tests()
        self.ri.begin_image_turn()
        self.ri.reset_read_guards()
        self.computer.reset_computer_guards()

    def tearDown(self) -> None:
        self.ri.begin_image_turn()
        self.ri.reset_read_guards()
        self.computer.end_computer_turn()
        self.capability.reset_for_tests()

    def _image(self) -> str:
        handle = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        handle.close()
        path = Path(handle.name)
        Image.new("RGB", (64, 48), (30, 50, 70)).save(path)
        self.addCleanup(lambda: path.unlink(missing_ok=True))
        return str(path)

    def _read_patches(self, source: str, ocr, *, encode=None):
        return patch.multiple(
            self.ri,
            resolve_read_path=lambda _source: source,
            _maybe_convert_heic=lambda path: path,
            _exif_normalize=lambda path: path,
            _encode_frame_data_url=encode or (lambda *_args, **_kwargs: "data:image/png;base64,ORIGINAL"),
            _ocr_layout=ocr,
            _enhance_for_ocr=lambda path: path,
            _reconstruct_table=lambda _boxes: None,
            _reconstruct_ranked_columns=lambda _boxes: None,
            _decode_qr=lambda _path: [],
            _looks_like_range_chart=lambda _boxes: False,
            _zoom_region_hint=lambda _boxes: "",
            phase=lambda *_args, **_kwargs: None,
            progress=lambda *_args, **_kwargs: None,
        )

    def _model(self, **kwargs):
        streaming = kwargs.pop("streaming", False)
        return self.llm.VisionFallbackChatOpenAI(
            api_key="x",
            base_url="http://test.local/v1",
            model="test-model",
            streaming=streaming,
            **kwargs,
        )

    def test_vision_read_is_direct_first_and_success_marks_vision(self) -> None:
        source = self._image()
        ocr_calls: list[str] = []

        def fake_ocr(path: str) -> list[dict]:
            ocr_calls.append(path)
            return [{"text": "never called on overview", "x": 0.1, "y": 0.1, "w": 0.2, "h": 0.1}]

        with self._read_patches(source, fake_ocr):
            result = self.ri.read_image.invoke({"source": source})
            self.assertIn("original image queued", result)
            self.assertEqual([], ocr_calls)
            self.assertEqual([], self.ri._ACTIVE_TURN_IMAGES)
            self.assertEqual(1, len(self.ri._PENDING_IMAGES))

            published = self.ri.active_turn_images()
            self.assertEqual(["data:image/png;base64,ORIGINAL"], published)
            self.assertEqual([], ocr_calls)
            self.assertEqual(1, len(self.ri.active_turn_image_sources()))

            calls = []

            def fake_generate(model, messages, stop=None, run_manager=None, **kwargs):
                calls.append(list(messages))
                return "VISION_RESPONSE"

            model = self._model()
            with patch.object(ChatOpenAI, "_generate", fake_generate):
                self.assertEqual(
                    "VISION_RESPONSE",
                    model._generate([
                        HumanMessage(content=[
                            {"type": "text", "text": "What is visible?"},
                            {"type": "image_url", "image_url": {"url": published[0]}},
                        ])
                    ]),
                )
            self.assertEqual(1, len(calls))
            self.assertEqual([], ocr_calls)
            self.assertEqual(
                self.capability.VISION,
                self.capability.get_capability("http://test.local/v1", "test-model"),
            )

    def test_react_boundary_publishes_original_without_sensor_work(self) -> None:
        source = self._image()
        ocr_calls: list[str] = []

        with self._read_patches(
            source,
            lambda path: ocr_calls.append(path) or [{"text": "not eager", "x": 0, "y": 0, "w": 1, "h": 1}],
        ):
            self.ri.read_image.invoke({"source": source})
            from react import prepare_turn_vision_messages

            messages = prepare_turn_vision_messages([HumanMessage(content="What is shown?")])

        self.assertEqual([], ocr_calls)
        self.assertEqual(1, len(messages[0].content) - 1)
        self.assertEqual("What is shown?", messages[0].content[0]["text"])
        self.assertEqual("image_url", messages[0].content[1]["type"])
        self.assertEqual(1, len(self.ri.active_turn_image_sources()))

    def test_classifier_only_accepts_explicit_image_unsupported_errors(self) -> None:
        class StructuredError(RuntimeError):
            status_code = 422
            body = {"error": {"code": "image_not_supported"}}

        self.assertTrue(self.capability.is_unsupported_image_error(StructuredError()))
        self.assertTrue(self.capability.is_unsupported_image_error(
            RuntimeError("HTTP 415: unsupported media type for image input")
        ))
        self.assertFalse(self.capability.is_unsupported_image_error(
            TimeoutError("request timed out while sending image input")
        ))
        self.assertFalse(self.capability.is_unsupported_image_error(
            RuntimeError("HTTP 500: model does not support image input")
        ))
        self.assertFalse(self.capability.is_unsupported_image_error(
            RuntimeError("HTTP 401: model does not support image input")
        ))
        self.assertFalse(self.capability.is_unsupported_image_error(
            RuntimeError("context length exceeded while sending image input")
        ))
        self.assertFalse(self.capability.is_unsupported_image_error(
            RuntimeError("HTTP 400: malformed tool arguments in function call")
        ))

    def test_same_model_call_recovers_with_ocr_without_tool_replay(self) -> None:
        source = self._image()
        ocr_calls: list[str] = []

        def fake_ocr(path: str) -> list[dict]:
            ocr_calls.append(path)
            return [{"text": "VISION 742", "x": 0.1, "y": 0.1, "w": 0.4, "h": 0.1}]

        with self._read_patches(source, fake_ocr):
            self.ri.read_image.invoke({"source": source})
            published = self.ri.active_turn_images()
            model = self._model()
            calls = []

            def reject_then_succeed(instance, messages, stop=None, run_manager=None, **kwargs):
                calls.append(list(messages))
                if len(calls) == 1:
                    raise UnsupportedImageError()
                return "RECOVERED_RESPONSE"

            with patch.object(ChatOpenAI, "_generate", reject_then_succeed):
                result = model._generate([
                    HumanMessage(content=[
                        {"type": "text", "text": "Read the marker from this image."},
                        {"type": "image_url", "image_url": {"url": published[0]}},
                    ])
                ])

            self.assertEqual("RECOVERED_RESPONSE", result)
            self.assertEqual(2, len(calls), "only the model call may be retried")
            self.assertEqual(1, len(ocr_calls))
            first_content = calls[0][0].content
            retry_content = calls[1][0].content
            self.assertTrue(any(block.get("type") == "image_url" for block in first_content))
            self.assertFalse(any(block.get("type") == "image_url" for block in retry_content))
            retry_text = "\n".join(block.get("text", "") for block in retry_content if block.get("type") == "text")
            self.assertIn("Read the marker from this image", retry_text)
            self.assertIn("[TEXT-ONLY IMAGE FALLBACK]", retry_text)
            self.assertIn("VISION 742", retry_text)
            self.assertEqual(self.capability.TEXT_ONLY, self.capability.get_capability(
                "http://test.local/v1", "test-model"
            ))
            self.assertEqual([], self.ri._PENDING_IMAGES)
            self.assertEqual([], self.ri._ACTIVE_TURN_IMAGES)
            self.assertEqual([], self.ri.active_turn_image_sources())

    def test_streaming_model_call_recovers_before_any_chunk_is_emitted(self) -> None:
        source = self._image()

        with self._read_patches(
            source,
            lambda _path: [{"text": "STREAM 742", "x": 0.1, "y": 0.1, "w": 0.4, "h": 0.1}],
        ):
            self.ri.read_image.invoke({"source": source})
            published = self.ri.active_turn_images()
            model = self._model(streaming=True)
            calls = []

            def reject_then_stream(instance, messages, stop=None, **kwargs):
                calls.append(list(messages))
                if len(calls) == 1:
                    raise UnsupportedImageError()
                yield ChatGenerationChunk(message=AIMessageChunk(content="RECOVERED"))

            with patch.object(ChatOpenAI, "_stream", reject_then_stream):
                chunks = list(model.stream([
                    HumanMessage(content=[
                        {"type": "text", "text": "Read this stream."},
                        {"type": "image_url", "image_url": {"url": published[0]}},
                    ])
                ]))

        self.assertEqual("RECOVERED", chunks[0].content)
        self.assertEqual(2, len(calls))
        self.assertFalse(any(block.get("type") == "image_url" for block in calls[1][0].content))
        self.assertEqual(self.capability.TEXT_ONLY, self.capability.get_capability(
            "http://test.local/v1", "test-model"
        ))

    def test_known_text_only_read_ocr_and_empty_result_are_explicit(self) -> None:
        source = self._image()
        ocr_calls: list[str] = []

        def fake_ocr(path: str) -> list[dict]:
            ocr_calls.append(path)
            return [{"text": "FULL OCR LINE", "x": 0.1, "y": 0.1, "w": 0.4, "h": 0.1}]

        self.capability.mark_text_only()
        with self._read_patches(source, fake_ocr, encode=lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("encoded"))):
            result = self.ri.read_image.invoke({"source": source})
        self.assertIn("TEXT-ONLY IMAGE FALLBACK", result)
        self.assertIn("FULL OCR LINE", result)
        self.assertEqual(1, len(ocr_calls))
        self.assertEqual([], self.ri._PENDING_IMAGES)

        self.ri.begin_image_turn()
        self.ri.reset_read_guards()
        self.capability.mark_text_only()
        with self._read_patches(source, lambda _path: [], encode=lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("encoded"))):
            empty = self.ri.read_image.invoke({"source": source})
        self.assertIn("cannot understand image pixels", empty)
        self.assertIn("no readable text", empty)
        self.assertNotIn("VISUAL", empty.upper())
        self.assertEqual([], self.ri._PENDING_IMAGES)

    def test_text_only_screen_ocr_pins_one_snapshot_for_the_turn(self) -> None:
        source = self._image()
        captured_paths: list[str] = []

        def fake_capture(command, **_kwargs):
            destination = command[-1]
            Path(destination).write_bytes(Path(source).read_bytes())
            captured_paths.append(destination)
            return SimpleNamespace(returncode=0)

        self.capability.mark_text_only()
        with patch.object(self.ri.subprocess, "run", side_effect=fake_capture), \
             patch.object(self.ri, "_maybe_convert_heic", side_effect=lambda path: path), \
             patch.object(self.ri, "_exif_normalize", side_effect=lambda path: path), \
             patch.object(self.ri, "full_ocr_for_text_only", side_effect=lambda path, _source: f"OCR:{path}") as ocr:
            self.ri.read_image.invoke({"source": "screen"})
            pinned = self.ri._SCREEN_TURN_SNAPSHOT[0]
            self.assertIsNotNone(pinned)
            self.assertTrue(Path(pinned).exists())
            self.ri.read_image.invoke({"source": "screen", "detail": "text"})

        self.assertEqual(1, len(captured_paths))
        self.assertEqual(2, ocr.call_count)
        self.assertEqual(pinned, ocr.call_args_list[1].args[0])

    def test_multi_image_fallback_is_labeled_in_source_order(self) -> None:
        first, second = self._image(), self._image()
        with patch.multiple(
            self.ri,
            resolve_read_path=lambda source: source,
            _maybe_convert_heic=lambda path: path,
            _exif_normalize=lambda path: path,
            _encode_frame_data_url=lambda *_args, **_kwargs: "data:image/png;base64,ORIGINAL",
            _ocr_layout=lambda _path: [],
            phase=lambda *_args, **_kwargs: None,
            progress=lambda *_args, **_kwargs: None,
        ):
            self.ri.read_image.invoke({"source": first})
            self.ri.read_image.invoke({"source": second})
            self.ri.active_turn_images()

        with patch.object(self.ri, "full_ocr_for_text_only", side_effect=lambda path, source: f"OCR-{Path(source).name}"):
            text = self.llm._text_only_fallback_text()
        self.assertLess(text.index(f"source: {first}"), text.index(f"source: {second}"))
        self.assertIn(f"[IMAGE 1 — source: {first}]", text)
        self.assertIn(f"[IMAGE 2 — source: {second}]", text)

    def test_computer_text_only_never_enters_impl_or_desktop_backend(self) -> None:
        self.capability.mark_text_only()
        impl = MagicMock(side_effect=AssertionError("computer impl entered"))
        capture = MagicMock(side_effect=AssertionError("desktop capture entered"))
        with patch.object(self.computer, "_computer_impl", impl), patch.object(self.computer._backend, "capture", capture):
            result = self.computer.computer.invoke({"action": "see"})
        self.assertIn("[unsupported] computer requires a vision-capable model", result)
        impl.assert_not_called()
        capture.assert_not_called()

    def test_computer_unknown_probe_unsupported_is_pre_action_and_cached(self) -> None:
        def unsupported_probe():
            self.capability.mark_text_only()
            return self.capability.TEXT_ONLY

        impl = MagicMock(side_effect=AssertionError("computer impl entered"))
        with patch.object(self.computer, "probe_vision_capability", side_effect=unsupported_probe) as probe, \
             patch.object(self.computer, "_computer_impl", impl):
            result = self.computer.computer.invoke({"action": "click", "target": "Delete"})
            again = self.computer.computer.invoke({"action": "see"})
        self.assertIn("[unsupported]", result)
        self.assertIn("[unsupported]", again)
        probe.assert_called_once()
        impl.assert_not_called()

    def test_computer_unknown_probe_vision_runs_once_then_normal_path(self) -> None:
        def vision_probe():
            self.capability.mark_vision()
            return self.capability.VISION

        impl = MagicMock(return_value="computer ok")
        with patch.object(self.computer, "probe_vision_capability", side_effect=vision_probe) as probe, \
             patch.object(self.computer, "_computer_impl", impl):
            self.assertTrue(self.computer.computer.invoke({"action": "see"}).startswith("computer ok"))
            self.assertTrue(self.computer.computer.invoke({"action": "see"}).startswith("computer ok"))
        probe.assert_called_once()
        self.assertEqual(2, impl.call_count)

    def test_computer_inconclusive_probe_fails_closed_without_cache_poisoning(self) -> None:
        impl = MagicMock(side_effect=AssertionError("computer impl entered"))
        with patch.object(self.computer, "probe_vision_capability", return_value=self.capability.UNKNOWN), \
             patch.object(self.computer, "_computer_impl", impl):
            result = self.computer.computer.invoke({"action": "open_app", "target": "TextEdit"})
        self.assertIn("inconclusive", result)
        self.assertIn("no desktop action", result)
        impl.assert_not_called()
        self.assertEqual(self.capability.UNKNOWN, self.capability.get_capability())

    def test_probe_is_tiny_non_mutating_and_uses_no_fallback(self) -> None:
        captured = {}

        class FakeClient:
            def invoke(self, message):
                captured["message"] = message
                return "OK"

        with patch.object(self.llm, "build_llm", return_value=FakeClient()) as build:
            state = self.capability.probe_vision_capability("http://probe.local/v1", "probe-model")
        self.assertEqual(self.capability.VISION, state)
        kwargs = build.call_args.kwargs
        self.assertFalse(kwargs["vision_fallback"])
        self.assertEqual(1, kwargs["max_tokens"])
        self.assertEqual(5.0, kwargs["timeout"])
        content = captured["message"].content
        self.assertEqual("image_url", content[1]["type"])
        self.assertLessEqual(len(content[1]["image_url"]["url"]), 200)

    def test_capability_cache_is_keyed_by_endpoint_and_model(self) -> None:
        self.capability.mark_text_only("http://one/v1", "model-a")
        self.capability.mark_vision("http://one/v1", "model-b")
        self.assertEqual(self.capability.TEXT_ONLY, self.capability.get_capability("http://one/v1", "model-a"))
        self.assertEqual(self.capability.VISION, self.capability.get_capability("http://one/v1", "model-b"))
        self.assertEqual(self.capability.UNKNOWN, self.capability.get_capability("http://two/v1", "model-a"))


if __name__ == "__main__":
    unittest.main()
