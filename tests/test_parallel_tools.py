"""CPU-only regressions for MAX VLM parallel tools and loop guards."""
from __future__ import annotations

import importlib
import sys
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class ParallelToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.read_file = importlib.import_module("tools.read_file")
        self.read_image = importlib.import_module("tools.read_image")
        self.read_image.begin_image_turn()
        self.read_image.reset_read_guards()

    def tearDown(self) -> None:
        self.read_image.begin_image_turn()
        self.read_image.reset_read_guards()

    def test_read_file_batch_runs_concurrently_preserves_order_and_isolates_errors(self) -> None:
        lock = threading.Lock()
        active = 0
        max_active = 0

        def fake_read(path: str, *args) -> str:
            nonlocal active, max_active
            with lock:
                active += 1
                max_active = max(max_active, active)
            try:
                time.sleep(0.04)
                if path == "bad":
                    raise RuntimeError("fixture failure")
                return f"content:{path}"
            finally:
                with lock:
                    active -= 1

        with patch.object(self.read_file, "_read_file_impl", side_effect=fake_read):
            result = self.read_file.read_file.invoke({"path": ["slow-a", "bad", "slow-b"]})

        self.assertGreaterEqual(max_active, 2)
        self.assertLess(result.index("[file:slow-a]"), result.index("[file:bad]"))
        self.assertLess(result.index("[file:bad]"), result.index("[file:slow-b]"))
        self.assertIn("content:slow-a", result)
        self.assertIn("[error] read_file failed: fixture failure", result)
        self.assertIn("content:slow-b", result)

    def test_read_file_requests_apply_independent_filters(self) -> None:
        calls: list[tuple[str, tuple[object, ...]]] = []

        def fake_read(path: str, *args) -> str:
            calls.append((path, args))
            return f"ok:{path}"

        with patch.object(self.read_file, "_read_file_impl", side_effect=fake_read):
            result = self.read_file.read_file.invoke({
                "requests": [
                    {"path": "one", "contains": "alpha"},
                    {"path": "two", "line_start": 2, "line_end": 4},
                ],
            })

        self.assertIn("[file:one]", result)
        self.assertIn("[file:two]", result)
        by_path = dict(calls)
        self.assertEqual("alpha", by_path["one"][5])
        self.assertEqual(2, by_path["two"][1])
        self.assertEqual(4, by_path["two"][2])

    def test_read_image_batch_queues_in_order_and_ocr_is_progressive(self) -> None:
        lock = threading.Lock()
        active = 0
        max_active = 0

        def fake_prepare(src: str, *, need_overview: bool, do_ocr: bool, text_only: bool) -> dict:
            nonlocal active, max_active
            with lock:
                active += 1
                max_active = max(max_active, active)
            try:
                time.sleep(0.04)
                return {
                    "image_url": f"data:image/png;base64,{src}" if need_overview else "",
                    "ocr_result": f"OCR:{src}" if do_ocr else "",
                    "retained_path": "",
                    "screen_tmp": None,
                }
            finally:
                with lock:
                    active -= 1

        with patch.object(self.read_image, "get_capability", return_value="vision"), \
             patch.object(self.read_image, "_prepare_batch_source", side_effect=fake_prepare):
            first = self.read_image.read_image.invoke({
                "source": ["a.png", "b.png", "c.png"],
                "ocr": True,
            })
            self.assertGreaterEqual(max_active, 2)
            self.assertLess(first.index("[image:1 a.png]"), first.index("[image:2 b.png]"))
            self.assertIn("OCR deferred", first)
            self.assertEqual(
                ["data:image/png;base64,a.png", "data:image/png;base64,b.png", "data:image/png;base64,c.png"],
                self.read_image.active_turn_images(),
            )
            second = self.read_image.read_image.invoke({
                "source": ["a.png", "b.png", "c.png"],
                "ocr": True,
            })

        self.assertIn("OCR:a.png", second)
        self.assertIn("OCR:b.png", second)
        self.assertIn("OCR:c.png", second)

    def test_read_image_batch_text_only_uses_parallel_fallback_without_publishing(self) -> None:
        def fake_prepare(src: str, *, need_overview: bool, do_ocr: bool, text_only: bool) -> dict:
            self.assertTrue(text_only)
            self.assertFalse(need_overview)
            self.assertTrue(do_ocr)
            return {"ocr_result": f"[TEXT-ONLY IMAGE FALLBACK — {src}]", "screen_tmp": None}

        with patch.object(self.read_image, "get_capability", return_value=self.read_image.TEXT_ONLY), \
             patch.object(self.read_image, "_prepare_batch_source", side_effect=fake_prepare):
            result = self.read_image.read_image.invoke({"source": ["a.png", "b.png"]})

        self.assertIn("[TEXT-ONLY IMAGE FALLBACK — a.png]", result)
        self.assertIn("[TEXT-ONLY IMAGE FALLBACK — b.png]", result)
        self.assertEqual([], self.read_image._PENDING_IMAGES)
        self.assertEqual([], self.read_image._ACTIVE_TURN_IMAGES)

    def test_repeated_web_query_is_hinted_then_stopped(self) -> None:
        react = importlib.import_module("react")

        def call(call_id: str, query: str) -> dict:
            return {
                "name": "web_search",
                "args": {"query": query, "max_results": 3},
                "id": call_id,
                "type": "tool_call",
            }

        first = AIMessage(content="", tool_calls=[call("call-1", "  Same   Query ")])
        second = AIMessage(content="", tool_calls=[call("call-2", "same query")])
        third = AIMessage(content="", tool_calls=[call("call-3", "SAME QUERY")])
        executed: list[str] = []

        def execute(request):
            executed.append(request.tool_call["id"])
            return "executed"

        first_request = SimpleNamespace(
            tool_call=first.tool_calls[0],
            state={"messages": [HumanMessage(content="find it"), first]},
        )
        second_request = SimpleNamespace(
            tool_call=second.tool_calls[0],
            state={"messages": [HumanMessage(content="find it"), first, second]},
        )
        third_request = SimpleNamespace(
            tool_call=third.tool_calls[0],
            state={"messages": [HumanMessage(content="find it"), first, second, third]},
        )

        self.assertEqual("executed", react._guard_repeated_tool_call(first_request, execute))
        hint = react._guard_repeated_tool_call(second_request, execute)
        self.assertIn("[tool_loop_hint]", hint.content)
        with self.assertRaises(react.ToolLoopDetected):
            react._guard_repeated_tool_call(third_request, execute)
        self.assertEqual(["call-1"], executed)

    def test_turn_runner_converts_loop_stop_to_user_visible_final(self) -> None:
        runtime = importlib.import_module("runtime_common")
        shown: list[str] = []

        class LoopingApp:
            def stream(self, *args, **kwargs):
                raise runtime.ToolLoopDetected("read_file")

        result = runtime.run_turn_core(
            LoopingApp(),
            "read it",
            {},
            thread_id="main",
            saver=object(),
            db_conn=object(),
            reset_web_counter=lambda: None,
            on_final=shown.append,
        )
        self.assertIn("ป้องกัน tool loop", result)
        self.assertEqual([result], shown)


if __name__ == "__main__":
    unittest.main()
