"""The WebSocket callback publishes a complete, redacted activity record per tool call."""
from __future__ import annotations

import json
import unittest
import uuid

import agent_server as srv


class _Queue:
    def __init__(self) -> None:
        self.events: list[dict] = []

    def put_nowait(self, event: dict) -> None:
        self.events.append(event)


class _Loop:
    """call_soon_threadsafe(fn, *args) runs immediately: no event loop needed."""

    def call_soon_threadsafe(self, fn, *args) -> None:
        fn(*args)


class _Log:
    def tool_call(self, *args, **kwargs) -> None:
        pass

    def tool_result(self, *args, **kwargs) -> None:
        pass


def _callback() -> tuple["srv._WSCallback", _Queue]:
    queue = _Queue()
    return srv._WSCallback(queue, _Loop(), _Log(), "turn-1"), queue


def _only(events: list[dict], kind: str) -> list[dict]:
    return [e for e in events if e.get("type") == kind]


class ToolActivityTests(unittest.TestCase):
    def test_tool_start_carries_call_id_and_redacted_arguments_and_keeps_legacy_fields(self) -> None:
        cb, queue = _callback()
        run_id = uuid.uuid4()
        cb.on_tool_start({"name": "bash"}, json.dumps({"command": "deploy API_TOKEN=hunter2", "timeout": 30}),
                         run_id=run_id)
        (event,) = _only(queue.events, "tool")
        self.assertEqual(event["name"], "bash")
        self.assertIn("detail", event)
        self.assertEqual(event["call_id"], str(run_id))
        self.assertIn("deploy", event["arguments"])
        self.assertNotIn("hunter2", event["arguments"])
        self.assertIn("[REDACTED]", event["arguments"])

    def test_tool_end_publishes_a_redacted_result_keyed_by_the_same_call_id(self) -> None:
        cb, queue = _callback()
        run_id = uuid.uuid4()
        cb.on_tool_start({"name": "bash"}, json.dumps({"command": "ls"}), run_id=run_id)
        cb.on_tool_end("file_a\nSECRET_KEY=abc123\nfile_c\nfile_d", run_id=run_id)
        (result,) = _only(queue.events, "tool_result")
        self.assertEqual(result["call_id"], str(run_id))
        self.assertEqual(result["name"], "bash")
        self.assertEqual(result["status"], "returned")
        self.assertNotIn("abc123", result["preview"])
        self.assertEqual(len(result["preview"].splitlines()), 3)
        self.assertTrue(result["truncated"])

    def test_error_results_are_reported_as_failed(self) -> None:
        cb, queue = _callback()
        run_id = uuid.uuid4()
        cb.on_tool_start({"name": "read_file"}, json.dumps({"path": "x.txt"}), run_id=run_id)
        cb.on_tool_end("[error] file not found", run_id=run_id)
        self.assertEqual(_only(queue.events, "tool_result")[0]["status"], "failed")

    def test_a_raised_tool_error_publishes_a_failed_result(self) -> None:
        cb, queue = _callback()
        run_id = uuid.uuid4()
        cb.on_tool_start({"name": "bash"}, json.dumps({"command": "x"}), run_id=run_id)
        cb.on_tool_error(RuntimeError("boom API_TOKEN=zzz"), run_id=run_id)
        (result,) = _only(queue.events, "tool_result")
        self.assertEqual(result["call_id"], str(run_id))
        self.assertEqual(result["status"], "failed")
        self.assertIn("boom", result["preview"])
        self.assertNotIn("zzz", result["preview"])

    def test_results_without_a_known_start_still_publish_with_their_call_id(self) -> None:
        cb, queue = _callback()
        run_id = uuid.uuid4()
        cb.on_tool_end("late result", run_id=run_id)
        (result,) = _only(queue.events, "tool_result")
        self.assertEqual(result["call_id"], str(run_id))
        self.assertEqual(result["preview"], "late result")


if __name__ == "__main__":
    unittest.main()
