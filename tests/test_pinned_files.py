from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("AGENT_SERVER_TOKEN", "test-pin-token")

import agent_server as srv
import graph
from langchain_core.messages import HumanMessage, ToolMessage
from tools.read_file import read_file as read_file_tool
from tools.read_image import read_image as read_image_tool


class PinnedFilesTests(unittest.TestCase):
    def test_backend_uses_read_file_policy_dedupes_and_reports_missing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            # TemporaryDirectory is outside the configured Agent Workspace, so
            # accepting this file proves Pin no longer imposes a Workspace-only
            # boundary when read_file policy allows the path.
            good = Path(td) / "good.txt"
            good.write_text("hello", encoding="utf-8")
            real = os.path.realpath(good)
            valid, failures = srv._pinned_file_paths([real, real])
            self.assertEqual(valid, [real])
            self.assertEqual(failures, [])

            missing = os.path.realpath(Path(td) / "missing.txt")
            valid, failures = srv._pinned_file_paths([missing, missing])
            self.assertEqual(valid, [])
            self.assertEqual(len(failures), 1)
            self.assertIn("ไม่พบไฟล์", failures[0]["reason"])

            alias = Path(td) / "alias.txt"
            alias.symlink_to(good)
            valid, failures = srv._pinned_file_paths([str(alias), real])
            self.assertEqual(valid, [real])
            self.assertEqual(failures, [])

            with patch.object(
                srv, "_resolve_read_path",
                side_effect=PermissionError("[BLOCKED] protected path: /etc/"),
            ):
                valid, failures = srv._pinned_file_paths(["/etc/passwd"])
            self.assertEqual(valid, [])
            self.assertEqual(len(failures), 1)
            self.assertIn("[BLOCKED] protected path", failures[0]["reason"])

            with self.assertRaises(ValueError):
                srv._pinned_file_paths([real] * 11)

    def test_pin_identity_suppresses_attachment_and_workspace_mention_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as td, patch.object(srv, "WORKSPACE", td):
            same = Path(td) / "same.txt"
            same.write_text("same source", encoding="utf-8")
            real = os.path.realpath(same)
            pin_identity = {real}
            self.assertEqual(
                srv._workspace_mention_paths(["same.txt"], skip_real_paths=pin_identity),
                [],
            )
            attached = srv._canonical_attachment_paths([real])
            self.assertEqual([path for path in attached if path not in pin_identity], [])

    def test_pinned_docs_force_4_4_2_batches_with_user_query(self) -> None:
        paths = [f"/tmp/pin-{i}.txt" for i in range(10)]
        calls = []

        def fake(args):
            calls.append(args)
            return "\n\n".join(
                f"[file:{req['path']}]\ncontent {idx}"
                for idx, req in enumerate(args["requests"], 1)
            )

        with patch.object(
            type(read_file_tool), "invoke", autospec=True,
            side_effect=lambda _self, args, config=None: fake(args),
        ):
            out, seeded, status = graph._seed_pinned_files(
                [HumanMessage(content="What was revenue in Q3?")],
                "What was revenue in Q3?",
                tool_config={"configurable": {
                    "pinned_files": paths,
                    "pinned_user_query": "What was revenue in Q3?",
                }},
            )

        self.assertEqual([len(call["requests"]) for call in calls], [4, 4, 2])
        self.assertTrue(all(
            req.get("user_query") == "What was revenue in Q3?"
            for call in calls for req in call["requests"]
        ))
        self.assertEqual(status["success"], paths)
        self.assertFalse(status["failures"])
        self.assertEqual(len(seeded), 6)
        self.assertTrue(all(isinstance(m, ToolMessage) or m.type == "ai" for m in seeded))
        self.assertGreater(len(out), 1)

    def test_q3_filter_no_match_falls_back_to_query_aware_read(self) -> None:
        path = "/tmp/report.txt"
        calls = []

        def fake(args):
            calls.append(args)
            req = args["requests"][0]
            if req.get("contains_any"):
                return f"[file:{path}]\n[error] no matches for any of ['Q3'] in {path}"
            return f"[file:{path}]\nThird quarter revenue was 100 million."

        with patch.object(
            type(read_file_tool), "invoke", autospec=True,
            side_effect=lambda _self, args, config=None: fake(args),
        ):
            _out, _seeded, status = graph._seed_pinned_files(
                [HumanMessage(content="What was revenue in Q3?")],
                "What was revenue in Q3?",
                tool_config={"configurable": {"pinned_files": [path]}},
            )
        self.assertEqual(len(calls), 2)
        self.assertNotIn("contains_any", calls[1]["requests"][0])
        self.assertEqual(status["success"], [path])
        self.assertFalse(status["failures"])

    def test_pinned_images_use_one_overview_batch(self) -> None:
        paths = ["/tmp/a.png", "/tmp/b.jpg"]
        calls = []

        def fake(args):
            calls.append(args)
            sources = args["source"] if isinstance(args["source"], list) else [args["source"]]
            return "\n\n".join(
                f"[image:{idx} {src}]\n[original image queued for agent direct vision]"
                for idx, src in enumerate(sources, 1)
            )

        with patch.object(
            type(read_image_tool), "invoke", autospec=True,
            side_effect=lambda _self, args, config=None: fake(args),
        ):
            _out, _seeded, status = graph._seed_pinned_files(
                [HumanMessage(content="เปรียบเทียบภาพ")],
                "เปรียบเทียบภาพ",
                tool_config={"configurable": {"pinned_files": paths}},
            )
        self.assertEqual(calls, [{"source": paths, "detail": "overview"}])
        self.assertEqual(status["success"], paths)
        self.assertFalse(status["failures"])

    def test_pin_evidence_is_invoke_only_not_persisted(self) -> None:
        seen = {}
        pin_ai = graph.AIMessage(content="", tool_calls=[{
            "name": "read_file", "args": {"path": "/tmp/p.txt"}, "id": "pin-call",
        }])
        pin_tool = ToolMessage(content="PIN EVIDENCE", tool_call_id="pin-call", name="read_file")

        class FakeReact:
            def invoke(self, payload, config=None):
                seen["messages"] = list(payload["messages"])
                return {"messages": list(payload["messages"]) + [graph.AIMessage(content="answer")]}

        old_react = graph._REACT
        try:
            graph._REACT = FakeReact()
            with patch.object(
                graph, "_seed_pinned_files",
                return_value=([HumanMessage(content="q"), pin_ai, pin_tool], [pin_ai, pin_tool], {
                    "requested": 1, "success": ["/tmp/p.txt"], "failures": [],
                }),
            ), patch.object(graph, "_force_plan_or_directive", side_effect=lambda msgs: (msgs, [])):
                result = graph._react_node_impl(
                    {"messages": [HumanMessage(content="q")]},
                    {"configurable": {"thread_id": "pin-test"}},
                )
        finally:
            graph._REACT = old_react

        self.assertTrue(any(isinstance(m, ToolMessage) and m.content == "PIN EVIDENCE" for m in seen["messages"]))
        self.assertFalse(any(isinstance(m, ToolMessage) and m.content == "PIN EVIDENCE" for m in result["messages"]))
        self.assertEqual(result["messages"][-1].content, "answer")


if __name__ == "__main__":
    unittest.main()
