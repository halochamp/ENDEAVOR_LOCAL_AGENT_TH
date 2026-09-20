from __future__ import annotations

import asyncio
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fastapi import HTTPException

import agent_server as srv
import endeavor_agent
import graph
from langchain_core.messages import HumanMessage
from tools import edit_access
from tools._safety import plan_write
from tools.edit import edit as edit_tool


class EditAccessStateTests(unittest.TestCase):
    def setUp(self) -> None:
        self._state_path = edit_access.APPROVED_EDIT_FOLDERS_PATH
        self._lock_path = edit_access._STATE_LOCK_PATH
        self._tmp = tempfile.TemporaryDirectory()
        edit_access.APPROVED_EDIT_FOLDERS_PATH = str(Path(self._tmp.name) / "state.json")
        edit_access._STATE_LOCK_PATH = f"{edit_access.APPROVED_EDIT_FOLDERS_PATH}.lock"
        edit_access._SESSION_FOCUS_FOLDER = ""

    def tearDown(self) -> None:
        edit_access.APPROVED_EDIT_FOLDERS_PATH = self._state_path
        edit_access._STATE_LOCK_PATH = self._lock_path
        edit_access._SESSION_FOCUS_FOLDER = ""
        self._tmp.cleanup()

    def test_focus_is_temporary_and_never_auto_approved(self) -> None:
        approved = Path(self._tmp.name) / "approved"
        focused = Path(self._tmp.name) / "focused"
        next_focus = Path(self._tmp.name) / "next"
        approved.mkdir()
        focused.mkdir()
        next_focus.mkdir()

        edit_access.add_approved_edit_folders([str(approved)])
        edit_access.save_focus_folder(str(focused))
        state = edit_access.load_edit_access_state()
        self.assertEqual(state["folders"], [str(approved.resolve())])
        self.assertEqual(state["focus_folder"], str(focused.resolve()))
        self.assertNotIn(str(focused.resolve()), state["folders"])
        self.assertTrue(edit_access.path_is_approved_for_edit(str(focused / "draft.txt")))

        edit_access.save_focus_folder(str(next_focus))
        self.assertFalse(edit_access.path_is_approved_for_edit(str(focused / "draft.txt")))
        self.assertTrue(edit_access.path_is_approved_for_edit(str(next_focus / "draft.txt")))
        edit_access.save_focus_folder("")
        self.assertFalse(edit_access.path_is_approved_for_edit(str(next_focus / "draft.txt")))
        self.assertTrue(edit_access.path_is_approved_for_edit(str(approved / "kept.txt")))
        persisted = json.loads(Path(edit_access.APPROVED_EDIT_FOLDERS_PATH).read_text(encoding="utf-8"))
        self.assertNotIn("focus_folder", persisted)

    def test_legacy_persisted_focus_is_ignored_and_fresh_session_starts_clear(self) -> None:
        approved = Path(self._tmp.name) / "approved"
        focused = Path(self._tmp.name) / "focused"
        approved.mkdir()
        focused.mkdir()
        Path(edit_access.APPROVED_EDIT_FOLDERS_PATH).write_text(json.dumps({
            "folders": [str(approved)], "focus_folder": str(focused),
        }), encoding="utf-8")
        self.assertEqual(edit_access.load_approved_edit_folders(), [str(approved.resolve())])
        self.assertEqual(edit_access.get_focus_folder(), "")
        edit_access.save_focus_folder(str(focused))
        self.assertEqual(edit_access.get_focus_folder(), str(focused.resolve()))
        edit_access._SESSION_FOCUS_FOLDER = ""  # fresh process/session semantics
        self.assertEqual(edit_access.get_focus_folder(), "")
        self.assertTrue(edit_access.path_is_approved_for_edit(str(approved / "kept.txt")))
        self.assertFalse(edit_access.path_is_approved_for_edit(str(focused / "draft.txt")))

    def test_edit_tool_path_is_direct_only_while_approved_or_focused(self) -> None:
        focused = Path(self._tmp.name) / "focused"
        focused.mkdir()
        target = focused / "new.txt"
        edit_access.save_focus_folder(str(focused))
        effective, error, note = plan_write(str(target), allow_approved_edit=True)
        self.assertEqual(effective, str(target))
        self.assertIsNone(error)
        self.assertEqual(note, "approved edit folder")

        edit_access.save_focus_folder("")
        effective, error, note = plan_write(str(target), allow_approved_edit=True)
        self.assertIn("outside Approved Edit Folders", error or "")
        self.assertEqual(effective, str(target))
        self.assertIsNone(note)

    def test_edit_tool_writes_focused_target_and_blocks_after_focus_clear(self) -> None:
        focused = Path(self._tmp.name) / "focused"
        focused.mkdir()
        target = focused / "draft.txt"
        target.write_text("before", encoding="utf-8")
        edit_access.save_focus_folder(str(focused))
        result = edit_tool.invoke({
            "path": str(target),
            "old_string": "before",
            "new_string": "after",
        })
        self.assertIn("after", target.read_text(encoding="utf-8"))
        self.assertNotIn("BLOCKED", result)

        edit_access.save_focus_folder("")
        target.write_text("before", encoding="utf-8")
        result = edit_tool.invoke({
            "path": str(target),
            "old_string": "before",
            "new_string": "should not write",
        })
        self.assertIn("BLOCKED", result)
        self.assertEqual(target.read_text(encoding="utf-8"), "before")

    def test_backend_focus_actions_preserve_persistent_list(self) -> None:
        approved = Path(self._tmp.name) / "approved"
        focused = Path(self._tmp.name) / "focused"
        approved.mkdir()
        focused.mkdir()
        edit_access.add_approved_edit_folders([str(approved)])

        result = asyncio.run(srv._apply_focus_folder(str(focused)))
        self.assertEqual(result["type"], "approved_edit_folders")
        self.assertEqual(result["folders"], [str(approved.resolve())])
        self.assertEqual(result["focus_folder"], str(focused.resolve()))

        result = asyncio.run(srv._apply_focus_folder(""))
        self.assertEqual(result["folders"], [str(approved.resolve())])
        self.assertEqual(result["focus_folder"], "")

    def test_backend_rejects_non_directory_without_mutating_state(self) -> None:
        missing = Path(self._tmp.name) / "missing"
        result = asyncio.run(srv._apply_focus_folder(str(missing)))
        self.assertTrue(result["error"])
        self.assertEqual(result["folders"], [])
        self.assertEqual(result["focus_folder"], "")

    def test_backend_edit_scope_revalidates_runtime_focus_and_rejects_arbitrary_path(self) -> None:
        focused = Path(self._tmp.name) / "focused"
        arbitrary = Path(self._tmp.name) / "arbitrary"
        focused.mkdir()
        arbitrary.mkdir()
        focused_file = focused / "draft.txt"
        arbitrary_file = arbitrary / "secret.txt"
        focused_file.write_text("draft", encoding="utf-8")
        arbitrary_file.write_text("secret", encoding="utf-8")
        edit_access.save_focus_folder(str(focused))
        self.assertEqual(srv._validated_edit_target(str(focused_file)), str(focused_file.resolve()))
        with self.assertRaises(HTTPException) as ctx:
            srv._validated_edit_target(str(arbitrary_file))
        self.assertEqual(ctx.exception.status_code, 403)
        edit_access.save_focus_folder("")
        with self.assertRaises(HTTPException):
            srv._validated_edit_target(str(focused_file))

    def test_cli_action_uses_same_state_and_does_not_auto_add_focus(self) -> None:
        focused = Path(self._tmp.name) / "focused"
        focused.mkdir()
        with patch.object(endeavor_agent, "_validate_approved_edit_folder", return_value=str(focused.resolve())):
            result = endeavor_agent._apply_cli_edit_access_action("focus", str(focused))
        self.assertEqual(result["folders"], [])
        self.assertEqual(result["focus_folder"], str(focused.resolve()))


class FocusPromptTests(unittest.TestCase):
    def test_unset_active_workspace_is_explicit_without_edit_grant(self) -> None:
        original = HumanMessage(content="Which workspace is active?")
        messages = [original]
        injected = graph._inject_focus_folder_prompt(
            messages,
            {"configurable": {"focus_folder": ""}},
        )
        self.assertIsNot(injected, messages)
        self.assertEqual(original.content, "Which workspace is active?")
        self.assertIn("Active Workspace: Not set", injected[0].content)
        self.assertNotIn("temporary edit access", injected[0].content)

    def test_active_workspace_lookup_failure_does_not_claim_not_set(self) -> None:
        original = HumanMessage(content="Which workspace is active?")
        messages = [original]
        with patch.object(edit_access, "get_focus_folder", side_effect=OSError("state unavailable")):
            injected = graph._inject_focus_folder_prompt(messages, None)
        self.assertIs(injected, messages)
        self.assertEqual(original.content, "Which workspace is active?")

    def test_focus_prompt_is_ephemeral_and_uses_configured_focus(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            original = HumanMessage(content="Please edit the draft")
            messages = [original]
            injected = graph._inject_focus_folder_prompt(
                messages,
                {"configurable": {"focus_folder": td}},
            )
            self.assertIsNot(injected, messages)
            self.assertEqual(original.content, "Please edit the draft")
            self.assertIn(str(Path(td).resolve()), injected[0].content)
            self.assertIn("temporary", injected[0].content)
