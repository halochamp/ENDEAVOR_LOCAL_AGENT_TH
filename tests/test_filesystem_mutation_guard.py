from __future__ import annotations

import hashlib
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

import config
import react
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from tools import edit_access
from tools._mutation_guard import classify_mutation_intent
from tools.bash import bash as bash_tool
from tools.bash_bg import bash_bg as bash_bg_tool
from tools.edit import edit as edit_tool
from tools.python_exec import python_exec as python_tool
from tools.read_file import read_file as read_tool
from tools.write_file import write_file as write_tool
from tools.tool_loop import tool_loop as tool_loop_tool
from tools import ALL_TOOLS
from tools._safety import _PROTECTED_PATHS


class FilesystemMutationGuardTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.internal = self.root / "internal"
        self.focus = self.root / "focus-a"
        self.focus_b = self.root / "focus-b"
        self.approved = self.root / "approved"
        self.outside = self.root / "outside"
        for folder in (self.internal, self.focus, self.focus_b, self.approved, self.outside):
            folder.mkdir()
        self._state_path = edit_access.APPROVED_EDIT_FOLDERS_PATH
        self._lock_path = edit_access._STATE_LOCK_PATH
        edit_access.APPROVED_EDIT_FOLDERS_PATH = str(self.root / "approved-state.json")
        edit_access._STATE_LOCK_PATH = str(self.root / "approved-state.json.lock")
        edit_access._SESSION_FOCUS_FOLDER = ""
        edit_access.save_focus_folder(str(self.focus))
        self._workspace_patch = patch.object(config, "WORKSPACE", str(self.internal))
        self._workspace_patch.start()
        tools = [read_tool, edit_tool, write_tool, bash_tool, bash_bg_tool, python_tool, tool_loop_tool]
        builder = StateGraph(MessagesState)
        builder.add_node("tools", ToolNode(tools, wrap_tool_call=react._guard_tool_call))
        builder.add_edge(START, "tools")
        builder.add_edge("tools", END)
        self.app = builder.compile()

    def tearDown(self):
        self._workspace_patch.stop()
        edit_access.APPROVED_EDIT_FOLDERS_PATH = self._state_path
        edit_access._STATE_LOCK_PATH = self._lock_path
        edit_access._SESSION_FOCUS_FOLDER = ""
        self.tmp.cleanup()

    def call(self, prompt, name, args, messages=None):
        history = list(messages or [HumanMessage(content=prompt)])
        if not any(isinstance(item, HumanMessage) for item in history):
            history.insert(0, HumanMessage(content=prompt))
        call_id = uuid.uuid4().hex
        ai = AIMessage(content="", tool_calls=[{
            "name": name, "args": args, "id": call_id, "type": "tool_call",
        }])
        state = self.app.invoke({"messages": history + [ai]})
        result = next(
            item for item in reversed(state["messages"])
            if isinstance(item, ToolMessage) and item.tool_call_id == call_id
        )
        return result, state["messages"]

    def read(self, prompt, path, messages=None):
        return self.call(prompt, "read_file", {"path": str(path)}, messages)

    def digest(self, path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    def test_edit_structured_create_only_creates_new_target(self):
        target = self.focus / "new.py"
        result, _ = self.call("Create new.py with a small function", "edit", {
            "path": "new.py", "mode": "create", "content": "def answer():\n    return 42\n",
        })
        self.assertTrue(target.is_file())
        self.assertIn("created", str(result.content))
        self.assertIn("syntax OK", str(result.content))
        target.write_text("old\n", encoding="utf-8")
        result, _ = self.call("Create new.py with new contents", "edit", {
            "path": "new.py", "mode": "create", "content": "new\n",
        })
        self.assertIn("create mode cannot overwrite", str(result.content))
        self.assertEqual(target.read_text(encoding="utf-8"), "old\n")

    def test_existing_edit_requires_current_turn_read_and_then_succeeds(self):
        target = self.focus / "calc.py"
        target.write_text("value = 1\n", encoding="utf-8")
        prompt = "Fix calc.py so value is 2"
        result, _ = self.call(prompt, "edit", {
            "path": "calc.py", "old_string": "1", "new_string": "2",
        })
        self.assertIn("read the existing target", str(result.content))
        self.assertEqual(target.read_text(encoding="utf-8"), "value = 1\n")

        history = [HumanMessage(content=prompt)]
        read_result, history = self.read(prompt, "calc.py", history)
        self.assertIn("value = 1", str(read_result.content))
        result, _ = self.call(prompt, "edit", {
            "path": "calc.py", "old_string": "1", "new_string": "2",
        }, history)
        self.assertIn("edited", str(result.content))
        self.assertEqual(target.read_text(encoding="utf-8"), "value = 2\n")

    def test_structured_replace_line_and_batch_modes_and_write_file_compatibility(self):
        self.assertIn(write_tool, ALL_TOOLS)
        target = self.focus / "modes.txt"
        target.write_text("one\ntwo\nthree\n", encoding="utf-8")
        prompt = "Update modes.txt with the requested content"
        history = [HumanMessage(content=prompt)]
        _, history = self.read(prompt, "modes.txt", history)
        result, history = self.call(prompt, "edit", {
            "path": "modes.txt", "mode": "replace", "content": "first\nsecond\n",
        }, history)
        self.assertIn("replaced", str(result.content))
        self.assertEqual(target.read_text(encoding="utf-8"), "first\nsecond\n")
        result, history = self.call(prompt, "edit", {
            "path": "modes.txt", "mode": "line", "line_start": 2,
            "line_end": 2, "new_string": "changed",
        }, history)
        self.assertIn("edited", str(result.content))
        self.assertEqual(target.read_text(encoding="utf-8"), "first\nchanged\n")
        result, _ = self.call(prompt, "edit", {
            "path": "modes.txt", "mode": "batch",
            "edits": [{"old_string": "changed", "new_string": "batched"}],
        }, history)
        self.assertIn("edited", str(result.content))
        self.assertEqual(target.read_text(encoding="utf-8"), "first\nbatched\n")

    def test_sandbox_mutation_scope_denies_unapproved_neighbor_even_within_process(self):
        target = self.focus / "inside.txt"
        outside = self.outside / "outside.txt"
        before = "keep"
        outside.write_text(before, encoding="utf-8")
        prompt = "Create inside.txt with text"
        result, _ = self.call(prompt, "bash", {
            "command": f"printf allowed > inside.txt; printf forbidden > {outside}",
        })
        self.assertEqual(target.read_text(encoding="utf-8"), "allowed")
        self.assertEqual(outside.read_text(encoding="utf-8"), before)
        self.assertEqual(getattr(result, "artifact", {}).get("provenance_type"), "bash_filesystem_delta_v1")
        self.assertEqual([Path(item["path"]).name for item in result.artifact["changed_paths"]], ["inside.txt"])

    def test_tool_loop_output_uses_shared_focus_and_read_before_replace(self):
        prompt = "Summarize the command and save the result as loop.md"
        result, _ = self.call(prompt, "tool_loop", {
            "items": ["pwd"], "action": "bash_each", "context": "loop test",
            "output_file": "loop.md", "max_n": 1,
        })
        target = self.focus / "loop.md"
        self.assertTrue(target.is_file())
        self.assertIn("Items: 1", target.read_text(encoding="utf-8"))
        self.assertNotIn("[BLOCKED]", str(result.content))

        target.write_text("old output", encoding="utf-8")
        blocked, _ = self.call(prompt, "tool_loop", {
            "items": ["pwd"], "action": "bash_each", "context": "loop test",
            "output_file": "loop.md", "max_n": 1,
        })
        self.assertIn("read the existing loop output target", str(blocked.content))
        self.assertEqual(target.read_text(encoding="utf-8"), "old output")

        history = [HumanMessage(content=prompt)]
        _, history = self.read(prompt, "loop.md", history)
        saved, _ = self.call(prompt, "tool_loop", {
            "items": ["pwd"], "action": "bash_each", "context": "loop test",
            "output_file": "loop.md", "max_n": 1,
        }, history)
        self.assertNotIn("[BLOCKED]", str(saved.content))
        self.assertIn("Items: 1", target.read_text(encoding="utf-8"))

    def test_tree_scope_sandbox_blocks_directory_symlink_escape(self):
        external = self.outside / "linked"
        external.mkdir()
        target = external / "victim.txt"
        target.write_text("unchanged", encoding="utf-8")
        before = self.digest(target)
        (self.focus / "escape").symlink_to(external, target_is_directory=True)
        result, _ = self.call(
            "Format the whole workspace with a safe marker pass",
            "bash",
            {"command": "printf changed > escape/victim.txt"},
        )
        self.assertEqual(self.digest(target), before)
        self.assertIsNone(getattr(result, "artifact", None))

    def test_explicit_existing_bash_mutation_after_read_emits_disk_delta_artifact(self):
        target = self.focus / "calc.py"
        target.write_text("value = 'old'\n", encoding="utf-8")
        before = self.digest(target)
        prompt = "Fix calc.py so value is new"
        history = [HumanMessage(content=prompt)]
        read_result, history = self.read(prompt, "calc.py", history)
        self.assertIn("old", str(read_result.content))
        result, _ = self.call(prompt, "bash", {
            "command": "printf \"value = 'new'\\n\" > calc.py",
        }, history)
        self.assertEqual(target.read_text(encoding="utf-8"), "value = 'new'\n")
        self.assertNotEqual(self.digest(target), before)
        artifact = getattr(result, "artifact", None)
        self.assertIsInstance(artifact, dict)
        self.assertEqual(artifact.get("provenance_type"), "bash_filesystem_delta_v1")
        changed = artifact["changed_paths"]
        self.assertEqual(len(changed), 1)
        self.assertEqual(changed[0]["path"], str(target.resolve()))
        self.assertEqual(changed[0]["before_sha256"], before)
        self.assertEqual(changed[0]["after_sha256"], self.digest(target))
        self.assertEqual(changed[0]["change_type"], "modified")
        self.assertEqual(artifact["observation_scope"]["coverage"], "requested_targets_only")
        self.assertEqual(artifact["observation_scope"]["requested_targets"], [str(target.resolve())])
        self.assertTrue(artifact["observation_scope"]["change_list_complete_within_scope"])
        self.assertIn("host observed filesystem delta", str(result.content))
        self.assertIsNone(edit_access.get_bash_mutation_scope())

    def test_existing_bash_target_without_read_is_blocked_before_execution(self):
        target = self.focus / "calc.py"
        target.write_text("OLD\n", encoding="utf-8")
        before = self.digest(target)
        prompt = "Fix calc.py"
        result, _ = self.call(prompt, "bash", {"command": "printf NEW > calc.py"})
        self.assertIn("read the existing requested target", str(result.content))
        self.assertEqual(self.digest(target), before)
        self.assertIsNone(getattr(result, "artifact", None))

    def test_read_from_previous_turn_does_not_authorize_existing_bash_mutation(self):
        target = self.focus / "calc.py"
        target.write_text("OLD\n", encoding="utf-8")
        before = self.digest(target)
        prior, history = self.read("Inspect calc.py", "calc.py")
        self.assertIn("OLD", str(prior.content))
        history.append(HumanMessage(content="Fix calc.py so it says NEW"))
        result, _ = self.call("Fix calc.py so it says NEW", "bash", {
            "command": "printf NEW > calc.py",
        }, history)
        self.assertIn("read the existing requested target", str(result.content))
        self.assertEqual(self.digest(target), before)

    def test_reading_one_target_does_not_authorize_a_different_existing_target(self):
        first = self.focus / "calc.py"
        second = self.focus / "other.py"
        first.write_text("OLD-A", encoding="utf-8")
        second.write_text("OLD-B", encoding="utf-8")
        before = (self.digest(first), self.digest(second))
        prompt = "Fix calc.py and other.py"
        history = [HumanMessage(content=prompt)]
        _, history = self.read(prompt, "calc.py", history)
        result, _ = self.call(prompt, "bash", {
            "command": "printf NEW-A > calc.py; printf NEW-B > other.py",
        }, history)
        self.assertIn("read the existing requested target", str(result.content))
        self.assertEqual((self.digest(first), self.digest(second)), before)

    def test_parallel_filesystem_tool_calls_in_one_agent_message_are_refused(self):
        target = self.focus / "calc.py"
        target.write_text("old", encoding="utf-8")
        prompt = "Fix calc.py to say new"
        history = [HumanMessage(content=prompt)]
        _, history = self.read(prompt, "calc.py", history)
        ids = [uuid.uuid4().hex, uuid.uuid4().hex]
        ai = AIMessage(content="", tool_calls=[
            {"name": "bash", "args": {"command": "printf one > calc.py"}, "id": ids[0], "type": "tool_call"},
            {"name": "bash", "args": {"command": "printf two > calc.py"}, "id": ids[1], "type": "tool_call"},
        ])
        state = self.app.invoke({"messages": history + [ai]})
        results = [item for item in state["messages"] if isinstance(item, ToolMessage) and item.tool_call_id in ids]
        self.assertEqual(len(results), 2)
        self.assertTrue(all("one filesystem mutation tool call at a time" in str(item.content) for item in results))
        self.assertEqual(target.read_text(encoding="utf-8"), "old")

    def test_nested_tool_loop_bash_batch_cannot_inherit_mutation_authority(self):
        target = self.focus / "calc.py"
        target.write_text("old", encoding="utf-8")
        result, _ = self.call("Fix calc.py to say new", "tool_loop", {
            "items": ["printf new > calc.py"], "action": "bash_each", "max_n": 1,
        })
        self.assertIn("cannot receive mutation authority", str(result.content))
        self.assertEqual(target.read_text(encoding="utf-8"), "old")

    def test_tree_snapshot_bound_fails_closed_before_command_execution(self):
        (self.focus / "a.txt").write_text("a", encoding="utf-8")
        (self.focus / "b.txt").write_text("b", encoding="utf-8")
        marker = self.focus / "must-not-run.txt"
        with patch("tools.bash._DELTA_MAX_FILES", 1):
            result, _ = self.call(
                "Format the whole workspace with a safe marker transform",
                "bash",
                {"command": "printf ran > must-not-run.txt"},
            )
        self.assertIn("exceeded bounded file/byte limits", str(result.content))
        self.assertFalse(marker.exists())
        self.assertIsNone(getattr(result, "artifact", None))

    def test_excess_changed_paths_fail_closed_without_complete_delta_claim(self):
        with patch("tools.bash._DELTA_MAX_CHANGES", 1):
            result, _ = self.call(
                "Format the whole workspace with a safe marker transform",
                "bash",
                {"command": "printf a > first.txt; printf b > second.txt"},
            )
        self.assertIn("changed path count exceeded", str(result.content))
        self.assertIn("filesystem delta unavailable", str(result.content))
        self.assertNotIn("host observed filesystem delta", str(result.content))
        self.assertIsNone(getattr(result, "artifact", None))
        self.assertTrue((self.focus / "first.txt").is_file())
        self.assertTrue((self.focus / "second.txt").is_file())

    def test_approved_folder_bash_write_is_allowed(self):
        edit_access.add_approved_edit_folders([str(self.approved)])
        target = self.approved / "approved.py"
        target.write_text("before\n", encoding="utf-8")
        prompt = f"Fix {target} to say after"
        history = [HumanMessage(content=prompt)]
        _, history = self.read(prompt, str(target), history)
        result, _ = self.call(prompt, "bash", {"command": f"printf after > {target}"}, history)
        self.assertEqual(target.read_text(encoding="utf-8"), "after")
        self.assertEqual(getattr(result, "artifact", {}).get("provenance_type"), "bash_filesystem_delta_v1")

    def test_focus_move_revokes_temporary_access_but_grants_new_focus(self):
        old_target = self.focus / "old.py"
        old_target.write_text("old\n", encoding="utf-8")
        before = self.digest(old_target)
        edit_access.save_focus_folder(str(self.focus_b))
        self.assertFalse(edit_access.path_is_approved_for_edit(str(old_target)))
        self.assertNotIn(str(self.focus.resolve()), edit_access.load_approved_edit_folders())
        prompt = f"Fix {old_target}"
        result, _ = self.call(prompt, "bash", {"command": f"printf changed > {old_target}"})
        self.assertIn("BLOCKED", str(result.content))
        self.assertEqual(self.digest(old_target), before)

        result, _ = self.call("Create new-b.txt with text", "bash", {"command": "printf B > new-b.txt"})
        self.assertEqual((self.focus_b / "new-b.txt").read_text(encoding="utf-8"), "B")
        self.assertEqual(getattr(result, "artifact", {}).get("provenance_type"), "bash_filesystem_delta_v1")

    def test_outside_folder_mutation_is_denied_and_unchanged(self):
        target = self.outside / "escape.py"
        target.write_text("original", encoding="utf-8")
        before = self.digest(target)
        result, _ = self.call(f"Fix {target}", "bash", {"command": f"printf changed > {target}"})
        self.assertIn("BLOCKED", str(result.content))
        self.assertEqual(self.digest(target), before)
        self.assertIsNone(getattr(result, "artifact", None))

    def test_existing_file_symlink_escape_is_denied_before_bash_execution(self):
        target = self.outside / "target.py"
        target.write_text("safe", encoding="utf-8")
        before = self.digest(target)
        (self.focus / "escape.py").symlink_to(target)
        result, _ = self.call("Fix escape.py", "bash", {"command": "printf unsafe > escape.py"})
        self.assertIn("symlink", str(result.content).lower())
        self.assertEqual(self.digest(target), before)
        self.assertIsNone(getattr(result, "artifact", None))

    def test_existing_symlink_between_authorized_roots_blocks_bash_before_execution(self):
        edit_access.add_approved_edit_folders([str(self.approved)])
        requested = self.focus / "calc.py"
        requested.write_text("requested-old", encoding="utf-8")
        linked = self.approved / "approved.txt"
        linked.write_text("approved-old", encoding="utf-8")
        before = self.digest(linked)
        (self.focus / "side-link.txt").symlink_to(linked)
        prompt = "Fix calc.py to say requested-new"
        history = [HumanMessage(content=prompt)]
        _, history = self.read(prompt, "calc.py", history)
        result, _ = self.call(prompt, "bash", {
            "command": "printf side-change > side-link.txt",
        }, history)
        self.assertIn("existing symlink", str(result.content).lower())
        self.assertIn("command was not executed", str(result.content))
        self.assertEqual(self.digest(linked), before)
        self.assertEqual(requested.read_text(encoding="utf-8"), "requested-old")
        self.assertIsNone(getattr(result, "artifact", None))

    def test_opaque_script_cannot_follow_new_symlink_into_another_authorized_root(self):
        edit_access.add_approved_edit_folders([str(self.approved)])
        requested = self.focus / "calc.py"
        requested.write_text("requested-old", encoding="utf-8")
        linked = self.approved / "approved.txt"
        linked.write_text("approved-old", encoding="utf-8")
        before = self.digest(linked)
        script = self.focus / "create_link.py"
        script.write_text(
            "from pathlib import Path\n"
            f"p = Path('side-link.txt')\n"
            f"p.symlink_to({str(linked)!r})\n"
            "p.write_text('side-change')\n",
            encoding="utf-8",
        )
        prompt = "Fix calc.py so it says requested-new"
        history = [HumanMessage(content=prompt)]
        _, history = self.read(prompt, "calc.py", history)
        result, _ = self.call(prompt, "bash", {
            "command": "python3 create_link.py",
        }, history)
        self.assertEqual(self.digest(linked), before)
        self.assertTrue((self.focus / "side-link.txt").is_symlink())
        self.assertIsNone(getattr(result, "artifact", None))

    def test_bash_call_scope_is_narrowed_to_the_requested_approved_root(self):
        edit_access.add_approved_edit_folders([str(self.approved)])
        target = self.approved / "approved.py"
        target.write_text("old", encoding="utf-8")
        prompt = f"Fix {target} to say new"
        history = [HumanMessage(content=prompt)]
        _, history = self.read(prompt, str(target), history)
        result, _ = self.call(prompt, "bash", {
            "command": f"printf new > {target}",
        }, history)
        self.assertEqual(target.read_text(encoding="utf-8"), "new")
        self.assertEqual(getattr(result, "artifact", {}).get("provenance_type"), "bash_filesystem_delta_v1")

    def test_symlink_scan_checks_focus_nested_beneath_ignored_parent(self):
        edit_access.add_approved_edit_folders([str(self.approved)])
        nested_focus = self.internal / ".venv" / "focused-project"
        nested_focus.mkdir(parents=True)
        edit_access.save_focus_folder(str(nested_focus))
        linked = self.approved / "approved.txt"
        linked.write_text("approved-old", encoding="utf-8")
        before = self.digest(linked)
        (nested_focus / "side-link.txt").symlink_to(linked)
        result, _ = self.call(
            "Format the whole workspace with a marker transform",
            "bash",
            {"command": "printf side-change > side-link.txt"},
        )
        self.assertIn("existing symlink", str(result.content).lower())
        self.assertEqual(self.digest(linked), before)
        self.assertIsNone(getattr(result, "artifact", None))

    def test_bash_mutation_scope_fails_closed_when_symlink_scan_is_bounded_out(self):
        (self.focus / "existing.txt").write_text("existing", encoding="utf-8")
        marker = self.focus / "must-not-run.txt"
        with patch("tools.bash._SYMLINK_SCAN_MAX_ENTRIES", 0):
            result, _ = self.call(
                "Format the whole workspace with a safe marker transform",
                "bash",
                {"command": "printf ran > must-not-run.txt"},
            )
        self.assertIn("bounded symlink entry scan", str(result.content))
        self.assertFalse(marker.exists())
        self.assertIsNone(getattr(result, "artifact", None))

    def test_guarded_bash_rejects_explicit_symlink_creation(self):
        target = self.focus / "calc.py"
        target.write_text("old", encoding="utf-8")
        link = self.focus / "new-link.py"
        result, _ = self.call("Fix calc.py to say new", "bash", {
            "command": f"ln -s {target} {link}; printf new > new-link.py",
        })
        self.assertIn("symlink creation is not allowed", str(result.content))
        self.assertFalse(link.exists())
        self.assertEqual(target.read_text(encoding="utf-8"), "old")

    def test_directory_symlink_escape_is_denied(self):
        external = self.outside / "dir"
        external.mkdir()
        target = external / "target.py"
        target.write_text("safe", encoding="utf-8")
        before = self.digest(target)
        (self.focus / "escape-dir").symlink_to(external, target_is_directory=True)
        result, _ = self.call("Fix escape-dir/target.py", "bash", {
            "command": "printf unsafe > escape-dir/target.py",
        })
        self.assertIn("symlink", str(result.content).lower())
        self.assertEqual(self.digest(target), before)

    def test_protected_path_under_authorized_root_is_denied(self):
        target = self.focus / "protected-state.json"
        target.write_text('{"keep": true}\n', encoding="utf-8")
        before = self.digest(target)
        with patch("tools._safety._PROTECTED_PATHS", _PROTECTED_PATHS + [str(target)]):
            result, _ = self.call("Fix protected-state.json", "bash", {
                "command": "printf bad > protected-state.json",
            })
        self.assertIn("protected", str(result.content).lower())
        self.assertEqual(self.digest(target), before)

    def test_read_only_bash_produces_no_delta_artifact(self):
        target = self.focus / "calc.py"
        target.write_text("value=1\n", encoding="utf-8")
        prompt = "Fix calc.py so value is 2"
        history = [HumanMessage(content=prompt)]
        _, history = self.read(prompt, "calc.py", history)
        result, _ = self.call(prompt, "bash", {"command": "git status --short"}, history)
        self.assertIsNone(getattr(result, "artifact", None))
        self.assertEqual(target.read_text(encoding="utf-8"), "value=1\n")

    def test_fake_stdout_and_zero_exit_do_not_prove_mutation(self):
        target = self.focus / "calc.py"
        target.write_text("value=1\n", encoding="utf-8")
        before = self.digest(target)
        prompt = "Fix calc.py so value is 2"
        history = [HumanMessage(content=prompt)]
        _, history = self.read(prompt, "calc.py", history)
        result, _ = self.call(prompt, "bash", {"command": "printf 'I wrote calc.py'"}, history)
        self.assertIn("I wrote calc.py", str(result.content))
        self.assertIsNone(getattr(result, "artifact", None))
        self.assertEqual(self.digest(target), before)

    def test_unclassified_bash_python_write_is_blocked_but_not_treated_as_intent(self):
        target = self.internal / "calc.py"
        target.write_text("old", encoding="utf-8")
        prompt = "Explain how to fix calc.py"
        intent = classify_mutation_intent([HumanMessage(content=prompt)])
        self.assertEqual(intent.kind, "none")
        result, _ = self.call(prompt, "bash", {
            "command": f"python3 -c \"from pathlib import Path; Path({str(target)!r}).write_text('bad')\"",
        })
        self.assertIn("file change must be explicit", str(result.content))
        self.assertEqual(target.read_text(encoding="utf-8"), "old")

    def test_clear_workspace_transform_uses_bounded_tree_scope(self):
        prompt = "Format the whole workspace with a harmless marker transform"
        intent = classify_mutation_intent([HumanMessage(content=prompt)])
        self.assertEqual(intent.kind, "tree")
        result, _ = self.call(prompt, "bash", {"command": "printf marker > generated.txt"})
        target = self.focus / "generated.txt"
        self.assertEqual(target.read_text(encoding="utf-8"), "marker")
        artifact = getattr(result, "artifact", None)
        self.assertEqual(artifact.get("provenance_type"), "bash_filesystem_delta_v1")
        self.assertEqual([Path(item["path"]).name for item in artifact["changed_paths"]], ["generated.txt"])

    def test_unsafe_explicit_target_never_falls_back_to_tree_scope(self):
        target = self.outside / "bad.py"
        target.write_text("unchanged", encoding="utf-8")
        before = self.digest(target)
        prompt = f"Fix {target} as part of the workspace project"
        intent = classify_mutation_intent([HumanMessage(content=prompt)])
        self.assertEqual(intent.kind, "blocked_explicit")
        result, _ = self.call(prompt, "bash", {"command": f"printf changed > {target}"})
        self.assertIn("BLOCKED", str(result.content))
        self.assertEqual(self.digest(target), before)
        self.assertIsNone(getattr(result, "artifact", None))

    def test_python_exec_is_not_a_requested_file_mutation_executor(self):
        target = self.focus / "calc.py"
        target.write_text("value=1\n", encoding="utf-8")
        prompt = "Fix calc.py so value is 2"
        result, _ = self.call(prompt, "python_exec", {
            "code": f"from pathlib import Path; Path({str(target)!r}).write_text('value=2')",
        })
        self.assertIn("cannot receive mutation authority", str(result.content))
        self.assertEqual(target.read_text(encoding="utf-8"), "value=1\n")

    def test_unsafe_explicit_target_cannot_fall_back_to_python_or_background_bash(self):
        target = self.outside / "target.py"
        target.write_text("safe", encoding="utf-8")
        before = self.digest(target)
        symlink = self.focus / "escape.py"
        symlink.symlink_to(target)
        prompt = "Fix escape.py so it is safe"
        python_result, _ = self.call(prompt, "python_exec", {
            "code": f"from pathlib import Path; Path({str(symlink)!r}).write_text('unsafe')",
        })
        self.assertIn("symlink", str(python_result.content).lower())
        self.assertEqual(self.digest(target), before)
        bg_result, _ = self.call(prompt, "bash_bg", {
            "action": "start", "command": f"printf unsafe > {symlink}",
        })
        self.assertIn("symlink", str(bg_result.content).lower())
        self.assertEqual(self.digest(target), before)

    def test_duplicate_call_guard_still_composes_with_filesystem_guard(self):
        prompt = "Show the current directory"
        first, first_history = self.call(prompt, "bash", {"command": "pwd"})
        result, _ = self.call(prompt, "bash", {"command": "pwd"}, first_history)
        self.assertIn("tool_loop_hint", str(result.content))


if __name__ == "__main__":
    unittest.main()
