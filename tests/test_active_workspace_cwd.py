"""Active Workspace is the one relative base for every file tool.

read_file / read_image / edit already resolve relative names under the Active
Workspace.  Read-only bash and bash_bg must start in the same folder, otherwise
the model can `ls` a file it cannot read and retypes (and garbles) the long
absolute path.  Isolated: temp workspace, temp approval state, no real jobs.
"""
from __future__ import annotations

import importlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import config
import graph
from tools import bash_bg as bash_bg_tool  # the LangChain tool object
from tools import edit_access
from tools._safety import resolve_read_path
from tools.bash import _active_workspace_cwd, _bash_execute, _classify_bash_error

# `tools.bash` / `tools.bash_bg` as attributes of `tools` are tool objects; the
# modules themselves (needed for patching) live in sys.modules.
bash_module = importlib.import_module("tools.bash")
bash_bg = importlib.import_module("tools.bash_bg")


class _Base(unittest.TestCase):
    def setUp(self) -> None:
        self._state_path = edit_access.APPROVED_EDIT_FOLDERS_PATH
        self._lock_path = edit_access._STATE_LOCK_PATH
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name).resolve()
        edit_access.APPROVED_EDIT_FOLDERS_PATH = str(root / "state.json")
        edit_access._STATE_LOCK_PATH = f"{edit_access.APPROVED_EDIT_FOLDERS_PATH}.lock"
        edit_access._SESSION_FOCUS_FOLDER = ""
        self.workspace = root / "agent_workspace"
        self.workspace.mkdir()
        self.active = root / "project"
        self.active.mkdir()
        (self.active / "hello.txt").write_text("hi", encoding="utf-8")
        self._ws_patch = patch.object(config, "WORKSPACE", str(self.workspace))
        self._ws_patch.start()

    def tearDown(self) -> None:
        self._ws_patch.stop()
        edit_access.APPROVED_EDIT_FOLDERS_PATH = self._state_path
        edit_access._STATE_LOCK_PATH = self._lock_path
        edit_access._SESSION_FOCUS_FOLDER = ""
        self._tmp.cleanup()


class BashCwdTests(_Base):
    def test_read_only_bash_starts_in_the_active_workspace(self) -> None:
        edit_access.save_focus_folder(str(self.active))
        out, artifact = _bash_execute("pwd; ls")
        self.assertIsNone(artifact)
        self.assertIn(str(self.active), out)
        self.assertIn("hello.txt", out)

    def test_without_an_active_workspace_bash_stays_in_workspace(self) -> None:
        out, _ = _bash_execute("pwd")
        self.assertIn(str(self.workspace), out)
        self.assertNotIn(str(self.active), out)

    def test_bash_and_read_tools_agree_on_the_relative_base(self) -> None:
        edit_access.save_focus_folder(str(self.active))
        listing, _ = _bash_execute("ls")
        self.assertIn("hello.txt", listing)
        self.assertEqual(
            resolve_read_path("hello.txt"), str((self.active / "hello.txt").resolve())
        )

    def test_unusable_active_workspace_falls_back_to_workspace(self) -> None:
        edit_access._SESSION_FOCUS_FOLDER = str(self.active / "missing")
        self.assertIsNone(_active_workspace_cwd())
        edit_access._SESSION_FOCUS_FOLDER = str(self.active)
        with patch("tools._safety.resolve_read_path", side_effect=PermissionError("protected")):
            self.assertIsNone(_active_workspace_cwd())
        self.assertEqual(_active_workspace_cwd(), str(self.active))

    def test_changing_cwd_does_not_widen_the_sandbox_profile(self) -> None:
        # Only the starting folder moves; the profile is still built from the
        # internal workspace alone (no Active Workspace write allowance added).
        edit_access.save_focus_folder(str(self.active))
        calls: list[tuple] = []
        real = bash_module._build_sandbox_profile

        def spy(*args, **kwargs):
            calls.append((args, kwargs))
            return real(*args, **kwargs)

        with patch.object(bash_module, "_build_sandbox_profile", side_effect=spy):
            _bash_execute("pwd")
        self.assertEqual(calls, [((str(self.workspace),), {})])


class BashHintTests(unittest.TestCase):
    def test_not_found_hint_points_back_to_relative_names(self) -> None:
        hint = _classify_bash_error(
            1, "ls: /a/b: No such file or directory", "ls /a/b", "/ws", active_workspace="/proj"
        )
        self.assertIn("relative", hint)

    def test_no_hint_without_an_active_workspace(self) -> None:
        self.assertIsNone(
            _classify_bash_error(1, "ls: x: No such file or directory", "ls x", "/ws")
        )


class _FakeProc:
    pid = 2147483000

    def poll(self):
        return None


class BashBgCwdTests(_Base):
    def tearDown(self) -> None:
        bash_bg._PROCS.clear()
        bash_bg._PROFILE_PATHS.clear()
        super().tearDown()

    def _start(self) -> dict:
        captured: dict = {}

        def fake_popen(argv, **kwargs):
            captured.update(kwargs)
            return _FakeProc()

        with patch.object(bash_bg.subprocess, "Popen", side_effect=fake_popen):
            result = bash_bg_tool.invoke({"action": "start", "command": "sleep 5"})
        self.assertIn("started job", result)
        return captured

    def test_background_job_starts_in_the_active_workspace(self) -> None:
        edit_access.save_focus_folder(str(self.active))
        self.assertEqual(self._start()["cwd"], str(self.active))

    def test_background_job_without_active_workspace_uses_workspace(self) -> None:
        self.assertEqual(self._start()["cwd"], str(self.workspace))


class DirectiveTests(unittest.TestCase):
    def test_directive_tells_the_model_to_use_relative_names(self) -> None:
        text = graph._FOCUS_FOLDER_DIRECTIVE
        self.assertIn("Relative paths in bash, read_file, read_image and edit resolve here", text)
        self.assertIn("never retype this path", text)


if __name__ == "__main__":
    unittest.main()
