"""Agent server: Internet/Offline is the owner's saved choice, not a one-shot startup probe (LLM-free)."""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

os.environ.setdefault("AGENT_SERVER_TOKEN", "test-internet-setting-token")

import agent_server as srv
import user_settings


class _Tool:
    def __init__(self, name: str) -> None:
        self.name = name


class InternetSettingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="agent-th-internet-")
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "user_settings.json"
        self._old = os.environ.get("V2_USER_SETTINGS_PATH")
        os.environ["V2_USER_SETTINGS_PATH"] = str(self.path)
        self.addCleanup(self._restore)

    def _restore(self) -> None:
        if self._old is None:
            os.environ.pop("V2_USER_SETTINGS_PATH", None)
        else:
            os.environ["V2_USER_SETTINGS_PATH"] = self._old

    def _state(self, online: bool = True):
        """A _State without the memory store / graph build: only what set_online touches."""
        state = srv._State.__new__(srv._State)
        state.online, state._saver, state.thread_id = online, object(), "t"
        state.app, state.cfg = "old-app", {"old": True}
        return state

    def _all_tools(self):
        return [_Tool("bash"), *[_Tool(name) for name in sorted(srv._WEB_TOOLS)]]

    def test_startup_reads_the_saved_choice_and_never_probes_the_network(self) -> None:
        with patch.object(srv, "_open_memory_store", return_value=(MagicMock(), MagicMock())), \
             patch.object(srv._State, "_rebuild_app", lambda self, extra_tools=None: None), \
             patch.object(srv, "_internet_up", side_effect=AssertionError("startup must not probe")):
            self.assertTrue(srv._State().online)          # no file = online, even though a probe would fail
            user_settings.set_internet(False)
            self.assertFalse(srv._State().online)

    def test_offline_hides_the_web_tools_and_online_binds_them_again(self) -> None:
        seen: list[set] = []

        def fake_graph(**kw):
            seen.append({t.name for t in kw["tools"]})
            return f"app{len(seen)}"

        state = self._state(online=True)
        with patch.object(srv, "ALL_TOOLS", self._all_tools()), patch.object(srv, "build_graph", fake_graph), \
             patch.object(srv, "_load_memory_md", return_value=""):
            self.assertIsNone(state.set_online(False))
            self.assertFalse(user_settings.internet_online())
            self.assertTrue(seen[-1].isdisjoint(srv._WEB_TOOLS) and "bash" in seen[-1])
            self.assertIsNone(state.set_online(True))
            self.assertTrue(srv._WEB_TOOLS <= seen[-1])
            before = len(seen)
            self.assertIsNone(state.set_online(True))      # same value: nothing rebuilt
            self.assertEqual(len(seen), before)

    def test_a_failed_rebuild_restores_file_flag_and_graph(self) -> None:
        state = self._state(online=True)
        with patch.object(srv, "ALL_TOOLS", self._all_tools()), patch.object(srv, "_load_memory_md", return_value=""), \
             patch.object(srv, "build_graph", side_effect=RuntimeError("boom")):
            frame = state.set_online(False)
        self.assertTrue(frame["error"])
        self.assertTrue(state.online)
        self.assertEqual((state.app, state.cfg), ("old-app", {"old": True}))
        self.assertTrue(user_settings.internet_online())   # the file says online again

    def test_an_unsaveable_choice_changes_nothing(self) -> None:
        state = self._state(online=True)
        with patch.object(user_settings, "set_internet", side_effect=OSError("disk")):
            frame = state.set_online(False)
        self.assertTrue(frame["error"])
        self.assertTrue(state.online)

    def test_ui_request_is_validated_and_refused_while_a_run_is_active(self) -> None:
        state = self._state(online=True)

        async def go():
            with patch.object(srv, "_state", state):
                self.assertEqual((await srv._set_internet_from_ui("offline"))["error"], "online must be true or false")
                async with srv._busy:
                    busy = await srv._set_internet_from_ui(False)
                self.assertEqual(busy["error"], "agent is busy")
                self.assertTrue(state.online)
                with patch.object(srv._State, "_rebuild_app", lambda self, extra_tools=None: None):
                    ok = await srv._set_internet_from_ui(False)
                return ok

        ok = asyncio.run(go())
        self.assertEqual((ok["type"], ok["enabled"], ok["error"]), ("internet_state", False, ""))

    def test_status_separates_the_saved_choice_from_the_reachability_dot(self) -> None:
        state = self._state(online=False)
        state.skill = ""
        with patch.object(srv, "_state", state), patch.object(srv, "_internet_reachable", True), \
             patch.object(srv, "_internet_up", side_effect=AssertionError("/status must not probe once the loop has a value")), \
             patch.object(srv, "_mlx_up", return_value=True), patch.object(srv, "_list_skills", return_value=[]), \
             patch.object(srv, "_load_builtin_cmds", return_value=[]):
            status = srv.get_status()
        self.assertTrue(status["online"])                     # the dot: reachable
        self.assertFalse(status["internet_enabled"])          # the switch: the owner chose Offline


if __name__ == "__main__":
    unittest.main()
