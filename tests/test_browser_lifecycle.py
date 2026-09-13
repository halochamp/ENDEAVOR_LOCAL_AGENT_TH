"""CPU-only regressions for browser lifecycle recovery and browser activation."""
from __future__ import annotations

import asyncio
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import _mac_input
from tools import browser_use_tool as bu


class BrowserLifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.state_path = Path(self.tmp.name) / "browser_session.json"
        self.state_patch = patch.object(bu, "_state_path", return_value=self.state_path)
        self.state_patch.start()

    def tearDown(self) -> None:
        self.state_patch.stop()
        self.tmp.cleanup()

    def test_pidless_quarantine_preserves_recovery_metadata(self) -> None:
        state = bu._State()
        rec = {
            "pid": 0,
            "needs_close": True,
            "_state_blocked": True,
            "launch_in_progress": True,
            "user_data_dir": "/tmp/browser-use-user-data-dir-fixture",
            "url": "https://example.com",
        }
        state.save_quarantine(rec)
        self.assertEqual(state.load(), rec)

    def test_exact_profile_blocked_state_self_clears_when_browser_is_absent(self) -> None:
        state = bu._State()
        rec = {
            "pid": 0,
            "needs_close": True,
            "_state_blocked": True,
            "launch_in_progress": True,
            "user_data_dir": "/tmp/browser-use-user-data-dir-fixture",
        }
        state.save_quarantine(rec)
        with patch.object(bu, "_profile_browser_pids", return_value=[]), \
             patch.object(bu, "_cleanup_temp_profile", return_value=True):
            self.assertIsNone(bu._live(state))
        self.assertIsNone(state.load())

    def test_exact_profile_blocked_state_recovers_one_owned_process_as_close_only(self) -> None:
        state = bu._State()
        rec = {
            "pid": 0,
            "needs_close": True,
            "_state_blocked": True,
            "launch_in_progress": True,
            "user_data_dir": "/tmp/browser-use-user-data-dir-fixture",
        }
        state.save_quarantine(rec)
        with patch.object(bu, "_profile_browser_pids", return_value=[4242]), \
             patch.object(bu, "_ps_lstart", return_value="fixture-start"), \
             patch.object(bu, "_is_alive", return_value=True):
            recovered = bu._live(state)
        self.assertEqual(recovered["pid"], 4242)
        self.assertEqual(recovered["start_sig"], "fixture-start")
        self.assertTrue(recovered["needs_close"])
        self.assertNotIn("_state_blocked", recovered)

    def test_generic_blocked_registry_stays_fail_closed_without_process_probe(self) -> None:
        state = bu._State()
        state.save_quarantine({"pid": 0, "needs_close": True, "_state_blocked": True})
        with patch.object(bu, "_profile_browser_pids") as probe:
            rec = bu._live(state)
        self.assertTrue(rec["_state_blocked"])
        probe.assert_not_called()

    def test_operation_lock_rejects_overlap_and_releases(self) -> None:
        first = bu._State()
        second = bu._State()
        handle = first.try_operation()
        self.assertIsNotNone(handle)
        try:
            self.assertIsNone(second.try_operation())
        finally:
            first.release_operation(handle)
        retry = second.try_operation()
        self.assertIsNotNone(retry)
        second.release_operation(retry)

    def test_browser_inner_llm_uses_current_no_think_api_and_exact_profile(self) -> None:
        class FakeChatOpenAI:
            last_kwargs = None

            def __init__(self, **kwargs):
                type(self).last_kwargs = kwargs

        class FakeBrowserProfile:
            last_kwargs = None

            def __init__(self, **kwargs):
                type(self).last_kwargs = kwargs
                for key, value in kwargs.items():
                    setattr(self, key, value)

        class FakeBrowserSession:
            def __init__(self, *, browser_profile=None, cdp_url=None, keep_alive=False):
                self.browser_profile = browser_profile or FakeBrowserProfile()
                self.cdp_url = cdp_url
                self._local_browser_watchdog = None

        class FakeResult:
            def is_successful(self):
                return True

            def final_result(self):
                return "Example Domain"

            def extracted_content(self):
                return ["Example Domain"]

            def errors(self):
                return []

        class FakeAgent:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            async def run(self):
                await asyncio.sleep(0)
                return FakeResult()

        browser_mod = types.ModuleType("browser_use")
        browser_mod.Agent = FakeAgent
        browser_mod.BrowserProfile = FakeBrowserProfile
        browser_pkg = types.ModuleType("browser_use.browser")
        session_mod = types.ModuleType("browser_use.browser.session")
        session_mod.BrowserSession = FakeBrowserSession
        llm_mod = types.ModuleType("browser_use.llm")
        llm_mod.ChatOpenAI = FakeChatOpenAI

        with patch.dict(sys.modules, {
            "browser_use": browser_mod,
            "browser_use.browser": browser_pkg,
            "browser_use.browser.session": session_mod,
            "browser_use.llm": llm_mod,
        }), patch.object(bu, "_wc_check", return_value=None), \
             patch.object(bu, "_wc_check_and_inc", return_value=None), \
             patch.object(bu, "_profile_browser_pids", return_value=[]), \
             patch.object(bu, "_cleanup_temp_profile", return_value=True), \
             patch.object(bu.web_cache, "get", return_value=None), \
             patch.object(bu.web_cache, "put"), patch.object(bu.web_cache, "put_summary"), \
             patch.object(bu, "summarize", return_value="Example Domain"):
            result = bu._browser_use_serialized(
                url="https://example.com", task="report title", keep_open=False,
            )

        kwargs = FakeChatOpenAI.last_kwargs
        self.assertEqual(kwargs["reasoning_effort"], "none")
        self.assertEqual(kwargs["reasoning_models"], [kwargs["model"]])
        self.assertNotIn("extra_body", kwargs)
        profile = FakeBrowserProfile.last_kwargs["user_data_dir"]
        self.assertTrue(Path(profile).name.startswith(bu._FRESH_PROFILE_PREFIX))
        self.assertIn("Example Domain", result)
        self.assertIsNone(bu._State().load())


class OpenUrlTests(unittest.TestCase):
    def test_open_url_reactivates_requested_browser_after_dispatch(self) -> None:
        completed = types.SimpleNamespace(returncode=0, stderr="")
        with patch.object(_mac_input.subprocess, "run", return_value=completed) as run, \
             patch.object(_mac_input, "open_app") as activate:
            _mac_input.open_url("https://www.youtube.com", app="Google Chrome")
        self.assertEqual(run.call_args_list[0].args[0], [
            "open", "-a", "Google Chrome", "https://www.youtube.com",
        ])
        activate.assert_called_once_with("Google Chrome")


if __name__ == "__main__":
    unittest.main()
