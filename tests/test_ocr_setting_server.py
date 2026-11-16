"""Agent server: Settings -> OCR frames (LLM-free, the engines themselves are stubbed)."""
from __future__ import annotations

import asyncio
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("AGENT_SERVER_TOKEN", "test-ocr-setting-token")

import agent_server as srv
import user_settings


def _status(openthai_ready: bool = True, selected: str = "apple") -> dict:
    return {"selected": selected, "effective": selected, "engines": {
        "apple": {"available": True, "detail": ""},
        "openthai": {"available": openthai_ready, "detail": "" if openthai_ready else "weights missing"}}}


class OcrSettingServerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="agent-th-ocr-setting-")
        self.addCleanup(self.tmp.cleanup)
        self._old = os.environ.get("V2_USER_SETTINGS_PATH")
        os.environ["V2_USER_SETTINGS_PATH"] = str(Path(self.tmp.name) / "user_settings.json")
        self.addCleanup(self._restore)

    def _restore(self) -> None:
        if self._old is None:
            os.environ.pop("V2_USER_SETTINGS_PATH", None)
        else:
            os.environ["V2_USER_SETTINGS_PATH"] = self._old

    def _set(self, raw, ready=True):
        # engine_status() reads the saved choice, so the stub follows the file like the real one
        def fake_status():
            return _status(ready, user_settings.ocr_engine())
        with patch("tools._ocr.engine_status", fake_status):
            return asyncio.run(srv._set_ocr_engine_from_ui(raw))

    def test_choosing_a_ready_engine_saves_it_and_reports_the_new_state(self) -> None:
        frame = self._set("openthai")
        self.assertEqual((frame["type"], frame["engine"], frame["error"]), ("ocr_engine_state", "openthai", ""))
        self.assertEqual(user_settings.ocr_engine(), "openthai")
        self.assertEqual(self._set("apple")["engine"], "apple")

    def test_unknown_values_are_refused_and_nothing_is_saved(self) -> None:
        for bad in ("tesseract", "", None, 1, ["openthai"]):
            frame = self._set(bad)
            self.assertIn("engine must be one of", frame["error"], bad)
        self.assertEqual(user_settings.ocr_engine(), "apple")

    def test_an_engine_that_cannot_run_is_refused_with_the_reason(self) -> None:
        frame = self._set("openthai", ready=False)
        self.assertIn("openthai OCR is not ready: weights missing", frame["error"])
        self.assertEqual(user_settings.ocr_engine(), "apple")

    def test_refused_while_a_run_is_active(self) -> None:
        async def go():
            with patch("tools._ocr.engine_status", lambda: _status(True, user_settings.ocr_engine())):
                async with srv._busy:
                    return await srv._set_ocr_engine_from_ui("openthai")
        frame = asyncio.run(go())
        self.assertEqual(frame["error"], "agent is busy")
        self.assertEqual(user_settings.ocr_engine(), "apple")

    def test_a_broken_ocr_module_never_breaks_the_status_poll(self) -> None:
        with patch("tools._ocr.engine_status", side_effect=RuntimeError("boom")):
            fields = srv._ocr_status_fields()
        self.assertEqual((fields["ocr_engine"], fields["ocr_effective"], fields["ocr_engines"]), ("apple", "apple", {}))

    def test_the_status_poll_carries_the_ocr_fields(self) -> None:
        with patch("tools._ocr.engine_status", lambda: _status(True, "openthai")):
            fields = srv._ocr_status_fields()
        self.assertEqual((fields["ocr_engine"], fields["ocr_effective"]), ("openthai", "openthai"))
        self.assertTrue(fields["ocr_engines"]["openthai"]["available"])


if __name__ == "__main__":
    unittest.main()
