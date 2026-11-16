"""Deterministic tests for user_settings.py (the owner's Internet/Offline choice), LLM-free."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import user_settings as us


class UserSettingsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="agent-th-user-settings-")
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "sub" / "user_settings.json"
        self._old = os.environ.get("V2_USER_SETTINGS_PATH")
        os.environ["V2_USER_SETTINGS_PATH"] = str(self.path)
        self.addCleanup(self._restore)

    def _restore(self) -> None:
        if self._old is None:
            os.environ.pop("V2_USER_SETTINGS_PATH", None)
        else:
            os.environ["V2_USER_SETTINGS_PATH"] = self._old

    def test_default_is_online_and_path_override_is_honoured(self) -> None:
        self.assertEqual(us.settings_path(), str(self.path))
        self.assertTrue(us.internet_online())

    def test_offline_and_online_round_trip_with_schema_version(self) -> None:
        us.set_internet(False)
        self.assertFalse(us.internet_online())
        self.assertEqual(json.loads(self.path.read_text()), {"schema_version": 1, "internet": "offline"})
        us.set_internet(True)
        self.assertTrue(us.internet_online())
        self.assertEqual(json.loads(self.path.read_text())["internet"], "online")

    def test_other_saved_keys_survive_a_write(self) -> None:
        self.path.parent.mkdir(parents=True)
        self.path.write_text(json.dumps({"schema_version": 1, "internet": "offline", "future_key": [1, 2]}))
        us.set_internet(True)
        self.assertEqual(json.loads(self.path.read_text()).get("future_key"), [1, 2])

    def test_unreadable_or_unknown_content_means_online_never_silently_offline(self) -> None:
        self.path.parent.mkdir(parents=True)
        for text in ("{not json", "[1]", '{"internet": "maybe"}', ""):
            self.path.write_text(text)
            self.assertTrue(us.internet_online(), text)

    def test_only_a_real_bool_is_accepted(self) -> None:
        for bad in ("offline", "online", 0, 1, None):
            with self.assertRaises(ValueError, msg=repr(bad)):
                us.set_internet(bad)

    def test_no_temp_files_are_left_behind_and_a_blocked_target_raises_oserror(self) -> None:
        us.set_internet(False)
        self.assertEqual([p.name for p in self.path.parent.iterdir() if p.name != self.path.name], [])
        blocked = Path(self.tmp.name) / "blocked.json"
        blocked.mkdir()
        os.environ["V2_USER_SETTINGS_PATH"] = str(blocked)
        with self.assertRaises(OSError):
            us.set_internet(False)
        self.assertEqual(list(Path(self.tmp.name).glob(".user_settings-*")), [])


if __name__ == "__main__":
    unittest.main()
