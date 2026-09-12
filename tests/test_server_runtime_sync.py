from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("AGENT_SERVER_TOKEN", "test-runtime-sync-token")

import agent_server as srv


class ServerRuntimeSyncTests(unittest.TestCase):
    def setUp(self) -> None:
        cfg = srv._config
        self._path = cfg._RUNTIME_SETTINGS_PATH
        self._model = cfg._current_model
        self._budget = cfg._current_thinking_budget
        self._shared = cfg._SHARED_MLX_MODEL
        self._locked = cfg._LOCKED_MODEL
        self._tmp = tempfile.TemporaryDirectory()
        cfg._RUNTIME_SETTINGS_PATH = str(Path(self._tmp.name) / "runtime_settings.json")
        cfg._SHARED_MLX_MODEL = ""
        cfg._LOCKED_MODEL = ""
        cfg._current_model = cfg.MODEL_CHOICES[0]
        cfg._current_thinking_budget = 1024

    def tearDown(self) -> None:
        cfg = srv._config
        cfg._RUNTIME_SETTINGS_PATH = self._path
        cfg._current_model = self._model
        cfg._current_thinking_budget = self._budget
        cfg._SHARED_MLX_MODEL = self._shared
        cfg._LOCKED_MODEL = self._locked
        self._tmp.cleanup()

    def _write(self, model: str, budget: int) -> None:
        Path(srv._config._RUNTIME_SETTINGS_PATH).write_text(
            json.dumps({"owner": "agent_th", "model": model, "thinking_budget": budget}),
            encoding="utf-8",
        )

    def test_backend_adopts_cli_written_budget_when_model_matches_listener(self) -> None:
        model = srv._config.MODEL_CHOICES[0]
        self._write(model, 512)
        with patch.object(srv, "_active_mlx_model", return_value=model), \
             patch.object(srv, "_rebuild_runtime_llms") as rebuild:
            changed = srv._sync_runtime_settings_from_owner_file_if_idle()
        self.assertTrue(changed)
        self.assertEqual(srv._config.get_thinking_budget(), 512)
        rebuild.assert_called_once_with()

    def test_backend_rejects_cli_model_that_does_not_match_listener(self) -> None:
        active = srv._config.MODEL_CHOICES[0]
        persisted = srv._config.MODEL_CHOICES[1]
        self._write(persisted, 512)
        with patch.object(srv, "_active_mlx_model", return_value=active), \
             patch.object(srv, "_rebuild_runtime_llms") as rebuild:
            changed = srv._sync_runtime_settings_from_owner_file_if_idle()
        self.assertFalse(changed)
        self.assertEqual(srv._config.get_model(), active)
        self.assertEqual(srv._config.get_thinking_budget(), 1024)
        rebuild.assert_not_called()

    def test_bundled_browser_ui_routes_are_retired(self) -> None:
        paths = {getattr(route, "path", "") for route in srv.api.routes}
        self.assertNotIn("/ui", paths)
        self.assertNotIn("/ui-token", paths)
        self.assertIn("/ws", paths)
        self.assertIn("/chat", paths)
        self.assertFalse((Path(srv.__file__).resolve().parent / "chat.html").exists())


if __name__ == "__main__":
    unittest.main()
