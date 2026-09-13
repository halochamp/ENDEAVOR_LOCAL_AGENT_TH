from __future__ import annotations

import asyncio
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
        self._standalone_model = cfg._current_standalone_model
        self._mode = cfg._current_server_mode
        self._port = cfg._current_server_port
        self._locked = cfg._LOCKED_MODEL
        self._override = cfg._MLX_BASE_URL_OVERRIDE
        self._url = cfg.MLX_BASE_URL
        self._tmp = tempfile.TemporaryDirectory()
        cfg._RUNTIME_SETTINGS_PATH = str(Path(self._tmp.name) / "runtime_settings.json")
        cfg._LOCKED_MODEL = ""
        cfg._MLX_BASE_URL_OVERRIDE = ""
        cfg._current_model = cfg.DEFAULT_MODEL
        cfg._current_standalone_model = cfg.DEFAULT_MODEL
        cfg._current_thinking_budget = 1024
        cfg._current_server_mode = "standalone"
        cfg._current_server_port = 8085
        cfg.MLX_BASE_URL = cfg.get_mlx_base_url()

    def tearDown(self) -> None:
        cfg = srv._config
        cfg._RUNTIME_SETTINGS_PATH = self._path
        cfg._current_model = self._model
        cfg._current_thinking_budget = self._budget
        cfg._current_standalone_model = self._standalone_model
        cfg._current_server_mode = self._mode
        cfg._current_server_port = self._port
        cfg._LOCKED_MODEL = self._locked
        cfg._MLX_BASE_URL_OVERRIDE = self._override
        cfg.MLX_BASE_URL = self._url
        self._tmp.cleanup()

    def _write(self, *, model: str, budget: int, mode: str = "standalone", port: int = 8085) -> None:
        Path(srv._config._RUNTIME_SETTINGS_PATH).write_text(
            json.dumps({
                "owner": "agent_th",
                "model": model,
                "standalone_model": model,
                "thinking_budget": budget,
                "server_mode": mode,
                "server_port": port,
            }),
            encoding="utf-8",
        )

    def test_backend_adopts_cli_written_budget_and_port(self) -> None:
        model = srv._config.DEFAULT_MODEL
        self._write(model=model, budget=512, port=8091)
        with patch.object(srv, "_model_server_status", return_value={"healthy": True, "loaded_model": model}), \
             patch.object(srv, "_rebuild_runtime_llms") as rebuild:
            changed = srv._sync_runtime_settings_from_owner_file_if_idle()
        self.assertTrue(changed)
        self.assertEqual(srv._config.get_thinking_budget(), 512)
        self.assertEqual(srv._config.get_server_port(), 8091)
        rebuild.assert_called_once_with()

    def test_shared_sync_adopts_actual_max_model(self) -> None:
        self._write(
            model=srv._config.DEFAULT_MODEL,
            budget=512,
            mode="shared_max",
            port=8085,
        )
        shared = srv._config.HIGH_QUALITY_MODEL
        with patch.object(srv, "_model_server_status", return_value={
            "healthy": True, "loaded_model": shared,
        }), patch.object(srv, "_rebuild_runtime_llms"):
            changed = srv._sync_runtime_settings_from_owner_file_if_idle()
        self.assertTrue(changed)
        self.assertEqual(srv._config.get_server_mode(), "shared_max")
        self.assertEqual(srv._config.get_model(), shared)

    def test_runtime_apply_shared_mode_never_calls_owner_start_stop_reset(self) -> None:
        shared = srv._config.HIGH_QUALITY_MODEL
        shared_status = {
            "healthy": True,
            "loaded_model": shared,
            "state": "shared_ready",
            "port": 8085,
        }
        with patch.object(srv, "_shared_max_server_status", return_value=shared_status), \
             patch.object(srv, "_model_server_status", return_value={"running": False, "owned": False}), \
             patch.object(srv, "_start_owner_model_server") as start, \
             patch.object(srv, "_stop_owner_model_server") as stop, \
             patch.object(srv, "_restart_owner_model_server") as reset, \
             patch.object(srv, "_rebuild_runtime_llms"):
            result = asyncio.run(srv._apply_runtime_settings(
                srv._config.DEFAULT_MODEL, 512, "shared_max", 8085,
            ))
        self.assertEqual(result["server_mode"], "shared_max")
        self.assertEqual(result["model"], shared)
        start.assert_not_called()
        stop.assert_not_called()
        reset.assert_not_called()

    def test_leaving_shared_mode_restores_previous_standalone_model(self) -> None:
        cfg = srv._config
        cfg.set_runtime_settings(
            model=cfg.DEFAULT_MODEL,
            thinking_budget=1024,
            server_mode="standalone",
            server_port=8091,
        )
        cfg.set_runtime_settings(
            model=cfg.HIGH_QUALITY_MODEL,
            thinking_budget=1024,
            server_mode="shared_max",
            server_port=8085,
        )
        cfg.adopt_shared_model(cfg.HIGH_QUALITY_MODEL)
        with patch.object(srv, "_start_owner_model_server", return_value={"healthy": True}), \
             patch.object(srv, "_rebuild_runtime_llms"):
            result = asyncio.run(srv._apply_runtime_settings(
                cfg.HIGH_QUALITY_MODEL, 512, "standalone", 8091,
            ))
        self.assertEqual(result["server_mode"], "standalone")
        self.assertEqual(result["model"], cfg.DEFAULT_MODEL)
        self.assertEqual(cfg.get_standalone_model(), cfg.DEFAULT_MODEL)

    def test_shared_model_server_action_is_read_only(self) -> None:
        srv._config.set_runtime_settings(
            model=srv._config.HIGH_QUALITY_MODEL,
            thinking_budget=1024,
            server_mode="shared_max",
            server_port=8085,
        )
        with patch.object(srv, "_model_server_status", return_value={
            "mode": "shared_max", "managed_by_th": False, "state": "shared_ready",
            "healthy": True, "running": True, "owned": True, "port": 8085,
            "selected_model": srv._config.HIGH_QUALITY_MODEL,
            "loaded_model": srv._config.HIGH_QUALITY_MODEL,
            "watchdog_enabled": False, "desired_state": "shared", "error": "",
        }):
            result = asyncio.run(srv._apply_model_server_action("stop"))
        self.assertIn("read-only", result["error"])

    def test_bundled_browser_ui_routes_are_retired(self) -> None:
        paths = {getattr(route, "path", "") for route in srv.api.routes}
        self.assertNotIn("/ui", paths)
        self.assertNotIn("/ui-token", paths)
        self.assertIn("/ws", paths)
        self.assertIn("/chat", paths)
        self.assertFalse((Path(srv.__file__).resolve().parent / "chat.html").exists())


if __name__ == "__main__":
    unittest.main()
