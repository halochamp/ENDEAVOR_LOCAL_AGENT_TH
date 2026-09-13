from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

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

    def test_runtime_payload_exposes_vlm_labels_consistently(self) -> None:
        payload = srv._runtime_settings_payload()
        labels = {item["value"]: item["label"] for item in payload["model_options"]}
        self.assertEqual(labels[srv._config.LIGHT_VLM_MODEL], "Qwen3.5 2B · VLM")
        self.assertEqual(labels[srv._config.COMPACT_VLM_MODEL], "Qwen3.5 9B · VLM")
        self.assertEqual(labels[srv._config.HIGH_QUALITY_MODEL], "Qwen3.6 35B · VLM")

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

    def test_generation_rate_is_actual_stream_callback_count_over_rolling_five_seconds(self) -> None:
        with srv._generation_token_lock:
            old_times = list(srv._generation_token_times)
            old_runs = set(srv._generation_active_runs)
            srv._generation_token_times.clear()
            srv._generation_active_runs.clear()
        try:
            srv._generation_run_start("run-a")
            for i in range(25):
                srv._mark_generation_token(now=96.0 + (i * 0.16))
            self.assertEqual(srv._generation_tokens_per_second(now=100.0), 5.0)
            srv._generation_run_end("run-a")
            self.assertEqual(srv._generation_tokens_per_second(now=100.0), 5.0)
            srv._generation_run_start("run-b")
            self.assertEqual(srv._generation_tokens_per_second(now=100.0), 5.0)
            srv._generation_run_end("run-b")
            self.assertEqual(srv._generation_tokens_per_second(now=105.0), 0.0)
        finally:
            with srv._generation_token_lock:
                srv._generation_token_times.clear()
                srv._generation_token_times.extend(old_times)
                srv._generation_active_runs.clear()
                srv._generation_active_runs.update(old_runs)

    def test_generation_meter_counts_reasoning_or_tool_chunks_without_model_specific_tokenizer(self) -> None:
        reasoning = MagicMock()
        reasoning.message.content = ""
        reasoning.message.additional_kwargs = {"reasoning_content": "คิด"}
        reasoning.message.tool_call_chunks = []
        reasoning.message.tool_calls = []
        tool = MagicMock()
        tool.message.content = ""
        tool.message.additional_kwargs = {}
        tool.message.tool_call_chunks = [{"args": "{}"}]
        tool.message.tool_calls = []
        empty = MagicMock()
        empty.message.content = ""
        empty.message.additional_kwargs = {}
        empty.message.tool_call_chunks = []
        empty.message.tool_calls = []
        self.assertTrue(srv._callback_has_generation_payload("x", {}))
        self.assertTrue(srv._callback_has_generation_payload("", {"chunk": reasoning}))
        self.assertTrue(srv._callback_has_generation_payload("", {"chunk": tool}))
        self.assertFalse(srv._callback_has_generation_payload("", {"chunk": empty}))

    def test_system_telemetry_is_self_contained_and_portable(self) -> None:
        old_has_psutil = srv._HAS_PSUTIL
        old_ram_total = srv._SYSTEM_RAM_TOTAL
        old_net_prev = srv._system_net_prev
        fake_psutil = MagicMock()
        fake_psutil.cpu_percent.return_value = 12.5
        fake_psutil.virtual_memory.return_value = MagicMock(
            used=20 * (1024 ** 3), total=48 * (1024 ** 3), percent=41.7,
        )
        fake_psutil.net_io_counters.return_value = MagicMock(
            bytes_sent=3_000, bytes_recv=6_000,
        )
        try:
            srv._HAS_PSUTIL = True
            srv._SYSTEM_RAM_TOTAL = 48 * (1024 ** 3)
            srv._system_net_prev = (1_000, 2_000, 100.0)
            with patch.object(srv, "_psutil", fake_psutil), \
                 patch.object(srv, "_portable_ram_used_bytes", return_value=24 * (1024 ** 3)), \
                 patch.object(srv, "_portable_gpu_percent", return_value=75.0), \
                 patch.object(srv.time, "monotonic", return_value=102.0):
                sample = srv._collect_system_telemetry()
        finally:
            srv._HAS_PSUTIL = old_has_psutil
            srv._SYSTEM_RAM_TOTAL = old_ram_total
            srv._system_net_prev = old_net_prev

        self.assertEqual(sample["type"], "system_telemetry")
        self.assertEqual(sample["cpu_percent"], 12.5)
        self.assertEqual(sample["gpu_percent"], 75.0)
        self.assertEqual(sample["ram_used_bytes"], 24 * (1024 ** 3))
        self.assertEqual(sample["ram_total_bytes"], 48 * (1024 ** 3))
        self.assertEqual(sample["network_up_bytes_per_second"], 1_000.0)
        self.assertEqual(sample["network_down_bytes_per_second"], 2_000.0)
        self.assertEqual(sample["tokens_per_second_5s"], 0.0)

    def test_telemetry_does_not_depend_on_private_monitor_or_machine_paths(self) -> None:
        source = Path(srv.__file__).read_text(encoding="utf-8")
        self.assertNotIn("SERVER_MONITOR", source)
        self.assertNotIn("/Users/", source)
        with patch.object(srv.shutil, "which", return_value=None):
            self.assertIsNone(srv._portable_gpu_percent())

    def test_telemetry_is_scoped_to_explicit_desktop_transport(self) -> None:
        desktop = MagicMock()
        desktop.query_params = {"transport": "desktop"}
        custom = MagicMock()
        custom.query_params = {}
        self.assertEqual(srv._ws_transport(desktop), "desktop")
        self.assertEqual(srv._ws_transport(custom), "")
        renderer = (Path(srv.__file__).resolve().parent / "AGENT_UI" / "renderer.js").read_text(
            encoding="utf-8",
        )
        self.assertIn("/ws?transport=desktop", renderer)

    def test_workspace_mentions_are_recursive_and_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            nested = root / "docs"
            nested.mkdir()
            target = nested / "world.doc"
            target.write_text("hello", encoding="utf-8")
            outside = root.parent / "outside-mention.txt"
            outside.write_text("outside", encoding="utf-8")
            try:
                with patch.object(srv, "WORKSPACE", str(root)):
                    index = srv._list_workspace_mentions()
                    self.assertTrue(any(item.get("relative_path") == "docs/world.doc" for item in index))
                    grounded = srv._augment_query_with_workspace_mentions(
                        "สรุปไฟล์ให้หน่อย", ["docs/world.doc"],
                    )
                    self.assertIn(str(target.resolve()), grounded)
                    self.assertIn("read_file", grounded)
                    with self.assertRaises(ValueError):
                        srv._augment_query_with_workspace_mentions("x", ["../outside-mention.txt"])
            finally:
                outside.unlink(missing_ok=True)

    def test_pdf_panel_routes_are_authenticated_backend_surfaces(self) -> None:
        paths = {getattr(route, "path", "") for route in srv.api.routes}
        self.assertIn("/pdf-to-text/start", paths)
        self.assertIn("/pdf-to-text/status", paths)
        source = (Path(srv.__file__).resolve().parent / "pdf_to_text.py").read_text(encoding="utf-8")
        self.assertNotIn("/Users/", source)
        self.assertNotIn("SERVER_MONITOR", source)
        self.assertIn("enable_thinking", source)
        self.assertIn('"thinking_budget": 0', source)

    def test_bundled_browser_ui_routes_are_retired(self) -> None:
        paths = {getattr(route, "path", "") for route in srv.api.routes}
        self.assertNotIn("/ui", paths)
        self.assertNotIn("/ui-token", paths)
        self.assertIn("/ws", paths)
        self.assertIn("/chat", paths)
        self.assertFalse((Path(srv.__file__).resolve().parent / "chat.html").exists())


if __name__ == "__main__":
    unittest.main()
