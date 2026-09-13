from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import config
import endeavor_agent
import runtime_model


class SharedRuntimeConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self._path = config._RUNTIME_SETTINGS_PATH
        self._model = config._current_model
        self._budget = config._current_thinking_budget
        self._standalone_model = config._current_standalone_model
        self._mode = config._current_server_mode
        self._port = config._current_server_port
        self._locked = config._LOCKED_MODEL
        self._override = config._MLX_BASE_URL_OVERRIDE
        self._url = config.MLX_BASE_URL
        self._tmp = tempfile.TemporaryDirectory()
        config._RUNTIME_SETTINGS_PATH = str(Path(self._tmp.name) / "runtime_settings.json")
        config._LOCKED_MODEL = ""
        config._MLX_BASE_URL_OVERRIDE = ""
        config._current_model = config.DEFAULT_MODEL
        config._current_standalone_model = config.DEFAULT_MODEL
        config._current_thinking_budget = 1024
        config._current_server_mode = "standalone"
        config._current_server_port = 8085
        config.MLX_BASE_URL = config.get_mlx_base_url()

    def tearDown(self) -> None:
        config._RUNTIME_SETTINGS_PATH = self._path
        config._current_model = self._model
        config._current_thinking_budget = self._budget
        config._current_standalone_model = self._standalone_model
        config._current_server_mode = self._mode
        config._current_server_port = self._port
        config._LOCKED_MODEL = self._locked
        config._MLX_BASE_URL_OVERRIDE = self._override
        config.MLX_BASE_URL = self._url
        self._tmp.cleanup()

    def test_labels_options_and_default_server_contract(self) -> None:
        settings = config.get_runtime_settings()
        self.assertEqual(config.get_model_label(), "Qwen3 14B · text")
        self.assertEqual(config.get_thinking_budget_label(), "High")
        self.assertEqual(settings["server_mode"], "standalone")
        self.assertEqual(settings["server_port"], 8085)
        self.assertEqual(config.get_mlx_base_url(), "http://localhost:8085/v1")
        self.assertEqual(
            [x["value"] for x in settings["server_mode_options"]],
            ["standalone", "shared_max"],
        )
        self.assertEqual(
            [(x["label"], x["value"]) for x in settings["thinking_options"]],
            [("Low", 256), ("Medium", 512), ("High", 1024), ("xhigh", 1536), ("Max", 2048)],
        )

    def test_default_model_is_qwen3_14b_and_9b_vlm_is_selectable(self) -> None:
        self.assertEqual(config.DEFAULT_MODEL, "Qwen/Qwen3-14B-MLX-4bit")
        self.assertEqual(config.COMPACT_VLM_MODEL, "mlx-community/Qwen3.5-9B-4bit")
        self.assertEqual(config.MODEL_CHOICES[0], config.DEFAULT_MODEL)
        self.assertEqual(config.MODEL_CHOICES[1], config.COMPACT_VLM_MODEL)
        self.assertEqual(config.MODEL_CHOICES[2], config.HIGH_QUALITY_MODEL)

    def test_runtime_port_moves_endpoint_and_persists_owner_state(self) -> None:
        changed = config.set_runtime_settings(
            model=config.DEFAULT_MODEL,
            thinking_budget=512,
            server_mode="standalone",
            server_port=8091,
        )
        self.assertTrue(changed["server_port_changed"])
        self.assertEqual(config.get_mlx_base_url(), "http://localhost:8091/v1")
        payload = json.loads(Path(config._RUNTIME_SETTINGS_PATH).read_text(encoding="utf-8"))
        self.assertEqual(payload, {
            "owner": "agent_th",
            "model": config.DEFAULT_MODEL,
            "standalone_model": config.DEFAULT_MODEL,
            "thinking_budget": 512,
            "server_mode": "standalone",
            "server_port": 8091,
        })

    def test_shared_max_mode_locks_model_but_not_thinking_budget(self) -> None:
        config.set_runtime_settings(
            model=config.HIGH_QUALITY_MODEL,
            thinking_budget=1024,
            server_mode="shared_max",
            server_port=8085,
        )
        settings = config.get_runtime_settings()
        self.assertTrue(settings["shared_server"])
        self.assertTrue(settings["model_locked"])
        self.assertEqual(settings["model_options"][0]["value"], config.HIGH_QUALITY_MODEL)
        changed = config.set_runtime_settings(
            model=config.HIGH_QUALITY_MODEL,
            thinking_budget=512,
            server_mode="shared_max",
            server_port=8085,
        )
        self.assertTrue(changed["thinking_budget_changed"])

    def test_35b_warning_is_only_below_24gb(self) -> None:
        gb = 1024 ** 3
        self.assertTrue(config.high_quality_model_warning_required(
            config.HIGH_QUALITY_MODEL, ram_bytes=16 * gb,
        ))
        self.assertFalse(config.high_quality_model_warning_required(
            config.HIGH_QUALITY_MODEL, ram_bytes=24 * gb,
        ))
        self.assertFalse(config.high_quality_model_warning_required(
            config.DEFAULT_MODEL, ram_bytes=16 * gb,
        ))

    def test_cli_35b_low_ram_warning_is_confirmable_not_blocked(self) -> None:
        gb = 1024 ** 3
        with patch.object(config, "high_quality_model_warning_required", return_value=True), \
             patch.object(config, "physical_memory_bytes", return_value=16 * gb), \
             patch.object(endeavor_agent, "prompt_user", return_value="y"):
            self.assertTrue(endeavor_agent._confirm_high_quality_model_on_low_ram(config.HIGH_QUALITY_MODEL))
        with patch.object(config, "high_quality_model_warning_required", return_value=True), \
             patch.object(config, "physical_memory_bytes", return_value=16 * gb), \
             patch.object(endeavor_agent, "prompt_user", return_value="n"):
            self.assertFalse(endeavor_agent._confirm_high_quality_model_on_low_ram(config.HIGH_QUALITY_MODEL))

    def test_cli_budget_change_does_not_restart_owner_server(self) -> None:
        with patch.object(endeavor_agent, "restart_owner_model_server") as restart:
            changed = endeavor_agent._apply_cli_runtime_settings(config.DEFAULT_MODEL, 512)
        self.assertTrue(changed["thinking_budget_changed"])
        self.assertFalse(changed["model_changed"])
        restart.assert_not_called()

    def test_cli_model_change_restarts_owned_standalone_server(self) -> None:
        new = config.COMPACT_VLM_MODEL
        with patch.object(endeavor_agent, "restart_owner_model_server", return_value={"healthy": True}) as restart:
            changed = endeavor_agent._apply_cli_runtime_settings(new, 1536)
        self.assertTrue(changed["model_changed"])
        self.assertEqual(config.get_model(), new)
        restart.assert_called_once()

    def test_cli_shared_mode_refuses_model_change(self) -> None:
        config.set_runtime_settings(
            model=config.HIGH_QUALITY_MODEL,
            thinking_budget=1024,
            server_mode="shared_max",
            server_port=8085,
        )
        with self.assertRaisesRegex(ValueError, "shared MAX"):
            endeavor_agent._apply_cli_runtime_settings(config.DEFAULT_MODEL, 512)

    def test_cli_runtime_match_uses_model_server_status(self) -> None:
        with patch.object(endeavor_agent, "model_server_status", return_value={
            "healthy": True, "loaded_model": config.DEFAULT_MODEL,
        }):
            matches, active = endeavor_agent._runtime_model_matches_active_server()
        self.assertTrue(matches)
        self.assertEqual(active, config.DEFAULT_MODEL)


class RuntimeModelDiscoveryTests(unittest.TestCase):
    def test_model_parser_supports_space_and_equals_forms(self) -> None:
        self.assertEqual(
            runtime_model.model_from_process_command("python -m mlx_vlm.server --model Qwen/Test --port 8085"),
            "Qwen/Test",
        )
        self.assertEqual(
            runtime_model.model_from_process_command("python server.py --model=Qwen/Test"),
            "Qwen/Test",
        )

    def test_active_local_model_uses_listener_process_not_catalogue(self) -> None:
        lsof = subprocess.CompletedProcess([], 0, stdout="p1234\n", stderr="")
        ps = subprocess.CompletedProcess([], 0, stdout="python server.py --model Qwen/Test --port 8085\n", stderr="")
        with patch.object(runtime_model.subprocess, "run", side_effect=[lsof, ps]) as run:
            model = runtime_model.active_local_mlx_model("http://localhost:8085/v1")
        self.assertEqual(model, "Qwen/Test")
        self.assertEqual(run.call_count, 2)

    def test_remote_backend_is_not_process_inspected(self) -> None:
        with patch.object(runtime_model.subprocess, "run") as run:
            model = runtime_model.active_local_mlx_model("https://example.invalid/v1")
        self.assertEqual(model, "")
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
