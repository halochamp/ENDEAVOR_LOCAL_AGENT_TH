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
        self._shared = config._SHARED_MLX_MODEL
        self._locked = config._LOCKED_MODEL
        self._tmp = tempfile.TemporaryDirectory()
        config._RUNTIME_SETTINGS_PATH = str(Path(self._tmp.name) / "runtime_settings.json")
        config._SHARED_MLX_MODEL = ""
        config._LOCKED_MODEL = ""
        config._current_model = config.MODEL_CHOICES[0]
        config._current_thinking_budget = 1024

    def tearDown(self) -> None:
        config._RUNTIME_SETTINGS_PATH = self._path
        config._current_model = self._model
        config._current_thinking_budget = self._budget
        config._SHARED_MLX_MODEL = self._shared
        config._LOCKED_MODEL = self._locked
        self._tmp.cleanup()

    def test_labels_and_options_match_desktop_contract(self) -> None:
        settings = config.get_runtime_settings()
        self.assertEqual(config.get_model_label(), "Qwen3 14B · text")
        self.assertEqual(config.get_thinking_budget_label(), "High")
        self.assertEqual(
            [(x["label"], x["value"]) for x in settings["thinking_options"]],
            [("Low", 256), ("Medium", 512), ("High", 1024), ("xhigh", 1536), ("Max", 2048)],
        )

    def test_default_model_is_qwen3_14b(self) -> None:
        self.assertEqual(config.DEFAULT_MODEL, "Qwen/Qwen3-14B-MLX-4bit")
        self.assertEqual(config.MODEL_CHOICES[0], config.DEFAULT_MODEL)
        self.assertEqual(config.MODEL_CHOICES[1], config.HIGH_QUALITY_MODEL)

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

    def test_cli_budget_change_persists_shared_owner_file_without_restart(self) -> None:
        current = config.get_model()
        changed = endeavor_agent._apply_cli_runtime_settings(current, 512)
        self.assertTrue(changed["thinking_budget_changed"])
        self.assertFalse(changed["model_changed"])
        self.assertFalse(changed.get("restart_required", False))
        payload = json.loads(Path(config._RUNTIME_SETTINGS_PATH).read_text(encoding="utf-8"))
        self.assertEqual(payload, {
            "owner": "agent_th",
            "model": current,
            "thinking_budget": 512,
        })

    def test_cli_model_change_is_persisted_but_does_not_kill_running_server(self) -> None:
        old = config.MODEL_CHOICES[0]
        new = config.MODEL_CHOICES[1]
        with patch.object(endeavor_agent, "active_local_mlx_model", return_value=old):
            changed = endeavor_agent._apply_cli_runtime_settings(new, 1536)
        self.assertTrue(changed["model_changed"])
        self.assertTrue(changed["restart_required"])
        self.assertEqual(changed["active_model"], old)
        self.assertEqual(config.get_model(), new)

    def test_cli_startup_fails_closed_on_saved_model_listener_mismatch(self) -> None:
        config._current_model = config.MODEL_CHOICES[1]
        with patch.object(endeavor_agent, "active_local_mlx_model", return_value=config.MODEL_CHOICES[0]):
            matches, active = endeavor_agent._runtime_model_matches_active_server()
        self.assertFalse(matches)
        self.assertEqual(active, config.MODEL_CHOICES[0])


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
