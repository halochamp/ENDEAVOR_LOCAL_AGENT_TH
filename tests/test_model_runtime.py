from __future__ import annotations

from pathlib import Path
import importlib.util
import sys
import tempfile
import types
import unittest
from unittest import mock

import config
import model_runtime as runtime


_launcher_spec = importlib.util.spec_from_file_location(
    "th_owned_launcher", Path(__file__).resolve().parents[1] / "scripts" / "run_mlx_server_owned.py"
)
launcher = importlib.util.module_from_spec(_launcher_spec)
assert _launcher_spec and _launcher_spec.loader
_launcher_spec.loader.exec_module(launcher)


class ModelRuntimeOwnershipTests(unittest.TestCase):
    def setUp(self) -> None:
        self._model = config._current_model
        self._budget = config._current_thinking_budget
        self._standalone_model = config._current_standalone_model
        self._mode = config._current_server_mode
        self._port = config._current_server_port
        self._override = config._MLX_BASE_URL_OVERRIDE
        self._settings = config._RUNTIME_SETTINGS_PATH
        self._control = runtime._CONTROL_SETTINGS_PATH
        self._tmp = tempfile.TemporaryDirectory()
        config._MLX_BASE_URL_OVERRIDE = ""
        config._current_model = config.DEFAULT_MODEL
        config._current_standalone_model = config.DEFAULT_MODEL
        config._current_thinking_budget = 1024
        config._current_server_mode = "standalone"
        config._current_server_port = 8085
        config._RUNTIME_SETTINGS_PATH = str(Path(self._tmp.name) / "runtime_settings.json")
        runtime._CONTROL_SETTINGS_PATH = Path(self._tmp.name) / "model_server_settings.json"

    def tearDown(self) -> None:
        config._current_model = self._model
        config._current_thinking_budget = self._budget
        config._current_standalone_model = self._standalone_model
        config._current_server_mode = self._mode
        config._current_server_port = self._port
        config._MLX_BASE_URL_OVERRIDE = self._override
        config._RUNTIME_SETTINGS_PATH = self._settings
        runtime._CONTROL_SETTINGS_PATH = self._control
        self._tmp.cleanup()

    def test_owned_server_requires_exact_th_launcher(self) -> None:
        owned = f"python {runtime._LAUNCHER} --model {config.DEFAULT_MODEL} --port 8085"
        generic = f"python -m mlx_vlm.server --model {config.DEFAULT_MODEL} --port 8085"
        self.assertTrue(runtime._is_owned_model_server(owned))
        self.assertFalse(runtime._is_owned_model_server(generic))

    def test_owned_standalone_uses_native_ar_without_draft_or_speculative_hook(self) -> None:
        with mock.patch.object(runtime, "_listener_pids", return_value=[]), \
             mock.patch.object(runtime, "_LOG_PATH", Path(self._tmp.name) / "native_ar.log"), \
             mock.patch.object(runtime.subprocess, "Popen") as spawn:
            runtime.start_model_server(config.DEFAULT_MODEL)
        argv = spawn.call_args.args[0]
        self.assertEqual(argv[argv.index("--model") + 1], config.DEFAULT_MODEL)
        self.assertIn("--enable-thinking", argv)
        self.assertFalse(any("draft" in arg.lower() or "speculative" in arg.lower() for arg in argv))
        launcher_source = Path(runtime._LAUNCHER).read_text(encoding="utf-8")
        self.assertNotIn("dflash_thinking_budget_patch", launcher_source)
        self.assertNotIn("_apply_dflash_budget", launcher_source)

    def test_owned_launcher_installs_registry_driven_apc_at_owner_boundary(self) -> None:
        source = Path(runtime._LAUNCHER).read_text(encoding="utf-8")
        self.assertIn("get_native_apc_contract(owner_model)", source)
        self.assertIn("apc_tool_call_template_patch", source)
        self.assertIn("apply_policy(apc_contract.template_policy)", source)
        for model in config.MODEL_CHOICES:
            self.assertNotIn(f'owner_model == {model!r}', source)

    def test_owned_launcher_installs_generic_apc_self_check_before_server_import(self) -> None:
        source = Path(runtime._LAUNCHER).read_text(encoding="utf-8")
        self.assertIn("apc_self_check_patch", source)
        self.assertIn("APC self-check compatibility seam", source)
        self.assertLess(
            source.index("apc_self_check_patch"),
            source.index("from mlx_vlm.server import main"),
        )

    def test_launcher_requires_mlx_vlm_0_7_10_or_newer(self) -> None:
        self.assertEqual(launcher.MIN_MLX_VLM, "0.7.10")
        for ok in ("0.7.10", "0.7.11", "0.8.0", "1.0.0", "0.7.10.post1", "0.7.10rc1"):
            with self.subTest(version=ok):
                self.assertEqual(launcher._require_supported_mlx_vlm(ok), ok)
        for old in ("0.6.8", "0.7.9", "0.7.0", "0.5.0"):
            with self.subTest(version=old):
                with self.assertRaisesRegex(RuntimeError, r"mlx-vlm >= 0\.7\.10"):
                    launcher._require_supported_mlx_vlm(old)
        for junk in ("", "latest", "7"):
            with self.subTest(version=junk):
                with self.assertRaisesRegex(RuntimeError, "version"):
                    launcher._require_supported_mlx_vlm(junk)

    def test_launcher_refuses_an_old_mlx_vlm_before_importing_the_server(self) -> None:
        source = Path(runtime._LAUNCHER).read_text(encoding="utf-8")
        main_body = source[source.index("def main()"):]
        self.assertLess(
            main_body.index("_require_supported_mlx_vlm()"),
            main_body.index("_install_owner_model_gate(owner_model)"),
        )

    def test_server_python_defaults_to_the_agent_interpreter(self) -> None:
        with mock.patch.dict(runtime.os.environ, {}, clear=False):
            runtime.os.environ.pop("V2_MODEL_SERVER_PYTHON", None)
            self.assertEqual(runtime._server_python(), sys.executable)

    def test_server_python_override_must_be_an_absolute_executable(self) -> None:
        with mock.patch.dict(runtime.os.environ, {"V2_MODEL_SERVER_PYTHON": sys.executable}):
            self.assertEqual(runtime._server_python(), sys.executable)
        for bad in ("python3", "/definitely/not/python", self._tmp.name):
            with self.subTest(value=bad):
                with mock.patch.dict(runtime.os.environ, {"V2_MODEL_SERVER_PYTHON": bad}):
                    with self.assertRaisesRegex(RuntimeError, "V2_MODEL_SERVER_PYTHON"):
                        runtime._server_python()

    def test_owned_server_is_started_with_the_configured_server_python(self) -> None:
        with mock.patch.dict(runtime.os.environ, {"V2_MODEL_SERVER_PYTHON": sys.executable}), \
             mock.patch.object(runtime, "_listener_pids", return_value=[]), \
             mock.patch.object(runtime, "_server_python", return_value="/opt/server/bin/python"), \
             mock.patch.object(runtime, "_LOG_PATH", Path(self._tmp.name) / "py.log"), \
             mock.patch.object(runtime.subprocess, "Popen") as spawn:
            runtime.start_model_server(config.DEFAULT_MODEL)
        self.assertEqual(spawn.call_args.args[0][0], "/opt/server/bin/python")

    def test_owned_standalone_launcher_gets_explicit_apc_policy(self) -> None:
        with mock.patch.dict(runtime.os.environ, {
            "APC_ENABLED": "0",
            "APC_EXACT_CACHE_ENTRIES": "9",
        }, clear=False):
            env = runtime._server_env(config.DEFAULT_MODEL)
        self.assertEqual(env["APC_ENABLED"], "1")
        self.assertEqual(env["APC_EXACT_CACHE_ENTRIES"], "2")

    def test_apc_disk_tier_is_always_off_for_the_owned_server(self) -> None:
        for inherited in ("1", "true", "0", None):
            with self.subTest(inherited=inherited):
                patch = {"APC_DISK_ENABLED": inherited} if inherited is not None else {}
                with mock.patch.dict(runtime.os.environ, patch, clear=False):
                    if inherited is None:
                        runtime.os.environ.pop("APC_DISK_ENABLED", None)
                    env = runtime._server_env(config.DEFAULT_MODEL)
                self.assertEqual(env["APC_DISK_ENABLED"], "0")

    def test_launcher_applies_the_disk_policy_before_the_server_is_imported(self) -> None:
        source = Path(runtime._LAUNCHER).read_text(encoding="utf-8")
        main_body = source[source.index("def main()"):]
        self.assertIn("apc_disk_policy", main_body)
        self.assertLess(
            main_body.index("apc_disk_policy"),
            main_body.index("_install_owner_model_gate(owner_model)"),
        )

    def test_apc_memory_cap_comes_only_from_the_model_contract(self) -> None:
        with mock.patch.dict(runtime.os.environ, {"APC_MEMORY_MAX_GB": "99"}, clear=False):
            capped = runtime._server_env("mlx-community/Qwen3.8-27B-4bit")
            uncapped = runtime._server_env(config.DEFAULT_MODEL)
        self.assertEqual(capped["APC_MEMORY_MAX_GB"], "6")
        # A stale value from the caller's environment must not leak into a model
        # that declares no cap.
        self.assertNotIn("APC_MEMORY_MAX_GB", uncapped)

    def test_shared_max_requires_max_project_patched_launcher(self) -> None:
        max_cmd = (
            "/opt/python /tmp/ENDEAVOR_LOCAL_AGENT_MAX_VLM/scripts/"
            "run_vlm_server_patched.py --model unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit --port 8085"
        )
        lookalike = "/opt/python /tmp/run_vlm_server_patched.py --model x --port 8085"
        self.assertTrue(runtime._is_max_shared_server(max_cmd))
        self.assertFalse(runtime._is_max_shared_server(lookalike))

    def test_shared_status_is_read_only_and_adopts_supported_model(self) -> None:
        config._current_server_mode = "shared_max"
        max_model = config.HIGH_QUALITY_MODEL
        max_cmd = (
            "/opt/python /tmp/ENDEAVOR_LOCAL_AGENT_MAX_VLM/scripts/"
            f"run_vlm_server_patched.py --model {max_model} --port 8085"
        )
        with mock.patch.object(runtime, "_listener_pids", return_value=[123]), \
             mock.patch.object(runtime, "_process_command", return_value=max_cmd), \
             mock.patch.object(runtime, "probe_model_health", return_value={
                 "status": "healthy", "loaded_model": max_model,
             }), \
             mock.patch.object(runtime.os, "kill") as kill:
            status = runtime.model_server_status()
        self.assertEqual(status["state"], "shared_ready")
        self.assertTrue(status["healthy"])
        self.assertFalse(status["managed_by_th"])
        self.assertEqual(config.get_model(), max_model)
        kill.assert_not_called()

    def test_shared_status_accepts_lightweight_2b_when_max_owns_it(self) -> None:
        config._current_server_mode = "shared_max"
        max_model = config.LIGHT_VLM_MODEL
        max_cmd = (
            "/opt/python /tmp/ENDEAVOR_LOCAL_AGENT_MAX_VLM/scripts/"
            f"run_vlm_server_patched.py --model {max_model} --port 8085"
        )
        with mock.patch.object(runtime, "_listener_pids", return_value=[123]), \
             mock.patch.object(runtime, "_process_command", return_value=max_cmd), \
             mock.patch.object(runtime, "probe_model_health", return_value={
                 "status": "healthy", "loaded_model": max_model,
             }):
            status = runtime.model_server_status()
        self.assertEqual(status["state"], "shared_ready")
        self.assertTrue(status["healthy"])
        self.assertEqual(config.get_model(), max_model)

    def test_shared_mode_refuses_start_stop_reset_and_watchdog(self) -> None:
        config._current_server_mode = "shared_max"
        with mock.patch.object(runtime, "_listener_pids") as listeners, \
             mock.patch.object(runtime.subprocess, "Popen") as popen, \
             mock.patch.object(runtime.os, "kill") as kill:
            for fn in (
                runtime.start_owner_model_server,
                runtime.stop_owner_model_server,
                runtime.restart_owner_model_server,
            ):
                with self.assertRaisesRegex(RuntimeError, "read-only"):
                    fn()
            with self.assertRaisesRegex(RuntimeError, "read-only"):
                runtime.set_model_server_watchdog(False)
        listeners.assert_not_called()
        popen.assert_not_called()
        kill.assert_not_called()

    def test_standalone_start_refuses_foreign_listener_instead_of_takeover(self) -> None:
        with mock.patch.object(runtime, "_listener_pids", return_value=[222]), \
             mock.patch.object(runtime, "_process_command", return_value="python other.py --port 8085"), \
             mock.patch.object(runtime, "stop_model_server") as stop, \
             mock.patch.object(runtime, "start_model_server") as start:
            with self.assertRaisesRegex(RuntimeError, "refusing to take over non-owner"):
                runtime._ensure_model_server_locked(config.DEFAULT_MODEL, 1.0)
        stop.assert_not_called()
        start.assert_not_called()

    def test_auto_attach_only_when_foreign_listener_is_verified_max(self) -> None:
        shared = {
            "healthy": True,
            "loaded_model": config.HIGH_QUALITY_MODEL,
            "state": "shared_ready",
        }
        with mock.patch.object(runtime, "model_server_status", return_value={"state": "foreign"}), \
             mock.patch.object(runtime, "shared_max_server_status", return_value=shared):
            result = runtime.auto_attach_max_test_server_if_present()
        self.assertIs(result, shared)
        self.assertEqual(config.get_server_mode(), "shared_max")
        self.assertEqual(config.get_model(), config.HIGH_QUALITY_MODEL)

    def test_auto_attach_does_not_trust_generic_foreign_listener(self) -> None:
        with mock.patch.object(runtime, "model_server_status", return_value={"state": "foreign"}), \
             mock.patch.object(runtime, "shared_max_server_status", return_value={
                 "healthy": False, "state": "foreign", "error": "not MAX",
             }):
            result = runtime.auto_attach_max_test_server_if_present()
        self.assertIsNone(result)
        self.assertEqual(config.get_server_mode(), "standalone")

    def test_owned_launcher_rejects_request_model_switch(self) -> None:
        package = types.ModuleType("mlx_vlm")
        package.__path__ = []
        server = types.ModuleType("mlx_vlm.server")
        calls = []

        def original(model_path, *args, **kwargs):
            calls.append(model_path)
            return "ok"

        server.get_cached_model = original
        with mock.patch.dict(sys.modules, {"mlx_vlm": package, "mlx_vlm.server": server}):
            launcher._install_owner_model_gate(config.DEFAULT_MODEL)
            self.assertEqual(server.get_cached_model(config.DEFAULT_MODEL), "ok")
            with self.assertRaisesRegex(ValueError, "model switch denied"):
                server.get_cached_model(config.HIGH_QUALITY_MODEL)
        self.assertEqual(calls, [config.DEFAULT_MODEL])

    def test_owner_control_has_no_model_argument(self) -> None:
        with mock.patch.object(runtime, "start_owner_model_server") as start:
            self.assertEqual(runtime._control_main(["start", "other/model"]), 2)
        start.assert_not_called()


if __name__ == "__main__":
    unittest.main()
