from __future__ import annotations

import contextlib
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("AGENT_SERVER_TOKEN", "test-cli-parity-token")

import agent_server
import endeavor_agent as cli
from runtime_common import (
    PINNED_FILE_MAX,
    builtin_commands_for_surface,
    canonical_attachment_paths,
    canonical_pinned_file_paths,
    command_completion_tokens,
    load_skill_registry,
    parse_builtin_command,
    prepare_file_context,
)
import ui_cli


class CliGenericParityTests(unittest.TestCase):
    def test_shared_registry_filters_cli_and_electron_surfaces(self) -> None:
        cli_registry = load_skill_registry("cli")
        electron_commands = agent_server._load_builtin_cmds()
        cli_names = {item["name"] for item in cli_registry["builtin_cmds"]}
        electron_names = {item["name"] for item in electron_commands}

        self.assertEqual(cli_registry["builtin_cmds"], builtin_commands_for_surface("cli"))
        self.assertEqual(electron_commands, builtin_commands_for_surface("electron"))
        self.assertTrue({"help", "runtime", "attach", "pin", "pdf_text"}.issubset(cli_names))
        self.assertTrue({"clear", "compact", "history", "exit"}.issubset(electron_names))
        self.assertTrue({"help", "runtime", "attach", "pin", "pdf_text"}.isdisjoint(electron_names))
        self.assertFalse({"goal", "telegram", "send_file"} & cli_names)
        self.assertTrue({"/help", "/runtime", "/attach", "/pin", "/pdf_text"}.issubset(
            set(command_completion_tokens("cli"))
        ))

    def test_registry_command_parser_handles_aliases_paths_and_unknown_skills(self) -> None:
        parsed = parse_builtin_command('/pin add "/tmp/a report.md" /tmp/b.txt')
        self.assertEqual(parsed["name"], "pin")
        self.assertEqual(parsed["args"], ["add", "/tmp/a report.md", "/tmp/b.txt"])
        self.assertEqual(parse_builtin_command("load history")["name"], "history")
        self.assertEqual(parse_builtin_command("menu")["name"], "menu")
        self.assertIsNone(parse_builtin_command("/research"))
        malformed = parse_builtin_command('/attach add "unfinished path')
        self.assertEqual(malformed["name"], "attach")
        self.assertTrue(malformed["error"])

    def test_help_uses_current_registry_and_available_skill_metadata(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            ui_cli.print_cli_help(
                load_skill_registry("cli")["builtin_cmds"],
                [{"name": "research", "description": "ค้นคว้าข้อมูล"}],
            )
        rendered = output.getvalue()
        for command in ("/help", "/runtime", "/attach", "/pin", "/pdf_text", "/clear", "/history"):
            self.assertIn(command, rendered)
        self.assertIn("/research", rendered)
        self.assertNotIn("/goal", rendered)
        self.assertNotIn("/telegram", rendered)

    def test_runtime_snapshot_is_shared_max_read_only(self) -> None:
        settings = {
            "model": "example/model", "thinking_budget": 1024,
            "server_mode": "shared_max", "server_port": 8085,
        }
        status = {
            "owner": "agent_max_vlm", "mode": "shared_max", "port": 8085,
            "healthy": True, "running": True, "state": "shared_ready",
            "loaded_model": "example/model", "managed_by_th": False,
        }
        output = io.StringIO()
        with (
            patch.object(cli.config, "get_runtime_settings", return_value=settings),
            patch.object(cli, "shared_max_server_status", return_value=status) as shared_status,
            patch.object(cli, "get_model_server_control_settings", return_value={
                "watchdog_enabled": True, "desired_state": "running",
            }),
            patch.object(cli.config, "set_runtime_settings", side_effect=AssertionError("write")),
            patch.object(cli, "model_server_status", side_effect=AssertionError("wrong status path")),
            patch.object(cli, "reconcile_runtime_server", side_effect=AssertionError("lifecycle")),
            patch.object(cli, "start_owner_model_server", side_effect=AssertionError("lifecycle")),
            patch.object(cli, "restart_owner_model_server", side_effect=AssertionError("lifecycle")),
            patch.object(cli, "_stop_owned_model_server", side_effect=AssertionError("lifecycle")),
            contextlib.redirect_stdout(output),
        ):
            snapshot = cli._cli_runtime_snapshot()
            ui_cli.print_runtime_status(*snapshot)

        shared_status.assert_called_once_with(port=8085)
        self.assertEqual(snapshot[0]["server_mode"], "shared_max")
        self.assertIn("Shared MAX (read-only)", output.getvalue())
        self.assertIn("owner: agent_max_vlm", output.getvalue())
        self.assertIn("loaded model: example/model", output.getvalue())
        self.assertIn("Watchdog: disabled", output.getvalue())

    def test_shared_file_context_canonicalizes_and_pin_wins_over_attach(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "report one.txt"
            source.write_text("reference text", encoding="utf-8")
            alias = Path(temp_dir) / "report alias.txt"
            alias.symlink_to(source)
            real = str(source.resolve())

            prepared = prepare_file_context("Summarize the report", [str(alias)], [real])
            self.assertEqual(prepared["pinned_paths"], [real])
            self.assertEqual(prepared["effective_attachments"], [])
            self.assertEqual(prepared["content"], "Summarize the report")
            self.assertEqual(prepared["turn_context"]["pinned_files"], [real])
            self.assertEqual(prepared["turn_context"]["pinned_user_query"], "Summarize the report")

    def test_shared_file_policy_rejects_protected_paths_and_bounds_pin_attach(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = []
            for index in range(PINNED_FILE_MAX + 1):
                path = Path(temp_dir) / f"pin-{index}.txt"
                path.write_text(str(index), encoding="utf-8")
                paths.append(str(path))

            valid, failures = canonical_pinned_file_paths([paths[0], paths[0]])
            self.assertEqual(valid, [str(Path(paths[0]).resolve())])
            self.assertEqual(failures, [])
            with self.assertRaises(ValueError):
                canonical_pinned_file_paths(paths)
            rejected, failures = canonical_pinned_file_paths(["/etc/passwd"])
            self.assertEqual(rejected, [])
            self.assertEqual(len(failures), 1)
            with self.assertRaises(ValueError):
                canonical_attachment_paths(["/etc/passwd"])
            with self.assertRaises(ValueError):
                canonical_attachment_paths(paths[:2])

    def test_cli_attach_and_pin_actions_dedupe_clear_remove_and_enforce_limits(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            first = Path(temp_dir) / "first report.txt"
            second = Path(temp_dir) / "second.txt"
            first.write_text("first", encoding="utf-8")
            second.write_text("second", encoding="utf-8")
            real_first = str(first.resolve())
            real_second = str(second.resolve())

            attached = cli._apply_cli_attachment_action([], "add", [str(first)])
            self.assertEqual(attached, [real_first])
            self.assertEqual(cli._apply_cli_attachment_action(attached, "add", [real_first]), attached)
            with self.assertRaises(ValueError):
                cli._apply_cli_attachment_action(attached, "add", [str(second)])
            self.assertEqual(cli._apply_cli_attachment_action(attached, "clear"), [])

            pins = cli._apply_cli_pin_action([], "add", [str(first), str(second), str(first)])
            self.assertEqual(pins, [real_first, real_second])
            pins = cli._apply_cli_pin_action(pins, "remove", ["1"])
            self.assertEqual(pins, [real_second])
            removed = Path(real_second)
            removed.unlink()
            pins = cli._apply_cli_pin_action(pins, "remove", [str(removed)])
            self.assertEqual(pins, [])
            self.assertEqual(cli._apply_cli_pin_action([real_first], "clear"), [])

            too_many = []
            for index in range(PINNED_FILE_MAX + 1):
                path = Path(temp_dir) / f"limit-{index}.txt"
                path.write_text("x", encoding="utf-8")
                too_many.append(str(path))
            with self.assertRaises(ValueError):
                cli._apply_cli_pin_action([], "add", too_many)

    def test_turn_helper_clears_attachment_and_keeps_pins_even_after_runner_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            pin_path = Path(temp_dir) / "pin.txt"
            attachment_path = Path(temp_dir) / "attached.txt"
            pin_path.write_text("pinned", encoding="utf-8")
            attachment_path.write_text("attached", encoding="utf-8")
            pins = [str(pin_path.resolve())]
            attachments = [str(attachment_path.resolve())]
            seen = {}

            def failing_runner(_app, query, cfg, **kwargs):
                seen["query"] = query
                seen["cfg"] = cfg
                seen["kwargs"] = kwargs
                raise RuntimeError("synthetic turn failure")

            with self.assertRaisesRegex(RuntimeError, "synthetic turn failure"):
                cli._run_cli_turn_with_file_state(
                    failing_runner, object(), "question", {"configurable": {"thread_id": "t"}},
                    thread_id="t", saver=None, db_conn=None, logger=None,
                    pins=pins, attachments=attachments, system_prompt="system",
                )
            self.assertEqual(attachments, [])
            self.assertEqual(pins, [str(pin_path.resolve())])
            self.assertEqual(seen["cfg"]["configurable"]["pinned_files"], pins)
            self.assertEqual(seen["cfg"]["configurable"]["pinned_user_query"], "question")
            self.assertIn(str(attachment_path.resolve()), seen["query"])

    def test_cli_pdf_text_calls_canonical_pipeline_and_maps_optional_rewrite(self) -> None:
        import pdf_to_text

        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "thai report.pdf"
            source.write_bytes(b"fixture")
            with patch.object(pdf_to_text, "convert_pdf", return_value={"total_pages": 1}) as convert:
                result = cli._run_cli_pdf_text([str(source)])
                self.assertEqual(result["total_pages"], 1)
                convert.assert_called_once_with(
                    str(source.resolve()), source_name=source.name, rewrite_thai=False,
                )
                cli._run_cli_pdf_text([str(source), "--rewrite-thai"])
                self.assertEqual(convert.call_count, 2)
                self.assertTrue(convert.call_args.kwargs["rewrite_thai"])
                with self.assertRaises((OSError, ValueError)):
                    cli._run_cli_pdf_text(["/etc/passwd"])
                with self.assertRaises(ValueError):
                    cli._run_cli_pdf_text([str(source), "--rewrite-thai", "--rewrite-thai"])
                self.assertEqual(convert.call_count, 2)

    def test_cli_pdf_text_dispatch_handles_operational_runtime_error(self) -> None:
        output = io.StringIO()
        with patch.object(
            cli, "_run_cli_pdf_text", side_effect=RuntimeError("invalid page dimensions")
        ), contextlib.redirect_stdout(output):
            cli._dispatch_cli_pdf_text(["fixture.pdf"])

        self.assertIn("PDF → Text ไม่สำเร็จ: invalid page dimensions", output.getvalue())


if __name__ == "__main__":
    unittest.main(verbosity=2)
