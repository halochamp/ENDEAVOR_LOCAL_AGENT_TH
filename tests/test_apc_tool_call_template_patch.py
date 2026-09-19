from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import patch

from scripts import apc_tool_call_template_patch as patcher


class APCQwen36TemplatePatchTests(unittest.TestCase):
    def _fake_prompt_utils(self, get_chat_template):
        mlx = types.ModuleType("mlx_vlm")
        prompt_utils = types.ModuleType("mlx_vlm.prompt_utils")
        prompt_utils.get_chat_template = get_chat_template
        mlx.prompt_utils = prompt_utils
        return mlx, prompt_utils

    def test_exact_one_condition_patch_preserves_tool_call_wrapper(self):
        captured = {}

        def original(processor, messages, add_generation_prompt, tokenize=False, **kwargs):
            captured.update(kwargs)
            return "rendered"

        mlx, prompt_utils = self._fake_prompt_utils(original)
        processor = types.SimpleNamespace(chat_template=patcher._UPSTREAM_CONDITION)
        with patch.dict(sys.modules, {
            "mlx_vlm": mlx,
            "mlx_vlm.prompt_utils": prompt_utils,
        }):
            self.assertTrue(
                patcher.apply_for_model("unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit")
            )
            self.assertEqual(
                prompt_utils.get_chat_template(processor, [], True),
                "rendered",
            )
        self.assertIn("message.tool_calls", captured["chat_template"])
        self.assertEqual(captured["chat_template"], patcher._PATCHED_CONDITION)

    def test_already_patched_is_idempotent(self):
        calls = []

        def original(processor, messages, add_generation_prompt, tokenize=False, **kwargs):
            calls.append(kwargs)
            return "ok"

        mlx, prompt_utils = self._fake_prompt_utils(original)
        processor = types.SimpleNamespace(chat_template=patcher._PATCHED_CONDITION)
        with patch.dict(sys.modules, {
            "mlx_vlm": mlx,
            "mlx_vlm.prompt_utils": prompt_utils,
        }):
            self.assertTrue(
                patcher.apply_for_model("unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit")
            )
            first = prompt_utils.get_chat_template
            self.assertTrue(
                patcher.apply_for_model("unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit")
            )
            self.assertIs(prompt_utils.get_chat_template, first)
            self.assertEqual(prompt_utils.get_chat_template(processor, [], True), "ok")
        self.assertEqual(len(calls), 1)

    def test_incompatible_template_refuses_without_guessing(self):
        def original(processor, messages, add_generation_prompt, tokenize=False, **kwargs):
            return "should not run"

        mlx, prompt_utils = self._fake_prompt_utils(original)
        processor = types.SimpleNamespace(chat_template="ambiguous preserve_thinking template")
        with patch.dict(sys.modules, {
            "mlx_vlm": mlx,
            "mlx_vlm.prompt_utils": prompt_utils,
        }), self.assertRaises(patcher.TemplatePatchError):
            patcher.apply_for_model("unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit")
            prompt_utils.get_chat_template(processor, [], True)

    def test_missing_or_incompatible_prompt_utils_seam_fails_closed(self):
        def no_kwargs(processor, messages, add_generation_prompt, tokenize=False):
            return "ok"

        mlx, prompt_utils = self._fake_prompt_utils(no_kwargs)
        with patch.dict(sys.modules, {
            "mlx_vlm": mlx,
            "mlx_vlm.prompt_utils": prompt_utils,
        }):
            self.assertFalse(
                patcher.apply_for_model("unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit")
            )

    def test_model_gate_leaves_other_models_untouched(self):
        def original(processor, messages, add_generation_prompt, tokenize=False, **kwargs):
            return "ok"

        mlx, prompt_utils = self._fake_prompt_utils(original)
        with patch.dict(sys.modules, {
            "mlx_vlm": mlx,
            "mlx_vlm.prompt_utils": prompt_utils,
        }):
            self.assertTrue(patcher.apply_for_model("Qwen/Qwen3-14B-MLX-4bit"))
        self.assertIs(prompt_utils.get_chat_template, original)


if __name__ == "__main__":
    unittest.main()
