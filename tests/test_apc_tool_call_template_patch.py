from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import patch

import config
from scripts import apc_tool_call_template_patch as patcher


class APCTemplatePatchTests(unittest.TestCase):
    def _fake_prompt_utils(self, get_chat_template):
        mlx = types.ModuleType("mlx_vlm")
        prompt_utils = types.ModuleType("mlx_vlm.prompt_utils")
        prompt_utils.get_chat_template = get_chat_template
        mlx.prompt_utils = prompt_utils
        return mlx, prompt_utils

    def _exercise(self, model, template):
        captured = {}

        def original(processor, messages, add_generation_prompt, tokenize=False, **kwargs):
            captured.update(kwargs)
            return "rendered"

        mlx, prompt_utils = self._fake_prompt_utils(original)
        processor = types.SimpleNamespace(chat_template=template)
        with patch.dict(sys.modules, {
            "mlx_vlm": mlx,
            "mlx_vlm.prompt_utils": prompt_utils,
        }):
            self.assertTrue(patcher.apply_for_model(model))
            self.assertEqual(prompt_utils.get_chat_template(processor, [], True), "rendered")
        return captured["chat_template"]

    def test_declarative_mapping_covers_all_public_models(self):
        self.assertEqual(set(patcher.MODEL_TEMPLATE_POLICIES), set(config.MODEL_CHOICES))
        self.assertEqual(
            patcher.MODEL_TEMPLATE_POLICIES,
            {
                config.HIGH_QUALITY_MODEL: "qwen36_tool_call",
                config.COMPACT_VLM_MODEL: "qwen35_tool_call",
                config.LIGHT_VLM_MODEL: "qwen35_tool_call",
                config.DEFAULT_MODEL: "qwen3_tool_call",
            },
        )

    def test_qwen36_exact_one_condition_patch(self):
        rendered = self._exercise(
            config.HIGH_QUALITY_MODEL,
            patcher._UPSTREAM_CONDITION,
        )
        self.assertEqual(rendered, patcher._PATCHED_CONDITION)

    def test_qwen35_policy_patches_outer_position_guard_for_9b_and_2b(self):
        template = (
            patcher._QWEN35_POSITION_UPSTREAM_TAG
            + "\n"
            + patcher._QWEN35_TOOL_PREDICATE
        )
        for model in (config.COMPACT_VLM_MODEL, config.LIGHT_VLM_MODEL):
            with self.subTest(model=model):
                rendered = self._exercise(model, template)
                self.assertIn(patcher._QWEN35_POSITION_PATCHED_TAG, rendered)
                self.assertNotIn(patcher._QWEN35_POSITION_UPSTREAM_TAG, rendered)
                self.assertIn(patcher._QWEN35_TOOL_PREDICATE, rendered)

    def test_qwen3_policy_patches_outer_and_nested_reasoning_guards(self):
        template = "\n".join((
            patcher._QWEN35_POSITION_UPSTREAM_TAG,
            patcher._QWEN3_TOOL_TAG,
            patcher._QWEN3_NESTED_UPSTREAM_TAG,
        ))
        rendered = self._exercise(config.DEFAULT_MODEL, template)
        self.assertIn(patcher._QWEN35_POSITION_PATCHED_TAG, rendered)
        self.assertIn(patcher._QWEN3_NESTED_PATCHED_TAG, rendered)
        self.assertNotIn(patcher._QWEN3_NESTED_UPSTREAM_TAG, rendered)

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
            self.assertTrue(patcher.apply_for_model(config.HIGH_QUALITY_MODEL))
            first = prompt_utils.get_chat_template
            self.assertTrue(patcher.apply_for_model(config.HIGH_QUALITY_MODEL))
            self.assertIs(prompt_utils.get_chat_template, first)
            self.assertEqual(prompt_utils.get_chat_template(processor, [], True), "ok")
        self.assertEqual(len(calls), 1)

    def test_malformed_qwen35_template_fails_closed_at_lazy_validation(self):
        def original(processor, messages, add_generation_prompt, tokenize=False, **kwargs):
            return "should not run"

        mlx, prompt_utils = self._fake_prompt_utils(original)
        processor = types.SimpleNamespace(chat_template=patcher._QWEN35_TOOL_PREDICATE)
        with patch.dict(sys.modules, {
            "mlx_vlm": mlx,
            "mlx_vlm.prompt_utils": prompt_utils,
        }), self.assertRaises(patcher.TemplatePatchError):
            self.assertTrue(patcher.apply_for_model(config.COMPACT_VLM_MODEL))
            prompt_utils.get_chat_template(processor, [], True)

    def test_malformed_qwen3_template_fails_closed_at_lazy_validation(self):
        def original(processor, messages, add_generation_prompt, tokenize=False, **kwargs):
            return "should not run"

        mlx, prompt_utils = self._fake_prompt_utils(original)
        processor = types.SimpleNamespace(chat_template=patcher._QWEN3_TOOL_TAG)
        with patch.dict(sys.modules, {
            "mlx_vlm": mlx,
            "mlx_vlm.prompt_utils": prompt_utils,
        }), self.assertRaises(patcher.TemplatePatchError):
            self.assertTrue(patcher.apply_for_model(config.DEFAULT_MODEL))
            prompt_utils.get_chat_template(processor, [], True)

    def test_explicit_custom_template_is_not_rewritten(self):
        captured = {}

        def original(processor, messages, add_generation_prompt, tokenize=False, **kwargs):
            captured.update(kwargs)
            return "ok"

        mlx, prompt_utils = self._fake_prompt_utils(original)
        processor = types.SimpleNamespace(chat_template="malformed owner template")
        with patch.dict(sys.modules, {
            "mlx_vlm": mlx,
            "mlx_vlm.prompt_utils": prompt_utils,
        }):
            self.assertTrue(patcher.apply_for_model(config.COMPACT_VLM_MODEL))
            self.assertEqual(
                prompt_utils.get_chat_template(processor, [], True, chat_template="custom"),
                "ok",
            )
        self.assertEqual(captured["chat_template"], "custom")

    def test_missing_or_incompatible_prompt_utils_seam_fails_closed(self):
        def no_kwargs(processor, messages, add_generation_prompt, tokenize=False):
            return "ok"

        mlx, prompt_utils = self._fake_prompt_utils(no_kwargs)
        with patch.dict(sys.modules, {
            "mlx_vlm": mlx,
            "mlx_vlm.prompt_utils": prompt_utils,
        }):
            self.assertFalse(patcher.apply_for_model(config.DEFAULT_MODEL))

    def test_non_target_model_is_a_noop(self):
        def original(processor, messages, add_generation_prompt, tokenize=False, **kwargs):
            return "ok"

        mlx, prompt_utils = self._fake_prompt_utils(original)
        with patch.dict(sys.modules, {
            "mlx_vlm": mlx,
            "mlx_vlm.prompt_utils": prompt_utils,
        }):
            self.assertTrue(patcher.apply_for_model("public/unsupported-model"))
        self.assertIs(prompt_utils.get_chat_template, original)


if __name__ == "__main__":
    unittest.main()
