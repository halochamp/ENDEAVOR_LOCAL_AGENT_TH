from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import patch

from scripts import apc_self_check_patch as patcher


class APCSelfCheckPatchTests(unittest.TestCase):
    def _fake_modules(self, *, make_cache=True, validate=True, mode=True):
        mlx = types.ModuleType("mlx_vlm")
        apc = types.ModuleType("mlx_vlm.apc")
        generate = types.ModuleType("mlx_vlm.generate")
        ar = types.ModuleType("mlx_vlm.generate.ar")
        calls = []

        def upstream(model, *, kv_bits=None, log=True):
            calls.append(("upstream", model, kv_bits, log))
            return "upstream"

        apc.self_check_model_apc = upstream
        if validate:
            def validate_layout(caches, *, apc_mode):
                calls.append(("validate", caches, apc_mode))
                return types.SimpleNamespace(ok=True, notes=[])
            apc.validate_prompt_cache_layout = validate_layout
        if mode:
            apc.model_apc_mode = lambda language_model: "block"
        if make_cache:
            def runtime_make_cache(model, left_padding, *, kv_bits=None, prefill_length=None):
                calls.append(("make_cache", model, left_padding, kv_bits, prefill_length))
                return ["cache"]
            ar._make_cache = runtime_make_cache
        mlx.apc = apc
        generate.ar = ar
        mlx.generate = generate
        return mlx, apc, generate, ar, calls

    def test_no_make_cache_uses_runtime_fallback_layout(self):
        mlx, apc, generate, ar, calls = self._fake_modules()
        language_model = types.SimpleNamespace(layers=[object(), object()])
        model = types.SimpleNamespace(language_model=language_model)
        with patch.dict(sys.modules, {
            "mlx_vlm": mlx,
            "mlx_vlm.apc": apc,
            "mlx_vlm.generate": generate,
            "mlx_vlm.generate.ar": ar,
        }):
            self.assertTrue(patcher.apply())
            result = apc.self_check_model_apc(model, kv_bits=4, log=False)
        self.assertTrue(result.ok)
        self.assertIn("no-make_cache fallback", " ".join(result.notes))
        self.assertEqual(calls[0][0], "make_cache")
        self.assertEqual(calls[0][2:], ([0], 4, 0))
        self.assertEqual(calls[1][0], "validate")

    def test_existing_make_cache_delegates_untouched(self):
        mlx, apc, generate, ar, calls = self._fake_modules()
        language_model = types.SimpleNamespace(
            layers=[object()],
            make_cache=lambda *args, **kwargs: ["native"],
        )
        model = types.SimpleNamespace(language_model=language_model)
        with patch.dict(sys.modules, {
            "mlx_vlm": mlx,
            "mlx_vlm.apc": apc,
            "mlx_vlm.generate": generate,
            "mlx_vlm.generate.ar": ar,
        }):
            self.assertTrue(patcher.apply())
            self.assertEqual(apc.self_check_model_apc(model), "upstream")
        self.assertEqual([call[0] for call in calls], ["upstream"])

    def test_missing_required_seam_fails_closed(self):
        mlx, apc, generate, ar, calls = self._fake_modules(make_cache=False)
        with patch.dict(sys.modules, {
            "mlx_vlm": mlx,
            "mlx_vlm.apc": apc,
            "mlx_vlm.generate": generate,
            "mlx_vlm.generate.ar": ar,
        }):
            self.assertFalse(patcher.apply())
        self.assertEqual(calls, [])

    def test_apply_is_idempotent(self):
        mlx, apc, generate, ar, calls = self._fake_modules()
        with patch.dict(sys.modules, {
            "mlx_vlm": mlx,
            "mlx_vlm.apc": apc,
            "mlx_vlm.generate": generate,
            "mlx_vlm.generate.ar": ar,
        }):
            self.assertTrue(patcher.apply())
            first = apc.self_check_model_apc
            self.assertTrue(patcher.apply())
            self.assertIs(apc.self_check_model_apc, first)


if __name__ == "__main__":
    unittest.main()
