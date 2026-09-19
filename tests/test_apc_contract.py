from __future__ import annotations

import unittest
import sys
import types
from unittest.mock import patch

import llm
from scripts.apc_extra_hash_patch import (
    _mask_background_meta,
    _apc_hash_kwargs,
    is_background_tenant,
)


class APCContractTests(unittest.TestCase):
    def test_main_and_background_clients_use_distinct_stable_tenants(self) -> None:
        captured = []

        class FakeClient:
            def __init__(self, **kwargs):
                captured.append(kwargs)

        with patch.object(llm, "VisionFallbackChatOpenAI", FakeClient):
            llm.build_llm(max_tokens=2)
            llm.build_llm(apc_cache=False, max_tokens=2)
        self.assertEqual(captured[0]["default_headers"]["X-APC-Tenant"], llm.APC_TENANT)
        self.assertEqual(captured[1]["default_headers"]["X-APC-Tenant"], llm.APC_BACKGROUND_TENANT)
        self.assertNotEqual(
            captured[0]["default_headers"]["X-APC-Tenant"],
            captured[1]["default_headers"]["X-APC-Tenant"],
        )

    def test_background_tenant_is_bypass_namespace(self) -> None:
        self.assertTrue(is_background_tenant("endeavor-th-background"))
        self.assertTrue(is_background_tenant("endeavor-apc-bypass:helper"))
        self.assertFalse(is_background_tenant("endeavor-th-main"))

    def test_mixed_batch_masks_only_background_meta(self) -> None:
        main_meta = {"full_input_ids": [1, 2], "prefix_len": 2}
        helper_meta = {"full_input_ids": [3, 4], "prefix_len": 2}
        result = _mask_background_meta(
            [main_meta, helper_meta],
            [
                {"_apc_tenant": "endeavor-th-main"},
                {"_apc_tenant": "endeavor-th-background"},
            ],
        )
        self.assertIs(result[0], main_meta)
        self.assertIsNone(result[1])

    def test_hash_sanitization_preserves_media_semantics_and_masks_only_redundant_shape(self) -> None:
        class Array:
            def __init__(self, values):
                self._values = values

            def tolist(self):
                return self._values

        clean = _apc_hash_kwargs({
            "inputs_embeds": object(),
            "attention_mask": Array([[1, 1]]),
            "_apc_tenant": "endeavor-th-main",
        })
        self.assertNotIn("inputs_embeds", clean)
        self.assertNotIn("attention_mask", clean)
        self.assertEqual(clean["_apc_tenant"], "endeavor-th-main")

        image = object()
        image_kwargs = {
            "inputs_embeds": object(),
            "pixel_values": image,
            "attention_mask": Array([[1, 1]]),
        }
        image_result = _apc_hash_kwargs(image_kwargs)
        self.assertNotIn("inputs_embeds", image_result)
        self.assertIs(image_result["pixel_values"], image)
        self.assertNotIn("attention_mask", image_result)

        for media_key in ("input_features", "pixel_values_videos"):
            audio_or_video = object()
            preserved = _apc_hash_kwargs({
                "inputs_embeds": "keep",
                media_key: audio_or_video,
                "attention_mask": Array([[1, 1]]),
            })
            self.assertEqual(preserved["inputs_embeds"], "keep")
            self.assertIs(preserved[media_key], audio_or_video)
            self.assertNotIn("attention_mask", preserved)

        unusual = _apc_hash_kwargs({
            "inputs_embeds": object(),
            "pixel_values": image,
            "attention_mask": Array([[1, 0]]),
        })
        self.assertNotIn("inputs_embeds", unusual)
        self.assertIn("attention_mask", unusual)

    def test_install_patch_preserves_main_row_in_real_mixed_seam(self) -> None:
        class FakeBatchGenerator:
            def _apc_extra_hash(self, kwargs):
                return kwargs

            def _apc_pick_for(self, sequence):
                return "main-pick"

            def _build_apc_meta_for_cold(self, ids, kwargs):
                return [{"row": i} for i in range(len(kwargs))]

            def _build_mixed_prompt_batch(self, sequences):
                return types.SimpleNamespace(_apc_meta=[{"row": 0}, {"row": 1}])

        mlx = types.ModuleType("mlx_vlm")
        generate = types.ModuleType("mlx_vlm.generate")
        ar = types.ModuleType("mlx_vlm.generate.ar")
        ar.BatchGenerator = FakeBatchGenerator
        with patch.dict(sys.modules, {
            "mlx_vlm": mlx,
            "mlx_vlm.generate": generate,
            "mlx_vlm.generate.ar": ar,
        }):
            from scripts.apc_extra_hash_patch import apply
            self.assertTrue(apply())
            instance = FakeBatchGenerator()
            self.assertEqual(
                instance._apc_pick_for((1, [1], 1, {"_apc_tenant": "endeavor-th-main"})),
                "main-pick",
            )
            self.assertIsNone(
                instance._apc_pick_for((2, [2], 1, {"_apc_tenant": "endeavor-th-background"}))
            )
            batch = instance._build_mixed_prompt_batch([
                (1, [1], 1, {"_apc_tenant": "endeavor-th-main"}),
                (2, [2], 1, {"_apc_tenant": "endeavor-th-background"}),
            ])
            self.assertIsNotNone(batch._apc_meta[0])
            self.assertIsNone(batch._apc_meta[1])
            cold = instance._build_apc_meta_for_cold(
                [[1], [2]],
                [
                    {"_apc_tenant": "endeavor-th-main"},
                    {"_apc_tenant": "endeavor-th-background"},
                ],
            )
            self.assertIsNotNone(cold[0])
            self.assertIsNone(cold[1])

    def test_install_fails_closed_when_any_policy_seam_is_missing(self) -> None:
        class IncompleteBatchGenerator:
            def _apc_extra_hash(self, kwargs):
                return kwargs

            def _apc_pick_for(self, sequence):
                return "main-pick"

            def _build_apc_meta_for_cold(self, ids, kwargs):
                return []

        mlx = types.ModuleType("mlx_vlm")
        generate = types.ModuleType("mlx_vlm.generate")
        ar = types.ModuleType("mlx_vlm.generate.ar")
        ar.BatchGenerator = IncompleteBatchGenerator
        with patch.dict(sys.modules, {
            "mlx_vlm": mlx,
            "mlx_vlm.generate": generate,
            "mlx_vlm.generate.ar": ar,
        }), patch("scripts.apc_extra_hash_patch._MARKER", "_test_marker"):
            from scripts.apc_extra_hash_patch import apply
            self.assertFalse(apply())
            self.assertFalse(hasattr(IncompleteBatchGenerator, "_test_marker"))


if __name__ == "__main__":
    unittest.main()
