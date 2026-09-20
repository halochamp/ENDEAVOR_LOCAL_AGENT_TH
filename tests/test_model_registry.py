from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import config
from model_registry import (
    ModelRegistryError,
    REGISTRY,
    SUPPORTED_APC_TEMPLATE_POLICIES,
    load_registry,
)


class ModelRegistryTests(unittest.TestCase):
    def _load_payload(self, payload: dict):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "models.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            return load_registry(path)

    def test_current_public_models_are_registry_driven(self):
        self.assertEqual(REGISTRY.default_model, config.DEFAULT_MODEL)
        self.assertEqual(
            tuple(spec.repo_id for spec in REGISTRY.selectable_models),
            config.MODEL_CHOICES,
        )
        self.assertEqual(
            {spec.repo_id: spec.label for spec in REGISTRY.selectable_models},
            config.MODEL_LABELS,
        )
        for spec in REGISTRY.selectable_models:
            self.assertTrue(spec.native_apc.enabled)
            self.assertEqual(spec.native_apc.exact_cache_entries, 2)
            self.assertIn(
                spec.native_apc.template_policy,
                SUPPORTED_APC_TEMPLATE_POLICIES,
            )

    def test_future_native_model_requires_only_registry_metadata(self):
        payload = {
            "schema_version": 1,
            "default_model": "future/model",
            "models": [
                {
                    "repo_id": "future/model",
                    "label": "Future native model",
                    "role": "default",
                    "selectable": True,
                    "ram_warning_below_gib": None,
                    "native_apc": {
                        "enabled": True,
                        "exact_cache_entries": 2,
                        "template_policy": "native_preserve",
                    },
                }
            ],
        }
        registry = self._load_payload(payload)
        spec = registry.get("future/model")
        self.assertEqual(spec.native_apc.template_policy, "native_preserve")
        self.assertEqual(spec.native_apc.exact_cache_entries, 2)
        self.assertTrue(spec.native_apc.enabled)

    def test_apc_runtime_has_no_repo_id_branches(self):
        root = Path(__file__).resolve().parents[1]
        launcher = (root / "scripts" / "run_mlx_server_owned.py").read_text(encoding="utf-8")
        template_patch = (root / "scripts" / "apc_tool_call_template_patch.py").read_text(encoding="utf-8")
        for spec in REGISTRY.selectable_models:
            self.assertNotIn(spec.repo_id, launcher)
            self.assertNotIn(spec.repo_id, template_patch)
        self.assertIn("get_native_apc_contract(owner_model)", launcher)
        self.assertIn("apply_policy(apc_contract.template_policy)", launcher)

    def test_native_apc_contract_fails_closed_on_wrong_capacity(self):
        payload = {
            "schema_version": 1,
            "default_model": "future/model",
            "models": [
                {
                    "repo_id": "future/model",
                    "label": "Future native model",
                    "role": "default",
                    "selectable": True,
                    "ram_warning_below_gib": None,
                    "native_apc": {
                        "enabled": True,
                        "exact_cache_entries": 8,
                        "template_policy": "native_preserve",
                    },
                }
            ],
        }
        with self.assertRaisesRegex(ModelRegistryError, "exact_cache_entries=2"):
            self._load_payload(payload)

    def test_native_apc_contract_fails_closed_on_unknown_template_policy(self):
        payload = {
            "schema_version": 1,
            "default_model": "future/model",
            "models": [
                {
                    "repo_id": "future/model",
                    "label": "Future native model",
                    "role": "default",
                    "selectable": True,
                    "ram_warning_below_gib": None,
                    "native_apc": {
                        "enabled": True,
                        "exact_cache_entries": 2,
                        "template_policy": "guess_the_template",
                    },
                }
            ],
        }
        with self.assertRaisesRegex(ModelRegistryError, "template_policy"):
            self._load_payload(payload)


if __name__ == "__main__":
    unittest.main()
