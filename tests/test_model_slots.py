"""Agent TH's model slots: native AR only, and the same models as model_registry.json.

Isolated: reads the two JSON files and builds a fake Hugging Face cache in a temp dir; no network, no model server.
"""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import model_slots

ROOT = Path(__file__).resolve().parents[1]


class ModelSlotsTest(unittest.TestCase):
    def test_registry_loads_and_every_slot_is_native_ar(self) -> None:
        reg = model_slots.load_registry()
        self.assertEqual(reg.default_slot_id, "qwen35-9b-ar")
        self.assertEqual(len(reg.slots), 5)
        for slot in reg.slots:
            self.assertIsNone(slot.draft, slot.slot_id)
            self.assertEqual(slot.acceleration_mode, "off", slot.slot_id)
            self.assertTrue(slot.selectable, slot.slot_id)

    def test_slots_match_model_registry(self) -> None:
        slot_models = {s.model: s for s in model_slots.load_registry().slots}
        registry = json.loads((ROOT / "model_registry.json").read_text(encoding="utf-8"))
        registry_models = {m["repo_id"]: m for m in registry["models"]}
        self.assertEqual(set(slot_models), set(registry_models))
        self.assertEqual(model_slots.load_registry().get(model_slots.load_registry().default_slot_id).model, registry["default_model"])
        for repo, slot in slot_models.items():
            apc = registry_models[repo]["native_apc"]
            self.assertEqual(slot.runtime["apc_template_policy"], apc["template_policy"], repo)
            self.assertEqual(slot.runtime["apc_exact_cache_entries"], apc["exact_cache_entries"], repo)

    def test_every_revision_is_a_full_commit_hash(self) -> None:
        for slot in model_slots.load_registry().slots:
            self.assertRegex(slot.target.revision, r"^[0-9a-f]{40}$", slot.slot_id)

    def test_preflight_refuses_a_target_that_is_not_downloaded(self) -> None:
        slot = model_slots.load_registry().get("qwen35-2b-ar")
        with tempfile.TemporaryDirectory() as cache:
            with self.assertRaises(model_slots.ModelSlotError):
                model_slots.preflight_slot(slot, cache_root=cache)


if __name__ == "__main__":
    unittest.main()
