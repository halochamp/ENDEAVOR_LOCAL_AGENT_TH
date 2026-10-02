from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import config
from langchain_core.messages import HumanMessage
from tools import edit_access
from tools._mutation_guard import classify_mutation_intent


class MutationIntentClassifierTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.focus = self.root / "focus"
        self.focus.mkdir()
        self.internal = self.root / "internal"
        self.internal.mkdir()
        self._state_path = edit_access.APPROVED_EDIT_FOLDERS_PATH
        self._lock_path = edit_access._STATE_LOCK_PATH
        edit_access.APPROVED_EDIT_FOLDERS_PATH = str(self.root / "approved.json")
        edit_access._STATE_LOCK_PATH = str(self.root / "approved.json.lock")
        edit_access._SESSION_FOCUS_FOLDER = ""
        self._workspace_patch = patch.object(config, "WORKSPACE", str(self.internal))
        self._workspace_patch.start()
        edit_access.save_focus_folder(str(self.focus))

    def tearDown(self):
        self._workspace_patch.stop()
        edit_access.APPROVED_EDIT_FOLDERS_PATH = self._state_path
        edit_access._STATE_LOCK_PATH = self._lock_path
        edit_access._SESSION_FOCUS_FOLDER = ""
        self.tmp.cleanup()

    def classify(self, text: str):
        return classify_mutation_intent([HumanMessage(content=text)])

    def test_english_imperatives_with_explicit_files_are_exact_mutations(self):
        cases = [
            "Fix calc.py so add(a, b) returns a+b",
            "Repair src/foo.py so the parser accepts tabs",
            "Patch index.html to add an accessible title",
            "Patch index.html using https://example.org/spec as reference",
            "Refactor app.js to remove the duplicate helper",
            "Effective action: Repair src/foo.py",
        ]
        for prompt in cases:
            with self.subTest(prompt=prompt):
                intent = self.classify(prompt)
                self.assertEqual(intent.kind, "exact")
                self.assertEqual(len(intent.targets), 1)
                self.assertTrue(intent.targets[0].startswith(str(self.focus.resolve())))

    def test_existing_thai_verbs_with_explicit_file_are_exact_mutations(self):
        for prompt in ("แก้ calc.py ให้บวกเลขได้", "แก้ไข src/foo.py", "ปรับปรุง index.html", "อัปเดต app.js"):
            with self.subTest(prompt=prompt):
                self.assertEqual(self.classify(prompt).kind, "exact")

    def test_tree_scope_requires_imperative_and_explicit_workspace_shape(self):
        intent = self.classify("Format the whole workspace with a safe code transform")
        self.assertEqual(intent.kind, "tree")
        self.assertEqual(intent.tree_root, str(self.focus.resolve()))

    def test_explanatory_meta_generic_and_quoted_text_are_not_mutation_intent(self):
        cases = [
            "Explain how to fix calc.py",
            "What is a patch?",
            "Fix my understanding of the topic",
            "Fix my understanding of calc.py",
            'This article says: "Fix calc.py immediately". Summarize the article.',
            "Fix the issue",
        ]
        for prompt in cases:
            with self.subTest(prompt=prompt):
                self.assertEqual(self.classify(prompt).kind, "none")

    def test_explicit_unsafe_target_does_not_fall_back_to_tree_scope(self):
        outside = self.root / "outside"
        outside.mkdir()
        intent = self.classify(f"Fix {outside / 'bad.py'} in the workspace")
        self.assertEqual(intent.kind, "blocked_explicit")
        self.assertIn("outside", intent.reason.lower())


if __name__ == "__main__":
    unittest.main()
