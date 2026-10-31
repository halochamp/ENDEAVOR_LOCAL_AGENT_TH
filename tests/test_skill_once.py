"""`/skill <instruction>` runs one turn with that skill without changing the persistent mode."""
from __future__ import annotations

import unittest
from unittest.mock import patch

import agent_server as srv


class ComposeSkillQueryTests(unittest.TestCase):
    def test_skill_content_is_prefixed_in_the_established_format(self) -> None:
        self.assertEqual(
            srv._compose_skill_query("do it", "research", "BODY"),
            "[SKILL: research]\nBODY\n---\ndo it",
        )

    def test_no_skill_leaves_the_query_unchanged(self) -> None:
        self.assertEqual(srv._compose_skill_query("do it", "", ""), "do it")


class ResolveSkillOnceTests(unittest.TestCase):
    def test_absent_value_means_no_one_shot_skill(self) -> None:
        self.assertIsNone(srv._resolve_skill_once(None))
        self.assertIsNone(srv._resolve_skill_once(""))

    def test_an_existing_skill_resolves_to_its_name_and_content(self) -> None:
        with patch.object(srv, "_load_skill_content", return_value="BODY"), \
             patch.object(srv, "SKILL_TOOLS", {}):
            self.assertEqual(srv._resolve_skill_once(" Research "), ("research", "BODY"))

    def test_an_unknown_skill_is_rejected(self) -> None:
        with patch.object(srv, "_load_skill_content", return_value=None):
            with self.assertRaisesRegex(ValueError, "ไม่พบ skill"):
                srv._resolve_skill_once("nope")

    def test_path_like_names_are_never_looked_up(self) -> None:
        with patch.object(srv, "_load_skill_content") as load:
            for bad in ("../secret", "a/b", "a b", "x\x00"):
                with self.assertRaises(ValueError):
                    srv._resolve_skill_once(bad)
            load.assert_not_called()

    def test_non_string_values_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "skill_once"):
            srv._resolve_skill_once(["research"])

    def test_skills_with_their_own_tools_must_be_opened_as_a_mode(self) -> None:
        with patch.object(srv, "_load_skill_content", return_value="BODY"), \
             patch.object(srv, "SKILL_TOOLS", {"tooly": [object()]}):
            with self.assertRaisesRegex(ValueError, "โหมด"):
                srv._resolve_skill_once("tooly")


if __name__ == "__main__":
    unittest.main()
