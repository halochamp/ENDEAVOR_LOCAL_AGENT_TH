"""Activity-log text must never carry secrets or payloads, and must stay bounded."""
from __future__ import annotations

import json
import unittest

from activity_text import arguments_preview, result_preview, safe_text


class SafeTextTests(unittest.TestCase):
    def test_secret_assignments_are_redacted_including_quoted_values_with_spaces(self) -> None:
        for raw, leaked in (
            ('export API_TOKEN="hunter2 spaced" && run', "hunter2"),
            ("DB_PASSWORD='p@ss word' ./deploy", "p@ss"),
            ("AWS_SECRET_ACCESS_KEY=abcd1234efgh", "abcd1234efgh"),
            ("OPENAI_API_KEY: sk-live-1234567890", "sk-live-1234567890"),
        ):
            with self.subTest(raw=raw):
                out = safe_text(raw)
                self.assertNotIn(leaked, out)
                self.assertIn("[REDACTED]", out)

    def test_cli_secret_flags_bearer_tokens_and_url_secrets_are_redacted(self) -> None:
        self.assertNotIn("topsecret", safe_text("tool --password topsecret --verbose"))
        self.assertNotIn("topsecret", safe_text("tool --api-key=topsecret"))
        self.assertNotIn("abc.def.ghi", safe_text("Authorization: Bearer abc.def.ghi"))
        self.assertNotIn("s3cr3t", safe_text("GET https://x.test/a?user=1&access_token=s3cr3t&z=2"))
        self.assertNotIn("pw123", safe_text("clone https://alice:pw123@git.example.com/r.git"))

    def test_private_key_blocks_are_redacted_even_when_unterminated(self) -> None:
        key = "-----BEGIN RSA PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASC\n-----END RSA PRIVATE KEY-----"
        self.assertNotIn("MIIEvQ", safe_text("cat file\n" + key))
        # A block cut off by the caller's own truncation must not leak its head.
        self.assertNotIn("MIIEvQ", safe_text("x\n-----BEGIN PRIVATE KEY-----\nMIIEvQIBADAN", limit=200))

    def test_redaction_happens_before_truncation(self) -> None:
        text = "A" * 50 + " API_TOKEN=supersecretvalue " + "B" * 50
        out = safe_text(text, limit=60)
        self.assertNotIn("supersecret", out)

    def test_image_payloads_and_control_characters_are_removed(self) -> None:
        out = safe_text("img data:image/png;base64,AAAA1111BBBB+/== end\x00\x07bell")
        self.assertIn("[image payload omitted]", out)
        self.assertNotIn("AAAA1111", out)
        self.assertNotIn("\x00", out)
        self.assertNotIn("\x07", out)

    def test_long_text_is_cut_with_an_explicit_marker(self) -> None:
        out = safe_text("x" * 5000, limit=100)
        self.assertTrue(out.endswith("… [truncated]"))
        self.assertEqual(len(out), 100 + len("… [truncated]"))
        self.assertEqual(safe_text("short"), "short")
        self.assertEqual(safe_text(None), "")


class ArgumentsPreviewTests(unittest.TestCase):
    def test_secret_named_fields_and_payload_fields_are_masked(self) -> None:
        args = {"url": "https://x.test", "api_key": "k-123", "headers": {"Authorization": "Bearer z"},
                "image_url": "data:image/png;base64,AAAA", "password": "pw"}
        out = arguments_preview(args)
        for leaked in ("k-123", "Bearer z", "AAAA", '"pw"'):
            self.assertNotIn(leaked, out)
        self.assertIn("https://x.test", out)
        self.assertIn("[REDACTED]", out)

    def test_nested_values_are_bounded_and_non_dicts_are_empty(self) -> None:
        deep = {"a": {"b": {"c": {"d": {"e": {"f": "too deep"}}}}}}
        self.assertIn("[nested value omitted]", arguments_preview(deep))
        wide = {f"k{i}": i for i in range(60)}
        out = json.loads(arguments_preview(wide))
        self.assertEqual(out["_truncated_fields"], 60 - 24)
        self.assertEqual(arguments_preview("not a dict"), "")
        self.assertEqual(arguments_preview({}), "")

    def test_the_whole_preview_is_length_bounded(self) -> None:
        self.assertLessEqual(len(arguments_preview({"q": "x" * 20000})), 2400 + len("… [truncated]"))


class ResultPreviewTests(unittest.TestCase):
    def test_status_is_derived_from_the_text_and_flags(self) -> None:
        self.assertEqual(result_preview("ok done")["status"], "returned")
        self.assertEqual(result_preview("[error] boom")["status"], "failed")
        self.assertEqual(result_preview("fine", error=True)["status"], "failed")
        self.assertEqual(result_preview("[blocked] no")["status"], "blocked")

    def test_preview_keeps_the_first_three_non_empty_lines_and_is_redacted(self) -> None:
        out = result_preview("\n\nline1\n\nline2 API_TOKEN=zzz\nline3\nline4\nline5")
        self.assertEqual(out["preview"].splitlines()[0], "line1")
        self.assertEqual(len(out["preview"].splitlines()), 3)
        self.assertNotIn("zzz", out["preview"])
        self.assertTrue(out["truncated"])

    def test_empty_and_short_results_are_reported_honestly(self) -> None:
        self.assertEqual(result_preview("")["preview"], "[empty return]")
        short = result_preview("one line")
        self.assertEqual(short["preview"], "one line")
        self.assertFalse(short["truncated"])


if __name__ == "__main__":
    unittest.main()
