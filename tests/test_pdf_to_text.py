"""Deterministic regression tests for Agent TH's direct PDF -> Text panel."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import fitz

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import config
import pdf_to_text


class PdfToTextTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="agent-th-pdf-text-test-")
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        patches = [
            mock.patch.object(config, "PDF_TO_TEXT_PRIVATE_DIR", str(root / "private")),
            mock.patch.object(config, "PDF_TO_TEXT_DB", str(root / "private" / "pdf_to_text.sqlite3")),
            mock.patch.object(config, "PDF_TO_TEXT_OUTPUT_DIR", str(root / "output")),
            mock.patch.object(config, "PDF_TO_TEXT_CACHE_MAX_AGE_SECONDS", 86400),
            mock.patch.object(config, "PDF_TO_TEXT_OCR_DPI", 150),
            mock.patch.object(config, "PDF_TO_TEXT_OCR_MAX_PIXELS", 20_000_000),
            mock.patch.object(config, "PDF_TO_TEXT_NATIVE_MIN_CHARS", 40),
            mock.patch.object(config, "PDF_TO_TEXT_LLM_BATCH_PAGES", 4),
            mock.patch.object(config, "PDF_TO_TEXT_LLM_BATCH_CHARS", 6000),
            mock.patch.object(config, "PDF_TO_TEXT_LLM_TIMEOUT_SECONDS", 90),
        ]
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_native_pdf_preserves_physical_pages_without_llm(self) -> None:
        source = Path(self.tmp.name) / "native.pdf"
        with fitz.open() as doc:
            page = doc.new_page()
            page.insert_text((72, 72), "Native page one contains enough searchable text to stay on the native path completely.")
            page = doc.new_page()
            page.insert_text((72, 72), "Native page two also contains enough searchable text to stay on the native path completely.")
            doc.save(source)

        with mock.patch.object(pdf_to_text, "_llm_rewrite_batch") as rewrite:
            result = pdf_to_text.convert_pdf(source, rewrite_thai=False)
        rewrite.assert_not_called()
        output = Path(result["output_path"]).read_text(encoding="utf-8")
        self.assertIn("===== PDF PAGE 1/2 =====", output)
        self.assertIn("===== PDF PAGE 2/2 =====", output)
        self.assertLess(output.index("PDF PAGE 1/2"), output.index("PDF PAGE 2/2"))
        self.assertEqual(result["stats"]["native"], 2)
        self.assertEqual(result["stats"]["rewritten"], 0)

    def test_thai_format_cleanup_ports_agent_lite_deterministic_repairs(self) -> None:
        # OCR can emit SARA AM before a tone mark; Agent Lite deterministically
        # reorders those existing code points without guessing a word.
        self.assertEqual(pdf_to_text._clean_text("นำ้"), "น้ำ")
        # Spaces inserted directly before Thai combining marks are formatting
        # damage, not lexical word boundaries, and are removed conservatively.
        self.assertEqual(pdf_to_text._clean_text("ก ้"), "ก้")

    def test_llm_rewrite_uses_current_owner_model_and_forces_no_think(self) -> None:
        captured: dict = {}

        class FakeResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return {
                    "choices": [{
                        "message": {
                            "content": json.dumps({
                                "pages": [
                                    {"page": 1, "text": "ข้อความหนึ่ง"},
                                    {"page": 2, "text": "ข้อความสอง"},
                                ]
                            }, ensure_ascii=False)
                        }
                    }]
                }

        class FakeClient:
            def __init__(self, timeout):
                captured["timeout"] = timeout

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def post(self, url, json=None, headers=None):
                captured["url"] = url
                captured["payload"] = json
                captured["headers"] = headers
                return FakeResponse()

        with mock.patch.object(pdf_to_text.httpx, "Client", FakeClient), \
             mock.patch.object(config, "get_mlx_base_url", return_value="http://127.0.0.1:8091/v1"), \
             mock.patch.object(config, "get_model", return_value="dynamic/current-model"):
            result = pdf_to_text._llm_rewrite_batch(
                [(1, "ขอ้ความหนึ่ง"), (2, "ขอ้ความสอง")]
            )

        self.assertEqual(result, {1: "ข้อความหนึ่ง", 2: "ข้อความสอง"})
        payload = captured["payload"]
        self.assertEqual(captured["url"], "http://127.0.0.1:8091/v1/chat/completions")
        self.assertEqual(payload["model"], "dynamic/current-model")
        self.assertIs(payload["enable_thinking"], False)
        self.assertEqual(payload["thinking_budget"], 0)
        self.assertEqual(payload["temperature"], 0.0)
        self.assertEqual(len(json.loads(payload["messages"][1]["content"].split("\n\n", 1)[1])["pages"]), 2)

    def test_rewrite_validation_fails_closed_on_lost_numeric_fact(self) -> None:
        self.assertTrue(pdf_to_text._validate_rewrite("ค่า 123 บาท", "ค่า 123 บาท"))
        self.assertFalse(pdf_to_text._validate_rewrite("ค่า 123 บาท", "ค่า บาท"))
        self.assertFalse(pdf_to_text._validate_rewrite(
            "รายการเดิม 123 ต้องรักษาข้อความต้นฉบับนี้ไว้",
            "ประโยคใหม่ 123 ที่มีความหมายต่างออกไปทั้งหมดครับ",
        ))


if __name__ == "__main__":
    unittest.main()
