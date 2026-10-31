"""Deterministic tests for the Agent TH Translate document pipeline."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
import translate_documents


class TranslateDocumentTests(unittest.TestCase):
    def test_blank_pages_never_reach_llm(self):
        source = Path(self.tmp.name) / "blank.txt"
        source.write_text("\f")
        with mock.patch.object(translate_documents, "_translate_batch") as llm:
            result = translate_documents.translate_document(source)
            llm.assert_not_called()
        self.assertEqual(Path(result['output_path']).read_text().count('[หน้าว่าง]'), 2)

    def test_thai_legacy_encoding_is_not_mistaken_for_utf16(self):
        source = Path(self.tmp.name) / "thai.txt"
        source.write_bytes("ภาษา".encode("cp874"))
        text, encoding, warnings = translate_documents._read_text(source)
        self.assertEqual(text, "ภาษา")
        self.assertEqual(encoding, "cp874")

    def test_cancel_during_last_batch_does_not_export(self):
        source = Path(self.tmp.name) / "cancel.txt"
        source.write_text("hello")
        cancelled = False
        def batch(items, **kwargs):
            nonlocal cancelled
            cancelled = True
            return {item["id"]: "สวัสดี" for item in items}
        with mock.patch.object(translate_documents, "_translate_batch", side_effect=batch):
            with self.assertRaises(translate_documents.TranslationCancelled):
                translate_documents.translate_document(source, should_cancel=lambda: cancelled)
        self.assertFalse(list((Path(self.tmp.name) / "output").glob("*.md")))

    def test_pause_during_last_batch_preserves_checkpoint_without_export(self):
        source = Path(self.tmp.name) / "last.txt"
        source.write_text("hello")
        checkpoint = Path(self.tmp.name) / "last.json"
        paused = False
        def batch(items, **kwargs):
            nonlocal paused
            paused = True
            return {item["id"]: "สวัสดี" for item in items}
        with mock.patch.object(translate_documents, "_translate_batch", side_effect=batch):
            with self.assertRaises(translate_documents.TranslationPaused):
                translate_documents.translate_document(source, checkpoint_path=checkpoint,
                    should_pause=lambda: paused)
        self.assertTrue(checkpoint.exists())
        self.assertFalse(list((Path(self.tmp.name) / "output").glob("*.md")))

    def test_repeated_translation_keeps_previous_output(self):
        source = Path(self.tmp.name) / "repeat.txt"
        source.write_text("hello")
        with mock.patch.object(translate_documents, "_translate_batch",
                side_effect=lambda items, **kwargs: {item["id"]: "สวัสดี" for item in items}):
            first = translate_documents.translate_document(source)
            second = translate_documents.translate_document(source)
        self.assertNotEqual(first["output_path"], second["output_path"])
        self.assertTrue(Path(first["output_path"]).exists())

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="max-vlm-translate-test-")
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        patches = [
            mock.patch.object(config, "TRANSLATE_MAX_BYTES", 10 * 1024 * 1024),
            mock.patch.object(config, "TRANSLATE_LLM_BLOCK_CHARS", 32),
            mock.patch.object(config, "TRANSLATE_LLM_BATCH_BLOCKS", 2),
            mock.patch.object(config, "TRANSLATE_LLM_MAX_TOKENS", 2048),
            mock.patch.object(config, "TRANSLATE_OUTPUT_DIR", str(root / "output")),
            mock.patch.object(config, "get_model", return_value="current/no-think-model"),
            mock.patch.object(config, "get_mlx_base_url", return_value="http://127.0.0.1:8085/v1"),
        ]
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_text_formfeed_and_long_lines_have_deterministic_pages_and_bounded_blocks(self) -> None:
        pages = translate_documents._split_text_pages("one\ntwo\fthree", max_chars=20)
        self.assertEqual([page.number for page in pages], [1, 2])
        self.assertEqual(pages[0].line_start, 1)
        self.assertEqual(pages[0].line_end, 2)
        page = translate_documents.Page(1, "x" * 75, "text", "logical_text")
        blocks = translate_documents._chunk_page(page, 32)
        self.assertEqual([len(item["text"]) for item in blocks], [32, 32, 11])

    def test_translation_writes_thai_markdown_with_physical_page_separators(self) -> None:
        source = Path(self.tmp.name) / "source.txt"
        source.write_text("First page 123\fSecond page 456", encoding="utf-8")
        checkpoint = Path(self.tmp.name) / "checkpoint.json"

        def fake_translate(batch, *, model, base_url):
            self.assertEqual(model, "current/no-think-model")
            self.assertEqual(base_url, "http://127.0.0.1:8085/v1")
            return {item["id"]: f"คำแปล {item['text']}" for item in batch}

        with mock.patch.object(translate_documents, "_translate_batch", side_effect=fake_translate):
            result = translate_documents.translate_document(
                source,
                output_dir=Path(self.tmp.name) / "output",
                checkpoint_path=checkpoint,
            )

        output = Path(result["output_path"]).read_text(encoding="utf-8")
        self.assertIn("target_language: th", output)
        self.assertIn("## หน้า 1 / 2", output)
        self.assertIn("## หน้า 2 / 2", output)
        self.assertIn("---", output)
        self.assertIn("คำแปล First page 123", output)
        self.assertIn("คำแปล Second page 456", output)
        self.assertFalse(checkpoint.exists())
        self.assertEqual(result["translated_pages"], 2)

    def test_llm_batch_forces_no_think_and_preserves_protected_tokens(self) -> None:
        captured: dict = {}

        class FakeResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return {
                    "choices": [{"message": {"content": json.dumps({
                        "blocks": [{"id": "p1-b1", "text": "คำแปล 123 ดู https://example.com"}],
                    }, ensure_ascii=False)}}],
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

        batch = [{"id": "p1-b1", "page": 1, "text": "Source 123 https://example.com", "part": 1, "parts": 1}]
        fake_httpx = mock.Mock(Client=FakeClient)
        with mock.patch.object(translate_documents, "httpx", fake_httpx):
            result = translate_documents._translate_batch(
                batch, model="current/no-think-model", base_url="http://127.0.0.1:8085/v1"
            )

        self.assertEqual(result, {"p1-b1": "คำแปล 123 ดู https://example.com"})
        self.assertEqual(captured["url"], "http://127.0.0.1:8085/v1/chat/completions")
        self.assertIs(captured["payload"]["enable_thinking"], False)
        self.assertEqual(captured["payload"]["thinking_budget"], 0)
        self.assertEqual(captured["payload"]["temperature"], 0.0)

    def test_parser_accepts_fenced_prose_trailing_commas_and_python_quotes(self):
        value = translate_documents._parse_json(
            "Here's the result:\n```json\n{'blocks': [{'id': 'p1-b1', 'translation': 'สวัสดี,}ครับ'},],}\n```"
        )
        self.assertEqual(value, {"blocks": [{"id": "p1-b1", "translation": "สวัสดี,}ครับ"}]})

    def test_normalizer_accepts_id_keyed_map_and_common_envelopes(self):
        expected = {"p1-b1", "p2-b1"}
        variants = [
            {"p1-b1": "หนึ่ง", "p2-b1": "สอง"},
            {"translations": {"p1-b1": "หนึ่ง", "p2-b1": {"thai": "สอง"}}},
            {"result": {"blocks": [
                {"block_id": "p1-b1", "translation": "หนึ่ง"},
                {"id": "p2-b1", "translated_text": "สอง"},
            ]}},
        ]
        for value in variants:
            with self.subTest(value=value):
                self.assertEqual(
                    translate_documents._normalize_translation_output(value, expected),
                    {"p1-b1": "หนึ่ง", "p2-b1": "สอง"},
                )
        self.assertIsNone(translate_documents._normalize_translation_output(
            {"p1-b1": "หนึ่ง", "wrong-id": "ผิด"}, expected
        ))

    def test_llm_batch_accepts_the_models_id_keyed_json_shape(self):
        class FakeResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return {"choices": [{"message": {"content": json.dumps({
                    "p1-b1": "คำแปล 123 ดู https://example.com",
                }, ensure_ascii=False)}}]}

        class FakeClient:
            def __init__(self, timeout): pass
            def __enter__(self): return self
            def __exit__(self, *_args): return None
            def post(self, *_args, **_kwargs): return FakeResponse()

        batch = [{"id": "p1-b1", "page": 1, "text": "Source 123 https://example.com", "part": 1, "parts": 1}]
        with mock.patch.object(translate_documents, "httpx", mock.Mock(Client=FakeClient)):
            result = translate_documents._translate_batch(
                batch, model="current/no-think-model", base_url="http://127.0.0.1:8085/v1"
            )
        self.assertEqual(result, {"p1-b1": "คำแปล 123 ดู https://example.com"})

    def test_llm_batch_repairs_one_malformed_response_without_retranslating(self):
        captured = {"calls": []}
        json_dumps = json.dumps

        class FakeResponse:
            def __init__(self, content): self.content = content
            def raise_for_status(self): return None
            def json(self): return {"choices": [{"message": {"content": self.content}}]}

        class FakeClient:
            def __init__(self, timeout): pass
            def __enter__(self): return self
            def __exit__(self, *_args): return None
            def post(self, _url, json=None, **_kwargs):
                captured["calls"].append(json)
                if len(captured["calls"]) == 1:
                    return FakeResponse("{blocks: [{'id' 'p1-b1','translation':'คำแปล 123 ดู https://example.com'}],}")
                return FakeResponse(json_dumps({
                    "blocks": [{"id": "p1-b1", "text": "คำแปล 123 ดู https://example.com"}],
                }, ensure_ascii=False))

        batch = [{"id": "p1-b1", "page": 1, "text": "Source 123 https://example.com", "part": 1, "parts": 1}]
        with mock.patch.object(translate_documents, "httpx", mock.Mock(Client=FakeClient)):
            result = translate_documents._translate_batch(
                batch, model="current/no-think-model", base_url="http://127.0.0.1:8085/v1"
            )
        self.assertEqual(result, {"p1-b1": "คำแปล 123 ดู https://example.com"})
        self.assertEqual(len(captured["calls"]), 2)
        self.assertIs(captured["calls"][1]["enable_thinking"], False)
        self.assertIn("Repair translation response formatting only", captured["calls"][1]["messages"][0]["content"])

    def test_llm_batch_retries_only_empty_or_protected_token_rows(self):
        captured = {"calls": []}
        json_dumps = json.dumps

        class FakeResponse:
            def __init__(self, value): self.value = value
            def raise_for_status(self): return None
            def json(self): return {"choices": [{"message": {"content": json_dumps(self.value, ensure_ascii=False)}}]}

        class FakeClient:
            def __init__(self, timeout): pass
            def __enter__(self): return self
            def __exit__(self, *_args): return None
            def post(self, _url, json=None, **_kwargs):
                captured["calls"].append(json)
                if len(captured["calls"]) == 1:
                    return FakeResponse({
                        "p1-b1": "คำแปล 123 ดู https://example.com",
                        "p2-b1": "คำแปลที่ไม่มีตัวเลข",
                        "p3-b1": "",
                    })
                if len(captured["calls"]) == 2:
                    return FakeResponse({"p2-b1": "คำแปล 456", "p3-b1": ""})
                return FakeResponse({"p3-b1": "คำแปลสาม"})

        batch = [
            {"id": "p1-b1", "page": 1, "text": "Source 123 https://example.com", "part": 1, "parts": 1},
            {"id": "p2-b1", "page": 2, "text": "Source 456", "part": 1, "parts": 1},
            {"id": "p3-b1", "page": 3, "text": "Third source", "part": 1, "parts": 1},
        ]
        with mock.patch.object(translate_documents, "httpx", mock.Mock(Client=FakeClient)):
            result = translate_documents._translate_batch(
                batch, model="current/no-think-model", base_url="http://127.0.0.1:8085/v1"
            )
        self.assertEqual(result, {
            "p1-b1": "คำแปล 123 ดู https://example.com",
            "p2-b1": "คำแปล 456",
            "p3-b1": "คำแปลสาม",
        })
        self.assertEqual(len(captured["calls"]), 3)
        retried = json.loads(captured["calls"][1]["messages"][1]["content"])
        self.assertEqual([block["id"] for block in retried["blocks"]], ["p2-b1", "p3-b1"])
        retried_again = json.loads(captured["calls"][2]["messages"][1]["content"])
        self.assertEqual([block["id"] for block in retried_again["blocks"]], ["p3-b1"])

    def test_llm_batch_splits_token_limited_batches_and_keeps_block_ids(self):
        captured = {"calls": []}
        json_dumps = json.dumps

        class FakeResponse:
            def __init__(self, value, finish="stop"): self.value, self.finish = value, finish
            def raise_for_status(self): return None
            def json(self):
                return {"choices": [{"finish_reason": self.finish, "message": {"content": json_dumps(self.value, ensure_ascii=False)}}]}

        class FakeClient:
            def __init__(self, timeout): pass
            def __enter__(self): return self
            def __exit__(self, *_args): return None
            def post(self, _url, json=None, **_kwargs):
                captured["calls"].append(json)
                if len(captured["calls"]) == 1:
                    return FakeResponse("", finish="length")
                request = jsonlib.loads(json["messages"][1]["content"])
                return FakeResponse({"blocks": [
                    {"id": block["id"], "text": "คำแปล " + block["text"]}
                    for block in request["blocks"]
                ]})

        jsonlib = json
        batch = [
            {"id": "p1-b1", "page": 1, "text": "Keep 123", "part": 1, "parts": 1},
            {"id": "p2-b1", "page": 2, "text": "Use https://example.com", "part": 1, "parts": 1},
        ]
        with mock.patch.object(translate_documents, "httpx", mock.Mock(Client=FakeClient)):
            result = translate_documents._translate_batch(
                batch, model="current/no-think-model", base_url="http://127.0.0.1:8085/v1"
            )
        self.assertEqual(result, {"p1-b1": "คำแปล Keep 123", "p2-b1": "คำแปล Use https://example.com"})
        self.assertEqual(len(captured["calls"]), 3)
        self.assertEqual([
            [block["id"] for block in jsonlib.loads(call["messages"][1]["content"])["blocks"]]
            for call in captured["calls"][1:]
        ], [["p1-b1"], ["p2-b1"]])

    def test_llm_batch_splits_one_long_block_without_cutting_url(self):
        captured = {"calls": []}
        json_dumps = json.dumps
        json_loads = json.loads

        class FakeResponse:
            def __init__(self, value, finish="stop"): self.value, self.finish = value, finish
            def raise_for_status(self): return None
            def json(self):
                return {"choices": [{"finish_reason": self.finish, "message": {"content": json_dumps(self.value, ensure_ascii=False)}}]}

        class FakeClient:
            def __init__(self, timeout): pass
            def __enter__(self): return self
            def __exit__(self, *_args): return None
            def post(self, _url, json=None, **_kwargs):
                captured["calls"].append(json)
                if len(captured["calls"]) == 1:
                    return FakeResponse("", finish="length")
                request = json_loads(json["messages"][1]["content"])
                return FakeResponse({"blocks": [
                    {"id": block["id"], "text": "คำแปล " + block["text"]}
                    for block in request["blocks"]
                ]})

        source = "Keep 300 meters clear of the marker at https://example.com and follow the marked trail safely."
        batch = [{"id": "p1-b1", "page": 1, "text": source, "part": 1, "parts": 1}]
        with mock.patch.object(translate_documents, "httpx", mock.Mock(Client=FakeClient)):
            result = translate_documents._translate_batch(
                batch, model="current/no-think-model", base_url="http://127.0.0.1:8085/v1"
            )
        self.assertEqual(len(captured["calls"]), 3)
        self.assertIn("300", result["p1-b1"])
        self.assertIn("https://example.com", result["p1-b1"])
        split_texts = [
            block["text"]
            for call in captured["calls"][1:]
            for block in json_loads(call["messages"][1]["content"])["blocks"]
        ]
        self.assertEqual("".join(split_texts).replace(" ", ""), source.replace(" ", ""))
        self.assertEqual(sum("https://example.com" in text for text in split_texts), 1)


if __name__ == "__main__":
    unittest.main()
