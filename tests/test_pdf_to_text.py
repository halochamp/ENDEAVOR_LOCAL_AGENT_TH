"""Deterministic regression tests for Agent TH's direct PDF -> MD panel."""
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
        self.tmp = tempfile.TemporaryDirectory(prefix="max-vlm-pdf-text-test-")
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
            mock.patch.object(config, "PDF_TO_TEXT_VISION_MAX_SIDE", 1536),
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

        with mock.patch.object(pdf_to_text, "_llm_rewrite_page") as rewrite:
            result = pdf_to_text.convert_pdf(source, rewrite_thai=False)
        rewrite.assert_not_called()
        output = Path(result["output_path"]).read_text(encoding="utf-8")
        self.assertIn("<!-- PDF page 1/2 -->", output)
        self.assertIn("<!-- PDF page 2/2 -->", output)
        self.assertIn("**หน้า 1/2**", output)                   # visible page number, written by code
        self.assertIn("**หน้า 2/2**", output)
        self.assertLess(output.index("PDF page 1/2"), output.index("PDF page 2/2"))
        self.assertEqual(output.count("\n---\n"), 1)            # between pages only
        self.assertTrue(result["output_path"].endswith("_pdf_to_md.md"))
        self.assertEqual(result["stats"]["native"], 2)
        self.assertEqual(result["stats"]["rewritten"], 0)

    def test_thai_format_cleanup_ports_agent_lite_deterministic_repairs(self) -> None:
        # OCR can emit SARA AM before a tone mark; Agent Lite deterministically
        # reorders those existing code points without guessing a word.
        self.assertEqual(pdf_to_text._clean_text("นำ้"), "น้ำ")
        # Spaces inserted directly before Thai combining marks are formatting
        # damage, not lexical word boundaries, and are removed conservatively.
        self.assertEqual(pdf_to_text._clean_text("ก ้"), "ก้")

    def _fake_client(self, captured, reply="# หัวข้อ\n\nเนื้อหา", finish="stop"):
        class FakeResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return {"choices": [{"message": {"content": reply}, "finish_reason": finish}]}

        class FakeClient:
            def __init__(self, timeout):
                captured["timeout"] = timeout

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def post(self, url, json=None, headers=None):
                captured.setdefault("calls", []).append((url, json, headers))
                return FakeResponse()

        return FakeClient

    def test_unticked_text_keeps_its_lines_and_cannot_turn_into_accidental_markdown(self) -> None:
        text = "ชื่อ ๑\nที่อยู่ ๒\n1. first\n- dash\n# hash\n> quote\n---\n\nย่อหน้าใหม่\nบรรทัดสุดท้าย"
        out = pdf_to_text._text_to_markdown(text)
        self.assertEqual(out.split("\n"), [
            "ชื่อ ๑  ", "ที่อยู่ ๒  ", "1\\. first  ", "\\- dash  ", "\\# hash  ", "\\> quote  ", "\\---", "",
            "ย่อหน้าใหม่  ", "บรรทัดสุดท้าย"])
        self.assertEqual(pdf_to_text._text_to_markdown("๑. ข้อแรก\n๒. ข้อสอง"), "๑. ข้อแรก  \n๒. ข้อสอง")   # Thai numerals: no escape
        try:
            import markdown_it
        except ImportError:
            return
        html = markdown_it.MarkdownIt("commonmark").render(out)
        self.assertEqual(html.count("<br"), 7)                       # the lines are still lines when rendered
        self.assertNotIn("<h1", html)
        self.assertNotIn("<ul", html)
        self.assertNotIn("<ol", html)
        self.assertNotIn("<blockquote", html)

    def test_unticked_text_cannot_open_code_blocks_html_or_lose_backslashes(self) -> None:
        BS = chr(92)
        convert = pdf_to_text._text_to_markdown
        # string level (no renderer needed)
        self.assertEqual(convert("```\nx"), BS + "`" + BS + "`" + BS + "`  \nx")           # every fence char escaped
        self.assertTrue(convert("  ~~~ใช้โค้ด").startswith("  " + BS + "~" + BS + "~" + BS + "~"))
        self.assertEqual(convert("a <b>x</b> <5"), "a " + BS + "<b>x" + BS + "</b> <5")        # only tag-like "<" is escaped
        self.assertEqual(convert("ย่อหน้า\n\n    เยื้องลึก"), "ย่อหน้า\n\n" + "\u00a0" * 4 + "เยื้องลึก")
        self.assertEqual(convert("ย่อหน้า\n\n  เยื้อง 2 ช่อง"), "ย่อหน้า\n\n  เยื้อง 2 ช่อง")  # 1-3 columns left alone
        self.assertEqual(convert("C:" + BS + "temp"), "C:" + BS + "temp")                         # backslash before a letter: untouched
        self.assertEqual(convert(BS + "*x"), BS * 3 + "*x")                                        # backslash + punctuation
        try:
            import html as html_lib, re
            import markdown_it
        except ImportError:
            return
        md = markdown_it.MarkdownIt("commonmark")

        def shown(text):
            rendered = md.render(convert(text))
            return re.sub(r"\s+", " ", html_lib.unescape(re.sub(r"<[^>]+>", "", rendered))).strip(), rendered

        for text in ("ย่อหน้าแรก\n\n    เยื้อง 4 ช่อง\n\nย่อหน้าสาม", "ย่อหน้าแรก\n\n\tเยื้องแท็บ\n\nย่อหน้าสาม"):
            self.assertNotIn("<pre>", shown(text)[1])
        for fence in ("```", "~~~", "````"):
            doc = "**หน้า 1/2**\n\n" + convert("ก่อน\n" + fence + "\nไม่ปิด") + "\n\n---\n\n**หน้า 2/2**\n\nหน้าสอง"
            rendered = md.render(doc)
            self.assertNotIn("<pre>", rendered)
            self.assertIn("<strong>หน้า 2/2</strong>", rendered)        # the next page's number is not swallowed
            self.assertIn("<hr", rendered)
        html_text = "ราคา <5 บาท และ <b>ตัวหนา</b> กับ <div>บล็อก</div> จบ"
        self.assertEqual(shown(html_text)[0], html_text)
        slash = "ตัวอย่าง " + BS + "*literal" + BS + "* และ C:" + BS + "temp" + BS + "file " + BS + "# " + BS + BS
        self.assertEqual(shown(slash)[0], re.sub(r"\s+", " ", slash).strip())
        plain = "ปกติ ไม่มีอะไรพิเศษ 1,250 บาท\nบรรทัดสอง a_b_c 2*3"
        self.assertEqual(shown(plain)[0], "ปกติ ไม่มีอะไรพิเศษ 1,250 บาท บรรทัดสอง a_b_c 2*3")

    def test_the_written_file_applies_it_only_to_pages_the_model_did_not_write(self) -> None:
        conn, job = self._seed_pages([(1, "native", "1. plain\nline"), (2, "ocr", "model page"), (3, "ocr_table", "| a | b |\n| --- | --- |")])
        conn.execute("UPDATE pdf_text_pages SET final_text='# Heading\n\n1. kept', rewrite_status='rewritten' WHERE page_number=2")
        conn.commit()
        output = Path(self.tmp.name) / "out.md"
        pdf_to_text._write_output(conn, job, 3, output)
        body = output.read_text(encoding="utf-8")
        self.assertIn("1\\. plain  \nline", body)           # deterministic page: escaped + hard break
        self.assertIn("# Heading\n\n1. kept", body)         # model page: untouched
        self.assertIn("| a | b |\n| --- | --- |", body)       # OCR table: untouched
        # page delimiters come from code in every case (model page or not), never from the model
        for number in (1, 2, 3):
            self.assertEqual(body.count(f"<!-- PDF page {number}/3 -->\n**หน้า {number}/3**\n"), 1)
        self.assertEqual(body.count("\n---\n"), 2)
        self.assertLess(body.index("PDF page 1/3"), body.index("PDF page 2/3"))
        self.assertLess(body.index("PDF page 2/3"), body.index("PDF page 3/3"))

    def test_broken_thai_spacing_is_repaired_but_real_word_and_number_spaces_are_kept(self) -> None:
        for broken, fixed in (
            ("น้ ำ", "น้ำ"), ("น ้ำ", "น้ำ"), ("นำ้", "น้ำ"), ("ท ี่", "ที่"), ("ข้ าราชการ", "ข้าราชการ"),
            ("ก าร", "การ"), ("เ รื่อง", "เรื่อง"), ("ก ำลัง", "กำลัง"), ("โ ครงการ", "โครงการ"),
        ):
            self.assertEqual(pdf_to_text._clean_text(broken), fixed, broken)
        for keep in ("ฉัน กิน ข้าว", "ค่า ไฟ", "จำนวน ๑๘๐ คน", "ราคา 1,250 บาท", "ชื่อ นามสกุล", "ดี ๆ"):
            self.assertEqual(pdf_to_text._clean_text(keep), keep, keep)
        self.assertEqual(pdf_to_text._clean_text("ที่\nด้วย"), "ที่\nด้วย")      # never joins across a line break

    def test_ocr_fragments_are_spacing_repaired_before_table_or_prose_reconstruction(self) -> None:
        boxes = [
            {"text": "น้ ำ ดื่ม", "x": 0.1, "y": 0.8, "w": 0.2, "h": 0.03},
            {"text": "เ รื่อง ก าร", "x": 0.1, "y": 0.7, "w": 0.3, "h": 0.03},
        ]
        with fitz.open() as doc:
            doc.new_page()
            with mock.patch.object(pdf_to_text, "read_layout", return_value=boxes):
                kind, text = pdf_to_text._ocr_page(doc[0], 1)
        self.assertEqual(kind, "ocr")
        self.assertIn("น้ำ ดื่ม", text)
        self.assertIn("เรื่อง การ", text)

    def test_llm_page_request_carries_image_and_text_on_the_owner_model_no_think(self) -> None:
        captured: dict = {}
        with mock.patch.object(pdf_to_text.httpx, "Client", self._fake_client(captured)), \
             mock.patch.object(config, "get_mlx_base_url", return_value="http://127.0.0.1:8091/v1"), \
             mock.patch.object(config, "get_model", return_value="dynamic/current-model"):
            result = pdf_to_text._llm_rewrite_page(3, "ขอ้ความหนึ่ง", "data:image/png;base64,AAAA")
        self.assertEqual(result, "# หัวข้อ\n\nเนื้อหา")
        url, payload, _headers = captured["calls"][0]
        self.assertEqual(url, "http://127.0.0.1:8091/v1/chat/completions")
        self.assertEqual(payload["model"], "dynamic/current-model")
        self.assertIs(payload["enable_thinking"], False)
        self.assertEqual(payload["thinking_budget"], 0)
        self.assertEqual(payload["temperature"], 0.0)
        parts = payload["messages"][1]["content"]
        self.assertEqual([part["type"] for part in parts], ["text", "image_url"])
        self.assertIn("ขอ้ความหนึ่ง", parts[0]["text"])
        self.assertEqual(parts[1]["image_url"]["url"], "data:image/png;base64,AAAA")
        self.assertEqual(captured["timeout"], config.PDF_TO_TEXT_LLM_TIMEOUT_SECONDS)

    def test_page_image_longest_side_is_the_configured_vision_side(self) -> None:
        import base64
        import io
        from PIL import Image

        with fitz.open() as doc:
            doc.new_page(width=842, height=595)       # A4 landscape in points
            doc.new_page(width=300, height=200)
            doc.new_page(width=595, height=842)       # A4 portrait
            for index in range(3):
                url = pdf_to_text._page_image_data_url(doc[index], index + 1)
                self.assertTrue(url.startswith("data:image/png;base64,"))
                width, height = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).size
                self.assertAlmostEqual(max(width, height), 1536, delta=2)
            with mock.patch.object(config, "PDF_TO_TEXT_VISION_MAX_SIDE", 800):
                url = pdf_to_text._page_image_data_url(doc[2], 3)
                width, height = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).size
                self.assertAlmostEqual(height, 800, delta=2)

    def test_a_fence_around_a_whole_page_is_removed(self) -> None:
        self.assertEqual(pdf_to_text._strip_page_fence("```markdown\n# หัวข้อ\n\nเนื้อหา\n```"), "# หัวข้อ\n\nเนื้อหา")
        self.assertEqual(pdf_to_text._strip_page_fence("# หัวข้อ\n\n```py\nx=1\n```\nต่อ"), "# หัวข้อ\n\n```py\nx=1\n```\nต่อ")

    def _pdf_with_pages(self, count):
        path = Path(self.tmp.name) / f"pages{count}.pdf"
        with fitz.open() as doc:
            for index in range(count):
                doc.new_page().insert_text((72, 72), f"page {index + 1}")
            doc.save(path)
        return path

    def _seed_pages(self, rows):
        """Insert (page, source_kind, text) rows into a fresh job and return (conn, job_id)."""
        conn = pdf_to_text._connect()
        self.addCleanup(conn.close)
        conn.execute(
            "INSERT INTO pdf_text_jobs(job_id, source_name, source_sha256, total_pages, rewrite_thai, status, created_at) "
            "VALUES('j','x.pdf','h',?,1,'running',0)", (len(rows),))
        conn.executemany(
            "INSERT INTO pdf_text_pages(job_id, page_number, source_kind, raw_text, final_text, rewrite_status) "
            "VALUES('j',?,?,?,?,'not_needed')", [(n, k, t, t) for n, k, t in rows])
        conn.commit()
        return conn, "j"

    def test_ticked_rewrite_sends_every_page_that_has_text_one_at_a_time_with_its_image(self) -> None:
        rows = [
            (1, "native", "ข้อความภาษาไทยจาก text layer ของ PDF ต้นฉบับ"),
            (2, "native", "Plain English native page, born digital text."),
            (3, "ocr", "English text that came out of OCR on a scanned page"),
            (4, "ocr", "ข้อความไทยที่ได้จาก OCR"),
            (5, "ocr_table", "| a | b |\n| --- | --- |\n| 1 | 2 |"),
            (6, "ocr", ""),
        ]
        conn, job = self._seed_pages(rows)
        pdf = self._pdf_with_pages(6)
        seen: list[tuple[int, str, str]] = []

        def fake_page(page, text, image_url):
            seen.append((page, text, image_url))
            return text

        with mock.patch.object(pdf_to_text, "_llm_rewrite_page", side_effect=fake_page), \
             mock.patch.object(pdf_to_text, "_page_image_data_url", side_effect=lambda page, n: f"data:image/png;base64,P{n}"):
            rewritten, fallback, warnings = pdf_to_text._apply_thai_rewrite(conn, job, 6, None, pdf)
        self.assertEqual([page for page, _, _ in seen], [1, 2, 3, 4, 5, 6])   # every page, blank ones too, in order
        self.assertTrue(all(url == f"data:image/png;base64,P{page}" for page, _, url in seen))   # each page its own image
        self.assertEqual((rewritten, fallback, warnings), (5, 0, []))      # page 6: blank in, blank out = nothing to do

    def test_an_image_only_page_gets_the_models_description_and_the_prompt_asks_for_figure_descriptions(self) -> None:
        conn, job = self._seed_pages([(1, "ocr", "")])
        pdf = self._pdf_with_pages(1)
        description = "> **[ภาพ]** แผนที่แสดงพื้นที่ครอบคลุมของภาพถ่ายทางอากาศ มีสัญลักษณ์เส้นทางบิน 3 เส้น"
        with mock.patch.object(pdf_to_text, "_llm_rewrite_page", return_value=description), \
             mock.patch.object(pdf_to_text, "_page_image_data_url", return_value="data:image/png;base64,P"):
            rewritten, fallback, _ = pdf_to_text._apply_thai_rewrite(conn, job, 1, None, pdf)
        self.assertEqual((rewritten, fallback), (1, 0))
        self.assertEqual(conn.execute("SELECT final_text FROM pdf_text_pages WHERE job_id=?", (job,)).fetchone()[0], description)
        prompt = pdf_to_text._rewrite_prompt("")
        self.assertIn("> **[ภาพ]**", prompt)
        self.assertIn("ภาพล้วน", prompt)                                # empty extraction is explained to the model

    def test_the_models_page_is_written_as_it_is_even_when_it_differs_a_lot_from_the_extraction(self) -> None:
        conn, job = self._seed_pages([(1, "ocr", "ยอด 1,250 บาท ชื่อ สมชาย"), (2, "ocr", "ข้อความเดิมหน้าสอง")])
        pdf = self._pdf_with_pages(2)
        answers = {1: "# สรุป\n\nยอด 9,999 บาท ชื่อ สมหญิง และมีข้อความที่อ่านจากภาพเพิ่ม", 2: "ข้อความหน้าสองที่ถูกแก้"}
        with mock.patch.object(pdf_to_text, "_llm_rewrite_page", side_effect=lambda page, text, url: answers[page]), \
             mock.patch.object(pdf_to_text, "_page_image_data_url", return_value="data:image/png;base64,P"):
            rewritten, fallback, warnings = pdf_to_text._apply_thai_rewrite(conn, job, 2, None, pdf)
        self.assertEqual((rewritten, fallback, warnings), (2, 0, []))
        finals = dict(conn.execute("SELECT page_number, final_text FROM pdf_text_pages WHERE job_id=?", (job,)))
        self.assertEqual(finals, answers)                     # numbers, names and length all differ: written anyway

    def test_an_empty_or_cut_off_reply_keeps_the_extracted_text(self) -> None:
        conn, job = self._seed_pages([(1, "ocr", "ข้อความหน้าหนึ่ง"), (2, "ocr", "ข้อความหน้าสอง"), (3, "ocr", "")])
        pdf = self._pdf_with_pages(3)

        def reply(page, text, url):
            if page == 1:
                return "   "                                        # empty answer for a page that has text
            if page == 2:
                raise pdf_to_text._ReplyTruncated("cut off")
            return ""                                               # blank page, blank answer

        with mock.patch.object(pdf_to_text, "_llm_rewrite_page", side_effect=reply), \
             mock.patch.object(pdf_to_text, "_page_image_data_url", return_value="data:image/png;base64,P"):
            rewritten, fallback, warnings = pdf_to_text._apply_thai_rewrite(conn, job, 3, None, pdf)
        self.assertEqual((rewritten, fallback), (0, 2))
        self.assertTrue(any("was cut off" in w for w in warnings), warnings)
        finals = dict(conn.execute("SELECT page_number, final_text FROM pdf_text_pages WHERE job_id=?", (job,)))
        self.assertEqual(finals, {1: "ข้อความหน้าหนึ่ง", 2: "ข้อความหน้าสอง", 3: ""})

    def test_a_reply_cut_off_at_max_tokens_is_reported_by_the_page_call(self) -> None:
        captured: dict = {}
        with mock.patch.object(pdf_to_text.httpx, "Client", self._fake_client(captured, finish="length")):
            with self.assertRaises(pdf_to_text._ReplyTruncated):
                pdf_to_text._llm_rewrite_page(1, "ข้อความ", "data:image/png;base64,AAAA")

    def test_a_failed_page_keeps_its_text_and_three_failures_in_a_row_stop_the_calls(self) -> None:
        rows = [(n, "ocr", f"ข้อความหน้า {n} ของเอกสาร") for n in range(1, 7)]
        conn, job = self._seed_pages(rows)
        pdf = self._pdf_with_pages(6)
        calls: list[int] = []

        def flaky(page, text, image_url):
            calls.append(page)
            if page == 1:
                return "# ข้อความหน้า 1 ของเอกสาร"
            raise RuntimeError("server down")

        with mock.patch.object(pdf_to_text, "_llm_rewrite_page", side_effect=flaky), \
             mock.patch.object(pdf_to_text, "_page_image_data_url", return_value="data:image/png;base64,P"):
            rewritten, fallback, warnings = pdf_to_text._apply_thai_rewrite(conn, job, 6, None, pdf)
        self.assertEqual(calls, [1, 2, 3, 4])                    # 3 consecutive failures, then no more requests
        self.assertEqual((rewritten, fallback), (1, 5))
        self.assertTrue(any("did not answer 3 pages in a row" in w for w in warnings), warnings)
        statuses = dict(conn.execute("SELECT page_number, rewrite_status FROM pdf_text_pages WHERE job_id=?", (job,)))
        self.assertEqual(statuses, {1: "rewritten", 2: "fallback", 3: "fallback", 4: "fallback", 5: "fallback", 6: "fallback"})
        finals = dict(conn.execute("SELECT page_number, final_text FROM pdf_text_pages WHERE job_id=?", (job,)))
        self.assertEqual(finals[2], "ข้อความหน้า 2 ของเอกสาร")  # fallback = deterministic text, untouched

    def test_the_pipeline_reports_what_each_wait_is_for(self) -> None:
        """ocr (a scanned page being read), then render + model (the model's reply) per page, in order."""
        events: list[tuple[str, int]] = []
        scan = Path(self.tmp.name) / "scan.pdf"
        with fitz.open() as doc:
            doc.new_page()                                   # no text layer: goes to OCR
            doc.new_page().insert_text((72, 72), "Page two has enough searchable text to stay on the native path entirely.")
            doc.save(scan)
        with mock.patch.object(pdf_to_text, "_ocr_page", return_value=("ocr", "scanned page text from OCR")), \
             mock.patch.object(pdf_to_text, "_llm_rewrite_page", side_effect=lambda page, text, url: text), \
             mock.patch.object(pdf_to_text, "_page_image_data_url", return_value="data:image/png;base64,P"):
            pdf_to_text.convert_pdf(scan, rewrite_thai=True, progress=lambda stage, page, total, stats: events.append((stage, page)))
        stages = [stage for stage, _ in events]
        self.assertEqual(events[:4], [("extract", 1), ("ocr", 1), ("extract", 2), ("rewrite", 1)])    # OCR only for the scanned page
        self.assertEqual([e for e in events if e[0] in ("render", "model")], [("render", 1), ("model", 1), ("render", 2), ("model", 2)])
        for page in (1, 2):
            self.assertLess(events.index(("render", page)), events.index(("model", page)))
        self.assertEqual(stages[-1], "write")

    def test_unticked_rewrite_never_calls_the_llm(self) -> None:
        source = Path(self.tmp.name) / "plain.pdf"
        with fitz.open() as doc:
            doc.new_page().insert_text((72, 72), "Plain searchable page with enough text to stay on the native path entirely.")
            doc.save(source)
        with mock.patch.object(pdf_to_text, "_apply_thai_rewrite") as apply_rewrite, \
             mock.patch.object(pdf_to_text, "_llm_rewrite_page") as page_call:
            pdf_to_text.convert_pdf(source, rewrite_thai=False)
        apply_rewrite.assert_not_called()
        page_call.assert_not_called()
        with mock.patch.object(pdf_to_text, "_apply_thai_rewrite", return_value=(0, 0, [])) as apply_rewrite:
            pdf_to_text.convert_pdf(source, rewrite_thai=True)
        apply_rewrite.assert_called_once()
        self.assertEqual(Path(apply_rewrite.call_args.args[4]).name, "plain.pdf")    # the PDF path reaches the image step

    def test_llm_reply_budget_scales_with_the_page_text(self) -> None:
        captured: dict = {}
        with mock.patch.object(pdf_to_text.httpx, "Client", self._fake_client(captured)):
            for size in (100, 6000, 20000):
                pdf_to_text._llm_rewrite_page(1, "ก" * size, "data:image/png;base64,AAAA")
        self.assertEqual([call[1]["max_tokens"] for call in captured["calls"]], [4096, 9600, 12000])

    def test_a_fence_the_model_wraps_around_the_reply_is_removed(self) -> None:
        captured: dict = {}
        with mock.patch.object(pdf_to_text.httpx, "Client",
                               self._fake_client(captured, reply="```markdown\n# หัวข้อ\n\nเนื้อหา\n```")):
            self.assertEqual(pdf_to_text._llm_rewrite_page(1, "หัวข้อ เนื้อหา", "data:image/png;base64,AAAA"),
                             "# หัวข้อ\n\nเนื้อหา")


class PdfToMdTextOnlyModelTests(unittest.TestCase):
    """TH can serve a model that cannot see images: PDF to MD must then send the extracted text only (a vision model always can)."""

    def setUp(self) -> None:
        from tools import _vision_capability as vision
        self.vision = vision
        vision.reset_for_tests()
        self.addCleanup(vision.reset_for_tests)
        self.sent: list[dict] = []
        for name, value in (("get_mlx_base_url", lambda: "http://127.0.0.1:9999/v1"), ("get_model", lambda: "test-model")):
            patcher = mock.patch.object(config, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def _reply(self, text: str = "# Page") -> dict:
        return {"choices": [{"finish_reason": "stop", "message": {"content": text}}]}

    def _has_image(self, payload: dict) -> bool:
        content = payload["messages"][1]["content"]
        return isinstance(content, list) and any(part.get("type") == "image_url" for part in content)

    def test_a_vision_model_gets_the_page_image(self) -> None:
        def post(url, payload):
            self.sent.append(payload)
            return self._reply()
        with mock.patch.object(pdf_to_text, "_post_page", post):
            self.assertEqual(pdf_to_text._llm_rewrite_page(1, "text", "data:image/png;base64,AAA"), "# Page")
        self.assertTrue(self._has_image(self.sent[0]))

    def test_a_model_known_to_be_text_only_gets_no_image(self) -> None:
        self.vision.mark_text_only("http://127.0.0.1:9999/v1", "test-model")

        def post(url, payload):
            self.sent.append(payload)
            return self._reply()
        with mock.patch.object(pdf_to_text, "_post_page", post):
            pdf_to_text._llm_rewrite_page(1, "text", "data:image/png;base64,AAA")
        self.assertEqual(len(self.sent), 1)
        self.assertFalse(self._has_image(self.sent[0]))
        self.assertIn("text", self.sent[0]["messages"][1]["content"])

    def test_a_server_that_refuses_the_image_is_remembered_and_the_page_retried_without_it(self) -> None:
        class Refusal(Exception):
            status_code = 400
        calls: list[bool] = []

        def post(url, payload):
            calls.append(self._has_image(payload))
            if calls[-1]:
                raise Refusal("This model does not support image input (image_url is not supported)")
            return self._reply("# Retried")
        with mock.patch.object(pdf_to_text, "_post_page", post):
            self.assertEqual(pdf_to_text._llm_rewrite_page(2, "text", "data:image/png;base64,AAA"), "# Retried")
            self.assertEqual(calls, [True, False])
            self.assertEqual(self.vision.get_capability("http://127.0.0.1:9999/v1", "test-model"), "text_only")
            pdf_to_text._llm_rewrite_page(3, "text", "data:image/png;base64,AAA")   # remembered: no image attempt
        self.assertEqual(calls, [True, False, False])

    def test_an_unrelated_error_is_not_taken_for_a_text_only_model(self) -> None:
        def post(url, payload):
            raise RuntimeError("connection reset")
        with mock.patch.object(pdf_to_text, "_post_page", post):
            with self.assertRaises(RuntimeError):
                pdf_to_text._llm_rewrite_page(1, "text", "data:image/png;base64,AAA")
        self.assertEqual(self.vision.get_capability("http://127.0.0.1:9999/v1", "test-model"), "unknown")


if __name__ == "__main__":
    unittest.main()
