"""_test_markdown_text_unit.py -- markdown_text + the translator's .md writer: code, not the model, owns the structure."""
import html as html_lib
import os
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import translate_documents as T
from markdown_text import close_open_fence, text_to_markdown

PASS = FAIL = 0
BS = chr(92)


def check(label, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {label}")
    else:
        FAIL += 1
        print(f"  FAIL  {label}  {detail}")


# -- close_open_fence: balanced blocks untouched, an open one is closed with a matching marker ------------------
balanced = "ก่อน\n```py\nx = 1\n```\nหลัง"
check("balanced fence is left exactly as it is", close_open_fence(balanced) == balanced)
check("unclosed ``` is closed", close_open_fence("a\n```py\nx").endswith("\n```\n") and close_open_fence("a\n```py\nx").count("```") == 2)
check("unclosed ~~~ is closed with ~~~", close_open_fence("a\n~~~\nx") == "a\n~~~\nx\n~~~\n")
check("a longer opener needs an equal-or-longer closer", close_open_fence("````\nx\n```\ny") == "````\nx\n```\ny\n````\n")
check("a shorter fence inside a longer block does not close it", close_open_fence("````\n```\nx\n````") == "````\n```\nx\n````")
check("a tilde line does not close a backtick block", close_open_fence("```\nx\n~~~") == "```\nx\n~~~\n```\n")
check("indent of 4+ is not a fence", close_open_fence("    ```\ntext") == "    ```\ntext")
check("a backtick line with a backtick in its info string is not a fence", close_open_fence("```a`b\ntext") == "```a`b\ntext")
check("text without fences is returned unchanged", close_open_fence("ปกติ\n\nข้อความ") == "ปกติ\n\nข้อความ")

# -- text_to_markdown without hard breaks keeps paragraphs flowing but still neutralises syntax --------------------
check("hard_breaks=False adds no trailing spaces", "  \n" not in text_to_markdown("บรรทัดหนึ่ง\nบรรทัดสอง", hard_breaks=False))
check("hard_breaks=True (PDF to MD) still adds them", text_to_markdown("บรรทัดหนึ่ง\nบรรทัดสอง") == "บรรทัดหนึ่ง  \nบรรทัดสอง")
check("hard_breaks=False still escapes a fence", text_to_markdown("```\nx", hard_breaks=False).startswith(BS + "`" + BS + "`" + BS + "`"))

# -- the translator's writer ----------------------------------------------------------------------------------------
try:
    import markdown_it
    md = markdown_it.MarkdownIt("commonmark")
except ImportError:
    md = None


def write(translated, source_name="doc.pdf", blocks=None, page_count=2):
    pages = [T.Page(number=n, text="x", source_kind="native", label="physical_page") for n in range(1, page_count + 1)]
    meta = {"source_name": source_name, "page_basis": "physical", "model": "m",
            "blocks": blocks or [{"id": f"p{n}-1", "page": n} for n in range(1, page_count + 1)]}
    out = Path(tempfile.mkdtemp()) / "o.md"
    T._write_markdown(out, meta, pages, translated, partial=False)
    return out.read_text(encoding="utf-8")


raw = write({"p1-1": "ย่อหน้าแรก", "p2-1": "ย่อหน้าสอง", "p3-1": "ย่อหน้าสาม"}, page_count=3)
check("every page has its number heading and hidden marker, written by code",
      all(f"## หน้า {n} / 3\n<!-- source-page: {n}; extraction: native -->" in raw for n in (1, 2, 3)))
check("page breaks sit between pages only", raw.count("\n---\n") == 2 + 1 and raw.rstrip().endswith("ย่อหน้าสาม"))   # 2 page breaks + closing front-matter rule
check("ordinary translated text is written unchanged and flowing", "ย่อหน้าแรก" in raw and "  \n" not in raw)

raw = write({"p1-1": "ก่อน\n```python\nprint(1)", "p2-1": "หน้าสอง"})
check("unclosed fence from the model is escaped in a non-Markdown source", BS + "`" + BS + "`" + BS + "`python" in raw)
raw_md = write({"p1-1": "ก่อน\n```python\nprint(1)", "p2-1": "หน้าสอง"}, source_name="notes.md")
check("Markdown source: the fence is kept and closed before the page break", raw_md.index("```python") < raw_md.index("\n```\n") < raw_md.index("## หน้า 2 / 2"))
raw_md = write({"p1-1": "# หัวข้อ\n\n```py\nx = 1\n```", "p2-1": "x"}, source_name="notes.markdown")
check("Markdown source: a balanced code block and its heading stay exactly as written", "# หัวข้อ\n\n```py\nx = 1\n```" in raw_md and "```\n```" not in raw_md)
split = write({"p1-1": "ก่อน\n```py\nx = 1", "p1-2": "y = 2\n```\nหลัง", "p2-1": "หน้าสอง"}, source_name="notes.md",
              blocks=[{"id": "p1-1", "page": 1}, {"id": "p1-2", "page": 1}, {"id": "p2-1", "page": 2}])
check("a code block split over two translation blocks of one page is not closed early", split.count("```") == 2, split.count("```"))

if md is not None:
    def rendered(raw_text):
        return md.render(raw_text)

    for fence in ("```", "~~~"):
        h = rendered(write({"p1-1": "ตัวอย่าง\n" + fence + "\nโค้ดไม่ปิด", "p2-1": "หน้าสอง"}))
        check(f"rendered: unclosed {fence} does not swallow page 2's number and break", "<pre>" not in h and "<h2>หน้า 2 / 2</h2>" in h)
    h = rendered(write({"p1-1": "ย่อหน้า\n\n    เยื้อง 4 ช่อง\n\nจบ", "p2-1": "x"}))
    check("rendered: a deep indent is not a code block", "<pre>" not in h)
    visible = re.sub(r"\s+", " ", html_lib.unescape(re.sub(r"<[^>]+>", "", rendered(write({"p1-1": "ใช้แท็ก <div> และ <b>ตัวหนา</b> ได้", "p2-1": "x"})))))
    check("rendered: HTML-looking text stays visible", "<div>" in visible and "<b>ตัวหนา</b>" in visible)
    h = rendered(write({"p1-1": "ก่อน\n```py\nx = 1", "p2-1": "หน้าสอง"}, source_name="a.md"))
    check("rendered: Markdown source with an unclosed fence keeps page 2 intact", h.count("<pre>") == 1 and "<h2>หน้า 2 / 2</h2>" in h)
else:
    print("  (markdown-it not installed: rendered checks skipped)")

print(f"\nPASS {PASS}  FAIL {FAIL}")
if FAIL:
    sys.exit(1)
print("MARKDOWN_TEXT_OK")
