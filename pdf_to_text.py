# ENDEAVOR_LOCAL_AGENT_TH — © HaloChamp
# License: MIT License + Commons Clause — personal/educational use only, no commercial use without permission
# Website: https://www.poomwat.com | GitHub: https://github.com/halochamp | Email: champoomwat@gmail.com

"""Direct PDF -> Markdown conversion for the Agent TH Electron panel ("PDF to MD").

Ported from Agent Lite's deterministic-first pipeline, with the currently selected
model/port used only when the user explicitly enables LLM rewrite.

The pipeline is deliberately deterministic-first and SSD-first:

1. Read one physical PDF page at a time.
2. Use the embedded text layer when present; otherwise rasterize only that page
   and run Apple Vision OCR.
3. Reconstruct a confident OCR grid with the same conservative table logic used
   by the read_image sensor.
4. Persist each page into SQLite immediately.  The complete document is never
   held as one Python string.
5. When the user ticks "rewrite" (rewrite_thai), send EVERY page (native, OCR,
   OCR tables, image-only; any language) through the currently selected
   model ONE PAGE AT A TIME, giving it the page image and the extracted text; it
   writes the page as Markdown (headings, lists, tables) reading mainly from the
   image and describes any figure, chart, map or photo it contains.  The model's
   page is written as it is (Markdown from an image can never be checked to 100%,
   so there is no comparison with the extraction); a page whose request fails,
   comes back empty or is cut off at max_tokens keeps its deterministic text.
   Unticked = no LLM call at all and the deterministic text is written as-is.
6. Stream the final rows back out in physical PDF page order with explicit page
   separators.

SQLite is staging/cache state only.  The user-facing .md output is written to
workspace/pdf_to_md/; old staging rows are aged out automatically.
"""
from __future__ import annotations

from collections.abc import Callable
import base64
import hashlib
import math
import re
import sqlite3
import tempfile
import time
import unicodedata
import uuid
from pathlib import Path

import httpx

import config
from markdown_text import text_to_markdown as _text_to_markdown
from tools._ocr import read_layout


Progress = Callable[[str, int, int, dict[str, int]], None]

_THAI_RE = re.compile(r"[\u0E00-\u0E7F]")
_THINK_RE = re.compile(r"<think>[\s\S]*?</think>", re.IGNORECASE)
_PROSE_COL_CHARS = 14
_THAI_BASE_OR_MARK_BEFORE_SPACE_RE = re.compile(
    r"(?<=[\u0E01-\u0E2E\u0E30-\u0E33\u0E40-\u0E45\u0E31\u0E34-\u0E3A\u0E47-\u0E4E])"
    r"\s+(?=[\u0E31\u0E34-\u0E3A\u0E47-\u0E4E])"
)
_THAI_MARK_BEFORE_BASE_AFTER_SPACE_RE = re.compile(
    r"(?<=[\u0E31\u0E34-\u0E3A\u0E47-\u0E4E])"
    r"\s+(?=[\u0E01-\u0E2E\u0E30-\u0E33\u0E40-\u0E45])"
)
_THAI_SARA_AM_BEFORE_TONE_RE = re.compile(r"\u0E33([\u0E48-\u0E4B])")
# Two more breaks that no Thai word contains: a space before SARA A/AA/AM ("ก า ร") and a space after a
# leading vowel ("เ รื่อง").  Deliberately NOT RagDoc's blanket removal of every Thai-to-Thai space: that
# exists for search indexing and would also glue real words ("ฉัน กิน ข้าว") and Thai digits ("จำนวน ๑๘๐ คน").
_THAI_SPACE_BEFORE_SARA_RE = re.compile(
    r"(?<=[\u0E01-\u0E2E\u0E31\u0E34-\u0E3A\u0E47-\u0E4E])[ \t]+(?=[\u0E30\u0E32\u0E33])"
)
_THAI_SPACE_AFTER_LEADING_VOWEL_RE = re.compile(r"(?<=[\u0E40-\u0E44])[ \t]+(?=[\u0E01-\u0E2E])")


def repair_thai_orthographic_order(text: str) -> str:
    """Agent Lite's deterministic Thai mark-order repair, copied verbatim in policy."""
    return _THAI_SARA_AM_BEFORE_TONE_RE.sub(lambda match: match.group(1) + "ำ", text)


def repair_thai_combining_mark_spacing(text: str) -> str:
    """Remove only clearly broken spaces around Thai combining marks."""
    while True:
        cleaned = _THAI_BASE_OR_MARK_BEFORE_SPACE_RE.sub("", text)
        cleaned = _THAI_MARK_BEFORE_BASE_AFTER_SPACE_RE.sub("", cleaned)
        cleaned = _THAI_SPACE_BEFORE_SARA_RE.sub("", cleaned)
        cleaned = _THAI_SPACE_AFTER_LEADING_VOWEL_RE.sub("", cleaned)
        if cleaned == text:
            return cleaned
        text = cleaned


def _connect() -> sqlite3.Connection:
    db_path = Path(config.PDF_TO_TEXT_DB)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=30)
    try:
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute(
            "CREATE TABLE IF NOT EXISTS pdf_text_jobs ("
            "job_id TEXT PRIMARY KEY, source_name TEXT NOT NULL, source_sha256 TEXT NOT NULL, "
            "total_pages INTEGER NOT NULL, rewrite_thai INTEGER NOT NULL, status TEXT NOT NULL, "
            "output_path TEXT, warning TEXT, created_at REAL NOT NULL, completed_at REAL)"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS pdf_text_pages ("
            "job_id TEXT NOT NULL, page_number INTEGER NOT NULL, source_kind TEXT NOT NULL, "
            "raw_text TEXT NOT NULL, final_text TEXT NOT NULL, rewrite_status TEXT NOT NULL, "
            "PRIMARY KEY(job_id, page_number))"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS pdf_text_pages_job_idx "
            "ON pdf_text_pages(job_id, page_number)"
        )
        conn.commit()
    except BaseException:
        conn.close()
        raise
    return conn


def _cleanup_cache(conn: sqlite3.Connection) -> None:
    cutoff = time.time() - config.PDF_TO_TEXT_CACHE_MAX_AGE_SECONDS
    stale = [
        row[0]
        for row in conn.execute(
            "SELECT job_id FROM pdf_text_jobs WHERE created_at < ?", (cutoff,)
        )
    ]
    if not stale:
        return
    conn.executemany("DELETE FROM pdf_text_pages WHERE job_id=?", ((job,) for job in stale))
    conn.executemany("DELETE FROM pdf_text_jobs WHERE job_id=?", ((job,) for job in stale))
    conn.commit()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clean_text(text: str) -> str:
    """Conservative layout cleanup that keeps page/paragraph line breaks."""
    lines = []
    for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = unicodedata.normalize("NFC", line.rstrip())
        line = repair_thai_combining_mark_spacing(line)
        lines.append(repair_thai_orthographic_order(line))
    # Keep intentional blank lines but avoid unlimited vertical whitespace.
    out: list[str] = []
    blank = 0
    for line in lines:
        if line.strip():
            blank = 0
            out.append(line)
        else:
            blank += 1
            if blank <= 2:
                out.append("")
    return "\n".join(out).strip()


def _column_intervals(boxes: list[dict], min_gap: float = 0.025) -> list[tuple[float, float]]:
    """OCR table column projection."""
    spans = sorted(
        ((box["x"], box["x"] + box["w"]) for box in boxes if box["w"] <= 0.4),
        key=lambda span: span[0],
    )
    if not spans:
        return []
    columns: list[list[float]] = [list(spans[0])]
    for lo, hi in spans[1:]:
        if lo - columns[-1][1] <= min_gap:
            columns[-1][1] = max(columns[-1][1], hi)
        else:
            columns.append([lo, hi])
    return [(lo, hi) for lo, hi in columns]


def _assign_column(box: dict, columns: list[tuple[float, float]]) -> int:
    left, right = box["x"], box["x"] + box["w"]
    best_index, best_overlap = 0, -1.0
    for index, (lo, hi) in enumerate(columns):
        overlap = min(right, hi) - max(left, lo)
        if overlap > best_overlap:
            best_overlap, best_index = overlap, index
    if best_overlap > 0:
        return best_index
    center = left + box["w"] / 2
    return min(
        range(len(columns)),
        key=lambda index: abs((columns[index][0] + columns[index][1]) / 2 - center),
    )


def reconstruct_ocr_table(boxes: list[dict]) -> str | None:
    """Conservative OCR grid reconstruction (shared policy with the read_image sensor)."""
    if len(boxes) < 4:
        return None

    items = sorted(boxes, key=lambda box: -(box["y"] + box["h"] / 2))
    heights = sorted(box["h"] for box in boxes)
    median_height = heights[len(heights) // 2]
    row_threshold = max(median_height * 0.6, 0.005)

    rows: list[list[dict]] = [[items[0]]]
    for box in items[1:]:
        center = box["y"] + box["h"] / 2
        row_center = sum(item["y"] + item["h"] / 2 for item in rows[-1]) / len(rows[-1])
        if row_center - center <= row_threshold:
            rows[-1].append(box)
        else:
            rows.append([box])

    columns = _column_intervals(boxes)
    if len(columns) < 2 or len(rows) < 2:
        return None

    grid: list[list[str]] = []
    for row in rows:
        cells = [""] * len(columns)
        for box in sorted(row, key=lambda item: item["x"]):
            column = _assign_column(box, columns)
            cells[column] = (
                (cells[column] + " " + box["text"]).strip()
                if cells[column]
                else box["text"]
            )
        grid.append(cells)

    multi_rows = sum(1 for row in grid if sum(1 for cell in row if cell) >= 2)
    if multi_rows < max(2, len(grid) * 0.5):
        return None

    filled = sum(1 for row in grid for cell in row if cell)
    if filled < len(grid) * len(columns) * 0.6:
        return None

    flat = [cell for row in grid for cell in row if cell]
    numeric = sum(1 for cell in flat if any(char.isdigit() for char in cell))
    if flat and numeric / len(flat) < 0.15:
        column_medians = []
        for column in range(len(columns)):
            lengths = sorted(
                len(grid[row][column])
                for row in range(len(grid))
                if grid[row][column]
            )
            if lengths:
                column_medians.append(lengths[len(lengths) // 2])
        if column_medians and min(column_medians) > _PROSE_COL_CHARS:
            return None

    def render(row: list[str]) -> str:
        return "| " + " | ".join(cell.replace("|", r"\|") for cell in row) + " |"

    lines = [
        render(grid[0]),
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    lines.extend(render(row) for row in grid[1:])
    return "\n".join(lines)


def _ocr_page(page, page_number: int) -> tuple[str, str]:
    """Return (source_kind, text) for one image-only physical PDF page."""
    temporary: Path | None = None
    try:
        import fitz

        work_dir = Path(config.PDF_TO_TEXT_PRIVATE_DIR)
        work_dir.mkdir(parents=True, exist_ok=True)
        work_dir.chmod(0o700)
        with tempfile.NamedTemporaryFile(
            prefix=f"page-{page_number:05d}-",
            suffix=".png",
            dir=work_dir,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)

        rect = page.rect
        source_pixels = float(rect.width) * float(rect.height)
        if source_pixels <= 0:
            raise RuntimeError(f"PDF page {page_number} has invalid dimensions")
        requested_scale = config.PDF_TO_TEXT_OCR_DPI / 72.0
        scale = requested_scale
        if source_pixels * scale * scale > config.PDF_TO_TEXT_OCR_MAX_PIXELS:
            scale = math.sqrt(config.PDF_TO_TEXT_OCR_MAX_PIXELS / source_pixels)
        if scale <= 0:
            raise RuntimeError(f"PDF page {page_number} is too large to rasterize safely")
        page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False).save(str(temporary))
        boxes = read_layout(str(temporary))
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

    cleaned_boxes = []
    for item in boxes or ():
        # per fragment, so table cells get the same spacing repair as prose ("น้ ำ" -> "น้ำ")
        text = repair_thai_combining_mark_spacing(
            repair_thai_orthographic_order(str(item.get("text") or "").strip())
        )
        if not text:
            continue
        clone = dict(item)
        clone["text"] = text
        cleaned_boxes.append(clone)

    if not cleaned_boxes:
        return "ocr", ""

    table = reconstruct_ocr_table(cleaned_boxes)
    if table:
        return "ocr_table", table

    return "ocr", _clean_text("\n".join(item["text"] for item in cleaned_boxes))


def _rewrite_prompt(text: str) -> str:
    return (
        "ภาพคือหน้า PDF หนึ่งหน้า ข้อความด้านล่างคือข้อความที่สกัดจากหน้านี้ด้วย OCR หรือ text layer "
        "ซึ่งอาจมีตัวอักษรผิดหรือเว้นวรรคผิด ให้เขียนเนื้อหาของหน้านี้เป็น Markdown ที่อ่านง่าย "
        "โดยอ่านจากภาพเป็นหลักและใช้ข้อความเป็นตัวช่วย: "
        "ใช้ # หัวข้อเมื่อเป็นหัวเรื่องชัดเจน, - สำหรับรายการ, ตาราง | เมื่อเป็นตาราง, "
        "เว้นบรรทัดว่างระหว่างย่อหน้า, **ตัวหนา** เฉพาะส่วนที่เน้นชัดเจน "
        "แก้ตัวอักษรที่ OCR อ่านผิดโดยเทียบกับภาพ ห้ามสรุป ห้ามแปล ห้ามเพิ่มเนื้อหาที่ไม่มีในหน้า ห้ามตัดเนื้อหา "
        "ห้ามเดาตัวอักษรหรือตัวเลขที่อ่านไม่ออก (ถ้าอ่านจากภาพไม่ออกให้ใช้ตามข้อความที่สกัดมา) "
        "ลายมือที่อ่านออกชัดเจนให้ใส่ไว้ในตำแหน่งที่ปรากฏ ลายเซ็นและตราประทับให้ข้าม "
        "หากในหน้ามีภาพ แผนที่ กราฟ แผนภูมิ ไดอะแกรม หรือรูปถ่าย ให้ถอดข้อความที่อยู่ในภาพตามจริง "
        "แล้วเพิ่มคำบรรยายสั้นๆ ตรงตำแหน่งภาพว่าภาพนี้แสดงอะไรและสื่อความหมายอะไร "
        "(กราฟ: แกน แนวโน้ม ค่าสำคัญ; แผนที่: พื้นที่และสัญลักษณ์; รูปถ่าย: สิ่งที่เห็น) "
        "ในรูปแบบ > **[ภาพ]** คำบรรยาย โดยเขียนเฉพาะสิ่งที่เห็นจริงในภาพ ห้ามเดา "
        "หน้าที่ไม่มีเนื้อหาเลยให้ตอบว่างเปล่า "
        "รักษาลำดับเนื้อหาตามหน้า เลขไทย ๐-๙ ต้องคงเป็นเลขไทย ห้ามแปลงเป็นเลขอารบิก "
        "ข้อที่ขึ้นต้นด้วยเลขไทย (เช่น ๑. ๒.๑) ให้เขียนเป็นบรรทัดข้อความธรรมดาคั่นด้วยบรรทัดว่าง ไม่ใช้ลิสต์ Markdown "
        "ตอบเป็น Markdown ของหน้านี้อย่างเดียว ไม่มีคำอธิบาย ไม่ใส่ code fence\n\n"
        "ข้อความที่สกัดได้:\n" + (text.strip() or "(ไม่มี — หน้านี้อาจเป็นภาพล้วน)")
    )


def _strip_page_fence(text: str) -> str:
    """Drop a ``` fence the model wrapped around a whole page."""
    match = re.fullmatch(r"\s*```[A-Za-z]*\n(.*?)\n?```\s*", text, flags=re.DOTALL)
    return match.group(1) if match else text


def _page_image_data_url(page, page_number: int) -> str:
    """Render one physical PDF page to a PNG data URL whose longest side is VISION_MAX_SIDE pixels."""
    rect = page.rect
    longest = max(float(rect.width), float(rect.height))
    if longest <= 0:
        raise RuntimeError(f"PDF page {page_number} has invalid dimensions")
    import fitz

    scale = config.PDF_TO_TEXT_VISION_MAX_SIDE / longest
    png = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False).tobytes("png")
    return "data:image/png;base64," + base64.b64encode(png).decode()


class _ReplyTruncated(Exception):
    """The model hit max_tokens mid-page: its Markdown would silently lose the end of the page."""


def _page_payload(model: str, text: str, image_url: str | None) -> dict:
    """The request body; ``image_url`` None = a model that cannot see images gets the extracted text only."""
    prompt = _rewrite_prompt(text)
    if image_url:
        user_content: object = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]
    else:
        user_content = prompt + "\n\n(ไม่มีภาพของหน้านี้ — โมเดลที่เลือกอ่านภาพไม่ได้ ให้ใช้เฉพาะข้อความที่สกัดมาข้างต้น)"
    return {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a careful PDF-to-Markdown transcriber. You get one page image (when the model can see "
                    "images) and the text extracted from it. Output the page as Markdown only. "
                    "Never summarize, translate or invent content."
                ),
            },
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.0,
        # Markdown is a little longer than the plain text it came from.
        "max_tokens": max(4096, min(12000, int(len(text) * 1.6))),
        "enable_thinking": False,
        "thinking_budget": 0,
        "repetition_penalty": config.REPETITION_PENALTY,
    }


def _post_page(url: str, payload: dict) -> dict:
    headers = {
        "Authorization": f"Bearer {config.API_KEY}",
        "Content-Type": "application/json",
        "X-APC-Tenant": config.APC_BACKGROUND_TENANT,
    }
    with httpx.Client(timeout=config.PDF_TO_TEXT_LLM_TIMEOUT_SECONDS) as client:
        response = client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()


def _llm_rewrite_page(page_number: int, text: str, image_url: str) -> str:
    """One local no-think request: the page image plus its extracted text -> Markdown.

    The image is sent only to a model that can see it: a model already known to be text-only gets the extracted text alone,
    and if the server refuses the image the model is remembered as text-only and the page is retried once without it."""
    from tools._vision_capability import TEXT_ONLY, get_capability, is_unsupported_image_error, mark_text_only

    base, model = config.get_mlx_base_url(), config.get_model()
    url = base.rstrip("/") + "/chat/completions"
    with_image = get_capability(base, model) != TEXT_ONLY
    try:
        body = _post_page(url, _page_payload(model, text, image_url if with_image else None))
    except Exception as error:
        if not (with_image and is_unsupported_image_error(error)):
            raise
        mark_text_only(base, model)
        body = _post_page(url, _page_payload(model, text, None))
    choice = body["choices"][0]
    if choice.get("finish_reason") == "length":
        raise _ReplyTruncated(f"page {page_number} reply was cut off at max_tokens")
    content = str(choice["message"]["content"])
    return _strip_page_fence(_THINK_RE.sub("", content).strip())


_MAX_CONSECUTIVE_LLM_FAILURES = 3
_MAX_REWRITE_WARNINGS = 10


def _apply_thai_rewrite(
    conn: sqlite3.Connection,
    job_id: str,
    total_pages: int,
    progress: Progress | None,
    pdf_path: Path,
) -> tuple[int, int, list[str]]:
    """Ticked "rewrite": every page goes through the model, one page at a
    time, with its image and its extracted text.  Returns (rewritten, fallback, warnings)."""
    # Ticked means everything: every page, whatever its source or language.  A page with no extracted text
    # (a photo, a map, a blank scan) is sent too: the model transcribes and describes what the image shows.
    rows = [
        (int(page), str(text or ""))
        for page, text in conn.execute(
            "SELECT page_number, raw_text FROM pdf_text_pages "
            "WHERE job_id=? ORDER BY page_number",
            (job_id,),
        )
    ]
    if not rows:
        return 0, 0, []

    def mark_fallback(page: int) -> None:
        conn.execute(
            "UPDATE pdf_text_pages SET rewrite_status='fallback' WHERE job_id=? AND page_number=?",
            (job_id, page),
        )
        conn.commit()

    rewritten_count = 0
    fallback_count = 0
    warnings: list[str] = []
    try:
        import fitz

        pdf = fitz.open(pdf_path)
    except Exception as exc:
        for page, _ in rows:
            mark_fallback(page)
        return 0, len(rows), [f"Could not open the PDF to send page images to the model: {type(exc).__name__}; using the original text"]

    consecutive_failures = 0
    try:
        for index, (page, source) in enumerate(rows, 1):
            if progress:
                progress(
                    "rewrite", page, total_pages,
                    {"index": index, "pages": len(rows), "rewritten": rewritten_count},
                )
            if consecutive_failures >= _MAX_CONSECUTIVE_LLM_FAILURES:
                # The model server is not answering; do not wait out a timeout per remaining page.
                mark_fallback(page)
                fallback_count += 1
                continue
            try:
                # Sub-steps, so the panel can say what the wait is for: drawing the page, then the model's reply.
                step_info = {"index": index, "pages": len(rows), "rewritten": rewritten_count}
                if progress:
                    progress("render", page, total_pages, step_info)
                image_url = _page_image_data_url(pdf[page - 1], page)
                if progress:
                    progress("model", page, total_pages, step_info)
                answer = _llm_rewrite_page(page, source, image_url)
                consecutive_failures = 0
            except _ReplyTruncated:
                fallback_count += 1
                mark_fallback(page)
                if len(warnings) < _MAX_REWRITE_WARNINGS:
                    warnings.append(f"The model's reply for page {page} was cut off (max_tokens); using the original text")
                continue
            except Exception as exc:
                consecutive_failures += 1
                fallback_count += 1
                mark_fallback(page)
                if len(warnings) < _MAX_REWRITE_WARNINGS:
                    warnings.append(f"The model request for page {page} failed: {type(exc).__name__}; using the original text")
                if consecutive_failures == _MAX_CONSECUTIVE_LLM_FAILURES:
                    warnings.append("The model did not answer 3 pages in a row — the remaining pages use the original text")
                continue
            # The model's page is written as it is: Markdown from the image and the text can never be
            # checked to 100%, so there is no comparison against the extraction -- only the spacing repair.
            candidate = _clean_text(answer)
            if not candidate:
                if source.strip():
                    mark_fallback(page)       # an empty answer for a page that has text: keep the text
                    fallback_count += 1
                continue                      # an empty page the model also found empty: nothing to write
            conn.execute(
                "UPDATE pdf_text_pages SET final_text=?, rewrite_status='rewritten' "
                "WHERE job_id=? AND page_number=?",
                (candidate, job_id, page),
            )
            conn.commit()
            rewritten_count += 1
    finally:
        pdf.close()
    return rewritten_count, fallback_count, warnings


def _safe_output_path(source_name: str) -> Path:
    stem = Path(source_name).stem.strip() or "document"
    stem = re.sub(r"[\\/:*?\"<>|\x00-\x1f]+", "_", stem).strip(" .") or "document"
    output_dir = Path(config.PDF_TO_TEXT_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    base = output_dir / f"{stem}_pdf_to_md.md"
    if not base.exists():
        return base
    for index in range(2, 10000):
        candidate = output_dir / f"{stem}_pdf_to_md_{index}.md"
        if not candidate.exists():
            return candidate
    raise RuntimeError("Could not create an output file name that does not collide with an existing file")


def _write_output(conn: sqlite3.Connection, job_id: str, total_pages: int, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        for page_number, text, source_kind, status in conn.execute(
            "SELECT page_number, final_text, source_kind, rewrite_status FROM pdf_text_pages "
            "WHERE job_id=? ORDER BY page_number",
            (job_id,),
        ):
            # The model's pages are Markdown already; a reconstructed OCR table is already a pipe table.
            if text and status != "rewritten" and source_kind != "ocr_table":
                text = _text_to_markdown(str(text))
            # Page number and page break are written by code, never by the model: a hidden marker for tools,
            # a visible bold "หน้า N/T" line (not a heading, so it cannot clash with the model's headings),
            # and "---" between pages.
            handle.write(f"<!-- PDF page {page_number}/{total_pages} -->\n**หน้า {page_number}/{total_pages}**\n\n")
            if text:
                handle.write(str(text).rstrip())
            handle.write("\n")
            if page_number != total_pages:
                handle.write("\n---\n\n")


def convert_pdf(
    source: str | Path,
    *,
    source_name: str | None = None,
    rewrite_thai: bool = True,
    progress: Progress | None = None,
) -> dict[str, object]:
    """Convert a PDF into page-delimited text and return a compact job summary."""
    path = Path(source).expanduser().resolve()
    if not path.is_file() or path.suffix.casefold() != ".pdf":
        raise ValueError("The source must be an existing PDF file")
    size = path.stat().st_size
    if size <= 0:
        raise ValueError("The PDF is empty")
    if size > config.PDF_TO_TEXT_MAX_BYTES:
        raise ValueError(
            f"The PDF is larger than {config.PDF_TO_TEXT_MAX_BYTES // (1024 * 1024)}MB"
        )

    try:
        import fitz
        pdf = fitz.open(path)
    except Exception as exc:
        raise ValueError(f"Could not open the PDF: {exc}") from exc

    name = Path(source_name or path.name).name
    job_id = uuid.uuid4().hex
    stats = {
        "native": 0,
        "ocr": 0,
        "tables": 0,
        "rewritten": 0,
        "rewrite_fallback": 0,
    }
    warnings: list[str] = []
    try:
        conn = _connect()
    except BaseException:
        pdf.close()
        raise
    try:
        _cleanup_cache(conn)
        total_pages = len(pdf)
        if total_pages <= 0:
            raise ValueError("The PDF has no pages")
        conn.execute(
            "INSERT INTO pdf_text_jobs("
            "job_id, source_name, source_sha256, total_pages, rewrite_thai, status, created_at"
            ") VALUES(?,?,?,?,?,'running',?)",
            (job_id, name, _sha256(path), total_pages, int(bool(rewrite_thai)), time.time()),
        )
        conn.commit()

        for page_number, page in enumerate(pdf, 1):
            if progress:
                progress("extract", page_number, total_pages, dict(stats))
            native = _clean_text(page.get_text("text") or "")
            native_signal_chars = sum(char.isalnum() for char in native)
            if native and native_signal_chars >= config.PDF_TO_TEXT_NATIVE_MIN_CHARS:
                source_kind, text = "native", native
                stats["native"] += 1
            else:
                # A hybrid page can contain a tiny native footer/page number
                # over a scanned body. Treat sparse native text as a hint, not
                # proof that the page is born-digital: OCR once and keep the
                # richer result only when it materially adds content.
                if progress:
                    progress("ocr", page_number, total_pages, dict(stats))
                ocr_kind, ocr_text = _ocr_page(page, page_number)
                if native and len(ocr_text.strip()) <= len(native.strip()) + 24:
                    source_kind, text = "native", native
                    stats["native"] += 1
                else:
                    source_kind, text = ocr_kind, ocr_text
                    if source_kind == "ocr_table":
                        stats["tables"] += 1
                    else:
                        stats["ocr"] += 1
            conn.execute(
                "INSERT INTO pdf_text_pages("
                "job_id, page_number, source_kind, raw_text, final_text, rewrite_status"
                ") VALUES(?,?,?,?,?,'not_needed')",
                (job_id, page_number, source_kind, text, text),
            )
            conn.commit()
    except BaseException:
        conn.close()
        raise
    finally:
        pdf.close()

    try:
        if rewrite_thai:
            rewritten, fallback, rewrite_warnings = _apply_thai_rewrite(
                conn, job_id, total_pages, progress, path
            )
            stats["rewritten"] = rewritten
            stats["rewrite_fallback"] = fallback
            warnings.extend(rewrite_warnings)

        if progress:
            progress("write", total_pages, total_pages, dict(stats))
        output = _safe_output_path(name)
        _write_output(conn, job_id, total_pages, output)
        warning_text = "\n".join(warnings)
        conn.execute(
            "UPDATE pdf_text_jobs SET status='done', output_path=?, warning=?, completed_at=? "
            "WHERE job_id=?",
            (str(output), warning_text, time.time(), job_id),
        )
        conn.commit()
        return {
            "job_id": job_id,
            "source_name": name,
            "total_pages": total_pages,
            "output_path": str(output),
            "rewrite_thai": bool(rewrite_thai),
            "stats": stats,
            "warnings": warnings,
        }
    except Exception as exc:
        conn.execute(
            "UPDATE pdf_text_jobs SET status='error', warning=?, completed_at=? WHERE job_id=?",
            (f"{type(exc).__name__}: {exc}", time.time(), job_id),
        )
        conn.commit()
        raise
    finally:
        conn.close()
