# ENDEAVOR_LOCAL_AGENT_TH — © HaloChamp
# License: MIT License + Commons Clause — personal/educational use only, no commercial use without permission
# Website: https://www.poomwat.com | GitHub: https://github.com/halochamp | Email: champoomwat@gmail.com

"""scrape_table.py — ดึงตารางจากเว็บ JS-rendered ด้วย Playwright + pandas.read_html
   พร้อม fallback สำหรับ div-based table (เว็บที่ไม่ใช้ <table> tag จริง)

ต่างจาก browser_use: code-controlled ไม่ใช่ AI agent — output deterministic
ใช้เมื่อ user แปะ URL แล้วสั่งวิเคราะห์ตาราง/ข้อมูลตัวเลข

ลำดับการสกัด: <table> จริง (pd.read_html) → ARIA role=table/row/cell → div "row"-class
heuristic. ใช้ method แรกที่เจอ ≥1 ตาราง เท่านั้น — ไม่ผสมหลาย method ในผลลัพธ์เดียว
"""
from __future__ import annotations
import sys
import os
import io
import re
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.tools import tool
from tools._progress import phase as _phase, progress as _progress
from config import SCRAPE_TABLE_MAX_CHARS, WORKSPACE

# หา [role=table]/[role=grid] แล้วไล่ลูก [role=row] → [role=cell|gridcell|columnheader|rowheader]
_ARIA_JS = """
() => {
  const norm = s => (s || '').replace(/\\s+/g, ' ').trim();
  const roleTables = Array.from(document.querySelectorAll('[role="table"], [role="grid"]'));
  const out = [];
  for (const t of roleTables) {
    const rowEls = Array.from(t.querySelectorAll('[role="row"]'));
    const rows = rowEls
      .map(r => Array.from(r.querySelectorAll(
        '[role="cell"], [role="gridcell"], [role="columnheader"], [role="rowheader"]'
      )).map(c => norm(c.innerText)))
      .filter(r => r.length > 0);
    if (rows.length >= 2) out.push(rows);
  }
  return out;
}
"""

# หา element ที่ class มีคำว่า "row" ทั้งหมด แล้วจัดกลุ่มตาม parent เดียวกัน
# (sibling group ใหญ่สุด = แถวข้อมูลจริง, ไม่ใช่ noise เช่น class="...arrow...")
_DIV_GRID_JS = """
() => {
  const norm = s => (s || '').replace(/\\s+/g, ' ').trim();
  const rowLike = Array.from(document.querySelectorAll('[class*="row" i]'));
  const groups = new Map();
  for (const el of rowLike) {
    const parent = el.parentElement;
    if (!parent) continue;
    if (!groups.has(parent)) groups.set(parent, []);
    groups.get(parent).push(el);
  }
  const out = [];
  for (const rows of groups.values()) {
    if (rows.length < 2) continue;
    const tableRows = rows.map(rowEl => {
      let cells = Array.from(rowEl.querySelectorAll('[class*="cell" i], [class*="col" i], [class*="item" i]'));
      if (cells.length === 0) cells = Array.from(rowEl.children);
      let texts = cells.map(c => norm(c.innerText));
      if (texts.length === 0) texts = [norm(rowEl.innerText)];
      return texts;
    });
    out.push(tableRows);
  }
  return out;
}
"""


def _rows_to_df(rows: list):
    """list[list[str]] -> DataFrame, ใช้แถวแรกเป็น header. คืน None ถ้าไม่เข้าเค้าว่าเป็นตาราง
    (คอลัมน์ไม่สม่ำเสมอ หรือมีแค่ 1 คอลัมน์ — น่าจะเป็น nav/list ไม่ใช่ตาราง)."""
    import pandas as pd
    if not rows:
        return None
    counts = Counter(len(r) for r in rows)
    mode_count = counts.most_common(1)[0][0]
    if mode_count < 2:
        return None
    rows = [r for r in rows if len(r) == mode_count]
    if len(rows) < 2:
        return None
    try:
        return pd.DataFrame(rows[1:], columns=rows[0])
    except Exception:
        return None


def _csv_filename(url: str, table_index: int) -> str:
    """Auto-generate a workspace-safe filename from the URL's host (no traversal risk —
    not user-supplied, only built from a regex-stripped netloc)."""
    host = urlparse(url).netloc or "table"
    host = re.sub(r"[^a-zA-Z0-9_.-]", "_", host)
    return f"scrape_table_{host}_{table_index}.csv"


def _numeric_summary(df) -> str:
    """Best-effort sum/mean/min/max per column, after stripping common formatting
    (%, comma, currency symbols). Only reports a column if at least half its rows
    (min 2) parse as numbers — avoids misleading stats on mostly-text columns."""
    import pandas as pd
    lines = []
    min_valid = max(2, len(df) // 2)
    for i, col in enumerate(df.columns):
        series = df.iloc[:, i]  # positional — df[col] returns a DataFrame if col is duplicated
        stripped = series.astype(str).str.replace(r"[,%$฿]", "", regex=True).str.strip()
        numeric = pd.to_numeric(stripped, errors="coerce")
        valid = numeric.dropna()
        if len(valid) >= min_valid:
            lines.append(
                f"  {col}: sum={valid.sum():.2f} mean={valid.mean():.2f} "
                f"min={valid.min():.2f} max={valid.max():.2f} (n={len(valid)}/{len(df)})"
            )
    if not lines:
        return ""
    return "# สรุปตัวเลข (คอลัมน์ที่แปลงเป็นเลขได้ ≥ ครึ่งของแถว):\n" + "\n".join(lines)


def _extract_via_js(page, js: str, source_tag: str) -> list:
    """รัน JS extractor หนึ่งตัวบน live page, แปลงทุก raw table ที่ได้เป็น DataFrame.
    เรียงตามจำนวนแถวมาก→น้อย (ตารางข้อมูลจริงมักใหญ่กว่า nav/decoration)."""
    try:
        raw_tables = page.evaluate(js)
    except Exception:
        return []
    out = []
    for rows in raw_tables or []:
        df = _rows_to_df(rows)
        if df is not None and not df.empty:
            out.append((df, source_tag))
    out.sort(key=lambda x: x[0].shape[0], reverse=True)
    return out


@tool
def scrape_table(url: str, table_index: int = 0) -> str:
    """Extract a table from a JS-rendered site (React, Vue, SPA) via Playwright and return CSV.
    Use when the user pastes a URL and asks to analyze/compare numeric table data from that site.
    Falls back to ARIA role=table/row/cell, then a div class="...row..." heuristic, when the
    page has no real <table> tag (common on sites that build grids with div+CSS instead).
    Always saves the full CSV to a file in workspace; the returned string is capped at
    SCRAPE_TABLE_MAX_CHARS (10,000 chars) — read the saved file for the complete data.

    url: URL containing the table
    table_index: which table (0=first; pass -1 to list all tables found)
    Returns: CSV string ready to pass to python_exec for analysis, or "[error] reason"
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return "[error] playwright not installed — รัน: pip install playwright && playwright install chromium"

    try:
        import pandas as pd
    except ImportError:
        return "[error] pandas not installed"

    if not url.strip():
        return "[error] url is required"
    url = url.strip()
    if not url.startswith("http"):
        url = "https://" + url

    _phase(f"🌐 scrape_table: {url[:50]}")

    tables: list = []  # list[(DataFrame, source_tag)]

    # ── Playwright: load page + wait for JS, then try extraction methods in order ──
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page()
            _progress("navigating…")
            page.goto(url, timeout=30000)
            # รอ network idle = JS โหลดตารางแล้ว
            try:
                page.wait_for_load_state("networkidle", timeout=10000)
            except Exception:
                pass  # timeout on networkidle ยอมรับได้ — ใช้ DOM ที่มีอยู่
            html = page.content()
            _progress(f"page loaded ({len(html)} chars)")

            # 1) <table> จริง
            try:
                html_tables = pd.read_html(io.StringIO(html), flavor="lxml")
            except Exception:
                try:
                    html_tables = pd.read_html(io.StringIO(html))
                except Exception:
                    html_tables = []
            if html_tables:
                tables = [(df, "html-table") for df in html_tables]
            else:
                # 2) ARIA role=table/row/cell
                _progress("ไม่พบ <table> — ลอง ARIA role fallback")
                tables = _extract_via_js(page, _ARIA_JS, "aria-role")
                if not tables:
                    # 3) div class="...row..." heuristic
                    _progress("ไม่พบ ARIA table — ลอง div-grid heuristic")
                    tables = _extract_via_js(page, _DIV_GRID_JS, "div-grid")

            browser.close()
    except Exception as e:
        return f"[error] Playwright failed: {e}"

    if not tables:
        return f"[scrape_table] ไม่พบตารางใน {url} (ลองแล้ว: html-table, aria-role, div-grid)"

    # แสดงรายชื่อถ้า table_index = -1
    if table_index == -1:
        summary = [f"[scrape_table] พบ {len(tables)} ตารางใน {url}"]
        for i, (df, src) in enumerate(tables):
            summary.append(
                f"  ตาราง {i} ({src}): {df.shape[0]} แถว × {df.shape[1]} คอลัมน์ | columns: {list(df.columns[:5])}"
            )
        return "\n".join(summary)

    if table_index >= len(tables) or table_index < -1:
        return (
            f"[error] table_index={table_index} เกินจำนวนตาราง ({len(tables)} ตาราง)\n"
            f"เรียก scrape_table(url, table_index=-1) เพื่อดูรายชื่อตารางทั้งหมด"
        )

    df, src = tables[table_index]
    _progress(f"table {table_index} ({src}): {df.shape[0]} rows × {df.shape[1]} cols")

    full_csv = df.to_csv(index=False)
    n_rows, n_cols = df.shape

    # เขียนไฟล์ CSV เต็มลง workspace เสมอ — ข้อมูลที่ถูก truncate ใน state ยังหาได้จากไฟล์
    filename = _csv_filename(url, table_index)
    saved_path = ""
    try:
        out_path = Path(WORKSPACE) / filename
        out_path.write_text(full_csv, encoding="utf-8")
        saved_path = str(out_path)
    except Exception as e:
        _progress(f"[warn] write CSV to workspace failed: {e}")

    try:
        stats_block = _numeric_summary(df)
    except Exception:
        stats_block = ""

    # ส่งเข้า state (return string) ไม่เกิน SCRAPE_TABLE_MAX_CHARS รวมทั้งหมด — กันงบให้
    # stats_block ก่อน เพื่อไม่ให้ตัวเลขสำคัญ (sum/mean) หลุดไปตอน truncate CSV body
    reserve = len(stats_block) + 80 if stats_block else 0
    budget = max(SCRAPE_TABLE_MAX_CHARS - reserve, 500)
    csv = full_csv
    truncated = ""
    if len(csv) > budget:
        csv = csv[:budget]
        # cut to last complete line
        csv = csv[:csv.rfind("\n") + 1] if "\n" in csv else csv
        truncated = f"\n...[truncated: ตารางใหญ่ {n_rows} แถว — ข้อมูลครบอยู่ที่ {saved_path or filename}]"

    saved_line = f"\nบันทึกไฟล์ครบที่: {saved_path}" if saved_path else ""
    stats_line = f"\n\n{stats_block}" if stats_block else ""

    return (
        f"[scrape_table] ตารางที่ {table_index} จาก {url} (source: {src}){saved_line}\n"
        f"{n_rows} แถว × {n_cols} คอลัมน์{stats_line}\n\n"
        f"{csv}{truncated}\n\n"
        f"# วิเคราะห์ต่อด้วย python_exec:\n"
        f"# import pandas as pd\n"
        f"# from io import StringIO\n"
        f'# df = pd.read_csv(StringIO("""above CSV"""))\n'
        f"# df.describe()  # สถิติพื้นฐาน"
    )
