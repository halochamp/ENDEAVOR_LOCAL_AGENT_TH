# ENDEAVOR_LOCAL_AGENT_TH — © HaloChamp
# License: MIT License + Commons Clause — personal/educational use only, no commercial use without permission
# Website: https://www.poomwat.com | GitHub: https://github.com/halochamp | Email: champoomwat@gmail.com

"""scrape_table.py — ดึงตารางจากเว็บ JS-rendered ด้วย Playwright + pandas.read_html

ต่างจาก browser_use: code-controlled ไม่ใช่ AI agent — output deterministic
ใช้เมื่อ user แปะ URL แล้วสั่งวิเคราะห์ตาราง/ข้อมูลตัวเลข
"""
from __future__ import annotations
import sys
import os
import io

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.tools import tool
from tools._progress import phase as _phase, progress as _progress
from config import SCRAPE_TABLE_MAX_CHARS


@tool
def scrape_table(url: str, table_index: int = 0) -> str:
    """ดึงตารางจากเว็บ JS-rendered (React, Vue, SPA) ด้วย Playwright แล้วคืน CSV
    ใช้เมื่อ user แปะ URL และสั่งวิเคราะห์ตาราง/เปรียบเทียบข้อมูลตัวเลขจากเว็บนั้น

    url: URL ที่มีตาราง
    table_index: index ตารางที่ต้องการ (0=แรก, ใส่ -1 เพื่อดูรายชื่อตารางทั้งหมด)
    คืน: CSV string พร้อมส่งต่อให้ python_exec วิเคราะห์ หรือ "[error] reason"
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

    # ── Playwright: load page + wait for JS ──────────────────────────
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
                pass  # timeout on networkidle ยอมรับได้ — ใช้ HTML ที่มีอยู่
            html = page.content()
            browser.close()
        _progress(f"page loaded ({len(html)} chars)")
    except Exception as e:
        return f"[error] Playwright failed: {e}"

    # ── pandas.read_html: สกัดทุกตาราง ──────────────────────────────
    try:
        tables = pd.read_html(io.StringIO(html), flavor="lxml")
    except Exception:
        try:
            tables = pd.read_html(io.StringIO(html))
        except Exception as e:
            return f"[error] ไม่พบตารางในหน้านี้: {e}"

    if not tables:
        return f"[scrape_table] ไม่พบตารางใน {url}"

    # แสดงรายชื่อถ้า table_index = -1
    if table_index == -1:
        summary = [f"[scrape_table] พบ {len(tables)} ตารางใน {url}"]
        for i, df in enumerate(tables):
            summary.append(f"  ตาราง {i}: {df.shape[0]} แถว × {df.shape[1]} คอลัมน์ | columns: {list(df.columns[:5])}")
        return "\n".join(summary)

    if table_index >= len(tables) or table_index < -1:
        return (
            f"[error] table_index={table_index} เกินจำนวนตาราง ({len(tables)} ตาราง)\n"
            f"เรียก scrape_table(url, table_index=-1) เพื่อดูรายชื่อตารางทั้งหมด"
        )

    df = tables[table_index]
    _progress(f"table {table_index}: {df.shape[0]} rows × {df.shape[1]} cols")

    csv = df.to_csv(index=False)
    n_rows, n_cols = df.shape
    truncated = ""
    if len(csv) > SCRAPE_TABLE_MAX_CHARS:
        csv = csv[:SCRAPE_TABLE_MAX_CHARS]
        # cut to last complete line
        csv = csv[:csv.rfind("\n") + 1] if "\n" in csv else csv
        truncated = f"\n...[truncated: ตารางใหญ่ {n_rows} แถว — ใช้ python_exec อ่าน URL โดยตรงถ้าต้องการข้อมูลครบ]"

    return (
        f"[scrape_table] ตารางที่ {table_index} จาก {url}\n"
        f"{n_rows} แถว × {n_cols} คอลัมน์\n\n"
        f"{csv}{truncated}\n\n"
        f"# วิเคราะห์ต่อด้วย python_exec:\n"
        f"# import pandas as pd\n"
        f"# from io import StringIO\n"
        f'# df = pd.read_csv(StringIO("""above CSV"""))\n'
        f"# df.describe()  # สถิติพื้นฐาน"
    )
