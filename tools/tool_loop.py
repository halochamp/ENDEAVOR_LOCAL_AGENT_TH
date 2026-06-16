# ENDEAVOR_LOCAL_AGENT_TH — © HaloChamp
# License: MIT License + Commons Clause — personal/educational use only, no commercial use without permission
# Website: https://www.poomwat.com | GitHub: https://github.com/halochamp | Email: champoomwat@gmail.com

"""tool_loop.py — Generic Python-driven loop tool

LLM เรียกครั้งเดียว → Python loop จัดการ iteration ทั้งหมดโดยไม่หลุด
ไม่ต้อง "continue" หรือ "ต่อ" เอง

Actions:
  search_and_browse — DDG search แต่ละ keyword → รวม URLs → fetch+summarize
  browse_summarize  — fetch+summarize แต่ละ URL ตรงๆ
  read_file         — อ่าน+summarize แต่ละ file path
  bash_each         — รัน bash command แต่ละ item (ผ่าน sandbox)
"""
from __future__ import annotations

import datetime
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.tools import tool

from tools._progress import phase as _phase, progress as _progress
from tools._summarize import summarize
from config import (
    TOOL_LOOP_DDG_MAX_RESULTS,
    TOOL_LOOP_READ_MAX_CHARS,
    TOOL_LOOP_READ_SUMMARIZE_THRESHOLD,
    TOOL_LOOP_BASH_MAX_CHARS,
    TOOL_LOOP_BASH_TIMEOUT,
)

log = logging.getLogger(__name__)

_VALID_ACTIONS = {"search_and_browse", "browse_summarize", "read_file", "bash_each"}
_TOOL_LOOP_WEB_MAX = 100  # tool_loop raises the per-turn web cap to this when needed


# ── DDG search ─────────────────────────────────────────────────────────────────

def _ddg_search(query: str, max_results: int = TOOL_LOOP_DDG_MAX_RESULTS) -> list[str]:
    try:
        from ddgs import DDGS
        time.sleep(1.0)
        results = list(DDGS().text(query, max_results=max_results, region="wt-wt"))
        return [r.get("href", "") for r in results if r.get("href")]
    except Exception as e:
        log.warning(f"DDG search failed: {e}")
        time.sleep(3.0)
        try:
            from ddgs import DDGS
            results = list(DDGS().text(query, max_results=max_results, region="wt-wt"))
            return [r.get("href", "") for r in results if r.get("href")]
        except Exception:
            return []


def _collect_urls(keywords: list[str], max_n: int) -> list[str]:
    seen: set[str] = set()
    urls: list[str] = []
    for kw in keywords:
        if len(urls) >= max_n:
            break
        _phase(f"🔍 ค้นหา: {kw[:60]}")
        for u in _ddg_search(kw):
            if u and u not in seen and u.startswith("http"):
                seen.add(u)
                urls.append(u)
        _progress(f"URLs: {len(urls)}/{max_n}")
    return urls[:max_n]


# ── Action handlers ─────────────────────────────────────────────────────────────

def _browse_summarize(url: str, context: str, idx: int, total: int) -> dict:
    from tools.browse_url import _fetch_jina
    from tools import web_cache
    from tools.web_cache import web_count_check as _wc_check, web_count_inc as _wc_inc

    if _wc_check():
        return {"title": "[web limit reached]", "summary": "[error] web call limit reached", "error": True}
    _wc_inc()
    try:
        cached = web_cache.get_summary(url, context)
        if cached is None:
            raw = web_cache.get(url)
            if raw is None:
                raw = _fetch_jina(url)
                if raw.startswith("[error]"):
                    return {"title": "[ดึงไม่ได้]", "summary": raw, "error": True}
                web_cache.put(url, raw)
            cached = summarize(raw, url=url, user_query=context)
            web_cache.put_summary(url, context, cached)
        lines = [l.strip() for l in cached.split("\n") if l.strip()]
        title = lines[0][:80] if lines else url.split("/")[-1][:60]
        _progress(f"[{idx}/{total}] {title[:50]}")
        return {"url": url, "title": title, "summary": cached, "error": False}
    except Exception as e:
        return {"url": url, "title": "[error]", "summary": str(e), "error": True}


def _read_file(path: str, context: str, idx: int, total: int) -> dict:
    try:
        from tools._safety import resolve_read_path
        safe = resolve_read_path(path)
        with open(safe, encoding="utf-8", errors="replace") as f:
            content = f.read()
        _progress(f"[{idx}/{total}] {os.path.basename(path)} ({len(content):,} chars)")
        body = summarize(content[:TOOL_LOOP_READ_MAX_CHARS], url=path, user_query=context or "summarize") if len(content) > TOOL_LOOP_READ_SUMMARIZE_THRESHOLD else content
        return {"path": path, "title": os.path.basename(path), "summary": body, "error": False}
    except Exception as e:
        return {"path": path, "title": os.path.basename(path), "summary": str(e), "error": True}


def _bash_each(command: str, idx: int, total: int) -> dict:
    import subprocess
    from config import WORKSPACE
    from tools.bash import _build_sandbox_profile
    import tempfile

    _progress(f"[{idx}/{total}] {command[:60]}")
    profile_path = None
    try:
        profile = _build_sandbox_profile(WORKSPACE)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".sb", delete=False) as pf:
            pf.write(profile)
            profile_path = pf.name
        result = subprocess.run(
            ["sandbox-exec", "-f", profile_path, "bash", "-c", command],
            capture_output=True, text=True, timeout=TOOL_LOOP_BASH_TIMEOUT, cwd=WORKSPACE,
        )
        output = (result.stdout + result.stderr).strip()
        if len(output) > TOOL_LOOP_BASH_MAX_CHARS:
            output = output[:TOOL_LOOP_BASH_MAX_CHARS] + "\n...[truncated]"
        return {"cmd": command, "title": command[:60], "summary": output or "(no output)", "error": result.returncode != 0}
    except Exception as e:
        return {"cmd": command, "title": command[:60], "summary": str(e), "error": True}
    finally:
        if profile_path:
            try:
                os.unlink(profile_path)
            except Exception:
                pass


# ── Output writer ───────────────────────────────────────────────────────────────

def _write_output(path: str, results: list[dict], context: str, action: str) -> None:
    today = datetime.date.today().strftime("%Y-%m-%d")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {context or 'Loop Results'}\nDate: {today} | Action: {action} | Items: {len(results)}\n\n---\n\n")
        for i, r in enumerate(results, 1):
            title = r.get("title", f"Item {i}")
            summary = r.get("summary", "")
            ref = r.get("url", r.get("path", r.get("cmd", "")))
            f.write(f"## {i}. {title}\n")
            if ref and ref != title:
                f.write(f"*{ref}*\n\n")
            f.write(f"{summary}\n\n---\n\n")


# ── Main tool ───────────────────────────────────────────────────────────────────

@tool
def tool_loop(
    items: list[str],
    action: str,
    context: str = "",
    max_n: int = 50,
    output_file: str = "",
) -> str:
    """วน loop ประมวลผล items ด้วย Python — ไม่หลุด loop ไม่ว่า items จะมากแค่ไหน

    LLM เรียก tool นี้ครั้งเดียว → Python loop จัดการทุก iteration อัตโนมัติ ไม่ต้อง "continue" เอง
    ใช้เมื่อต้องทำงานเดิมซ้ำกับ input หลายตัว เช่น สรุป 30 เว็บ อ่าน 10 ไฟล์ รัน 20 คำสั่ง

    SCALE GATE — เลือก tool ก่อนเรียก:
      browse_summarize + N ≤ 8 + ไม่ต้องการ output_file  →  batch_browse แทน (parallel, เร็วกว่า)
      browse_summarize + N > 8  หรือ  ต้องการ output_file →  tool_loop
      search_and_browse / read_file / bash_each            →  tool_loop เสมอ (ไม่มีทางเลือกอื่น)

    ACTIONS — gotchas ที่ต้องระวัง:

      search_and_browse — items = keyword list → DDG search → URLs → summarize ทุก URL
        ✅ items=["Thai AI startup funding 2026", "LLM benchmark Thailand 2026"]  (specific, multi-word)
        ❌ items=["AI", "technology", "news"]                                      (generic → DDG คืน noise)

      browse_summarize — items = URL list → fetch+summarize แต่ละ URL
        ✅ items=["https://example.com/article", "https://site.com/page"]  (full https:// URLs)
        ❌ items=["example.com", "/article"]                               (ไม่ใช่ URL เต็ม → error)

      read_file — items = absolute file path list → อ่าน+summarize
        HARD RULE: ก่อน call tool_loop(read_file) ทุกครั้ง ต้องผ่าน SELF-CHECK นี้ก่อน:
          "paths ใน items มาจาก workspace_ls() หรือ bash ที่เพิ่ง call ในรอบนี้ไหม?"
           YES (มี tool output ในรอบนี้) → ดำเนินการได้
           NO  (จำ / เดา / copy จาก context) → ต้องเรียก workspace_ls() ก่อนเสมอ ห้ามข้ามขั้นตอน
        ✅ workspace_ls() → อ่าน output → tool_loop(items=[paths จาก output], action="read_file")
        ❌ tool_loop(items=["/Users/.../file.md"], action="read_file")  ← ถ้า path ไม่ได้มาจาก tool output รอบนี้
        ❌ items=["report.md"]  (relative → resolves to workspace/report.md → ไม่เจอไฟล์)
        !! NEVER call read_file (single tool) one-at-a-time — แม้แค่ 2 ไฟล์ ต้องใช้ tool_loop เสมอ

      bash_each — items = complete bash command list → รัน ผ่าน sandbox (cwd = workspace/, same limits as bash tool)
        ✅ items=["grep -n TODO /abs/a.py", "wc -l /abs/b.py"]  (full commands, absolute paths)
        ❌ items=["grep TODO a.py"]  (relative path → resolves inside workspace/ → มักไม่เจอไฟล์)
        !! paths ใน command ไม่รู้ → เรียก workspace_ls() หรือ bash ก่อนเพื่อหา absolute path จริง

    KEYWORD TIPS (search_and_browse):
      - ภาษา: ข่าวไทย → ใช้ภาษาไทย, tech/academic → English — ตรงกับแหล่งข้อมูล
      - ใส่ปี/timeframe: "Thai AI startup 2026" ดีกว่า "Thai AI startup" → DDG ตัด stale content
      - จำนวน keywords ≈ max_n / 4  (DDG คืน ~4-8 URL ต่อ keyword)
        เช่น max_n=20 → ใส่ 5 keywords; max_n=50 → ใส่ 10-12 keywords
        ❌ keyword เดียว + max_n=50 → browse แค่ ~8 URLs ทั้งที่ตั้ง max_n ไว้สูง

    CONTEXT TIPS:
      - ระบุมุมที่ต้องการจาก content: "ผลกระทบต่อตลาดงานไทย" ดีกว่า "AI"
      - ใช้คำเดียวกับที่ user ถาม → summarizer เน้นมุมนั้น ตัดส่วนไม่เกี่ยวออก

    PARAMETERS:
      context     — ใส่ user's primary goal เสมอ เช่น "สรุปข่าว AI funding ไทย 2026"
                    → ช่วยให้ summarizer สรุปตรงประเด็น  ❌ ปล่อยว่าง → summaries off-topic
      output_file — user พูดว่า "บันทึก / เก็บ / เขียนไฟล์" → ใส่ output_file="filename.md"
                    → เขียนใน workspace/ (sandbox อนุญาต)  ❌ ลืมใส่ → ผลหายเมื่อ turn จบ
      max_n       — cap จำนวน items ที่จะประมวล (default 50, web actions capped at 100)

    AFTER TOOL RETURNS — tool คืน preview (titles 5 แรก + จำนวน ok/error):
      → ต้องสรุปเนื้อหาจริงให้ user เสมอ  ❌ ห้าม copy-paste raw "✅ เสร็จ X/Y items" โชว์ตรงๆ
      → ถ้ามี error บางรายการ ให้แจ้ง user ด้วย ("X จาก Y รายการโหลดไม่ได้")

    BASH_EACH TIPS:
      - output ต่อ command ถูก truncate ที่ 2000 chars — ถ้า command คืนผลเยอะ ให้ pipe ก่อน
        เช่น "grep -rn TODO /abs/path | head -30"  แทน  "grep -rn TODO /abs/path"
      - command ต้องสมบูรณ์ในตัวเอง: ห้ามอ้าง env variable หรือ alias ที่ไม่มีใน sandbox

    คืน: สรุปผลทั้งหมด หรือ "[error] reason"
    """
    if action not in _VALID_ACTIONS:
        return f"[error] action '{action}' ไม่รู้จัก — ใช้ได้: {', '.join(sorted(_VALID_ACTIONS))}"
    if not items:
        return "[error] items ว่าง — ต้องระบุอย่างน้อย 1 item"

    try:
        # For web actions, raise the per-turn web limit to accommodate max_n (up to
        # _TOOL_LOOP_WEB_MAX=100). Regular single-tool callers keep the default 20 cap;
        # tool_loop explicitly needs more. If prior calls already consumed budget, the
        # remaining headroom shrinks accordingly.
        if action in ("browse_summarize", "search_and_browse"):
            from tools.web_cache import _WEB_COUNT, _WEB_LIMIT, web_count_set_limit
            max_n = min(max_n, _TOOL_LOOP_WEB_MAX)
            needed_limit = _WEB_COUNT[0] + max_n
            if needed_limit > _WEB_LIMIT[0]:
                web_count_set_limit(min(needed_limit, _TOOL_LOOP_WEB_MAX))
            remaining = max(0, _WEB_LIMIT[0] - _WEB_COUNT[0])
            if remaining < max_n:
                _phase(f"⚠ web budget: สูงสุด {remaining} items (ใช้ไปแล้ว {_WEB_COUNT[0]} จากทุก tool ใน turn นี้)")
                max_n = remaining
            if max_n == 0:
                return f"[web_limit] ค้นครบ {_WEB_LIMIT[0]} ครั้งแล้ว — หยุดค้นและสรุปจากข้อมูลที่มีได้เลย"

        # search_and_browse: expand keywords → URLs first, then browse
        if action == "search_and_browse":
            _phase(f"🔎 รวบรวม URLs จาก {len(items)} keywords (สูงสุด {max_n})…")
            urls = _collect_urls(items, max_n)
            if not urls:
                return f"[error] ไม่พบ URL สำหรับ keywords: {items[:3]}"
            _phase(f"📡 รวม {len(urls)} URLs — เริ่ม browse…")
            items = urls
            action = "browse_summarize"

        items = items[:max_n]
        total = len(items)
        results: list[dict] = []

        for i, item in enumerate(items, 1):
            if action == "browse_summarize":
                _phase(f"🌐 [{i}/{total}] {item[:70]}")
                r = _browse_summarize(item, context, i, total)
            elif action == "read_file":
                _phase(f"📄 [{i}/{total}] {os.path.basename(item)}")
                r = _read_file(item, context, i, total)
            else:  # bash_each
                _phase(f"⚙️  [{i}/{total}] {item[:70]}")
                r = _bash_each(item, i, total)

            results.append(r)

        # Write output file if requested
        out_path = ""
        if output_file:
            from config import WORKSPACE
            fname = output_file if "." in output_file else output_file + ".md"
            out_path = os.path.join(WORKSPACE, fname)
            _phase("✍️ เขียนไฟล์…")
            _write_output(out_path, results, context, action)

        ok = sum(1 for r in results if not r.get("error"))
        err = total - ok
        previews = "\n".join(
            f"- {r.get('title','')[:60]}: {r.get('summary','')[:100]}"
            for r in results
        )

        out_msg = f"\n📄 ไฟล์: {out_path}" if out_path else ""
        err_msg = f" ({err} error)" if err else ""
        output = f"✅ เสร็จ {ok}/{total} items{err_msg}{out_msg}\n\n{previews}"
        return output[:20000]

    except Exception as e:
        log.exception("tool_loop failed")
        return f"[error] tool_loop: {e}"
