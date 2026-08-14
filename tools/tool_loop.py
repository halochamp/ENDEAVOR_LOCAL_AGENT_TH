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
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.tools import tool

from tools._progress import phase as _phase, progress as _progress
from tools._summarize import summarize
from tools._result_title import title_from_browse_summary, title_from_command, title_from_path
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


def _collect_urls(keywords: list[str], max_n: int) -> list[str]:
    from tools.web_search import _search_urls

    seen: set[str] = set()
    urls: list[str] = []
    for kw in keywords:
        if len(urls) >= max_n:
            break
        _phase(f"🔍 ค้นหา: {kw[:60]}")
        remaining = max_n - len(urls)
        for u in _search_urls(kw, user_query=kw, max_results=min(TOOL_LOOP_DDG_MAX_RESULTS, remaining)):
            if u and u not in seen and u.startswith("http"):
                seen.add(u)
                urls.append(u)
        _progress(f"URLs: {len(urls)}/{max_n}")
    return urls[:max_n]


# ── Action handlers ─────────────────────────────────────────────────────────────

def _browse_summarize(url: str, context: str, idx: int, total: int) -> dict:
    # Delegate entirely to browse_url — it handles _wc_check/_wc_inc and caching.
    from tools.browse_url import browse_url as _bu
    from tools.browse_url import _split_browse_result as _split
    try:
        result = _bu.invoke({"url": url, "user_query": context})
        result_url, summary, ok = _split(result)
        if not ok:
            title = "[web limit reached]" if result.startswith("[web_limit]") else "[ดึงไม่ได้]"
            return {"url": url, "title": title, "summary": result, "error": True}
        title = title_from_browse_summary(result_url or url, summary)
        _progress(f"[{idx}/{total}] {title[:50]}")
        return {"url": result_url or url, "title": title, "summary": summary, "error": False}
    except Exception as e:
        return {"url": url, "title": "[error]", "summary": str(e), "error": True}


def _read_file(path: str, context: str, idx: int, total: int) -> dict:
    # Delegate to read_file's plain callable (not the @tool wrapper) — supports
    # PDF/DOCX/XLSX, safety checks, size guard, and keyword-anchored
    # _sample_coverage() for large docs.
    from tools.read_file import _read_file_impl as _rf
    from tools.read_file import _is_compact_read_result as _is_compact
    title = title_from_path(path)
    try:
        content = _rf(path, user_query=context)
        if content.startswith("[error]"):
            return {"path": path, "title": title, "summary": content, "error": True}
        _progress(f"[{idx}/{total}] {title} ({len(content):,} chars)")
        already_compact = _is_compact(content)
        body = (
            summarize(content[:TOOL_LOOP_READ_MAX_CHARS], url=path, user_query=context or "summarize")
            if len(content) > TOOL_LOOP_READ_SUMMARIZE_THRESHOLD and not already_compact
            else content
        )
        return {"path": path, "title": title, "summary": body, "error": False}
    except Exception as e:
        return {"path": path, "title": title, "summary": str(e), "error": True}


def _bash_each(command: str, idx: int, total: int) -> dict:
    # Delegate to bash's plain callable (not the @tool wrapper) — handles sandbox
    # profile, finally cleanup, 10k truncation.
    from tools.bash import _bash_impl as _bash
    from tools._truncate import truncate_with_save
    from config import WORKSPACE
    _progress(f"[{idx}/{total}] {command[:60]}")
    try:
        output = _bash(command, TOOL_LOOP_BASH_TIMEOUT)
        error = output.startswith("[error]")
        # Heuristic, not a guaranteed detection: a command whose legitimate stdout happens to
        # start with this exact literal (e.g. `cat`-ing an earlier "_bash_output_*.txt" recovery
        # file inside a batch) would also take the truncated-already branch below. Degrades
        # gracefully either way (falls back to the pre-fix blind slice), so left as-is.
        if output.startswith("...[bash] truncated:"):
            # bash.py already truncated+saved a recovery file (raw output > its own 10k
            # cap) — its marker_first design means the marker (and recovery path) survives
            # a further blind cut, so just slice here; re-running truncate_with_save on an
            # already-truncated string would write a second, redundant recovery file that
            # duplicates already-lossy data instead of the true original.
            if len(output) > TOOL_LOOP_BASH_MAX_CHARS:
                output = output[:TOOL_LOOP_BASH_MAX_CHARS] + "\n...[truncated]"
        else:
            # bash.py did NOT truncate (raw output <= its own 10k cap) — outputs in the
            # 2k-10k range reach here unmarked with no recovery file yet, so this secondary
            # cut needs its own save.
            output = truncate_with_save(
                output, TOOL_LOOP_BASH_MAX_CHARS, WORKSPACE, "bash_each", marker_first=True
            )
        return {"cmd": command, "title": title_from_command(command), "summary": output or "(no output)", "error": error}
    except Exception as e:
        return {"cmd": command, "title": title_from_command(command), "summary": str(e), "error": True}


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

# @tool tool_loop's docstring keeps only the routing-critical half; per-action
# examples/PARAMETERS/post-call guidance only matter once the model has already
# chosen this tool — deferred here, injected on the first call each turn.
_SYNTAX_MANUAL = (
    "\n\n[tool_loop usage notes]\n"
    "search_and_browse — items = keywords → DDG search → browse + summarize each URL.\n"
    "  ✅ items=[\"Thai AI startup funding 2026\", \"LLM benchmark Thailand 2026\"]  (specific, multi-word)\n"
    "  ❌ items=[\"AI\", \"technology\", \"news\"]  (generic → DDG returns noise)\n"
    "  Keyword tips: match language to source (Thai news → Thai keywords, tech/academic → English); "
    "include year/timeframe; use ~max_n/4 keywords (DDG returns ~4-8 URLs each — e.g. max_n=20 → 5 keywords).\n"
    "browse_summarize — items = full https:// URLs.\n"
    "  ❌ items=[\"example.com\", \"/article\"]  (not full URLs → error)\n"
    "read_file — items = ABSOLUTE file paths → read + summarize.\n"
    "  ✅ bash(\"rg --files | rg '\\.md$'\") → read output → tool_loop(items=[paths from output], action=\"read_file\")\n"
    "  ❌ tool_loop(items=[\"/Users/.../file.md\"], action=\"read_file\")  path not from tool output this round\n"
    "  ❌ items=[\"report.md\"]  (relative → resolves to workspace/report.md → file not found)\n"
    "bash_each — items = complete self-contained commands, absolute paths (sandboxed, cwd = workspace/).\n"
    "  ✅ items=[\"grep -n TODO /abs/a.py\", \"wc -l /abs/b.py\"]\n"
    "  ❌ items=[\"grep TODO a.py\"]  (relative path → resolves inside workspace/ → usually not found)\n"
    "  Unknown paths → call bash first to get real absolute paths. Output per command truncated at 2000 "
    "chars — pipe big output (\"grep -rn TODO /abs | head -30\"). No env variables or aliases — sandbox has none.\n"
    "PARAMETERS:\n"
    "  context     — always set to the user's primary goal, in the user's own words "
    "(e.g. \"สรุปข่าว AI funding ไทย 2026\") → keeps summaries on-topic; empty → off-topic.\n"
    "  output_file — user says \"บันทึก / เก็บ / เขียนไฟล์\" → set output_file=\"filename.md\" (written in "
    "workspace/); omitted → results are lost when the turn ends.\n"
    "  max_n       — cap on items processed (default 50; web actions capped at 100).\n"
    "AFTER TOOL RETURNS — tool returns a preview (first 5 titles + ok/error counts): always synthesize the "
    "real content for the user — never paste raw \"✅ done X/Y items\"; if some items errored, tell the user "
    "(\"X of Y failed to load\")."
)


def _tool_loop_impl(
    items: list[str],
    action: str,
    context: str = "",
    max_n: int = 50,
    output_file: str = "",
) -> str:
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
            from tools.web_cache import web_count_ensure_headroom
            max_n = min(max_n, _TOOL_LOOP_WEB_MAX)
            # Atomically raise the limit if needed and read remaining in one lock acquisition
            # — no TOCTOU gap between reading _WEB_COUNT and setting the new limit.
            remaining = web_count_ensure_headroom(max_n, _TOOL_LOOP_WEB_MAX)
            if remaining < max_n:
                _phase(f"⚠ web budget: สูงสุด {remaining} items เหลือใน turn นี้")
                max_n = remaining
            if max_n == 0:
                return f"[web_limit] ค้นครบแล้ว — หยุดค้นและสรุปจากข้อมูลที่มีได้เลย"

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
            fname = os.path.basename(output_file) or "output.md"
            if "." not in fname:
                fname += ".md"
            out_path = os.path.join(WORKSPACE, fname)
            _phase("✍️ เขียนไฟล์…")
            _write_output(out_path, results, context, action)

        ok = sum(1 for r in results if not r.get("error"))
        err = total - ok
        err_msg = f" ({err} error)" if err else ""

        if out_path:
            # File written — compact preview is enough; full content is in the file.
            body = "\n".join(
                f"- {r.get('title','')[:60]}: {r.get('summary','')[:100]}"
                for r in results
            )
            out_msg = (
                f"\n📄 ไฟล์: {os.path.basename(out_path)}"
                f"\n📁 directory: {os.path.dirname(out_path)}"
                f"\n🔗 full path: {out_path}"
            )
        else:
            # No file — collect up to 100k (inner), then outer-summarize to 20k for state.
            _BODY_BUDGET = 100_000
            _OUTER_LIMIT = 20_000
            parts: list[str] = []
            budget = _BODY_BUDGET
            for r in results:
                title = r.get("title", "")[:80]
                summary = r.get("summary", "")
                ref = r.get("url", r.get("path", r.get("cmd", "")))
                ref_str = f" ({ref[:80]})" if ref and ref != title else ""
                entry = f"[{title}]{ref_str}\n{summary}"
                if len(entry) > budget:
                    parts.append(entry[:budget] + "…")
                    budget = 0
                    break
                parts.append(entry)
                budget -= len(entry) + 2  # +2 for \n\n separator
            recall_hint = " — เรียก recall_web(url) เพื่อดูเนื้อหาเพิ่ม" if action == "browse_summarize" else ""
            truncated_note = (
                f"\n\n… (แสดง {len(parts)}/{total} items — ถึง budget 100k chars{recall_hint})"
                if budget == 0 and len(parts) < total else ""
            )
            body = "\n\n".join(parts) + truncated_note
            if len(body) > _OUTER_LIMIT:
                # Recovery-file paths ("Full output saved at: X.txt") are load-bearing pointers,
                # not prose — summarize() has no concept of them and can silently mangle/cut the
                # path mid-string. Pull them out first, reattach verbatim BEFORE the summary
                # (mirrors _truncate.py's marker_first reasoning) so they survive both the LLM
                # pass and the _OUTER_LIMIT slice below.
                recovery_paths = list(dict.fromkeys(re.findall(r"Full output saved at: (\S+\.txt)\.", body)))
                _phase("📝 สรุปรวม…")
                body = summarize(body[:100_000], url="loop_results", user_query=context or "summarize all results")
                if recovery_paths:
                    appendix = "\n".join(f"- {p}" for p in recovery_paths)
                    body = f"[recovery files — full untruncated output of any item above]\n{appendix}\n\n{body}"
            body = body[:_OUTER_LIMIT]
            out_msg = ""

        output = f"✅ เสร็จ {ok}/{total} items{err_msg}{out_msg}\n\n{body}"
        return output[:20000]

    except Exception as e:
        log.exception("tool_loop failed")
        return f"[error] tool_loop: {e}"


@tool
def tool_loop(
    items: list[str],
    action: str,
    context: str = "",
    max_n: int = 50,
    output_file: str = "",
) -> str:
    """Loop over many items in ONE call — Python runs every iteration; never drops out of the loop.

    Use when repeating the same action on many inputs (summarize 30 sites, read 10 files, run 20 commands).
    Tool choice: browse_summarize + N ≤ 8 + no output_file → use batch_browse instead; everything else → tool_loop.
    !! NEVER call read_file one-at-a-time for multiple files — even 2 files must go through tool_loop.

    ACTIONS: search_and_browse (items=keywords), browse_summarize (items=full https:// URLs), read_file
    (items=ABSOLUTE file paths — must come from bash output THIS round, never remembered/guessed/copied
    from context), bash_each (items=complete self-contained commands with absolute paths).

    Detailed per-action examples, PARAMETERS (context/output_file/max_n), and post-call synthesis
    guidance appear in this tool's own result the first time you call it this turn.

    Returns: summary of all results, or "[error] reason".
    """
    from tools._call_guard import first_call_this_turn
    first = first_call_this_turn("tool_loop")
    result = _tool_loop_impl(items, action, context, max_n, output_file)
    if first:
        result += _SYNTAX_MANUAL
    return result
