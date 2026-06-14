"""research_orchestrator.py — Python-driven research state machine

แก้ปัญหา ReAct loop หลุดโฟกัส: Python loop จัดการ batch transitions เอง
โมเดลเรียก tool นี้แค่ครั้งเดียว — ไม่ต้อง "continue" เอง

Architecture (State Machine):
  COLLECT_URLS → [BATCH: FETCH×5 → APPEND×5 → MINI_SUMMARY] × n_batches → FINAL_SUMMARY → DONE
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import tempfile
import datetime
import logging
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from langchain_core.tools import tool
from tools.browse_url import _fetch_jina
from tools._summarize import summarize
from tools._progress import phase as _phase, progress as _progress
from tools import web_cache
from tools.web_cache import web_count_inc as _wc_inc

log = logging.getLogger(__name__)

_FALLBACK_KEYWORDS = [
    "latest 2026", "overview analysis", "trends statistics",
    "expert opinion", "future outlook",
]


# ── URL collection ─────────────────────────────────────────────────────────────

def _ddg_search(query: str, max_results: int = 8) -> list[str]:
    """Single DDG search → list of URLs"""
    try:
        from ddgs import DDGS
        time.sleep(1.0)
        raw = list(DDGS().text(query, max_results=max_results, region="wt-wt"))
        return [r.get("href", "") for r in raw if r.get("href")]
    except Exception as e:
        log.warning(f"DDG search failed: {e}")
        time.sleep(3.0)
        try:
            from ddgs import DDGS
            raw = list(DDGS().text(query, max_results=max_results, region="wt-wt"))
            return [r.get("href", "") for r in raw if r.get("href")]
        except Exception:
            return []


def _collect_urls(topic: str, n: int, keywords: list[str]) -> list[str]:
    """Collect N unique URLs via multiple DDG search rounds"""
    seen: set[str] = set()
    urls: list[str] = []
    for kw in keywords:
        if len(urls) >= n:
            break
        query = f"{topic} {kw}"
        _phase(f"🔍 ค้นหา: {query[:50]}")
        batch = _ddg_search(query)
        for u in batch:
            if u and u not in seen and u.startswith("http"):
                seen.add(u)
                urls.append(u)
        _progress(f"URLs collected: {len(urls)}/{n}")
    return urls[:n]


# ── Source fetch + extract ─────────────────────────────────────────────────────

def _fetch_source(url: str, topic: str) -> dict:
    """Fetch URL and extract title + bullets"""
    try:
        # Check cache first — only count real network fetches
        uq = topic
        summary = web_cache.get_summary(url, uq)
        if summary is None:
            raw = web_cache.get(url)
            if raw is None:
                _wc_inc()
                raw = _fetch_jina(url)
                if raw.startswith("[error]"):
                    return {"title": "[ดึงไม่ได้]", "date": "-", "bullets": [], "error": raw}
                web_cache.put(url, raw)
            summary = summarize(raw, url=url, user_query=uq)
            web_cache.put_summary(url, uq, summary)

        # Parse title from summary (first non-empty line or fallback)
        lines = [l.strip() for l in summary.split("\n") if l.strip()]
        title = lines[0][:80] if lines else url.split("/")[-1][:60]
        bullets = lines[1:4] if len(lines) > 1 else [summary[:200]]
        return {"title": title, "date": str(datetime.date.today().year), "bullets": bullets, "summary": summary}
    except Exception as e:
        return {"title": "[ดึงไม่ได้]", "date": "-", "bullets": [], "error": str(e)}


# ── File helpers ───────────────────────────────────────────────────────────────

def _workspace() -> str:
    from config import WORKSPACE
    return WORKSPACE


def _append_source(path: str, n: int, src: dict) -> None:
    title = src.get("title", "[ไม่พบหัวข้อ]")
    date = src.get("date", "-")
    bullets = src.get("bullets", [])
    error = src.get("error")
    with open(path, "a", encoding="utf-8") as f:
        if error:
            f.write(f"### Source {n}: [ดึงไม่ได้ — ข้าม]\n\n")
        else:
            bullet_text = "\n".join(f"- {b}" for b in bullets[:3]) if bullets else "- (ไม่พบเนื้อหา)"
            f.write(f"### Source {n}: {title}\nวันที่: {date}\n{bullet_text}\n\n")


def _append_mini_summary(path: str, batch_num: int, start: int, end: int, summaries: list[str]) -> str:
    text = "; ".join(s[:80] for s in summaries[:5] if s)
    mini = f"Batch {batch_num}: แหล่ง {start}–{end} รวม {end-start+1} แหล่ง — {text}"
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"---\n## 📋 Batch {batch_num} Summary (Sources {start}–{end})\n{text}\n---\n\n")
    return mini


def _append_final_summary(path: str, topic: str, all_mini: list[dict]) -> None:
    lines = [f"[Batch {s['batch']} — Sources {s['sources']}] {s['summary']}" for s in all_mini]
    combined = "\n".join(lines)
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"""===================================================
## 🔍 FINAL SUMMARY — {topic}
{combined}

### หมายเหตุ
สังเคราะห์จาก {len(all_mini)} batch mini-summaries — ดูรายละเอียดแต่ละแหล่งด้านบน
===================================================
""")


# ── Checkpoint helpers ─────────────────────────────────────────────────────────

def _cp_path(topic: str) -> str:
    """Per-topic checkpoint path. A fresh run used to overwrite a single fixed
    /tmp file, silently clobbering a different topic's paused checkpoint — keying
    on the topic slug keeps interleaved research runs from colliding."""
    slug = re.sub(r"[^\w]", "_", topic)[:25].strip("_") or "untitled"
    return os.path.join(tempfile.gettempdir(), f"news_research_{slug}.json")


def _save_cp(cp: dict, path: str) -> None:
    """Atomic write — tmp + os.replace so a kill mid-write can't corrupt the
    checkpoint (mirrors store.py / file_registry.py); handle closed via context manager."""
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cp, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _load_cp(path: str) -> Optional[dict]:
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


# ── Main tool ─────────────────────────────────────────────────────────────────

@tool
def research_orchestrator(topic: str, n: int = 30, resume: bool = False, keywords: str = "") -> str:
    """ค้นหาข่าวล่าสุดจาก N แหล่งและเขียนรายงาน — วน batch อัตโนมัติโดย Python loop

    ทำงานทั้งหมดใน 1 tool call: ค้นหา URL → เปิดเว็บทุกแหล่ง → สรุปรายแหล่ง → สรุปรวม
    ไม่ต้อง "continue" หรือรอ — Python จัดการ batch transitions เอง

    ⚠️ Skill-gated: ต้องเปิด skill mode ก่อนเรียก tool นี้
    topic: หัวข้อที่ต้องการค้นหา
    n: จำนวนแหล่ง (1–100, default 30)
    resume: ทำต่อจาก checkpoint ที่ค้าง (true = ต่อ, false = เริ่มใหม่)
    keywords: search angles คั่นด้วย comma เช่น "latest 2026, expert opinion, market trends, challenges"
              ถ้าไม่ระบุจะใช้ fallback keywords ทั่วไป
    คืน: สรุปผล + path ไฟล์ หรือ "[error] reason"
    """
    n = max(1, min(100, int(n)))
    ws = _workspace()
    cp_path = _cp_path(topic)   # per-topic — resume keys on the same topic slug

    try:
        # ── RESUME or fresh START ──────────────────────────────────────────────
        if resume:
            cp = _load_cp(cp_path)
            if cp is None:
                return "[error] ไม่พบ checkpoint — ลอง resume=False เพื่อเริ่มใหม่"
            topic = cp["topic"]
            filename = cp["filename"]
            path = os.path.join(ws, filename)
            _phase(f"▶️ Resume: {topic} ({len(cp['completed'])}/{len(cp['all_urls'])} แหล่งแล้ว)")
        else:
            # Create output file
            slug = re.sub(r"[^\w]", "_", topic)[:25].strip("_")
            filename = f"news_{slug}.md"
            path = os.path.join(ws, filename)
            today = datetime.date.today().strftime("%Y-%m-%d")
            with open(path, "w", encoding="utf-8") as f:
                f.write(f"# News Digest: {topic}\nDate: {today} | Sources: 0/{n} | Status: in progress\n---\n\n")
            _phase(f"📁 สร้างไฟล์ {filename}")

            # Parse keywords
            kw_list = [k.strip() for k in keywords.split(",") if k.strip()] if keywords else _FALLBACK_KEYWORDS

            # Collect URLs
            _phase(f"🔎 รวบรวม {n} URLs สำหรับ '{topic}'…")
            all_urls = _collect_urls(topic, n, kw_list)
            if not all_urls:
                return f"[error] ไม่พบ URL สำหรับ '{topic}'"

            total_batches = (len(all_urls) + 4) // 5
            cp = {
                "topic": topic, "filename": filename,
                "all_urls": all_urls, "completed": [],
                "batch_num": 0, "total_batches": total_batches,
                "batch_summaries": [],
            }
            _save_cp(cp, cp_path)
            _phase(f"📡 {topic} | {len(all_urls)} แหล่ง | {total_batches} batch — เริ่ม…")

        # ── BATCH LOOP — Python loop, ไม่ต้องพึ่ง LLM ──────────────────────────
        while len(cp["completed"]) < len(cp["all_urls"]):
            remaining = [u for u in cp["all_urls"] if u not in cp["completed"]]
            batch = remaining[:5]
            batch_start = len(cp["completed"]) + 1
            batch_end = batch_start + len(batch) - 1
            batch_label = cp["batch_num"] + 1
            _phase(f"📦 Batch {batch_label}/{cp['total_batches']} (แหล่ง {batch_start}–{batch_end})")

            titles = []
            base_n = len(cp["completed"])
            for i, url in enumerate(batch):
                global_n = base_n + i + 1
                src = _fetch_source(url, topic)
                _append_source(path, global_n, src)
                title = src.get("title", url[:60])
                titles.append(title)
                # Mark this source done + save before the next progress()/cancel
                # checkpoint — if cancelled mid-batch, resume won't re-fetch and
                # re-append sources already written to the file.
                cp["completed"].append(url)
                _save_cp(cp, cp_path)
                _progress(f"📰 {global_n}/{len(cp['all_urls'])}: {title[:50]}")

            mini = _append_mini_summary(path, batch_label, batch_start, batch_start + len(titles) - 1, titles)
            cp["batch_summaries"].append({
                "batch": batch_label,
                "sources": f"{batch_start}-{batch_start + len(titles) - 1}",
                "summary": mini,
            })
            cp["batch_num"] += 1
            _save_cp(cp, cp_path)

        # ── FINAL SUMMARY ──────────────────────────────────────────────────────
        _phase("✍️ เขียน Final Summary…")
        _append_final_summary(path, topic, cp["batch_summaries"])

        # Cleanup checkpoint
        try:
            os.remove(cp_path)
        except Exception:
            pass

        done = len(cp["completed"])
        batches = cp["batch_num"]
        return (
            f"✅ เสร็จสิ้น {batches} batch ({done} แหล่ง)\n"
            f"📄 ไฟล์: {path}\n\n"
            + "\n".join(f"- {s['summary'][:120]}" for s in cp["batch_summaries"])
        )

    except Exception as e:
        log.exception("research_orchestrator failed")
        return f"[error] research_orchestrator: {e}"
