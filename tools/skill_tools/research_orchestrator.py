# ENDEAVOR_LOCAL_AGENT_TH — © HaloChamp
# License: MIT License + Commons Clause — personal/educational use only, no commercial use without permission
# Website: https://www.poomwat.com | GitHub: https://github.com/halochamp | Email: champoomwat@gmail.com

"""research_orchestrator.py — Python-driven research state machine

แก้ปัญหา ReAct loop หลุดโฟกัส: Python loop จัดการ batch transitions เอง
โมเดลเรียก tool นี้แค่ครั้งเดียว — ไม่ต้อง "continue" เอง

Architecture (State Machine):
  COLLECT_URLS → [BATCH: FETCH×5 → APPEND×5 → MINI_SUMMARY] × n_batches → FINAL_SUMMARY → DONE
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import re
import sys
import time
import tempfile
import datetime
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from langchain_core.tools import tool
from tools.browse_url import _fetch_jina
from tools._summarize import summarize, summarize_batch
from tools._progress import phase as _phase, progress as _progress
from tools import web_cache
from tools.web_cache import (
    web_count_check as _wc_check,
    web_count_check_and_inc as _wc_check_and_inc,
    web_count_remaining as _wc_remaining,
)
from config import SUMMARY_SKIP_LLM_BELOW, SUMMARY_MAX_CHARS, SUMMARY_BATCH_MAX_TOKENS

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

_SKIP_DOMAINS = (
    "youtube.com", "youtu.be",   # video — no readable text
    "twitter.com", "x.com",      # JS-gated
    "instagram.com", "tiktok.com",
    "facebook.com",
)

def _fetch_raw(url: str) -> tuple[str, str]:
    """Fetch raw content for one URL (cache → jina). Thread-safe — no LLM."""
    if any(d in url for d in _SKIP_DOMAINS):
        return url, "[error] skipped — video/social domain"
    raw = web_cache.get(url)
    if raw is None:
        raw = _fetch_jina(url)
        if not raw.startswith("[error]"):
            web_cache.put(url, raw)
    return url, raw


def _build_src(url: str, summary: str) -> dict:
    """Build src dict from a ready summary string."""
    lines = [l.strip() for l in summary.split("\n") if l.strip()]
    title = lines[0][:80] if lines else url.split("/")[-1][:60]
    bullets = lines[1:4] if len(lines) > 1 else [summary[:200]]
    return {"title": title, "date": str(datetime.date.today().year), "bullets": bullets, "summary": summary}


def _fetch_and_summarize_batch(batch: list[str], topic: str) -> list[dict]:
    """Parallel fetch + batch summarize for up to 5 URLs.

    Phase 1 — parallel fetch raw (ThreadPoolExecutor 5 workers)
    Phase 2 — summarize_batch() for long content; raw directly for short
    Phase 3 — build src dicts in original order
    """
    # ── Phase 1: check summary cache, parallel-fetch uncached ─────────────────
    cached: dict[str, str] = {}
    need_raw: list[str] = []
    for url in batch:
        s = web_cache.get_summary(url, topic)
        if s is not None:
            cached[url] = s
        else:
            need_raw.append(url)

    raws: dict[str, str] = {}
    if need_raw:
        # Reserve budget before fetching, not after — this used to fetch every
        # uncached URL first and meter afterward, so it could blow past the
        # per-turn web cap or fetch anyway once another tool already exhausted
        # it. Mirrors batch_browse.py's check-then-inc pattern.
        err = _wc_check()
        if err:
            for url in need_raw:
                raws[url] = err
            need_raw = []
        else:
            remaining = _wc_remaining()
            if remaining < len(need_raw):
                _phase(f"⚠️ web budget: capping fetch to {remaining}/{len(need_raw)} URLs")
                for url in need_raw[remaining:]:
                    raws[url] = "[web_limit] ค้นครบ budget ของ turn นี้แล้ว"
                need_raw = need_raw[:remaining]

        if need_raw:
            _phase(f"⚡ Parallel fetch {len(need_raw)} URLs…")
            with ThreadPoolExecutor(max_workers=5) as ex:
                for url, raw in ex.map(_fetch_raw, need_raw):
                    raws[url] = raw
            # Meter only successful fetches, atomically — check_and_inc (not a
            # bare inc) since ToolNode can run this tool concurrently with
            # other web tools sharing the same counter.
            for url in need_raw:
                if not raws.get(url, "").startswith("[error]"):
                    err = _wc_check_and_inc()
                    if err:
                        raws[url] = err

    # ── Phase 2: batch summarize long content ─────────────────────────────────
    need_llm = [
        (url, raws[url]) for url in need_raw
        if not raws.get(url, "").startswith("[error]")
        and len(raws.get(url, "")) > SUMMARY_SKIP_LLM_BELOW
    ]
    batch_summaries: dict[str, str] = {}
    if need_llm:
        _phase(f"📝 Batch summarizing {len(need_llm)} sources…")
        result = summarize_batch(need_llm, user_query=topic)
        if result:
            batch_summaries = result
            for url, s in batch_summaries.items():
                web_cache.put_summary(url, topic, s)
        else:
            # summarize_batch failed → fall back to per-URL (sequential)
            _progress("batch summarize failed — falling back to per-URL")
            for url, raw in need_llm:
                s = summarize(raw, url=url, user_query=topic)
                batch_summaries[url] = s
                web_cache.put_summary(url, topic, s)

    # ── Phase 3: build src dicts in original batch order ──────────────────────
    srcs: list[dict] = []
    for url in batch:
        if url in cached:
            srcs.append(_build_src(url, cached[url]))
        elif raws.get(url, "").startswith("[error]"):
            srcs.append({"title": "[ดึงไม่ได้]", "date": "-", "bullets": [], "error": raws[url]})
        elif url in batch_summaries:
            srcs.append(_build_src(url, batch_summaries[url]))
        else:
            # Short content — use raw directly (no LLM needed)
            raw = raws.get(url, "")
            summary = raw[:SUMMARY_MAX_CHARS] if raw else "(ไม่พบเนื้อหา)"
            web_cache.put_summary(url, topic, summary)
            srcs.append(_build_src(url, summary))
    return srcs


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


def _append_final_summary(path: str, topic: str, all_mini: list[dict], all_summaries: list[str]) -> None:
    synthesis = ""
    try:
        from llm import build_llm
        llm = build_llm(
            temperature=0.1,
            max_tokens=SUMMARY_BATCH_MAX_TOKENS,
            extra_body={"enable_thinking": False},
        )
        llm_input = "\n\n".join(all_summaries)[:100_000]
        prompt = (
            f"คุณเป็น research assistant สังเคราะห์งานวิจัยเกี่ยวกับ: {topic}\n\n"
            f"ด้านล่างคือสรุปจาก {len(all_summaries)} แหล่ง "
            "ขอให้สังเคราะห์เป็นรายงานสรุปในภาษาไทย\n"
            "- จัดกลุ่มประเด็นสำคัญ รวมตัวเลข วันที่ และชื่อเฉพาะ\n"
            "- ห้ามแต่งข้อมูลที่ไม่มีในแหล่ง\n"
            "- ตอบเฉพาะสรุป ไม่ต้องมีคำนำหรือปิดท้าย\n\n"
            + llm_input
        )
        resp = llm.invoke(prompt, config={"callbacks": []})
        synthesis = (resp.content or "").strip()
    except Exception as e:
        log.warning(f"final summary LLM failed: {e} — using batch list fallback")

    with open(path, "a", encoding="utf-8") as f:
        f.write("===================================================\n")
        f.write(f"## 🔍 FINAL SUMMARY — {topic}\n\n")
        if synthesis:
            f.write(synthesis + "\n")
        else:
            lines = [f"[Batch {s['batch']} — Sources {s['sources']}] {s['summary']}" for s in all_mini]
            f.write("\n".join(lines) + "\n")
        f.write(f"\n### หมายเหตุ\nสังเคราะห์จาก {len(all_summaries)} แหล่ง — ดูรายละเอียดด้านบน\n")
        f.write("===================================================\n")


# ── Checkpoint helpers ─────────────────────────────────────────────────────────

def _cp_path(topic: str) -> str:
    """Per-topic checkpoint path. A fresh run used to overwrite a single fixed
    /tmp file, silently clobbering a different topic's paused checkpoint — keying
    on the topic slug keeps interleaved research runs from colliding.

    The readable slug is truncated to 25 chars, so two distinct long topics that
    share the same first 25 chars would otherwise collide on the same checkpoint
    file — append a short hash of the full topic to disambiguate.
    """
    slug = re.sub(r"[^\w]", "_", topic)[:25].strip("_") or "untitled"
    h = hashlib.md5(topic.encode("utf-8")).hexdigest()[:8]
    return os.path.join(tempfile.gettempdir(), f"news_research_{slug}_{h}.json")


def _latest_cp_path() -> Optional[str]:
    """Most recently modified checkpoint file, for resume=True with topic="" —
    there's nothing to hash into _cp_path() so fall back to whichever research
    run was paused most recently."""
    candidates = glob.glob(os.path.join(tempfile.gettempdir(), "news_research_*.json"))
    return max(candidates, key=os.path.getmtime) if candidates else None


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

    AFTER THIS TOOL RETURNS: synthesize the "=== RESEARCH SUMMARIES ===" section into a
    structured Thai report — key findings grouped by theme, with concrete numbers and dates.
    State how many URLs the synthesis is derived from. Do NOT dump the raw summaries —
    write a coherent structured report for the user.
    """
    n = max(1, min(100, int(n)))
    ws = _workspace()

    try:
        # ── RESUME or fresh START ──────────────────────────────────────────────
        if resume:
            # Documented resume call shape is topic="", n=0, resume=True — with
            # no topic there's nothing to hash into _cp_path(), so fall back to
            # the most recently touched checkpoint.
            cp_path = _cp_path(topic) if topic else _latest_cp_path()
            cp = _load_cp(cp_path) if cp_path else None
            if cp is None:
                return "[error] ไม่พบ checkpoint — ลอง resume=False เพื่อเริ่มใหม่"
            if topic and cp.get("topic") != topic:
                # cp_path is hash-keyed on topic now, so this should be unreachable
                # outside a hash collision — guard kept as defense-in-depth so a stale
                # checkpoint never silently swaps in someone else's topic.
                return "[error] checkpoint topic ไม่ตรงกับ topic ที่ระบุ — ลอง resume=False เพื่อเริ่มใหม่"
            topic = cp["topic"]
            filename = cp["filename"]
            path = os.path.join(ws, filename)
            _phase(f"▶️ Resume: {topic} ({len(cp['completed'])}/{len(cp['all_urls'])} แหล่งแล้ว)")
        else:
            cp_path = _cp_path(topic)
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
                "all_summaries": [],  # full per-URL summaries — persisted so resume keeps pre-pause sources
            }
            _save_cp(cp, cp_path)
            _phase(f"📡 {topic} | {len(all_urls)} แหล่ง | {total_batches} batch — เริ่ม…")

        # ── BATCH LOOP — Python loop, ไม่ต้องพึ่ง LLM ──────────────────────────
        all_summaries: list[str] = cp.setdefault("all_summaries", [])  # ref to cp's list → resume keeps pre-pause sources
        while len(cp["completed"]) < len(cp["all_urls"]):
            remaining = [u for u in cp["all_urls"] if u not in cp["completed"]]
            batch = remaining[:5]
            batch_start = len(cp["completed"]) + 1
            batch_end = batch_start + len(batch) - 1
            batch_label = cp["batch_num"] + 1
            _phase(f"📦 Batch {batch_label}/{cp['total_batches']} (แหล่ง {batch_start}–{batch_end})")

            base_n = len(cp["completed"])
            srcs = _fetch_and_summarize_batch(batch, topic)
            titles = []
            for i, (url, src) in enumerate(zip(batch, srcs)):
                global_n = base_n + i + 1
                _append_source(path, global_n, src)
                title = src.get("title", url[:60])
                titles.append(title)
                all_summaries.append(f"[{global_n}] {title}\n{src.get('summary', title)}")
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
        _append_final_summary(path, topic, cp["batch_summaries"], all_summaries)

        # Cleanup checkpoint
        try:
            os.remove(cp_path)
        except Exception:
            pass

        done = len(cp["completed"])
        batches = cp["batch_num"]

        summary_block = "\n\n".join(all_summaries)

        if len(summary_block) > 20_000:
            _phase(f"✍️ Compressing {len(summary_block):,} chars → ≤20,000…")
            try:
                from llm import build_llm
                llm = build_llm(
                    temperature=0.1,
                    max_tokens=SUMMARY_BATCH_MAX_TOKENS,
                    extra_body={"enable_thinking": False},
                )
                llm_input = summary_block[:100_000]
                compress_prompt = (
                    f"คุณเป็น research assistant สังเคราะห์งานวิจัยเกี่ยวกับ: {topic}\n\n"
                    f"ด้านล่างคือสรุปจาก {len(all_summaries)} แหล่ง "
                    "ขอให้สังเคราะห์เป็นสรุปรวมในภาษาไทย ความยาวไม่เกิน 20,000 ตัวอักษร\n"
                    "- รวมข้อมูลสำคัญ ตัวเลข วันที่ และชื่อเฉพาะไว้ครบ\n"
                    "- จัดกลุ่มประเด็นที่เกี่ยวข้องเข้าด้วยกัน\n"
                    "- ห้ามแต่งข้อมูลที่ไม่มีในแหล่ง\n"
                    "- ตอบเฉพาะสรุป ไม่ต้องมีคำนำหรือปิดท้าย\n\n"
                    + llm_input
                )
                resp = llm.invoke(compress_prompt, config={"callbacks": []})
                compressed = (resp.content or "").strip()
                if compressed:
                    summary_block = compressed
                    _progress(f"compressed to {len(summary_block):,} chars")
                else:
                    summary_block = summary_block[:20_000]
            except Exception as e:
                log.warning(f"compression LLM failed: {e} — truncating")
                summary_block = summary_block[:20_000]

        return (
            f"✅ เสร็จสิ้น {batches} batch ({done} แหล่ง)\n"
            f"📄 ไฟล์: {path}\n\n"
            "=== RESEARCH SUMMARIES ===\n"
            + summary_block
        )

    except Exception as e:
        log.exception("research_orchestrator failed")
        return f"[error] research_orchestrator: {e}"
