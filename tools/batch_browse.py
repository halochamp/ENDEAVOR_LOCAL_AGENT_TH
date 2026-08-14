# ENDEAVOR_LOCAL_AGENT_TH — © HaloChamp
# License: MIT License + Commons Clause — personal/educational use only, no commercial use without permission
# Website: https://www.poomwat.com | GitHub: https://github.com/halochamp | Email: champoomwat@gmail.com

"""batch_browse.py — fetch หลาย URLs แล้วคืน summaries รวมใน 1 tool call

Architecture:
  Phase 1 (parallel):   HTTP fetch เท่านั้น — ไม่มี LLM ใน thread
  Phase 2 (batched):    URL ที่ต้องใช้ LLM ถูก summarize รวมใน request เดียว (F1)
                        — fallback ทีละ URL เมื่อ batch ใช้ไม่ได้/parse ไม่ผ่าน

Cache-aware: summary hit ข้ามทั้ง 2 phase; raw hit ข้าม Phase 1 (แค่
re-summarize สำหรับ query ใหม่ — ห้าม raw-slice ตาม Tool Output Contract)
Web counter นับเฉพาะ fetch ใหม่ที่สำเร็จ — cache hit ทุกแบบไม่กิน budget
"""
from __future__ import annotations
import sys
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.tools import tool
from tools.browse_url import _fetch_body
from tools._summarize import summarize, summarize_batch
from config import (
    SUMMARY_SKIP_LLM_BELOW,
    BATCH_BROWSE_MAX_WORKERS as _MAX_WORKERS,
    BATCH_BROWSE_MAX_URLS as _MAX_URLS,
)
from tools._progress import phase as _phase, progress as _progress
from tools import web_cache
from tools.web_cache import web_count_check as _wc_check, web_count_check_and_inc as _wc_check_and_inc, web_count_remaining as _wc_remaining


def _http_only(url: str) -> tuple[str, str]:
    """HTTP fetch เท่านั้น ไม่มี LLM — safe ใน thread"""
    return url, _fetch_body(url)


def _extract_url(entry: str) -> str:
    """ดึง URL จาก entry ที่ model อาจ copy ทั้งบรรทัดจาก fetch_sitemap ("date  url"):
    เอา token แรกที่ขึ้นต้น http; ไม่มี → token แรก (bare domain ให้ขั้น https:// prefix จัดการต่อ)"""
    tokens = entry.split()
    if not tokens:
        return ""
    return next((t for t in tokens if t.startswith("http")), tokens[0])


@tool
def batch_browse(urls: list, user_query: str = "") -> str:
    """Fetch multiple URLs and return combined summaries in ONE tool call.
    Use instead of one-by-one browse_url when you have a URL list from fetch_sitemap or search results.

    urls: list of URLs to fetch (max BATCH_BROWSE_MAX_URLS, default 8)
    user_query: the current question — keeps summaries on-topic
    Returns: combined summaries of all URLs, or "[error] reason"
    """
    if not urls:
        return "[error] urls is required"

    clean = [_extract_url(str(u).strip()) for u in urls if u and str(u).strip()]
    clean = [u for u in clean if u][:_MAX_URLS]
    if not clean:
        return "[error] no valid URLs provided"

    # normalize + dedup — URL ซ้ำห้าม fetch/นับ budget/summarize ซ้ำ
    clean = [u if u.startswith("http") else "https://" + u for u in clean]
    clean = list(dict.fromkeys(clean))

    uq = (user_query or "").strip()

    # ── แยก summary hit / raw hit (re-summarize) / miss (fetch) ────────
    results: dict[str, str] = {}
    pending: list[dict] = []  # [{"url", "raw", "is_new"}]
    to_fetch: list[str] = []
    for u in clean:
        hit_sum = web_cache.get_summary(u, uq) or web_cache.get_summary(u, "")
        if hit_sum is not None:
            results[u] = f"[web:{u}] {hit_sum}"
            continue
        hit_raw = web_cache.get(u)
        if hit_raw is not None:
            if hit_raw.strip():
                pending.append({"url": u, "raw": hit_raw, "is_new": False})
            else:
                results[u] = f"[web:{u}] [no usable content cached]"
            continue
        to_fetch.append(u)

    if to_fetch:
        err = _wc_check()
        if err:
            if not results and not pending:
                return err
            _progress("web budget exhausted — serving cached URLs only")
            for u in to_fetch:
                results[u] = err
            to_fetch = []
        else:
            # Cap to_fetch to remaining budget so Phase 1 doesn't launch fetches
            # that will be discarded once the budget runs out mid-batch.
            remaining = _wc_remaining()
            if remaining < len(to_fetch):
                _progress(f"web budget: capping fetch to {remaining}/{len(to_fetch)} URLs")
                to_fetch = to_fetch[:remaining]

    n_fetch = len(to_fetch)
    _phase(f"📄 batch browse: {n_fetch} fetch + {len(pending)} re-summarize + {len(results)} cached")

    # ── Phase 1: parallel HTTP fetch (ไม่มี LLM ใน thread) ───────────
    raw_map: dict[str, str] = {}
    if to_fetch:
        try:
            pool = ThreadPoolExecutor(max_workers=min(_MAX_WORKERS, n_fetch))
            try:
                futures = {pool.submit(_http_only, u): u for u in to_fetch}
                done = 0
                for future in as_completed(futures, timeout=35):
                    try:
                        done += 1
                        url, raw = future.result(timeout=35)
                        raw_map[url] = raw
                        _progress(f"HTTP {done}/{n_fetch}: {url[:55]}")
                    except Exception:
                        pass
            finally:
                # wait=False — don't let cleanup block on stragglers past the 35s budget
                pool.shutdown(wait=False)
        except Exception:
            pass  # timeout or executor error — use empty raw for affected URLs

    # นับ budget เฉพาะ fetch ที่สำเร็จ — error ไม่กิน web budget
    for url in to_fetch:
        raw = raw_map.get(url, "[error] fetch missing")
        if raw.startswith("[error]"):
            results[url] = raw
            continue
        err = _wc_check_and_inc()
        if err:
            results[url] = err
            continue
        pending.append({"url": url, "raw": raw, "is_new": True})

    # ── Phase 2: summarize — batch เดียวเมื่อ ≥2 URL ต้องใช้ LLM (F1) ─
    def _finish(p: dict, s: str) -> None:
        if p["is_new"]:
            web_cache.put(p["url"], p["raw"])
        web_cache.put_summary(p["url"], uq, s, raw=p["raw"])
        results[p["url"]] = f"[web:{p['url']}] {s}"

    need_llm = [p for p in pending if len(p["raw"].strip()) > SUMMARY_SKIP_LLM_BELOW]
    skip_llm = [p for p in pending if p not in need_llm]

    for p in skip_llm:
        _finish(p, summarize(p["raw"], url=p["url"], user_query=uq or None))

    if len(need_llm) >= 2:
        _phase(f"📄 summarize batch: {len(need_llm)} URLs")
        batch = summarize_batch([(p["url"], p["raw"]) for p in need_llm], uq or None)
        if batch is not None:
            for p in need_llm:
                _finish(p, batch[p["url"]])
            need_llm = []
    for p in need_llm:
        _phase(f"📄 summarize: {p['url'][:50]}")
        _finish(p, summarize(p["raw"], url=p["url"], user_query=uq or None))

    # ── รวม results ตาม order เดิม ────────────────────────────────────
    lines = [f"[batch_browse] {len(clean)} URLs ({n_fetch} fetched, {len(clean) - n_fetch} cached)"]
    for u in clean:
        lines.append(results.get(u, f"[error] missing: {u}"))

    return "\n\n".join(lines)
